from typing import Sequence
import logging, time
from tqdm.auto import tqdm, trange

import anndata as ad
import scanpy as sc
import squidpy as sq

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import seaborn as sns
from matplotlib import pyplot as plt

from popari.sample_for_integral import integrate_of_exponential_over_simplex
from popari.util import NesterovGD, IndependentSet, sample_graph_iid, project2simplex, project2simplex_, project_M, project_M_, get_datetime, convert_numpy_to_pytorch_sparse_coo

import torch
import torch.nn.functional as F
from torch.nn import MultiheadAttention

class PopariDataset(ad.AnnData):
    r"""Wrapper around AnnData object. Allows for preprocessing of dataset for Popari.


    Attributes:
        dataset: AnnData dataset to convert into Popari-compatible object.
        replicate_name: name of dataset
        coordinates_key: location in ``.obsm`` dataframe of 2D coordinates for datapoints.
    """

    def __init__(self, dataset: ad.AnnData, replicate_name: str, coordinates_key: str = "spatial"):
        super().__init__(
            X = dataset.X,
            obs = dataset.obs,
            obsp = dataset.obsp,
            obsm = dataset.obsm, 
            var = dataset.var, 
            varp = dataset.varp,
            uns = dataset.uns
        )
      
        self.coordinates_key = coordinates_key 
        if self.coordinates_key not in self.obsm:
            raise ValueError("Dataset must include spatial coordinates.")

        self.name = f"{replicate_name}"

    def compute_spatial_neighbors(self, threshold: float = 94.5):
        r"""Compute neighbor graph based on spatial coordinates.

        Stores resulting graph in ``self.obs["adjacency_list"]``.

        """

        sq.gr.spatial_neighbors(self, coord_type="generic", delaunay=True)
        distance_matrix = self.obsp["spatial_distances"]
        distances = distance_matrix.data
        cutoff = np.percentile(distances, threshold)

        sq.gr.spatial_neighbors(self, coord_type="generic", delaunay=True, radius=[0, cutoff])
        self.obsp["adjacency_matrix"] = self.obsp["spatial_connectivities"]
        
        num_cells, _ = self.obsp["adjacency_matrix"].shape

        adjacency_list = [[] for _ in range(num_cells)]
        for x, y in zip(*self.obsp["adjacency_matrix"].nonzero()):
            adjacency_list[x].append(y)

        self.obs["adjacency_list"] = adjacency_list

    @staticmethod
    def remove_connectivity_artifacts(sparse_distance_matrix: csr_matrix, sparse_adjacency_matrix: csr_matrix, threshold: float = 94.5):
        """Remove artifacts in adjacency matrices produced by heuristic algorithms.

        For example, when Delaunay triangulation is used to produce the adjacency matrix, some spots which are
        connected as a result may actually be far apart in Euclidean space.
        
        Args:
            sparse_distance_matrix:

        """
        dense_distances = sparse_distance_matrix.toarray()
        distances = sparse_distance_matrix.data
        cutoff = np.percentile(distances, threshold)
        mask = dense_distances < cutoff
    
        sparse_adjacency_matrix[~mask] = 0
        sparse_adjacency_matrix.eliminate_zeros()
        
        return sparse_adjacency_matrix

    def plot_metagene_embedding(self, metagene_index: int, **scatterplot_kwargs):
        r"""Plot the embedding values for a metagene in-situ.

        Args:
            metagene_index: index of the metagene to plot.
            **scatterplot_kwargs: keyword args to pass to ``sns.scatterplot``.
        """
        points = self.obsm["spatial"]
        x, y = points.T
        metagene = self.obsm["X"][:, metagene_index]
  
        palette = "viridis" if "palette" not in scatterplot_kwargs else scatterplot_kwargs.pop("palette")

        metagene_key = f"Metagene {metagene_index}"
        metagene_expression = pd.DataFrame({"x":x, "y":y, metagene_key: metagene})
        sns.scatterplot(data=metagene_expression, x="x", y="y", hue=metagene_key, palette=palette, **scatterplot_kwargs)

        ax = scatterplot_kwargs.get("ax", None)
        if isinstance(palette, str) and ax:
            norm = plt.Normalize(metagene_expression[metagene_key].min(), metagene_expression[metagene_key].max())
            sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
            sm.set_array([])
            
            ax.figure.colorbar(sm)

class EmbeddingOptimizer():
    """Optimizer and state for Popari embeddings.

    """

    def __init__(self, K, Ys, datasets, initial_context=None, context=None, use_inplace_ops=False, embedding_step_size_multiplier=1, embedding_mini_iterations=1000, embedding_acceleration_trick=True, verbose=0):
        self.verbose = verbose
        self.use_inplace_ops = use_inplace_ops
        self.datasets = datasets
        self.K = K
        self.Ys = Ys
        self.initial_context = initial_context if initial_context else {"device": "cpu", "dtype": torch.float32}
        self.context = context if context else {"device": "cpu", "dtype": torch.float32}
        self.adjacency_lists = {dataset.name: dataset.obs["adjacency_list"] for dataset in self.datasets}
        self.adjacency_matrices = {dataset.name: convert_numpy_to_pytorch_sparse_coo(dataset.obsp["adjacency_matrix"], self.context) for dataset in self.datasets}
        self.embedding_step_size_multiplier = embedding_step_size_multiplier
        self.embedding_mini_iterations = embedding_mini_iterations
        self.embedding_acceleration_trick = embedding_acceleration_trick
       
        if self.verbose:
            print(f"{get_datetime()} Initializing EmbeddingState") 
        self.embedding_state = EmbeddingState(K, self.datasets, context=self.context)

    def link(self, parameter_optimizer):
        self.parameter_optimizer = parameter_optimizer

    def update_embeddings(self, use_neighbors=True):
        """Update Popari embeddings according to optimization scheme.

        """
        logging.info(f'{get_datetime()}Updating latent states')

        loss_list = []
        for dataset_index, dataset  in enumerate(self.datasets):
            is_spatial_replicate = ("adjacency_list" in dataset.obs)
            sigma_yx = self.parameter_optimizer.sigma_yxs[dataset_index]
            Y = self.Ys[dataset_index].to(self.context["device"])
            X = self.embedding_state[dataset.name].to(self.context["device"])
            M = self.parameter_optimizer.metagene_state[dataset.name].to(self.context["device"])
            prior_x_mode = self.parameter_optimizer.prior_x_modes[dataset_index]
            prior_x = self.parameter_optimizer.prior_xs[dataset_index]
            if not is_spatial_replicate or not use_neighbors:
                loss, self.embedding_state[dataset.name][:] = self.estimate_weight_wonbr(
                    Y, M, X, sigma_yx, prior_x_mode, prior_x, dataset)
            else:
                loss, self.embedding_state[dataset.name][:] = self.estimate_weight_wnbr(
                    Y, M, X, sigma_yx, prior_x_mode, prior_x, dataset)

            loss_list.append(loss)
    
    def nll_embeddings(self, use_neighbors=True):
        with torch.no_grad():
            loss_embeddings = torch.zeros(1, **self.context)
            for dataset_index, dataset  in enumerate(self.datasets):
                is_spatial_replicate = ("adjacency_list" in dataset.obs)
                sigma_yx = self.parameter_optimizer.sigma_yxs[dataset_index]
                Y = self.Ys[dataset_index].to(self.context["device"])
                X = self.embedding_state[dataset.name].to(self.context["device"])
                M = self.parameter_optimizer.metagene_state[dataset.name].to(self.context["device"])
                prior_x_mode = self.parameter_optimizer.prior_x_modes[dataset_index]
                prior_x = self.parameter_optimizer.prior_xs[dataset_index]
                if not is_spatial_replicate or not use_neighbors:
                    loss = self.nll_weight_wonbr(
                        Y, M, X, sigma_yx, prior_x_mode, prior_x, dataset)
                else:
                    loss = self.nll_weight_wnbr(
                        Y, M, X, sigma_yx, prior_x_mode, prior_x, dataset)

                loss_embeddings += loss

        return loss_embeddings.cpu().numpy()

    @torch.no_grad()
    def estimate_weight_wonbr(self, Y, M, X, sigma_yx, prior_x_mode, prior_x, dataset, n_epochs=1000, tol=1e-6, update_alg='gd'):
        """Estimate weights without spatial information - equivalent to vanilla NMF.
   
        Optimizes the follwing objective with respect to hidden state X:

        min 1/2σ^2 || Y - X MT ||_2^2 + lam || X ||_1
        grad = X MT M / σ^2 - Y MT / σ^2 + lam
    
        TODO: use (projected) Nesterov GD. not urgent

        Args:
            Y (torch.Tensor):
        """
    
        # Precomputing quantities 
        MTM = M.T @ M / (sigma_yx ** 2)
        YM = Y @ M / (sigma_yx ** 2)
        Ynorm = torch.linalg.norm(Y, ord='fro').item() ** 2 / (sigma_yx ** 2)
        step_size = 1 / torch.linalg.eigvalsh(MTM).max().item()
        loss_prev, loss = np.inf, np.nan
    
        def multiplicative_update(X_prev):
            """
            TODO:UNTESTED
    
            """
            X = torch.clip(X_prev, min=1e-10)
            loss = ((X @ MTM) * X).sum() / 2 - clipped_X.view(-1) @ YM.view(-1) + Ynorm / 2
            numerator = YM
            denominator = X @ MTM
            if prior_x_mode == 'exponential shared fixed':
                # see sklearn.decomposition.NMF
                loss += (X @ prior_x[0]).sum()
                denominator += prior_x[0][None]
            else:
                raise NotImplementedError
    
            loss = loss.item()
            assert loss <= loss_prev * (1 + 1e-4), (loss_prev, loss, (loss_prev - loss) / loss)
            multiplicative_factor = numerator / denominator
            X *= multiplicative_factor
            torch.clip(X, min=1e-10)
    
            return X, loss
    
        def gradient_update(X):
            """
            TODO:UNTESTED
    
            """
            quadratic_term_gradient = X @ MTM
            linear_term_gradient = YM
            if prior_x_mode == 'exponential shared fixed':
                linear_term_gradient = linear_term_gradient - prior_x[0][None]
            elif not prior_x_mode:
                pass
            else:
                raise NotImplementedError
            loss = (quadratic_term_gradient * X).sum().item() / 2 - (linear_term_gradient * X).sum().item() + Ynorm / 2
            gradient = quadratic_term_gradient - linear_term_gradient
            X = X.sub(gradient, alpha=step_size)
            X = torch.clip(X, min=1e-10)
            
            return X, loss
            
        progress_bar = trange(n_epochs, leave=True, disable=not self.verbose, miniters=1000)
        for epoch in progress_bar:
            X_prev = X.clone()
            if update_alg == 'mu':
                # TODO: it seems like loss might not always decrease...
                X.clip_(min=1e-10)
                loss = ((X @ MTM) * X).sum() / 2 - X.view(-1) @ YM.view(-1) + Ynorm / 2
                numerator = YM
                denominator = X @ MTM
                if prior_x_mode == 'exponential shared fixed':
                    # see sklearn.decomposition.NMF
                    loss += (X @ prior_x[0]).sum()
                    denominator.add_(prior_x[0][None])
                elif not prior_x_mode:
                    pass
                else:
                    raise NotImplementedError
    
                loss = loss.item()
                assert loss <= loss_prev * (1 + 1e-4), (loss_prev, loss, (loss_prev - loss) / loss)
                multiplicative_factor = numerator / denominator
                X.mul_(multiplicative_factor).clip_(min=1e-10)
    
                # X, loss = multiplicative_update(X_prev)
            elif update_alg == 'gd':
                X, loss = gradient_update(X)
            else:
                raise NotImplementedError
    
            dX = torch.abs((X_prev - X) / torch.linalg.norm(X, dim=1, ord=1, keepdim=True)).max().item()
            do_stop = dX < tol
            progress_bar.set_description(
                f'Updating weight w/o nbrs: loss = {loss:.1e} '
                f'%δloss = {(loss_prev - loss) / loss:.1e} '
                f'%δX = {dX:.1e}'
            )
            loss_prev = loss
            if do_stop:
                break
        progress_bar.close()
        return loss, X
    
    @torch.no_grad()
    def nll_weight_wonbr(self, Y, M, X, sigma_yx, prior_x_mode, prior_x, dataset):
        # Precomputing quantities 
        MTM = M.T @ M / (sigma_yx ** 2)
        YM = Y @ M / (sigma_yx ** 2)
        Ynorm = torch.linalg.norm(Y, ord='fro').item() ** 2 / (sigma_yx ** 2)
        step_size = 1 / torch.linalg.eigvalsh(MTM).max().item()
        loss_prev, loss = np.inf, np.nan
    
        loss = ((X @ MTM) * X).sum() / 2 - X.view(-1) @ YM.view(-1) + Ynorm / 2

        return loss
    
    @torch.no_grad()
    def estimate_weight_wnbr(self, Y, M, X, sigma_yx, prior_x_mode, prior_x, dataset, tol=1e-5, update_alg='nesterov'):
        """Estimate updated weights taking neighbor-neighbor interactions into account.
    
        The optimization for all variables
        min 1/2σ^2 || Y - diag(S) Z MT ||_2^2 + lam || S ||_1 + sum_{ij in E} ziT Σx-1 zj
    
        for s_i
        min 1/2σ^2 || y - M z s ||_2^2 + lam s
        s* = max(0, ( yT M z / σ^2 - lam ) / ( zT MT M z / σ^2) )
    
        for Z
        min 1/2σ^2 || Y - diag(S) Z MT ||_2^2 + sum_{ij in E} ziT Σx-1 zj
        grad_i = MT M z s^2 / σ^2 - MT y s / σ^2 + sum_{j in Ei} Σx-1 zj
    
        TODO: Try projected Newton's method.
        TM: Inverse is precomputed once, and projection is cheap. Not sure if it works theoretically
        """
        # Precomputing quantities
        MTM = M.T @ M / (sigma_yx ** 2)
        YM = Y.to(M.device) @ M / (sigma_yx ** 2)
        Ynorm = torch.linalg.norm(Y, ord='fro').item() ** 2 / (sigma_yx ** 2)
        base_step_size = self.embedding_step_size_multiplier / torch.linalg.eigvalsh(MTM).max().item()
        S = torch.linalg.norm(X, dim=1, ord=1, keepdim=True)

        if self.verbose > 3:
            print(f"S max: {S.max()}")
            print(f"S min: {S.min()}")

        Z = X / S
        N = len(Z)
        
        E_adjacency_list = self.adjacency_lists[dataset.name]
        adjacency_matrix = self.adjacency_matrices[dataset.name].to(self.context["device"])
        Sigma_x_inv = self.parameter_optimizer.spatial_affinity_state[dataset.name].to(self.context["device"])
    
        def update_s():
            S[:] = (YM * Z).sum(axis=1, keepdim=True)
            if prior_x_mode == 'exponential shared fixed':
                # TODO: why divide by two?
                S.sub_(prior_x[0][0] / 2)
            elif not prior_x_mode:
                pass
            else:
                raise NotImplementedError
    
            denominator = ((Z @ MTM) * Z).sum(axis=1, keepdim=True)
            S.div_(denominator)
            S.clip_(min=1e-5)
    
        def calc_func_grad(Z_batch, S_batch, quad, linear):
            t = (Z_batch @ quad).mul_(S_batch ** 2)
            f = (t * Z_batch).sum() / 2
            g = t
            t = linear
            f -= (t * Z_batch).sum()
            g -= t
            g.sub_(g.sum(1, keepdim=True))
            
            return f.item(), g
    
        def update_z_gd(Z):
            step_size = base_step_size / S.square()
            pbar = tqdm(range(N), leave=False, disable=True)
            for idx in IndependentSet(E_adjacency_list, device=self.context["device"], batch_size=128):
                step_size_scale = 1
                quad_batch = MTM
                linear_batch = YM[idx] * S[idx] - torch.index_select(adjacency_matrix, 0, idx) @ Z @ Sigma_x_inv
                Z_batch = Z[idx].contiguous()
                S_batch = S[idx].contiguous()
                step_size_batch = step_size[idx].contiguous()
                func, grad = calc_func_grad(Z_batch, S_batch, quad_batch, linear_batch)
                while True:
                    Z_batch_new = Z_batch - step_size_batch * step_size_scale * grad
                    if self.use_inplace_ops:
                        Z_batch_new = project2simplex_(Z_batch_new, dim=1)
                    else:
                        Z_batch_new = project2simplex(Z_batch_new, dim=1)
                    dZ = Z_batch_new.sub(Z_batch).abs().max().item()
                    func_new, grad_new = calc_func_grad(Z_batch_new, S_batch, quad_batch, linear_batch)
                    if func_new < func:
                        Z_batch = Z_batch_new
                        func = func_new
                        grad = grad_new
                        step_size_scale *= 1.1
                        continue
                    else:
                        step_size_scale *= .5
                    if dZ < tol or step_size_scale < .5: break
                assert step_size_scale > .1
                Z[idx] = Z_batch
                pbar.set_description(f'Updating Z w/ nbrs via line search: lr={step_size_scale:.1e}')
                pbar.update(len(idx))
            pbar.close()
    
            return Z
    
        def update_z_gd_nesterov(Z):
            pbar = trange(N, leave=False, disable=True, desc='Updating Z w/ nbrs via Nesterov GD')
           
            func, grad = calc_func_grad(Z, S, MTM, YM * S - adjacency_matrix @ Z @ Sigma_x_inv / 2)
            for idx in IndependentSet(E_adjacency_list, device=self.context["device"], batch_size=256):
                quad_batch = MTM
                linear_batch_spatial = - torch.index_select(adjacency_matrix, 0, idx) @ Z @ Sigma_x_inv
                Z_batch = Z[idx].contiguous()
                S_batch = S[idx].contiguous()
                    
                optimizer = NesterovGD(Z_batch, base_step_size / S_batch.square())
                ppbar = trange(10000, leave=False, disable=not (self.verbose > 3))
                for i_iter in ppbar:
                    if self.embedding_acceleration_trick:
                        update_s() # TODO: update S_batch directly
                    S_batch = S[idx].contiguous()
                    linear_batch = linear_batch_spatial + YM[idx] * S_batch
                    if i_iter == 0:
                        func, grad = calc_func_grad(Z_batch, S_batch, quad_batch, linear_batch)
                        func, grad = calc_func_grad(Z, S, MTM, YM * S - adjacency_matrix @ Z @ Sigma_x_inv / 2)
                    NesterovGD.step_size = base_step_size / S_batch.square() # TM: I think this converges as s converges
                    func, grad = calc_func_grad(Z_batch, S_batch, quad_batch, linear_batch)
                    # grad_limit = torch.quantile(torch.abs(grad), 0.9)
                    # max_before = torch.max(torch.abs(grad))
                    # grad.clamp_(min=-grad_limit, max=grad_limit)
                    # max_after = torch.max(torch.abs(grad))
                    Z_batch_prev = Z_batch.clone()
                    Z_batch = optimizer.step(grad)
                    # if max(torch.linalg.norm(Z_batch_copy, ord=1, axis=1)) > 1000:
                    #     print(f"L1 norm of Z_batch_copy: {torch.linalg.norm(Z_batch_copy, ord=1, axis=1)}")
                    #     print(f"Max L1 norm of Z_batch_copy: {max(torch.linalg.norm(Z_batch_copy, ord=1, axis=1))}")
                    #     print(f"L2 norm of Z_batch_copy: {torch.linalg.norm(Z_batch_copy, ord=2, axis=1)}")
                    #     print(f"Max L2 norm of Z_batch_copy: {max(torch.linalg.norm(Z_batch_copy, ord=2, axis=1))}")
                    #     print(f"Grad: {grad}")
                    #     print(f"Grad max before: {max_before}")
                    #     print(f"grad limit:{grad_limit}")
                    #     print(f"Grad max after: {max_after}")
                   
                    if self.use_inplace_ops:
                        Z_batch = project2simplex_(Z_batch, dim=1)
                    else:
                        Z_batch = project2simplex(Z_batch, dim=1)

                    optimizer.set_parameters(Z_batch)

                    dZ = (Z_batch_prev - Z_batch).abs().max().item()
                    Z[idx] = Z_batch
                    description = (
                        f'func={func:.1e}, dZ={dZ:.1e}'
                    )
                    ppbar.set_description(description)
                    if dZ < tol:
                        break
                ppbar.close()
                
                Z[idx] = Z_batch
                func, grad = calc_func_grad(Z_batch, S_batch, quad_batch, linear_batch)
                func, grad = calc_func_grad(Z, S, MTM, YM * S - adjacency_matrix @ Z @ Sigma_x_inv /2 )
                pbar.update(len(idx))
            pbar.close()
            func, grad = calc_func_grad(Z, S, MTM, YM * S - adjacency_matrix @ Z @ Sigma_x_inv / 2)
    
            return Z
    
        def compute_loss():
            X = Z * S
            loss = ((X @ MTM) * X).sum() / 2 - (X * YM).sum() + Ynorm / 2
            if prior_x_mode == 'exponential shared fixed':
                loss += prior_x[0][0] * S.sum()
            elif not prior_x_mode:
                pass
            else:
                raise NotImplementedError
    
            if Sigma_x_inv is not None:
                loss += ((adjacency_matrix @ Z) @ Sigma_x_inv).mul(Z).sum() / 2
            loss = loss.item()
            # assert loss <= loss_prev, (loss_prev, loss)
            return loss
    
        # TM: consider combine compute_loss and update_z to remove a call to torch.sparse.mm
        # TM: the above idea is not practical if we update only a subset of nodes each time
    
        loss = np.inf
        pbar = trange(self.embedding_mini_iterations, disable=not self.verbose, desc='Updating weight w/ neighbors')
    
        for epoch in pbar:
            update_s()
            Z_prev = Z.clone().detach()
            # We may use Nesterov first and then vanilla GD in later iterations
            # update_z_mu(Z)
            # update_z_gd(Z)
            if update_alg == "gd":
                Z = update_z_gd(Z)
            elif update_alg == "nesterov":
                Z = update_z_gd_nesterov(Z)
    
            loss_prev = loss
            loss = compute_loss()
            dloss = loss_prev - loss
            dZ = (Z_prev - Z).abs().max().item()
            pbar.set_description(
                f'Updating weight w/ neighbors: loss = {loss:.1e} '
                f'δloss = {dloss:.1e} '
                f'δZ = {dZ:.1e}'
            )
            if dZ < tol: break
    
        X_final = Z * S
        return loss, X_final
    
    @torch.no_grad()
    def nll_weight_wnbr(self, Y, M, X, sigma_yx, prior_x_mode, prior_x, dataset, tol=1e-5, update_alg='nesterov'):
        # Precomputing quantities
        MTM = M.T @ M / (sigma_yx ** 2)
        YM = Y.to(M.device) @ M / (sigma_yx ** 2)
        Ynorm = torch.linalg.norm(Y, ord='fro').item() ** 2 / (sigma_yx ** 2)
        S = torch.linalg.norm(X, dim=1, ord=1, keepdim=True)

        Z = X / S
        N = len(Z)
        
        E_adjacency_list = self.adjacency_lists[dataset.name]
        adjacency_matrix = self.adjacency_matrices[dataset.name].to(self.context["device"])
        Sigma_x_inv = self.parameter_optimizer.spatial_affinity_state[dataset.name].to(self.context["device"])
        
        def compute_loss():
            X = Z * S
            loss = ((X @ MTM) * X).sum() / 2 - (X * YM).sum() + Ynorm / 2
            if prior_x_mode == 'exponential shared fixed':
                loss += prior_x[0][0] * S.sum()
            elif not prior_x_mode:
                pass
            else:
                raise NotImplementedError
    
            if Sigma_x_inv is not None:
                loss += ((adjacency_matrix @ Z) @ Sigma_x_inv).mul(Z).sum() / 2
            loss = loss.item()
            # assert loss <= loss_prev, (loss_prev, loss)
            return loss
        
        loss = compute_loss()
    
        return loss

class EmbeddingState(dict):
    """Collections of cell embeddings for all ST replicates.

    Attributes:
        K: embedding dimension:

    """
    def __init__(self, K: int, datasets: Sequence[PopariDataset], initial_context=None, context=None):
        self.datasets = datasets
        self.K = K
        self.initial_context = initial_context if initial_context else {"device": "cpu", "dtype": torch.float32}
        self.context = context if context else {"device": "cpu", "dtype": torch.float32}
        super().__init__()
            
        self.embeddings = []

        for dataset in self.datasets:
            num_cells, _ = dataset.shape
            replicate_embeddings = torch.zeros((num_cells, K), **self.context)
            self.__setitem__(dataset.name, replicate_embeddings)
            self.embeddings.append(replicate_embeddings)

class ParameterOptimizer():
    """Optimizer and state for Popari parameters.

    """

    def __init__(self, K, Ys, datasets, betas, prior_x_modes,
            metagene_groups,
            metagene_tags,
            spatial_affinity_groups,
            spatial_affinity_tags,
            spatial_affinity_regularization_power=2,
            spatial_affinity_constraint=None,
            spatial_affinity_centering=False,
            spatial_affinity_scaling=10,
            lambda_Sigma_x_inv=1e-2,
            spatial_affinity_tol=2e-3,
            spatial_affinity_mode="shared lookup",
            metagene_mode="shared",
            lambda_M=0.5,
            lambda_Sigma_bar=0.5,
            spatial_affinity_lr=1e-3,
            M_constraint="simplex",
            sigma_yx_inv_mode="separate",
            initial_context=None,
            context=None,
            use_inplace_ops=False,
            verbose=0
    ):
        self.verbose = verbose
        self.use_inplace_ops = use_inplace_ops

        self.datasets = datasets
        self.spatial_affinity_mode = spatial_affinity_mode
        self.K = K
        self.Ys = Ys
        self.metagene_groups = metagene_groups
        self.metagene_tags = metagene_tags
        self.spatial_affinity_groups = spatial_affinity_groups
        self.spatial_affinity_tags = spatial_affinity_tags
        self.sigma_yx_inv_mode = sigma_yx_inv_mode
        self.lambda_Sigma_bar = lambda_Sigma_bar
        self.spatial_affinity_constraint = spatial_affinity_constraint
        self.spatial_affinity_centering = spatial_affinity_centering
        self.spatial_affinity_lr = spatial_affinity_lr
        self.spatial_affinity_scaling = spatial_affinity_scaling
        self.lambda_Sigma_x_inv = lambda_Sigma_x_inv
        self.spatial_affinity_tol=spatial_affinity_tol
        self.lambda_M = lambda_M
        self.metagene_mode = metagene_mode
        self.M_constraint = M_constraint
        self.prior_x_modes = prior_x_modes
        self.betas = betas
        self.initial_context = initial_context if initial_context else {"device": "cpu", "dtype": torch.float32}
        self.context = context if context else {"device": "cpu", "dtype": torch.float32}
        self.spatial_affinity_regularization_power = spatial_affinity_regularization_power
        self.adjacency_lists = {dataset.name: dataset.obs["adjacency_list"] for dataset in self.datasets}
        self.adjacency_matrices = {dataset.name: convert_numpy_to_pytorch_sparse_coo(dataset.obsp["adjacency_matrix"], self.context) for dataset in self.datasets}
       
        if self.verbose:
            print(f"{get_datetime()} Initializing MetageneState") 
        
        self.metagene_state = MetageneState(
            self.K,
            self.datasets,
            self.metagene_groups,
            self.metagene_tags,
            mode=self.metagene_mode,
            M_constraint=self.M_constraint,
            initial_context=self.initial_context,
            context=self.context
        )
        
        if self.verbose:
            print(f"{get_datetime()} Initializing SpatialAffinityState") 
        
        self.spatial_affinity_state = SpatialAffinityState(
            self.K,
            self.metagene_state,
            self.datasets,
            self.spatial_affinity_groups,
            self.spatial_affinity_tags,
            self.betas,
            scaling=self.spatial_affinity_scaling,
            mode=self.spatial_affinity_mode,
            initial_context=self.initial_context,
            lr=self.spatial_affinity_lr,
            context=self.context
        )
        
        if all(prior_x_mode == 'exponential shared fixed' for prior_x_mode in self.prior_x_modes):
            self.prior_xs = [(torch.ones(self.K, **self.initial_context),) for _ in range(len(self.datasets))]
        elif all(prior_x_mode == None for prior_x_mode in self.prior_x_modes):
            self.prior_xs = [(torch.zeros(self.K, **self.initial_context),) for _ in range(len(self.datasets))]
        else:
            raise NotImplementedError

        self.sigma_yxs = np.zeros(len(self.datasets))

    def link(self, embedding_optimizer):
        """Link to embedding_optimizer.
        
        """
        self.embedding_optimizer = embedding_optimizer
       
    def scale_metagenes(self):
        norm_axis = 1
        # norm_axis = int(self.metagene_mode == "differential")
        if self.M_constraint == 'simplex':
            scale_factor = torch.linalg.norm(self.metagene_state.metagenes, axis=norm_axis, ord=1, keepdim=True)
        elif self.M_constraint == 'unit_sphere':
            scale_factor = torch.linalg.norm(self.metagene_state.metagenes, axis=norm_axis, ord=2, keepdim=True)
        
        self.metagene_state.metagenes.div_(scale_factor)
        for group_index, group_replicates in enumerate(self.metagene_groups.values()):
            for dataset_index, dataset in enumerate(self.datasets):
                if dataset.name not in group_replicates:
                    continue
                replicate_embedding = self.embedding_optimizer.embedding_state.embeddings[dataset_index]
                if self.metagene_mode == "differential":
                    replicate_scale_factor = scale_factor[dataset_index]
                    replicate_embedding.mul_(replicate_scale_factor)
                else:
                    group_scale_factor = scale_factor[group_index]
                    replicate_embedding.mul_(group_scale_factor)

    def estimate_Sigma_x_inv(self, Sigma_x_inv, replicate_mask, optimizer, Sigma_x_inv_bar=None, subsample_rate=None, constraint=None, n_epochs=1000, tol=2e-3, check_frequency=50):
        """Optimize Sigma_x_inv parameters.
    
       
        Differential mode:
        grad =  ... + λ_Sigma_x_inv ( Sigma_x_inv - Sigma_x_inv_bar )
    
        Args:
            Xs: list of latent expression embeddings for each FOV.
            Sigma_x_inv: previous estimate of Σx-1
    
        """
        datasets = [dataset for (use_replicate, dataset) in zip(replicate_mask, self.datasets) if use_replicate]
        betas = [beta for (use_replicate, beta) in zip(replicate_mask, self.betas) if use_replicate]
        betas = np.array(betas) / np.sum(betas)
        
        Xs = [self.embedding_optimizer.embedding_state[dataset.name] for dataset in datasets]
        spatial_flags = ["adjacency_list" in dataset.obs for dataset in datasets]

        num_edges_per_fov = [sum(map(len, dataset.obs["adjacency_list"])) for dataset in datasets]
    
        if not any(sum(map(len, dataset.obs["adjacency_list"])) > 0 and u for dataset, u in zip(datasets, spatial_flags)):
            return
    
        linear_term_coefficient = torch.zeros_like(Sigma_x_inv).requires_grad_(False)
        size_factors = [torch.linalg.norm(X, axis=1, ord=1, keepdim=True) for X in Xs ]
        Zs = [X.to(self.context["device"]) / size_factor for X, size_factor in zip(Xs, size_factors)]
        nus = [] # sum of neighbors' z
        weighted_total_cells = 0

        for Z, dataset, use_spatial, beta in zip(Zs, datasets, spatial_flags, self.betas):
            adjacency_list = self.adjacency_lists[dataset.name]
            adjacency_matrix = self.adjacency_matrices[dataset.name]

            if use_spatial:
                nu = adjacency_matrix @ Z
                linear_term_coefficient.addmm_(Z.T, nu, alpha=beta)
            else:
                nu = None

            nus.append(nu)
            weighted_total_cells += beta * sum(map(len, adjacency_list))
            del Z, adjacency_matrix
        # linear_term_coefficient = (linear_term_coefficient + linear_term_coefficient.T) / 2 # should be unnecessary as long as adjacency_list is symmetric
        if self.verbose > 2:
            print(f"spatial affinity linear term coefficient range: {linear_term_coefficient.min().item():.2e} ~ {linear_term_coefficient.max().item():.2e}")
    
        history = []
        Sigma_x_inv.requires_grad_(True)
   
        loss_prev, loss = np.inf, np.nan
        
        verbose_bar = tqdm(disable=not (self.verbose > 2), bar_format='{desc}{postfix}')
        progress_bar = trange(1, n_epochs+1, disable=not self.verbose, desc='Updating Σx-1')

        Sigma_x_inv_best, loss_best, epoch_best = None, np.inf, -1
        dSigma_x_inv = np.inf
        early_stop_epoch_count = 0
        Sigma_x_inv_prev = Sigma_x_inv.clone().detach()
        for epoch in progress_bar:
            optimizer.zero_grad()
    
            # Compute loss 
            linear_term = Sigma_x_inv.view(-1) @ linear_term_coefficient.view(-1)
            regularization = torch.zeros(1, **self.context)
            if Sigma_x_inv_bar is not None:
                group_weighting = 1 / len(Sigma_x_inv_bar)
                for group_Sigma_x_inv_bar in Sigma_x_inv_bar:
                    regularization += group_weighting * self.lambda_Sigma_bar * (group_Sigma_x_inv_bar - Sigma_x_inv).pow(2).sum() * weighted_total_cells / 2

            regularization += self.lambda_Sigma_x_inv * Sigma_x_inv.abs().pow(self.spatial_affinity_regularization_power).sum() * weighted_total_cells / 2
            
            log_partition_function = 0
            for nu, beta in zip(nus, self.betas):
                if subsample_rate is None:
                    subsample_index = np.arange(len(dataset))
                    subsample_multiplier = 1
                else:
                    node_limit = int(subsample_rate * len(dataset))
                    subsample_index = np.sort(sample_graph_iid(adjacency_list, range(len(dataset)), node_limit))
                    subsample_multiplier = 1 / subsample_rate
                    nu = nu[subsample_index]

                if nu is None:
                    continue
                assert torch.isfinite(nu).all()
                assert torch.isfinite(Sigma_x_inv).all()
                eta = nu @ Sigma_x_inv
                logZ = integrate_of_exponential_over_simplex(eta)
                log_partition_function += subsample_multiplier * beta * logZ.sum()
    
            loss = (linear_term + regularization + log_partition_function) / weighted_total_cells
  
            if loss < loss_best:
                Sigma_x_inv_best = Sigma_x_inv.clone().detach()
                loss_best = loss.item()
                epoch_best = epoch
    
            loss.backward()
            Sigma_x_inv.grad = (Sigma_x_inv.grad + Sigma_x_inv.grad.T) / 2
            optimizer.step()
            with torch.no_grad():
                if self.spatial_affinity_centering:
                    Sigma_x_inv -= Sigma_x_inv.mean()

                if self.spatial_affinity_constraint == "clamp":
                    Sigma_x_inv.clamp_(min=-self.spatial_affinity_state.scaling, max=self.spatial_affinity_state.scaling)
                elif self.spatial_affinity_constraint == "scale":
                    Sigma_x_inv.mul_(self.spatial_affinity_state.scaling / Sigma_x_inv.abs().max())

                if epoch % check_frequency == 0:
                    loss = loss.item()
                    dloss = loss_prev - loss
                    loss_prev = loss
                    regularization_prev = regularization.item()
                    log_partition_function_prev = log_partition_function.item()
                    linear_term_prev = linear_term.item()
            
                    history.append((Sigma_x_inv.detach().cpu().numpy(), loss))
    
    
                    dSigma_x_inv = Sigma_x_inv_prev.sub(Sigma_x_inv).abs().max().item()
                    Sigma_x_inv_prev = Sigma_x_inv.clone().detach()
           
                    description = (
                        f'Updating Σx-1: loss = {dloss:.1e} -> {loss:.1e} '
                        f'δΣx-1 = {dSigma_x_inv:.1e} '
                        f'Σx-1 range = {Sigma_x_inv.min().item():.1e} ~ {Sigma_x_inv.max().item():.1e}'
                    )

                    verbose_description = (
                            f"Spatial affinity average: {Sigma_x_inv.mean().item():.1e} "
                            f"Total spatial affinity loss: {loss:.1e} "
                            f"spatial affinity linear term {linear_term:.6e} "
                            f"spatial affinity regularization {regularization.item():.1e} "
                            f"spatial affinity log_partition_function {log_partition_function:.1e} "
                        )

                    verbose_bar.set_description_str(verbose_description)
                    progress_bar.set_description(description)
   
                    if dSigma_x_inv < tol * check_frequency or epoch > epoch_best + 2 * check_frequency:
                        break
   
        verbose_bar.close()
        progress_bar.close()

        # with torch.no_grad():
        #     offset = -Sigma_x_inv.mean()
        #     Sigma_x_inv += offset
        #     print(f"Offset = {offset}")
        #     # Compute loss 
        #     linear_term = Sigma_x_inv.view(-1) @ linear_term_coefficient.view(-1)
        #     regularization = torch.zeros(1, **self.context)
        #     if Sigma_x_inv_bar is not None:
        #         group_weighting = 1 / len(Sigma_x_inv_bar)
        #         for group_Sigma_x_inv_bar in Sigma_x_inv_bar:
        #             regularization += group_weighting * self.lambda_Sigma_bar * (group_Sigma_x_inv_bar - Sigma_x_inv).pow(2).sum() * weighted_total_cells / 2

        #     regularization += self.lambda_Sigma_x_inv * Sigma_x_inv.pow(self.spatial_affinity_regularization_power).sum() * weighted_total_cells / 2
        #     
        #     log_partition_function = 0
        #     for nu, beta in zip(nus, self.betas):
        #         if subsample_rate is None:
        #             subsample_index = np.arange(len(dataset))
        #             subsample_multiplier = 1
        #         else:
        #             node_limit = int(subsample_rate * len(dataset))
        #             subsample_index = np.sort(sample_graph_iid(adjacency_list, range(len(dataset)), node_limit))
        #             subsample_multiplier = 1 / subsample_rate
        #             nu = nu[subsample_index]

        #         if nu is None:
        #             continue
        #         assert torch.isfinite(nu).all()
        #         assert torch.isfinite(Sigma_x_inv).all()
        #         eta = nu @ Sigma_x_inv
        #         logZ = integrate_of_exponential_over_simplex(eta)
        #         log_partition_function += subsample_multiplier * beta * logZ.sum()
    
        #     loss = (linear_term + regularization + log_partition_function) / weighted_total_cells

        #     print(f"Previous loss: total-{loss_prev}, regularization-{regularization_prev}, linear_term-{linear_term_prev}, log_partition_function-{log_partition_function_prev}")
        #     print(f"Loss after adding large offset: total-{loss.item()}, regularization-{regularization.item()}, linear_term-{linear_term.item()}, log_partition_function-{log_partition_function.item()}")
        #     2/0

        Sigma_x_inv = Sigma_x_inv_best
        Sigma_x_inv.requires_grad_(False)
       
        return Sigma_x_inv, loss * weighted_total_cells
    
    def nll_Sigma_x_inv(self, Sigma_x_inv, replicate_mask, Sigma_x_inv_bar=None):
        datasets = [dataset for (use_replicate, dataset) in zip(replicate_mask, self.datasets) if use_replicate]
        betas = [beta for (use_replicate, beta) in zip(replicate_mask, self.betas) if use_replicate]
        betas = np.array(betas) / np.sum(betas)
        
        Xs = [self.embedding_optimizer.embedding_state[dataset.name] for dataset in datasets]
        spatial_flags = ["adjacency_list" in dataset.obs for dataset in datasets]

        num_edges_per_fov = [sum(map(len, dataset.obs["adjacency_list"])) for dataset in datasets]
    
        if not any(sum(map(len, dataset.obs["adjacency_list"])) > 0 and u for dataset, u in zip(datasets, spatial_flags)):
            return
    
        linear_term_coefficient = torch.zeros_like(Sigma_x_inv).requires_grad_(False)
        size_factors = [torch.linalg.norm(X, axis=1, ord=1, keepdim=True) for X in Xs ]
        Zs = [X.to(self.context["device"]) / size_factor for X, size_factor in zip(Xs, size_factors)]
        nus = [] # sum of neighbors' z
        weighted_total_cells = 0

        for Z, dataset, use_spatial, beta in zip(Zs, datasets, spatial_flags, self.betas):
            adjacency_list = self.adjacency_lists[dataset.name]
            adjacency_matrix = self.adjacency_matrices[dataset.name]

            if use_spatial:
                nu = adjacency_matrix @ Z
                linear_term_coefficient.addmm_(Z.T, nu, alpha=beta)
            else:
                nu = None

            nus.append(nu)
            weighted_total_cells += beta * sum(map(len, adjacency_list))
            del Z, adjacency_matrix
    
        loss_prev, loss = np.inf, np.nan
        
        linear_term = Sigma_x_inv.view(-1) @ linear_term_coefficient.view(-1)
        regularization = torch.zeros(1, **self.context)
        if Sigma_x_inv_bar is not None:
            group_weighting = 1 / len(Sigma_x_inv_bar)
            for group_Sigma_x_inv_bar in Sigma_x_inv_bar:
                regularization += group_weighting * self.lambda_Sigma_bar * (group_Sigma_x_inv_bar - Sigma_x_inv).pow(2).sum() * weighted_total_cells / 2

        regularization += self.lambda_Sigma_x_inv * Sigma_x_inv.pow(self.spatial_affinity_regularization_power).sum() * weighted_total_cells / 2
        
        log_partition_function = 0
        for nu, beta in zip(nus, self.betas):
            if nu is None:
                continue
            assert torch.isfinite(nu).all()
            assert torch.isfinite(Sigma_x_inv).all()
            eta = nu @ Sigma_x_inv
            logZ = integrate_of_exponential_over_simplex(eta)
            log_partition_function += beta * logZ.sum()
    
        loss = (linear_term + regularization + log_partition_function) / weighted_total_cells
        
        return loss

    def update_spatial_affinity(self, differentiate_spatial_affinities=True, subsample_rate=None):
        if self.spatial_affinity_mode == "shared lookup":
            for group_name, group_replicates in self.spatial_affinity_groups.items():
                replicate_mask =  [dataset.name in group_replicates for dataset in self.datasets]
                first_dataset_name = group_replicates[0]
                Sigma_x_inv = self.spatial_affinity_state[first_dataset_name].to(self.context["device"])
                optimizer = self.spatial_affinity_state.optimizers[group_name]
                Sigma_x_inv, loss = self.estimate_Sigma_x_inv(Sigma_x_inv, replicate_mask, optimizer, subsample_rate=subsample_rate, tol=self.spatial_affinity_tol)
                with torch.no_grad():
                   self.spatial_affinity_state[first_dataset_name][:] = Sigma_x_inv

        elif self.spatial_affinity_mode == "differential lookup":
            for dataset_index, dataset in enumerate(self.datasets):
                if differentiate_spatial_affinities:
                    spatial_affinity_bars = [self.spatial_affinity_state.spatial_affinity_bar[group_name].detach() for group_name in self.spatial_affinity_tags[dataset.name]]
                else:
                    spatial_affinity_bars = None

                replicate_mask = [False] * len(self.datasets)
                replicate_mask[dataset_index] = True
                Sigma_x_inv = self.spatial_affinity_state[dataset.name].to(self.context["device"])
                optimizer = self.spatial_affinity_state.optimizers[dataset.name]
                Sigma_x_inv, loss = self.estimate_Sigma_x_inv(Sigma_x_inv, replicate_mask, optimizer, Sigma_x_inv_bar=spatial_affinity_bars, subsample_rate=subsample_rate, tol=self.spatial_affinity_tol)
        # K_options, group_options = np.meshgrid()
        # runs = 
                with torch.no_grad():
                    self.spatial_affinity_state[dataset.name][:] = Sigma_x_inv
            
            self.spatial_affinity_state.reaverage()
    
    def nll_spatial_affinities(self):
        with torch.no_grad():
            loss_spatial_affinities = torch.zeros(1, **self.context)
            if self.spatial_affinity_mode == "shared lookup":
                for group_name, group_replicates in self.spatial_affinity_groups.items():
                    replicate_mask =  [dataset.name in group_replicates for dataset in self.datasets]
                    first_dataset_name = group_replicates[0]
                    Sigma_x_inv = self.spatial_affinity_state[first_dataset_name].to(self.context["device"])
                    loss_Sigma_x_inv = self.nll_Sigma_x_inv(Sigma_x_inv, replicate_mask)
                    loss_spatial_affinities += loss_Sigma_x_inv

            elif self.spatial_affinity_mode == "differential lookup":
                for dataset_index, dataset in enumerate(self.datasets):
                    spatial_affinity_bars = [self.spatial_affinity_state.spatial_affinity_bar[group_name].detach() for group_name in self.spatial_affinity_tags[dataset.name]]

                    replicate_mask = [False] * len(self.datasets)
                    replicate_mask[dataset_index] = True
                    Sigma_x_inv = self.spatial_affinity_state[dataset.name].to(self.context["device"])
                    loss_Sigma_x_inv = self.nll_Sigma_x_inv(Sigma_x_inv, replicate_mask, Sigma_x_inv_bar=spatial_affinity_bars)
                    loss_spatial_affinities += loss_Sigma_x_inv

        return loss_spatial_affinities.cpu().numpy()

    def update_metagenes(self, differentiate_metagenes=True, simplex_projection_mode="exact"):
        if self.metagene_mode == "shared":
            for group_name, group_replicates in self.metagene_groups.items():
                first_dataset_name = group_replicates[0]
                replicate_mask =  [dataset.name in group_replicates for dataset in self.datasets]
                M = self.metagene_state[first_dataset_name]
                updated_M = self.estimate_M(M, replicate_mask, simplex_projection_mode=simplex_projection_mode)
                for dataset_name in group_replicates:
                    self.metagene_state[dataset_name] = updated_M

        elif self.metagene_mode == "differential":
            for dataset_index, dataset in enumerate(self.datasets):
                if differentiate_metagenes:
                    M_bars = [self.metagene_state.M_bar[group_name] for group_name in self.metagene_tags[dataset.name]]
                else:
                    M_bars = None

                M = self.metagene_state[dataset.name]
                replicate_mask = [False] * len(self.datasets)
                replicate_mask[dataset_index] = True
                self.metagene_state[dataset.name]= self.estimate_M(M, replicate_mask, M_bar=M_bars, simplex_projection_mode=simplex_projection_mode)

            self.metagene_state.reaverage()

    def nll_metagenes(self):
        with torch.no_grad():
            loss_metagenes = torch.zeros(1, **self.context)
            if self.metagene_mode == "shared":
                for group_name, group_replicates in self.metagene_groups.items():
                    first_dataset_name = group_replicates[0]
                    replicate_mask =  [dataset.name in group_replicates for dataset in self.datasets]
                    M = self.metagene_state[first_dataset_name]
                    loss_M = self.nll_M(M, replicate_mask)
                    loss_metagenes += loss_M

            elif self.metagene_mode == "differential":
                for dataset_index, dataset in enumerate(self.datasets):
                    M_bars = [self.metagene_state.M_bar[group_name] for group_name in self.metagene_tags[dataset.name]]

                    M = self.metagene_state[dataset.name]
                    replicate_mask = [False] * len(self.datasets)
                    replicate_mask[dataset_index] = True
                    loss_M = self.nll_M(M, replicate_mask, M_bar=M_bars)
                    loss_metagenes += loss_M

        return loss_metagenes.cpu().numpy()

    def nll_M(self, M, replicate_mask, M_bar=None):
        _, K = M.shape
        quadratic_factor = torch.zeros([K, K], **self.context)
        linear_term = torch.zeros_like(M)
        
        datasets = [dataset for (use_replicate, dataset) in zip(replicate_mask, self.datasets) if use_replicate]
        Xs = [self.embedding_optimizer.embedding_state[dataset.name] for dataset in datasets]
        Ys = [Y for (use_replicate, Y) in zip(replicate_mask, self.Ys) if use_replicate]
        sigma_yxs = self.sigma_yxs[replicate_mask]

        betas = self.betas[replicate_mask]
        betas /= betas.sum()

        scaled_betas = betas / (sigma_yxs**2)
        
        # ||Y||_2^2
        constant_magnitude = np.array([torch.linalg.norm(Y).item()**2 for Y in Ys]).sum()
    
        constant = (np.array([torch.linalg.norm(self.embedding_optimizer.embedding_state[dataset.name]).item()**2 for dataset in datasets]) * scaled_betas).sum()

        regularization = [self.prior_xs[dataset_index] for dataset_index, dataset in enumerate(datasets)]
        for dataset, X, Y, sigma_yx, scaled_beta in zip(datasets, Xs, Ys, sigma_yxs, scaled_betas):
            # X_c^TX_c
            quadratic_factor.addmm_(X.T, X, alpha=scaled_beta)
            # MX_c^TY_c
            linear_term.addmm_(Y.T, X, alpha=scaled_beta)
    
        differential_regularization_quadratic_factor = torch.zeros((K, K), **self.context)
        differential_regularization_linear_term = torch.zeros(1, **self.context)
        if self.lambda_M > 0 and M_bar is not None:
            differential_regularization_quadratic_factor = self.lambda_M * torch.eye(K, **self.context)
            
            differential_regularization_linear_term = torch.zeros_like(M, **self.context)
            group_weighting = 1 / len(M_bar)
            for group_M_bar in M_bar:
                differential_regularization_linear_term += group_weighting * self.lambda_M * group_M_bar
        
        def compute_loss(M):
            quadratic_factor_grad = M @ (quadratic_factor + differential_regularization_quadratic_factor)
            loss = (quadratic_factor_grad * M).sum()
            linear_term_grad = linear_term + differential_regularization_linear_term
            loss -= 2 * (linear_term_grad * M).sum()
        
            loss += constant

            if self.metagene_mode == "differential" and M_bar is not None:
                differential_regularization_term = (M @ differential_regularization_quadratic_factor * M).sum() - 2 * (differential_regularization_linear_term * M).sum()
                group_weighting = 1 / len(M_bar)
                for group_M_bar in M_bar:
                    differential_regularization_term += group_weighting * self.lambda_M * (group_M_bar * group_M_bar).sum()

            loss /= 2
    
            return loss.item()
        
        loss = compute_loss(M)

        return loss

    def estimate_M(self, M, replicate_mask,
            M_bar=None, n_epochs=10000, tol=1e-3, backend_algorithm='gd Nesterov', simplex_projection_mode=False):
        """Optimize metagene parameters.
    
        M is shared across all replicates.
        min || Y - X MT ||_2^2 / (2 σ_yx^2)
        s.t. || Mk ||_p = 1
        grad = (M XT X - YT X) / (σ_yx^2)
    
        Each replicate may have a slightly different M
        min || Y - X MT ||_2^2 / (2 σ_yx^2) + || M - M_bar ||_2^2 λ_M / 2
        s.t. || Mk ||_p = 1
        grad = ( M XT X - YT X ) / ( σ_yx^2 ) + λ_M ( M - M_bar )
    
        Args:
            M: current estimate of metagene parameters
            betas: weight of each FOV in optimization scheme
            context: context ith which to create PyTorch tensor
            n_epochs: number of epochs 
    
        Returns:
            Updated estimate of metagene parameters.
        """

        G, K = M.shape
        quadratic_factor = torch.zeros([K, K], **self.context)
        linear_term = torch.zeros_like(M)
        # TODO: replace below (and any reference to dataset)
       
        tol /= G

        datasets = [dataset for (use_replicate, dataset) in zip(replicate_mask, self.datasets) if use_replicate]
        Xs = [self.embedding_optimizer.embedding_state[dataset.name] for dataset in datasets]
        Ys = [Y for (use_replicate, Y) in zip(replicate_mask, self.Ys) if use_replicate]
        sigma_yxs = self.sigma_yxs[replicate_mask]

        betas = self.betas[replicate_mask]
        betas /= betas.sum()

        scaled_betas = betas / (sigma_yxs**2)
        
        # ||Y||_2^2
        constant_magnitude = np.array([torch.linalg.norm(Y).item()**2 for Y in Ys]).sum()
    
        constant = (np.array([torch.linalg.norm(self.embedding_optimizer.embedding_state[dataset.name]).item()**2 for dataset in datasets]) * scaled_betas).sum()
        if self.verbose > 1:
            print(f"M constant term: {constant: .1e}")
            print(f"M constant magnitude: {constant_magnitude:.1e}")

        regularization = [self.prior_xs[dataset_index] for dataset_index, dataset in enumerate(datasets)]
        for dataset, X, Y, scaled_beta in zip(datasets, Xs, Ys, scaled_betas):
            # X_c^TX_c
            quadratic_factor.addmm_(X.T, X, alpha=scaled_beta)
            # MX_c^TY_c
            linear_term.addmm_(Y.T, X, alpha=scaled_beta)
    
        # if self.lambda_M > 0 and M_bar is not None:
        #     quadratic_factor.diagonal().add_(self.lambda_M)
        #     linear_term += self.lambda_M * M_bar
        differential_regularization_quadratic_factor = torch.zeros((K, K), **self.context)
        differential_regularization_linear_term = torch.zeros(1, **self.context)
        if self.lambda_M > 0 and M_bar is not None:
            differential_regularization_quadratic_factor = self.lambda_M * torch.eye(K, **self.context)
        
            
            differential_regularization_linear_term = torch.zeros_like(M, **self.context)
            group_weighting = 1 / len(M_bar)
            for group_M_bar in M_bar:
                differential_regularization_linear_term += group_weighting * self.lambda_M * group_M_bar
        #     quadratic_factor.diagonal().add_(self.lambda_M)
        #     linear_term += self.lambda_M * M_bar

        if self.verbose > 1:
            print(f"{get_datetime()} Eigenvalue difference: {torch.max(torch.linalg.eigvals(quadratic_factor + differential_regularization_quadratic_factor).abs()) - torch.max(torch.linalg.eigvals(quadratic_factor).abs())}")
            print(f"M linear term: {torch.linalg.norm(linear_term)}")
            print(f"M regularization linear term: {torch.linalg.norm(differential_regularization_linear_term)}")
            print(f"M linear regularization term ratio: {torch.linalg.norm(differential_regularization_linear_term) / torch.linalg.norm(linear_term)}")
        loss_prev, loss = np.inf, np.nan

        verbose_bar = tqdm(disable=not (self.verbose > 2), bar_format='{desc}{postfix}')
        progress_bar = trange(n_epochs, leave=True, disable=not self.verbose, desc='Updating M', miniters=1000)
    
        def compute_loss_and_gradient(M):
            quadratic_factor_grad = M @ (quadratic_factor + differential_regularization_quadratic_factor)
            loss = (quadratic_factor_grad * M).sum()
            verbose_description = ""
            if self.verbose > 2:
                verbose_description += f"M quadratic term: {loss:.1e}"
            grad = quadratic_factor_grad
            linear_term_grad = linear_term + differential_regularization_linear_term
            loss -= 2 * (linear_term_grad * M).sum()
            grad -= linear_term_grad
        
            loss += constant

            if self.metagene_mode == "differential" and M_bar is not None:
                differential_regularization_term = (M @ differential_regularization_quadratic_factor * M).sum() - 2 * (differential_regularization_linear_term * M).sum()
                group_weighting = 1 / len(M_bar)
                for group_M_bar in M_bar:
                    differential_regularization_term += group_weighting * self.lambda_M * (group_M_bar * group_M_bar).sum()
            
            if self.verbose > 2:
                # print(f"M regularization term: {regularization_term}")
                if self.metagene_mode == "differential":
                    verbose_description += f"M differential regularization term: {differential_regularization_term}"

            loss /= 2
    
    
            if self.M_constraint == 'simplex':
                grad.sub_(grad.sum(0, keepdim=True))
    
            return loss.item(), grad
        
        def estimate_M_nag(M):
            """Estimate M using Nesterov accelerated gradient descent.
    
            Args:
                M (torch.Tensor) : current estimate of meteagene parameters
            """
            loss, grad = compute_loss_and_gradient(M)
            if self.verbose > 1:
                print(f"M NAG Initial Loss: {loss}")
    
            step_size = 1 / torch.linalg.eigvalsh(quadratic_factor).max().item()
            loss = np.inf
            
            optimizer = NesterovGD(M.clone(), step_size)
            for epoch in progress_bar:
                loss_prev = loss
                M_prev = M.clone()
    
                # Update M
                loss, grad = compute_loss_and_gradient(M)
                M = optimizer.step(grad)
                if simplex_projection_mode == "exact":
                    if self.use_inplace_ops:
                        M = project_M_(M, self.M_constraint)
                    else:
                        M = project_M(M, self.M_constraint)
                elif simplex_projection_mode == "approximate":
                    raise NotImplementedError()

                optimizer.set_parameters(M)
    
                dloss = loss_prev - loss
                dM = (M_prev - M).abs().max().item()
                stop_criterion = dM < tol and epoch > 5
                assert not np.isnan(loss)
                if epoch % 5 == 0 or stop_criterion:
                    description = (
                        f'Updating M: loss = {loss:.1e}, '
                        f'%δloss = {dloss / loss:.1e}, '
                        f'δM = {dM:.1e}'
                        # f'lr={step_size_scale:.1e}'
                    )
                    progress_bar.set_description(description)
                if stop_criterion:
                    break
            
            verbose_bar.close()
            progress_bar.close()
            
            loss, grad = compute_loss_and_gradient(M)
            if self.verbose > 1:
                print(f"M NAG Final Loss: {loss}")
    
            return M
        
        if backend_algorithm == 'mu':
            for epoch in progress_bar:
                loss = (((M @ quadratic_factor) * M).sum() - 2 * (M * linear_term).sum() + constant) / 2
                loss = loss.item()
                numerator = linear_term
                denominator = M @ quadratic_factor
                multiplicative_factor = numerator / denominator
    
                M_prev = M.clone()
                # multiplicative_factor.clip_(max=10)
                M *= multiplicative_factor
                if simplex_projection_mode == "exact":
                    if self.use_inplace_ops:
                        M = project_M_(M, self.M_constraint)
                    else:
                        M = project_M(M, self.M_constraint)
                elif simplex_projection_mode == "approximate":
                    pass
                dM = M_prev.sub(M).abs_().max().item()
    
                stop_criterion = dM < tol and epoch > 5
                if epoch % 1000 == 0 or stop_criterion:
                    progress_bar.set_description(
                        f'Updating M: loss = {loss:.1e}, '
                        f'%δloss = {(loss_prev - loss) / loss:.1e}, '
                        f'δM = {dM:.1e}'
                    )
                if stop_criterion:
                    break
    
        elif backend_algorithm == 'gd':
            step_size = 1 / torch.linalg.eigvalsh(quadratic_factor).max().item()
            step_size_scale = 1
            loss, grad = compute_loss_and_gradient(M)
            dM = dloss = np.inf
            for epoch in progress_bar:
                M_new = M.sub(grad, alpha=step_size * step_size_scale)
                if simplex_projection_mode == "exact":
                    if self.use_inplace_ops:
                        M = project_M_(M_new, self.M_constraint)
                    else:
                        M = project_M(M_new, self.M_constraint)
                elif simplex_projection_mode == "approximate":
                    pass
                loss_new, grad_new = compute_loss_and_gradient(M_new)
                if loss_new < loss or step_size_scale == 1:
                    dM = (M_new - M).abs().max().item()
                    dloss = loss - loss_new
                    M[:] = M_new
                    loss = loss_new
                    grad = grad_new
                    step_size_scale *= 1.1
                else:
                    step_size_scale *= .5
                    step_size_scale = max(step_size_scale, 1.)
    
                stop_criterion = dM < tol and epoch > 5
                if epoch % 1000 == 0 or stop_criterion:
                    progress_bar.set_description(
                        f'Updating M: loss = {loss:.1e}, '
                        f'%δloss = {dloss / loss:.1e}, '
                        f'δM = {dM:.1e}, '
                        f'lr={step_size_scale:.1e}'
                    )
                if stop_criterion:
                    break
    
        elif backend_algorithm == 'gd Nesterov':
            M = estimate_M_nag(M)
        else:
            raise NotImplementedError
       
        return M

    def update_sigma_yx(self):
        """Update sigma_yx for each replicate.

        """

        squared_terms = [torch.addmm(Y, self.embedding_optimizer.embedding_state[dataset.name], self.metagene_state[dataset.name].T, alpha=-1) for Y, dataset in zip(self.Ys, self.datasets)]
        squared_loss = np.array([torch.linalg.norm(squared_term, ord="fro").item() ** 2 for squared_term in squared_terms])
        # squared_loss = np.array([
        #     torch.linalg.norm(, ord='fro').item() ** 2
        #     for Y, X, dataset, replicate in zip(Ys, self.Xs, self.datasets, self.repli_list)
        # ])
        num_replicates = len(self.datasets)
        sizes = np.array([dataset.X.size for dataset in self.datasets])
        if self.sigma_yx_inv_mode == 'separate':
            self.sigma_yxs[:] = np.sqrt(squared_loss / sizes)
        elif self.sigma_yx_inv_mode == 'average':
            sigma_yx = np.sqrt(np.dot(self.betas, squared_loss) / np.dot(self.betas, sizes))
            self.sigma_yxs[:] = np.full(num_replicates, float(sigma_yx))
        else:
            raise NotImplementedError


    def nll_sigma_yx(self):
        with torch.no_grad():
            squared_terms = [torch.addmm(Y, self.embedding_optimizer.embedding_state[dataset.name], self.metagene_state[dataset.name].T, alpha=-1) for Y, dataset in zip(self.Ys, self.datasets)]
            squared_loss = np.array([torch.linalg.norm(squared_term, ord="fro").item() ** 2 for squared_term in squared_terms])

        return squared_loss.sum()



class MetageneState(dict):
    """State to store metagene parameters during Popari optimization.

    Metagene state can be shared across replicates or maintained separately for each replicate.

    Attributes:
        datasets: A reference to the list of PopariDatasets that are being optimized.
        context: Parameters to define the context for PyTorch tensor instantiation.
        metagenes: A PyTorch tensor containing all metagene parameters.
    """
    def __init__(self, K, datasets, groups, tags, mode="shared", M_constraint="simplex", initial_context=None, context=None):
        self.datasets = datasets
        self.groups = groups
        self.tags = tags
        self.initial_context = initial_context if initial_context else {"device": "cpu", "dtype": torch.float32}
        self.context = context if context else {"device": "cpu", "dtype": torch.float32}
        self.M_constraint = M_constraint
        if mode == "shared":
            _, num_genes = self.datasets[0].shape
            self.metagenes = torch.zeros((len(self.groups), num_genes, K), **self.context)
            for (group_name, group_replicates), group_metagenes in zip(self.groups.items(), self.metagenes):
                for dataset_name in group_replicates:
                    self.__setitem__(dataset_name, group_metagenes)
        
        elif mode == "differential":
            _, num_genes = self.datasets[0].shape
            self.metagenes = torch.zeros((len(self.datasets), num_genes, K), **self.context)
            for dataset, replicate_metagenes in zip(self.datasets, self.metagenes):
                self.__setitem__(dataset.name, replicate_metagenes)
            
            self.M_bar = {}
            for group_name, group_replicates in self.groups.items():
                self.M_bar[group_name] = torch.zeros((num_genes, K), **self.context)
                for dataset_name in group_replicates:
                    self.M_bar[group_name] += self.__getitem__(dataset_name)
                self.M_bar[group_name].div_(len(group_replicates))
        
    def reaverage(self):
        # Set M_bar to average of self.Ms (memory efficient)
        for group_name, group_replicates in self.groups.items():
            self.M_bar[group_name].zero_()
            for dataset_name in group_replicates:
                self.M_bar[group_name].add_(self.__getitem__(dataset_name))
            self.M_bar[group_name].div_(len(group_replicates))
            self.M_bar[group_name][:] = project_M(self.M_bar[group_name], self.M_constraint)

class SpatialAffinityState(dict):
    def __init__(self, K, metagene_state, datasets, groups, tags, betas, scaling=10, lr=1e-3, mode="shared lookup", initial_context=None, context=None):
        self.datasets = datasets
        self.groups = groups
        self.tags = tags
        self.metagene_state = metagene_state
        self.K = K
        self.mode = mode
        self.initial_context = initial_context if initial_context else {"device": "cpu", "dtype": torch.float32}
        self.context = context if context else {"device": "cpu", "dtype": torch.float32}
        self.betas = betas
        self.scaling = scaling
        self.lr = lr
        self.optimizers = {}
        super().__init__()

        num_replicates = len(self.datasets)
        if mode == "shared lookup":
            metagene_affinities = torch.zeros((K, K), **self.initial_context)
            for dataset in self.datasets:
                self.__setitem__(dataset.name, metagene_affinities)

        elif mode == "differential lookup":
            for dataset_index, dataset in enumerate(self.datasets):
                metagene_affinity = torch.zeros((K, K), **self.initial_context)
                self.__setitem__(dataset.name, metagene_affinity)

            self.spatial_affinity_bar = {}
            for group_name in self.groups:
                self.spatial_affinity_bar[group_name] = torch.zeros((self.K, self.K), **self.context)

        elif mode == "attention":
            metagene_affinities = 0 # attention mechanism here
            for dataset_index, dataset in enumerate(self.datasets):
                self.__setitem__(dataset.name, metagene_affinities[dataset_index])

    def initialize(self, initial_embeddings):
        use_spatial_info = ["adjacency_list" in dataset.obs for dataset in self.datasets]

        if not any(use_spatial_info):
            return

        num_replicates = len(self.datasets)
        Sigma_x_invs = torch.zeros([num_replicates, self.K, self.K], **self.initial_context)
        for replicate, (beta, initial_embedding, is_spatial_replicate, dataset) in enumerate(zip(self.betas, initial_embeddings, use_spatial_info, self.datasets)):
            if not is_spatial_replicate:
                continue
            
            adjacency_list = dataset.obs["adjacency_list"]
            X = initial_embedding
            Z = X / torch.linalg.norm(X, dim=1, keepdim=True, ord=1)
            edges = np.array([(i, j) for i, e in enumerate(adjacency_list) for j in e])
   
            x = Z[edges[:, 0]]
            y = Z[edges[:, 1]]
            x = x - x.mean(dim=0, keepdim=True)
            y = y - y.mean(dim=0, keepdim=True)
            y_std = y.std(dim=0, keepdim=True)
            x_std = x.std(dim=0, keepdim=True)
            corr = (y / y_std).T @ (x / x_std) / len(x)
            Sigma_x_invs[replicate] = -beta * corr
    
        # Symmetrizing and zero-centering Sigma_x_inv
        Sigma_x_invs = (Sigma_x_invs + torch.transpose(Sigma_x_invs, 1, 2)) / 2
        Sigma_x_invs -= Sigma_x_invs.mean(dim=(1, 2), keepdims=True)
        Sigma_x_invs *= self.scaling

        if self.mode == "shared lookup":
            for group_name, group_replicates in self.groups.items():
                first_dataset = self.datasets[0]
                shared_affinity = self.__getitem__(first_dataset.name)
                for dataset_index, dataset in enumerate(self.datasets):
                    if dataset.name in group_replicates:
                        shared_affinity[:] += Sigma_x_invs[dataset_index]

                shared_affinity.div_(len(group_replicates))
                optimizer = torch.optim.Adam(
                    [shared_affinity],
                    lr=self.lr,
                    betas=(.5, .9),
                )
                self.optimizers[group_name] = optimizer
                
        elif self.mode == "differential lookup":
            for group_name, group_replicates in self.groups.items():
                for dataset_index, dataset in enumerate(self.datasets):
                    if dataset.name in group_replicates:
                        self.spatial_affinity_bar[group_name] += Sigma_x_invs[dataset_index]

                self.spatial_affinity_bar[group_name].div_(len(group_replicates))

            for dataset_index, dataset in enumerate(self.datasets):
                differential_affinity = self.__getitem__(dataset.name)
                differential_affinity[:] = Sigma_x_invs[dataset_index]
                optimizer = torch.optim.Adam(
                    [differential_affinity],
                    lr=self.lr,
                    betas=(.5, .9),
                )
                self.optimizers[dataset.name] = optimizer
                
        elif self.mode == "attention":
            #TODO: initialize with gradient descent
            raise NotImplementedError()
    
    def reaverage(self):
        # Set spatial_affinity_bar to average of self.spatial_affinitys (memory efficient)
        for group_name, group_replicates in self.groups.items():
            self.spatial_affinity_bar[group_name].zero_()
            for dataset_name in group_replicates:
                self.spatial_affinity_bar[group_name].add_(self.__getitem__(dataset_name))
            self.spatial_affinity_bar[group_name].div_(len(group_replicates))
