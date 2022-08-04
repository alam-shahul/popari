from typing import Sequence
import logging, time, gc
from tqdm.auto import tqdm, trange

import anndata as ad
import scanpy as sc
import squidpy as sq

import numpy as np
from scipy.sparse import csr_matrix

from spicemix.sample_for_integral import integrate_of_exponential_over_simplex
from spicemix.util import NesterovGD, IndependentSet, project2simplex, project_M, get_datetime

import torch
import torch.nn.functional as F
from torch.nn import MultiheadAttention

class SpiceMixDataset(ad.AnnData):
    """Wrapper around AnnData object. Allows for preprocessing of dataset for SpiceMix.


    """

    def __init__(self, dataset, replicate_name, coordinates_key="spatial"):
        """
        """
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

    def compute_spatial_neighbors(self):
        """Compute neighbor graph based on spatial coordinates.

        """

        sq.gr.spatial_neighbors(self, coord_type="generic", delaunay=True)
        distance_matrix, adjacency_matrix = self.obsp["spatial_distances"], self.obsp["spatial_connectivities"]
        self.obsp["spatial_connectivities"] = SpiceMixDataset.remove_connectivity_artifacts(distance_matrix, adjacency_matrix)
        self.obsp["adjacency_matrix"] = self.obsp["spatial_connectivities"]
        
        num_cells, _ = self.obsp["adjacency_matrix"].shape

        adjacency_list = [[] for _ in range(num_cells)]
        for x, y in zip(*self.obsp["adjacency_matrix"].nonzero()):
            adjacency_list[x].append(y)

        self.obs["adjacency_list"] = adjacency_list

    @staticmethod
    def remove_connectivity_artifacts(sparse_distance_matrix, sparse_adjacency_matrix, threshold=94.5):
        dense_distances = sparse_distance_matrix.toarray()
        distances = sparse_distance_matrix.data
        cutoff = np.percentile(distances, threshold)
        mask = dense_distances < cutoff

        return csr_matrix(sparse_adjacency_matrix * mask)
 
class EmbeddingOptimizer():
    """Optimizer and state for SpiceMix embeddings.

    """

    def __init__(self, K, Ys, datasets, context=None):
        self.datasets = datasets
        self.K = K
        self.Ys = Ys
        self.context = context if context else {}
        self.embedding_state = EmbeddingState(K, self.datasets, context=self.context)

    def link(self, parameter_optimizer):
        self.parameter_optimizer = parameter_optimizer

    def update_embeddings(self):
        """Update SpiceMixPlus embeddings according to optimization scheme.

        """
        logging.info(f'{get_datetime()}Updating latent states')

        loss_list = []
        for dataset_index, dataset  in enumerate(self.datasets):
            is_spatial_replicate = ("adjacency_list" in dataset.obs)
            sigma_yx = self.parameter_optimizer.sigma_yxs[dataset_index]
            Y = self.Ys[dataset_index]
            X = self.embedding_state[dataset.name]
            M = self.parameter_optimizer.metagene_state[dataset.name]
            prior_x_mode = self.parameter_optimizer.prior_x_modes[dataset_index]
            prior_x = self.parameter_optimizer.prior_xs[dataset_index]
            if not is_spatial_replicate:
                loss, self.embedding_state[dataset.name] = self.estimate_weight_wonbr(
                    Y, M, X, sigma_yx, prior_x_mode, prior_x, dataset, context=self.context)
            else:
                loss, self.embedding_state[dataset.name] = self.estimate_weight_wnbr(
                    Y, M, X, sigma_yx, prior_x_mode, prior_x, dataset, context=self.context)

            loss_list.append(loss)

    @torch.no_grad()
    def estimate_weight_wonbr(self, Y, M, X, sigma_yx, prior_x_mode, prior_x, dataset, context, n_epochs=1000, tol=1e-6, update_alg='gd', verbose=True):
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
            
        progress_bar = trange(n_epochs, leave=True, disable=not verbose, miniters=1000)
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
            if epoch % 1000 == 0 or do_stop:
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
    def estimate_weight_wnbr(self, Y, M, X, sigma_yx, prior_x_mode, prior_x, dataset, context, n_epochs=1000, tol=1e-5, update_alg='nesterov'):
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
        base_step_size = 1 / torch.linalg.eigvalsh(MTM).max().item()
        S = torch.linalg.norm(X, dim=1, ord=1, keepdim=True)
        Z = X / S
        N = len(Z)
        
        E_adjacency_list = dataset.obs["adjacency_list"]
        Sigma_x_inv = self.parameter_optimizer.spatial_affinity_state[dataset.name]
    
        def get_adjacency_matrix(adjacency_list):
            edges = [(i, j) for i, e in enumerate(adjacency_list) for j in e]
            adjacency_matrix = torch.sparse_coo_tensor(np.array(edges).T, np.ones(len(edges)), size=[len(adjacency_list), N], **context)
            return adjacency_matrix
    
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
    
        # Debugging for loss of M
        def compute_loss_and_gradient(M):
            quadratic_factor_grad = M @ quadratic_factor
            loss = (quadratic_factor_grad * M).sum()
            grad = quadratic_factor_grad
            linear_term_grad = linear_term
            loss -= 2 * (linear_term_grad * M).sum()
            grad -= linear_term_grad
            
            loss += constant
            loss /= 2
    
            if M_constraint == 'simplex':
                grad.sub_(grad.sum(0, keepdim=True))
    
            return loss.item(), grad
        
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
            for idx in IndependentSet(E_adjacency_list, batch_size=128):
                step_size_scale = 1
                quad_batch = MTM
                linear_batch = YM[idx] * S[idx] - get_adjacency_matrix(E_adjacency_list[idx]) @ Z @ Sigma_x_inv
                Z_batch = Z[idx].contiguous()
                S_batch = S[idx].contiguous()
                step_size_batch = step_size[idx].contiguous()
                func, grad = calc_func_grad(Z_batch, S_batch, quad_batch, linear_batch)
                while True:
                    Z_batch_new = Z_batch - step_size_batch * step_size_scale * grad
                    result = project2simplex(Z_batch_new, dim=1)
                    Z_batch_new = 0
                    Z_batch_new = result
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
    
        def update_z_gd_nesterov(Z, verbose=False):
            pbar = trange(N, leave=False, disable=True, desc='Updating Z w/ nbrs via Nesterov GD')
            batch_number = 0
            func, grad = calc_func_grad(Z, S, MTM, YM * S - get_adjacency_matrix(E_adjacency_list) @ Z @ Sigma_x_inv / 2)
            for idx in IndependentSet(E_adjacency_list, batch_size=256):
                quad_batch = MTM
                linear_batch_spatial = - get_adjacency_matrix(E_adjacency_list[idx]) @ Z @ Sigma_x_inv
                Z_batch = Z[idx].contiguous()
                S_batch = S[idx].contiguous()
                    
                optimizer = NesterovGD(Z_batch, base_step_size / S_batch.square())
                ppbar = trange(10000, leave=False, disable=True)
                for i_iter in ppbar:
                    update_s() # TODO: update S_batch directly
                    S_batch = S[idx].contiguous()
                    linear_batch = linear_batch_spatial + YM[idx] * S_batch
                    if i_iter == 0:
                        func, grad = calc_func_grad(Z_batch, S_batch, quad_batch, linear_batch)
                        func, grad = calc_func_grad(Z, S, MTM, YM * S - get_adjacency_matrix(E_adjacency_list) @ Z @ Sigma_x_inv / 2)
                    NesterovGD.step_size = base_step_size / S_batch.square() # TM: I think this converges as s converges
                    func, grad = calc_func_grad(Z_batch, S_batch, quad_batch, linear_batch)
                    Z_batch_prev = Z_batch.clone()
                    Z_batch_copy = optimizer.step(grad)
                    Z_batch = project2simplex(Z_batch_copy, dim=1)
                    optimizer.set_parameters(Z_batch)
                    dZ = (Z_batch_prev - Z_batch).abs().max().item()
                    Z[idx] = Z_batch
                    ppbar.set_description(f'func={func:.1e}, dZ={dZ:.1e}')
                    if dZ < tol:
                        break
                ppbar.close()
                
                Z[idx] = Z_batch
                func, grad = calc_func_grad(Z_batch, S_batch, quad_batch, linear_batch)
                func, grad = calc_func_grad(Z, S, MTM, YM * S - get_adjacency_matrix(E_adjacency_list) @ Z @ Sigma_x_inv /2 )
                if verbose:
                    print(f"Z loss: {func}")
                pbar.update(len(idx))
            pbar.close()
            func, grad = calc_func_grad(Z, S, MTM, YM * S - get_adjacency_matrix(E_adjacency_list) @ Z @ Sigma_x_inv / 2)
            if verbose:
                print(f"Z final loss: {func}")
    
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
                loss += ((dataset.obsp["adjacency_matrix"] @ Z) @ Sigma_x_inv).mul(Z).sum() / 2
            loss = loss.item()
            # assert loss <= loss_prev, (loss_prev, loss)
            return loss
    
        # TM: consider combine compute_loss and update_z to remove a call to torch.sparse.mm
        # TM: the above idea is not practical if we update only a subset of nodes each time
    
        loss = np.inf
        pbar = trange(n_epochs, desc='Updating weight w/ neighbors')
    
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

class EmbeddingState(dict):
    """Collections of cell embeddings for all ST replicates.

    Attributes:
        K: embedding dimension:

    """
    def __init__(self, K: int, datasets: Sequence[SpiceMixDataset], context=None):
        self.datasets = datasets
        self.K = K
        self.context = context if context else {}
        super().__init__()

        for dataset in self.datasets:
            num_cells, _ = dataset.shape
            initial_embedding = torch.zeros((num_cells, K), **self.context)
            self.__setitem__(dataset.name, initial_embedding)

    def normalize(self):
        """Normalize embeddings per each cell.
        
        This step helps to make cell embeddings comparable, and facilitates downstream tasks like clustering.

        """
        # TODO: implement
        pass

class ParameterOptimizer():
    """Optimizer and state for SpiceMix parameters.

    """

    def __init__(self, K, Ys, datasets, betas, prior_x_modes,
            spatial_affinity_regularization_power=2,
            lambda_Sigma_x_inv=1e-2,
            spatial_affinity_mode="shared lookup",
            metagene_mode="shared",
            M_constraint="simplex",
            sigma_yx_inv_mode="separate",
            context=None
    ):
        self.datasets = datasets
        self.spatial_affinity_mode = spatial_affinity_mode
        self.K = K
        self.Ys = Ys
        self.sigma_yx_inv_mode = sigma_yx_inv_mode
        self.lambda_Sigma_x_inv = lambda_Sigma_x_inv
        self.metagene_mode = metagene_mode
        self.M_constraint = M_constraint
        self.prior_x_modes = prior_x_modes
        self.betas = betas
        self.context = context if context else {}
        self.spatial_affinity_regularization_power = spatial_affinity_regularization_power

    def link(self, embedding_optimizer):
        self.embedding_optimizer = embedding_optimizer
        self.metagene_state = MetageneState(self.K, self.datasets, mode=self.metagene_mode, context=self.context)
        self.spatial_affinity_state = SpatialAffinityState(self.K, self.metagene_state, self.datasets, self.betas, mode=self.spatial_affinity_mode, context=self.context)
        
        if all(prior_x_mode == 'exponential shared fixed' for prior_x_mode in self.prior_x_modes):
            self.prior_xs = [(torch.ones(self.K, **self.context),) for _ in range(len(self.datasets))]
        elif all(prior_x_mode == None for prior_x_mode in self.prior_x_modes):
            self.prior_xs = [(torch.zeros(self.K, **self.context),) for _ in range(len(self.datasets))]
        else:
            raise NotImplementedError

        self.sigma_yxs = np.zeros(len(self.datasets))
       
    def scale_metagenes(self):
        if self.M_constraint == 'simplex':
            scale_factor = torch.linalg.norm(self.metagene_state.metagenes, axis=0, ord=1, keepdim=True)
        elif self.M_constraint == 'unit_sphere':
            scale_factor = torch.linalg.norm(self.metagene_state.metagenes, axis=0, ord=2, keepdim=True)
        else:
            raise NotImplementedError
        
        self.metagene_state.metagenes.div_(scale_factor)
        for dataset in self.datasets:
            self.embedding_optimizer.embedding_state[dataset.name].mul_(scale_factor)

    def estimate_Sigma_x_inv(self, Sigma_x_inv, replicate_mask, Sigma_x_inv_bar=None, constraint="clamp", n_epochs=1000):
        """Optimize Sigma_x_inv parameters.
    
       
        Differential mode:
        grad = ( M XT X - YT X ) / ( σ_yx^2 ) + λ_Sigma_x_inv ( Sigma_x_inv - Sigma_x_inv_bar )
    
        Args:
            Xs: list of latent expression embeddings for each FOV.
            Sigma_x_inv: previous estimate of Σx-1
    
        """
        datasets = [dataset for (use_replicate, dataset) in zip(replicate_mask, self.datasets) if use_replicate]
        Xs = [self.embedding_optimizer.embedding_state[dataset.name] for dataset in datasets]
        spatial_flags = ["adjacency_list" in dataset.obs for dataset in datasets]
        optimizer = self.spatial_affinity_state.optimizer

        num_edges_per_fov = [sum(map(len, dataset.obs["adjacency_list"])) for dataset in datasets]
    
        if not any(sum(map(len, dataset.obs["adjacency_list"])) > 0 and u for dataset, u in zip(datasets, spatial_flags)):
            return
    
        linear_term_coefficient = torch.zeros_like(Sigma_x_inv).requires_grad_(False)
        size_factors = [torch.linalg.norm(X, axis=1, ord=1, keepdim=True) for X in Xs ]
        Zs = [X / size_factor for X, size_factor in zip(Xs, size_factors)]
        nus = [] # sum of neighbors' z
        weighted_total_cells = 0
        for Z, dataset, use_spatial, beta in zip(Zs, datasets, spatial_flags, self.betas):
            adjacency_matrix = dataset.obsp["adjacency_matrix"]
            adjacency_list = dataset.obs["adjacency_list"]

            if use_spatial:
                nu = adjacency_matrix @ Z
                linear_term_coefficient = linear_term_coefficient.addmm_(Z.T, nu, alpha=beta)
            else:
                nu = None
            nus.append(nu)
            weighted_total_cells += beta * sum(map(len, adjacency_list))
            del Z, adjacency_matrix
        # linear_term_coefficient = (linear_term_coefficient + linear_term_coefficient.T) / 2 # should be unnecessary as long as adjacency_list is symmetric
    
        history = []
        Sigma_x_inv.requires_grad_(True)
   
        loss_prev, loss = np.inf, np.nan
        progress_bar = trange(n_epochs, desc='Updating Σx-1')
        Sigma_x_inv_best, loss_best, epoch_best = None, np.inf, -1
        dSigma_x_inv = np.inf
        early_stop_epoch_count = 0
        for epoch in progress_bar:
            optimizer.zero_grad()
    
            # Compute loss 
            linear_term = Sigma_x_inv.view(-1) @ linear_term_coefficient.view(-1)
            regularization = self.lambda_Sigma_x_inv * Sigma_x_inv.pow(self.spatial_affinity_regularization_power).sum() * weighted_total_cells / 2
            if Sigma_x_inv_bar is not None:
                regularization += self.lambda_Sigma_x_inv * 100 * (Sigma_x_inv_bar - Sigma_x_inv).pow(2).sum() * weighted_total_cells / 2
            
            log_partition_function = 0
            for Z, nu, beta in zip(Zs, nus, self.betas):
                if nu is None:
                    continue
                assert torch.isfinite(nu).all()
                assert torch.isfinite(Sigma_x_inv).all()
                eta = nu @ Sigma_x_inv
                logZ = integrate_of_exponential_over_simplex(eta)
                log_partition_function += beta * logZ.sum()
    
            loss = (linear_term + regularization + log_partition_function) / weighted_total_cells
    
            if loss < loss_best:
                Sigma_x_inv_best = Sigma_x_inv.clone().detach()
                loss_best = loss.item()
                epoch_best = epoch
    
            history.append((Sigma_x_inv.detach().cpu().numpy(), loss.item()))
    
            with torch.no_grad():
                Sigma_x_inv_prev = Sigma_x_inv.clone().detach()
    
            loss.backward()
            Sigma_x_inv.grad = (Sigma_x_inv.grad + Sigma_x_inv.grad.T) / 2
            optimizer.step()
            with torch.no_grad():
                if constraint == "clamp":
                    Sigma_x_inv.clamp_(min=-self.spatial_affinity_state.scaling, max=self.spatial_affinity_state.scaling)
            # with torch.no_grad():
            #   Sigma_x_inv -= Sigma_x_inv.mean()
    
            loss = loss.item()
            dloss = loss_prev - loss
            loss_prev = loss
    
            with torch.no_grad():
                dSigma_x_inv = Sigma_x_inv_prev.sub(Sigma_x_inv).abs().max().item()
            
            progress_bar.set_description(
                f'Updating Σx-1: loss = {dloss:.1e} -> {loss:.1e} '
                f'δΣx-1 = {dSigma_x_inv:.1e} '
                f'Σx-1 range = {Sigma_x_inv.min().item():.1e} ~ {Sigma_x_inv.max().item():.1e}'
            )
    
            if Sigma_x_inv.grad.abs().max() < 1e-4 and dSigma_x_inv < 1e-4:
                early_stop_epoch_count += 1
            else:
                early_stop_epoch_count = 0
            if early_stop_epoch_count >= 10 or epoch > epoch_best + 100:
                break
    
        Sigma_x_inv = Sigma_x_inv_best
        Sigma_x_inv.requires_grad_(False)
       
        print(Sigma_x_inv)
        
        return Sigma_x_inv, loss * weighted_total_cells

    def update_spatial_affinity(self):
        if self.spatial_affinity_mode == "shared lookup":
            first_dataset = self.datasets[0]
            replicate_mask =  np.full(len(self.datasets), True)
            Sigma_x_inv = self.spatial_affinity_state[first_dataset.name]
            Sigma_x_inv, loss = self.estimate_Sigma_x_inv(Sigma_x_inv, replicate_mask)
            with torch.no_grad():
               for dataset in self.datasets:
                  self.spatial_affinity_state[dataset.name][:] = Sigma_x_inv

            # with torch.no_grad():
            #     # Note: in-place update is necessary here in order for optimizer to track same object
            #     self.Sigma_x_inv[:] = updated_Sigma_x_inv
            #     for replicate, dataset in zip(self.repli_list, self.datasets):
            #         dataset.uns["Sigma_x_inv"][f"{replicate}"][:] = updated_Sigma_x_inv
        elif self.spatial_affinity_mode == "differential lookup":
            for index, dataset in enumerate(self.datasets):
                M = self.metagene_state[dataset.name]
                replicate_mask =  np.full(len(self.datasets), False)
                replicate_mask[index] = True
                Sigma_x_inv = self.spatial_affinity_state[dataset.name]
                Sigma_x_inv, loss = self.estimate_Sigma_x_inv(Sigma_x_inv, replicate_mask)
                with torch.no_grad():
                    self.spatial_affinity_state[dataset.name][:] = Sigma_x_inv

    def update_metagenes(self, Xs):
        if self.metagene_mode == "shared":
            first_dataset = self.datasets[0]
            replicate_mask =  np.full(len(self.datasets), True)
            M = self.metagene_state[first_dataset.name]
            updated_M = self.estimate_M(M, replicate_mask)
            for dataset in self.datasets:
                self.metagene_state[dataset.name] = updated_M

        elif self.metagene_mode == "differential":
            for index, dataset in enumerate(self.datasets):
                M = self.metagene_state[dataset.name]
                replicate_mask =  np.full(len(self.datasets), False)
                replicate_mask[index] = True
                self.metagene_state[dataset.name]= self.estimate_M(M, replicate_mask)

    def estimate_M(self, M, replicate_mask,
            M_bar=None, lambda_M=0,
            n_epochs=10000, tol=1e-6, backend_algorithm='gd Nesterov', verbose=False):
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
            Ys: list of gene expression data
            Xs: list of estimated hidden states
            M (torch.Tensor): current estimate of metagene parameters
            betas: weight of each FOV in optimization scheme
            context: context ith which to create PyTorch tensor
            n_epochs: number of epochs 
    
        Returns:
            Updated estimate of metagene parameters.
        """

        _, K = M.shape
        quadratic_factor = torch.zeros([K, K], **self.context)
        linear_term = torch.zeros_like(M)
        # TODO: replace below (and any reference to dataset)
        
        datasets = [dataset for (use_replicate, dataset) in zip(replicate_mask, self.datasets) if use_replicate]
        Xs = [self.embedding_optimizer.embedding_state[dataset.name] for dataset in datasets]
        sigma_yxs = np.array([dataset.uns["sigma_yx"] for dataset in datasets])
        scaled_betas = self.betas[replicate_mask] / (sigma_yxs**2)
        
        # ||Y||_2^2
        constant_magnitude = np.array([torch.linalg.norm(Y).item()**2 for Y in self.Ys]).sum()
    
        constant = (np.array([torch.linalg.norm(self.embedding_optimizer.embedding_state[dataset.name]).item()**2 for dataset in datasets]) * scaled_betas).sum()
        regularization = [self.prior_xs[dataset_index] for dataset_index, dataset in enumerate(datasets)]
        for dataset, X, sigma_yx, scaled_beta in zip(datasets, Xs, sigma_yxs, scaled_betas):
            # X_c^TX_c
            quadratic_factor.addmm_(X.T, X, alpha=scaled_beta)
            # MX_c^TY_c
            linear_term.addmm_(torch.tensor(dataset.X, **self.context).T.to(X.device), X, alpha=scaled_beta)
    
        if lambda_M > 0 and M_bar is not None:
            quadratic_factor.diagonal().add_(lambda_M)
            linear_term += lambda_M * M_bar
    
        loss_prev, loss = np.inf, np.nan
        progress_bar = trange(n_epochs, leave=True, disable=not verbose, desc='Updating M', miniters=1000)
    
        def compute_loss_and_gradient(M, verbose=False):
            quadratic_factor_grad = M @ quadratic_factor
            loss = (quadratic_factor_grad * M).sum()
            if verbose:
                print(f"M quadratic term: {loss}")
            grad = quadratic_factor_grad
            linear_term_grad = linear_term
            loss -= 2 * (linear_term_grad * M).sum()
            grad -= linear_term_grad
        
            loss += constant
            regularization_term = torch.sum(torch.Tensor([(regularizer[0] * X).sum() for regularizer, X in zip(regularization, Xs)]))
            loss += regularization_term
            if verbose:
                print(f"M regularization term: {regularization_term}")
                print(f"M constant term: {constant}")
                print(f"M constant magnitude: {constant_magnitude}")
            loss /= 2
    
    
            if self.M_constraint == 'simplex':
                grad.sub_(grad.sum(0, keepdim=True))
    
            return loss.item(), grad
        
        def estimate_M_nag(M, verbose=False):
            """Estimate M using Nesterov accelerated gradient descent.
    
            Args:
                M (torch.Tensor) : current estimate of meteagene parameters
            """
            loss, grad = compute_loss_and_gradient(M, verbose=verbose)
            if verbose:
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
                M = project_M(M, self.M_constraint)
                optimizer.set_parameters(M)
    
                dloss = loss_prev - loss
                dM = (M_prev - M).abs().max().item()
                stop_criterion = dM < tol and epoch > 5
                assert not np.isnan(loss)
                if epoch % 1000 == 0 or stop_criterion:
                    progress_bar.set_description(
                        f'Updating M: loss = {loss:.1e}, '
                        f'%δloss = {dloss / loss:.1e}, '
                        f'δM = {dM:.1e}'
                        # f'lr={step_size_scale:.1e}'
                    )
                if stop_criterion:
                    break
            
            loss, grad = compute_loss_and_gradient(M, verbose=True)
            if verbose:
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
                M = project_M(M, self.M_constraint)
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
                M_new = project_M(M_new, self.M_constraint)
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
            M = estimate_M_nag(M, verbose=verbose)
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

class MetageneState(dict):
    """State to store metagene parameters during SpiceMixPlus optimization.

    Metagene state can be shared across replicates or maintained separately for each replicate.

    Attributes:
        datasets: A reference to the list of SpiceMixDatasets that are being optimized.
        context: Parameters to define the context for PyTorch tensor instantiation.
        metagenes: A PyTorch tensor containing all metagene parameters.
    """
    def __init__(self, K, datasets, mode="shared", context=None):
        self.datasets = datasets
        self.context = context if context else {}
        if mode == "shared":
            _, num_genes = self.datasets[0].shape
            self.metagenes = torch.zeros((num_genes, K), **self.context)
            for dataset in self.datasets:
                self.__setitem__(dataset.name, self.metagenes)
        
        elif mode == "differential":
            self.metagenes = torch.zeros((len(self.datasets), num_genes, K), **self.context)
            self.M_bar = torch.mean(self.metagenes, axis=0)
            for dataset, replicate_metagenes in zip(self.datasets, self.metagenes):
                self.__setitem__(dataset.name, replicate_metagenes)
        
    def reaverage(self):
        # Set M_bar to average of self.Ms (memory efficient)
        self.M_bar = torch.zeros.zero_()
        for dataset in self.datasets:
            self.M_bar.add_(self.__getitem__(dataset.name))
        self.M_bar.div_(len(self.datasets))

class SpatialAffinityState(dict):
    def __init__(self, K, metagene_state, datasets, betas, scaling=10, mode="shared lookup", context=None):
        self.datasets = datasets
        self.metagene_state = metagene_state
        self.K = K
        self.mode = "shared lookup"
        self.context = context if context else {}
        self.betas = betas
        self.scaling = scaling
        self.optimizer = None
        super().__init__()

        num_replicates = len(self.datasets)
        if mode == "shared lookup":
            self.spatial_affinity = SpatialAffinityLookup(K=self.K, context=self.context)
            metagene_affinities = self.spatial_affinity.get_metagene_affinities()[0]
            for dataset in self.datasets:
                self.__setitem__(dataset.name, metagene_affinities)

        elif mode == "differential lookup":
            self.spatial_affinity = SpatialAffinityLookup(K=self.K, num_replicates=num_replicates, context=self.context)
            self.spatial_affinity_bar = torch.zeros_like(self.spatial_affinity.get_metagene_affinities()[0])

        elif mode == "attention":
            self.spatial_affinity = SpatialAffinityAttention(K=self.K, num_replicates=num_replicates, context=self.context)
            metagene_affinities = self.spatial_affinity.get_metagene_affinities(self.metagene_state[dataset.name])
            for dataset_index, dataset in enumerate(self.datasets):
                self.__setitem__(dataset.name, metagene_affinities[dataset_index])

    def initialize(self, initial_embeddings, scaling=10):
        use_spatial_info = ["adjacency_list" in dataset.obs for dataset in self.datasets]

        if not any(use_spatial_info):
            return

        num_replicates = len(self.datasets)
        Sigma_x_invs = torch.zeros([num_replicates, self.K, self.K], **self.context)
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
            self.spatial_affinity.spatial_affinity_lookup[:] = Sigma_x_invs.mean(axis=0)
            for dataset in self.datasets:
                dataset.uns["Sigma_x_inv"] = {dataset.name : self.spatial_affinity.spatial_affinity_lookup[0]}
            
            # This optimizer retains its state throughout the optimization
            self.optimizer = torch.optim.Adam(
                [self.__getitem__(dataset.name)],
                lr=1e-1,
                betas=(.5, .9),
            )
        elif self.mode == "attention":
            #TODO: initialize with gradient descent
            raise NotImplementedError()
    
    def reaverage(self):
        # Set M_bar to average of self.Ms (memory efficient)
        self.spatial_affinity_bar.zero_()
        for dataset in self.datasets:
            self.spatial_affinity_bar.add_(self.__getitem__(dataset.name))
        self.spatial_affinity_bar.div_(len(self.datasets))

class SpatialAffinity():
    """Compute spatial affinities betweeen factors in 2D space.

    Parameters:
        K (int): number of latent spatial factors
    """
    def __init__(self, K, context=None):
        raise NotImplementedError()

    def get_metagene_affinities(self, metagenes):
        raise NotImplementedError()

class SpatialAffinityLookup(SpatialAffinity):
    def __init__(self, K, num_replicates=1, scaling=10, context=None):
        self.K = K
        self.context = context if context else {}
        self.spatial_affinity_lookup = torch.zeros((num_replicates, K, K), **self.context)

    def get_metagene_affinities(self, metagenes=None):
        return self.spatial_affinity_lookup

class SpatialAffinityAttention(SpatialAffinity):
    def __init__(self, K, num_replicates=1, scaling=10, context=None):
        self.K = K
        self.context = context if context else {}
        self.spatial_affinity_attention = MultiheadAttention(K * num_replicates, num_replicates, **self.context)

    def get_metagene_affinities(self, metagenes):
        _ , metagene_atttention_weights = self.spatial_affinity_attention(metagenes, metagenes, metagenes, average_attn_weights=False)

        return metagene_attention
