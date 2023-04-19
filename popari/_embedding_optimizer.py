from typing import Sequence
import logging, time
from tqdm.auto import tqdm, trange

import numpy as np
import torch

from popari.util import NesterovGD, IndependentSet, sample_graph_iid, project2simplex, project2simplex_, project_M, project_M_, get_datetime, convert_numpy_to_pytorch_sparse_coo
from popari.components import PopariDataset

class EmbeddingOptimizer():
    """Optimizer and state for Popari embeddings.

    """

    def __init__(self, K, Ys, datasets, initial_context=None, context=None, use_inplace_ops=False, embedding_step_size_multiplier=1, embedding_mini_iterations=1000, embedding_acceleration_trick=True, verbose=0):
        self.verbose = verbose
        self.use_inplace_ops = use_inplace_ops
        self.datasets = datasets
        self.hierarchical = False
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
            for idx in IndependentSet(E_adjacency_list, device=self.context["device"], batch_size=1024):
                quad_batch = MTM
                linear_batch_spatial = - torch.index_select(adjacency_matrix, 0, idx) @ Z @ Sigma_x_inv
                Z_batch = Z[idx].contiguous()
                S_batch = S[idx].contiguous()
                    
                optimizer = NesterovGD(Z_batch, base_step_size / S_batch.square())
                ppbar = trange(100, leave=False, disable=not (self.verbose > 3))
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


