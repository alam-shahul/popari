import sys, logging, time, gc, os
from tqdm.auto import tqdm, trange

import numpy as np
import torch

from sample_for_integral import project2simplex
from util import NesterovGD

@torch.no_grad()
def estimate_weight_wonbr(Y, M, X, sigma_yx, replicate, prior_x_mode, prior_x, dataset, context, n_epochs=1000, tol=1e-6, update_alg='gd', verbose=True):
    """Estimate weights without spatial information - equivalent to vanilla NMF.

    min 1/2σ^2 || Y - X MT ||_2^2 + lam || X ||_1
    grad = X MT M / σ^2 - Y MT / σ^2 + lam

    TODO: use (projected) Nesterov GD. not urgent
    """

    M = dataset.uns["M"][f"{replicate}"]
    sigma_yx = dataset.uns["sigma_yx"]

    # Precomputing quantities 
    MTM = M.T @ M / (sigma_yx ** 2)
    YM = Y.to(M.device) @ M / (sigma_yx ** 2)
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


class IndependentSet:
    """Iterator class that yields a list of batch_size independent nodes from a spatial graph.

    For each iteration, no pair of yielded nodes can be neighbors of each other according to the
    adjacency matrix.

    """

    def __init__(self, adjacency_list, batch_size=50):
        self.N = len(adjacency_list)
        self.adjacency_list = adjacency_list
        self.batch_size = batch_size
        self.indices_remaining = None

    def __iter__(self):
        self.indices_remaining = set(range(self.N))
        return self

    def __next__(self):
        # make sure selected nodes are not adjacent to each other
        # i.e., find an independent set of `indices_candidates` in a greedy manner
        if len(self.indices_remaining) == 0:
            raise StopIteration

        valid_indices = []
        excluded_indices = set()
        effective_batch_size = min(self.batch_size, len(self.indices_remaining))
        candidate_indices = np.random.choice(list(self.indices_remaining),
            size=effective_batch_size,
            replace=False,
        )
        for index in candidate_indices:
            if index not in excluded_indices:
                valid_indices.append(index)
                excluded_indices |= set(self.adjacency_list[index])

        self.indices_remaining -= set(valid_indices)

        return valid_indices

@torch.no_grad()
def estimate_weight_wnbr(Y, M, X, sigma_yx, replicate, prior_x_mode, prior_x, dataset, context, n_epochs=1000, tol=1e-5, update_alg='nesterov'):
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
    M = dataset.uns["M"][f"{replicate}"]

    # Precomputing quantities
    MTM = M.T @ M / (sigma_yx ** 2)
    YM = Y.to(M.device) @ M / (sigma_yx ** 2)
    Ynorm = torch.linalg.norm(Y, ord='fro').item() ** 2 / (sigma_yx ** 2)
    base_step_size = 1 / torch.linalg.eigvalsh(MTM).max().item()
    S = torch.linalg.norm(X, dim=1, ord=1, keepdim=True)
    Z = X / S
    N = len(Z)
    
    E_adjacency_list = dataset.obs["adjacency_list"]
    Sigma_x_inv = dataset.uns["Sigma_x_inv"][f"{replicate}"]

    def get_adjacency_matrix(adjacency_list):
        edges = [(i, j) for i, e in enumerate(adjacency_list) for j in e]
        adjacency_matrix = torch.sparse_coo_tensor(np.array(edges).T, np.ones(len(edges)), size=[len(adjacency_list), N], **context)
        return adjacency_matrix

    def update_s():
        # print(f"Minimum S: {S.min()}")
        S[:] = (YM * Z).sum(axis=1, keepdim=True)
        # print(f"Minimum S: {S.min()}")
        # print(f"prior_x_mode: {prior_x_mode}")
        # print(f"prior_x: {prior_x[0][0]}")
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

    # def update_z_mu(Z):
    #     indices = np.random.permutation(len(S))
    #     # bs = len(indices) // 50
    #     bs = 1
    #     thr = 1e3
    #     for i in range(0, len(indices), bs):
    #         idx = indices[i: i+bs]
    #         adjacency_matrix = get_adjacency_matrix(E_adjacency_list[idx])
    #         numerator = YM[idx] * S[idx]
    #         denominator = (Z[idx] @ MTM).mul_(S[idx] ** 2)
    #         t = (adjacency_matrix @ Z) @ Sigma_x_inv
    #         # TM: not sure if projected mu works
    #         # TM: seems not
    #         numerator -= t.clip(max=0)
    #         denominator += t.clip(min=0)
    #         mul_fac = numerator / denominator
    #         mul_fac.clip_(min=1/thr, max=thr)
    #         # Z.mul_(mul_fac)
    #         Z[idx] *= mul_fac
    #         result = project2simplex(Z, dim=1)
    #         Z = 0
    #         Z = result

    #     return Z
    
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
        # def calc_func_grad(Z, idx, adjacency_matrix=None):
        #   # grad = (Z @ MTM).mul_(S ** 2)
        #   # grad.addcmul_(YM, S, value=-1)
        #   # grad.addmm_(E @ Z, Sigma_x_inv)
        #   # grad.sub_(grad.sum(1, keepdim=True))
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
        # print(f"Z initial loss: {func}")
        for idx in IndependentSet(E_adjacency_list, batch_size=256):
        # for idx in [[N-1]]:
            quad_batch = MTM
            # linear_batch = YM[idx] * S[idx] - get_adjacency_matrix(E_adjacency_list[idx]) @ Z @ Sigma_x_inv
            linear_batch_spatial = - get_adjacency_matrix(E_adjacency_list[idx]) @ Z @ Sigma_x_inv
            Z_batch = Z[idx].contiguous()
            S_batch = S[idx].contiguous()
                

            # print(f"Updated step size: {base_step_size / S_batch.square()}")
            optimizer = NesterovGD(Z_batch, base_step_size / S_batch.square())
            ppbar = trange(10000, leave=False, disable=True)
            # while True:
            for i_iter in ppbar:
                # loss_old, _ = compute_loss(np.inf)
                # S_old = S.clone()
                update_s() # TODO: update S_batch directly
                # loss_new, dloss = compute_loss(loss_old)
                # print(loss_old, loss_new, dloss, S_old.sub(S).abs().max().item())
                S_batch = S[idx].contiguous()
                linear_batch = linear_batch_spatial + YM[idx] * S_batch
                if i_iter == 0:
                    func, grad = calc_func_grad(Z_batch, S_batch, quad_batch, linear_batch)
                    # print(f"Z_batch initial loss: {func}")
                    func, grad = calc_func_grad(Z, S, MTM, YM * S - get_adjacency_matrix(E_adjacency_list) @ Z @ Sigma_x_inv / 2)
                    # print(f"Z loss: {func}")
                NesterovGD.step_size = base_step_size / S_batch.square() # TM: I think this converges as s converges
                func, grad = calc_func_grad(Z_batch, S_batch, quad_batch, linear_batch)
                Z_batch_prev = Z_batch.clone()
                # loss_old, _ = compute_loss(np.inf)
                Z_batch_copy = optimizer.step(grad)
                # print(Z_batch_copy.max())
                Z_batch = project2simplex(Z_batch_copy, dim=1)
                # print(result.sum(axis=1))
                optimizer.set_parameters(Z_batch)
                dZ = (Z_batch_prev - Z_batch).abs().max().item()
                Z[idx] = Z_batch
                # loss_new, dloss = compute_loss(loss_old)
                # if i_iter % 100 == 0:
                #   print(loss_old, loss_new, dloss, dZ)
                #   d = S_batch * Z_batch - X_bak[None]
                #   print(torch.linalg.norm(d))
                #   print(d)
                ppbar.set_description(f'func={func:.1e}, dZ={dZ:.1e}')
                if dZ < tol:
                    break
            ppbar.close()
            Z[idx] = Z_batch
            func, grad = calc_func_grad(Z_batch, S_batch, quad_batch, linear_batch)
            if verbose:
                print(f"Z_batch final loss: {func}")
            func, grad = calc_func_grad(Z, S, MTM, YM * S - get_adjacency_matrix(E_adjacency_list) @ Z @ Sigma_x_inv /2 )
            if verbose:
                print(f"Z loss: {func}")
            # assert False
            # pbar.set_description(f'Updating Z w/ nbrs via Nesterov GD')
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

