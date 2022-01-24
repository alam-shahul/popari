import sys, logging, time, gc, os
from tqdm.auto import tqdm, trange

import numpy as np
import torch

from sample_for_integral import project2simplex
from util import NesterovGD


@torch.no_grad()
def estimate_weight_wonbr(
        Y, M, X, sigma_yx, prior_x_mode, prior_x, context, n_epochs=1000, tol=1e-6, update_alg='gd', verbose=True):
    """
    min 1/2σ^2 || Y - X MT ||_2^2 + lam || X ||_1
    grad = X MT M / σ^2 - Y MT / σ^2 + lam

    TODO: use (projected) Nesterov GD. not urgent
    """
    MTM = M.T @ M / (sigma_yx ** 2)
    YM = Y.to(M.device) @ M / (sigma_yx ** 2)
    Ynorm = torch.linalg.norm(Y, ord='fro').item() ** 2 / (sigma_yx ** 2)
    step_size = 1 / torch.linalg.eigvalsh(MTM).max().item()
    loss_prev, loss = np.inf, np.nan
    pbar = trange(n_epochs, leave=True, disable=not verbose, miniters=1000)
    for i_epoch in pbar:
        X_prev = X.clone()
        if update_alg == 'mu':
            X.clip_(min=1e-10)
            loss = ((X @ MTM) * X).sum() / 2 - X.view(-1) @ YM.view(-1) + Ynorm / 2
            numerator = YM
            denominator = X @ MTM
            if prior_x_mode == 'exponential shared fixed':
                # see sklearn.decomposition.NMF
                loss += (X @ prior_x[0]).sum()
                denominator.add_(prior_x[0][None])
            else:
                raise NotImplementedError

            loss = loss.item()
            assert loss <= loss_prev * (1 + 1e-4), (loss_prev, loss, (loss_prev - loss) / loss)
            mul_fac = numerator / denominator
            X.mul_(mul_fac).clip_(min=1e-10)
        elif update_alg == 'gd':
            t = X @ MTM
            loss = (t * X).sum().item() / 2
            g = t
            t = YM
            if prior_x_mode == 'exponential shared fixed':
                t = t - prior_x[0][None]
            else:
                raise NotImplementedError
            loss -= (t * X).sum().item()
            g -= t
            loss += Ynorm / 2
            X.add_(g, alpha=-step_size).clip_(min=1e-10)
        else:
            raise NotImplementedError

        dX = X_prev.sub(X).div(torch.linalg.norm(X, dim=1, ord=1, keepdim=True)).abs().max().item()
        do_stop = dX < tol
        if i_epoch % 1000 == 0 or do_stop:
            pbar.set_description(
                f'Updating weight w/o nbrs: loss = {loss:.1e} '
                f'%δloss = {(loss_prev - loss) / loss:.1e} '
                f'%δX = {dX:.1e}'
            )
        loss_prev = loss
        if do_stop: break
    pbar.close()
    return loss


class IndependentSet:
    def __init__(self, adj_list, batch_size=50):
        self.N = len(adj_list)
        self.adj_list = adj_list
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
        indices = []
        indices_exclude = set()
        indices_candidates = np.random.choice(
            list(self.indices_remaining),
            size=min(self.batch_size, len(self.indices_remaining)),
            replace=False,
        )
        for i in indices_candidates:
            if i in indices_exclude:
                continue
            else:
                indices.append(i)
                indices_exclude |= set(self.adj_list[i])
        self.indices_remaining -= set(indices)
        return list(indices)


@torch.no_grad()
def estimate_weight_wnbr(
        Y, M, standin, sigma_yx, Sigma_x_inv, E, prior_x_mode, prior_x, context, n_epochs=1000, tol=1e-5, update_alg='gd',
):
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
    X = standin.clone()
    MTM = M.T @ M / (sigma_yx ** 2)
    YM = Y.to(M.device) @ M / (sigma_yx ** 2)
    Ynorm = torch.linalg.norm(Y, ord='fro').item() ** 2 / (sigma_yx ** 2)
    step_size_base = 1 / torch.linalg.eigvalsh(MTM).max().item()
    S = torch.linalg.norm(X, dim=1, ord=1, keepdim=True)
    Z = X / S
    N = len(Z)

    E_adj_list = np.array(E, dtype=object)

    def get_adj_mat(adj_list):
        edges = [(i, j) for i, e in enumerate(adj_list) for j in e]
        adj_mat = torch.sparse_coo_tensor(np.array(edges).T, np.ones(len(edges)), size=[len(adj_list), N], **context)
        return adj_mat

    E_adj_mat = get_adj_mat(E_adj_list)

    def update_s():
        S[:] = (YM * Z).sum(1, keepdim=True)
        if prior_x_mode == 'exponential shared fixed':
            S.sub_(prior_x[0][0] / 2)
        else:
            raise NotImplementedError
        S.div_(((Z @ MTM) * Z).sum(1, keepdim=True))
        S.clip_(min=1e-5)

    # def update_z_mu(Z):
    #     indices = np.random.permutation(len(S))
    #     # bs = len(indices) // 50
    #     bs = 1
    #     thr = 1e3
    #     for i in range(0, len(indices), bs):
    #         idx = indices[i: i+bs]
    #         adj_mat = get_adj_mat(E_adj_list[idx])
    #         numerator = YM[idx] * S[idx]
    #         denominator = (Z[idx] @ MTM).mul_(S[idx] ** 2)
    #         t = (adj_mat @ Z) @ Sigma_x_inv
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

    # def update_z_gd(Z):
    #     # def calc_func_grad(Z, idx, adj_mat=None):
    #     #   # grad = (Z @ MTM).mul_(S ** 2)
    #     #   # grad.addcmul_(YM, S, value=-1)
    #     #   # grad.addmm_(E @ Z, Sigma_x_inv)
    #     #   # grad.sub_(grad.sum(1, keepdim=True))
    #     def calc_func_grad(Z_batch, S_batch, quad, linear):
    #         t = (Z_batch @ quad).mul_(S_batch ** 2)
    #         f = (t * Z_batch).sum() / 2
    #         g = t
    #         t = linear
    #         f -= (t * Z_batch).sum()
    #         g -= t
    #         g.sub_(g.sum(1, keepdim=True))
    #         return f.item(), g
    #     step_size = step_size_base / S.square()
    #     pbar = tqdm(range(N), leave=False, disable=True)
    #     for idx in IndependentSet(E_adj_list, batch_size=128):
    #         step_size_scale = 1
    #         quad_batch = MTM
    #         linear_batch = YM[idx] * S[idx] - get_adj_mat(E_adj_list[idx]) @ Z @ Sigma_x_inv
    #         Z_batch = Z[idx].contiguous()
    #         S_batch = S[idx].contiguous()
    #         step_size_batch = step_size[idx].contiguous()
    #         func, grad = calc_func_grad(Z_batch, S_batch, quad_batch, linear_batch)
    #         while True:
    #             Z_batch_new = Z_batch - step_size_batch * step_size_scale * grad
    #             result = project2simplex(Z_batch_new, dim=1)
    #             Z_batch_new = 0
    #             Z_batch_new = result
    #             dZ = Z_batch_new.sub(Z_batch).abs().max().item()
    #             func_new, grad_new = calc_func_grad(Z_batch_new, S_batch, quad_batch, linear_batch)
    #             if func_new < func:
    #                 Z_batch = Z_batch_new
    #                 func = func_new
    #                 grad = grad_new
    #                 step_size_scale *= 1.1
    #                 continue
    #             else:
    #                 step_size_scale *= .5
    #             if dZ < tol or step_size_scale < .5: break
    #         assert step_size_scale > .1
    #         Z[idx] = Z_batch
    #         pbar.set_description(f'Updating Z w/ nbrs via line search: lr={step_size_scale:.1e}')
    #         pbar.update(len(idx))
    #     pbar.close()

    def update_z_gd_nesterov(Z):
        def calc_func_grad(Z_batch, S_batch, quad, linear):
            t = (Z_batch @ quad).mul_(S_batch ** 2)
            f = (t * Z_batch).sum() / 2
            g = t
            t = linear
            f -= (t * Z_batch).sum()
            g -= t
            g.sub_(g.sum(1, keepdim=True))
            return f.item(), g
        pbar = trange(N, leave=False, disable=True, desc='Updating Z w/ nbrs via Nesterov GD')
        for idx in IndependentSet(E_adj_list, batch_size=256):
        # for idx in [[N-1]]:
            quad_batch = MTM
            # linear_batch = YM[idx] * S[idx] - get_adj_mat(E_adj_list[idx]) @ Z @ Sigma_x_inv
            linear_batch_spatial = - get_adj_mat(E_adj_list[idx]) @ Z @ Sigma_x_inv
            Z_batch = Z[idx].contiguous()
            S_batch = S[idx].contiguous()
            optimizer = NesterovGD(Z_batch, step_size_base / S_batch.square())
            ppbar = trange(10000, leave=False, disable=True)
            # while True:
            for i_iter in ppbar:
                # loss_old, _ = calc_loss(np.inf)
                # S_old = S.clone()
                update_s() # TODO: update S_batch directly
                # loss_new, dloss = calc_loss(loss_old)
                # print(loss_old, loss_new, dloss, S_old.sub(S).abs().max().item())
                S_batch = S[idx].contiguous()
                linear_batch = linear_batch_spatial + YM[idx] * S_batch
                NesterovGD.step_size = step_size_base / S_batch.square() # TM: I think this converges as s converges
                func, grad = calc_func_grad(Z_batch, S_batch, quad_batch, linear_batch)
                Z_batch_prev = Z_batch.clone()
                # loss_old, _ = calc_loss(np.inf)
                Z_batch_copy = optimizer.step(grad)
                result = project2simplex(Z_batch_copy, dim=1)
                Z_batch = 0
                Z_batch = result
                optimizer.set_parameters(result)
                dZ = Z_batch_prev.sub(Z_batch).abs().max().item()
                Z[idx] = Z_batch
                # loss_new, dloss = calc_loss(loss_old)
                # if i_iter % 100 == 0:
                #   print(loss_old, loss_new, dloss, dZ)
                #   d = S_batch * Z_batch - X_bak[None]
                #   print(torch.linalg.norm(d))
                #   print(d)
                ppbar.set_description(f'func={func:.1e}, dZ={dZ:.1e}')
                if dZ < tol: break
            ppbar.close()
            Z[idx] = Z_batch
            # assert False
            # pbar.set_description(f'Updating Z w/ nbrs via Nesterov GD')
            pbar.update(len(idx))
        pbar.close()
        return Z

    def calc_loss(loss_prev):
        X = Z * S
        loss = ((X @ MTM) * X).sum() / 2 - (X * YM).sum() + Ynorm / 2
        if prior_x_mode == 'exponential shared fixed':
            loss += prior_x[0][0] * S.sum()
        else:
            raise NotImplementedError
        if Sigma_x_inv is not None:
            loss += ((E_adj_mat @ Z) @ Sigma_x_inv).mul(Z).sum() / 2
        loss = loss.item()
        # assert loss <= loss_prev, (loss_prev, loss)
        return loss, loss_prev - loss

    # TM: consider combine calc_loss and update_z to remove a call to torch.sparse.mm
    # TM: the above idea is not practical if we update only a subset of nodes each time

    loss = np.inf
    pbar = trange(n_epochs, desc='Updating weight w/ neighbors')

    for i_epoch in pbar:
        update_s()
        Z_prev = Z.clone().detach()
        # We may use Nesterov first and then vanilla GD in later iterations
        # update_z_mu(Z)
        # update_z_gd(Z)
        result = update_z_gd_nesterov(Z)
        Z = 0
        Z = result
        loss, dloss = calc_loss(loss)
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
def estimate_weight_wnbr_phenotype(
        Y, M, X, sigma_yx, Sigma_x_inv, E, prior_x_mode, prior_x, phenotype, phenotype_predictors,
        context, n_epochs=100, tol=1e-4, update_alg='gd',
):
    """
    The optimization for all variables
    min 1/2σ^2 || Y - diag(S) Z MT ||_2^2 + lam || S ||_1 + sum_{ij in E} ziT Σx-1 zj + loss(predictor(Z), phenotype)

    for s_i
    min 1/2σ^2 || y - M z s ||_2^2 + lam s
    s* = max(0, ( yT M z / σ^2 - lam ) / ( zT MT M z / σ^2) )

    for Z
    min 1/2σ^2 || Y - diag(S) Z MT ||_2^2 + sum_{ij in E} ziT Σx-1 zj
    grad_i = MT M z s^2 / σ^2 - MT y s / σ^2 + sum_{j in Ei} Σx-1 zj
    """
    MTM = M.T @ M / (sigma_yx**2)
    YM = Y.to(M.device) @ M / (sigma_yx ** 2)
    Ynorm = torch.linalg.norm(Y, ord='fro').item() ** 2 / (sigma_yx**2)
    step_size_base = 1 / torch.linalg.eigvalsh(MTM).max()
    S = torch.linalg.norm(X, dim=1, ord=1, keepdim=True)
    Z = X / S
    N = len(Z)

    E_adj_list = np.array(E, dtype=object)

    def get_adj_mat(adj_list):
        edges = [(i, j) for i, e in enumerate(adj_list) for j in e]
        adj_mat = torch.sparse_coo_tensor(np.array(edges).T, np.ones(len(edges)), size=[len(adj_list), N], **context)
        return adj_mat

    E_adj_mat = get_adj_mat(E_adj_list)

    def update_s():
        S[:] = (YM * Z).sum(1, keepdim=True)
        if prior_x_mode == 'exponential shared fixed':
            S.sub_(prior_x[0][0])
        else:
            raise NotImplementedError
        S.div_(((Z @ MTM) * Z).sum(1, keepdim=True))
        S.clip_(min=1e-5)

    def update_z_gd(Z):
        Z_storage = Z
        # @torch.enable_grad()
        def calc_func_grad(Z, idx):
            # TODO: use auto-grad to store the computation graph of phenotype loss
            if Z.grad is None: Z.grad = torch.zeros_like(Z)
            else: Z.grad.zero_()
            # -- manual --
            t = (Z[idx] @ MTM).mul_(S[idx] ** 2)
            f = (t * Z[idx]).sum().item() / 2
            g = t
            t = YM[idx] * S[idx]
            f -= (t * Z[idx]).sum().item()
            g -= t
            if Sigma_x_inv is not None:
                t = get_adj_mat(E_adj_list[idx]) @ Z @ Sigma_x_inv
                f += (t * Z[idx]).sum().item()
                g += t
            # f += Ynorm / 2
            Z.grad[idx] += g
            # -- auto grad --
            # loss = (Z[idx] @ MTM).mul(Z[idx]).mul(S[idx].square()).sum() / 2
            # loss = loss - (YM[idx] * S[idx] * Z[idx]).sum()
            # loss = loss + Ynorm / 2
            # if Sigma_x_inv is not None:
            #   with torch.no_grad():
            #       t = get_adj_mat(E_adj_list[idx]) @ Z @ Sigma_x_inv
            #   loss = loss + (t * Z[idx]).sum()
            # _t = time.perf_counter()
            with torch.enable_grad():
                # Z.requires_grad_(True)
                Z_batch = Z[idx].clone().requires_grad_(True)
                for p, (predictor, optimizer, loss_fn) in zip(phenotype.values(), phenotype_predictors.values()):
                    if p is None: continue
                    # loss = loss + loss_fn(predictor(Z[idx]), p[idx], state='eval')
                    # loss = loss_fn(predictor(Z[idx]), p[idx], state='eval')
                    loss = loss_fn(predictor(Z_batch), p[idx], state='eval')
                    loss.backward()
                    f += loss.item()
                # Z.requires_grad_(False)
                Z.grad[idx] += Z_batch
            # # print(time.perf_counter() - _t)
            return f
        step_size = step_size_base / S.square()
        pbar = tqdm(range(N), leave=False, desc='Updating Z w/ nbrs w/ ph.')
        for idx in IndependentSet(E_adj_list, batch_size=100):
            step_size_scale = 1
            func = calc_func_grad(Z, idx)
            # func.backward()
            # func = func.item()
            assert Z.grad is not None
            Z.grad.sub_(Z.grad.mean(1, keepdim=True))
            for i_iter in range(100):
                Z_new = Z.clone().detach_()
                assert Z.grad is not None
                Z_new[idx] -= (step_size[idx] * step_size_scale) * Z.grad[idx]
                result = project2simplex(Z_new[idx], dim=1)
                Z_new[idx] = 0
                Z_new[idx] = result
                dZ = Z_new[idx].sub(Z[idx]).abs().max().item()
                func_new = calc_func_grad(Z_new, idx)
                # if func_new.item() < func:
                # print(step_size_scale, func - func_new)
                if func_new < func:
                    func = func_new
                    # func.backward()
                    # func = func.item()
                    Z = Z_new
                    assert Z.grad is not None
                    step_size_scale *= 1.1
                    continue
                else:
                    step_size_scale *= .1
                if dZ < 1e-4 or step_size_scale < .1: break
            # assert step_size_scale > .1
            pbar.set_description(f'Updating Z w/ nbrs w/ ph. lr={step_size_scale:.1e}')
            pbar.update(len(idx))
        pbar.close()
        Z_storage[:] = Z

    def calc_loss(loss_prev):
        X = Z * S
        loss = ((X @ MTM) * X).sum() / 2 - (X * YM).sum() + Ynorm / 2
        if prior_x_mode == 'exponential shared fixed':
            loss += prior_x[0][0] * S.sum()
        else:
            raise NotImplementedError
        if Sigma_x_inv is not None:
            loss += ((E_adj_mat @ Z) @ Sigma_x_inv).mul(Z).sum()
        for p, (predictor, optimizer, loss_fn) in zip(phenotype.values(), phenotype_predictors.values()):
            if p is None: continue
            loss = loss + loss_fn(predictor(Z), p, state='eval')
        loss = loss.item()
        # assert loss <= loss_prev, (loss_prev, loss)
        return loss, loss_prev - loss

    loss = np.inf
    pbar = trange(n_epochs, desc='Updating weight w/ neighbors')
    Z_prev = Z.clone().detach_().requires_grad_(False)
    for i_epoch in pbar:
        update_s()
        update_z_gd(Z)
        loss, dloss = calc_loss(loss)
        dZ = (Z_prev - Z).abs().max().item()
        pbar.set_description(
            f'Updating weight w/ nbrs w/ ph: loss = {loss:.1e} '
            f'δloss = {dloss:.1e} '
            f'δZ = {dZ:.1e}'
        )
        if dZ < tol: break
        Z_prev[:] = Z
    pbar.close()

    Z.requires_grad_(False)
    X[:] = Z * S
    return loss


@torch.no_grad()
def estimate_weight_wnbr_phenotype_v2(
        Y, M, Z, S, sigma_yx, Sigma_x_inv, E, prior_x_mode, prior_x, Z_optimizer, phenotype, phenotype_predictors,
        context, n_epochs=1000, tol=1e-3, update_alg='gd',
):
    """
    The optimization for all variables
    min 1/2σ^2 || Y - diag(S) Z MT ||_2^2 + lam || S ||_1 + sum_{ij in E} ziT Σx-1 zj + loss(predictor(Z), phenotype)

    for s_i
    min 1/2σ^2 || y - M z s ||_2^2 + lam s
    s* = max(0, ( yT M z / σ^2 - lam ) / ( zT MT M z / σ^2) )

    for Z
    min 1/2σ^2 || Y - diag(S) Z MT ||_2^2 + sum_{ij in E} ziT Σx-1 zj
    grad_i = MT M z s^2 / σ^2 - MT y s / σ^2 + sum_{j in Ei} Σx-1 zj

    Difference from v1: update all nodes at the same time
    TM: used to check if IndependentSet is a bottleneck
    TM: seems not
    TM: but not sure which one is more efficient, after all we don't guarantee the loss is decreasing
    """
    MTM = M.T @ M / (sigma_yx**2)
    YM = Y.to(M.device) @ M / (sigma_yx ** 2)
    Ynorm = torch.linalg.norm(Y, ord='fro').item() ** 2 / (sigma_yx**2)
    N = len(Z)

    E_adj_list = np.array(E, dtype=object)

    def get_adj_mat(adj_list):
        edges = [(i, j) for i, e in enumerate(adj_list) for j in e]
        adj_mat = torch.sparse_coo_tensor(np.array(edges).T, np.ones(len(edges)), size=[len(adj_list), N], **context)
        return adj_mat

    E_adj_mat = get_adj_mat(E_adj_list)

    def update_s():
        S[:] = (YM * Z).sum(1, keepdim=True)
        if prior_x_mode == 'exponential shared fixed':
            S.sub_(prior_x[0][0])
        else:
            raise NotImplementedError(prior_x_mode)
        S.div_(((Z @ MTM) * Z).sum(1, keepdim=True))
        S.clip_(min=1e-5)

    def update_z(Z):
        # Z_storage = Z
        Z.requires_grad_(True)
        Z_optimizer.zero_grad()
        def calc_func_grad(Z, idx):
            # TODO: use auto-grad to store the computation graph of phenotype loss
            if Z.grad is None: Z.grad = torch.zeros_like(Z)
            else: Z.grad.zero_()
            # -- manual --
            t = (Z[idx] @ MTM).mul_(S[idx] ** 2)
            f = (t * Z[idx]).sum().item() / 2
            g = t
            t = YM[idx] * S[idx]
            f -= (t * Z[idx]).sum().item()
            g -= t
            if Sigma_x_inv is not None:
                t = get_adj_mat(E_adj_list[idx]) @ Z @ Sigma_x_inv
                f += (t * Z[idx]).sum().item()
                g += t
            # f += Ynorm / 2
            Z.grad[idx] += g
            # -- auto grad --
            with torch.enable_grad():
                Z.requires_grad_(True)
                for p, (predictor, optimizer, loss_fn) in zip(phenotype.values(), phenotype_predictors.values()):
                    if p is None: continue
                    # loss = loss + loss_fn(predictor(Z), p, state='eval')
                    loss = loss_fn(predictor(Z), p, state='eval')
                    loss.backward()
                    f += loss.item()
                Z.requires_grad_(False)
            # # print(time.perf_counter() - _t)
            return f
        func = calc_func_grad(Z, slice(None))
        # func.backward()
        # func = func.item()
        Z_optimizer.step()
        result = project2simplex(Z, dim=1)
        Z = 0
        Z = result
        # Z_storage[:] = Z
        Z.requires_grad_(False)

    def calc_loss(loss_prev):
        X = Z * S
        loss = ((X @ MTM) * X).sum() / 2 - (X * YM).sum() + Ynorm / 2
        if prior_x_mode == 'exponential shared fixed':
            loss += prior_x[0][0] * S.sum()
        else:
            raise NotImplementedError
        if Sigma_x_inv is not None:
            loss += ((E_adj_mat @ Z) @ Sigma_x_inv).mul(Z).sum()
        for p, (predictor, optimizer, loss_fn) in zip(phenotype.values(), phenotype_predictors.values()):
            if p is None: continue
            loss = loss + loss_fn(predictor(Z), p, state='eval')
        loss = loss.item()
        # assert loss <= loss_prev, (loss_prev, loss)
        return loss, loss_prev - loss

    loss = np.inf
    pbar = trange(n_epochs, desc='Updating weight w/ neighbors')
    Z_prev = Z.clone().detach_().requires_grad_(False)
    for i_epoch in pbar:
        update_s()
        update_z(Z)
        loss, dloss = calc_loss(loss)
        dZ = (Z_prev - Z).abs().max().item()
        pbar.set_description(
            f'Updating weight w/ nbrs w/ ph: loss = {loss:.1e} '
            f'δloss = {dloss:.1e} '
            f'δZ = {dZ:.1e}'
        )
        if dZ < tol: break
        Z_prev[:] = Z
    pbar.close()

    Z.requires_grad_(False)
    # X[:] = Z * S
    return loss
