import sys, time, itertools, logging
from multiprocessing import Pool, Process
from tqdm.auto import tqdm, trange
from collections import defaultdict

import torch
import numpy as np
from scipy.special import loggamma

from sample_for_integral import integrate_of_exponential_over_simplex, project2simplex
from util import NesterovGD

@torch.no_grad()
def project_M(M, M_constraint):
    result = M.clone()
    if M_constraint == 'simplex':
        result = project2simplex(result, dim=0, zero_threshold=1e-5)
    elif M_constraint == 'unit sphere':
        result = M.div(torch.linalg.norm(result, ord=2, dim=0, keepdim=True))
    elif M_constraint == 'nonneg unit sphere':
        result = M.clip(1e-10).div(torch.linalg.norm(result, ord=2, dim=0, keepdim=True))
    else:
        raise NotImplementedError
    return result


@torch.no_grad()
def estimate_M(Ys, Xs, M, sigma_yxs, betas, M_constraint, context,
        M_bar=None, lambda_M=0,
        n_epochs=10000, tol=1e-6, backend_algorithm='gd Nesterov', verbose=True):
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
        M: current estimate of metagene parameters
        betas: weight of each FOV in optimization scheme
        context: unclear
        n_epochs: number of epochs 

    Returns:
        Updated estimate of metagene parameters.
    """

    _, K = M.shape
    quadratic_factor = torch.zeros([K, K], **context)
    linear_term = torch.zeros_like(M)
    scaled_betas = betas / (sigma_yxs**2)
    constant = (np.array([torch.linalg.norm(Y).item()**2 for Y in Ys]) * scaled_betas).sum()
    for Y, X, sigma_yx, scaled_beta in zip(Ys, Xs, sigma_yxs, scaled_betas):
        quadratic_factor.addmm_(X.T, X, alpha=scaled_beta)
        linear_term.addmm_(Y.T.to(X.device), X, alpha=scaled_beta)
    if lambda_M > 0 and M_bar is not None:
        quadratic_factor.diagonal().add_(lambda_M)
        linear_term += lambda_M * M_bar

    loss_prev, loss = np.inf, np.nan
    pbar = trange(n_epochs, leave=True, disable=not verbose, desc='Updating M', miniters=1000)

    def calc_func_grad(M):
        t = M @ quadratic_factor
        f = (t * M).sum() / 2
        g = t
        t = linear_term
        f -= (t * M).sum()
        g -= t
        f += constant / 2
        if M_constraint == 'simplex':
            g.sub_(g.sum(0, keepdim=True))
        return f.item(), g
    
    if backend_algorithm == 'mu':
        for epoch in pbar:
            loss = ((M @ quadratic_factor) * M).sum() / 2 - (M * linear_term).sum() + constant / 2
            loss = loss.item()
            numerator = linear_term
            denominator = M @ quadratic_factor
            mul_fac = numerator / denominator

            M_prev = M.clone()
            # mul_fac.clip_(max=10)
            M.mul_(mul_fac)
            M = project_M(M, M_constraint)
            dM = M_prev.sub(M).abs_().max().item()

            do_stop = dM < tol and epoch > 5
            if epoch % 1000 == 0 or do_stop:
                pbar.set_description(
                    f'Updating M: loss = {loss:.1e}, '
                    f'%δloss = {(loss_prev - loss) / loss:.1e}, '
                    f'δM = {dM:.1e}'
                )
            if do_stop: break
    elif backend_algorithm == 'gd':
        step_size = 1 / torch.linalg.eigvalsh(quadratic_factor).max().item()
        step_size_scale = 1
        func, grad = calc_func_grad(M)
        dM = df = np.inf
        for epoch in pbar:
            M_new = M.sub(grad, alpha=step_size * step_size_scale)
            M_new = project_M(M_new, M_constraint)
            func_new, grad_new = calc_func_grad(M_new)
            if func_new < func or step_size_scale == 1:
                dM = (M_new - M).abs().max().item()
                df = func - func_new
                M[:] = M_new
                func = func_new
                grad = grad_new
                step_size_scale *= 1.1
            else:
                step_size_scale *= .5
                step_size_scale = max(step_size_scale, 1.)

            do_stop = dM < tol and epoch > 5
            if epoch % 1000 == 0 or do_stop:
                pbar.set_description(
                    f'Updating M: loss = {func:.1e}, '
                    f'%δloss = {df / func:.1e}, '
                    f'δM = {dM:.1e}, '
                    f'lr={step_size_scale:.1e}'
                )
            if do_stop: break
    elif backend_algorithm == 'gd Nesterov':
        step_size = 1 / torch.linalg.eigvalsh(quadratic_factor).max().item()
        func_prev = np.inf
        optimizer = NesterovGD(M.clone(), step_size)
        for epoch in pbar:
            func, grad = calc_func_grad(M)
            df = func_prev - func
            M_prev = M.clone()
            M = optimizer.step(grad)
            M = project_M(M, M_constraint)
            optimizer.set_parameters(M)
            dM = M_prev.sub(M).abs().max().item()
            do_stop = dM < tol and epoch > 5
            assert not np.isnan(func)
            if epoch % 1000 == 0 or do_stop:
                pbar.set_description(
                    f'Updating M: loss = {func:.1e}, '
                    f'%δloss = {df / func:.1e}, '
                    f'δM = {dM:.1e}'
                    # f'lr={step_size_scale:.1e}'
                )
            if do_stop: break
            func_prev = func
    else:
        raise NotImplementedError
    
    return M
    # step_size = 1 / torch.linalg.eigvalsh(quadratic_factor).max().item()
    # for epoch in pbar:
    #   loss = ((M @ quadratic_factor) * M).sum() / 2 - (M * linear_term).sum() + constant / 2
    #   loss = loss.item()
    #   assert loss <= loss_prev * (1 + 1e-4), (loss_prev, loss)
    #   M_prev[:] = M
    #
    #   if backend_algorithm == 'mu':
    #       numerator = linear_term
    #       denominator = M @ quadratic_factor
    #       mul_fac = numerator / denominator
    #       M.mul_(mul_fac)
    #       mul_fac.clip_(max=10)
    #   elif backend_algorithm == 'gd':
    #       grad = torch.addmm(linear_term, M, quadratic_factor, alpha=1, beta=-1)
    #       # grad.sub_(grad.sum(0, keepdim=True))
    #       M.add_(grad, alpha=-step_size)
    #   else:
    #       raise NotImplementedError
    #   if M_constraint == 'simplex':
    #       project2simplex(M, dim=0)
    #   elif M_constraint == 'unit sphere':
    #       M.div_(torch.linalg.norm(M, ord=2, dim=0, keepdim=True))
    #   elif M_constraint == 'nonneg unit sphere':
    #       M.clip_(1e-10).div_(torch.linalg.norm(M, ord=2, dim=0, keepdim=True))
    #   else:
    #       raise NotImplementedError
    #   loss_prev = loss
    #   dM = M_prev.sub(M).abs_().max().item()
    #
    #   do_stop = dM < tol and epoch > 5
    #   if epoch % 1000 == 0 or do_stop:
    #       pbar.set_description(
    #           f'Updating M: loss = {loss:.1e}, '
    #           f'%δloss = {(loss_prev - loss) / loss:.1e}, '
    #           f'δM = {dM:.1e}'
    #       )
    #   if do_stop: break
    # pbar.close()
    # return loss


def estimate_Sigma_x_inv(Xs, Sigma_x_inv, Es, use_spatial, lambda_Sigma_x_inv, betas, optimizer, context, n_epochs=1000):
    """Optimize Sigma_x_inv parameters.

    """

    if not any(sum(map(len, E)) > 0 and u for E, u in zip(Es, use_spatial)): return
    linear_term = torch.zeros_like(Sigma_x_inv).requires_grad_(False)
    Zs = [X / torch.linalg.norm(X, axis=1, ord=1, keepdim=True) for X in Xs]
    nus = [] # sum of neighbors' z
    num_edges = 0
    for Z, E, u, beta in zip(Zs, Es, use_spatial, betas):
        E = [(i, j) for i, e in enumerate(E) for j in e]
        E = torch.sparse_coo_tensor(np.array(E).T, np.ones(len(E)), size=[len(Z)]*2, **context)
        if u:
            nu = E @ Z
            linear_term.addmm_(Z.T, nu, alpha=beta)
        else:
            nu = None
        nus.append(nu)
        num_edges += beta * sum(map(len, E))
        del Z, E
    # linear_term = (linear_term + linear_term.T) / 2 # should be unnecessary as long as E is symmetric

    assumption_str = 'mean-field'

    history = []
    Sigma_x_inv.requires_grad_(True)

    if optimizer is not None:
        loss_prev, loss = np.inf, np.nan
        pbar = trange(n_epochs, desc='Updating Σx-1')
        Sigma_x_inv_best, loss_best, epoch_best = None, np.inf, -1
        dSigma_x_inv = np.inf
        early_stop_epoch_count = 0
        for epoch in pbar:
            optimizer.zero_grad()

            loss = Sigma_x_inv.view(-1) @ linear_term.view(-1)
            loss = loss + lambda_Sigma_x_inv * Sigma_x_inv.pow(2).sum() * num_edges / 2
            for Z, nu, beta in zip(Zs, nus, betas):
                if nu is None: continue
                eta = nu @ Sigma_x_inv
                logZ = integrate_of_exponential_over_simplex(eta)
                loss = loss + beta * logZ.sum()
            loss = loss / num_edges
            # if epoch == 0: print(loss.item())

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
            # with torch.no_grad():
            #   Sigma_x_inv -= Sigma_x_inv.mean()

            loss = loss.item()
            dloss = loss_prev - loss
            loss_prev = loss

            with torch.no_grad():
                dSigma_x_inv = Sigma_x_inv_prev.sub(Sigma_x_inv).abs().max().item()
            pbar.set_description(
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

        with torch.no_grad():
            Sigma_x_inv[:] = Sigma_x_inv_best
    else:
        Sigma_x_inv_storage = Sigma_x_inv

        def calc_func_grad(Sigma_x_inv):
            if Sigma_x_inv.grad is None: Sigma_x_inv.grad = torch.zeros_like(Sigma_x_inv)
            else: Sigma_x_inv.grad.zero_()
            loss = Sigma_x_inv.view(-1) @ linear_term.view(-1)
            loss = loss + lambda_Sigma_x_inv * Sigma_x_inv.pow(2).sum() * num_edges / 2
            for Z, nu, beta in zip(Zs, nus, betas):
                if nu is None: continue
                eta = nu @ Sigma_x_inv
                logZ = integrate_of_exponential_over_simplex(eta)
                loss = loss + beta * logZ.sum()
            return loss

        pbar = trange(n_epochs)
        step_size = 1e-1
        step_size_update = 2
        dloss = np.inf
        loss = calc_func_grad(Sigma_x_inv)
        loss.backward()
        loss = loss.item()
        for epoch in pbar:
            with torch.no_grad():
                Sigma_x_inv_new = Sigma_x_inv.add(Sigma_x_inv.grad, alpha=-step_size).requires_grad_(True)
                Sigma_x_inv_new.sub_(Sigma_x_inv_new.mean())
            loss_new = calc_func_grad(Sigma_x_inv_new)
            with torch.no_grad():
                if loss_new.item() < loss:
                    loss_new.backward()
                    loss_new = loss_new.item()
                    dloss = loss - loss_new
                    loss = loss_new
                    dSigma_x_inv = Sigma_x_inv.sub(Sigma_x_inv_new).abs().max().item()
                    Sigma_x_inv = Sigma_x_inv_new
                    step_size *= step_size_update
                else:
                    step_size /= step_size_update
                pbar.set_description(
                    f'Updating Σx-1: loss = {dloss:.1e} -> {loss:.1e}, '
                    f'δΣx-1 = {dSigma_x_inv:.1e} '
                    f'lr = {step_size:.1e} '
                    f'Σx-1 range = {Sigma_x_inv.min().item():.1e} ~ {Sigma_x_inv.max().item():.1e}'
                )
                if (Sigma_x_inv.grad * step_size).abs().max() < 1e-3:
                    dloss = np.nan
                    dSigma_x_inv = np.nan
                    break

        with torch.no_grad():
            Sigma_x_inv_storage[:] = Sigma_x_inv
    Sigma_x_inv.requires_grad_(False)

    return history


def estimate_phenotype_predictor(
        input_list, target_list,
        phenotype_name, predictor, optimizer, loss_fn,
        n_epoch=10000,
):
    pbar = trange(n_epoch, desc=f'Updating `{phenotype_name}` predictor')
    for param in predictor.parameters(): param.requires_grad_(True)
    predictor.train()
    loss_best = np.inf
    epoch_best = 0
    loss_prev = np.nan
    early_stop_epoch_count = 0
    history = []
    for epoch in pbar:
        loss_total = 0
        optimizer.zero_grad()
        metrics = defaultdict(lambda: 0)
        for x, y in zip(input_list, target_list):
            yhat = predictor(x)
            loss = loss_fn(yhat, y, state='train')
            loss_total += loss.item()
            loss.backward()
            metrics['loss'] += loss.item()
            metrics['acc'] += (yhat.argmax(1) == y).sum().item()
            metrics['# of samples'] += len(y)
        optimizer.step()
        for k in ['acc']:
            metrics[k] /= metrics['# of samples']

        dloss = loss_prev - loss_total
        loss_prev = loss_total
        if loss_total < loss_best:
            loss_best = loss_total
            epoch_best = epoch
        if dloss < 1e-4:
            early_stop_epoch_count += 1
        else:
            early_stop_epoch_count = 0
        history.append(metrics)
        pbar.set_description(
            f'Updating `{phenotype_name}` predictor. '
            f'loss:{dloss:.1e} -> {loss_total:.1e} best={loss_best:.1e} '
            f'L2={sum(torch.linalg.norm(param)**2 for param in predictor.parameters())**.5:.1e} '
            f"acc={metrics['acc']:.2f}"
        )
        # if dloss < 1e-4: break
        if early_stop_epoch_count >= 10 or epoch > epoch_best + 100:
            break
    pbar.close()
    predictor.eval()
    for param in predictor.parameters():
        param.requires_grad_(False)

