import sys, time, itertools, logging
from multiprocessing import Pool, Process
from tqdm.auto import tqdm, trange

import torch
import numpy as np
from scipy.special import loggamma

from sample_for_integral import integrate_of_exponential_over_simplex, project2simplex

@torch.no_grad()
def estimate_M(Ys, Xs, M, betas, context, n_epochs=100, tolerance=1e-5, mode='gd'):
    """Optimize metagene parameters.

    min || Y - X MT ||_2^2 / 2
    s.t. || Mk ||_1 = 1
    grad = M XT X - YT X
    
    Args:
        Ys: list of gene expression arrays
        Xs: list of estimated hidden states
        M: current estimation of metagenes
        betas: weight of each FOV in optimization scheme
        context: unclear
        n_epochs: number of epochs 

    Returns:
        Updated estimate of metagene parameters.
    """

    num_genes, K = M.shape

    # Calculating constants
    XTX = torch.zeros([K, K], **context)
    YTX = torch.zeros_like(M)
    weighted_norm = np.dot([torch.linalg.norm(Y, ord='fro') ** 2 for Y in Ys], betas)
    for Y, X, beta in zip(Ys, Xs, betas):
        XTX += beta * X.T @ X
        YTX += beta * Y.T.to(X.device) @ X

    previous_loss, loss = np.inf, np.nan
    progress_bar = trange(n_epochs, disable=False, desc='Updating M')
    M_prev = M.clone().detach()
    XTX_maximum_eigenvalue = torch.linalg.eigvalsh(XTX).max().item()
    step_size = 1 / XTX_maximum_eigenvalue 
    for epoch in progress_bar:
        # TODO: shouldn't it be 2 times instead of divided by 2 for the second term?
        # TODO: can we save computation by not calculating the loss here?
        loss = ((M @ XTX) * M).sum() / 2 - (M * YTX).sum() + weighted_norm / 2
        loss = loss.item()
        assert loss <= previous_loss * (1 + 1e-4), (previous_loss, loss)
        M_prev[:] = M

        if mode == 'mu':
            numerator = YTX
            denominator = M @ XTX
            mul_fac = numerator / denominator
            M.mul_(mul_fac)
            # TODO: is it okay to remove this line?
            mul_fac.clip_(max=10)
        elif mode == 'gd':
            grad = torch.addmm(YTX, M, XTX, alpha=1, beta=-1)
            # Why this line?
            grad -= grad.sum(axis=0, keepdim=True)
            M = M.add(grad, alpha=-step_size)

        M = project2simplex(M, dim=0)

        previous_loss = loss
        max_difference = torch.abs(M_prev - M).max().item()
        progress_bar.set_description(
            f'Updating M: loss = {loss:.1e}, '
            f'%δloss = {(previous_loss - loss) / loss:.1e}, '
            f'δM = {max_difference:.1e}'
        )
        if max_difference < tolerance and epoch > 5:
            break

    progress_bar.close()

    return M
    return loss


def estimate_Sigma_x_inv(Xs, Sigma_x_inv, adjacency_lists, use_spatial, lambda_Sigma_x_inv, betas, optimizer, context, n_epochs=10000):
    """Optimize Sigma_x_inv parameters.

    
    Returns:
        Updated estimate of Sigma_x_inv.
    """
   
    num_edges_per_fov = [sum(map(len, adjacency_list)) for adjacency_list in adjacency_lists]

    if not any(sum(map(len, edge)) > 0 and u for edge, u in zip(adjacency_lists, use_spatial)):
        return

    linear_term = torch.zeros_like(Sigma_x_inv).requires_grad_(False)
    # Zs = [torch.tensor(X / np.linalg.norm(X, axis=1, ord=1, keepdims=True), **context) for X in Xs]
    size_factors = [torch.linalg.norm(X, axis=1, ord=1, keepdim=True) for X in Xs ]
    Zs = [X / size_factor for X, size_factor in zip(Xs, size_factors)]
    nus = [] # sum of neighbors' z
    weighted_total_cells = 0
    for Z, adjacency_list, u, beta in zip(Zs, adjacency_lists, use_spatial, betas):
        edges = [(i, j) for i, edge in enumerate(adjacency_list) for j in edge]
        adjacency_matrix = torch.sparse_coo_tensor(np.array(edges).T, np.ones(len(edges)), size=[len(Z)]*2, **context) # Note: this is symmetric
        if u:
            # TODO: will this line break if a cell does not have a neighbor?
            nu = adjacency_matrix @ Z
            linear_term = linear_term.addmm(Z.T, nu, alpha=beta)
        else:
            nu = None
        nus.append(nu)
        weighted_total_cells += beta * sum(map(len, adjacency_matrix))
        del Z, adjacency_matrix
    linear_term = (linear_term + linear_term.T) / 2

    history = []

    if optimizer:
        previous_loss, loss = np.inf, np.nan
        progress_bar = trange(n_epochs, desc='Updating Σx-1')
        Sigma_x_inv_best, loss_best, epoch_best = None, np.inf, -1
        dSigma_x_inv = np.inf
        early_stop_iepoch = 0
        for epoch in progress_bar:
            optimizer.zero_grad()

            loss = Sigma_x_inv.view(-1) @ linear_term.view(-1)
            loss = loss + lambda_Sigma_x_inv * Sigma_x_inv.pow(2).sum() * weighted_total_cells / 2
            for Z, nu, beta in zip(Zs, nus, betas):
                if nu is None: continue
                eta = nu @ Sigma_x_inv
                logZ = integrate_of_exponential_over_simplex(eta)
                loss = loss + beta * logZ.sum()
            loss = loss / weighted_total_cells
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
            with torch.no_grad():
                Sigma_x_inv -= Sigma_x_inv.mean()

            loss = loss.item()
            dloss = previous_loss - loss
            previous_loss = loss

            with torch.no_grad():
                dSigma_x_inv = Sigma_x_inv_prev.sub(Sigma_x_inv).abs().max().item()
            progress_bar.set_description(
                f'Updating Σx-1: loss = {dloss:.1e} -> {loss:.1e} '
                f'δΣx-1 = {dSigma_x_inv:.1e} '
                f'Σx-1 range = {Sigma_x_inv.min().item():.1e} ~ {Sigma_x_inv.max().item():.1e}'
            )

            if Sigma_x_inv.grad.abs().max() < 1e-4 and dSigma_x_inv < 1e-4:
                early_stop_iepoch += 1
            else:
                early_stop_iepoch = 0

            if early_stop_iepoch >= 10 or epoch > epoch_best + 100:
                break

        with torch.no_grad():
            Sigma_x_inv[:] = Sigma_x_inv_best
    else:
        Sigma_x_inv_storage = Sigma_x_inv

        def calc_func_grad(Sigma_x_inv):
            if not Sigma_x_inv.grad:
                Sigma_x_inv.grad = torch.zeros_like(Sigma_x_inv)
            else:
                Sigma_x_inv.grad.zero_()
            loss = Sigma_x_inv.view(-1) @ linear_term.view(-1)
            loss = loss + lambda_Sigma_x_inv * Sigma_x_inv.pow(2).sum() * weighted_total_cells / 2
            for Z, nu, beta in zip(Zs, nus, betas):
                if not nu:
                    continue
                eta = nu @ Sigma_x_inv
                logZ = integrate_of_exponential_over_simplex(eta)
                loss = loss + beta * logZ.sum()
            return loss

        progress_bar = trange(n_epochs)
        step_size = 1e-1
        step_size_update = 2
        dloss = np.inf
        loss = calc_func_grad(Sigma_x_inv)
        loss.backward()
        loss = loss.item()
        for epoch in progress_bar:
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
                progress_bar.set_description(
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

    return history
