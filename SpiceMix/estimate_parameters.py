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
def estimate_M(Xs, M, sigma_yxs, betas, datasets, M_constraint, context,
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
    quadratic_factor = torch.zeros([K, K], **context)
    linear_term = torch.zeros_like(M)
    scaled_betas = betas / (sigma_yxs**2)
    
    # ||Y||_2^2
    constant_magnitude = np.array([torch.linalg.norm(torch.tensor(dataset.X, **context)).item()**2 for dataset in datasets]).sum()

    constant = (np.array([torch.linalg.norm(torch.tensor(dataset.X, **context)).item()**2 for dataset in datasets]) * scaled_betas).sum()
    regularization = [dataset.uns["spicemixplus_hyperparameters"]["prior_x"] for dataset in datasets]
    for dataset, X, sigma_yx, scaled_beta in zip(datasets, Xs, sigma_yxs, scaled_betas):
        # X_c^TX_c
        quadratic_factor.addmm_(X.T, X, alpha=scaled_beta)
        # MX_c^TY_c
        linear_term.addmm_(torch.tensor(dataset.X, **context).T.to(X.device), X, alpha=scaled_beta)

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


        if M_constraint == 'simplex':
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
            M = project_M(M, M_constraint)
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
            M = project_M(M, M_constraint)
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
            M_new = project_M(M_new, M_constraint)
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

    # step_size = 1 / torch.linalg.eigvalsh(quadratic_factor).max().item()
    # for epoch in progress_bar:
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
    #   stop_criterion = dM < tol and epoch > 5
    #   if epoch % 1000 == 0 or stop_criterion:
    #       progress_bar.set_description(
    #           f'Updating M: loss = {loss:.1e}, '
    #           f'%δloss = {(loss_prev - loss) / loss:.1e}, '
    #           f'δM = {dM:.1e}'
    #       )
    #   if stop_criterion: break
    # progress_bar.close()
    # return loss


def estimate_Sigma_x_inv(Xs, Sigma_x_inv, spatial_flags, lambda_Sigma_x_inv, betas, optimizer, context, datasets, Sigma_x_inv_bar=None, n_epochs=1000):
    """Optimize Sigma_x_inv parameters.

   
    Differential mode:
    grad = ( M XT X - YT X ) / ( σ_yx^2 ) + λ_Sigma_x_inv ( Sigma_x_inv - Sigma_x_inv_bar )

    Args:
        Xs: list of latent expression embeddings for each FOV.
        Sigma_x_inv: previous estimate of Σx-1

    """
    num_edges_per_fov = [sum(map(len, dataset.obs["adjacency_list"])) for dataset in datasets]

    if not any(sum(map(len, dataset.obs["adjacency_list"])) > 0 and u for dataset, u in zip(datasets, spatial_flags)):
        return

    linear_term_coefficient = torch.zeros_like(Sigma_x_inv).requires_grad_(False)
    # Zs = [torch.tensor(X / np.linalg.norm(X, axis=1, ord=1, keepdims=True), **context) for X in Xs]
    size_factors = [torch.linalg.norm(X, axis=1, ord=1, keepdim=True) for X in Xs ]
    Zs = [X / size_factor for X, size_factor in zip(Xs, size_factors)]
    nus = [] # sum of neighbors' z
    weighted_total_cells = 0
    for Z, dataset, use_spatial, beta in zip(Zs, datasets, spatial_flags, betas):
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

    if optimizer:
        loss_prev, loss = np.inf, np.nan
        progress_bar = trange(n_epochs, desc='Updating Σx-1')
        Sigma_x_inv_best, loss_best, epoch_best = None, np.inf, -1
        dSigma_x_inv = np.inf
        early_stop_epoch_count = 0
        for epoch in progress_bar:
            optimizer.zero_grad()

            # Compute loss 
            linear_term = Sigma_x_inv.view(-1) @ linear_term_coefficient.view(-1)
            regularization = lambda_Sigma_x_inv * Sigma_x_inv.pow(2).sum() * weighted_total_cells / 2
            if Sigma_x_inv_bar is not None:
                regularization += lambda_Sigma_x_inv * 100 * (Sigma_x_inv_bar - Sigma_x_inv).pow(2).sum() * weighted_total_cells / 2
            
            log_partition_function = 0
            for Z, nu, beta in zip(Zs, nus, betas):
                if nu is None:
                    continue
                assert torch.isfinite(nu).all()
                assert torch.isfinite(Sigma_x_inv).all()
                eta = nu @ Sigma_x_inv
                logZ = integrate_of_exponential_over_simplex(eta)
                log_partition_function += beta * logZ.sum()

            loss = (linear_term + regularization + log_partition_function) / weighted_total_cells
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

        return Sigma_x_inv, loss * weighted_total_cells
    else:
        pass
        # Sigma_x_inv_storage = Sigma_x_inv

        # def compute_loss_and_gradient(Sigma_x_inv):
        #     if Sigma_x_inv.grad is None:
        #         Sigma_x_inv.grad = torch.zeros_like(Sigma_x_inv)
        #     else:
        #         Sigma_x_inv.grad.zero_()
        #     loss = Sigma_x_inv.view(-1) @ linear_term.view(-1)
        #     loss = loss + lambda_Sigma_x_inv * Sigma_x_inv.pow(2).sum() * weighted_total_cells / 2
        #     for Z, nu, beta in zip(Zs, nus, betas):
        #         if nu is None:
        #             continue
        #         eta = nu @ Sigma_x_inv
        #         logZ = integrate_of_exponential_over_simplex(eta)
        #         loss = loss + beta * logZ.sum()
        #     return loss

        # progress_bar = trange(n_epochs)
        # step_size = 1e-1
        # step_size_update = 2
        # dloss = np.inf
        # loss = compute_loss_and_gradient(Sigma_x_inv)
        # loss.backward()
        # loss = loss.item()
        # for epoch in progress_bar:
        #     with torch.no_grad():
        #         Sigma_x_inv_new = Sigma_x_inv.add(Sigma_x_inv.grad, alpha=-step_size).requires_grad_(True)
        #         Sigma_x_inv_new.sub_(Sigma_x_inv_new.mean())
        #     loss_new = compute_loss_and_gradient(Sigma_x_inv_new)
        #     with torch.no_grad():
        #         if loss_new.item() < loss:
        #             loss_new.backward()
        #             loss_new = loss_new.item()
        #             dloss = loss - loss_new
        #             loss = loss_new
        #             dSigma_x_inv = Sigma_x_inv.sub(Sigma_x_inv_new).abs().max().item()
        #             Sigma_x_inv = Sigma_x_inv_new
        #             step_size *= step_size_update
        #         else:
        #             step_size /= step_size_update
        #         progress_bar.set_description(
        #             f'Updating Σx-1: loss = {dloss:.1e} -> {loss:.1e}, '
        #             f'δΣx-1 = {dSigma_x_inv:.1e} '
        #             f'lr = {step_size:.1e} '
        #             f'Σx-1 range = {Sigma_x_inv.min().item():.1e} ~ {Sigma_x_inv.max().item():.1e}'
        #         )
        #         if (Sigma_x_inv.grad * step_size).abs().max() < 1e-3:
        #             dloss = np.nan
        #             dSigma_x_inv = np.nan
        #             break

        # with torch.no_grad():
        #     Sigma_x_inv_storage = Sigma_x_inv
    Sigma_x_inv.requires_grad_(False)

def estimate_phenotype_predictor(input_list, phenotypes, phenotype_name, predictor, optimizer, compute_loss, n_epoch=10000):
    """Train phenotype predictor on train data.

    Args:
        input_list: 

    """
    progress_bar = trange(n_epoch, desc=f'Updating `{phenotype_name}` predictor')
    for param in predictor.parameters():
        param.requires_grad_(True)

    predictor.train()
    loss_best = np.inf
    epoch_best = 0
    loss_prev = np.nan
    early_stop_epoch_count = 0
    history = []
    for epoch in progress_bar:
        loss_total = 0
        optimizer.zero_grad()
        metrics = defaultdict(lambda: 0)
        for x, y in zip(input_list, phenotypes):
            yhat = predictor(x)
            loss = compute_loss(yhat, y, mode='train')
            loss_total += loss.item()
            loss.backward()
            metrics['loss'] += loss.item()
            metrics['acc'] += (yhat.argmax(axis=1) == y).sum().item()
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

        l2_loss = torch.sqrt(sum(torch.linalg.norm(param)**2 for param in predictor.parameters()))

        history.append(metrics)
        progress_bar.set_description(
            f'Updating `{phenotype_name}` predictor. '
            f'loss:{dloss:.1e} -> {loss_total:.1e} best={loss_best:.1e} '
            f'L2={l2_loss:.1e} '
            f"acc={metrics['acc']:.2f}"
        )
        # if dloss < 1e-4: break
        if early_stop_epoch_count >= 10 or epoch > epoch_best + 100:
            break
    
    progress_bar.close()
    predictor.eval()
    for param in predictor.parameters():
        param.requires_grad_(False)

def estimate_M_and_predictor():
    pass
