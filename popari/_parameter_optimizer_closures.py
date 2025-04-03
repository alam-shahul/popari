import awkward as ak
import numpy as np
import torch
from tqdm.auto import tqdm, trange

from popari._popari_dataset import PopariDataset
from popari.sample_for_integral import integrate_of_exponential_over_simplex
from popari.util import (
    IndependentSet,
    NesterovGD,
    convert_numpy_to_pytorch_sparse_coo,
    get_datetime,
    project2simplex,
    project2simplex_,
    project_M,
    project_M_,
    sample_graph_iid,
)

######################### Nll M Closure Functions #########################


def compute_loss_nll_M_closure(
    M,
    quadratic_factor,
    differential_regularization_quadratic_factor,
    linear_term,
    differential_regularization_linear_term,
    constant,
    metagene_mode,
    M_bar,
    lambda_M,
):
    def compute_loss(M):
        quadratic_factor_grad = M @ (quadratic_factor + differential_regularization_quadratic_factor)
        loss = (quadratic_factor_grad * M).sum()
        linear_term_grad = linear_term + differential_regularization_linear_term
        loss -= 2 * (linear_term_grad * M).sum()

        loss += constant

        if metagene_mode == "differential" and M_bar is not None:
            differential_regularization_term = (M @ differential_regularization_quadratic_factor * M).sum() - 2 * (
                differential_regularization_linear_term * M
            ).sum()
            group_weighting = 1 / len(M_bar)
            for group_M_bar in M_bar:
                differential_regularization_term += group_weighting * lambda_M * (group_M_bar * group_M_bar).sum()

        loss /= 2

        return loss.item()

    return compute_loss(M)


######################### Estimate M Closure Functions #########################
"""
def compute_loss_and_gradient_M_closure(M, quadratic_factor, differential_regularization_quadratic_factor, verbose, differential_regularization_linear_factor, constant, metagene_mode, M_bar, lambda_M, M_constraint):
    def compute_loss_and_gradient(M):
        quadratic_factor_grad = M @ (quadratic_factor + differential_regularization_quadratic_factor)
        loss = (quadratic_factor_grad * M).sum()
        verbose_description = ""
        if verbose > 2:
            verbose_description += f"M quadratic term: {loss:.1e}"
        linear_term_grad = linear_factor + differential_regularization_linear_factor
        loss -= 2 * (linear_term_grad * M).sum()
        grad = quadratic_factor_grad - linear_term_grad

        loss += constant

        if metagene_mode == "differential" and M_bar is not None:
            differential_regularization_term = (M @ differential_regularization_quadratic_factor * M).sum() - 2 * (
                differential_regularization_linear_factor * M
            ).sum()
            group_weighting = 1 / len(M_bar)
            for group_M_bar in M_bar:
                differential_regularization_term += (
                    group_weighting * lambda_M * (group_M_bar * group_M_bar).sum()
                )

        if verbose > 2:
            # print(f"M regularization term: {regularization_term}")
            if metagene_mode == "differential":
                verbose_description += f"M differential regularization term: {differential_regularization_term}"

        loss /= 2

        if M_constraint == "simplex":
            grad.sub_(grad.sum(0, keepdim=True))

        return loss.item(), grad
    return compute_loss_and_gradient(M)
"""


def compute_loss_and_gradient(
    M,
    quadratic_factor,
    differential_regularization_quadratic_factor,
    verbose,
    linear_factor,
    differential_regularization_linear_factor,
    constant,
    metagene_mode,
    M_bar,
    lambda_M,
    M_constraint,
    batch_effects,
):
    quadratic_factor_grad = M @ (quadratic_factor + differential_regularization_quadratic_factor)
    loss = (quadratic_factor_grad * M).sum()
    verbose_description = ""
    if verbose > 2:
        verbose_description += f"M quadratic term: {loss:.1e}"
    linear_term_grad = linear_factor + differential_regularization_linear_factor
    loss -= 2 * (linear_term_grad * M).sum()
    grad = quadratic_factor_grad - linear_term_grad

    if not all(torch.all(tensor == 0) for tensor in batch_effects):
        print("incorrectly in here")
        loss += -0.5 * torch.sum(torch.log(torch.linalg.eigvalsh(M.T @ M) + 1e-10))
        grad += -M @ torch.inverse(M.T @ M + 1e-10 * torch.eye(M.shape[1], device=M.device))

    loss += constant

    if metagene_mode == "differential" and M_bar is not None:
        differential_regularization_term = (M @ differential_regularization_quadratic_factor * M).sum() - 2 * (
            differential_regularization_linear_factor * M
        ).sum()
        group_weighting = 1 / len(M_bar)
        for group_M_bar in M_bar:
            differential_regularization_term += group_weighting * lambda_M * (group_M_bar * group_M_bar).sum()

    if verbose > 2:
        # print(f"M regularization term: {regularization_term}")
        if metagene_mode == "differential":
            verbose_description += f"M differential regularization term: {differential_regularization_term}"

    loss /= 2

    if M_constraint == "simplex":
        grad.sub_(grad.sum(0, keepdim=True))

    return loss.item(), grad


def estimate_M_nag_closure(
    M,
    verbose,
    quadratic_factor,
    differential_regularization_quadratic_factor,
    linear_factor,
    differential_regularization_linear_factor,
    constant,
    metagene_mode,
    M_bar,
    lambda_M,
    progress_bar,
    simplex_projection_mode,
    use_inplace_ops,
    M_constraint,
    tol,
    verbose_bar,
    batch_effects,
):
    def estimate_M_nag(M):
        """Estimate M using Nesterov accelerated gradient descent.

        Args:
            M (torch.Tensor) : current estimate of meteagene parameters

        """
        loss, grad = compute_loss_and_gradient(
            M,
            quadratic_factor,
            differential_regularization_quadratic_factor,
            verbose,
            linear_factor,
            differential_regularization_linear_factor,
            constant,
            metagene_mode,
            M_bar,
            lambda_M,
            M_constraint,
            batch_effects,
        )
        if verbose > 1:
            print(f"M NAG Initial Loss: {loss}")

        step_size = 1 / torch.linalg.eigvalsh(quadratic_factor).max().item()
        loss = np.inf

        optimizer = NesterovGD(M.clone(), step_size)
        for epoch in progress_bar:
            loss_prev = loss
            M_prev = M.clone()

            # Update M
            loss, grad = compute_loss_and_gradient(
                M,
                quadratic_factor,
                differential_regularization_quadratic_factor,
                verbose,
                linear_factor,
                differential_regularization_linear_factor,
                constant,
                metagene_mode,
                M_bar,
                lambda_M,
                M_constraint,
                batch_effects,
            )
            M = optimizer.step(grad)
            if simplex_projection_mode == "exact":
                if use_inplace_ops:
                    M = project_M_(M, M_constraint)
                else:
                    M = project_M(M, M_constraint)
            elif simplex_projection_mode == "approximate":
                raise NotImplementedError()

            optimizer.set_parameters(M)

            dloss = loss_prev - loss
            dM = (M_prev - M).abs().max().item()
            stop_criterion = dM < tol and epoch > 5
            assert not np.isnan(loss)
            if epoch % 5 == 0 or stop_criterion:
                description = (
                    f"Updating M: loss = {loss:.1e}, "
                    f"%δloss = {dloss / loss:.1e}, "
                    f"δM = {dM:.1e}"
                    # f'lr={step_size_scale:.1e}'
                )
                progress_bar.set_description(description)
            if stop_criterion:
                break

        verbose_bar.close()
        progress_bar.close()

        loss, grad = compute_loss_and_gradient(
            M,
            quadratic_factor,
            differential_regularization_quadratic_factor,
            verbose,
            linear_factor,
            differential_regularization_linear_factor,
            constant,
            metagene_mode,
            M_bar,
            lambda_M,
            M_constraint,
            batch_effects,
        )
        if verbose > 1:
            print(f"M NAG Final Loss: {loss}")

        return M

    return estimate_M_nag(M)
