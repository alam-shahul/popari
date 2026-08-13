import numpy as np
import torch
from loguru import logger
from tqdm.auto import trange

from popari.optim.nesterov import NesterovGD
from popari.optim.projection import project_M, project_M_


def _metagene_loss(metagenes, quadratic_factor, linear_factor, constant):
    """Return the conditional metagene objective as a scalar tensor."""

    loss = ((metagenes @ quadratic_factor) * metagenes).sum()
    loss -= 2 * (linear_factor * metagenes).sum()
    return (loss + constant) / 2


@torch.no_grad()
def estimate_metagenes(
    level,
    M,
    replicate_mask,
    n_epochs=10000,
    tol=1e-3,
    backend_algorithm="gd Nesterov",
    simplex_projection_mode=False,
):
    """Optimize metagene parameters.

    M is shared across all replicates.
    min || Y - X MT ||_2^2 / (2 σ_yx^2)
    s.t. || Mk ||_p = 1
    grad = (M XT X - YT X) / (σ_yx^2)

    Args:
        M: current estimate of metagene parameters
        betas: weight of each FOV in optimization scheme
        context: context ith which to create PyTorch tensor
        n_epochs: number of epochs

    Returns:
        Updated estimate of metagene parameters.

    """

    G, K = M.shape
    quadratic_factor = torch.zeros([K, K], **level.context)
    linear_factor = torch.zeros_like(M)
    # TODO: replace below (and any reference to dataset)

    tol /= G

    samples = [sample for use_replicate, sample in zip(replicate_mask, level.sample_names) if use_replicate]
    Xs = [level.embedding(sample) for sample in samples]
    Ys = [Y for (use_replicate, Y) in zip(replicate_mask, level.Ys) if use_replicate]
    sigma_yxs = level.sigma_yxs[replicate_mask]

    betas = level.betas[replicate_mask]
    betas /= betas.sum()

    scaled_betas = betas / (sigma_yxs**2)

    constant = torch.zeros((), **level.context)
    for Y, scaled_beta in zip(Ys, scaled_betas):
        constant += scaled_beta * torch.square(Y).sum()

    if level.verbose >= 3:
        logger.debug("Metagene objective constant: {:.3e}", constant)
        # print(f"M constant magnitude: {constant_magnitude:.1e}")

    for X, Y, scaled_beta in zip(Xs, Ys, scaled_betas):
        # X_c^TX_c
        quadratic_factor.addmm_(X.T, X, alpha=scaled_beta)
        # MX_c^TY_c
        linear_factor.addmm_(Y.T, X, alpha=scaled_beta)

    if level.verbose >= 3:
        logger.debug("Metagene linear-term norm: {:.3e}", torch.linalg.norm(linear_factor).item())
    loss_prev, loss = np.inf, np.nan

    progress_bar = trange(
        n_epochs,
        desc="Metagene optimization",
        leave=False,
        disable=level.verbose < 2,
        dynamic_ncols=True,
        mininterval=1,
    )

    def compute_loss_and_gradient(M):
        quadratic_factor_grad = M @ quadratic_factor
        grad = quadratic_factor_grad - linear_factor

        if level.M_constraint == "simplex":
            grad.sub_(grad.sum(0, keepdim=True))

        return _metagene_loss(M, quadratic_factor, linear_factor, constant).item(), grad

    def estimate_metagenes_nag(M):
        """Estimate M using Nesterov accelerated gradient descent.

        Args:
            M (torch.Tensor) : current estimate of meteagene parameters

        """
        loss, grad = compute_loss_and_gradient(M)
        if level.verbose >= 3:
            logger.debug("Initial metagene loss: {:.3e}", loss)

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
                if level.use_inplace_ops:
                    M = project_M_(M, level.M_constraint)
                else:
                    M = project_M(M, level.M_constraint)
            elif simplex_projection_mode == "approximate":
                raise NotImplementedError()

            optimizer.set_parameters(M)

            dloss = loss_prev - loss
            dM = (M_prev - M).abs().max().item()
            stop_criterion = dM < tol and epoch > 5
            assert not np.isnan(loss)
            if epoch % 5 == 0 or stop_criterion:
                progress_bar.set_postfix(loss=f"{loss:.1e}", delta=f"{dM:.1e}")
            if stop_criterion:
                break

        progress_bar.close()

        loss, grad = compute_loss_and_gradient(M)
        if level.verbose >= 3:
            logger.debug("Final metagene loss: {:.3e}", loss)

        return M, loss

    if backend_algorithm == "mu":
        for epoch in progress_bar:
            loss = _metagene_loss(M, quadratic_factor, linear_factor, constant).item()
            numerator = linear_factor
            denominator = M @ quadratic_factor
            multiplicative_factor = numerator / denominator

            M_prev = M.clone()
            # multiplicative_factor.clip_(max=10)
            M *= multiplicative_factor
            if simplex_projection_mode == "exact":
                if level.use_inplace_ops:
                    M = project_M_(M, level.M_constraint)
                else:
                    M = project_M(M, level.M_constraint)
            elif simplex_projection_mode == "approximate":
                pass
            dM = M_prev.sub(M).abs_().max().item()

            stop_criterion = dM < tol and epoch > 5
            if epoch % 1000 == 0 or stop_criterion:
                progress_bar.set_description(
                    f"Updating M: loss = {loss:.1e}, " f"%δloss = {(loss_prev - loss) / loss:.1e}, " f"δM = {dM:.1e}",
                )
            if stop_criterion:
                break

    elif backend_algorithm == "gd":
        step_size = 1 / torch.linalg.eigvalsh(quadratic_factor).max().item()
        step_size_scale = 1
        loss, grad = compute_loss_and_gradient(M)
        dM = dloss = np.inf
        for epoch in progress_bar:
            M_new = M.sub(grad, alpha=step_size * step_size_scale)
            if simplex_projection_mode == "exact":
                if level.use_inplace_ops:
                    M = project_M_(M_new, level.M_constraint)
                else:
                    M = project_M(M_new, level.M_constraint)
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
                step_size_scale *= 0.5
                step_size_scale = max(step_size_scale, 1.0)

            stop_criterion = dM < tol and epoch > 5
            if epoch % 1000 == 0 or stop_criterion:
                progress_bar.set_description(
                    f"Updating M: loss = {loss:.1e}, "
                    f"%δloss = {dloss / loss:.1e}, "
                    f"δM = {dM:.1e}, "
                    f"lr={step_size_scale:.1e}",
                )
            if stop_criterion:
                break

    elif backend_algorithm == "gd Nesterov":
        M, loss = estimate_metagenes_nag(M)
    else:
        raise NotImplementedError

    if backend_algorithm != "gd Nesterov":
        loss = _metagene_loss(M, quadratic_factor, linear_factor, constant).item()
    return M, loss
