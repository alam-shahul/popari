import numpy as np
import torch
from loguru import logger
from tqdm.auto import trange

from popari.sample_for_integral import integrate_of_exponential_over_simplex
from popari.util import (
    IndependentSet,
    NesterovGD,
    graph_neighbors,
    project2simplex,
    project2simplex_,
    project_M,
    project_M_,
    sample_graph_iid,
)


def compute_empirical_spatial_affinities(embeddings, adjacency, sample_axis, scaling, context):
    """Compute one centered empirical spatial-affinity matrix per sample."""

    sample_affinities = {}
    edge_rows = np.repeat(np.arange(adjacency.shape[0]), np.diff(adjacency.indptr))
    for embedding, sample in zip(embeddings, sample_axis.names):
        sample_indices = sample_axis.indices(sample)
        edge_mask = sample_axis.codes[edge_rows] == sample_axis.position(sample)
        rows = edge_rows[edge_mask]
        columns = adjacency.indices[edge_mask]
        if len(rows) == 0:
            affinity = torch.zeros((embedding.shape[1], embedding.shape[1]), **context)
        else:
            normalized = embedding / torch.linalg.norm(embedding, dim=1, keepdim=True, ord=1)
            edges = np.column_stack(
                (np.searchsorted(sample_indices, rows), np.searchsorted(sample_indices, columns)),
            )
            source = normalized[edges[:, 0]]
            target = normalized[edges[:, 1]]
            source = source - source.mean(dim=0, keepdim=True)
            target = target - target.mean(dim=0, keepdim=True)
            correlation = (
                (target / target.std(dim=0, keepdim=True)).T @ (source / source.std(dim=0, keepdim=True)) / len(source)
            )
            affinity = -correlation

        affinity = (affinity + affinity.T) / 2
        affinity -= affinity.mean()
        sample_affinities[sample] = affinity * scaling

    return sample_affinities


def _spatial_affinity_loss(
    spatial_affinity,
    linear_factor,
    neighbor_sums,
    sample_weights,
    weighted_edge_count,
    *,
    regularization_strength,
    regularization_power,
    group_means=None,
    group_regularization_strength=0,
):
    """Return the conditional spatial-affinity objective as a scalar tensor."""

    linear_term = spatial_affinity.flatten() @ linear_factor.flatten()
    regularization = regularization_strength * spatial_affinity.abs().pow(regularization_power).sum()
    if group_means is not None:
        group_weight = 1 / len(group_means)
        regularization += sum(
            group_weight * group_regularization_strength * (group_mean - spatial_affinity).square().sum()
            for group_mean in group_means
        )
    regularization *= weighted_edge_count / 2

    log_partition = spatial_affinity.new_zeros(())
    for neighbor_sum, sample_weight in zip(neighbor_sums, sample_weights):
        eta = neighbor_sum @ spatial_affinity
        log_partition += sample_weight * integrate_of_exponential_over_simplex(eta).sum()

    return (linear_term + regularization + log_partition) / weighted_edge_count


def estimate_spatial_affinity(
    level,
    Sigma_x_inv,
    replicate_mask,
    optimizer,
    Sigma_x_inv_bar=None,
    subsample_rate=None,
    constraint=None,
    n_epochs=1000,
    tol=2e-3,
    check_frequency=50,
):
    """Optimize Sigma_x_inv parameters.

    Differential mode:
    grad =  ... + λ_Sigma_x_inv ( Sigma_x_inv - Sigma_x_inv_bar )

    Args:
        Xs: list of latent expression embeddings for each FOV.
        Sigma_x_inv: previous estimate of Σx-1

    """
    samples = [sample for use_replicate, sample in zip(replicate_mask, level.sample_names) if use_replicate]
    betas = torch.as_tensor(
        [beta for (use_replicate, beta) in zip(replicate_mask, level.betas) if use_replicate],
        **level.context,
    )
    betas = betas / betas.sum()

    global_X = level.embeddings.detach()
    global_Z = global_X / torch.linalg.norm(global_X, axis=1, ord=1, keepdim=True)
    global_nu = level.adjacency_matrix @ global_Z
    num_edges_per_fov = [
        int(np.diff(level.adjacency.indptr)[level.sample_axis.indices(sample)].sum()) for sample in samples
    ]

    if not any(num_edges > 0 for num_edges in num_edges_per_fov):
        return

    linear_term_coefficient = torch.zeros_like(Sigma_x_inv).requires_grad_(False)
    nus = []  # sum of neighbors' z
    weighted_total_cells = 0

    for sample, num_edges, beta in zip(samples, num_edges_per_fov, betas):
        indices = level.sample_indices[sample]
        Z = global_Z.index_select(0, indices)
        nu = global_nu.index_select(0, indices)
        linear_term_coefficient.addmm_(Z.T, nu, alpha=beta)

        nus.append(nu)
        weighted_total_cells += beta * num_edges
        del Z

    for sample, nu in zip(samples, nus):
        if not torch.isfinite(nu).all():
            raise FloatingPointError(f"Neighbor sums contain non-finite values for sample {sample!r}.")
    if not torch.isfinite(Sigma_x_inv).all():
        raise FloatingPointError("Spatial affinity contains non-finite values before optimization.")

    if level.verbose >= 3:
        logger.debug(
            "Spatial-affinity linear coefficient range: {:.2e} to {:.2e}",
            linear_term_coefficient.min().item(),
            linear_term_coefficient.max().item(),
        )

    loss_prev, loss = np.inf, np.nan

    progress_bar = trange(
        1,
        n_epochs + 1,
        desc="Spatial-affinity optimization",
        leave=False,
        disable=level.verbose < 2,
        dynamic_ncols=True,
        mininterval=1,
    )

    Sigma_x_inv_best, loss_best, epoch_best = None, np.inf, -1
    dSigma_x_inv = np.inf
    Sigma_x_inv_prev = Sigma_x_inv.clone().detach()
    for epoch in progress_bar:
        optimizer.zero_grad()

        objective_neighbor_sums = []
        for nu, sample in zip(nus, samples):
            sample_indices = level.sample_axis.indices(sample)
            sample_size = len(sample_indices)
            if subsample_rate is not None:
                node_limit = int(subsample_rate * sample_size)
                sample_adjacency = level.adjacency[sample_indices][:, sample_indices]
                subsample_index = np.sort(sample_graph_iid(sample_adjacency, range(sample_size), node_limit))
                subsample_multiplier = 1 / subsample_rate
                nu = nu[subsample_index]

            objective_neighbor_sums.append(nu)

        objective_weights = betas if subsample_rate is None else betas / subsample_rate
        loss = _spatial_affinity_loss(
            Sigma_x_inv,
            linear_term_coefficient,
            objective_neighbor_sums,
            objective_weights,
            weighted_total_cells,
            regularization_strength=level.lambda_Sigma_x_inv,
            regularization_power=level.spatial_affinity_regularization_power,
            group_means=Sigma_x_inv_bar,
            group_regularization_strength=level.lambda_Sigma_bar,
        )

        if loss < loss_best:
            Sigma_x_inv_best = Sigma_x_inv.clone().detach()
            loss_best = loss.item()
            epoch_best = epoch

        loss.backward()
        Sigma_x_inv.grad = (Sigma_x_inv.grad + Sigma_x_inv.grad.T) / 2
        optimizer.step()
        with torch.no_grad():
            if level.spatial_affinity_centering:
                Sigma_x_inv -= Sigma_x_inv.mean()

            if level.spatial_affinity_constraint == "clamp":
                Sigma_x_inv.clamp_(
                    min=-level.spatial_affinity_scaling,
                    max=level.spatial_affinity_scaling,
                )
            elif level.spatial_affinity_constraint == "scale":
                Sigma_x_inv.mul_(level.spatial_affinity_scaling / Sigma_x_inv.abs().max())

            if epoch % check_frequency == 0 or epoch == n_epochs:
                if not torch.isfinite(Sigma_x_inv).all():
                    raise FloatingPointError(
                        f"Spatial affinity became non-finite at optimization epoch {epoch}.",
                    )
                loss = loss.item()
                loss_prev = loss

                dSigma_x_inv = Sigma_x_inv_prev.sub(Sigma_x_inv).abs().max().item()
                Sigma_x_inv_prev = Sigma_x_inv.clone().detach()

                progress_bar.set_postfix(loss=f"{loss:.1e}", delta=f"{dSigma_x_inv:.1e}")
                if level.verbose >= 3:
                    logger.debug(
                        "Spatial-affinity objective: loss={:.3e}, range={:.3e} to {:.3e}",
                        loss,
                        Sigma_x_inv.min().item(),
                        Sigma_x_inv.max().item(),
                    )

                if dSigma_x_inv < tol * check_frequency or epoch > epoch_best + 2 * check_frequency:
                    break

    progress_bar.close()

    Sigma_x_inv = Sigma_x_inv_best
    Sigma_x_inv.requires_grad_(False)

    return Sigma_x_inv, loss_best * weighted_total_cells


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
