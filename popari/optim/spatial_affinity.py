import numpy as np
import torch
from loguru import logger
from torch import nn
from tqdm.auto import trange

from popari._named_state import ParameterDict
from popari.optim.batching import sample_graph_iid
from popari.optim.simplex_integral import integrate_of_exponential_over_simplex


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

    progress_bar = trange(
        1,
        n_epochs + 1,
        desc="Spatial-affinity optimization",
        leave=False,
        disable=level.verbose < 2,
        dynamic_ncols=True,
        mininterval=1,
    )

    Sigma_x_inv_best = Sigma_x_inv.detach().clone()
    loss_best = Sigma_x_inv.new_full((), torch.inf)
    epoch_best = torch.full((), -1, dtype=torch.long, device=Sigma_x_inv.device)
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

        with torch.no_grad():
            improved = loss.detach() < loss_best
            loss_best = torch.where(improved, loss.detach(), loss_best)
            Sigma_x_inv_best = torch.where(improved, Sigma_x_inv.detach(), Sigma_x_inv_best)
            epoch_best = torch.where(improved, epoch_best.new_tensor(epoch), epoch_best)

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
                loss_value = loss.detach().item()
                if not np.isfinite(loss_value):
                    raise FloatingPointError(f"Spatial-affinity objective became non-finite at epoch {epoch}.")
                epoch_best_value = epoch_best.item()

                dSigma_x_inv = Sigma_x_inv_prev.sub(Sigma_x_inv).abs().max().item()
                Sigma_x_inv_prev = Sigma_x_inv.clone().detach()

                progress_bar.set_postfix(loss=f"{loss_value:.1e}", delta=f"{dSigma_x_inv:.1e}")
                if level.verbose >= 3:
                    logger.debug(
                        "Spatial-affinity objective: loss={:.3e}, range={:.3e} to {:.3e}",
                        loss_value,
                        Sigma_x_inv.min().item(),
                        Sigma_x_inv.max().item(),
                    )

                if dSigma_x_inv < tol * check_frequency or epoch > epoch_best_value + 2 * check_frequency:
                    break

    progress_bar.close()

    Sigma_x_inv_best.requires_grad_(False)

    return Sigma_x_inv_best, loss_best * weighted_total_cells


class SpatialAffinityState(nn.Module):
    """Registered spatial affinities and their runtime grouping behavior."""

    def __init__(self, K, sample_names, groups, mode, context):
        super().__init__()
        self.sample_names = tuple(sample_names)
        self.groups = {name: list(samples) for name, samples in groups.items()}
        self.mode = mode
        self.tags = {
            sample: [group for group, samples in self.groups.items() if sample in samples]
            for sample in self.sample_names
        }

        self.values = ParameterDict(prefix="spatial_affinity")
        if mode == "shared lookup":
            self._sample_to_parameter = {sample: group for group, samples in self.groups.items() for sample in samples}
            parameter_names = tuple(self.groups)
        elif mode == "differential lookup":
            self._sample_to_parameter = {sample: sample for sample in self.sample_names}
            parameter_names = self.sample_names
        else:
            raise NotImplementedError(f"{mode=} is not implemented.")
        self.parameter_names = tuple(parameter_names)
        for name in parameter_names:
            self.values[name] = torch.zeros((K, K), **context)

    def for_sample(self, sample: str) -> torch.Tensor:
        """Return the affinity matrix applicable to a sample."""

        return self.values[self._sample_to_parameter[sample]]

    def set_sample_(self, sample: str, value: torch.Tensor) -> None:
        """Copy an affinity matrix into the parameter applicable to a sample."""

        with torch.no_grad():
            self.for_sample(sample).copy_(value)

    def group_means(self) -> dict[str, torch.Tensor]:
        """Return detached arithmetic means of current sample affinities."""

        with torch.no_grad():
            return {
                group: torch.stack([self.for_sample(sample) for sample in samples]).mean(dim=0)
                for group, samples in self.groups.items()
            }

    def initialize_(self, sample_values: dict[str, torch.Tensor], betas: torch.Tensor) -> None:
        """Initialize state from one empirical affinity matrix per sample."""

        with torch.no_grad():
            if self.mode == "shared lookup":
                for group, samples in self.groups.items():
                    shared_affinity = self.values[group]
                    shared_affinity.zero_()
                    for sample, beta in zip(self.sample_names, betas):
                        if sample in samples:
                            shared_affinity.add_(beta * sample_values[sample])
            else:
                for sample in self.sample_names:
                    self.values[sample].copy_(sample_values[sample])

    def get_extra_state(self):
        return {
            "sample_names": self.sample_names,
            "groups": self.groups,
            "mode": self.mode,
        }

    def set_extra_state(self, state):
        if tuple(state["sample_names"]) != self.sample_names:
            raise RuntimeError(
                f"{self.__class__.__name__} checkpoint samples do not match the current samples.",
            )
        loaded_groups = {name: list(samples) for name, samples in state["groups"].items()}
        if loaded_groups != self.groups:
            raise RuntimeError(
                f"{self.__class__.__name__} checkpoint groups do not match the current groups.",
            )
        if state.get("mode") != self.mode:
            raise RuntimeError(f"{self.__class__.__name__} checkpoint mode is incompatible with the current model.")
