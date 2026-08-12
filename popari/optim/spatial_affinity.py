import numpy as np
import torch
from loguru import logger
from sklearn.decomposition import NMF
from torch import nn
from tqdm.auto import trange

from popari._named_state import ParameterDict
from popari.optim.projection import project2simplex_
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


def _spatial_affinity_losses(
    affinities,
    linear_factors,
    packed_neighbor_sums,
    observation_weights,
    observation_mask,
    weighted_edge_counts,
    *,
    regularization_strength,
    regularization_power,
    group_means=None,
    group_membership=None,
    group_regularization_strength=0,
):
    """Return one conditional objective for each batched affinity.

    The leading dimension indexes unique affinity parameters, the second
    dimension contains padded observations, and the final dimensions index
    metagenes. Padded observations are excluded by ``observation_mask``.

    """

    linear_terms = (affinities * linear_factors).sum(dim=(1, 2))
    eta = torch.bmm(packed_neighbor_sums, affinities)
    valid_eta = eta[observation_mask]
    parameter_ids = torch.arange(len(affinities), device=affinities.device)[:, None].expand_as(observation_mask)[
        observation_mask
    ]
    partition_terms = affinities.new_zeros(len(affinities))
    partition_terms.scatter_add_(
        0,
        parameter_ids,
        observation_weights[observation_mask] * integrate_of_exponential_over_simplex(valid_eta),
    )

    regularization = regularization_strength * affinities.abs().pow(regularization_power).sum(dim=(1, 2))
    if group_means is not None:
        squared_distances = (affinities[:, None] - group_means[None]).square().sum(dim=(2, 3))
        normalized_membership = group_membership / group_membership.sum(dim=1, keepdim=True).clamp_min(1)
        regularization += group_regularization_strength * (normalized_membership * squared_distances).sum(dim=1)
    regularization *= weighted_edge_counts / 2

    nonzero_edges = weighted_edge_counts > 0
    losses = affinities.new_zeros(len(affinities))
    losses[nonzero_edges] = (
        linear_terms[nonzero_edges] + partition_terms[nonzero_edges] + regularization[nonzero_edges]
    ) / weighted_edge_counts[nonzero_edges]
    return losses


def _prepare_affinity_inputs(level):
    """Prepare graph statistics that remain fixed during affinity optimization.

    Observations governed by the same unique affinity parameter are packed
    together. Parameter batches are padded to a common length so their spatial
    objectives can be evaluated with one set of tensor operations.

    """

    state = level.spatial_affinity
    parameter_names = state.parameter_names
    samples_by_parameter = state.samples_by_parameter()

    # Compute normalized embeddings and neighbor sums once, before the inner
    # optimization loop where only the affinity matrices change.
    global_embedding = level.embeddings.detach()
    global_normalized = global_embedding / torch.linalg.norm(global_embedding, dim=1, ord=1, keepdim=True)
    global_neighbor_sums = level.adjacency_matrix @ global_normalized
    if not torch.isfinite(global_neighbor_sums).all():
        raise FloatingPointError("Neighbor sums contain non-finite values.")

    edge_counts = np.diff(level.adjacency.indptr)
    max_size = max(
        (
            sum(len(level.sample_axis.indices(sample)) for sample in samples)
            for samples in samples_by_parameter.values()
        ),
        default=0,
    )
    num_parameters = len(parameter_names)
    packed_indices = torch.zeros((num_parameters, max_size), dtype=torch.long, device=global_embedding.device)
    observation_weights = global_embedding.new_zeros((num_parameters, max_size))
    observation_mask = torch.zeros((num_parameters, max_size), dtype=torch.bool, device=global_embedding.device)
    weighted_edge_counts = global_embedding.new_zeros(num_parameters)

    sample_positions = {sample: index for index, sample in enumerate(level.sample_names)}
    for parameter_index, (parameter_name, samples) in enumerate(samples_by_parameter.items()):
        # Samples sharing one affinity retain their relative model weights, but
        # their weights are normalized within that affinity parameter.
        sample_betas = torch.stack([level.betas[sample_positions[sample]] for sample in samples])
        sample_betas = sample_betas / sample_betas.sum()
        offset = 0
        for sample, beta in zip(samples, sample_betas):
            indices = level.sample_indices[sample]
            stop = offset + len(indices)
            packed_indices[parameter_index, offset:stop] = indices
            observation_weights[parameter_index, offset:stop] = beta
            observation_mask[parameter_index, offset:stop] = True
            weighted_edge_counts[parameter_index] += beta * int(
                edge_counts[level.sample_axis.indices(sample)].sum(),
            )
            offset = stop

    packed_normalized = global_normalized[packed_indices]
    packed_neighbor_sums = global_neighbor_sums[packed_indices]
    # This coefficient is fixed while affinities are optimized and therefore
    # should not be reconstructed during every inner epoch.
    linear_factors = torch.einsum(
        "pn,pnk,pnl->pkl",
        observation_weights,
        packed_normalized,
        packed_neighbor_sums,
    )
    return linear_factors, packed_neighbor_sums, observation_weights, observation_mask, weighted_edge_counts


def estimate_spatial_affinities(
    level,
    optimizer,
    *,
    differentiate=True,
    subsample_rate=None,
    n_epochs=1000,
    tol=2e-3,
    check_frequency=50,
):
    """Jointly optimize every unique spatial-affinity parameter."""

    if subsample_rate is not None:
        raise NotImplementedError("Spatial-affinity subsampling is not supported by the vectorized optimizer.")

    state = level.spatial_affinity
    parameters = list(state.parameters())
    affinities = torch.stack([state.for_parameter(name) for name in state.parameter_names])
    if not torch.isfinite(affinities).all():
        raise FloatingPointError("Spatial affinity contains non-finite values before optimization.")

    (
        linear_factors,
        packed_neighbor_sums,
        observation_weights,
        observation_mask,
        weighted_edge_counts,
    ) = _prepare_affinity_inputs(level)
    active = weighted_edge_counts > 0
    if not active.any():
        return weighted_edge_counts.detach()

    # Differential group means are frozen across the inner optimization loop.
    group_means = None
    interaction_group_means = None
    group_membership = None
    if differentiate and level.spatial_affinity_mode == "differential lookup":
        group_names = tuple(state.groups)
        if state.parameterization == "full":
            group_mean_snapshot = state.group_means()
            group_means = torch.stack([group_mean_snapshot[name] for name in group_names])
        else:
            group_mean_snapshot = state.interaction_group_means()
            interaction_group_means = torch.stack([group_mean_snapshot[name] for name in group_names])
        group_membership = affinities.new_tensor(
            [
                [group in state.tags[parameter_name] for group in group_names]
                for parameter_name in state.parameter_names
            ],
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
    best_losses = affinities.new_full((len(state.parameter_names),), torch.inf)
    best_losses[~active] = 0
    if state.parameterization == "full":
        best_affinities = affinities.detach().clone()
        best_epochs = torch.full((len(state.parameter_names),), -1, dtype=torch.long, device=affinities.device)
    else:
        best_parameters = [parameter.detach().clone() for parameter in parameters]
        best_total_loss = affinities.new_tensor(torch.inf)
        best_epoch = torch.full((), -1, dtype=torch.long, device=affinities.device)
    previous_affinities = affinities.detach().clone()

    for epoch in progress_bar:
        optimizer.zero_grad(set_to_none=True)
        affinities = torch.stack([state.for_parameter(name) for name in state.parameter_names])
        losses = _spatial_affinity_losses(
            affinities,
            linear_factors,
            packed_neighbor_sums,
            observation_weights,
            observation_mask,
            weighted_edge_counts,
            regularization_strength=level.lambda_Sigma_x_inv,
            regularization_power=level.spatial_affinity_regularization_power,
            group_means=group_means,
            group_membership=group_membership,
            group_regularization_strength=level.lambda_Sigma_bar,
        )
        if interaction_group_means is not None:
            interactions = torch.stack([state.interaction_for_parameter(name) for name in state.parameter_names])
            squared_distances = (interactions[:, None] - interaction_group_means[None]).square().sum(dim=(2, 3))
            normalized_membership = group_membership / group_membership.sum(dim=1, keepdim=True).clamp_min(1)
            losses = losses + level.lambda_Sigma_bar * (normalized_membership * squared_distances).sum(dim=1) / 2

        with torch.no_grad():
            if state.parameterization == "full":
                improved = active & (losses.detach() < best_losses)
                best_losses = torch.where(improved, losses.detach(), best_losses)
                best_affinities = torch.where(improved[:, None, None], affinities.detach(), best_affinities)
                best_epochs = torch.where(improved, best_epochs.new_full((), epoch), best_epochs)
            else:
                total_loss = losses[active].sum().detach()
                improved = total_loss < best_total_loss
                best_total_loss = torch.where(improved, total_loss, best_total_loss)
                best_losses = torch.where(improved, losses.detach(), best_losses)
                best_parameters = [
                    torch.where(improved, parameter.detach(), best_parameter)
                    for parameter, best_parameter in zip(parameters, best_parameters)
                ]
                best_epoch = torch.where(improved, best_epoch.new_full((), epoch), best_epoch)

        losses[active].sum().backward()
        state.symmetrize_gradients_()
        optimizer.step()

        state.project_(
            center=level.spatial_affinity_centering,
            constraint=level.spatial_affinity_constraint,
            scaling=level.spatial_affinity_scaling,
        )

        with torch.no_grad():
            if epoch % check_frequency == 0 or epoch == n_epochs:
                affinities = torch.stack([state.for_parameter(name) for name in state.parameter_names])
                if not torch.isfinite(affinities).all():
                    raise FloatingPointError(f"Spatial affinity became non-finite at optimization epoch {epoch}.")
                if not torch.isfinite(losses[active]).all():
                    raise FloatingPointError(f"Spatial-affinity objective became non-finite at epoch {epoch}.")

                deltas = (previous_affinities - affinities).abs().amax(dim=(1, 2))
                previous_affinities = affinities.detach().clone()
                mean_loss = losses[active].mean().item()
                max_delta = deltas[active].max().item()
                progress_bar.set_postfix(loss=f"{mean_loss:.1e}", delta=f"{max_delta:.1e}")
                if level.verbose >= 3:
                    logger.debug(
                        "Spatial-affinity objective: mean_loss={:.3e}, max_delta={:.3e}",
                        mean_loss,
                        max_delta,
                    )

                converged = deltas < tol * check_frequency
                if state.parameterization == "full":
                    stale = epoch > best_epochs + 2 * check_frequency
                    should_stop = torch.all(~active | converged | stale)
                else:
                    should_stop = torch.all(~active | converged) or epoch > best_epoch + 2 * check_frequency
                if should_stop:
                    break

    progress_bar.close()
    with torch.no_grad():
        if state.parameterization == "full":
            for parameter, best_affinity in zip(parameters, best_affinities):
                parameter.copy_(best_affinity)
        else:
            for parameter, best_parameter in zip(parameters, best_parameters):
                parameter.copy_(best_parameter)
    return best_losses * weighted_edge_counts


class SpatialAffinityState(nn.Module):
    """Registered spatial affinities and their runtime grouping behavior."""

    def __init__(self, K, sample_names, groups, mode, context, parameterization="full", rank=None):
        super().__init__()
        if parameterization not in {"full", "factorized"}:
            raise ValueError("spatial_affinity_parameterization must be 'full' or 'factorized'.")
        if rank is None:
            rank = K
        if not 1 <= rank <= K:
            raise ValueError(f"spatial_affinity_rank must satisfy 1 <= rank <= K; received {rank} for K={K}.")

        self.K = K
        self.rank = rank
        self.sample_names = tuple(sample_names)
        self.groups = {name: list(samples) for name, samples in groups.items()}
        self.mode = mode
        self.parameterization = parameterization
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

        self.interactions = ParameterDict(prefix="spatial_affinity_interaction")
        if parameterization == "full":
            self.register_parameter("transform", None)
            for name in parameter_names:
                self.values[name] = torch.zeros((K, K), **context)
        else:
            self.transform = nn.Parameter(torch.zeros((rank, K), **context))
            for name in parameter_names:
                self.interactions[name] = torch.zeros((rank, rank), **context)

    def for_parameter(self, name: str) -> torch.Tensor:
        """Return one effective affinity matrix by parameter name."""

        if self.parameterization == "full":
            return self.values[name]
        return self.transform.T @ self.interactions[name] @ self.transform

    def for_sample(self, sample: str) -> torch.Tensor:
        """Return the affinity matrix applicable to a sample."""

        return self.for_parameter(self._sample_to_parameter[sample])

    def parameter_for_sample(self, sample: str) -> str:
        """Return the unique affinity parameter name used by a sample."""

        return self._sample_to_parameter[sample]

    def transform_for_parameter(self, name: str) -> torch.Tensor:
        """Return the shared spatial transform."""

        if self.parameterization != "factorized":
            raise RuntimeError("Full spatial affinities do not have spatial transforms.")
        if name not in self.parameter_names:
            raise KeyError(name)
        return self.transform

    def transform_for_sample(self, sample: str) -> torch.Tensor:
        """Return the shared spatial transform applicable to one sample."""

        return self.transform_for_parameter(self.parameter_for_sample(sample))

    def interaction_for_parameter(self, name: str) -> torch.Tensor:
        """Return one symmetric spatial-factor interaction matrix."""

        if self.parameterization != "factorized":
            raise RuntimeError("Full spatial affinities do not have factor interactions.")
        return self.interactions[name]

    def interaction_for_sample(self, sample: str) -> torch.Tensor:
        """Return the spatial-factor interaction matrix applicable to a
        sample."""

        return self.interaction_for_parameter(self.parameter_for_sample(sample))

    def set_sample_(self, sample: str, value: torch.Tensor) -> None:
        """Copy an affinity matrix into the parameter applicable to a sample."""

        if self.parameterization != "full":
            raise RuntimeError("Set factorized spatial affinities through load_factors_().")
        with torch.no_grad():
            self.for_sample(sample).copy_(value)

    def group_means(self) -> dict[str, torch.Tensor]:
        """Return detached arithmetic means of current sample affinities."""

        with torch.no_grad():
            return {
                group: torch.stack([self.for_sample(sample) for sample in samples]).mean(dim=0)
                for group, samples in self.groups.items()
            }

    def interaction_group_means(self) -> dict[str, torch.Tensor]:
        """Return detached group means of sample-specific interactions."""

        if self.parameterization != "factorized":
            raise RuntimeError("Full spatial affinities do not have factor interactions.")
        with torch.no_grad():
            return {
                group: torch.stack([self.interaction_for_sample(sample) for sample in samples]).mean(dim=0)
                for group, samples in self.groups.items()
            }

    def initialize_(self, sample_values: dict[str, torch.Tensor], betas: torch.Tensor, random_state=0) -> None:
        """Initialize state from one empirical affinity matrix per sample."""

        with torch.no_grad():
            targets = {}
            if self.mode == "shared lookup":
                for group, samples in self.groups.items():
                    target = next(iter(sample_values.values())).new_zeros((self.K, self.K))
                    for sample, beta in zip(self.sample_names, betas):
                        if sample in samples:
                            target.add_(beta * sample_values[sample])
                    targets[group] = target
            else:
                for sample in self.sample_names:
                    targets[sample] = sample_values[sample]

            if self.parameterization == "full":
                for name, target in targets.items():
                    self.values[name].copy_(target)
                return

            weights = betas / betas.sum()
            mean_absolute_affinity = sum(
                weight * sample_values[sample].abs() for sample, weight in zip(self.sample_names, weights)
            )
            if torch.any(mean_absolute_affinity):
                transform = (
                    NMF(
                        n_components=self.rank,
                        init="nndsvda",
                        random_state=random_state,
                        max_iter=500,
                    )
                    .fit_transform(mean_absolute_affinity.cpu().numpy())
                    .T
                )
            else:
                transform = np.zeros((self.rank, self.K))
                transform[np.arange(self.K) % self.rank, np.arange(self.K)] = 1
            transform = torch.as_tensor(transform, device=self.transform.device, dtype=self.transform.dtype)
            zero_rows = transform.sum(dim=1) == 0
            transform[zero_rows] = 1 / self.K
            transform /= transform.sum(dim=1, keepdim=True)
            self.transform.copy_(transform)
            transform_pseudoinverse = torch.linalg.pinv(transform)
            for name, target in targets.items():
                interaction = transform_pseudoinverse.T @ target @ transform_pseudoinverse
                self.interactions[name].copy_((interaction + interaction.T) / 2)

    def load_factors_(self, factors: dict) -> None:
        """Load a materialized factorized affinity state."""

        if self.parameterization != "factorized":
            raise RuntimeError("Only factorized affinity state accepts persisted factors.")
        if "A" not in factors or not isinstance(factors.get("B"), dict):
            raise ValueError("Factorized results must contain one shared `A` matrix and sample-specific `B` matrices.")
        transform = torch.as_tensor(factors["A"], device=self.transform.device, dtype=self.transform.dtype)
        if transform.shape != self.transform.shape:
            raise ValueError(f"Persisted A has shape {transform.shape}; expected {tuple(self.transform.shape)}.")
        if (
            not torch.isfinite(transform).all()
            or torch.any(transform < 0)
            or not torch.allclose(
                transform.sum(dim=1),
                transform.new_ones(self.rank),
                atol=1e-6,
                rtol=1e-6,
            )
        ):
            raise ValueError("Persisted A must be finite, nonnegative, and row-stochastic.")
        interactions = factors["B"]
        with torch.no_grad():
            self.transform.copy_(transform)
            for parameter_name, samples in self.samples_by_parameter().items():
                values = [
                    torch.as_tensor(interactions[sample], device=transform.device, dtype=transform.dtype)
                    for sample in samples
                ]
                reference = values[0]
                if reference.shape != self.interactions[parameter_name].shape:
                    raise ValueError(
                        f"Persisted B for {samples[0]!r} has shape {reference.shape}; "
                        f"expected {tuple(self.interactions[parameter_name].shape)}.",
                    )
                if not torch.isfinite(reference).all() or not torch.allclose(
                    reference,
                    reference.T,
                    atol=1e-6,
                    rtol=1e-6,
                ):
                    raise ValueError(f"Persisted B for {samples[0]!r} must be finite and symmetric.")
                if any(not torch.allclose(value, reference) for value in values[1:]):
                    raise ValueError(
                        f"Samples sharing affinity parameter {parameter_name!r} have different B matrices.",
                    )
                self.interactions[parameter_name].copy_(reference)

    def materialized_factors(self) -> dict:
        """Return factorized state using sample-keyed interaction matrices."""

        if self.parameterization != "factorized":
            raise RuntimeError("Full spatial affinities do not have A/B factors.")
        return {
            "A": self.transform.detach().cpu().numpy(),
            "B": {sample: self.interaction_for_sample(sample).detach().cpu().numpy() for sample in self.sample_names},
        }

    def samples_by_parameter(self) -> dict[str, list[str]]:
        """Return samples governed by each unique affinity parameter."""

        result = {name: [] for name in self.parameter_names}
        for sample in self.sample_names:
            result[self.parameter_for_sample(sample)].append(sample)
        return result

    def project_(self, *, center=False, constraint=None, scaling=10) -> None:
        """Project trainable state onto its affinity constraints."""

        with torch.no_grad():
            if self.parameterization == "factorized":
                if center:
                    raise ValueError("Centering is not supported for factorized spatial affinities.")
                project2simplex_(self.transform, dim=1, minimum_value=0)
                for interaction in self.interactions.values():
                    interaction.copy_((interaction + interaction.T) / 2)
                    if constraint == "clamp":
                        interaction.clamp_(min=-scaling, max=scaling)
                maximum = max(self.for_parameter(name).abs().max() for name in self.parameter_names)
                if constraint == "clamp" and maximum > scaling:
                    for interaction in self.interactions.values():
                        interaction.mul_(scaling / maximum)
                elif constraint == "scale" and maximum > 0:
                    for interaction in self.interactions.values():
                        interaction.mul_(scaling / maximum)
                return

            for name in self.parameter_names:
                parameter = self.values[name]
                parameter.copy_((parameter + parameter.T) / 2)
                effective = self.for_parameter(name)
                if center:
                    parameter.sub_(effective.mean())
                    effective = self.for_parameter(name)
                if constraint == "clamp":
                    parameter.clamp_(min=-scaling, max=scaling)
                elif constraint == "scale":
                    maximum = effective.abs().max()
                    if maximum > 0:
                        parameter.mul_(scaling / maximum)

    def symmetrize_gradients_(self) -> None:
        """Restrict affinity gradients to symmetric matrix directions."""

        if self.parameterization == "factorized":
            for interaction in self.interactions.values():
                if interaction.grad is not None:
                    interaction.grad.copy_((interaction.grad + interaction.grad.T) / 2)
            return
        for name in self.parameter_names:
            gradient = self.values[name].grad
            if gradient is not None:
                gradient.copy_((gradient + gradient.T) / 2)

    def get_extra_state(self):
        return {
            "sample_names": self.sample_names,
            "groups": self.groups,
            "mode": self.mode,
            "parameterization": self.parameterization,
            "rank": self.rank,
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
        if state.get("parameterization", "full") != self.parameterization or state.get("rank", self.K) != self.rank:
            raise RuntimeError(
                f"{self.__class__.__name__} checkpoint parameterization is incompatible with the model.",
            )
