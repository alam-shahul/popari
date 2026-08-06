import anndata as ad
import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from loguru import logger
from scipy.sparse import csr_array
from torch import nn

from popari._binning_utils import GridDownsampler, PartitionDownsampler
from popari._named_state import BufferDict
from popari._sample_axis import SampleAxis
from popari._sparse import convert_numpy_to_pytorch_sparse_coo
from popari.initialization import (
    initialize_dummy,
    initialize_ground_truth,
    initialize_kmeans,
    initialize_leiden,
    initialize_svd,
)
from popari.optim.simplex_integral import integrate_of_exponential_over_simplex
from popari.optim.spatial_affinity import (
    SpatialAffinityState,
    _spatial_affinity_loss,
    compute_empirical_spatial_affinities,
)
from popari.preprocessing import compute_spatial_neighbors
from popari.schema import BIN_ASSIGNMENTS_KEY, DATASET_NAME_KEY, SAMPLE_KEY_KEY, SCHEMA_VERSION, SCHEMA_VERSION_KEY


class HierarchicalLevel(nn.Module):
    """View of SRT multisample dataset at a set resolution.

    Includes the scaled (i.e. binned) data and learnable Popari state.

    """

    def __init__(
        self,
        adata: AnnData,
        sample_key: str,
        betas: list,
        prior_x_modes: list,
        method: str,
        random_state: int,
        K: int,
        context: dict,
        initial_context: dict,
        use_inplace_ops: bool,
        pretrained: bool,
        verbose: str,
        spatial_affinity_groups: dict,
        lambda_Sigma_x_inv: float,
        lambda_Sigma_bar: float,
        spatial_affinity_lr: float,
        spatial_affinity_tol: float,
        spatial_affinity_constraint: str | None,
        spatial_affinity_centering: bool,
        spatial_affinity_scaling: int,
        spatial_affinity_regularization_power: int,
        M_constraint: str,
        sigma_yx_inv_mode: str,
        spatial_affinity_mode: str,
        embedding_step_size_multiplier: float,
        embedding_mini_iterations: int,
        embedding_acceleration_trick: bool,
        binned_Ys: list = None,
        level: int = 0,
        hierarchical_levels: int | None = 1,
    ):
        super().__init__()

        self.adata = adata
        self.sample_key = sample_key
        self.context = context
        self.initial_context = initial_context
        self.sample_axis = SampleAxis.from_anndata(adata, sample_key=sample_key)
        adata.popari.validate_spatial_graph()
        self.adjacency = adata.obsp["adjacency_matrix"]
        self.register_buffer(
            "adjacency_matrix",
            convert_numpy_to_pytorch_sparse_coo(self.adjacency, self.context),
        )
        self.replicate_names = list(self.sample_axis.names)
        self.sample_names = self.sample_axis.names
        self.K = K
        self.level = level
        self.hierarchical_levels = hierarchical_levels
        self.level_suffix = "" if self.level == 0 else f"_level_{self.level}"
        self.use_inplace_ops = use_inplace_ops
        self.verbose = verbose
        self.random_state = random_state
        self.pretrained = pretrained
        self.lambda_Sigma_x_inv = lambda_Sigma_x_inv
        self.lambda_Sigma_bar = lambda_Sigma_bar
        self.spatial_affinity_lr = spatial_affinity_lr
        self.spatial_affinity_tol = spatial_affinity_tol
        self.spatial_affinity_constraint = spatial_affinity_constraint
        self.spatial_affinity_centering = spatial_affinity_centering
        self.spatial_affinity_scaling = spatial_affinity_scaling
        self.spatial_affinity_regularization_power = spatial_affinity_regularization_power
        self.M_constraint = M_constraint
        self.sigma_yx_inv_mode = sigma_yx_inv_mode
        self.spatial_affinity_mode = spatial_affinity_mode
        self.embedding_step_size_multiplier = embedding_step_size_multiplier
        self.embedding_mini_iterations = embedding_mini_iterations
        self.embedding_acceleration_trick = embedding_acceleration_trick

        self.num_replicates = len(self.sample_axis)
        self._dirty = True

        def fill_groups(groups, are_exclusive=False):
            if not groups:
                groups = {"_default": self.replicate_names}
            else:
                groups = {str(group): list(group_samples) for group, group_samples in groups.items()}

            included_replicate_names = sum(groups.values(), [])
            difference = set(self.replicate_names) - set(included_replicate_names)
            if difference:
                groups[f"_default"] = list(difference)

            # Make group names unique to hierarchical level
            groups = {
                key if key.endswith(self.level_suffix) else f"{key}{self.level_suffix}": value
                for key, value in groups.items()
            }

            tags = {replicate_name: [] for replicate_name in self.replicate_names}
            for group, group_replicates in groups.items():
                for replicate in group_replicates:
                    if are_exclusive and len(tags[replicate]) > 0:
                        raise ValueError("If in shared mode, each replicate can only appear in one group.")
                    tags[replicate].append(group)

            return groups

        if spatial_affinity_groups == "disjoint":
            spatial_affinity_groups = {replicate_name: [replicate_name] for replicate_name in self.replicate_names}

        normalized_spatial_affinity_groups = fill_groups(
            spatial_affinity_groups,
            are_exclusive=(self.spatial_affinity_mode == "shared lookup"),
        )

        if binned_Ys is None:
            self.Ys = []
            for sample in self.replicate_names:
                indices = self.sample_axis.indices(sample)
                expression = self.adata.X[indices]
                num_cells = len(indices)
                Y = convert_numpy_to_pytorch_sparse_coo(expression, self.context)
                Y *= (self.K * 1) / (Y.sum() / num_cells)
                self.Ys.append(Y)
        else:
            self.Ys = binned_Ys

        if betas is None:
            betas_tensor = torch.full((self.num_replicates,), 1 / self.num_replicates, **self.context)
        else:
            betas_tensor = torch.as_tensor(betas, **self.context)
            betas_tensor = betas_tensor / betas_tensor.sum()
        self.register_buffer("betas", betas_tensor)

        if prior_x_modes is None:
            prior_x_modes = [None] * self.num_replicates

        self.prior_x_modes = prior_x_modes

        if self.verbose >= 1:
            logger.info("Initializing level-owned state")
        self.sample_indices = BufferDict(prefix="sample_indices")
        for sample in self.sample_names:
            self.sample_indices[sample] = torch.tensor(
                self.sample_axis.indices(sample),
                dtype=torch.long,
                device=self.context["device"],
            )
        self.embeddings = nn.Parameter(
            torch.zeros((self.adata.n_obs, self.K), **self.context),
            requires_grad=False,
        )
        self.metagenes = nn.Parameter(torch.zeros((self.adata.n_vars, self.K), **self.context))
        self.register_buffer("sigma_yxs", torch.zeros(self.num_replicates, **self.context))

        if all(mode == "exponential shared fixed" for mode in self.prior_x_modes):
            self.prior_xs = [(torch.ones(self.K, **self.initial_context),) for _ in self.replicate_names]
        elif all(mode is None for mode in self.prior_x_modes):
            self.prior_xs = [(torch.zeros(self.K, **self.initial_context),) for _ in self.replicate_names]
        else:
            raise NotImplementedError

        self.spatial_affinity = SpatialAffinityState(
            self.K,
            self.replicate_names,
            normalized_spatial_affinity_groups,
            mode=self.spatial_affinity_mode,
            context=self.context,
        )

        if self.pretrained:
            self.embeddings.copy_(
                torch.from_numpy(self.adata.obsm["X"]).to(**self.context),
            )
            with torch.no_grad():
                self.metagenes.copy_(
                    torch.from_numpy(self.adata.uns["M"]).to(**self.context),
                )
            for sample in self.replicate_names:
                self.spatial_affinity.set_sample_(
                    sample,
                    torch.from_numpy(self.adata.uns["Sigma_x_inv"][sample]).to(**self.context),
                )

            self._recompute_observation_noise()
        else:
            if self.level < self.hierarchical_levels - 1:
                method = "dummy"

            if self.verbose >= 1:
                logger.info("Initializing metagenes and embeddings using {}", method)

            if method == "dummy":
                initial_metagenes, initial_embeddings = initialize_dummy(
                    self.adata,
                    self.sample_axis,
                    self.K,
                    self.initial_context,
                )
            elif method == "kmeans":
                initial_metagenes, initial_embeddings = initialize_kmeans(
                    self.adata,
                    self.sample_axis,
                    self.K,
                    self.initial_context,
                    kwargs_kmeans=dict(random_state=self.random_state),
                )
            elif method == "svd":
                initial_metagenes, initial_embeddings = initialize_svd(
                    self.adata,
                    self.sample_axis,
                    self.K,
                    self.initial_context,
                    M_nonneg=(self.M_constraint == "simplex"),
                    X_nonneg=True,
                )
            elif method in {"leiden", "leiden_fast"}:
                kwargs_leiden = {
                    "random_state": self.random_state,
                }
                if method == "leiden_fast":
                    kwargs_leiden.update(flavor="igraph", n_iterations=2)
                initial_metagenes, initial_embeddings = initialize_leiden(
                    self.adata,
                    self.sample_axis,
                    self.K,
                    self.initial_context,
                    kwargs_leiden=kwargs_leiden,
                    verbose=self.verbose,
                )
            elif method == "ground_truth":
                initial_metagenes, initial_embeddings = initialize_ground_truth(
                    self.adata,
                    self.sample_axis,
                    self.K,
                    self.initial_context,
                    random_state=self.random_state,
                )
            else:
                raise NotImplementedError

            with torch.no_grad():
                self.metagenes.copy_(initial_metagenes.to(**self.context))
                self.embeddings.copy_(initial_embeddings.to(**self.context))

            self._normalize_factorization()

            self._recompute_observation_noise()

            # Initializing spatial affinities
            if self.verbose >= 1:
                logger.info("Initializing spatial affinities with empirical correlations")
            self._initialize_spatial_affinities()

        self.adata.uns["losses"] = {key: list(values) for key, values in self.adata.uns.get("losses", {}).items()}

    @property
    def spatial_affinity_groups(self):
        """Return the runtime spatial-affinity groups."""

        return self.spatial_affinity.groups

    @property
    def spatial_affinity_tags(self):
        """Return spatial-affinity group memberships by sample."""

        return self.spatial_affinity.tags

    def embedding(self, sample: str) -> torch.Tensor:
        """Return embeddings for one sample from the global embedding tensor."""

        return torch.index_select(self.embeddings, 0, self.sample_indices[sample])

    def set_embedding(self, sample: str, value: torch.Tensor) -> None:
        """Write one sample's embeddings into the global embedding tensor."""

        with torch.no_grad():
            self.embeddings.index_copy_(0, self.sample_indices[sample], value.to(self.embeddings.device))

    def _normalize_factorization(self) -> None:
        """Normalize metagenes while preserving their reconstruction."""

        if self.M_constraint == "simplex":
            scale_factor = torch.linalg.norm(self.metagenes, axis=0, ord=1, keepdim=True)
        elif self.M_constraint == "unit_sphere":
            scale_factor = torch.linalg.norm(self.metagenes, axis=0, ord=2, keepdim=True)
        else:
            raise NotImplementedError(f"Unsupported metagene constraint: {self.M_constraint!r}")

        with torch.no_grad():
            self.metagenes.div_(scale_factor)
            self.embeddings.mul_(scale_factor)

    def _recompute_observation_noise(self) -> None:
        """Recompute sample observation noise from the current factorization."""

        squared_terms = [
            torch.addmm(
                expression.to_dense(),
                self.embedding(sample),
                self.metagenes.T,
                alpha=-1,
            )
            for expression, sample in zip(self.Ys, self.sample_names)
        ]
        squared_loss = torch.as_tensor(
            [torch.linalg.norm(term, ord="fro").item() ** 2 for term in squared_terms],
            **self.context,
        )
        sizes = torch.as_tensor([expression.numel() for expression in self.Ys], **self.context)
        if self.sigma_yx_inv_mode == "separate":
            self.sigma_yxs[:] = torch.sqrt(squared_loss / sizes)
        elif self.sigma_yx_inv_mode == "average":
            sigma_yx = torch.sqrt(torch.dot(self.betas, squared_loss) / torch.dot(self.betas, sizes))
            self.sigma_yxs[:] = sigma_yx
        else:
            raise NotImplementedError(f"Unsupported observation-noise mode: {self.sigma_yx_inv_mode!r}")

    def _initialize_spatial_affinities(self) -> None:
        """Initialize spatial affinities from empirical embedding
        correlations."""

        initial_affinities = compute_empirical_spatial_affinities(
            [self.embedding(sample).clone() for sample in self.sample_names],
            self.adjacency,
            self.sample_axis,
            self.spatial_affinity_scaling,
            self.initial_context,
        )
        self.spatial_affinity.initialize_(initial_affinities, self.betas)

    def _reload_state(self):
        """Reload Popari state using results from saved datasets.

        Used when loading a trained model from an in-memory artifact.

        """
        self.embeddings.copy_(
            torch.from_numpy(self.adata.obsm["X"]).to(**self.context),
        )
        with torch.no_grad():
            self.metagenes.copy_(
                torch.from_numpy(self.adata.uns["M"]).to(**self.context),
            )
        for sample in self.replicate_names:
            self.spatial_affinity.set_sample_(
                sample,
                torch.from_numpy(self.adata.uns["Sigma_x_inv"][sample]).to(**self.context),
            )

        self._recompute_observation_noise()

    def materialize_results(self, *, force: bool = False) -> AnnData:
        """Write authoritative tensor state into this level's AnnData."""

        if not self._dirty and not force:
            return self.adata

        self.adata.uns["M"] = self.metagenes.cpu().detach().numpy()
        self.adata.obsm["X"] = self.embeddings.cpu().detach().numpy()
        self.adata.uns["sigma_yx"] = {
            sample: self.sigma_yxs[index].item() for index, sample in enumerate(self.replicate_names)
        }
        self.adata.uns["Sigma_x_inv"] = {
            sample: self.spatial_affinity.for_sample(sample).cpu().detach().numpy() for sample in self.replicate_names
        }

        if self.spatial_affinity_mode == "differential lookup":
            self.adata.uns["spatial_affinity_bar"] = {
                group_name: group_mean.cpu().numpy()
                for group_name, group_mean in self.spatial_affinity.group_means().items()
            }

        self._dirty = False
        return self.adata

    def mark_dirty(self) -> None:
        """Record that learned tensor state is newer than materialized
        AnnData."""

        self._dirty = True

    def forward(self, use_spatial: bool = False):
        """Compute the joint negative log pseudolikelihood for this level.

        Samples contribute once each. ``betas`` affect optimization weighting,
        but do not temper the reported joint probability.

        """

        total_loss = self.metagenes.new_zeros(())
        metagenes = self.metagenes.to(self.context["device"])

        # Expression likelihood and optional embedding-magnitude prior.
        for sample_index, sample in enumerate(self.sample_names):
            sigma_yx = self.sigma_yxs[sample_index]
            expression = self.Ys[sample_index].to(self.context["device"])
            embedding = self.embedding(sample).to(self.context["device"])
            num_observations, num_genes = expression.shape

            quadratic_factor = metagenes.T @ metagenes / sigma_yx.square()
            linear_factor = expression @ metagenes / sigma_yx.square()
            expression_norm = expression.square().sum() / sigma_yx.square()
            total_loss += ((embedding @ quadratic_factor) * embedding).sum() / 2
            total_loss -= (embedding * linear_factor).sum()
            total_loss += expression_norm / 2
            total_loss += num_observations * num_genes / 2 * torch.log(2 * np.pi * sigma_yx.square())

            prior_x_mode = self.prior_x_modes[sample_index]
            if prior_x_mode == "exponential shared fixed":
                rate = self.prior_xs[sample_index][0][0]
                magnitude = torch.linalg.norm(embedding, dim=1, ord=1)
                total_loss += rate * magnitude.sum() - num_observations * self.K * torch.log(rate)
                if use_spatial:
                    total_loss += num_observations * torch.lgamma(rate.new_tensor(float(self.K)))
            elif prior_x_mode is not None:
                raise NotImplementedError(f"Unsupported embedding prior: {prior_x_mode!r}")

        if not use_spatial:
            return total_loss

        # Spatial conditional likelihood and affinity priors. Each unique
        # affinity is regularized once using the edges governed by it.
        normalized = self.embeddings / torch.linalg.norm(self.embeddings, dim=1, ord=1, keepdim=True)
        neighbor_sums = self.adjacency_matrix @ normalized
        edge_counts = np.diff(self.adjacency.indptr)
        group_means = (
            self.spatial_affinity.group_means() if self.spatial_affinity_mode == "differential lookup" else None
        )

        if self.spatial_affinity_mode == "shared lookup":
            parameter_samples = self.spatial_affinity_groups.items()
        elif self.spatial_affinity_mode == "differential lookup":
            parameter_samples = ((sample, [sample]) for sample in self.sample_names)
        else:
            raise NotImplementedError(f"Unsupported spatial-affinity mode: {self.spatial_affinity_mode!r}")

        for parameter_name, samples in parameter_samples:
            affinity = self.spatial_affinity.values[parameter_name]
            linear_factor = torch.zeros_like(affinity)
            sample_neighbor_sums = []
            edge_count = 0
            for sample in samples:
                indices = self.sample_indices[sample]
                sample_normalized = normalized.index_select(0, indices)
                sample_neighbors = neighbor_sums.index_select(0, indices)
                linear_factor.addmm_(sample_normalized.T, sample_neighbors)
                sample_neighbor_sums.append(sample_neighbors)
                edge_count += int(edge_counts[self.sample_axis.indices(sample)].sum())

            affinity_group_means = None
            if group_means is not None:
                affinity_group_means = [group_means[group] for group in self.spatial_affinity_tags[parameter_name]]

            if edge_count:
                sample_weights = affinity.new_ones(len(samples))
                total_loss += (
                    _spatial_affinity_loss(
                        affinity,
                        linear_factor,
                        sample_neighbor_sums,
                        sample_weights,
                        affinity.new_tensor(float(edge_count)),
                        regularization_strength=self.lambda_Sigma_x_inv,
                        regularization_power=self.spatial_affinity_regularization_power,
                        group_means=affinity_group_means,
                        group_regularization_strength=self.lambda_Sigma_bar,
                    )
                    * edge_count
                )
            else:
                total_loss += sum(
                    integrate_of_exponential_over_simplex(sample_neighbors @ affinity).sum()
                    for sample_neighbors in sample_neighbor_sums
                )

        return total_loss


class Hierarchy(nn.ModuleList):
    """Container for hierarchical views of Popari."""

    def __init__(
        self,
        base_view: "HierarchicalLevel",
        downsampling_method: str = "grid",
        **hierarchical_view_kwargs,
    ):
        super().__init__([base_view])
        if downsampling_method == "grid":
            self.downsampler = GridDownsampler()
        elif downsampling_method == "partition":
            self.downsampler = PartitionDownsampler()

        self.hierarchical_view_kwargs = hierarchical_view_kwargs

    def construct(self, levels: int, downsample_rate: float, **kwargs):
        base_view = self[0]
        context = base_view.context
        previous_view = base_view
        for level in range(1, levels):
            if base_view.verbose >= 1:
                logger.info("Initializing hierarchy level {}", level)
            binned_datasets = []
            binned_Ys = []
            local_assignments = []
            previous_Ys = previous_view.Ys

            effective_kwargs = kwargs.copy()
            for sample, previous_Y in zip(previous_view.replicate_names, previous_Ys):
                previous_dataset = previous_view.adata[previous_view.sample_axis.indices(sample)].copy()
                previous_dataset.obs[previous_view.sample_key] = previous_dataset.obs[
                    previous_view.sample_key
                ].cat.remove_unused_categories()
                previous_dataset.popari.name = sample
                previous_dataset.obsp["adjacency_matrix"] = csr_array(
                    previous_dataset.obsp["adjacency_matrix"],
                )
                bin_assignments_key = f"{BIN_ASSIGNMENTS_KEY}_{sample}"
                binned_dataset, effective_kwargs = self.downsampler.downsample(
                    previous_dataset,
                    downsample_rate=downsample_rate,
                    bin_assignments_key=bin_assignments_key,
                    **effective_kwargs,
                )
                binned_dataset.popari.name = sample
                binned_dataset.obs[previous_view.sample_key] = sample
                binned_dataset.obs_names = [f"{sample}_level_{level}_{index}" for index in range(binned_dataset.n_obs)]

                if base_view.verbose >= 1:
                    logger.info(
                        "Downsampled {} from {} to {} observations",
                        sample,
                        len(previous_dataset),
                        len(binned_dataset),
                    )

                binned_datasets.append(binned_dataset)
                assignments = csr_array(binned_dataset.obsm.pop(bin_assignments_key))
                local_assignments.append(assignments)
                bin_assignments = convert_numpy_to_pytorch_sparse_coo(
                    assignments.tocoo(),
                    context=context,
                )

                binned_Y = bin_assignments @ previous_Y
                binned_Ys.append(binned_Y)

            binned_adata = ad.concat(
                binned_datasets,
                join="inner",
                merge="same",
                uns_merge="same",
            )
            binned_adata.obs[previous_view.sample_key] = pd.Categorical(
                binned_adata.obs[previous_view.sample_key].astype(str),
                categories=previous_view.replicate_names,
                ordered=True,
            )
            binned_adata.uns[DATASET_NAME_KEY] = "multisample"
            binned_adata.uns[SAMPLE_KEY_KEY] = previous_view.sample_key
            binned_adata.uns[SCHEMA_VERSION_KEY] = SCHEMA_VERSION

            row_parts = []
            column_parts = []
            data_parts = []
            row_offset = 0
            for sample, assignments in zip(previous_view.replicate_names, local_assignments, strict=True):
                local = assignments.tocoo()
                fine_indices = previous_view.sample_axis.indices(sample)
                row_parts.append(row_offset + local.row)
                column_parts.append(fine_indices[local.col])
                data_parts.append(local.data)
                row_offset += assignments.shape[0]
            binned_adata.obsm[BIN_ASSIGNMENTS_KEY] = csr_array(
                (
                    np.concatenate(data_parts),
                    (np.concatenate(row_parts), np.concatenate(column_parts)),
                ),
                shape=(binned_adata.n_obs, previous_view.adata.n_obs),
            )
            compute_spatial_neighbors(binned_adata, sample_key=previous_view.sample_key)
            level_view = HierarchicalLevel(
                binned_adata,
                level=level,
                binned_Ys=binned_Ys,
                **self.hierarchical_view_kwargs,
            )
            self.append(level_view)

            previous_view = level_view

    @classmethod
    def reconstruct(cls, reloaded_hierarchy: dict, **hierarchical_view_kwargs):
        """Reconstruct a hierarchy from one unified AnnData per level."""

        context = hierarchical_view_kwargs["context"]

        def reconstruct_level(
            level: int,
            adata: AnnData,
            previous_view: "HierarchicalLevel | None",
        ):
            if hierarchical_view_kwargs["verbose"] >= 1:
                logger.info("Reloading hierarchy level {}", level)
            if previous_view is not None:
                binned_Ys = []
                assignments = csr_array(adata.obsm[BIN_ASSIGNMENTS_KEY])
                sample_axis = SampleAxis.from_anndata(
                    adata,
                    sample_key=hierarchical_view_kwargs["sample_key"],
                )
                for sample, previous_Y in zip(sample_axis.names, previous_view.Ys):
                    local_assignments = assignments[sample_axis.indices(sample)][
                        :,
                        previous_view.sample_axis.indices(sample),
                    ]
                    binned_Y = (
                        convert_numpy_to_pytorch_sparse_coo(
                            local_assignments,
                            context=context,
                        )
                        @ previous_Y
                    )
                    binned_Ys.append(binned_Y)
            else:
                binned_Ys = None

            level_view = HierarchicalLevel(
                adata,
                level=level,
                binned_Ys=binned_Ys,
                **hierarchical_view_kwargs,
            )

            return level_view

        base_view = reconstruct_level(0, reloaded_hierarchy[0], None)

        hierarchy = cls(base_view=base_view, **hierarchical_view_kwargs)
        previous_view = base_view

        for level in range(1, hierarchical_view_kwargs["hierarchical_levels"]):
            adata = reloaded_hierarchy[level]
            level_view = reconstruct_level(level, adata, previous_view)
            hierarchy.append(level_view)
            previous_view = level_view

        return hierarchy
