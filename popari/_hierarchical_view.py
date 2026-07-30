from collections import defaultdict

import numpy as np
import torch
from anndata import AnnData
from scipy.sparse import csr_array
from torch import nn
from tqdm.auto import trange

from popari._binning_utils import GridDownsampler, PartitionDownsampler
from popari._embedding_optimizer import EmbeddingOptimizer
from popari._parameter_optimizer import ParameterOptimizer
from popari._sample_axis import SampleAxis
from popari.initialization import (
    initialize_dummy,
    initialize_ground_truth,
    initialize_kmeans,
    initialize_leiden,
    initialize_svd,
)
from popari.io import merge_anndata
from popari.preprocessing import compute_spatial_neighbors
from popari.sample_for_integral import integrate_of_exponential_over_simplex
from popari.schema import BIN_ASSIGNMENTS_KEY
from popari.util import convert_adjacency_matrix_to_awkward_array, convert_numpy_to_pytorch_sparse_coo, get_datetime


class HierarchicalView(nn.Module):
    """View of SRT multisample dataset at a set resolution.

    Includes the scaled (i.e. binned data) as well as the learnable Popari
    parameters and their corresponding optimizers.

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
        metagene_groups: dict,
        spatial_affinity_groups: dict,
        parameter_optimizer_hyperparameters: dict,
        embedding_optimizer_hyperparameters: dict,
        binned_Ys: list = None,
        superresolution_lr: float = 1e-3,
        level: int = 0,
        hierarchical_levels: int | None = 1,
    ):
        super().__init__()

        self.adata = adata
        self.sample_key = sample_key
        self.sample_axis = SampleAxis.from_anndata(adata, sample_key=sample_key)
        self.replicate_names = list(self.sample_axis.names)
        self.K = K
        self.level = level
        self.hierarchical_levels = hierarchical_levels
        self.level_suffix = "" if self.level == 0 else f"_level_{self.level}"
        self.context = context
        self.initial_context = initial_context
        self.use_inplace_ops = use_inplace_ops
        self.verbose = verbose
        self.random_state = random_state
        self.pretrained = pretrained

        self.num_replicates = len(self.sample_axis)
        self._legacy_datasets = None

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

            return groups, tags

        if metagene_groups == "disjoint":
            metagene_groups = {replicate_name: [replicate_name] for replicate_name in self.replicate_names}

        if spatial_affinity_groups == "disjoint":
            spatial_affinity_groups = {replicate_name: [replicate_name] for replicate_name in self.replicate_names}

        self.metagene_groups, self.metagene_tags = fill_groups(
            metagene_groups,
            are_exclusive=(parameter_optimizer_hyperparameters["metagene_mode"] == "shared"),
        )
        self.spatial_affinity_groups, self.spatial_affinity_tags = fill_groups(
            spatial_affinity_groups,
            are_exclusive=(parameter_optimizer_hyperparameters["spatial_affinity_mode"] == "shared lookup"),
        )

        parameter_optimizer_hyperparameters = parameter_optimizer_hyperparameters.copy()
        parameter_optimizer_hyperparameters["metagene_groups"] = self.metagene_groups
        parameter_optimizer_hyperparameters["metagene_tags"] = self.metagene_tags
        parameter_optimizer_hyperparameters["spatial_affinity_groups"] = self.spatial_affinity_groups
        parameter_optimizer_hyperparameters["spatial_affinity_tags"] = self.spatial_affinity_tags

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

        self.parameter_optimizer = ParameterOptimizer(
            self.K,
            self.Ys,
            self.adata,
            self.sample_axis,
            self.betas,
            prior_x_modes,
            initial_context=self.initial_context,
            context=self.context,
            use_inplace_ops=self.use_inplace_ops,
            verbose=self.verbose,
            **parameter_optimizer_hyperparameters,
        )
        self.superresolution_lr = superresolution_lr

        if self.verbose:
            print(f"{get_datetime()} Initializing EmbeddingOptimizer")
        self.embedding_optimizer = EmbeddingOptimizer(
            self.K,
            self.Ys,
            self.adata,
            self.sample_axis,
            initial_context=self.initial_context,
            context=self.context,
            use_inplace_ops=self.use_inplace_ops,
            verbose=self.verbose,
            **embedding_optimizer_hyperparameters,
        )
        self.parameter_optimizer.link(self.embedding_optimizer)
        self.embedding_optimizer.link(self.parameter_optimizer)

        if self.pretrained:
            spatial_affinity_copy = torch.zeros((self.num_replicates, self.K, self.K), **self.context)
            self.embedding_optimizer.embedding_state.embedding.copy_(
                torch.from_numpy(self.adata.obsm["X"]).to(**self.context),
            )
            for dataset_index, sample in enumerate(self.replicate_names):
                self.parameter_optimizer.metagene_state[sample] = torch.from_numpy(
                    self.adata.uns["M"][sample],
                ).to(**self.initial_context)
                self.parameter_optimizer.spatial_affinity[sample] = torch.from_numpy(
                    self.adata.uns["Sigma_x_inv"][sample],
                ).to(**self.initial_context)
                spatial_affinity_copy[dataset_index] = self.parameter_optimizer.spatial_affinity[sample]

            self.parameter_optimizer.update_sigma_yx()
            self.parameter_optimizer.spatial_affinity.initialize_optimizers(
                spatial_affinity_copy,
                self.parameter_optimizer.spatial_affinity_bar,
            )
        else:
            if self.level < self.hierarchical_levels - 1:
                method = "dummy"

            if self.verbose:
                print(f"{get_datetime()} Initializing metagenes and hidden states using {method} method")

            if method == "dummy":
                self.M, self.X = initialize_dummy(
                    self.adata,
                    self.sample_axis,
                    self.K,
                    self.initial_context,
                )
            elif method == "kmeans":
                self.M, self.X = initialize_kmeans(
                    self.adata,
                    self.sample_axis,
                    self.K,
                    self.initial_context,
                    kwargs_kmeans=dict(random_state=self.random_state),
                )
            elif method == "svd":
                self.M, self.X = initialize_svd(
                    self.adata,
                    self.sample_axis,
                    self.K,
                    self.initial_context,
                    M_nonneg=(self.parameter_optimizer.M_constraint == "simplex"),
                    X_nonneg=True,
                )
            elif method == "leiden":
                kwargs_leiden = {
                    "random_state": self.random_state,
                }
                self.M, self.X = initialize_leiden(
                    self.adata,
                    self.sample_axis,
                    self.K,
                    self.initial_context,
                    kwargs_leiden=kwargs_leiden,
                    verbose=self.verbose,
                )
            elif method == "ground_truth":
                self.M, self.X = initialize_ground_truth(
                    self.adata,
                    self.sample_axis,
                    self.K,
                    self.initial_context,
                    random_state=self.random_state,
                )
            else:
                raise NotImplementedError

            for sample in self.replicate_names:
                self.parameter_optimizer.metagene_state[sample] = self.M
            self.embedding_optimizer.embedding_state.embedding.copy_(self.X.to(**self.context))

            self.parameter_optimizer.scale_metagenes()

            self.Sigma_x_inv_bar = None

            self.parameter_optimizer.update_sigma_yx()

            initial_embeddings = [self.embedding_optimizer.embedding_state[sample] for sample in self.replicate_names]

            # Initializing spatial affinities
            if self.verbose:
                print(f"{get_datetime()} Initializing Sigma_x_inv with empirical correlations")
            self.parameter_optimizer.spatial_affinity.initialize(
                initial_embeddings,
                self.parameter_optimizer.spatial_affinity_bar,
            )

            self.adata.uns["M"] = {
                sample: self.parameter_optimizer.metagene_state[sample].cpu().detach().numpy()
                for sample in self.replicate_names
            }
            self.adata.obsm["X"] = self.embedding_optimizer.embedding_state.embedding.cpu().detach().numpy()
            self.adata.uns["Sigma_x_inv"] = {
                sample: self.parameter_optimizer.spatial_affinity[sample].cpu().detach().numpy()
                for sample in self.replicate_names
            }
            self.adata.uns["popari_hyperparameters"] = {
                "prior_x": {
                    sample: self.parameter_optimizer.prior_xs[index][0].cpu().detach().numpy()
                    for index, sample in enumerate(self.replicate_names)
                },
                "K": self.K,
                "use_inplace_ops": self.use_inplace_ops,
                "random_state": self.random_state,
                "verbose": self.verbose,
                **parameter_optimizer_hyperparameters,
                **embedding_optimizer_hyperparameters,
            }

            if self.parameter_optimizer.metagene_mode == "differential":
                M_bar = {
                    group_name: self.parameter_optimizer.metagene_state.M_bar[group_name].cpu().detach().numpy()
                    for group_name in self.parameter_optimizer.metagene_groups
                }
                self.adata.uns["M_bar"] = M_bar

            if self.parameter_optimizer.spatial_affinity_mode == "differential lookup":
                spatial_affinity_bar = {
                    group_name: self.parameter_optimizer.spatial_affinity_bar[group_name].cpu().detach().numpy()
                    for group_name in self.parameter_optimizer.spatial_affinity_groups
                }
                self.adata.uns["spatial_affinity_bar"] = spatial_affinity_bar

        self.superresolution_optimizers = {}
        if "losses" not in self.adata.uns:
            self.adata.uns["losses"] = defaultdict(list)
        else:
            self.adata.uns["losses"] = defaultdict(list, self.adata.uns["losses"])
            for key in self.adata.uns["losses"]:
                self.adata.uns["losses"][key] = list(self.adata.uns["losses"][key])

        if self.parameter_optimizer.metagene_mode == "differential":
            self.parameter_optimizer.metagene_state.reaverage()
        if self.parameter_optimizer.spatial_affinity_mode == "differential lookup":
            self.parameter_optimizer.spatial_affinity.reaverage(self.parameter_optimizer.spatial_affinity_bar)

    def sample_adata(self, sample: str) -> AnnData:
        """Return a standalone copy of one sample for sample-local
        algorithms."""

        dataset = self.adata[self.sample_axis.indices(sample)].copy()
        dataset.obs[self.sample_key] = dataset.obs[self.sample_key].cat.remove_unused_categories()
        dataset.popari.name = sample
        adjacency = csr_array(dataset.obsp["adjacency_matrix"])
        dataset.obsp["adjacency_matrix"] = adjacency
        dataset.obsm["adjacency_list"] = convert_adjacency_matrix_to_awkward_array(
            adjacency.tocoo(),
        )
        return dataset

    @property
    def datasets(self) -> list[AnnData]:
        """Temporary compatibility view of the unified level AnnData."""

        if self._legacy_datasets is None:
            self._legacy_datasets = [self.sample_adata(sample) for sample in self.replicate_names]
        return self._legacy_datasets

    def link(self, low_res_view: "HierarchicalView"):
        """Link a view to the resolution right below it in the hierarchy."""
        self.low_res_view = low_res_view

    def _propagate_parameters(self):
        """Use parameters from low-resolution to initialize higher-
        resolution."""
        low_res_metagenes = self.low_res_view.parameter_optimizer.metagene_state.metagenes
        self.parameter_optimizer.metagene_state.copy_from_tensor(low_res_metagenes)

        self.synchronize_datasets()

    def _superresolve_embeddings(
        self,
        n_epochs=10000,
        tol=1e-4,
        update_alg="gd",
        use_manual_gradients=True,
        verbose=None,
    ):
        """Superresolve embeddings using embeddings for lower resolution
        spots."""

        final_losses = np.zeros(self.num_replicates)
        global_assignments = csr_array(self.low_res_view.adata.obsm[BIN_ASSIGNMENTS_KEY])
        for dataset_index, sample in enumerate(self.replicate_names):
            sigma_yx = self.parameter_optimizer.sigma_yxs[dataset_index]
            Y = self.Ys[dataset_index].to(self.context["device"])

            if Y.sum() == 0:
                raise ValueError(
                    "It seems like you are trying to superresolve a hierarchical level with all zero expression "
                    "values. This probably means the model was saved incorrectly; try using `model.save_results` "
                    "with `as_trainable=True` next time.",
                )

            X = self.embedding_optimizer.embedding_state[sample].to(self.context["device"])
            X_B = self.low_res_view.embedding_optimizer.embedding_state[sample].cpu().detach().numpy()
            B = global_assignments[self.low_res_view.sample_axis.indices(sample)][:, self.sample_axis.indices(sample)]

            M = self.parameter_optimizer.metagene_state[sample].to(self.context["device"])
            prior_x_mode = self.parameter_optimizer.prior_x_modes[dataset_index]
            prior_x = self.parameter_optimizer.prior_xs[dataset_index]

            # Precomputing quantities
            MTM = (M.T @ M / (sigma_yx**2)).detach()
            BTB = convert_numpy_to_pytorch_sparse_coo((B.T @ B).tocoo(), context=self.context)
            YM = (Y @ M / (sigma_yx**2)).detach()
            BTX_B = torch.from_numpy(B.T @ X_B).to(self.context["device"]).to(self.context["dtype"])

            linear_term_gradient = YM + BTX_B
            if prior_x_mode == "exponential shared fixed":
                linear_term_gradient = linear_term_gradient - prior_x[0][None]
            linear_term_gradient = linear_term_gradient.detach()

            Ynorm = (torch.square(Y).sum() / (sigma_yx**2)).detach()
            X_Bnorm = np.linalg.norm(X_B, ord="fro").item() ** 2
            loss_prev, loss = np.inf, np.nan

            X = X.clone().detach().requires_grad_(True)

            if verbose is None:
                verbose = self.verbose

            superresolution_optimizer = torch.optim.Adam(
                [X],
                lr=self.superresolution_lr,
                betas=(0.5, 0.9),
            )

            def gradient_update(X, iteration=None):
                """Perform one constrained gradient update."""

                superresolution_optimizer.zero_grad()
                quadratic_term_gradient = X @ MTM + BTB @ X

                loss = (
                    (quadratic_term_gradient * X).sum() / 2 - (linear_term_gradient * X).sum() + Ynorm / 2 + X_Bnorm / 2
                )
                if use_manual_gradients:
                    gradient = quadratic_term_gradient - linear_term_gradient
                    X.grad = gradient
                else:
                    loss.backward()

                superresolution_optimizer.step()
                with torch.no_grad():
                    X.clamp_(min=1e-10)

                return loss.detach().item()

            progress_bar = trange(n_epochs, leave=True, disable=(verbose < 5), miniters=10000)
            for epoch in progress_bar:
                X_prev = X.clone().detach()
                if update_alg == "mu":
                    pass
                elif update_alg == "gd":
                    loss = gradient_update(X, iteration=epoch)

                dX = torch.abs((X_prev - X) / torch.linalg.norm(X, dim=1, ord=1, keepdim=True)).max().item()

                do_stop = dX < tol
                description = (
                    f"Updating weights hierarchically: loss = {loss:.1e} "
                    f"%δloss = {(loss_prev - loss) / loss:.1e} "
                    f"%δX = {dX:.1e}"
                )

                progress_bar.set_description(description)

                loss_prev = loss
                if do_stop:
                    break

            progress_bar.close()
            self.embedding_optimizer.embedding_state[sample] = X.clone().detach()

            final_losses[dataset_index] = loss

            del superresolution_optimizer
            del sigma_yx
            del Y
            del X
            del X_B
            del B
            del M
            del MTM
            del BTB
            del YM
            del BTX_B

        return final_losses

    def _reload_state(self):
        """Reload Popari state using results from saved datasets.

        Opposite of `synchronize_datasets`. Should be used rarely, e.g. when loading a trained
        model from memory.

        """
        self.embedding_optimizer.embedding_state.embedding.copy_(
            torch.from_numpy(self.adata.obsm["X"]).to(**self.context),
        )
        for sample in self.replicate_names:
            self.parameter_optimizer.metagene_state[sample] = torch.from_numpy(
                self.adata.uns["M"][sample],
            ).to(**self.initial_context)

            with torch.no_grad():
                self.parameter_optimizer.spatial_affinity[sample] = torch.from_numpy(
                    self.adata.uns["Sigma_x_inv"][sample],
                ).to(**self.initial_context)

        if self.parameter_optimizer.metagene_mode == "differential":
            self.parameter_optimizer.metagene_state.reaverage()

        if self.parameter_optimizer.spatial_affinity_mode == "differential lookup":
            self.parameter_optimizer.spatial_affinity.reaverage(self.parameter_optimizer.spatial_affinity_bar)

        self.parameter_optimizer.update_sigma_yx()

    def synchronize_datasets(self):
        """Synchronize learned state into the unified level AnnData."""

        self.adata.uns["M"] = {
            sample: self.parameter_optimizer.metagene_state[sample].cpu().detach().numpy()
            for sample in self.replicate_names
        }
        self.adata.obsm["X"] = self.embedding_optimizer.embedding_state.embedding.cpu().detach().numpy()
        self.adata.uns["sigma_yx"] = {
            sample: self.parameter_optimizer.sigma_yxs[index].item()
            for index, sample in enumerate(self.replicate_names)
        }
        self.adata.uns["Sigma_x_inv"] = {
            sample: self.parameter_optimizer.spatial_affinity[sample].cpu().detach().numpy()
            for sample in self.replicate_names
        }

        if self.parameter_optimizer.metagene_mode == "differential":
            M_bar = {
                group_name: self.parameter_optimizer.metagene_state.M_bar[group_name].cpu().detach().numpy()
                for group_name in self.parameter_optimizer.metagene_groups
            }
            self.adata.uns["M_bar"] = M_bar

        if self.parameter_optimizer.spatial_affinity_mode == "differential lookup":
            spatial_affinity_bar = {
                group_name: self.parameter_optimizer.spatial_affinity_bar[group_name].cpu().detach().numpy()
                for group_name in self.parameter_optimizer.spatial_affinity_groups
            }
            self.adata.uns["spatial_affinity_bar"] = spatial_affinity_bar

        if self._legacy_datasets is not None:
            for dataset, sample in zip(self._legacy_datasets, self.replicate_names):
                indices = self.sample_axis.indices(sample)
                dataset.obsm["X"] = self.adata.obsm["X"][indices].copy()
                dataset.uns["M"] = {sample: self.adata.uns["M"][sample]}
                dataset.uns["Sigma_x_inv"] = {
                    sample: self.adata.uns["Sigma_x_inv"][sample],
                }
                dataset.uns["sigma_yx"] = self.adata.uns["sigma_yx"][sample]
                if "M_bar" in self.adata.uns:
                    dataset.uns["M_bar"] = self.adata.uns["M_bar"]
                if "spatial_affinity_bar" in self.adata.uns:
                    dataset.uns["spatial_affinity_bar"] = self.adata.uns["spatial_affinity_bar"]

    def forward(self, use_spatial: bool = False):
        """Compute overall negative log-likelihood for the current model
        parameters."""

        total_loss = torch.zeros((), **self.context)
        if use_spatial:
            weighted_total_cells = 0
            for sample in self.replicate_names:
                E_adjacency_list = self.embedding_optimizer.adjacency_lists[sample]
                weighted_total_cells += sum(map(len, E_adjacency_list))

        for dataset_index, sample in enumerate(self.replicate_names):
            sigma_yx = self.parameter_optimizer.sigma_yxs[dataset_index]
            Y = self.Ys[dataset_index].to(self.context["device"])
            X = self.embedding_optimizer.embedding_state[sample].to(self.context["device"])
            M = self.parameter_optimizer.metagene_state[sample].to(self.context["device"])
            prior_x_mode = self.parameter_optimizer.prior_x_modes[dataset_index]
            prior_x = self.parameter_optimizer.prior_xs[dataset_index]

            MTM = M.T @ M / (sigma_yx**2)
            YM = Y.to(M.device) @ M / (sigma_yx**2)
            Ynorm = torch.square(Y).sum() / (sigma_yx**2)
            S = torch.linalg.norm(X, dim=1, ord=1, keepdim=True)

            Z = X / S
            N, G = Y.shape

            loss = ((X @ MTM) * X).sum() / 2 - (X * YM).sum() + Ynorm / 2

            logZ_i_Y = torch.ones((N,), **self.context) * (G / 2 * torch.log(2 * np.pi * sigma_yx**2))
            if not use_spatial:
                logZ_i_X = torch.full((N,), 0, **self.context)
                if (prior_x[0] != 0).all():
                    logZ_i_X += torch.full((N,), self.K * torch.log(prior_x[0]).item(), **self.context)
                log_partition_function = (logZ_i_Y + logZ_i_X).sum()
            else:
                adjacency_matrix = self.embedding_optimizer.adjacency_matrices[sample].to(
                    self.context["device"],
                )
                Sigma_x_inv = self.parameter_optimizer.spatial_affinity[sample].to(
                    self.context["device"],
                )
                nu = adjacency_matrix @ Z
                eta = nu @ Sigma_x_inv
                logZ_i_s = torch.full((N,), 0, **self.context)
                if (prior_x[0] != 0).all():
                    logZ_i_s = torch.full(
                        (N,),
                        -self.K * torch.log(prior_x[0]).item() + torch.log(factorial(self.K - 1, exact=True)).item(),
                        **self.context,
                    )

                logZ_i_z = integrate_of_exponential_over_simplex(eta)
                log_partition_function = (logZ_i_Y + logZ_i_z + logZ_i_s).sum()

                if prior_x_mode == "exponential shared fixed":
                    loss += prior_x[0][0] * S.sum()
                elif not prior_x_mode:
                    pass
                else:
                    raise NotImplementedError

                if Sigma_x_inv is not None:
                    loss += (eta).mul(Z).sum() / 2

                spatial_affinity_bars = None
                if self.parameter_optimizer.spatial_affinity.mode == "differential lookup":
                    spatial_affinity_bars = [
                        self.parameter_optimizer.spatial_affinity_bar[group_name]
                        for group_name in self.parameter_optimizer.spatial_affinity_tags[sample]
                    ]

                regularization = torch.zeros((), **self.context)
                if spatial_affinity_bars is not None:
                    group_weighting = 1 / len(spatial_affinity_bars)
                    for group_Sigma_x_inv_bar in spatial_affinity_bars:
                        regularization += (
                            group_weighting
                            * self.parameter_optimizer.lambda_Sigma_bar
                            * (group_Sigma_x_inv_bar - Sigma_x_inv).pow(2).sum()
                            / 2
                        )

                regularization += self.parameter_optimizer.lambda_Sigma_x_inv * Sigma_x_inv.pow(2).sum() / 2
                regularization *= weighted_total_cells
                loss += regularization

            loss += log_partition_function

            differential_regularization_term = torch.zeros((), **self.context)
            M_bar = None
            if self.parameter_optimizer.metagene_mode == "differential":
                M_bar = [
                    self.parameter_optimizer.metagene_state.M_bar[group_name]
                    for group_name in self.parameter_optimizer.metagene_tags[sample]
                ]

            if self.parameter_optimizer.lambda_M > 0 and M_bar is not None:
                differential_regularization_quadratic_factor = self.parameter_optimizer.lambda_M * torch.eye(
                    self.K,
                    **self.context,
                )

                differential_regularization_linear_term = torch.zeros_like(M, **self.context)
                group_weighting = 1 / len(M_bar)
                for group_M_bar in M_bar:
                    differential_regularization_linear_term += (
                        group_weighting * self.parameter_optimizer.lambda_M * group_M_bar
                    )

                differential_regularization_term = (M @ differential_regularization_quadratic_factor * M).sum() - 2 * (
                    differential_regularization_linear_term * M
                ).sum()
                group_weighting = 1 / len(M_bar)
                for group_M_bar in M_bar:
                    differential_regularization_term += (
                        group_weighting * self.parameter_optimizer.lambda_M * (group_M_bar * group_M_bar).sum()
                    )

            loss += differential_regularization_term
            total_loss += loss

        return total_loss

    def nll(self, use_spatial=False):
        """Compute overall negative log-likelihood for current model
        parameters."""

        with torch.no_grad():
            return self.forward(use_spatial=use_spatial).reshape(1).cpu().numpy()


class Hierarchy:
    """Container for hierarchical views of Popari."""

    def __init__(
        self,
        base_view: "HierarchicalView",
        downsampling_method: str = "grid",
        **hierarchical_view_kwargs,
    ):
        self.view_container = {0: base_view}
        if downsampling_method == "grid":
            self.downsampler = GridDownsampler()
        elif downsampling_method == "partition":
            self.downsampler = PartitionDownsampler()

        self.hierarchical_view_kwargs = hierarchical_view_kwargs

    def __setitem__(self, index: int, view: "HierarchicalView"):
        self.view_container[index] = view

    def __getitem__(self, index: int):
        return self.view_container[index]

    def construct(self, levels: int, downsample_rate: float, **kwargs):
        base_view = self[0]
        context = base_view.context
        previous_view = base_view
        for level in range(1, levels):
            print(f"{get_datetime()} Initializing hierarchy level {level}")
            binned_datasets = []
            binned_Ys = []
            previous_Ys = previous_view.Ys

            effective_kwargs = kwargs.copy()
            for sample, previous_Y in zip(previous_view.replicate_names, previous_Ys):
                previous_dataset = previous_view.sample_adata(sample)
                bin_assignments_key = f"{BIN_ASSIGNMENTS_KEY}_{sample}"
                binned_dataset, effective_kwargs = self.downsampler.downsample(
                    previous_dataset,
                    downsample_rate=downsample_rate,
                    bin_assignments_key=bin_assignments_key,
                    **effective_kwargs,
                )
                binned_dataset.popari.name = sample
                compute_spatial_neighbors(binned_dataset)

                print(
                    f"{get_datetime()} Downsized dataset from {len(previous_dataset)} to {len(binned_dataset)} spots.",
                )

                binned_datasets.append(binned_dataset)
                bin_assignments = convert_numpy_to_pytorch_sparse_coo(
                    csr_array(binned_dataset.obsm[bin_assignments_key]).tocoo(),
                    context=context,
                )

                binned_Y = bin_assignments @ previous_Y
                binned_Ys.append(binned_Y)

            binned_adata = merge_anndata(
                binned_datasets,
                sample_key=previous_view.sample_key,
            )
            level_view = HierarchicalView(
                binned_adata,
                level=level,
                binned_Ys=binned_Ys,
                **self.hierarchical_view_kwargs,
            )
            previous_view.link(level_view)
            self[level] = level_view

            previous_view = level_view

    @classmethod
    def reconstruct(cls, reloaded_hierarchy: dict, **hierarchical_view_kwargs):
        """Reconstruct a hierarchy from one unified AnnData per level."""

        context = hierarchical_view_kwargs["context"]

        def reconstruct_level(
            level: int,
            adata: AnnData,
            previous_view: "HierarchicalView | None",
        ):
            print(f"{get_datetime()} Reloading level {level}")
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

            level_view = HierarchicalView(
                adata,
                level=level,
                binned_Ys=binned_Ys,
                **hierarchical_view_kwargs,
            )

            if previous_view is not None:
                previous_view.link(level_view)

            return level_view

        base_view = reconstruct_level(0, reloaded_hierarchy[0], None)

        hierarchy = cls(base_view=base_view, **hierarchical_view_kwargs)
        previous_view = base_view

        for level in range(1, hierarchical_view_kwargs["hierarchical_levels"]):
            adata = reloaded_hierarchy[level]
            level_view = reconstruct_level(level, adata, previous_view)
            hierarchy[level] = level_view
            previous_view = level_view

        return hierarchy
