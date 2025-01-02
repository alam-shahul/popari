from __future__ import annotations

from collections import defaultdict
from typing import Optional, Sequence

import numpy as np
import torch
from scipy.sparse import csr_array
from tqdm.auto import trange

from popari._binning_utils import GridDownsampler, PartitionDownsampler
from popari._embedding_optimizer import EmbeddingOptimizer
from popari._parameter_optimizer import ParameterOptimizer
from popari._popari_dataset import PopariDataset
from popari.initialization import initialize_dummy, initialize_kmeans, initialize_leiden, initialize_svd
from popari.sample_for_integral import integrate_of_exponential_over_simplex
from popari.util import convert_numpy_to_pytorch_sparse_coo, get_datetime


class HierarchicalView:
    """View of SRT multisample dataset at a set resolution.

    Includes the scaled (i.e. binned data) as well as the learnable Popari
    parameters and their corresponding optimizers.

    """

    def __init__(
        self,
        datasets: Sequence[PopariDataset],
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

        self.datasets = datasets
        self.replicate_names = [dataset.name for dataset in datasets]
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

        self.num_replicates = len(self.datasets)

        def fill_groups(groups, are_exclusive=False):
            if not groups:
                groups = {f"_default": self.replicate_names}

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
                        ValueError("If in shared mode, each replicate can only appear in one group.")
                    tags[replicate].append(group)

            return groups, tags

        def add_level_suffix(groups):
            suffixed_groups = {}
            for group, group_replicates in groups.items():
                suffixed_groups[group] = [
                    f"{group_replicate}{self.level_suffix}" for group_replicate in group_replicates
                ]

            return suffixed_groups

        if metagene_groups == "disjoint":
            metagene_groups = {replicate_name: [replicate_name] for replicate_name in self.replicate_names}

        if spatial_affinity_groups == "disjoint":
            spatial_affinity_groups = {replicate_name: [replicate_name] for replicate_name in self.replicate_names}

        if metagene_groups is not None:
            metagene_groups = add_level_suffix(metagene_groups)
        if spatial_affinity_groups is not None:
            spatial_affinity_groups = add_level_suffix(spatial_affinity_groups)

        self.metagene_groups, self.metagene_tags = fill_groups(
            metagene_groups,
            are_exclusive=(parameter_optimizer_hyperparameters["metagene_mode"] == "shared"),
        )
        self.spatial_affinity_groups, self.spatial_affinity_tags = fill_groups(
            spatial_affinity_groups,
            are_exclusive=(parameter_optimizer_hyperparameters["spatial_affinity_mode"] == "shared lookup"),
        )

        parameter_optimizer_hyperparameters["metagene_groups"] = self.metagene_groups
        parameter_optimizer_hyperparameters["metagene_tags"] = self.metagene_tags
        parameter_optimizer_hyperparameters["spatial_affinity_groups"] = self.spatial_affinity_groups
        parameter_optimizer_hyperparameters["spatial_affinity_tags"] = self.spatial_affinity_tags

        if binned_Ys is None:
            self.Ys = []
            for dataset in self.datasets:
                num_cells, _ = dataset.shape
                Y = convert_numpy_to_pytorch_sparse_coo(dataset.X, self.context)
                Y *= (self.K * 1) / (Y.sum() / num_cells)
                self.Ys.append(Y)
        else:
            self.Ys = binned_Ys

        if betas is None:
            self.betas = np.full(self.num_replicates, 1 / self.num_replicates)
        else:
            self.betas = np.array(betas, copy=False) / sum(betas)

        if prior_x_modes is None:
            prior_x_modes = [None] * self.num_replicates

        self.prior_x_modes = prior_x_modes

        self.parameter_optimizer = ParameterOptimizer(
            self.K,
            self.Ys,
            self.datasets,
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
            self.datasets,
            initial_context=self.initial_context,
            context=self.context,
            use_inplace_ops=self.use_inplace_ops,
            verbose=self.verbose,
            **embedding_optimizer_hyperparameters,
        )
        self.parameter_optimizer.link(self.embedding_optimizer)
        self.embedding_optimizer.link(self.parameter_optimizer)

        if self.pretrained:
            first_dataset = self.datasets[0]
            # if self.metagene_mode == "differential":
            #     self.parameter_optimizer.metagene_state.M_bar = {group_name: torch.from_numpy(first_dataset.uns["M_bar"][group_name]).to(**self.initial_context) for group_name in self.metagene_groups}
            # if self.spatial_affinity_mode == "differential lookup":
            #     self.parameter_optimizer.spatial_affinity_state.spatial_affinity_bar = {group_name: first_dataset.uns["M_bar"][group_name].to(**self.initial_context) for group_name in self.spatial_affinity_groups}
            spatial_affinity_copy = torch.zeros((len(self.datasets), self.K, self.K), **self.context)
            for dataset_index, dataset in enumerate(self.datasets):
                self.parameter_optimizer.metagene_state[dataset.name][:] = torch.from_numpy(
                    dataset.uns["M"][dataset.name],
                ).to(**self.initial_context)
                self.embedding_optimizer.embedding_state[dataset.name][:] = torch.from_numpy(dataset.obsm["X"]).to(
                    **self.initial_context,
                )
                self.embedding_optimizer.adjacency_matrices[dataset.name] = convert_numpy_to_pytorch_sparse_coo(
                    dataset.obsp["adjacency_matrix"],
                    self.initial_context,
                )
                self.parameter_optimizer.adjacency_matrices[dataset.name] = convert_numpy_to_pytorch_sparse_coo(
                    dataset.obsp["adjacency_matrix"],
                    self.initial_context,
                )

                self.parameter_optimizer.spatial_affinity_state[dataset.name] = torch.from_numpy(
                    dataset.uns["Sigma_x_inv"][dataset.name],
                ).to(**self.initial_context)
                spatial_affinity_copy[dataset_index] = self.parameter_optimizer.spatial_affinity_state[dataset.name]

            self.parameter_optimizer.update_sigma_yx()
            self.parameter_optimizer.spatial_affinity_state.initialize_optimizers(spatial_affinity_copy)
        else:
            if self.level < self.hierarchical_levels - 1:
                method = "dummy"

            if self.verbose:
                print(f"{get_datetime()} Initializing metagenes and hidden states using {method} method")

            if method == "dummy":
                self.M, self.Xs = initialize_dummy(self.datasets, self.K, self.initial_context)
            elif method == "kmeans":
                self.M, self.Xs = initialize_kmeans(
                    self.datasets,
                    self.K,
                    self.initial_context,
                    kwargs_kmeans=dict(random_state=self.random_state),
                )
            elif method == "svd":
                self.M, self.Xs = initialize_svd(
                    self.datasets,
                    self.K,
                    self.initial_context,
                    M_nonneg=(self.parameter_optimizer.M_constraint == "simplex"),
                    X_nonneg=True,
                )
            elif method == "leiden":
                kwargs_leiden = {
                    "random_state": self.random_state,
                }
                self.M, self.Xs = initialize_leiden(
                    self.datasets,
                    self.K,
                    self.initial_context,
                    kwargs_leiden=kwargs_leiden,
                    verbose=self.verbose,
                )
            else:
                raise NotImplementedError

            for dataset_index, dataset in enumerate(self.datasets):
                self.parameter_optimizer.metagene_state[dataset.name][:] = self.M
                self.embedding_optimizer.embedding_state[dataset.name][:] = self.Xs[dataset_index]

            self.parameter_optimizer.scale_metagenes()

            # # Ensure initial embeddings do not have too large magnitudes
            # for dataset_index, dataset in enumerate(self.datasets):
            #     initial_X = self.embedding_optimizer.embedding_state[dataset.name]
            #     cell_normalized_X = initial_X / torch.linalg.norm(initial_X, dim=0, keepdim=True)
            #     self.embedding_optimizer.embedding_state[dataset.name][:] = cell_normalized_X

            self.Sigma_x_inv_bar = None

            self.parameter_optimizer.update_sigma_yx()

            # # Update metagenes to ensure that they lie on simplex after normalizign embeddings
            # self.parameter_optimizer.update_metagenes()

            initial_embeddings = [self.embedding_optimizer.embedding_state[dataset.name] for dataset in self.datasets]

            # Initializing spatial affinities
            if self.verbose:
                print(f"{get_datetime()} Initializing Sigma_x_inv with empirical correlations")
            self.parameter_optimizer.spatial_affinity_state.initialize(initial_embeddings)

            for dataset_index, dataset in enumerate(self.datasets):
                metagene_state = self.parameter_optimizer.metagene_state[dataset.name].cpu().detach().numpy()
                dataset.uns["M"] = {dataset.name: metagene_state}

                X = self.embedding_optimizer.embedding_state[dataset.name].cpu().detach().numpy()
                dataset.obsm["X"] = X

                Sigma_x_inv = self.parameter_optimizer.spatial_affinity_state[dataset.name].cpu().detach().numpy()
                dataset.uns["Sigma_x_inv"] = {dataset.name: Sigma_x_inv}

                dataset.uns["popari_hyperparameters"] = {
                    "prior_x": self.parameter_optimizer.prior_xs[dataset_index][0].cpu().detach().numpy(),
                    "K": self.K,
                    "use_inplace_ops": self.use_inplace_ops,
                    "random_state": self.random_state,
                    "verbose": self.verbose,
                    **parameter_optimizer_hyperparameters,
                    **embedding_optimizer_hyperparameters,
                }

                # if self.metagene_mode == "differential":
                #     dataset.uns["popari_hyperparameters"]["lambda_M"] = self.parameter_optimizer.lambda_M
                # if self.spatial_affinity_mode != "shared":
                #     dataset.uns["popari_hyperparameters"]["lambda_Sigma_bar"] = self.parameter_optimizer.lambda_Sigma_bar

            if self.parameter_optimizer.metagene_mode == "differential":
                M_bar = {
                    group_name: self.parameter_optimizer.metagene_state.M_bar[group_name].cpu().detach().numpy()
                    for group_name in self.parameter_optimizer.metagene_groups
                }
                for dataset in self.datasets:
                    dataset.uns["M_bar"] = M_bar
                    # dataset.uns["lambda_Sigma_bar"] = self.parameter_optimizer.lambda_Sigma_bar

            if self.parameter_optimizer.spatial_affinity_mode == "differential lookup":
                spatial_affinity_bar = {
                    group_name: self.parameter_optimizer.spatial_affinity_state.spatial_affinity_bar[group_name]
                    .cpu()
                    .detach()
                    .numpy()
                    for group_name in self.parameter_optimizer.spatial_affinity_groups
                }
                for dataset in self.datasets:
                    dataset.uns["spatial_affinity_bar"] = spatial_affinity_bar
                    # dataset.uns["lambda_M"] = self.parameter_optimizer.lambda_M

        self.superresolution_optimizers = {}
        for dataset in self.datasets:
            if "losses" not in dataset.uns:
                dataset.uns["losses"] = defaultdict(list)
            else:
                dataset.uns["losses"] = defaultdict(list, dataset.uns["losses"])
                for key in dataset.uns["losses"]:
                    dataset.uns["losses"][key] = list(dataset.uns["losses"][key])

        if self.parameter_optimizer.metagene_mode == "differential":
            self.parameter_optimizer.metagene_state.reaverage()
        if self.parameter_optimizer.spatial_affinity_mode == "differential lookup":
            self.parameter_optimizer.spatial_affinity_state.reaverage()

    def link(self, low_res_view: HierarchicalView):
        """Link a view to the resolution right below it in the hierarchy."""
        self.low_res_view = low_res_view

    def _propagate_parameters(self):
        """Use parameters from low-resolution to initialize higher-
        resolution."""
        low_res_metagenes = self.low_res_view.parameter_optimizer.metagene_state.metagenes
        self.parameter_optimizer.metagene_state.metagenes[:] = low_res_metagenes

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

        final_losses = np.zeros(len(self.datasets))
        for dataset_index, (dataset, low_res_name) in enumerate(zip(self.datasets, self.low_res_view.replicate_names)):
            low_res_dataset = self.low_res_view.datasets[dataset_index]
            sigma_yx = self.parameter_optimizer.sigma_yxs[dataset_index]
            Y = self.Ys[dataset_index].to(self.context["device"])

            if Y.sum() == 0:
                raise ValueError(
                    "It seems like you are trying to superresolve a hierarchical level with all zero expression "
                    "values. This probably means the model was saved incorrectly; try using `model.save_results` "
                    "with `as_trainable=True` next time.",
                )

            X = self.embedding_optimizer.embedding_state[dataset.name].to(self.context["device"])
            X_B = low_res_dataset.obsm["X"]
            B = low_res_dataset.obsm[f"bin_assignments_{low_res_dataset.name}"]

            M = self.parameter_optimizer.metagene_state[dataset.name].to(self.context["device"])
            prior_x_mode = self.parameter_optimizer.prior_x_modes[dataset_index]
            prior_x = self.parameter_optimizer.prior_xs[dataset_index]

            # Precomputing quantities
            MTM = M.T @ M / (sigma_yx**2)
            BTB = convert_numpy_to_pytorch_sparse_coo((B.T @ B).tocoo(), context=self.context)
            YM = Y @ M / (sigma_yx**2)
            BTX_B = torch.from_numpy(B.T @ X_B).to(self.context["device"])

            linear_term_gradient = YM + BTX_B
            if prior_x_mode == "exponential shared fixed":
                linear_term_gradient = linear_term_gradient - prior_x[0][None]

            Ynorm = torch.square(Y).sum() / (sigma_yx**2)
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
            # superresolution_scheduler = torch.optim.lr_scheduler.StepLR(superresolution_optimizer, step_size=decay_period, gamma=0.5)

            # self.superresolution_optimizers[dataset.name] = superresolution_optimizer
            # self.superresolution_schedulers[dataset.name] = superresolution_scheduler

            def gradient_update(X, iteration=None):
                """TODO:UNTESTED."""

                # lambda_B = 1e-1
                # self.superresolution_optimizers[dataset.name].zero_grad()
                superresolution_optimizer.zero_grad()
                # X.grad = None
                quadratic_term_gradient = X @ MTM + BTB @ X

                loss = (
                    (quadratic_term_gradient * X).sum() / 2 - (linear_term_gradient * X).sum() + Ynorm / 2 + X_Bnorm / 2
                )
                if use_manual_gradients:
                    gradient = quadratic_term_gradient - linear_term_gradient
                    X.grad = gradient
                else:
                    # loss = (torch.linalg.norm(Y.T - M @ X.T, ord='fro') ** 2) / 2  + ((torch.linalg.norm(X_B - B @ X, ord='fro') ** 2) / 2 )
                    loss.backward()

                # self.superresolution_optimizers[dataset.name].step()
                superresolution_optimizer.step()
                # self.superresolution_schedulers[dataset.name].step()

                if verbose > 4 and (iteration % 5 == 0):
                    pass
                    # print(f'NMF component: {((X @ MTM - YM) * X).sum().item() / 2 + Ynorm / 2}')
                    # print(f'Superresolution component: {((BTB @ X - BTX_B) * X).sum().item() / 2 + X_Bnorm / 2}')
                    # print(f'Total loss (recomputed): {((X @ MTM - YM + BTB @ X - BTX_B) * X).sum().item() / 2 + (Ynorm + X_Bnorm) / 2}')
                    # print(f"NMF component: {(torch.linalg.norm(Y.T - M @ X.T, ord='fro').item() ** 2)/ (sigma_yx ** 2)}")
                    # print(f"Superresolution component: {torch.linalg.norm(X_B - B @ X, ord='fro').item() ** 2}")
                    # print(f"Total loss (recomputed): {(torch.linalg.norm(Y.T - M @ X.T, ord='fro').item() ** 2)/ (sigma_yx ** 2) + torch.linalg.norm(X_B - B @ X, ord='fro').item() ** 2}")
                    # print(f"Total loss: {loss}")

                # gradient = quadratic_term_gradient - linear_term_gradient
                # X = X.sub(gradient, alpha=self.superresolution_lr)
                # X = torch.clip(X, min=1e-10)
                with torch.no_grad():
                    X.clamp_(min=1e-10)

                # del quadratic_term_gradient

                return loss

            progress_bar = trange(n_epochs, leave=True, disable=(verbose < 5), miniters=10000)
            for epoch in progress_bar:
                X_prev = X.clone().detach()
                if update_alg == "mu":
                    pass
                elif update_alg == "gd":
                    loss = gradient_update(X, iteration=epoch)

                dX = torch.abs((X_prev - X) / torch.linalg.norm(X, dim=1, ord=1, keepdim=True)).max().item()

                # if epoch % decay_period:
                #     tol /= 10

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
            self.embedding_optimizer.embedding_state[dataset.name][:] = X.clone().detach()

            final_losses[dataset_index] = loss.cpu().detach().numpy()

            # Delete dangling reference
            # TODO: can probably delete all this?
            del superresolution_optimizer
            # del self.superresolution_optimizers[dataset.name]

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
        for dataset_index, dataset in enumerate(self.datasets):
            self.parameter_optimizer.metagene_state[dataset.name][:] = torch.from_numpy(
                dataset.uns["M"][dataset.name],
            ).to(**self.initial_context)
            self.embedding_optimizer.embedding_state[dataset.name][:] = torch.from_numpy(dataset.obsm["X"]).to(
                **self.initial_context,
            )
            # self.parameter_optimizer.sigma_yxs[dataset_index] = dataset.uns["sigma_yx"] TODO: sigma_yx doesn't seem to be saved correctly due to issues with `merge_anndata`

            with torch.no_grad():
                self.parameter_optimizer.spatial_affinity_state[dataset.name][:] = torch.from_numpy(
                    dataset.uns["Sigma_x_inv"][dataset.name],
                ).to(**self.initial_context)

        if self.parameter_optimizer.metagene_mode == "differential":
            self.parameter_optimizer.metagene_state.reaverage()

        if self.parameter_optimizer.spatial_affinity_mode == "differential lookup":
            self.parameter_optimizer.spatial_affinity_state.reaverage()

        self.parameter_optimizer.update_sigma_yx()  # TODO: sigma_yx doesn't seem to be saved correctly due to issues with `merge_anndata`; try that instead of this

    def synchronize_datasets(self):
        """Synchronize datasets with learned view parameters and embeddings."""
        for dataset_index, dataset in enumerate(self.datasets):
            dataset.uns["M"][dataset.name] = (
                self.parameter_optimizer.metagene_state[dataset.name].cpu().detach().numpy()
            )
            dataset.obsm["X"] = self.embedding_optimizer.embedding_state[dataset.name].cpu().detach().numpy()
            dataset.uns["sigma_yx"] = self.parameter_optimizer.sigma_yxs[dataset_index]
            with torch.no_grad():
                dataset.uns["Sigma_x_inv"][dataset.name][:] = (
                    self.parameter_optimizer.spatial_affinity_state[dataset.name].cpu().detach().numpy()
                )

            dataset.uns["losses"]["nll_embeddings"].append(self.embedding_optimizer.nll_embeddings())
            dataset.uns["losses"]["nll_spatial_affinities"].append(self.parameter_optimizer.nll_spatial_affinities())
            dataset.uns["losses"]["nll_metagenes"].append(self.parameter_optimizer.nll_metagenes())
            dataset.uns["losses"]["nll_sigma_yx"].append(self.parameter_optimizer.nll_sigma_yx())
            dataset.uns["losses"]["nll"].append(self.nll())

        if self.parameter_optimizer.metagene_mode == "differential":
            M_bar = {
                group_name: self.parameter_optimizer.metagene_state.M_bar[group_name].cpu().detach().numpy()
                for group_name in self.parameter_optimizer.metagene_groups
            }
            for dataset in self.datasets:
                dataset.uns["M_bar"] = M_bar

        if self.parameter_optimizer.spatial_affinity_mode == "differential lookup":
            spatial_affinity_bar = {
                group_name: self.parameter_optimizer.spatial_affinity_state.spatial_affinity_bar[group_name]
                .cpu()
                .detach()
                .numpy()
                for group_name in self.parameter_optimizer.spatial_affinity_groups
            }
            for dataset in self.datasets:
                dataset.uns["spatial_affinity_bar"] = spatial_affinity_bar

    def nll(self, use_spatial=False):
        """Compute overall negative log-likelihood for current model
        parameters."""

        with torch.no_grad():
            total_loss = torch.zeros(1, **self.context)
            if use_spatial:
                weighted_total_cells = 0
                for dataset in self.datasets:
                    E_adjacency_list = self.embedding_optimizer.adjacency_lists[dataset.name]
                    weighted_total_cells += sum(map(len, E_adjacency_list))

            for dataset_index, dataset in enumerate(self.datasets):
                sigma_yx = self.parameter_optimizer.sigma_yxs[dataset_index]
                Y = self.Ys[dataset_index].to(self.context["device"])
                X = self.embedding_optimizer.embedding_state[dataset.name].to(self.context["device"])
                M = self.parameter_optimizer.metagene_state[dataset.name].to(self.context["device"])
                prior_x_mode = self.parameter_optimizer.prior_x_modes[dataset_index]
                beta = self.betas[dataset_index]
                prior_x = self.parameter_optimizer.prior_xs[dataset_index]

                # Precomputing quantities
                MTM = M.T @ M / (sigma_yx**2)
                YM = Y.to(M.device) @ M / (sigma_yx**2)
                Ynorm = torch.square(Y).sum() / (sigma_yx**2)
                S = torch.linalg.norm(X, dim=1, ord=1, keepdim=True)

                Z = X / S
                N, G = Y.shape

                loss = ((X @ MTM) * X).sum() / 2 - (X * YM).sum() + Ynorm / 2

                logZ_i_Y = torch.full((N,), G / 2 * np.log(2 * np.pi * sigma_yx**2), **self.context)
                if not use_spatial:
                    logZ_i_X = torch.full((N,), 0, **self.context)
                    if (prior_x[0] != 0).all():
                        logZ_i_X += torch.full((N,), self.K * torch.log(prior_x[0]).item(), **self.context)
                    log_partition_function = (logZ_i_Y + logZ_i_X).sum()
                else:
                    adjacency_matrix = self.embedding_optimizer.adjacency_matrices[dataset.name].to(
                        self.context["device"],
                    )
                    Sigma_x_inv = self.parameter_optimizer.spatial_affinity_state[dataset.name].to(
                        self.context["device"],
                    )
                    nu = adjacency_matrix @ Z
                    eta = nu @ Sigma_x_inv
                    logZ_i_s = torch.full((N,), 0, **self.context)
                    if (prior_x[0] != 0).all():
                        logZ_i_s = torch.full(
                            (N,),
                            -self.K * torch.log(prior_x[0]).item()
                            + torch.log(factorial(self.K - 1, exact=True)).item(),
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
                    if self.parameter_optimizer.spatial_affinity_state.mode == "differential lookup":
                        spatial_affinity_bars = [
                            self.parameter_optimizer.spatial_affinity_state.spatial_affinity_bar[group_name].detach()
                            for group_name in self.parameter_optimizer.spatial_affinity_tags[dataset.name]
                        ]

                    regularization = torch.zeros(1, **self.context)
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

                    loss += regularization.item()

                loss += log_partition_function

                differential_regularization_term = torch.zeros(1, **self.context)
                M_bar = None
                if self.parameter_optimizer.metagene_mode == "differential":
                    M_bar = [
                        self.parameter_optimizer.metagene_state.M_bar[group_name]
                        for group_name in self.parameter_optimizer.metagene_tags[dataset.name]
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

                    differential_regularization_term = (
                        M @ differential_regularization_quadratic_factor * M
                    ).sum() - 2 * (differential_regularization_linear_term * M).sum()
                    group_weighting = 1 / len(M_bar)
                    for group_M_bar in M_bar:
                        differential_regularization_term += (
                            group_weighting * self.parameter_optimizer.lambda_M * (group_M_bar * group_M_bar).sum()
                        )

                loss += differential_regularization_term.item()

                total_loss += loss

        return total_loss.cpu().numpy()


class Hierarchy:
    """Container for hierarchical views of Popari."""

    def __init__(
        self,
        base_view: HierarchicalView,
        downsampling_method: str = "grid",
        **hierarchical_view_kwargs,
    ):
        self.view_container = {0: base_view}
        if downsampling_method == "grid":
            self.downsampler = GridDownsampler()
        elif downsampling_method == "partition":
            self.downsampler = PartitionDownsampler()

        self.hierarchical_view_kwargs = hierarchical_view_kwargs

    def __setitem__(self, index: int, view: HierarchicalView):
        self.view_container[index] = view

    def __getitem__(self, index: int):
        return self.view_container[index]

    def construct(self, levels: int, downsample_rate: float, **kwargs):
        base_view = self[0]
        context = base_view.context
        previous_view = base_view
        original_names = [dataset.name for dataset in previous_view.datasets]
        for level in range(1, levels):
            print(f"{get_datetime()} Initializing hierarchy level {level}")
            previous_datasets = previous_view.datasets
            binned_datasets = []
            binned_Ys = []
            previous_Ys = previous_view.Ys

            effective_kwargs = kwargs.copy()
            for previous_Y, previous_dataset, original_name in zip(previous_Ys, previous_datasets, original_names):
                dataset_name, *_ = previous_dataset.name.split("_level_")
                binned_dataset_name = f"{dataset_name}_level_{level}"
                bin_assignments_key = f"bin_assignments_{binned_dataset_name}"
                binned_dataset, effective_kwargs = self.downsampler.downsample(
                    previous_dataset,
                    downsample_rate=downsample_rate,
                    bin_assignments_key=bin_assignments_key,
                    **effective_kwargs,
                )
                binned_dataset = PopariDataset(binned_dataset, binned_dataset_name)
                binned_dataset.compute_spatial_neighbors()

                print(
                    f"{get_datetime()} Downsized dataset from {len(previous_dataset)} to {len(binned_dataset)} spots.",
                )

                binned_datasets.append(binned_dataset)
                bin_assignments = convert_numpy_to_pytorch_sparse_coo(
                    csr_array(binned_dataset.obsm[f"bin_assignments_{binned_dataset_name}"]).tocoo(),
                    context=context,
                )

                binned_Y = bin_assignments @ previous_Y
                binned_Ys.append(binned_Y)

            level_view = HierarchicalView(
                binned_datasets,
                level=level,
                binned_Ys=binned_Ys,
                **self.hierarchical_view_kwargs,
            )
            previous_view.link(level_view)
            self[level] = level_view

            previous_view = level_view

    @classmethod
    def reconstruct(cls, reloaded_hierarchy: dict, **hierarchical_view_kwargs):
        """Reconstruct hierarchy object from dictionary of level binned
        datasets."""

        context = hierarchical_view_kwargs["context"]

        def reconstruct_level(level: int, datasets: Sequence[PopariDataset], previous_view: HierarchicalView | None):
            print(f"{get_datetime()} Reloading level {level}")
            if previous_view is not None:
                binned_Ys = []
                for dataset, previous_Y in zip(datasets, previous_view.Ys):
                    B = dataset.obsm[f"bin_assignments_{dataset.name}"]
                    dataset.obsm[f"bin_assignments_{dataset.name}"] = csr_array(B)
                    binned_Y = (
                        convert_numpy_to_pytorch_sparse_coo(
                            dataset.obsm[f"bin_assignments_{dataset.name}"],
                            context=context,
                        )
                        @ previous_Y
                    )
                    binned_Ys.append(binned_Y)
            else:
                binned_Ys = None

            level_view = HierarchicalView(
                datasets,
                level=level,
                binned_Ys=binned_Ys,
                **hierarchical_view_kwargs,
            )

            level_view._reload_state()

            if previous_view is not None:
                previous_view.link(level_view)

            return level_view

        base_view = reconstruct_level(0, reloaded_hierarchy[0], None)

        hierarchy = cls(base_view=base_view, **hierarchical_view_kwargs)
        previous_view = base_view

        for level in range(1, hierarchical_view_kwargs["hierarchical_levels"]):
            datasets = reloaded_hierarchy[level]
            level_view = reconstruct_level(level, datasets, previous_view)
            hierarchy[level] = level_view
            previous_view = level_view

        return hierarchy
