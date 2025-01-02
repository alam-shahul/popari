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


class ParameterOptimizer:
    """Optimizer and state for Popari parameters."""

    def __init__(
        self,
        K,
        Ys,
        datasets,
        betas,
        prior_x_modes,
        metagene_groups,
        metagene_tags,
        spatial_affinity_groups,
        spatial_affinity_tags,
        spatial_affinity_regularization_power=2,
        spatial_affinity_constraint=None,
        spatial_affinity_centering=False,
        spatial_affinity_scaling=10,
        lambda_Sigma_x_inv=1e-2,
        spatial_affinity_tol=2e-3,
        spatial_affinity_mode="shared lookup",
        metagene_mode="shared",
        lambda_M=0.5,
        lambda_Sigma_bar=0.5,
        spatial_affinity_lr=1e-3,
        M_constraint="simplex",
        sigma_yx_inv_mode="separate",
        initial_context=None,
        context=None,
        use_inplace_ops=False,
        verbose=0,
    ):
        self.verbose = verbose
        self.use_inplace_ops = use_inplace_ops

        self.datasets = datasets
        self.spatial_affinity_mode = spatial_affinity_mode
        self.K = K
        self.Ys = Ys
        self.metagene_groups = metagene_groups
        self.metagene_tags = metagene_tags
        self.spatial_affinity_groups = spatial_affinity_groups
        self.spatial_affinity_tags = spatial_affinity_tags
        self.sigma_yx_inv_mode = sigma_yx_inv_mode
        self.lambda_Sigma_bar = lambda_Sigma_bar
        self.spatial_affinity_constraint = spatial_affinity_constraint
        self.spatial_affinity_centering = spatial_affinity_centering
        self.spatial_affinity_lr = spatial_affinity_lr
        self.spatial_affinity_scaling = spatial_affinity_scaling
        self.lambda_Sigma_x_inv = lambda_Sigma_x_inv
        self.spatial_affinity_tol = spatial_affinity_tol
        self.lambda_M = lambda_M
        self.metagene_mode = metagene_mode
        self.M_constraint = M_constraint
        self.prior_x_modes = prior_x_modes
        self.betas = betas
        self.initial_context = initial_context if initial_context else {"device": "cpu", "dtype": torch.float32}
        self.context = context if context else {"device": "cpu", "dtype": torch.float32}
        self.spatial_affinity_regularization_power = spatial_affinity_regularization_power
        self.adjacency_lists = {dataset.name: dataset.obsm["adjacency_list"] for dataset in self.datasets}
        self.adjacency_matrices = {
            dataset.name: convert_numpy_to_pytorch_sparse_coo(dataset.obsp["adjacency_matrix"], self.context)
            for dataset in self.datasets
        }

        if self.verbose:
            print(f"{get_datetime()} Initializing MetageneState")

        self.metagene_state = MetageneState(
            self.K,
            self.datasets,
            self.metagene_groups,
            self.metagene_tags,
            mode=self.metagene_mode,
            M_constraint=self.M_constraint,
            initial_context=self.initial_context,
            context=self.context,
        )

        if self.verbose:
            print(f"{get_datetime()} Initializing SpatialAffinityState")

        self.spatial_affinity_state = SpatialAffinityState(
            self.K,
            self.metagene_state,
            self.datasets,
            self.spatial_affinity_groups,
            self.spatial_affinity_tags,
            self.betas,
            scaling=self.spatial_affinity_scaling,
            mode=self.spatial_affinity_mode,
            initial_context=self.initial_context,
            lr=self.spatial_affinity_lr,
            context=self.context,
        )

        if all(prior_x_mode == "exponential shared fixed" for prior_x_mode in self.prior_x_modes):
            self.prior_xs = [(torch.ones(self.K, **self.initial_context),) for _ in range(len(self.datasets))]
        elif all(prior_x_mode == None for prior_x_mode in self.prior_x_modes):
            self.prior_xs = [(torch.zeros(self.K, **self.initial_context),) for _ in range(len(self.datasets))]
        else:
            raise NotImplementedError

        self.sigma_yxs = np.zeros(len(self.datasets))

    def link(self, embedding_optimizer):
        """Link to embedding_optimizer."""
        self.embedding_optimizer = embedding_optimizer

    def scale_metagenes(self):
        norm_axis = 1
        # norm_axis = int(self.metagene_mode == "differential")
        if self.M_constraint == "simplex":
            scale_factor = torch.linalg.norm(self.metagene_state.metagenes, axis=norm_axis, ord=1, keepdim=True)
        elif self.M_constraint == "unit_sphere":
            scale_factor = torch.linalg.norm(self.metagene_state.metagenes, axis=norm_axis, ord=2, keepdim=True)

        self.metagene_state.metagenes.div_(scale_factor)
        for group_index, group_replicates in enumerate(self.metagene_groups.values()):
            for dataset_index, dataset in enumerate(self.datasets):
                if dataset.name not in group_replicates:
                    continue
                replicate_embedding = self.embedding_optimizer.embedding_state.embeddings[dataset_index]
                if self.metagene_mode == "differential":
                    replicate_scale_factor = scale_factor[dataset_index]
                    replicate_embedding.mul_(replicate_scale_factor)
                else:
                    group_scale_factor = scale_factor[group_index]
                    replicate_embedding.mul_(group_scale_factor)

    def estimate_Sigma_x_inv(
        self,
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
        datasets = [dataset for (use_replicate, dataset) in zip(replicate_mask, self.datasets) if use_replicate]
        betas = [beta for (use_replicate, beta) in zip(replicate_mask, self.betas) if use_replicate]
        betas = np.array(betas) / np.sum(betas)

        Xs = [self.embedding_optimizer.embedding_state[dataset.name] for dataset in datasets]
        spatial_flags = ["adjacency_list" in dataset.obsm for dataset in datasets]

        num_edges_per_fov = [sum(map(len, dataset.obsm["adjacency_list"])) for dataset in datasets]

        if not any(
            sum(map(len, dataset.obsm["adjacency_list"])) > 0 and u for dataset, u in zip(datasets, spatial_flags)
        ):
            return

        linear_term_coefficient = torch.zeros_like(Sigma_x_inv).requires_grad_(False)
        size_factors = [torch.linalg.norm(X, axis=1, ord=1, keepdim=True) for X in Xs]
        Zs = [X.to(self.context["device"]) / size_factor for X, size_factor in zip(Xs, size_factors)]
        nus = []  # sum of neighbors' z
        weighted_total_cells = 0

        for Z, dataset, use_spatial, beta in zip(Zs, datasets, spatial_flags, self.betas):
            adjacency_list = self.adjacency_lists[dataset.name]
            adjacency_matrix = self.adjacency_matrices[dataset.name]

            if use_spatial:
                nu = adjacency_matrix @ Z
                linear_term_coefficient.addmm_(Z.T, nu, alpha=beta)
            else:
                nu = None

            nus.append(nu)
            weighted_total_cells += beta * sum(map(len, adjacency_list))
            del Z, adjacency_matrix
        # linear_term_coefficient = (linear_term_coefficient + linear_term_coefficient.T) / 2 # should be unnecessary as long as adjacency_list is symmetric
        if self.verbose > 2:
            print(
                f"spatial affinity linear term coefficient range: {linear_term_coefficient.min().item():.2e} ~ {linear_term_coefficient.max().item():.2e}",
            )

        history = []
        Sigma_x_inv.requires_grad_(True)

        loss_prev, loss = np.inf, np.nan

        verbose_bar = tqdm(disable=not (self.verbose > 2), bar_format="{desc}{postfix}")
        progress_bar = trange(1, n_epochs + 1, disable=not self.verbose, desc="Updating Σx-1")

        Sigma_x_inv_best, loss_best, epoch_best = None, np.inf, -1
        dSigma_x_inv = np.inf
        early_stop_epoch_count = 0
        Sigma_x_inv_prev = Sigma_x_inv.clone().detach()
        for epoch in progress_bar:
            optimizer.zero_grad()

            # Compute loss
            linear_term = Sigma_x_inv.view(-1) @ linear_term_coefficient.view(-1)
            regularization = torch.zeros(1, **self.context)
            if Sigma_x_inv_bar is not None:
                group_weighting = 1 / len(Sigma_x_inv_bar)
                for group_Sigma_x_inv_bar in Sigma_x_inv_bar:
                    regularization += (
                        group_weighting
                        * self.lambda_Sigma_bar
                        * (group_Sigma_x_inv_bar - Sigma_x_inv).pow(2).sum()
                        * weighted_total_cells
                        / 2
                    )

            regularization += (
                self.lambda_Sigma_x_inv
                * Sigma_x_inv.abs().pow(self.spatial_affinity_regularization_power).sum()
                * weighted_total_cells
                / 2
            )

            log_partition_function = 0
            for nu, beta in zip(nus, self.betas):
                if subsample_rate is None:
                    subsample_index = np.arange(len(dataset))
                    subsample_multiplier = 1
                else:
                    node_limit = int(subsample_rate * len(dataset))
                    subsample_index = np.sort(sample_graph_iid(adjacency_list, range(len(dataset)), node_limit))
                    subsample_multiplier = 1 / subsample_rate
                    nu = nu[subsample_index]

                if nu is None:
                    continue
                assert torch.isfinite(nu).all()
                assert torch.isfinite(Sigma_x_inv).all()
                eta = nu @ Sigma_x_inv
                logZ = integrate_of_exponential_over_simplex(eta)
                log_partition_function += subsample_multiplier * beta * logZ.sum()

            loss = (linear_term + regularization + log_partition_function) / weighted_total_cells

            if loss < loss_best:
                Sigma_x_inv_best = Sigma_x_inv.clone().detach()
                loss_best = loss.item()
                epoch_best = epoch

            loss.backward()
            Sigma_x_inv.grad = (Sigma_x_inv.grad + Sigma_x_inv.grad.T) / 2
            optimizer.step()
            with torch.no_grad():
                if self.spatial_affinity_centering:
                    Sigma_x_inv -= Sigma_x_inv.mean()

                if self.spatial_affinity_constraint == "clamp":
                    Sigma_x_inv.clamp_(
                        min=-self.spatial_affinity_state.scaling,
                        max=self.spatial_affinity_state.scaling,
                    )
                elif self.spatial_affinity_constraint == "scale":
                    Sigma_x_inv.mul_(self.spatial_affinity_state.scaling / Sigma_x_inv.abs().max())

                if epoch % check_frequency == 0:
                    loss = loss.item()
                    dloss = loss_prev - loss
                    loss_prev = loss
                    regularization_prev = regularization.item()
                    log_partition_function_prev = log_partition_function.item()
                    linear_term_prev = linear_term.item()

                    history.append((Sigma_x_inv.detach().cpu().numpy(), loss))

                    dSigma_x_inv = Sigma_x_inv_prev.sub(Sigma_x_inv).abs().max().item()
                    Sigma_x_inv_prev = Sigma_x_inv.clone().detach()

                    description = (
                        f"Updating Σx-1: loss = {dloss:.1e} -> {loss:.1e} "
                        f"δΣx-1 = {dSigma_x_inv:.1e} "
                        f"Σx-1 range = {Sigma_x_inv.min().item():.1e} ~ {Sigma_x_inv.max().item():.1e}"
                    )

                    verbose_description = (
                        f"Spatial affinity average: {Sigma_x_inv.mean().item():.1e} "
                        f"Total spatial affinity loss: {loss:.1e} "
                        f"spatial affinity linear term {linear_term:.6e} "
                        f"spatial affinity regularization {regularization.item():.1e} "
                        f"spatial affinity log_partition_function {log_partition_function:.1e} "
                    )

                    verbose_bar.set_description_str(verbose_description)
                    progress_bar.set_description(description)

                    if dSigma_x_inv < tol * check_frequency or epoch > epoch_best + 2 * check_frequency:
                        break

        verbose_bar.close()
        progress_bar.close()

        # with torch.no_grad():
        #     offset = -Sigma_x_inv.mean()
        #     Sigma_x_inv += offset
        #     print(f"Offset = {offset}")
        #     # Compute loss
        #     linear_term = Sigma_x_inv.view(-1) @ linear_term_coefficient.view(-1)
        #     regularization = torch.zeros(1, **self.context)
        #     if Sigma_x_inv_bar is not None:
        #         group_weighting = 1 / len(Sigma_x_inv_bar)
        #         for group_Sigma_x_inv_bar in Sigma_x_inv_bar:
        #             regularization += group_weighting * self.lambda_Sigma_bar * (group_Sigma_x_inv_bar - Sigma_x_inv).pow(2).sum() * weighted_total_cells / 2

        #     regularization += self.lambda_Sigma_x_inv * Sigma_x_inv.pow(self.spatial_affinity_regularization_power).sum() * weighted_total_cells / 2
        #
        #     log_partition_function = 0
        #     for nu, beta in zip(nus, self.betas):
        #         if subsample_rate is None:
        #             subsample_index = np.arange(len(dataset))
        #             subsample_multiplier = 1
        #         else:
        #             node_limit = int(subsample_rate * len(dataset))
        #             subsample_index = np.sort(sample_graph_iid(adjacency_list, range(len(dataset)), node_limit))
        #             subsample_multiplier = 1 / subsample_rate
        #             nu = nu[subsample_index]

        #         if nu is None:
        #             continue
        #         assert torch.isfinite(nu).all()
        #         assert torch.isfinite(Sigma_x_inv).all()
        #         eta = nu @ Sigma_x_inv
        #         logZ = integrate_of_exponential_over_simplex(eta)
        #         log_partition_function += subsample_multiplier * beta * logZ.sum()

        #     loss = (linear_term + regularization + log_partition_function) / weighted_total_cells

        #     print(f"Previous loss: total-{loss_prev}, regularization-{regularization_prev}, linear_term-{linear_term_prev}, log_partition_function-{log_partition_function_prev}")
        #     print(f"Loss after adding large offset: total-{loss.item()}, regularization-{regularization.item()}, linear_term-{linear_term.item()}, log_partition_function-{log_partition_function.item()}")
        #     2/0

        Sigma_x_inv = Sigma_x_inv_best
        Sigma_x_inv.requires_grad_(False)

        return Sigma_x_inv, loss * weighted_total_cells

    def nll_Sigma_x_inv(self, Sigma_x_inv, replicate_mask, Sigma_x_inv_bar=None):
        datasets = [dataset for (use_replicate, dataset) in zip(replicate_mask, self.datasets) if use_replicate]
        betas = [beta for (use_replicate, beta) in zip(replicate_mask, self.betas) if use_replicate]
        betas = np.array(betas) / np.sum(betas)

        Xs = [self.embedding_optimizer.embedding_state[dataset.name] for dataset in datasets]
        spatial_flags = ["adjacency_list" in dataset.obsm for dataset in datasets]

        num_edges_per_fov = [sum(map(len, dataset.obsm["adjacency_list"])) for dataset in datasets]

        if not any(
            sum(map(len, dataset.obsm["adjacency_list"])) > 0 and u for dataset, u in zip(datasets, spatial_flags)
        ):
            return

        linear_term_coefficient = torch.zeros_like(Sigma_x_inv).requires_grad_(False)
        size_factors = [torch.linalg.norm(X, axis=1, ord=1, keepdim=True) for X in Xs]
        Zs = [X.to(self.context["device"]) / size_factor for X, size_factor in zip(Xs, size_factors)]
        nus = []  # sum of neighbors' z
        weighted_total_cells = 0

        for Z, dataset, use_spatial, beta in zip(Zs, datasets, spatial_flags, self.betas):
            adjacency_list = self.adjacency_lists[dataset.name]
            adjacency_matrix = self.adjacency_matrices[dataset.name]

            if use_spatial:
                nu = adjacency_matrix @ Z
                linear_term_coefficient.addmm_(Z.T, nu, alpha=beta)
            else:
                nu = None

            nus.append(nu)
            weighted_total_cells += beta * sum(map(len, adjacency_list))
            del Z, adjacency_matrix

        loss_prev, loss = np.inf, np.nan

        linear_term = Sigma_x_inv.view(-1) @ linear_term_coefficient.view(-1)
        regularization = torch.zeros(1, **self.context)
        if Sigma_x_inv_bar is not None:
            group_weighting = 1 / len(Sigma_x_inv_bar)
            for group_Sigma_x_inv_bar in Sigma_x_inv_bar:
                regularization += (
                    group_weighting
                    * self.lambda_Sigma_bar
                    * (group_Sigma_x_inv_bar - Sigma_x_inv).pow(2).sum()
                    * weighted_total_cells
                    / 2
                )

        regularization += (
            self.lambda_Sigma_x_inv
            * Sigma_x_inv.pow(self.spatial_affinity_regularization_power).sum()
            * weighted_total_cells
            / 2
        )

        log_partition_function = 0
        for nu, beta in zip(nus, self.betas):
            if nu is None:
                continue
            assert torch.isfinite(nu).all()
            assert torch.isfinite(Sigma_x_inv).all()
            eta = nu @ Sigma_x_inv
            logZ = integrate_of_exponential_over_simplex(eta)
            log_partition_function += beta * logZ.sum()

        loss = (linear_term + regularization + log_partition_function) / weighted_total_cells

        return loss

    def update_spatial_affinity(self, differentiate_spatial_affinities=True, **optimization_kwargs):
        if self.spatial_affinity_mode == "shared lookup":
            for group_name, group_replicates in self.spatial_affinity_groups.items():
                replicate_mask = [dataset.name in group_replicates for dataset in self.datasets]
                first_dataset_name = group_replicates[0]
                Sigma_x_inv = self.spatial_affinity_state[first_dataset_name].to(self.context["device"])
                optimizer = self.spatial_affinity_state.optimizers[group_name]
                Sigma_x_inv, loss = self.estimate_Sigma_x_inv(
                    Sigma_x_inv,
                    replicate_mask,
                    optimizer,
                    tol=self.spatial_affinity_tol,
                    **optimization_kwargs,
                )
                with torch.no_grad():
                    self.spatial_affinity_state[first_dataset_name][:] = Sigma_x_inv

        elif self.spatial_affinity_mode == "differential lookup":
            for dataset_index, dataset in enumerate(self.datasets):
                if differentiate_spatial_affinities:
                    spatial_affinity_bars = [
                        self.spatial_affinity_state.spatial_affinity_bar[group_name].detach()
                        for group_name in self.spatial_affinity_tags[dataset.name]
                    ]
                else:
                    spatial_affinity_bars = None

                replicate_mask = [False] * len(self.datasets)
                replicate_mask[dataset_index] = True
                Sigma_x_inv = self.spatial_affinity_state[dataset.name].to(self.context["device"])
                optimizer = self.spatial_affinity_state.optimizers[dataset.name]
                Sigma_x_inv, loss = self.estimate_Sigma_x_inv(
                    Sigma_x_inv,
                    replicate_mask,
                    optimizer,
                    Sigma_x_inv_bar=spatial_affinity_bars,
                    tol=self.spatial_affinity_tol,
                    **optimization_kwargs,
                )
                # K_options, group_options = np.meshgrid()
                # runs =
                with torch.no_grad():
                    self.spatial_affinity_state[dataset.name][:] = Sigma_x_inv

            self.spatial_affinity_state.reaverage()

    def reinitialize_spatial_affinities(self):
        pretrained_embeddings = [
            self.embedding_optimizer.embedding_state[dataset.name].clone() for dataset in self.datasets
        ]
        self.spatial_affinity_state.initialize(pretrained_embeddings)

        # TODO: add code to synchronize new spatial affinities after reinitialization update here

    def nll_spatial_affinities(self):
        with torch.no_grad():
            loss_spatial_affinities = torch.zeros(1, **self.context)
            if self.spatial_affinity_mode == "shared lookup":
                for group_name, group_replicates in self.spatial_affinity_groups.items():
                    replicate_mask = [dataset.name in group_replicates for dataset in self.datasets]
                    first_dataset_name = group_replicates[0]
                    Sigma_x_inv = self.spatial_affinity_state[first_dataset_name].to(self.context["device"])
                    loss_Sigma_x_inv = self.nll_Sigma_x_inv(Sigma_x_inv, replicate_mask)
                    loss_spatial_affinities += loss_Sigma_x_inv

            elif self.spatial_affinity_mode == "differential lookup":
                for dataset_index, dataset in enumerate(self.datasets):
                    spatial_affinity_bars = [
                        self.spatial_affinity_state.spatial_affinity_bar[group_name].detach()
                        for group_name in self.spatial_affinity_tags[dataset.name]
                    ]

                    replicate_mask = [False] * len(self.datasets)
                    replicate_mask[dataset_index] = True
                    Sigma_x_inv = self.spatial_affinity_state[dataset.name].to(self.context["device"])
                    loss_Sigma_x_inv = self.nll_Sigma_x_inv(
                        Sigma_x_inv,
                        replicate_mask,
                        Sigma_x_inv_bar=spatial_affinity_bars,
                    )
                    loss_spatial_affinities += loss_Sigma_x_inv

        return loss_spatial_affinities.cpu().numpy()

    def update_metagenes(self, differentiate_metagenes=True, simplex_projection_mode="exact"):
        if self.metagene_mode == "shared":
            for group_name, group_replicates in self.metagene_groups.items():
                first_dataset_name = group_replicates[0]
                replicate_mask = [dataset.name in group_replicates for dataset in self.datasets]
                M = self.metagene_state[first_dataset_name]
                updated_M = self.estimate_M(M, replicate_mask, simplex_projection_mode=simplex_projection_mode)
                for dataset_name in group_replicates:
                    self.metagene_state[dataset_name][:] = updated_M

        elif self.metagene_mode == "differential":
            for dataset_index, dataset in enumerate(self.datasets):
                if differentiate_metagenes:
                    M_bars = [self.metagene_state.M_bar[group_name] for group_name in self.metagene_tags[dataset.name]]
                else:
                    M_bars = None

                M = self.metagene_state[dataset.name]
                replicate_mask = [False] * len(self.datasets)
                replicate_mask[dataset_index] = True
                self.metagene_state[dataset.name][:] = self.estimate_M(
                    M,
                    replicate_mask,
                    M_bar=M_bars,
                    simplex_projection_mode=simplex_projection_mode,
                )

            self.metagene_state.reaverage()

    def nll_metagenes(self):
        with torch.no_grad():
            loss_metagenes = torch.zeros(1, **self.context)
            if self.metagene_mode == "shared":
                for group_name, group_replicates in self.metagene_groups.items():
                    first_dataset_name = group_replicates[0]
                    replicate_mask = [dataset.name in group_replicates for dataset in self.datasets]
                    M = self.metagene_state[first_dataset_name]
                    loss_M = self.nll_M(M, replicate_mask)
                    loss_metagenes += loss_M

            elif self.metagene_mode == "differential":
                for dataset_index, dataset in enumerate(self.datasets):
                    M_bars = [self.metagene_state.M_bar[group_name] for group_name in self.metagene_tags[dataset.name]]

                    M = self.metagene_state[dataset.name]
                    replicate_mask = [False] * len(self.datasets)
                    replicate_mask[dataset_index] = True
                    loss_M = self.nll_M(M, replicate_mask, M_bar=M_bars)
                    loss_metagenes += loss_M

        return loss_metagenes.cpu().numpy()

    def nll_M(self, M, replicate_mask, M_bar=None):
        _, K = M.shape
        quadratic_factor = torch.zeros([K, K], **self.context)
        linear_term = torch.zeros_like(M)

        datasets = [dataset for (use_replicate, dataset) in zip(replicate_mask, self.datasets) if use_replicate]
        Xs = [self.embedding_optimizer.embedding_state[dataset.name] for dataset in datasets]
        Ys = [Y for (use_replicate, Y) in zip(replicate_mask, self.Ys) if use_replicate]
        sigma_yxs = self.sigma_yxs[replicate_mask]

        betas = self.betas[replicate_mask]
        betas /= betas.sum()

        scaled_betas = betas / (sigma_yxs**2)

        # ||Y||_2^2
        constant_magnitude = np.array([torch.square(Y).sum().cpu() for Y in Ys]).sum()

        constant = (
            np.array(
                [
                    torch.linalg.norm(self.embedding_optimizer.embedding_state[dataset.name]).item() ** 2
                    for dataset in datasets
                ],
            )
            * scaled_betas
        ).sum()

        regularization = [self.prior_xs[dataset_index] for dataset_index, dataset in enumerate(datasets)]
        for dataset, X, Y, sigma_yx, scaled_beta in zip(datasets, Xs, Ys, sigma_yxs, scaled_betas):
            # X_c^TX_c
            quadratic_factor.addmm_(X.T, X, alpha=scaled_beta)
            # MX_c^TY_c
            linear_term.addmm_(Y.T, X, alpha=scaled_beta)

        differential_regularization_quadratic_factor = torch.zeros((K, K), **self.context)
        differential_regularization_linear_term = torch.zeros(1, **self.context)
        if self.lambda_M > 0 and M_bar is not None:
            differential_regularization_quadratic_factor = self.lambda_M * torch.eye(K, **self.context)

            differential_regularization_linear_term = torch.zeros_like(M, **self.context)
            group_weighting = 1 / len(M_bar)
            for group_M_bar in M_bar:
                differential_regularization_linear_term += group_weighting * self.lambda_M * group_M_bar

        def compute_loss(M):
            quadratic_factor_grad = M @ (quadratic_factor + differential_regularization_quadratic_factor)
            loss = (quadratic_factor_grad * M).sum()
            linear_term_grad = linear_term + differential_regularization_linear_term
            loss -= 2 * (linear_term_grad * M).sum()

            loss += constant

            if self.metagene_mode == "differential" and M_bar is not None:
                differential_regularization_term = (M @ differential_regularization_quadratic_factor * M).sum() - 2 * (
                    differential_regularization_linear_term * M
                ).sum()
                group_weighting = 1 / len(M_bar)
                for group_M_bar in M_bar:
                    differential_regularization_term += (
                        group_weighting * self.lambda_M * (group_M_bar * group_M_bar).sum()
                    )

            loss /= 2

            return loss.item()

        loss = compute_loss(M)

        return loss

    def estimate_M(
        self,
        M,
        replicate_mask,
        M_bar=None,
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

        Each replicate may have a slightly different M
        min || Y - X MT ||_2^2 / (2 σ_yx^2) + || M - M_bar ||_2^2 λ_M / 2
        s.t. || Mk ||_p = 1
        grad = ( M XT X - YT X ) / ( σ_yx^2 ) + λ_M ( M - M_bar )

        Args:
            M: current estimate of metagene parameters
            betas: weight of each FOV in optimization scheme
            context: context ith which to create PyTorch tensor
            n_epochs: number of epochs

        Returns:
            Updated estimate of metagene parameters.

        """

        G, K = M.shape
        quadratic_factor = torch.zeros([K, K], **self.context)
        linear_factor = torch.zeros_like(M)
        # TODO: replace below (and any reference to dataset)

        tol /= G

        datasets = [dataset for (use_replicate, dataset) in zip(replicate_mask, self.datasets) if use_replicate]
        Xs = [self.embedding_optimizer.embedding_state[dataset.name] for dataset in datasets]
        Ys = [Y for (use_replicate, Y) in zip(replicate_mask, self.Ys) if use_replicate]
        sigma_yxs = self.sigma_yxs[replicate_mask]

        betas = self.betas[replicate_mask]
        betas /= betas.sum()

        scaled_betas = betas / (sigma_yxs**2)

        # ||Y||_2^2
        constant = np.array(
            [torch.square(Y).sum().cpu() / (sigma_yx**2) for Y, sigma_yx in zip(Ys, sigma_yxs)],
        ).sum()
        # constant_magnitude = np.array([torch.linalg.norm(Y).item()**2 for Y in Ys]).sum()

        # constant = (np.array([torch.linalg.norm(self.embedding_optimizer.embedding_state[dataset.name]).item()**2 for dataset in datasets]) * scaled_betas).sum()
        if self.verbose > 1:
            print(f"M constant: {constant: .1e}")
            # print(f"M constant magnitude: {constant_magnitude:.1e}")

        regularization = [self.prior_xs[dataset_index] for dataset_index, dataset in enumerate(datasets)]
        for dataset, X, Y, scaled_beta in zip(datasets, Xs, Ys, scaled_betas):
            # X_c^TX_c
            quadratic_factor.addmm_(X.T, X, alpha=scaled_beta)
            # MX_c^TY_c
            linear_factor.addmm_(Y.T, X, alpha=scaled_beta)

        # if self.lambda_M > 0 and M_bar is not None:
        #     quadratic_factor.diagonal().add_(self.lambda_M)
        #     linear_factor += self.lambda_M * M_bar
        differential_regularization_quadratic_factor = torch.zeros((K, K), **self.context)
        differential_regularization_linear_factor = torch.zeros(1, **self.context)
        if self.lambda_M > 0 and M_bar is not None:
            differential_regularization_quadratic_factor = self.lambda_M * torch.eye(K, **self.context)

            differential_regularization_linear_factor = torch.zeros_like(M, **self.context)
            group_weighting = 1 / len(M_bar)
            for group_M_bar in M_bar:
                differential_regularization_linear_factor += group_weighting * self.lambda_M * group_M_bar
        #     quadratic_factor.diagonal().add_(self.lambda_M)
        #     linear_factor += self.lambda_M * M_bar

        if self.verbose > 1:
            print(
                f"{get_datetime()} Eigenvalue difference: {torch.max(torch.linalg.eigvals(quadratic_factor + differential_regularization_quadratic_factor).abs()) - torch.max(torch.linalg.eigvals(quadratic_factor).abs())}",
            )
            print(f"M linear term: {torch.linalg.norm(linear_factor)}")
            print(f"M regularization linear term: {torch.linalg.norm(differential_regularization_linear_factor)}")
            print(
                f"M linear regularization term ratio: {torch.linalg.norm(differential_regularization_linear_factor) / torch.linalg.norm(linear_factor)}",
            )
        loss_prev, loss = np.inf, np.nan

        verbose_bar = tqdm(disable=not (self.verbose > 2), bar_format="{desc}{postfix}")
        progress_bar = trange(n_epochs, leave=True, disable=not self.verbose, desc="Updating M", miniters=1000)

        def compute_loss_and_gradient(M):
            quadratic_factor_grad = M @ (quadratic_factor + differential_regularization_quadratic_factor)
            loss = (quadratic_factor_grad * M).sum()
            verbose_description = ""
            if self.verbose > 2:
                verbose_description += f"M quadratic term: {loss:.1e}"
            linear_term_grad = linear_factor + differential_regularization_linear_factor
            loss -= 2 * (linear_term_grad * M).sum()
            grad = quadratic_factor_grad - linear_term_grad

            loss += constant

            if self.metagene_mode == "differential" and M_bar is not None:
                differential_regularization_term = (M @ differential_regularization_quadratic_factor * M).sum() - 2 * (
                    differential_regularization_linear_factor * M
                ).sum()
                group_weighting = 1 / len(M_bar)
                for group_M_bar in M_bar:
                    differential_regularization_term += (
                        group_weighting * self.lambda_M * (group_M_bar * group_M_bar).sum()
                    )

            if self.verbose > 2:
                # print(f"M regularization term: {regularization_term}")
                if self.metagene_mode == "differential":
                    verbose_description += f"M differential regularization term: {differential_regularization_term}"

            loss /= 2

            if self.M_constraint == "simplex":
                grad.sub_(grad.sum(0, keepdim=True))

            return loss.item(), grad

        def estimate_M_nag(M):
            """Estimate M using Nesterov accelerated gradient descent.

            Args:
                M (torch.Tensor) : current estimate of meteagene parameters

            """
            loss, grad = compute_loss_and_gradient(M)
            if self.verbose > 1:
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
                if simplex_projection_mode == "exact":
                    if self.use_inplace_ops:
                        M = project_M_(M, self.M_constraint)
                    else:
                        M = project_M(M, self.M_constraint)
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

            loss, grad = compute_loss_and_gradient(M)
            if self.verbose > 1:
                print(f"M NAG Final Loss: {loss}")

            return M

        if backend_algorithm == "mu":
            for epoch in progress_bar:
                loss = (((M @ quadratic_factor) * M).sum() - 2 * (M * linear_term).sum() + constant) / 2
                loss = loss.item()
                numerator = linear_term
                denominator = M @ quadratic_factor
                multiplicative_factor = numerator / denominator

                M_prev = M.clone()
                # multiplicative_factor.clip_(max=10)
                M *= multiplicative_factor
                if simplex_projection_mode == "exact":
                    if self.use_inplace_ops:
                        M = project_M_(M, self.M_constraint)
                    else:
                        M = project_M(M, self.M_constraint)
                elif simplex_projection_mode == "approximate":
                    pass
                dM = M_prev.sub(M).abs_().max().item()

                stop_criterion = dM < tol and epoch > 5
                if epoch % 1000 == 0 or stop_criterion:
                    progress_bar.set_description(
                        f"Updating M: loss = {loss:.1e}, "
                        f"%δloss = {(loss_prev - loss) / loss:.1e}, "
                        f"δM = {dM:.1e}",
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
                    if self.use_inplace_ops:
                        M = project_M_(M_new, self.M_constraint)
                    else:
                        M = project_M(M_new, self.M_constraint)
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
            M = estimate_M_nag(M)
        else:
            raise NotImplementedError

        return M

    def update_sigma_yx(self):
        """Update sigma_yx for each replicate."""

        # print((self.Ys[0]).is_sparse)
        # print((self.embedding_optimizer.embedding_state[self.datasets[0].name]).is_sparse)
        # print((self.metagene_state[self.datasets[0].name].T).is_sparse)
        # squared_loss = np.zeros(len(self.datasets))
        # for index, (Y, dataset) in enumerate(zip(self.Ys, self.datasets)):
        #     result = torch.square(-(self.embedding_optimizer.embedding_state[dataset.name] @ self.metagene_state[dataset.name].T) + Y).sum()
        #     print(result)
        #     2/0
        #     squared_loss[index] = result
        squared_terms = [
            torch.addmm(
                Y.to_dense(),
                self.embedding_optimizer.embedding_state[dataset.name],
                self.metagene_state[dataset.name].T,
                alpha=-1,
            )
            for Y, dataset in zip(self.Ys, self.datasets)
        ]
        squared_loss = np.array(
            [torch.linalg.norm(squared_term, ord="fro").item() ** 2 for squared_term in squared_terms],
        )
        num_replicates = len(self.datasets)
        sizes = np.array([dataset.X.size for dataset in self.datasets])
        if self.sigma_yx_inv_mode == "separate":
            self.sigma_yxs[:] = np.sqrt(squared_loss / sizes)
        elif self.sigma_yx_inv_mode == "average":
            sigma_yx = np.sqrt(np.dot(self.betas, squared_loss) / np.dot(self.betas, sizes))
            self.sigma_yxs[:] = np.full(num_replicates, float(sigma_yx))
        else:
            raise NotImplementedError

    def nll_sigma_yx(self):
        with torch.no_grad():
            squared_terms = [
                torch.addmm(
                    Y.to_dense(),
                    self.embedding_optimizer.embedding_state[dataset.name],
                    self.metagene_state[dataset.name].T,
                    alpha=-1,
                )
                for Y, dataset in zip(self.Ys, self.datasets)
            ]
            squared_loss = np.array(
                [torch.linalg.norm(squared_term, ord="fro").item() ** 2 for squared_term in squared_terms],
            )

        return squared_loss.sum()


class MetageneState(dict):
    """State to store metagene parameters during Popari optimization.

    Metagene state can be shared across replicates or maintained separately for each replicate.

    Attributes:
        datasets: A reference to the list of PopariDatasets that are being optimized.
        context: Parameters to define the context for PyTorch tensor instantiation.
        metagenes: A PyTorch tensor containing all metagene parameters.

    """

    def __init__(
        self,
        K,
        datasets,
        groups,
        tags,
        mode="shared",
        M_constraint="simplex",
        initial_context=None,
        context=None,
    ):
        self.datasets = datasets
        self.groups = groups
        self.tags = tags
        self.initial_context = initial_context if initial_context else {"device": "cpu", "dtype": torch.float32}
        self.context = context if context else {"device": "cpu", "dtype": torch.float32}
        self.M_constraint = M_constraint
        if mode == "shared":
            _, num_genes = self.datasets[0].shape
            self.metagenes = torch.zeros((len(self.groups), num_genes, K), **self.context)
            for (group_name, group_replicates), group_metagenes in zip(self.groups.items(), self.metagenes):
                for dataset_name in group_replicates:
                    self.__setitem__(dataset_name, group_metagenes)

        elif mode == "differential":
            _, num_genes = self.datasets[0].shape
            self.metagenes = torch.zeros((len(self.datasets), num_genes, K), **self.context)
            for dataset, replicate_metagenes in zip(self.datasets, self.metagenes):
                self.__setitem__(dataset.name, replicate_metagenes)

            self.M_bar = {}
            for group_name, group_replicates in self.groups.items():
                self.M_bar[group_name] = torch.zeros((num_genes, K), **self.context)
                for dataset_name in group_replicates:
                    self.M_bar[group_name] += self.__getitem__(dataset_name)
                self.M_bar[group_name].div_(len(group_replicates))

    def reaverage(self):
        # Set M_bar to average of self.Ms (memory efficient)
        for group_name, group_replicates in self.groups.items():
            self.M_bar[group_name].zero_()
            for dataset_name in group_replicates:
                self.M_bar[group_name].add_(self.__getitem__(dataset_name))
            self.M_bar[group_name].div_(len(group_replicates))
            self.M_bar[group_name][:] = project_M(self.M_bar[group_name], self.M_constraint)


class SpatialAffinityState(dict):
    def __init__(
        self,
        K,
        metagene_state,
        datasets,
        groups,
        tags,
        betas,
        scaling=10,
        lr=1e-3,
        mode="shared lookup",
        initial_context=None,
        context=None,
    ):
        self.datasets = datasets
        self.groups = groups
        self.tags = tags
        self.metagene_state = metagene_state
        self.K = K
        self.mode = mode
        self.initial_context = initial_context if initial_context else {"device": "cpu", "dtype": torch.float32}
        self.context = context if context else {"device": "cpu", "dtype": torch.float32}
        self.betas = betas
        self.scaling = scaling
        self.lr = lr
        self.optimizers = {}
        super().__init__()

        num_replicates = len(self.datasets)
        if mode == "shared lookup":
            metagene_affinities = torch.zeros((K, K), **self.initial_context)
            for dataset in self.datasets:
                self.__setitem__(dataset.name, metagene_affinities)

        elif mode == "differential lookup":
            for dataset_index, dataset in enumerate(self.datasets):
                metagene_affinity = torch.zeros((K, K), **self.initial_context)
                self.__setitem__(dataset.name, metagene_affinity)

            self.spatial_affinity_bar = {}
            for group_name in self.groups:
                self.spatial_affinity_bar[group_name] = torch.zeros((self.K, self.K), **self.context)

        elif mode == "attention":
            metagene_affinities = 0  # attention mechanism here
            for dataset_index, dataset in enumerate(self.datasets):
                self.__setitem__(dataset.name, metagene_affinities[dataset_index])
        else:
            raise NotImplementedError(f"{mode=} is not implemented.")

    def initialize(self, initial_embeddings):
        use_spatial_info = ["adjacency_list" in dataset.obsm for dataset in self.datasets]

        if not any(use_spatial_info):
            return

        num_replicates = len(self.datasets)
        Sigma_x_invs = torch.zeros([num_replicates, self.K, self.K], **self.initial_context)
        for replicate, (initial_embedding, is_spatial_replicate, dataset) in enumerate(
            zip(initial_embeddings, use_spatial_info, self.datasets),
        ):
            if not is_spatial_replicate:
                continue

            adjacency_list = dataset.obsm["adjacency_list"]
            X = initial_embedding
            Z = X / torch.linalg.norm(X, dim=1, keepdim=True, ord=1)
            edges = np.array([(i, j) for i, e in enumerate(adjacency_list) for j in e])

            x = Z[edges[:, 0]]
            y = Z[edges[:, 1]]
            x = x - x.mean(dim=0, keepdim=True)
            y = y - y.mean(dim=0, keepdim=True)
            y_std = y.std(dim=0, keepdim=True)
            x_std = x.std(dim=0, keepdim=True)
            corr = (y / y_std).T @ (x / x_std) / len(x)
            Sigma_x_invs[replicate] = -corr

        # Symmetrizing and zero-centering Sigma_x_inv
        Sigma_x_invs = (Sigma_x_invs + torch.transpose(Sigma_x_invs, 1, 2)) / 2
        Sigma_x_invs -= Sigma_x_invs.mean(dim=(1, 2), keepdims=True)
        Sigma_x_invs *= self.scaling

        self.initialize_optimizers(Sigma_x_invs)

    def initialize_optimizers(self, Sigma_x_invs):
        if self.mode == "shared lookup":
            for group_name, group_replicates in self.groups.items():
                first_dataset = self.datasets[0]
                shared_affinity = torch.zeros([self.K, self.K], **self.initial_context)
                for dataset_index, (beta, dataset) in enumerate(zip(self.betas, self.datasets)):
                    if dataset.name in group_replicates:
                        shared_affinity += beta * Sigma_x_invs[dataset_index]

                for dataset_index, dataset in enumerate(self.datasets):
                    if dataset.name in group_replicates:
                        self.__setitem__(dataset.name, shared_affinity)

                optimizer = torch.optim.Adam(
                    [shared_affinity],
                    lr=self.lr,
                    betas=(0.5, 0.9),
                )
                self.optimizers[group_name] = optimizer

        elif self.mode == "differential lookup":
            for group_name, group_replicates in self.groups.items():
                for dataset_index, dataset in enumerate(self.datasets):
                    if dataset.name in group_replicates:
                        self.spatial_affinity_bar[group_name] += Sigma_x_invs[dataset_index]

                self.spatial_affinity_bar[group_name].div_(len(group_replicates))

            for dataset_index, dataset in enumerate(self.datasets):
                differential_affinity = Sigma_x_invs[dataset_index]
                self.__setitem__(dataset.name, differential_affinity)
                optimizer = torch.optim.Adam(
                    [differential_affinity],
                    lr=self.lr,
                    betas=(0.5, 0.9),
                )
                self.optimizers[dataset.name] = optimizer

        elif self.mode == "attention":
            # TODO: initialize with gradient descent
            raise NotImplementedError()

    def reaverage(self):
        # Set spatial_affinity_bar to average of self.spatial_affinitys (memory efficient)
        for group_name, group_replicates in self.groups.items():
            self.spatial_affinity_bar[group_name].zero_()
            for dataset_name in group_replicates:
                self.spatial_affinity_bar[group_name].add_(self.__getitem__(dataset_name))
            self.spatial_affinity_bar[group_name].div_(len(group_replicates))
