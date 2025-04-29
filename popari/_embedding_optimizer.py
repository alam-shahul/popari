import logging
import time
from typing import Sequence

import faiss
import numpy as np
import torch
from tqdm.auto import tqdm, trange

from popari._embedding_optimizer_util import (
    BatchEffectEmbeddingLossWithNeighborsNesterov,
    BatchEffectTripletLossEmbeddingLossWithNeighborsNesterov,
    EmbeddingLossNoNeighborsGD,
    EmbeddingLossTripletLossNoNeighborsGD,
    EmbeddingLossWithNeighborsNesterov,
)
from popari._popari_dataset import PopariDataset
from popari.util import (
    IndependentSet,
    NesterovGD,
    convert_scipy_csr_to_pytorch_coo,
    get_datetime,
    project2simplex,
    project2simplex_,
    project_M,
    project_M_,
    sample_graph_iid,
)


class EmbeddingOptimizer:
    """Optimizer and state for Popari embeddings."""

    def __init__(
        self,
        K,
        Ys,
        datasets,
        initial_context=None,
        context=None,
        use_inplace_ops=False,
        embedding_step_size_multiplier=1,
        embedding_mini_iterations=1000,
        embedding_acceleration_trick=True,
        verbose=0,
        batch_effect_correction=None,
    ):
        self.batch_effect_correction = batch_effect_correction
        self.verbose = verbose
        self.use_inplace_ops = use_inplace_ops
        self.datasets = datasets
        self.hierarchical = False
        self.K = K
        self.Ys = Ys
        self.initial_context = initial_context if initial_context else {"device": "cpu", "dtype": torch.float32}
        self.context = context if context else {"device": "cpu", "dtype": torch.float32}
        self.adjacency_lists = {dataset.name: dataset.obsm["adjacency_list"] for dataset in self.datasets}
        self._adjacency_matrices = {}

        self.embedding_step_size_multiplier = embedding_step_size_multiplier
        self.embedding_mini_iterations = embedding_mini_iterations
        self.embedding_acceleration_trick = embedding_acceleration_trick

        if self.verbose:
            print(f"{get_datetime()} Initializing EmbeddingState")
        self.embedding_state = EmbeddingState(K, self.datasets, context=self.context)

    @property
    def adjacency_matrices(self):
        return self._adjacency_matrices

    @adjacency_matrices.setter
    def adjacency_matrices(self, val):
        self._adjacency_matrices = val

    def link(self, parameter_optimizer, batch_optimizer=None):
        self.parameter_optimizer = parameter_optimizer
        if batch_optimizer is not None:
            self.batch_optimizer = batch_optimizer

    def update_embeddings(self, use_neighbors=True):
        """Update Popari embeddings according to optimization scheme."""
        logging.info(f"{get_datetime()}Updating latent states")

        loss_list = []

        if self.parameter_optimizer.prior_x_modes[0] == "cross_dataset_average":
            # average_across_samples = []
            # for dataset in self.datasets:
            #     this_embeddings = self.embedding_state[dataset.name]

            #     reference_embeddings = []
            #     for other_dataset in self.datasets:
            #         #if other_dataset.name != dataset.name:
            #             reference_embeddings.append(self.embedding_state[other_dataset.name])

            #     reference_embeddings = torch.cat(reference_embeddings, dim=0) # (N_all, K)
            #     N_all = reference_embeddings.shape[0]

            #     query = this_embeddings.detach().cpu().numpy().astype('float32')
            #     reference = reference_embeddings.detach().cpu().numpy().astype('float32')

            #     index = faiss.IndexFlatL2(reference.shape[1])
            #     index.add(reference)

            #     _, indices = index.search(query, int(N_all * 0.1))  # (N, top_k)
            #     neighbor_embeds = reference_embeddings[torch.tensor(indices)]
            #     prior_embeddings = neighbor_embeds.mean(dim=1)  # (N, K)
            #     average_across_samples.append(prior_embeddings)

            average_across_samples = []
            for dataset in self.datasets:
                embeddings_list = []
                for other_dataset in self.datasets:
                    if other_dataset.name != dataset.name:
                        embeddings_list.append(self.embedding_state[dataset.name])
                stacked_embeddings = torch.cat(embeddings_list, dim=0)
                # stacked_embeddings = torch.stack(embeddings_list, dim=0)
                prior_embeddings = torch.mean(stacked_embeddings, dim=0)
                average_across_samples.append(prior_embeddings)

        for dataset_index, dataset in enumerate(self.datasets):
            is_spatial_replicate = "adjacency_list" in dataset.obsm
            sigma_yx = self.parameter_optimizer.sigma_yxs[dataset_index]
            Y = self.Ys[dataset_index].to(self.context["device"])
            X = self.embedding_state[dataset.name].to(self.context["device"])
            M = self.parameter_optimizer.metagene_state[dataset.name].to(self.context["device"])
            prior_x_mode = self.parameter_optimizer.prior_x_modes[dataset_index]
            prior_x = self.parameter_optimizer.prior_xs[dataset_index]
            if not is_spatial_replicate or not use_neighbors:
                if prior_x_mode == "cross_dataset_average":
                    loss, self.embedding_state[dataset.name][:] = self.estimate_weight_wonbr_cross(
                        Y,
                        M,
                        X,
                        sigma_yx,
                        prior_x_mode,
                        prior_x,
                        dataset,
                        average_across_samples[dataset_index],
                    )
                else:
                    loss, self.embedding_state[dataset.name][:] = self.estimate_weight_wonbr(
                        Y,
                        M,
                        X,
                        sigma_yx,
                        prior_x_mode,
                        prior_x,
                        dataset,
                    )
            elif self.batch_effect_correction is not None:
                if prior_x_mode == "cross_dataset_average":
                    B = self.batch_optimizer.batch_effect_state[dataset.name].to(self.context["device"])
                    loss, self.embedding_state[dataset.name][:] = self.estimate_weight_wnbr_batch_cross(
                        Y,
                        M,
                        B,
                        X,
                        sigma_yx,
                        prior_x_mode,
                        prior_x,
                        dataset,
                        average_across_samples[dataset_index],
                    )

                else:
                    B = self.batch_optimizer.batch_effect_state[dataset.name].to(self.context["device"])
                    loss, self.embedding_state[dataset.name][:] = self.estimate_weight_wnbr_batch(
                        Y,
                        M,
                        B,
                        X,
                        sigma_yx,
                        prior_x_mode,
                        prior_x,
                        dataset,
                    )
            else:
                loss, self.embedding_state[dataset.name][:] = self.estimate_weight_wnbr(
                    Y,
                    M,
                    X,
                    sigma_yx,
                    prior_x_mode,
                    prior_x,
                    dataset,
                )

            loss_list.append(loss)

    def nll_embeddings(self, use_neighbors=True):
        with torch.no_grad():
            loss_embeddings = torch.zeros(1, **self.context)
            for dataset_index, dataset in enumerate(self.datasets):
                is_spatial_replicate = "adjacency_list" in dataset.obsm
                sigma_yx = self.parameter_optimizer.sigma_yxs[dataset_index]
                Y = self.Ys[dataset_index].to(self.context["device"])
                X = self.embedding_state[dataset.name].to(self.context["device"])
                M = self.parameter_optimizer.metagene_state[dataset.name].to(self.context["device"])
                B = self.batch_optimizer.batch_effect_state[dataset.name].to(self.context["device"])
                prior_x_mode = self.parameter_optimizer.prior_x_modes[dataset_index]
                prior_x = self.parameter_optimizer.prior_xs[dataset_index]
                if not is_spatial_replicate or not use_neighbors:
                    loss = self.nll_weight_wonbr(
                        Y,
                        M,
                        B,
                        X,
                        sigma_yx,
                        prior_x_mode,
                        prior_x,
                        dataset,
                    )
                else:
                    loss = self.nll_weight_wnbr(
                        Y,
                        M,
                        B,
                        X,
                        sigma_yx,
                        prior_x_mode,
                        prior_x,
                        dataset,
                    )

                loss_embeddings += loss

        return loss_embeddings.cpu().numpy()

    @torch.no_grad()
    def estimate_weight_wonbr(
        self,
        Y,
        M,
        X,
        sigma_yx,
        prior_x_mode,
        prior_x,
        dataset,
        n_epochs=1000,
        tol=1e-6,
        update_alg="gd",
    ):
        """Estimate weights without spatial information - equivalent to vanilla NMF.

        Optimizes the follwing objective with respect to hidden state X:

        min 1/2σ^2 || Y - X MT ||_2^2 + lam || X ||_1
        grad = X MT M / σ^2 - Y MT / σ^2 + lam

        TODO: use (projected) Nesterov GD. not urgent

        Args:
            Y (torch.Tensor):
        """

        # Precomputing quantities
        MTM = M.T @ M / (sigma_yx**2)
        YM = Y @ M / (sigma_yx**2)
        Ynorm = torch.square(Y).sum() / (sigma_yx**2)
        step_size = 1 / torch.linalg.eigvalsh(MTM).max().item()
        loss_prev, loss = np.inf, np.nan

        # multiplicative_update = multiplicative_update_wonbr_closure(X_prev, X, MTM, clipped_X, YM, Ynorm, prior_x_mode, prior_x, loss_prev,)

        # gradient_update = gradient_update_wonbr_closure(X, MTM, YM, prior_x_mode, prior_x, Ynorm, step_size)
        updater = EmbeddingLossNoNeighborsGD(MTM, YM, Ynorm, prior_x_mode, prior_x, step_size)

        progress_bar = trange(n_epochs, leave=True, disable=not self.verbose, miniters=1000)
        for epoch in progress_bar:
            X_prev = X.clone()
            if update_alg == "mu":
                # TODO: it seems like loss might not always decrease...
                X.clip_(min=1e-10)
                loss = ((X @ MTM) * X).sum() / 2 - X.view(-1) @ YM.view(-1) + Ynorm / 2
                numerator = YM
                denominator = X @ MTM
                if prior_x_mode == "exponential shared fixed":
                    # see sklearn.decomposition.NMF
                    loss += (X @ prior_x[0]).sum()
                    denominator.add_(prior_x[0][None])
                elif not prior_x_mode:
                    pass
                else:
                    raise NotImplementedError

                loss = loss.item()
                assert loss <= loss_prev * (1 + 1e-4), (loss_prev, loss, (loss_prev - loss) / loss)
                multiplicative_factor = numerator / denominator
                X.mul_(multiplicative_factor).clip_(min=1e-10)

                # X, loss = multiplicative_update(X_prev)
            elif update_alg == "gd":
                X, loss = updater(X)
            else:
                raise NotImplementedError

            dX = torch.abs((X_prev - X) / torch.linalg.norm(X, dim=1, ord=1, keepdim=True)).max().item()
            do_stop = dX < tol
            progress_bar.set_description(
                f"Updating weight w/o nbrs: loss = {loss:.1e} "
                f"%δloss = {(loss_prev - loss) / loss:.1e} "
                f"%δX = {dX:.1e}",
            )
            loss_prev = loss
            if do_stop:
                break
        progress_bar.close()
        return loss, X

    @torch.no_grad()
    def estimate_weight_wonbr_cross(
        self,
        Y,
        M,
        X,
        sigma_yx,
        prior_x_mode,
        prior_x,
        dataset,
        average_across_samples,
        n_epochs=1000,
        tol=1e-6,
        update_alg="gd",
    ):
        """Estimate weights without spatial information - equivalent to vanilla NMF.

        Optimizes the follwing objective with respect to hidden state X:

        min 1/2σ^2 || Y - X MT ||_2^2 + lam || X ||_1
        grad = X MT M / σ^2 - Y MT / σ^2 + lam

        TODO: use (projected) Nesterov GD. not urgent

        Args:
            Y (torch.Tensor):
        """

        # Precomputing quantities
        MTM = M.T @ M / (sigma_yx**2)
        YM = Y @ M / (sigma_yx**2)
        Ynorm = torch.square(Y).sum() / (sigma_yx**2)
        step_size = 1 / torch.linalg.eigvalsh(MTM).max().item()
        loss_prev, loss = np.inf, np.nan

        # multiplicative_update = multiplicative_update_wonbr_closure(X_prev, X, MTM, clipped_X, YM, Ynorm, prior_x_mode, prior_x, loss_prev,)

        # gradient_update = gradient_update_wonbr_closure(X, MTM, YM, prior_x_mode, prior_x, Ynorm, step_size)

        updater = EmbeddingLossTripletLossNoNeighborsGD(
            MTM,
            YM,
            Ynorm,
            prior_x_mode,
            prior_x,
            step_size,
            average_across_samples,
        )

        progress_bar = trange(n_epochs, leave=True, disable=not self.verbose, miniters=1000)
        for epoch in progress_bar:
            X_prev = X.clone()
            if update_alg == "mu":
                # TODO: it seems like loss might not always decrease...
                X.clip_(min=1e-10)
                loss = ((X @ MTM) * X).sum() / 2 - X.view(-1) @ YM.view(-1) + Ynorm / 2
                numerator = YM
                denominator = X @ MTM
                if prior_x_mode == "exponential shared fixed":
                    # see sklearn.decomposition.NMF
                    loss += (X @ prior_x[0]).sum()
                    denominator.add_(prior_x[0][None])
                elif not prior_x_mode:
                    pass
                else:
                    raise NotImplementedError

                loss = loss.item()
                assert loss <= loss_prev * (1 + 1e-4), (loss_prev, loss, (loss_prev - loss) / loss)
                multiplicative_factor = numerator / denominator
                X.mul_(multiplicative_factor).clip_(min=1e-10)

                # X, loss = multiplicative_update(X_prev)
            elif update_alg == "gd":
                X, loss = updater(X)
            else:
                raise NotImplementedError

            dX = torch.abs((X_prev - X) / torch.linalg.norm(X, dim=1, ord=1, keepdim=True)).max().item()
            do_stop = dX < tol
            progress_bar.set_description(
                f"Updating weight w/o nbrs: loss = {loss:.1e} "
                f"%δloss = {(loss_prev - loss) / loss:.1e} "
                f"%δX = {dX:.1e}",
            )
            loss_prev = loss
            if do_stop:
                break
        progress_bar.close()
        return loss, X

    @torch.no_grad()
    def nll_weight_wonbr(self, Y, M, B, X, sigma_yx, prior_x_mode, prior_x, dataset):
        # Precomputing quantities
        MTM = M.T @ M / (sigma_yx**2)
        YM = Y @ M / (sigma_yx**2)
        Ynorm = torch.square(Y).sum() / (sigma_yx**2)
        step_size = 1 / torch.linalg.eigvalsh(MTM).max().item()
        loss_prev, loss = np.inf, np.nan

        loss = ((X @ MTM) * X).sum() / 2 - X.view(-1) @ YM.view(-1) + Ynorm / 2

        return loss

    @torch.no_grad()
    def estimate_weight_wnbr(
        self,
        Y,
        M,
        X,
        sigma_yx,
        prior_x_mode,
        prior_x,
        dataset,
        tol=1e-5,
        update_alg="nesterov",
    ):
        """Estimate updated weights taking neighbor-neighbor interactions into
        account.

        The optimization for all variables
        min 1/2σ^2 || Y - diag(S) Z MT ||_2^2 + lam || S ||_1 + sum_{ij in E} ziT Σx-1 zj

        for s_i
        min 1/2σ^2 || y - M z s ||_2^2 + lam s
        s* = max(0, ( yT M z / σ^2 - lam ) / ( zT MT M z / σ^2) )

        for Z
        min 1/2σ^2 || Y - diag(S) Z MT ||_2^2 + sum_{ij in E} ziT Σx-1 zj
        grad_i = MT M z s^2 / σ^2 - MT y s / σ^2 + sum_{j in Ei} Σx-1 zj

        TODO: Try projected Newton's method.
        TM: Inverse is precomputed once, and projection is cheap. Not sure if it works theoretically

        """
        # Precomputing quantities
        MTM = M.T @ M / (sigma_yx**2)
        YM = Y.to(M.device) @ M / (sigma_yx**2)
        Ynorm = torch.square(Y).sum() / (sigma_yx**2)
        base_step_size = self.embedding_step_size_multiplier / torch.linalg.eigvalsh(MTM).max().item()
        S = torch.linalg.norm(X, dim=1, ord=1, keepdim=True)

        if self.verbose > 3:
            print(f"S max: {S.max()}")
            print(f"S min: {S.min()}")

        Z = X / S

        E_adjacency_list = self.adjacency_lists[dataset.name]
        adjacency_matrix = self.adjacency_matrices[dataset.name].to(self.context["device"])
        Sigma_x_inv = self.parameter_optimizer.spatial_affinity_state[dataset.name].to(self.context["device"])

        # TM: consider combine compute_loss and update_z to remove a call to torch.sparse.mm
        # TM: the above idea is not practical if we update only a subset of nodes each time

        embedding_updater = EmbeddingLossWithNeighborsNesterov(
            Z,
            S,
            MTM,
            YM,
            Ynorm,
            adjacency_matrix,
            prior_x_mode,
            prior_x,
            Sigma_x_inv,
            E_adjacency_list,
            self.context["device"],
            base_step_size,
            self.verbose,
            self.embedding_acceleration_trick,
            self.use_inplace_ops,
            self.embedding_mini_iterations,
            tol,
        )

        loss, X = embedding_updater()

        return loss, X

    @torch.no_grad()
    def estimate_weight_wnbr_batch(
        self,
        Y,
        M,
        B,
        X,
        sigma_yx,
        prior_x_mode,
        prior_x,
        dataset,
        tol=1e-5,
        update_alg="nesterov",
    ):
        """Estimate updated weights taking neighbor-neighbor interactions into
        account.

        The optimization for all variables
        min 1/2σ^2 || Y - diag(S) Z MT ||_2^2 + lam || S ||_1 + sum_{ij in E} ziT Σx-1 zj

        for s_i
        min 1/2σ^2 || y - M z s ||_2^2 + lam s
        s* = max(0, ( yT M z / σ^2 - lam ) / ( zT MT M z / σ^2) )

        for Z
        min 1/2σ^2 || Y - diag(S) Z MT ||_2^2 + sum_{ij in E} ziT Σx-1 zj
        grad_i = MT M z s^2 / σ^2 - MT y s / σ^2 + sum_{j in Ei} Σx-1 zj

        TODO: Try projected Newton's method.
        TM: Inverse is precomputed once, and projection is cheap. Not sure if it works theoretically

        """

        N, K = X.size()

        # Precomputing quantities
        MTM = M.T @ M / (sigma_yx**2)
        YM = Y.to(M.device) @ M / (sigma_yx**2)
        Ynorm = torch.square(Y).sum() / (sigma_yx**2)
        base_step_size = self.embedding_step_size_multiplier / torch.linalg.eigvalsh(MTM).max().item()
        S = torch.linalg.norm(X, dim=1, ord=1, keepdim=True)

        if self.verbose > 3:
            print(f"S max: {S.max()}")
            print(f"S min: {S.min()}")

        Z = X / S

        E_adjacency_list = self.adjacency_lists[dataset.name]
        adjacency_matrix = self.adjacency_matrices[dataset.name].to(self.context["device"])
        Sigma_x_inv = self.parameter_optimizer.spatial_affinity_state[dataset.name].to(self.context["device"])

        embedding_updater = BatchEffectEmbeddingLossWithNeighborsNesterov(
            Z,
            S,
            B,
            MTM,
            YM,
            Ynorm,
            adjacency_matrix,
            prior_x_mode,
            prior_x,
            Sigma_x_inv,
            E_adjacency_list,
            self.context["device"],
            base_step_size,
            self.verbose,
            self.embedding_acceleration_trick,
            self.use_inplace_ops,
            self.embedding_mini_iterations,
            tol,
        )

        loss, X = embedding_updater()

        return loss, X

    @torch.no_grad()
    def estimate_weight_wnbr_batch_cross(
        self,
        Y,
        M,
        B,
        X,
        sigma_yx,
        prior_x_mode,
        prior_x,
        dataset,
        average_across_samples,
        tol=1e-5,
        update_alg="nesterov",
    ):
        """Estimate updated weights taking neighbor-neighbor interactions into
        account.

        The optimization for all variables
        min 1/2σ^2 || Y - diag(S) Z MT ||_2^2 + lam || S ||_1 + sum_{ij in E} ziT Σx-1 zj

        for s_i
        min 1/2σ^2 || y - M z s ||_2^2 + lam s
        s* = max(0, ( yT M z / σ^2 - lam ) / ( zT MT M z / σ^2) )

        for Z
        min 1/2σ^2 || Y - diag(S) Z MT ||_2^2 + sum_{ij in E} ziT Σx-1 zj
        grad_i = MT M z s^2 / σ^2 - MT y s / σ^2 + sum_{j in Ei} Σx-1 zj

        TODO: Try projected Newton's method.
        TM: Inverse is precomputed once, and projection is cheap. Not sure if it works theoretically

        """

        N, K = X.size()

        # Precomputing quantities
        MTM = M.T @ M / (sigma_yx**2)
        YM = Y.to(M.device) @ M / (sigma_yx**2)
        Ynorm = torch.square(Y).sum() / (sigma_yx**2)
        base_step_size = self.embedding_step_size_multiplier / torch.linalg.eigvalsh(MTM).max().item()
        S = torch.linalg.norm(X, dim=1, ord=1, keepdim=True)

        if self.verbose > 3:
            print(f"S max: {S.max()}")
            print(f"S min: {S.min()}")

        Z = X / S

        E_adjacency_list = self.adjacency_lists[dataset.name]
        adjacency_matrix = self.adjacency_matrices[dataset.name].to(self.context["device"])
        Sigma_x_inv = self.parameter_optimizer.spatial_affinity_state[dataset.name].to(self.context["device"])

        embedding_updater = BatchEffectTripletLossEmbeddingLossWithNeighborsNesterov(
            Z,
            S,
            B,
            MTM,
            YM,
            Ynorm,
            adjacency_matrix,
            prior_x_mode,
            prior_x,
            average_across_samples,
            Sigma_x_inv,
            E_adjacency_list,
            self.context["device"],
            base_step_size,
            self.verbose,
            self.embedding_acceleration_trick,
            self.use_inplace_ops,
            self.embedding_mini_iterations,
            tol,
        )

        loss, X = embedding_updater()

        return loss, X

    @torch.no_grad()
    def nll_weight_wnbr(self, Y, M, B, X, sigma_yx, prior_x_mode, prior_x, dataset, tol=1e-5, update_alg="nesterov"):
        # Precomputing quantities
        MTM = M.T @ M / (sigma_yx**2)
        YM = Y.to(M.device) @ M / (sigma_yx**2)
        Ynorm = torch.square(Y).sum() / (sigma_yx**2)
        S = torch.linalg.norm(X, dim=1, ord=1, keepdim=True)

        base_step_size = 0.1

        Z = X / S

        E_adjacency_list = self.adjacency_lists[dataset.name]
        adjacency_matrix = self.adjacency_matrices[dataset.name].to(self.context["device"])
        Sigma_x_inv = self.parameter_optimizer.spatial_affinity_state[dataset.name].to(self.context["device"])

        embedding_updater = EmbeddingLossWithNeighborsNesterov(
            Z,
            S,
            MTM,
            YM,
            Ynorm,
            adjacency_matrix,
            prior_x_mode,
            prior_x,
            Sigma_x_inv,
            E_adjacency_list,
            self.context["device"],
            base_step_size,
            self.verbose,
            self.embedding_acceleration_trick,
            self.use_inplace_ops,
            self.embedding_mini_iterations,
            tol,
        )

        loss = embedding_updater.compute_loss()

        return loss


class EmbeddingState(dict):
    """Collections of cell embeddings for all ST replicates.

    Attributes:
        K: embedding dimension:

    """

    def __init__(self, K: int, datasets: Sequence[PopariDataset], initial_context=None, context=None):
        self.datasets = datasets
        self.K = K
        self.initial_context = initial_context if initial_context else {"device": "cpu", "dtype": torch.float32}
        self.context = context if context else {"device": "cpu", "dtype": torch.float32}
        super().__init__()

        self.embeddings = []

        for dataset in self.datasets:
            num_cells, _ = dataset.shape
            replicate_embeddings = torch.zeros((num_cells, K), **self.context)
            self.__setitem__(dataset.name, replicate_embeddings)
            self.embeddings.append(replicate_embeddings)
