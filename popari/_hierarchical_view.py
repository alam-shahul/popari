from __future__ import annotations
from typing import Sequence

import numpy as np

import torch

from collections import defaultdict

from popari.util import get_datetime, convert_numpy_to_pytorch_sparse_coo
from popari.initialization import initialize_kmeans, initialize_svd, initialize_louvain

from popari._popari_dataset import PopariDataset
from popari._embedding_optimizer import EmbeddingOptimizer
from popari._parameter_optimizer import ParameterOptimizer

class HierarchicalView():
    """View of SRT multisample dataset at a set resolution.

    Includes the scaled (i.e. binned data) as well as the learnable Popari parameters and
    their corresponding optimizers.

    """

    def __init__(self, datasets: Sequence[PopariDataset], betas: list, prior_x_modes: list,
            method: str, random_state: int, K: int, context: dict, initial_context: dict, use_inplace_ops: bool,
            pretrained: bool, verbose: str, parameter_optimizer_hyperparameters: dict,
            embedding_optimizer_hyperparameters: dict, level: int = 0):

        self.datasets = datasets
        self.K = K
        self.level = level
        self.context = context
        self.initial_context = initial_context
        self.use_inplace_ops = use_inplace_ops
        self.verbose = verbose
        self.random_state = random_state
        self.pretrained = pretrained

        self.num_replicates = len(self.datasets)

        self.Ys = []
        for dataset in self.datasets:
            Y = torch.tensor(dataset.X, **self.context)
            Y *= (self.K * 1) / Y.sum(axis=1, keepdim=True).mean()
            self.Ys.append(Y)

        if betas is None:
            self.betas = np.full(self.num_replicates, 1/self.num_replicates)
        else:
            self.betas = np.array(betas, copy=False) / sum(betas)
        
        if prior_x_modes is None:
            prior_x_modes = [None] * self.num_replicates

        self.prior_x_modes = prior_x_modes

        self.parameter_optimizer = ParameterOptimizer(self.K, self.Ys, self.datasets, self.betas, prior_x_modes,
                initial_context=self.initial_context,
                context=self.context,
                use_inplace_ops=self.use_inplace_ops,
                verbose=self.verbose,
                **parameter_optimizer_hyperparameters
            )

        if self.verbose:
            print(f"{get_datetime()} Initializing EmbeddingOptimizer")
        self.embedding_optimizer = EmbeddingOptimizer(self.K, self.Ys, self.datasets,
            initial_context=self.initial_context,
            context=self.context,
            use_inplace_ops=self.use_inplace_ops,
            verbose=self.verbose,
            **embedding_optimizer_hyperparameters
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
            for dataset_index, dataset  in enumerate(self.datasets):
                self.parameter_optimizer.metagene_state[dataset.name][:] = torch.from_numpy(dataset.uns["M"][dataset.name]).to(**self.initial_context)
                self.embedding_optimizer.embedding_state[dataset.name][:] = torch.from_numpy(dataset.obsm["X"]).to(**self.initial_context)
                self.embedding_optimizer.adjacency_matrices[dataset.name] = convert_numpy_to_pytorch_sparse_coo(dataset.obsp["adjacency_matrix"], self.initial_context)
                self.parameter_optimizer.adjacency_matrices[dataset.name] = convert_numpy_to_pytorch_sparse_coo(dataset.obsp["adjacency_matrix"], self.initial_context)
                    
                self.parameter_optimizer.spatial_affinity_state[dataset.name] = torch.from_numpy(dataset.uns["Sigma_x_inv"][dataset.name]).to(**self.initial_context)
                spatial_affinity_copy[dataset_index] = self.parameter_optimizer.spatial_affinity_state[dataset.name]
      
            self.parameter_optimizer.update_sigma_yx()
            self.parameter_optimizer.spatial_affinity_state.initialize_optimizers(spatial_affinity_copy)
        else:
            if self.verbose:
                print(f"{get_datetime()} Initializing metagenes and hidden states")
            if method == 'kmeans':
                self.M, self.Xs = initialize_kmeans(self.datasets, self.K, self.initial_context, kwargs_kmeans=dict(random_state=self.random_state))
            elif method == 'svd':
                self.M, self.Xs = initialize_svd(self.datasets, self.K, self.initial_context, M_nonneg=(self.parameter_optimizer.M_constraint == 'simplex'), X_nonneg=True)
            elif method == 'louvain':
                kwargs_louvain = {
                    "random_state": self.random_state
                }
                self.M, self.Xs = initialize_louvain(self.datasets, self.K, self.initial_context, kwargs_louvain=kwargs_louvain)
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
            self.parameter_optimizer.spatial_affinity_state.initialize(initial_embeddings)
            
            for dataset_index, dataset in enumerate(self.datasets):
                metagene_state = self.parameter_optimizer.metagene_state[dataset.name].cpu().detach().numpy()
                dataset.uns["M"] = {dataset.name: metagene_state}

                X = self.embedding_optimizer.embedding_state[dataset.name].cpu().detach().numpy()
                dataset.obsm["X"] = X
                
                Sigma_x_inv = self.parameter_optimizer.spatial_affinity_state[dataset.name].cpu().detach().numpy()
                dataset.uns["Sigma_x_inv"] = {dataset.name : Sigma_x_inv}

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
                M_bar = {group_name: self.parameter_optimizer.metagene_state.M_bar[group_name].cpu().detach().numpy() for group_name in self.parameter_optimizer.metagene_groups}
                for dataset in self.datasets:
                    dataset.uns["M_bar"] = M_bar
                    # dataset.uns["lambda_Sigma_bar"] = self.parameter_optimizer.lambda_Sigma_bar
            
            if self.parameter_optimizer.spatial_affinity_mode == "differential lookup":
                spatial_affinity_bar = {group_name: self.parameter_optimizer.spatial_affinity_state.spatial_affinity_bar[group_name].cpu().detach().numpy() for group_name in self.parameter_optimizer.spatial_affinity_groups}
                for dataset in self.datasets:
                    dataset.uns["spatial_affinity_bar"] = spatial_affinity_bar
                    # dataset.uns["lambda_M"] = self.parameter_optimizer.lambda_M
        
        for dataset in self.datasets:
            if "losses" not in dataset.uns:
                dataset.uns["losses"] = defaultdict(list)
            else:
                for key in dataset.uns["losses"]:
                    dataset.uns["losses"][key] = list(dataset.uns["losses"][key])
            
        if self.parameter_optimizer.metagene_mode == "differential":
            self.parameter_optimizer.metagene_state.reaverage()
        if self.parameter_optimizer.spatial_affinity_mode == "differential lookup":
            self.parameter_optimizer.spatial_affinity_state.reaverage()

    def link(self, high_res_view: HierarchicalView):
        """Link a view to the resolution right above it in the hierarchy."""
        self.high_res_view = high_res_view

    def synchronize_datasets(self):
        """Synchronize datasets with learned view parameters and embeddings."""
        for dataset_index, dataset in enumerate(self.datasets):
            dataset.uns["M"][dataset.name] = self.parameter_optimizer.metagene_state[dataset.name].cpu().detach().numpy()
            dataset.obsm["X"] = self.embedding_optimizer.embedding_state[dataset.name].cpu().detach().numpy()
            dataset.uns["sigma_yx"] = self.parameter_optimizer.sigma_yxs[dataset_index]
            with torch.no_grad():
                dataset.uns["Sigma_x_inv"][dataset.name][:] = self.parameter_optimizer.spatial_affinity_state[dataset.name].cpu().detach().numpy()
           
            dataset.uns["losses"]["nll_embeddings"].append(self.embedding_optimizer.nll_embeddings())
            dataset.uns["losses"]["nll_spatial_affinities"].append(self.parameter_optimizer.nll_spatial_affinities())
            dataset.uns["losses"]["nll_metagenes"].append(self.parameter_optimizer.nll_metagenes())
            dataset.uns["losses"]["nll_sigma_yx"].append(self.parameter_optimizer.nll_sigma_yx())
            dataset.uns["losses"]["nll"].append(self.nll())
            
        if self.parameter_optimizer.metagene_mode == "differential":
            M_bar = {group_name: self.parameter_optimizer.metagene_state.M_bar[group_name].cpu().detach().numpy() for group_name in self.parameter_optimizer.metagene_groups}
            for dataset in self.datasets:
                dataset.uns["M_bar"] = M_bar
        
        if self.parameter_optimizer.spatial_affinity_mode == "differential lookup":
            spatial_affinity_bar = {group_name: self.parameter_optimizer.spatial_affinity_state.spatial_affinity_bar[group_name].cpu().detach().numpy() for group_name in self.parameter_optimizer.spatial_affinity_groups}
            for dataset in self.datasets:
                dataset.uns["spatial_affinity_bar"] = spatial_affinity_bar
    
    def nll(self, use_spatial=False):
        """Compute overall negative log-likelihood for current model parameters.

        """

        with torch.no_grad():
            total_loss  = torch.zeros(1, **self.context)
            if use_spatial:    
                weighted_total_cells = 0
                for dataset in self.datasets: 
                    E_adjacency_list = self.embedding_optimizer.adjacency_lists[dataset.name]
                    weighted_total_cells += sum(map(len, E_adjacency_list))

            for dataset_index, dataset  in enumerate(self.datasets):
                sigma_yx = self.parameter_optimizer.sigma_yxs[dataset_index]
                Y = self.Ys[dataset_index].to(self.context["device"])
                X = self.embedding_optimizer.embedding_state[dataset.name].to(self.context["device"])
                M = self.parameter_optimizer.metagene_state[dataset.name].to(self.context["device"])
                prior_x_mode = self.parameter_optimizer.prior_x_modes[dataset_index]
                beta = self.betas[dataset_index]
                prior_x = self.parameter_optimizer.prior_xs[dataset_index]
                    
                # Precomputing quantities
                MTM = M.T @ M / (sigma_yx ** 2)
                YM = Y.to(M.device) @ M / (sigma_yx ** 2)
                Ynorm = torch.linalg.norm(Y, ord='fro').item() ** 2 / (sigma_yx ** 2)
                S = torch.linalg.norm(X, dim=1, ord=1, keepdim=True)

                Z = X / S
                N, G = Y.shape
                
                loss = ((X @ MTM) * X).sum() / 2 - (X * YM).sum() + Ynorm / 2

                logZ_i_Y = torch.full((N,), G/2 * np.log((2 * np.pi * sigma_yx**2)), **self.context)
                if not use_spatial:
                    logZ_i_X = torch.full((N,), 0, **self.context)
                    if (prior_x[0] != 0).all():
                        logZ_i_X +=  torch.full((N,), self.K * torch.log(prior_x[0]).item(), **self.context)
                    log_partition_function = (logZ_i_Y + logZ_i_X).sum()
                else:
                    adjacency_matrix = self.embedding_optimizer.adjacency_matrices[dataset.name].to(self.context["device"])
                    Sigma_x_inv = self.parameter_optimizer.spatial_affinity_state[dataset.name].to(self.context["device"])
                    nu = adjacency_matrix @ Z
                    eta = nu @ Sigma_x_inv
                    logZ_i_s = torch.full((N,), 0, **self.context)
                    if (prior_x[0] != 0).all():
                        logZ_i_s = torch.full((N,), -self.K * torch.log(prior_x[0]).item() + torch.log(factorial(self.K-1, exact=True)).item(), **self.context)

                    logZ_i_z = integrate_of_exponential_over_simplex(eta)
                    log_partition_function = (logZ_i_Y + logZ_i_z + logZ_i_s).sum()
            
                    if prior_x_mode == 'exponential shared fixed':
                        loss += prior_x[0][0] * S.sum()
                    elif not prior_x_mode:
                        pass
                    else:
                        raise NotImplementedError
            
                    if Sigma_x_inv is not None:
                        loss += (eta).mul(Z).sum() / 2

                    spatial_affinity_bars = None
                    if self.spatial_affinity_mode == "differential lookup":
                        spatial_affinity_bars = [self.parameter_optimizer.spatial_affinity_state.spatial_affinity_bar[group_name].detach() for group_name in self.parameter_optimizer.spatial_affinity_tags[dataset.name]]

                    regularization = torch.zeros(1, **self.context)
                    if spatial_affinity_bars is not None:
                        group_weighting = 1 / len(spatial_affinity_bars)
                        for group_Sigma_x_inv_bar in spatial_affinity_bars:
                            regularization += group_weighting * self.parameter_optimizer.lambda_Sigma_bar * (group_Sigma_x_inv_bar - Sigma_x_inv).pow(2).sum() / 2
        
                    regularization += self.parameter_optimizer.lambda_Sigma_x_inv * Sigma_x_inv.pow(2).sum() / 2

                    regularization *= weighted_total_cells
        
                    loss += regularization.item()
                    
                loss += log_partition_function

                differential_regularization_term = torch.zeros(1, **self.context)
                M_bar = None
                if self.parameter_optimizer.metagene_mode == "differential":
                    M_bar = [self.parameter_optimizer.metagene_state.M_bar[group_name] for group_name in self.parameter_optimizer.metagene_tags[dataset.name]]
                   
                if self.parameter_optimizer.lambda_M > 0 and M_bar is not None:
                    differential_regularization_quadratic_factor = self.parameter_optimizer.lambda_M * torch.eye(self.K, **self.context)
                    
                    differential_regularization_linear_term = torch.zeros_like(M, **self.context)
                    group_weighting = 1 / len(M_bar)
                    for group_M_bar in M_bar:
                        differential_regularization_linear_term += group_weighting * self.parameter_optimizer.lambda_M * group_M_bar

                    differential_regularization_term = (M @ differential_regularization_quadratic_factor * M).sum() - 2 * (differential_regularization_linear_term * M).sum()
                    group_weighting = 1 / len(M_bar)
                    for group_M_bar in M_bar:
                        differential_regularization_term += group_weighting * self.parameter_optimizer.lambda_M * (group_M_bar * group_M_bar).sum()
                
                loss += differential_regularization_term.item()

                total_loss += loss

        return total_loss.cpu().numpy()
