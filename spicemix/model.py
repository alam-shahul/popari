from typing import Sequence, Union, Optional

import sys, time, itertools, logging, os, pickle
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import anndata as ad

from spicemix.util import print_datetime
from spicemix.io import load_anndata, save_anndata

from spicemix.initialization import initialize_kmeans, initialize_svd
from spicemix.sample_for_integral import integrate_of_exponential_over_simplex

from spicemix.components import SpiceMixDataset, ParameterOptimizer, EmbeddingOptimizer

class SpiceMixPlus:
    """SpiceMixPlus optimization model.

    Models spatial biological data using the NMF-HMRF formulation. Supports multiple
    fields-of-view (FOVs) and multimodal data.

    Attributes:
        K: number of metagenes to learn
        datasets: list of input AnnData spatial datasets for SpiceMix.
        embedding_optimizer: object wrapping learned embeddings (i.e. X) and related state
        parameter_optimizer: object wrapping learned parameters (i.e. M, sigma_yxs, spatial affinities)
            and related state
    """
    def __init__(self,
        K: int,
        replicate_names: Sequence[str],
        datasets: Optional[Union[str, Path, Sequence[ad.AnnData]]] = None,
        dataset_path: Optional[Union[str, Path ]] = None,
        lambda_Sigma_x_inv: float = 1e-4,
        initialization_method: str = "svd",
        betas: Optional[Sequence[float]] = None,
        prior_x_modes: Optional[Sequence[float]] = None,
        M_constraint: str = "simplex",
        sigma_yx_inv_mode: str = "separate",
        torch_context: Optional[dict] = None,
        metagene_mode: str = "shared",
        spatial_affinity_mode: str ="shared lookup",
        lambda_M: float = 0,
        random_state: int = 0
    ):
        """Initialize SpiceMixPlus object using ST data.

        Args:
            K: number of metagenes to learn
            datasets: list of input AnnData spatial datasets for SpiceMix.
            dataset_path: path to AnnData merged dataset on disk. Ignored if `datasets` is specified.
            replicate_names: names of spatial datasets
            lambda_Sigma_x_inv: hyperparameter to balance importance of spatial information. Default: 1e-4
            initialization_method: algorithm to use for initializing metagenes and embeddings. Default: `svd`
            betas: weighting of each dataset during optimization. Defaults to equally weighting each dataset
            prior_x_modes: family of prior distribution for embeddings of each dataset
            M_constraint: constraint on columns of M. Default: `simplex`
            sigma_yx_inv_mode: form of sigma_yx_inv parameter. Default: `separate`
            torch_context: keyword args to use during initialization of PyTorch tensors.
            metagene_mode: modality of metagene parameters. Default: `shared`
            spatial_affinity_mode: modality of spatial affinity parameters. Default: `shared lookup`
            lambda_M: hyperparameter to constrain metagene deviation in differential case. Ignored if
                `metagene_mode` is `shared`. Default: `0`
            random_state: seed for reproducibility of randomized computations. Default: `0`
        """

        if not any([datasets, dataset_path]):
            raise ValueError("At least one of `datasets`, `dataset_path` must be specified in the SpiceMixPlus constructor.")

        if not torch_context:
            torch_context = dict(device='cpu', dtype=torch.float32)

        self.context = torch_context

        self.random_state = random_state
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.K = K
        self.lambda_Sigma_x_inv = lambda_Sigma_x_inv
        self.M_constraint = M_constraint
        self.sigma_yx_inv_mode = sigma_yx_inv_mode
        self.spatial_affinity_mode = spatial_affinity_mode

        self.metagene_mode = metagene_mode 
        self.lambda_M = lambda_M

        if datasets:
            self.load_anndata_datasets(datasets, replicate_names)
        elif dataset_path:
            self.load_dataset(dataset_path, replicate_names)

        self._initialize(betas, prior_x_modes, method=initialization_method)

    def load_anndata_datasets(self, datasets: Sequence[ad.AnnData], replicate_names: Sequence[str]):
        """Load SpiceMixPlus data directly from AnnData objects.

        Args:
            datasets: spatial transcriptomics datasets in AnnData format (one for each FOV)
            replicate_names: names for all datasets/replicates
        """
        self.datasets = [SpiceMixDataset(dataset, replicate_name) for dataset, replicate_name in zip(datasets, replicate_names)]
        self.Ys = []
        for dataset in self.datasets:
            Y = torch.tensor(dataset.X, **self.context)
            self.Ys.append(Y)
        
        self.num_replicates = len(self.datasets)

    def load_dataset(self, dataset_path: Union[str, Path], replicate_names: Sequence[str]):
        """Load dataset into SpiceMixPlus from saved .h5ad file.

        Args:
            dataset_path: path to input ST datasets, stored in .h5ad format
            replicate_names: names for all datasets/replicates. Note that these must match the names in the .h5ad file.
        """

        dataset_path = Path(dataset_path)
        
        datasets = load_anndata(dataset_path, replicate_names, self.context)
        self.load_anndata_datasets(datasets, replicate_names)

    def _initialize(self, betas: Optional[Sequence[float]] = None, prior_x_modes: Optional[Sequence[str]] = None, method: str = 'svd'):
        """Initialize metagenes and hidden states.

        Args:
            betas: weighting of each dataset during optimization. Defaults to equally weighting each dataset
            prior_x_modes: family of prior distribution for embeddings of each dataset
            method: algorithm to use for initializing metagenes and embeddings. Default: SVD
        """
       
        if betas is None:
            self.betas = np.full(self.num_replicates, 1/self.num_replicates)
        else:
            self.betas = np.array(betas, copy=False) / sum(betas)

        if prior_x_modes is None:
            prior_x_modes = [None] * self.num_replicates

        self.prior_x_modes = prior_x_modes

        self.parameter_optimizer = ParameterOptimizer(self.K, self.Ys, self.datasets, self.betas, prior_x_modes,
                lambda_Sigma_x_inv=self.lambda_Sigma_x_inv,
                M_constraint=self.M_constraint,
                context=self.context
            )
        self.embedding_optimizer = EmbeddingOptimizer(self.K, self.Ys, self.datasets, context=self.context)
        
        self.parameter_optimizer.link(self.embedding_optimizer)
        self.embedding_optimizer.link(self.parameter_optimizer)

        if method == 'kmeans':
            self.M, self.Xs = initialize_kmeans(self.datasets, self.K, self.context, kwargs_kmeans=dict(random_state=self.random_state))
        elif method == 'svd':
            self.M, self.Xs = initialize_svd(self.datasets, self.K, self.context, M_nonneg=(self.M_constraint == 'simplex'), X_nonneg=True)
        else:
            raise NotImplementedError
        
        for dataset_index, dataset in enumerate(self.datasets):
            self.parameter_optimizer.metagene_state[dataset.name][:] = self.M
            self.embedding_optimizer.embedding_state[dataset.name][:] = self.Xs[dataset_index]

        self.parameter_optimizer.scale_metagenes()

        self.Sigma_x_inv_bar = None

        self.parameter_optimizer.update_sigma_yx()

        initial_embeddings = [self.embedding_optimizer.embedding_state[dataset.name] for dataset in self.datasets]
        self.parameter_optimizer.spatial_affinity_state.initialize(initial_embeddings)
        
        for dataset_index, dataset  in enumerate(self.datasets):
            if self.metagene_mode == "differential":
                dataset.uns["M_bar"] = {dataset.name: self.parameter_optimizer.metagene_state.M_bar}

            dataset.uns["M"] = {dataset.name: self.parameter_optimizer.metagene_state}
            dataset.obsm["X"] = self.embedding_optimizer.embedding_state[dataset.name]
                
            dataset.uns["Sigma_x_inv"] = {dataset.name : self.parameter_optimizer.spatial_affinity_state[dataset.name]}

            dataset.uns["spicemixplus_hyperparameters"] = {
                "metagene_mode": self.metagene_mode,
                "prior_x": self.parameter_optimizer.prior_xs[dataset_index][0],
                "K": self.K,
                "lambda_Sigma_x_inv": self.lambda_Sigma_x_inv,
            }


        self.synchronize_datasets()
    
    def synchronize_datasets(self):
        """Synchronize datasets with learned SpiceMix parameters and embeddings."""
        for dataset_index, dataset in enumerate(self.datasets):
            dataset.uns["M"][dataset.name]= self.parameter_optimizer.metagene_state[dataset.name]
            dataset.obsm["X"] = self.embedding_optimizer.embedding_state[dataset.name]
            dataset.uns["sigma_yx"] = self.parameter_optimizer.sigma_yxs[dataset_index]
            with torch.no_grad():
                dataset.uns["Sigma_x_inv"][dataset.name][:] = self.parameter_optimizer.spatial_affinity_state[dataset.name]

            if self.spatial_affinity_mode == "differential lookup":
                dataset.uns["spatial_affinity_bar"][dataset.name][:] = self.parameter_optimizer.spatial_affinity_state.spatial_affinity_bar

            if self.metagene_mode == "differential":
                dataset.uns["M_bar"][dataset.name] = self.parameter_optimizer.metagene_state.M_bar


    def estimate_weights(self):
        """Update embeddings (latent states) for each replicate."""
        logging.info(f'{print_datetime()}Updating latent states')
        self.embedding_optimizer.update_embeddings()
        self.synchronize_datasets()

    def estimate_parameters(self, update_spatial_affinities: bool = True):
        """Update parameters for each replicate.

        Args:
            update_spatial_affinities: If specified, spatial affinities will be updated during this iteration.
                Default: True.
        """
        logging.info(f'{print_datetime()}Updating model parameters')

        if update_spatial_affinities:
            self.parameter_optimizer.update_spatial_affinity()

            # elif self.Sigma_x_inv_mode == "differential":
            #     for X, dataset, u, beta, optimizer, replicate in zip(self.Xs, self.datasets, use_spatial,
            #             self.betas, self.optimizer_Sigma_x_invs, self.repli_list):
            #         updated_Sigma_x_inv, Q_value = estimate_Sigma_x_inv([X], dataset.uns["Sigma_x_inv"][f"{replicate}"],
            #                 [u], self.lambda_Sigma_x_inv, [beta], optimizer, self.context, [dataset])
            #         with torch.no_grad():
            #             # Note: in-place update is necessary here in order for optimizer to track same object
            #             dataset.uns["Sigma_x_inv"][f"{replicate}"][:] = updated_Sigma_x_inv

            #         print(f"Q_value:{Q_value}")
       
            #     self.Sigma_x_inv_bar.zero_()
            #     for dataset, replicate in zip(self.datasets, self.repli_list):
            #         self.Sigma_x_inv_bar.add_(dataset.uns["Sigma_x_inv"][f"{replicate}"])
            #     
            #     self.Sigma_x_inv_bar.div_(len(self.datasets))

            #     for dataset, replicate in zip(self.datasets, self.repli_list):
            #         dataset.uns["Sigma_x_inv_bar"][f"{replicate}"] = self.Sigma_x_inv_bar

        self.parameter_optimizer.update_metagenes(self.Xs)
        self.parameter_optimizer.update_sigma_yx()
        self.synchronize_datasets()

    def save_results(self, path2dataset, PredictorConstructor=None, predictor_hyperparams=None):
        """Save datasets and learned SpiceMixPlus parameters to .h5ad file.

        Args:
            dataset_path: path to input ST datasets, to be stored in .h5ad format

        """
        replicate_names = [dataset.name for dataset in self.datasets]
        save_anndata(path2dataset, self.datasets, replicate_names)
