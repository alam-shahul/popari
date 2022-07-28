import sys, time, itertools, resource, logging, h5py, os, pickle
from pathlib import Path
import multiprocessing
from multiprocessing import Pool
from util import print_datetime, openH5File, encode4h5, parse_suffix

import numpy as np
import pandas as pd
from scipy.special import factorial
from sklearn.decomposition import TruncatedSVD, PCA

import torch
import anndata as ad

from load_data import load_expression, load_edges, load_genelist, load_anndata, save_anndata, save_predictors

from initialization import initialize_kmeans, initialize_Sigma_x_inv
from estimate_weights import estimate_weight_wonbr, estimate_weight_wnbr
from estimate_parameters import estimate_M, estimate_Sigma_x_inv
from sample_for_integral import integrate_of_exponential_over_simplex

from components import SpiceMixDataset, ParameterOptimizer, EmbeddingOptimizer

class SpiceMixPlus:
    """SpiceMixPlus optimization model.

    Models spatial biological data using the NMF-HMRF formulation. Supports multiple
    fields-of-view (FOVs) and multimodal data.

    Attributes:
        device: device to use for PyTorch operations
        num_processes: number of parallel processes to use for optimizing weights (should be <= #FOVs)
        replicate_names: names of replicates/FOVs in input dataset
        TODO: finish docstring
    """
    def __init__(self,
            K, lambda_Sigma_x_inv, repli_list, betas=None, prior_x_modes=None,
            path2result=None, context=None, metagene_mode="shared",
            spatial_affinity_mode="shared lookup",
            lambda_M=0, random_state=0
    ):

        if not context:
            context = dict(device='cpu', dtype=torch.float32)

        self.context = context
        self.repli_list = repli_list
        self.num_repli = len(self.repli_list)

        self.random_state = random_state
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.K = K
        self.lambda_Sigma_x_inv = lambda_Sigma_x_inv
        if betas is None:
            self.betas = np.full(self.num_repli, 1/self.num_repli)
        else:
            self.betas = np.array(betas, copy=False) / sum(betas)

        if prior_x_modes is None:
            prior_x_modes = ['exponential shared fixed'] * self.num_repli

        self.prior_x_modes = prior_x_modes
        self.M_constraint = 'simplex'
        # self.M_constraint = 'unit sphere'
        self.X_constraint = 'none'
        self.dropout_mode = 'raw'
        self.sigma_yx_inv_mode = 'separate'
        self.pairwise_potential_mode = 'normalized'
        self.spatial_affinity_mode = spatial_affinity_mode

        if path2result is not None:
            self.path2result = path2result
            self.result_filename = self.path2result
            logging.info(f'result file = {self.result_filename}')
        else:
            self.result_filename = None
        # self.save_hyperparameters()

        self.Ys = None
        self.Sigma_x_inv = self.Xs = self.sigma_yxs = None
       
        self.metagene_mode = metagene_mode 
        self.M = None
        self.M_bar = None
        self.lambda_M = lambda_M
        self.prior_xs = None

    def load_anndata_datasets(self, datasets, replicate_names):
        """Load SpiceMixPlus data directly from AnnData objects.

        Args:
            datasets (list of AnnData): a list of AnnData objects (one for each FOV)
            replicate_names (list of str): a list of names for each dataset
        """
        self.datasets = [SpiceMixDataset(dataset, replicate_name) for dataset, replicate_name in zip(datasets, replicate_names)]
        self.Ys = []
        for dataset in self.datasets:
            Y = torch.tensor(dataset.X, **self.context)
            self.Ys.append(Y)

    def load_dataset(self, path2dataset, anndata_filepath=None):
        """Load dataset into SpiceMix object.

        """

        path2dataset = Path(path2dataset)
        
        datasets = load_anndata(path2dataset / anndata_filepath, self.repli_list, self.context)
        self.load_anndata_datasets(datasets, self.repli_list)

    def initialize(self, method='kmeans'):
        """Initialize metagenes and hidden states.

        """

        self.parameter_optimizer = ParameterOptimizer(self.K, self.Ys, self.datasets, self.betas, self.prior_x_modes, lambda_Sigma_x_inv=self.lambda_Sigma_x_inv, context=self.context)
        self.embedding_optimizer = EmbeddingOptimizer(self.K, self.Ys, self.datasets, context=self.context)
        
        self.parameter_optimizer.initialize(self.embedding_optimizer)
        self.embedding_optimizer.initialize(self.parameter_optimizer)

        if method == 'kmeans':
            self.M, self.Xs = initialize_kmeans(self.K, self.Ys, kwargs_kmeans=dict(random_state=self.random_state), context=self.context)
        elif method == 'svd':
            self.M, self.Xs = self.initialize_svd(M_nonneg=(self.M_constraint == 'simplex'), X_nonneg=True,)
        else:
            raise NotImplementedError
        
        for dataset_index, dataset in enumerate(self.datasets):
            self.parameter_optimizer.metagene_state[dataset.name][:] = self.M
            self.embedding_optimizer.embedding_state[dataset.name][:] = self.Xs[dataset_index]

        self.parameter_optimizer.scale_metagenes()

        self.Sigma_x_inv_bar = None

        for dataset_index, (dataset, replicate) in enumerate(zip(self.datasets, self.repli_list)):
            if self.metagene_mode == "differential":
                dataset.uns["M_bar"] = {dataset.name: self.parameter_optimizer.metagene_state.M_bar}

            dataset.uns["M"] = {dataset.name: self.parameter_optimizer.metagene_state}
            dataset.obsm["X"] = self.embedding_optimizer.embedding_state[dataset.name]

            dataset.uns["spicemixplus_hyperparameters"] = {
                "metagene_mode": self.metagene_mode,
                "prior_x": self.parameter_optimizer.prior_xs[dataset_index][0],
                "K": self.K,
                "lambda_Sigma_x_inv": self.lambda_Sigma_x_inv,
            }

        self.parameter_optimizer.update_sigma_yx()

        initial_embeddings = [self.embedding_optimizer.embedding_state[dataset.name] for dataset in self.datasets]
        self.parameter_optimizer.spatial_affinity_state.initialize(initial_embeddings)

        self.synchronize_datasets()
    
    def synchronize_datasets(self):
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

    def initialize_svd(self, M_nonneg=True, X_nonneg=True):
        """Initialize metagenes and hidden states using SVD.
    
        Args:
            K (int): number of metagenes
            Ys (list of numpy.ndarray): list of gene expression data for each FOV.
                Note that currently all Ys must have the same number of genes;
                otherwise they are simply absent from the analysis.
    
        Returns:
            A tuple of (M, Xs), where M is the initial estimate of the metagene
            values and Xs is the list of initial estimates of the hidden states
            of the gene expression data.
        """
  
        # TODO: add check that number of genes is the same for all datasets
        Y_cat = np.concatenate([dataset.X for dataset in self.datasets], axis=0)
    
        svd = TruncatedSVD(self.K)
        X_cat = svd.fit_transform(Y_cat)
        M = svd.components_.T
        norm_p = np.ones([1, self.K])
        norm_n = np.ones([1, self.K])
    
        if M_nonneg:
            M_positive = np.clip(M, a_min=0, a_max=None)
            M_negative = np.clip(M, a_min=None, a_max=0)
            norm_p *= np.linalg.norm(M_positive, axis=0, ord=1, keepdims=True)
            norm_n *= np.linalg.norm(M_negative, axis=0, ord=1, keepdims=True)
    
        if X_nonneg:
            X_cat_positive = np.clip(X_cat, a_min=0, a_max=None)
            X_cat_negative = np.clip(X_cat, a_min=None, a_max=0)
            norm_p *= np.linalg.norm(X_cat_positive, axis=0, ord=1, keepdims=True)
            norm_n *= np.linalg.norm(X_cat_negative, axis=0, ord=1, keepdims=True)
    
        # Since M must be non-negative, choose the_value that yields greater L1-norm
        sign = np.where(norm_p >= norm_n, 1., -1.)
        M *= sign
        X_cat *= sign
        X_cat_iter = X_cat
        if M_nonneg:
            M = np.clip(M, a_min=1e-10, a_max=None)
            
        Xs = []
        for dataset in self.datasets:
            Y = dataset.X
            N = len(dataset)

            X = X_cat_iter[:N]
            X_cat_iter = X_cat_iter[N:]
            if X_nonneg:
                # fill negative elements by zero
                # X = np.clip(X, a_min=1e-10, a_max=None)
                # fill negative elements by the average of nonnegative elements
                for x in X.T:
                    idx = x < 1e-10
                    # Bugfix below: if statement necessary, otherwise nan elements may be introduced...
                    if len(x[~idx]) > 0:
                        x[idx] = x[~idx].mean()
            else:
                X = np.full([N, self.K], 1/self.K)
            Xs.append(X)
    
        M = torch.tensor(M, **self.context)
        Xs = [torch.tensor(X, **self.context) for X in Xs]
    
        return M, Xs

    def estimate_weights(self):
        """Update embeddings (latent states) for each replicate.

        """

        logging.info(f'{print_datetime()}Updating latent states')
        self.embedding_optimizer.update_embeddings()
        self.synchronize_datasets()

    def estimate_parameters(self, update_spatial_affinities=True):
        """Update parameters for each replicate.

        Args:
            update_spatial_affinities (bool): If specified, spatial affinities will be update during this iteration.
                Default: True.

        """
        logging.info(f'{print_datetime()}Updating model parameters')

        if update_spatial_affinities:
            self.parameter_optimizer.update_spatial_affinity()
            # TODO: somehow this makes the ARI really good. Why?
            # Solved: because the Sigma_x_inv does not get too large

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

    def save_results(self, path2dataset, iteration, PredictorConstructor=None, predictor_hyperparams=None, filename=None):
        if not filename:
            filename = f"trained_iteration_{iteration}.h5"

        save_anndata(path2dataset / filename, self.datasets, self.repli_list)
