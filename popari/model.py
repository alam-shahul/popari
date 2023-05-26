from typing import Sequence, Union, Optional

import sys, time, itertools, logging, os, pickle
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import anndata as ad

from popari.util import get_datetime, convert_numpy_to_pytorch_sparse_coo
from popari.io import load_anndata, save_anndata, merge_anndata, unmerge_anndata

from popari.components import PopariDataset, HierarchicalView

from popari._hierarchical_view import construct_hierarchy, reconstruct_hierarchy

class Popari:
    r"""Popari optimization model.

    Models spatial biological data using the NMF-HMRF formulation. Supports multiple
    fields-of-view (FOVs) and differential analysis.
   
    Example of including math in docstring (for use later):
    :math:`||A||_F = [\sum_{i,j} abs(a_{i,j})^2]^{1/2}`

    Attributes:
        K: number of metagenes to learn
        replicate_names: names of spatial datasets
        datasets: list of input AnnData spatial datasets for Popari.
        dataset_path: path to AnnData merged dataset on disk. Ignored if ``datasets`` is specified.
        lambda_Sigma_x_inv: hyperparameter to balance importance of spatial information. Default: ``1e-4``
        pretrained: if set, attempts to load model state from input files. Default: ``False``
        initialization_method: algorithm to use for initializing metagenes and embeddings. Default: ``louvain``
        metagene_groups: defines a grouping of replicates for the metagene optimization. If
            ``metagene_mode == "shared"``, then one set of metagenes will be created for each group;
            if ``metagene_mode == "differential",  then all replicates will have their own set of metagenes,
            but each group will share an ``M_bar``.
        spatial_affinity_groups: defines a grouping of replicates for the spatial affinity optimization.
            If ``spatial_affinity_mode == "shared"``, then one set of spatial_affinities will be created for each group;
            if ``spatial_affinity_mode == "differential"``,  then all replicates will have their own set of spatial
            affinities, but each group will share a ``spatial_affinity_bar``.
        betas: weighting of each dataset during optimization. Defaults to equally weighting each dataset
        prior_x_modes: family of prior distribution for embeddings of each dataset
        M_constraint: constraint on columns of M. Default: ``simplex``
        sigma_yx_inv_mode: form of sigma_yx_inv parameter. Default: ``separate``
        torch_context: keyword args to use of PyTorch tensors during training.
        initial_context: keyword args to use during initialization of PyTorch tensors.
        metagene_mode: modality of metagene parameters. Default: ``shared``.

            =================  ===== 
            ``metagene_mode``  Option
            =================  =====
            ``shared``         A metagene set is shared between all replicates in a group.
            ``differential``   Each replicate learns its own metagene set.
            =================  =====

        spatial_affinity_mode: modality of spatial affinity parameters. Default: ``shared lookup``
        lambda_M: hyperparameter to constrain metagene deviation in differential case. Ignored if
            ``metagene_mode`` is ``shared``. Default: ``0.5``
        lambda_Sigma_bar: hyperparameter to constrain spatial affinity deviation in differential case. Ignored if
            ``spatial_affinity_mode`` is ``shared lookup``. Default: ``0.5``
        spatial_affinity_lr: learning rate for optimization of ``Sigma_x_inv``
        spatial_affinity_tol: convergence tolerance during optimization of ``Sigma_x_inv``
        spatial_affinity_constraint: method to ensure that spatial affinities lie within an appropriate range
        spatial_affinity_centering: if set, spatial affinities are zero-centered after every optimization step
        spatial_affinity_scaling: magnitude of spatial affinities during initial scaling. Default: ``10``
        spatial_affinity_regularization_power: exponent controlling penalization of spatial affinity magnitudes. Default: ``2``
        embedding_mini_iterations: number of mini-iterations to use during each iteration of embedding optimization. Default: ``1000``
        embedding_acceleration_trick: if set, use trick to accelerate convergence of embedding optimization. Default: ``True``
        embedding_step_size_multiplier: controls relative step size during embedding optimization. Default: ``1.0``
        superresolution_lr: learning rate for optimization of ``X`` from low-res embeddings
        use_inplace_ops: if set, inplace PyTorch operations will be used to speed up computation
        random_state: seed for reproducibility of randomized computations. Default: ``0``
        verbose: level of verbosity to use during optimization. Default: ``0`` (no print statements)
    """
    
    def __init__(self,
        K: int,
        replicate_names: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[ad.AnnData]] = None,
        dataset_path: Optional[Union[str, Path]] = None,
        reloaded_hierarchy: Optional[dict] = None,
        lambda_Sigma_x_inv: float = 1e-4,
        pretrained: bool = False,
        initialization_method: str = "louvain",
        hierarchical_levels: Optional[int] = None,
        metagene_groups: Optional[dict] = None,
        spatial_affinity_groups: Optional[dict] = None,
        betas: Optional[Sequence[float]] = None,
        prior_x_modes: Optional[Sequence[str]] = None,
        M_constraint: str = "simplex",
        sigma_yx_inv_mode: str = "separate",
        torch_context: Optional[dict] = None,
        initial_context: Optional[dict] = None,
        metagene_mode: str = "shared",
        spatial_affinity_mode: str = "shared lookup",
        lambda_M: float = 0.5,
        lambda_Sigma_bar: float = 0.5,
        spatial_affinity_lr: float = 1e-2,
        spatial_affinity_tol: float = 2e-3,
        spatial_affinity_constraint: Optional[str] = None,
        spatial_affinity_centering: bool = False,
        spatial_affinity_scaling: int = 10,
        spatial_affinity_regularization_power: int = 2,
        embedding_mini_iterations: int = 1000,
        embedding_acceleration_trick: bool = True,
        embedding_step_size_multiplier: float = 1.0,
        superresolution_lr: float = 1e-3,
        use_inplace_ops: bool = True,
        random_state: int = 0,
        verbose: int = 0
    ):

        self.use_inplace_ops = use_inplace_ops
        self.verbose = verbose

        if not any([datasets, dataset_path]):
            raise ValueError("At least one of `datasets`, `dataset_path` must be specified in the Popari constructor.")

        if K <= 1:
            raise ValueError("`K` must be an integer value greater than 1.")

        if not torch_context:
            torch_context = dict(device='cpu', dtype=torch.float32)
        
        if not initial_context:
            initial_context = dict(device='cpu', dtype=torch.float32)

        self.context = torch_context
        self.initial_context = initial_context

        self.random_state = random_state
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.K = K
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
        self.pretrained = pretrained

        self.metagene_mode = metagene_mode 
        self.lambda_M = lambda_M
        self.metagene_groups = metagene_groups
        self.spatial_affinity_groups = spatial_affinity_groups

            
        self.embedding_step_size_multiplier = embedding_step_size_multiplier
        self.embedding_mini_iterations = embedding_mini_iterations
        self.embedding_acceleration_trick = embedding_acceleration_trick

        self.hierarchical_levels = hierarchical_levels
        self.reloaded_hierarchy = reloaded_hierarchy
        self.superresolution_lr = superresolution_lr

        if dataset_path:
            self.load_dataset(dataset_path)
        elif datasets:
            self.load_anndata_datasets(datasets, replicate_names)

        if replicate_names is None:
            self.replicate_names = [dataset.name for dataset in self.datasets]
        else:
            self.replicate_names = [f"{replicate_name}" for replicate_name in replicate_names]
        
        self.parameter_optimizer_hyperparameters = {
            "lambda_Sigma_x_inv": self.lambda_Sigma_x_inv,
            "lambda_Sigma_bar": self.lambda_Sigma_bar,
            "spatial_affinity_lr": self.spatial_affinity_lr,
            "spatial_affinity_tol": self.spatial_affinity_tol,
            "spatial_affinity_constraint": self.spatial_affinity_constraint,
            "spatial_affinity_centering": self.spatial_affinity_centering,
            "spatial_affinity_scaling": self.spatial_affinity_scaling,
            "spatial_affinity_regularization_power": self.spatial_affinity_regularization_power,
            "M_constraint": self.M_constraint,
            "sigma_yx_inv_mode": self.sigma_yx_inv_mode,
            "spatial_affinity_mode": self.spatial_affinity_mode,
            "lambda_M": self.lambda_M,
            "metagene_mode": self.metagene_mode,
        }

        self.embedding_optimizer_hyperparameters = {
            "embedding_step_size_multiplier": embedding_step_size_multiplier,
            "embedding_mini_iterations": embedding_mini_iterations,
            "embedding_acceleration_trick": embedding_acceleration_trick,
        }

        self._initialize(betas=betas, prior_x_modes=prior_x_modes, method=initialization_method, pretrained=pretrained)

    def load_anndata_datasets(self, datasets: Sequence[ad.AnnData], replicate_names: Sequence[str]):
        """Load Popari data directly from AnnData objects.

        Args:
            datasets: spatial transcriptomics datasets in AnnData format (one for each FOV)
            replicate_names: names for all datasets/replicates
        """
        self.datasets = [PopariDataset(dataset, replicate_name) for dataset, replicate_name in zip(datasets, replicate_names)]
        self.num_replicates = len(self.datasets)

    def load_dataset(self, dataset_path: Union[str, Path]):
        """Load dataset into Popari from saved .h5ad file.

        Args:
            dataset_path: path to input ST datasets, stored in .h5ad format
        """

        dataset_path = Path(dataset_path)
        
        datasets, replicate_names = load_anndata(dataset_path)
        self.load_anndata_datasets(datasets, replicate_names)

    def _initialize(self, pretrained=False, betas: Optional[Sequence[float]] = None, prior_x_modes: Optional[Sequence[str]] = None, method: str = 'svd'):
        """Initialize metagenes and hidden states.

        Args:
            betas: weighting of each dataset during optimization. Defaults to equally weighting each dataset
            prior_x_modes: family of prior distribution for embeddings of each dataset
            method: algorithm to use for initializing metagenes and embeddings. Default: SVD
        """

        hierarchical_view_kwargs = {
                "random_state": self.random_state,
                "K": self.K,
                "context": self.context,
                "initial_context": self.initial_context,
                "betas": betas,
                "prior_x_modes": prior_x_modes,
                "use_inplace_ops": self.use_inplace_ops,
                "method": method,
                "pretrained": self.pretrained,
                "verbose": self.verbose,
                "metagene_groups": self.metagene_groups,
                "spatial_affinity_groups": self.spatial_affinity_groups,
                "parameter_optimizer_hyperparameters": self.parameter_optimizer_hyperparameters,
                "embedding_optimizer_hyperparameters": self.embedding_optimizer_hyperparameters,
        }

        
        self.base_view = HierarchicalView(self.datasets, superresolution_lr=self.superresolution_lr,
                level=0, **hierarchical_view_kwargs)

        self.active_view = self.base_view
        if self.hierarchical_levels is not None:
            if self.pretrained:
                self.hierarchy = reconstruct_hierarchy(self.reloaded_hierarchy, 
                        superresolution_lr=self.superresolution_lr, **hierarchical_view_kwargs)
            else:
                self.hierarchy = construct_hierarchy(self.base_view, chunks=2, levels=self.hierarchical_levels, 
                        superresolution_lr=self.superresolution_lr, **hierarchical_view_kwargs)
            self.active_view = self.hierarchy[self.hierarchical_levels - 1]

        self.datasets = self.active_view.datasets
        self.parameter_optimizer = self.active_view.parameter_optimizer
        self.embedding_optimizer = self.active_view.embedding_optimizer
        self.metagene_groups = self.active_view.metagene_groups
        self.metagene_tags = self.active_view.metagene_tags
        self.spatial_affinity_groups = self.active_view.spatial_affinity_groups
        self.spatial_affinity_tags = self.active_view.spatial_affinity_tags

        self.synchronize_datasets()
    
    def estimate_weights(self, use_neighbors=True):
        """Update embeddings (latent states) for each replicate.

        Args:
            use_neighbors: If specified, weight updates will take into account neighboring
                interactions. Default: ``True``
        """
        if self.verbose:
            print(f"{get_datetime()} Updating latent states")
        self.embedding_optimizer.update_embeddings(use_neighbors=use_neighbors)
        self.synchronize_datasets()

    def estimate_parameters(self, update_spatial_affinities: bool = True, differentiate_spatial_affinities: bool = True,
                            differentiate_metagenes: bool = True, simplex_projection_mode: bool = "exact",
                            edge_subsample_rate: Optional[float] = None):
        """Update parameters for each replicate.

        Args:
            update_spatial_affinities: If specified, spatial affinities will be updated during
                this iteration. Default: ``True``
            edge_subsample_rate: Fraction of adjacency matrix edges that will be included in
                optimization of ``Sigma_x_inv``.
        """
        logging.info(f'{get_datetime()}Updating model parameters')

        if update_spatial_affinities:
            if self.verbose:
                print(f"{get_datetime()} Updating spatial affinities")
            self.parameter_optimizer.update_spatial_affinity(differentiate_spatial_affinities=differentiate_spatial_affinities, subsample_rate=edge_subsample_rate)

        if self.verbose:
            print(f"{get_datetime()} Updating metagenes")
        self.parameter_optimizer.update_metagenes(differentiate_metagenes=differentiate_metagenes, simplex_projection_mode=simplex_projection_mode)
        if self.verbose:
            print(f"{get_datetime()} Updating sigma_yx")
        self.parameter_optimizer.update_sigma_yx()
        self.synchronize_datasets()

    def superresolve(self, differentiate_spatial_affinities: bool = True,
            update_spatial_affinities: bool = True, edge_subsample_rate: Optional[float] = None,
            n_epochs: int = 10000, tol: float = 1e-5):
        """Superresolve embeddings in hierarchical case.
        
        Works in a cascading manner, by superresolving the embeddings from lowest to highest resolutions in order.
        """

        for level in range(self.hierarchical_levels-2, -1, -1):
            view = self.hierarchy[level]
            view._propagate_parameters()
            view.parameter_optimizer.update_sigma_yx()
                
            view._superresolve_embeddings(n_epochs=n_epochs, tol=tol)
            
            if update_spatial_affinities:
                view.parameter_optimizer.update_spatial_affinity(differentiate_spatial_affinities=differentiate_spatial_affinities, subsample_rate=edge_subsample_rate)
        
        self.synchronize_datasets()

    def set_superresolution_lr(self, new_lr: float, target_level: Optional[int] = None):
        """Change learning rate for superresolution optimization.

        Can be used to change learning rate for all hierarchical levels (by default) or just
        the learning rate for a certain resolution.

        """

        change_all = (target_level == None)
        for level in self.hierarchy:
            if change_all or (level == target_level):
                self.hierarchy[level].superresolution_lr = new_lr

    def synchronize_datasets(self):
        """Synchronize datasets across all hierarchical levels."""

        self.base_view.synchronize_datasets()
        if self.hierarchical_levels is not None:
            for level, view in self.hierarchy.items():
                view.synchronize_datasets()

    def save_results(self, path2dataset: str, ignore_raw_data: bool =True):
        """Save datasets and learned Popari parameters to .h5ad file.

        Args:
            dataset_path: path to input ST datasets, to be stored in .h5ad format
            ignore_raw_data: if set, only learned parameters and embeddings will be saved; raw gene expression will be ignored.
        """

        if self.hierarchical_levels is None:
            save_anndata(path2dataset, self.datasets, ignore_raw_data=ignore_raw_data)
        else:
            merged_datasets = []
            level_names = self.hierarchy.keys()
            for level in level_names:
                view = self.hierarchy[level]
                datasets = view.datasets
                merged_dataset = merge_anndata(datasets, ignore_raw_data=ignore_raw_data)
                merged_datasets.append(merged_dataset)

            total_dataset = ad.concat(merged_datasets, label="hierarchical_level", keys=level_names, merge="unique", uns_merge="unique")
            total_dataset.uns["hierarchical_levels"] = self.hierarchical_levels

            total_dataset.write(path2dataset)

def load_trained_model(dataset_path: Union[str, Path], context=dict(device="cpu", dtype=torch.float64), **popari_kwargs):
    """Load trained Popari model for downstream analysis.

    Args:
        dataset_path: location of Popari results, stored as a .h5ad file.
    """

    # TODO: change this so that replicate_names can rename the datasets in the saved file...?

    merged_dataset = ad.read_h5ad(dataset_path)
    datasets = reloaded_hierarchy = hierarchical_levels = None
    if "hierarchical_levels" in merged_dataset.uns:
        indices = merged_dataset.obs.groupby("hierarchical_level").indices.values()
        unmerged_views = [merged_dataset[index].copy() for index in indices]
        reloaded_hierarchy = {}
        for unmerged_view in unmerged_views:
            level = unmerged_view.obs["hierarchical_level"].unique()[0]
            unmerged_datasets, unmerged_replicate_names = unmerge_anndata(unmerged_view)
            reloaded_hierarchy[level] = unmerged_datasets
        
        datasets = reloaded_hierarchy[0]
        replicate_names = [dataset.name for dataset in datasets]
    else:
        datasets, replicate_names = unmerge_anndata(merged_dataset)

    return load_pretrained(datasets, replicate_names, reloaded_hierarchy=reloaded_hierarchy, context=context, **popari_kwargs)

def load_pretrained(datasets: Sequence[PopariDataset], replicate_names: Sequence[str] = None,
                    context=dict(device="cpu", dtype=torch.float64), reloaded_hierarchy: Optional[dict] = None, **popari_kwargs):
    """Load pretrained Popari model from in-memory datasets.

    """
    first_dataset = datasets[0]
    saved_hyperparameters = first_dataset.uns["popari_hyperparameters"]

    metagene_groups = saved_hyperparameters["metagene_groups"]
    for group in metagene_groups:
        metagene_groups[group] = list(metagene_groups[group])

    spatial_affinity_groups = saved_hyperparameters["spatial_affinity_groups"]
    for group in spatial_affinity_groups:
        spatial_affinity_groups[group] = list(spatial_affinity_groups[group])

    new_kwargs = saved_hyperparameters.copy()
    for keyword in popari_kwargs:
        new_kwargs[keyword] = popari_kwargs[keyword]

    for noninitial_hyperparameter in ["prior_x", "metagene_tags", "spatial_affinity_tags"]:
        new_kwargs.pop(noninitial_hyperparameter)

    # metagene_mode = saved_hyperparameters["metagene_mode"]
    # K = saved_hyperparameters["K"]
    # lambda_Sigma_x_inv = saved_hyperparameters["lambda_Sigma_x_inv"]

    hierarchical_levels = None
    if reloaded_hierarchy is not None:
        hierarchical_levels = len(reloaded_hierarchy)

    trained_model = Popari(
        datasets=datasets,
        replicate_names=replicate_names,
        reloaded_hierarchy=reloaded_hierarchy,
        hierarchical_levels=hierarchical_levels,
        pretrained=True,
        # K=K,
        # metagene_mode=metagene_mode,
        # metagene_groups=metagene_groups,
        # spatial_affinity_groups=spatial_affinity_groups,
        # lambda_Sigma_x_inv=lambda_Sigma_x_inv,
        initial_context=context,
        torch_context=context,
        **new_kwargs
    )

    return trained_model
