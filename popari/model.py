import copy
import logging
import warnings
from pathlib import Path
from typing import Optional, Sequence, Union

import anndata as ad
import numpy as np
import torch
from anndata import AnnData
from torch import nn
from tqdm import trange

from popari._hierarchical_view import HierarchicalView, Hierarchy
from popari.io import (
    convert_legacy_anndata,
    load_anndata,
    load_anndata_hierarchy,
    merge_anndata,
    save_anndata,
    save_anndata_hierarchy,
)
from popari.schema import DEFAULT_SAMPLE_KEY
from popari.util import convert_numpy_to_pytorch_sparse_coo, get_datetime


class Popari(nn.Module):
    r"""Popari optimization model.

    Models spatial biological data using the NMF-HMRF formulation. Supports multiple
    fields-of-view (FOVs) and differential analysis.

    Example of including math in docstring (for use later):
    :math:`||A||_F = [\sum_{i,j} abs(a_{i,j})^2]^{1/2}`

    Attributes:
        K: number of metagenes to learn
        replicate_names: names of spatial datasets
        adata: unified multisample AnnData used for training.
        sample_key: observation column containing ordered sample identities.
        dataset_path: path to a unified AnnData dataset on disk.
        reloaded_hierarchy: data from previous hierarchical run of Popari.
        lambda_Sigma_x_inv: hyperparameter to balance importance of spatial information. Default: ``1e-4``
        pretrained: if set, attempts to load model state from input files. Default: ``False``
        initialization_method: algorithm to use for initializing metagenes and embeddings.
            Supports ``dummy``, ``kmeans``, ``svd``, ``leiden``, and ``ground_truth``. Default: ``leiden``
        hierarchical_levels: number of hierarchical levels to use. Default: ``1`` (non-hierarchical mode)
        metagene_groups: defines a grouping of replicates for the metagene optimization. If
            ``metagene_mode == "shared"``, then one set of metagenes will be created for each group;
            if ``metagene_mode == "differential",  then all replicates will have their own set of metagenes,
            but each group will share an ``M_bar``.
        spatial_affinity_groups: defines a grouping of replicates for the spatial affinity optimization.
            If ``spatial_affinity_mode == "shared lookup"``, then one set of spatial_affinities will be created for each group;
            if ``spatial_affinity_mode == "differential lookup"``,  then all replicates will have their own set of spatial
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
        binning_downsample_rate: ratio of number of spots at low resolution to high resolution when
            using hierarchical mode
        superresolution_lr: learning rate for optimization of ``X`` from low-res embeddings
        use_inplace_ops: if set, inplace PyTorch operations will be used to speed up computation
        random_state: seed for reproducibility of randomized computations. Default: ``0``
        verbose: level of verbosity to use during optimization. Default: ``0`` (no print statements)

    """

    def __init__(
        self,
        K: int,
        adata: Optional[ad.AnnData] = None,
        sample_key: str | None = None,
        replicate_names: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[ad.AnnData]] = None,
        dataset_path: Optional[Union[str, Path]] = None,
        reloaded_hierarchy: Optional[dict] = None,
        lambda_Sigma_x_inv: float = 1e-4,
        pretrained: bool = False,
        initialization_method: str = "leiden",
        hierarchical_levels: int = 1,
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
        lambda_Sigma_bar: float = 1e-3,
        spatial_affinity_lr: float = 1e-2,
        spatial_affinity_tol: float = 2e-3,
        spatial_affinity_constraint: Optional[str] = None,
        spatial_affinity_centering: bool = False,
        spatial_affinity_scaling: int = 10,
        spatial_affinity_regularization_power: int = 2,
        embedding_mini_iterations: int = 1000,
        embedding_acceleration_trick: bool = True,
        embedding_step_size_multiplier: float = 1.0,
        downsampling_method: str = "grid",
        binning_downsample_rate: float = 0.2,
        chunks: int = 2,
        superresolution_lr: float = 1e-1,
        use_inplace_ops: bool = True,
        random_state: int = 0,
        verbose: int = 0,
    ):
        super().__init__()

        self.use_inplace_ops = use_inplace_ops
        self.verbose = verbose

        if sum(value is not None for value in (adata, datasets, dataset_path)) != 1:
            raise ValueError("Specify exactly one of `adata`, `datasets`, or `dataset_path`.")

        if K <= 1:
            raise ValueError("`K` must be an integer value greater than 1.")

        if not torch_context:
            torch_context = dict(device="cpu", dtype=torch.float32)

        if not initial_context:
            initial_context = dict(device="cpu", dtype=torch.float32)

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
        self._configured_metagene_groups = metagene_groups
        self._configured_spatial_affinity_groups = spatial_affinity_groups

        self.embedding_step_size_multiplier = embedding_step_size_multiplier
        self.embedding_mini_iterations = embedding_mini_iterations
        self.embedding_acceleration_trick = embedding_acceleration_trick

        self.hierarchical_levels = hierarchical_levels
        self.reloaded_hierarchy = reloaded_hierarchy
        self.superresolution_lr = superresolution_lr
        self.downsampling_method = downsampling_method
        self.binning_downsample_rate = binning_downsample_rate
        self.chunks = chunks
        self.sample_key = sample_key

        if dataset_path is not None:
            self.load_dataset(dataset_path)
        elif adata is not None:
            self.load_anndata(adata)
        else:
            self.load_anndata_datasets(datasets, replicate_names)

        self.replicate_names = list(self._adata.popari.sample_names)
        self.num_replicates = len(self.replicate_names)

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

    @property
    def base_view(self):
        return self.views[self.hierarchical_levels - 1]

    @property
    def adata(self):
        """Unified AnnData at the finest hierarchy level."""

        if hasattr(self, "hierarchy"):
            return self.hierarchy[0].adata
        return self._adata

    @property
    def datasets(self):
        """Temporary compatibility view of the coarsest hierarchy level."""

        if hasattr(self, "views"):
            return self.base_view.datasets
        return [self._adata]

    @property
    def Ys(self):
        return self.base_view.Ys

    @property
    def betas(self):
        return self.base_view.betas

    @property
    def parameter_optimizer(self):
        return self.base_view.parameter_optimizer

    @property
    def embedding_optimizer(self):
        return self.base_view.embedding_optimizer

    @property
    def metagene_groups(self):
        if hasattr(self, "views"):
            return self.base_view.metagene_groups
        return self._configured_metagene_groups

    @property
    def metagene_tags(self):
        return self.base_view.metagene_tags

    @property
    def spatial_affinity_groups(self):
        if hasattr(self, "views"):
            return self.base_view.spatial_affinity_groups
        return self._configured_spatial_affinity_groups

    @property
    def spatial_affinity_tags(self):
        return self.base_view.spatial_affinity_tags

    def load_anndata_datasets(self, datasets: Sequence[ad.AnnData], replicate_names: Sequence[str]):
        """Compatibility adapter for legacy single-sample AnnData sequences.

        Args:
            datasets: spatial transcriptomics datasets in AnnData format (one for each FOV)
            replicate_names: names for all datasets/replicates

        """
        warnings.warn(
            "Passing datasets= is deprecated; pass one multisample AnnData with adata=.",
            DeprecationWarning,
            stacklevel=2,
        )
        datasets = list(datasets)
        if replicate_names is not None:
            if len(replicate_names) != len(datasets):
                raise ValueError("replicate_names must match the number of datasets.")
            for dataset, replicate_name in zip(datasets, replicate_names):
                dataset.popari.name = replicate_name
        resolved_sample_key = self.sample_key or DEFAULT_SAMPLE_KEY
        self._adata = merge_anndata(datasets, sample_key=resolved_sample_key)
        self.sample_key = resolved_sample_key

    def load_anndata(self, adata: ad.AnnData):
        """Load one unified Popari AnnData."""

        self._adata = convert_legacy_anndata(
            adata,
            sample_key=self.sample_key,
            copy=True,
        )
        self.sample_key = self._adata.popari.sample_key

    def load_dataset(self, dataset_path: Union[str, Path]):
        """Load dataset into Popari from saved .h5ad file.

        Args:
            dataset_path: path to input ST datasets, stored in .h5ad format

        """

        dataset_path = Path(dataset_path)

        self._adata = load_anndata(dataset_path, sample_key=self.sample_key)
        self.sample_key = self._adata.popari.sample_key

    def _initialize(
        self,
        pretrained=False,
        betas: Optional[Sequence[float]] = None,
        prior_x_modes: Optional[Sequence[str]] = None,
        method: str = "svd",
    ):
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
            "hierarchical_levels": self.hierarchical_levels,
            "betas": betas,
            "prior_x_modes": prior_x_modes,
            "use_inplace_ops": self.use_inplace_ops,
            "method": method,
            "pretrained": self.pretrained,
            "verbose": self.verbose,
            "metagene_groups": self.metagene_groups,
            "spatial_affinity_groups": self.spatial_affinity_groups,
            "superresolution_lr": self.superresolution_lr,
            "sample_key": self.sample_key,
            "parameter_optimizer_hyperparameters": self.parameter_optimizer_hyperparameters,
            "embedding_optimizer_hyperparameters": self.embedding_optimizer_hyperparameters,
        }

        bin_assignment_kwargs = {}
        if self.downsampling_method == "grid":
            bin_assignment_kwargs["chunks"] = self.chunks
        elif self.downsampling_method == "partition":
            bin_assignment_kwargs["adjacency_list_key"] = "adjacency_list"

        if self.pretrained:
            if self.reloaded_hierarchy is None:
                self.reloaded_hierarchy = {0: self._adata}
            self.hierarchy = Hierarchy.reconstruct(
                self.reloaded_hierarchy,
                **hierarchical_view_kwargs,
            )
        else:
            base_view = HierarchicalView(self.adata, level=0, **hierarchical_view_kwargs)
            self.hierarchy = Hierarchy(
                downsampling_method=self.downsampling_method,
                base_view=base_view,
                **hierarchical_view_kwargs,
            )

            self.hierarchy.construct(
                levels=self.hierarchical_levels,
                downsample_rate=self.binning_downsample_rate,
                **bin_assignment_kwargs,
            )

        self.views = nn.ModuleList([self.hierarchy[level] for level in range(self.hierarchical_levels)])

        self.synchronize_datasets()

    def estimate_weights(self, use_neighbors: bool = True, synchronize: bool = True):
        """Update embeddings (latent states) for each replicate.

        Args:
            use_neighbors: If specified, weight updates will take into account neighboring
                interactions. Default: ``True``

        """
        if self.verbose:
            print(f"{get_datetime()} Updating latent states")
        self.embedding_optimizer.update_embeddings(use_neighbors=use_neighbors)

        if synchronize:
            self.synchronize_datasets()

    def estimate_parameters(
        self,
        update_spatial_affinities: bool = True,
        differentiate_spatial_affinities: bool = True,
        differentiate_metagenes: bool = True,
        simplex_projection_mode: bool = "exact",
        edge_subsample_rate: Optional[float] = None,
        synchronize: bool = True,
        spatial_affinity_epochs: int = 1000,
    ):
        """Update parameters for each replicate.

        Args:
            update_spatial_affinities: If specified, spatial affinities will be updated during
                this iteration. Default: ``True``
            edge_subsample_rate: Fraction of adjacency matrix edges that will be included in
                optimization of ``Sigma_x_inv``.
            spatial_affinity_epochs: Maximum number of inner optimization epochs for
                ``Sigma_x_inv``. Default: ``1000``.

        """
        logging.info(f"{get_datetime()}Updating model parameters")

        if update_spatial_affinities:
            if self.verbose:
                print(f"{get_datetime()} Updating spatial affinities")
            self.parameter_optimizer.update_spatial_affinity(
                differentiate_spatial_affinities=differentiate_spatial_affinities,
                subsample_rate=edge_subsample_rate,
                n_epochs=spatial_affinity_epochs,
            )

        if self.verbose:
            print(f"{get_datetime()} Updating metagenes")

        self.parameter_optimizer.update_metagenes(
            differentiate_metagenes=differentiate_metagenes,
            simplex_projection_mode=simplex_projection_mode,
        )

        if self.verbose:
            print(f"{get_datetime()} Updating sigma_yx")

        self.parameter_optimizer.update_sigma_yx()

        if synchronize:
            self.synchronize_datasets()

    def superresolve(
        self,
        differentiate_spatial_affinities: bool = True,
        update_spatial_affinities: bool = True,
        edge_subsample_rate: Optional[float] = None,
        use_manual_gradients: bool = True,
        n_epochs: int = 10000,
        miniepochs: int = None,
        tol: float = 1e-5,
    ):
        """Superresolve embeddings in hierarchical case.

        Works in a cascading manner, by superresolving the embeddings from
        lowest to highest resolutions in order.

        """

        if miniepochs is None:
            miniepochs = min(100, n_epochs)

        effective_epochs = n_epochs // miniepochs + 1

        for level in range(self.hierarchical_levels - 2, -1, -1):
            if self.verbose:
                print(f"{get_datetime()} Superresolving level {level} embeddings")
            view = self.hierarchy[level]
            view._propagate_parameters()

            progress_bar = trange(effective_epochs, leave=True, disable=not self.verbose, miniters=10000)
            previous_losses = np.full(view.num_replicates, np.inf)
            for epoch in progress_bar:
                view.parameter_optimizer.update_sigma_yx()
                losses = view._superresolve_embeddings(
                    n_epochs=miniepochs,
                    tol=tol,
                    use_manual_gradients=use_manual_gradients,
                    verbose=self.verbose,
                )
                formatted_losses = [f"{loss:.1e}" for loss in losses]
                formatted_deltas = [f"{delta:.1e}" for delta in ((previous_losses - losses) / losses)]
                description = (
                    f"Updating weights hierarchically: loss = {formatted_losses} " f"%δloss = {formatted_deltas} "
                )
                previous_losses = losses
                progress_bar.set_description(description)

            pretrained_embeddings = [
                view.embedding_optimizer.embedding_state[sample].clone() for sample in view.replicate_names
            ]
            view.parameter_optimizer.spatial_affinity.initialize(
                pretrained_embeddings,
                view.parameter_optimizer.spatial_affinity_bar,
            )

            if update_spatial_affinities:
                view.parameter_optimizer.update_spatial_affinity(
                    differentiate_spatial_affinities=differentiate_spatial_affinities,
                    subsample_rate=edge_subsample_rate,
                )

        self.synchronize_datasets()

    def nll(self, level: int = 0, use_spatial: bool = False):
        """Compute the nll for the current configuration of model parameters."""

        with torch.no_grad():
            return self.forward(level=level, use_spatial=use_spatial).reshape(1).cpu().numpy()

    def forward(self, level: int = 0, use_spatial: bool = False):
        """Compute the current objective for a hierarchy level."""

        view = self.hierarchy[level]
        return view(use_spatial=use_spatial)

    def set_superresolution_lr(self, new_lr: float, target_level: Optional[int] = None):
        """Change learning rate for superresolution optimization.

        Can be used to change learning rate for all hierarchical levels (by
        default) or just the learning rate for a certain resolution.

        """

        change_all = target_level == None
        for level in range(self.hierarchical_levels):
            if change_all or (level == target_level):
                self.hierarchy[level].superresolution_lr = new_lr

    def synchronize_datasets(self):
        """Synchronize unified AnnData objects across hierarchy levels."""

        self.base_view.synchronize_datasets()
        if self.hierarchical_levels > 1:
            for level in range(self.hierarchical_levels):
                self.hierarchy[level].synchronize_datasets()

    def save_results(self, dataset_path: str, ignore_raw_data: bool = True):
        """Save datasets and learned Popari parameters to disk.

        Args:
            dataset_path: where to save results. If results are non-hierarchical, file extension
                will automatically be changed to ``.h5ad``. Otherwise, if results are hierarchical,
                ``dataset_path`` will be interpreted as a path to a subfolder where separate
                ``.h5ad`` files will be stored for each hierarchical level.
            ignore_raw_data: if set, only learned parameters and embeddings will be saved; raw gene expression will be ignored.

        """

        dataset_path = Path(dataset_path)
        path_without_extension = dataset_path.parent / dataset_path.stem

        self.synchronize_datasets()

        if self.hierarchical_levels == 1:
            if self.verbose:
                print(f"{get_datetime()} Writing results to {path_without_extension}.h5ad")
            save_anndata(
                f"{path_without_extension}.h5ad",
                self.adata,
                ignore_raw_data=ignore_raw_data,
                sample_key=self.sample_key,
            )
        else:
            save_anndata_hierarchy(
                path_without_extension,
                {level: self.hierarchy[level].adata for level in range(self.hierarchical_levels)},
                ignore_raw_data=ignore_raw_data,
                sample_key=self.sample_key,
            )

    def _reload_expression(self, raw_adata: AnnData | Sequence[AnnData]):
        """Can be used to recover expression values for training model if saved
        with `ignore_raw_data=True`"""
        if not isinstance(raw_adata, AnnData):
            raw_adata = merge_anndata(raw_adata, sample_key=self.sample_key)
        raw_adata = convert_legacy_anndata(
            raw_adata,
            sample_key=self.sample_key,
            copy=False,
        )

        high_resolution_view = self.hierarchy[0]
        if not raw_adata.obs_names.equals(high_resolution_view.adata.obs_names):
            raw_adata = raw_adata[high_resolution_view.adata.obs_names].copy()
        high_resolution_view.adata.X = raw_adata.X.copy()
        for index, sample in enumerate(high_resolution_view.replicate_names):
            indices = high_resolution_view.sample_axis.indices(sample)
            expression = high_resolution_view.adata.X[indices]
            num_cells = len(indices)

            Y = convert_numpy_to_pytorch_sparse_coo(expression, self.context)
            Y *= (self.K * 1) / (Y.sum() / num_cells)
            high_resolution_view.Ys[index] = Y

        high_resolution_view.parameter_optimizer.update_sigma_yx()  # This is necessary, since `sigma_yx` isn't loaded correctly from disk

        for level in range(self.hierarchical_levels - 1):
            view = self.hierarchy[level]
            low_res_view = self.hierarchy[level + 1]
            assignments = low_res_view.adata.obsm["bin_assignments"]
            low_res_view.adata.X = assignments @ view.adata.X
            for index, (sample, previous_Y) in enumerate(
                zip(view.replicate_names, view.Ys),
            ):
                bin_assignments = assignments[low_res_view.sample_axis.indices(sample)][
                    :,
                    view.sample_axis.indices(sample),
                ]
                bin_assignments_tensor = convert_numpy_to_pytorch_sparse_coo(
                    bin_assignments,
                    context=self.initial_context,
                )

                binned_Y = bin_assignments_tensor @ previous_Y
                low_res_view.Ys[index] = binned_Y

            low_res_view.parameter_optimizer.update_sigma_yx()


class SpiceMix(Popari):
    """Wrapper to produce SpiceMix hyperparameter configuration."""

    def __init__(self, **spicemix_hyperparameters):
        spatial_affinity_mode = "shared lookup"
        if "spatial_affinity_mode" in spicemix_hyperparameters:
            spicemix_hyperparameters.pop("spatial_affinity_mode")

        super().__init__(spatial_affinity_mode="shared lookup", **spicemix_hyperparameters)


def load_trained_model(
    dataset_path: Union[str, Path],
    context=dict(device="cpu", dtype=torch.float64),
    **popari_kwargs,
):
    """Load trained Popari model for downstream analysis.

    Args:
        dataset_path: location of Popari results, stored as a .h5ad file.

    """

    # TODO: change this so that replicate_names can rename the datasets in the saved file...?

    dataset_path = Path(dataset_path)
    path_without_extension = dataset_path.parent / dataset_path.stem

    reloaded_hierarchy = load_anndata_hierarchy(path_without_extension)
    popari_kwargs["hierarchical_levels"] = len(reloaded_hierarchy)

    return load_pretrained(
        reloaded_hierarchy[0],
        reloaded_hierarchy=reloaded_hierarchy,
        context=context,
        **popari_kwargs,
    )


def load_pretrained(
    adata: AnnData,
    context=dict(device="cpu", dtype=torch.float64),
    reloaded_hierarchy: Optional[dict] = None,
    **popari_kwargs,
):
    """Load a pretrained Popari model from a unified AnnData."""

    saved_hyperparameters = copy.deepcopy(adata.uns["popari_hyperparameters"])

    metagene_groups = saved_hyperparameters["metagene_groups"]
    for group in metagene_groups:
        metagene_groups[group] = list(metagene_groups[group])

    spatial_affinity_groups = saved_hyperparameters["spatial_affinity_groups"]
    for group in spatial_affinity_groups:
        spatial_affinity_groups[group] = list(spatial_affinity_groups[group])

    new_kwargs = saved_hyperparameters | popari_kwargs

    for noninitial_hyperparameter in ["prior_x", "metagene_tags", "spatial_affinity_tags"]:
        new_kwargs.pop(noninitial_hyperparameter)

    trained_model = Popari(
        adata=adata,
        sample_key=adata.popari.sample_key,
        reloaded_hierarchy=reloaded_hierarchy,
        pretrained=True,
        initial_context=context,
        torch_context=context,
        **new_kwargs,
    )

    return trained_model


def from_pretrained(pretrained_model: Popari, popari_context: dict = None, lambda_Sigma_bar: float = 1e-3):
    """Initialize Popari object from a SpiceMix pretrained model."""

    adata = pretrained_model.adata.copy()
    reloaded_hierarchy = {
        level: pretrained_model.hierarchy[level].adata.copy() for level in range(pretrained_model.hierarchical_levels)
    }

    return load_pretrained(
        adata,
        reloaded_hierarchy=reloaded_hierarchy,
        hierarchical_levels=pretrained_model.hierarchical_levels,
        metagene_mode="differential",
        spatial_affinity_mode="differential lookup",
        context=popari_context,
        lambda_Sigma_bar=lambda_Sigma_bar,
    )
