import copy
from pathlib import Path
from typing import Optional, Sequence, Union

import anndata as ad
import numpy as np
import torch
from anndata import AnnData
from torch import nn

from popari._hierarchical_level import HierarchicalLevel, Hierarchy
from popari._sparse import convert_numpy_to_pytorch_sparse_coo
from popari.io import load_anndata, load_anndata_hierarchy
from popari.schema import SAMPLE_KEY_KEY, SCHEMA_VERSION, SCHEMA_VERSION_KEY


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
            Supports ``dummy``, ``kmeans``, ``svd``, ``leiden``, ``leiden_fast``, and ``ground_truth``.
            ``leiden_fast`` uses the igraph backend with two iterations. Default: ``leiden``
        hierarchical_levels: number of hierarchical levels to use. Default: ``1`` (non-hierarchical mode)
        groups: exhaustive partition of samples sharing spatial-affinity parameters. ``None`` creates one shared
            parameter; ``"disjoint"`` creates one parameter per sample.
        regularization_groups: groups of spatial-affinity parameter names regularized toward common means.
        betas: weighting of each dataset during optimization. Defaults to equally weighting each dataset
        prior_x_modes: family of prior distribution for embeddings of each dataset
        M_constraint: constraint on columns of M. Default: ``simplex``
        sigma_yx_inv_mode: form of sigma_yx_inv parameter. Default: ``separate``
        torch_context: keyword args to use of PyTorch tensors during training.
        initial_context: keyword args to use during initialization of PyTorch tensors.
        spatial_affinity_parameterization: internal affinity representation, either ``full`` or ``factorized``.
        spatial_affinity_rank: rank of the factorized spatial representation. Defaults to ``K``.
        lambda_Sigma_bar: hyperparameter constraining affinities within ``regularization_groups``. Default: ``1e-3``
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
        use_inplace_ops: if set, inplace PyTorch operations will be used to speed up computation
        random_state: seed for reproducibility of randomized computations. Default: ``0``
        verbose: level of verbosity to use during optimization. Default: ``0`` (no print statements)

    """

    def __init__(
        self,
        K: int,
        adata: Optional[ad.AnnData] = None,
        sample_key: str | None = None,
        dataset_path: Optional[Union[str, Path]] = None,
        reloaded_hierarchy: Optional[dict] = None,
        lambda_Sigma_x_inv: float = 1e-4,
        pretrained: bool = False,
        initialization_method: str = "leiden",
        hierarchical_levels: int = 1,
        groups: dict | str | None = None,
        regularization_groups: dict | None = None,
        betas: Optional[Sequence[float]] = None,
        prior_x_modes: Optional[Sequence[str]] = None,
        M_constraint: str = "simplex",
        sigma_yx_inv_mode: str = "separate",
        torch_context: Optional[dict] = None,
        initial_context: Optional[dict] = None,
        lambda_Sigma_bar: float = 1e-3,
        spatial_affinity_lr: float = 1e-2,
        spatial_affinity_tol: float = 2e-3,
        spatial_affinity_constraint: Optional[str] = None,
        spatial_affinity_centering: bool = False,
        spatial_affinity_scaling: int = 10,
        spatial_affinity_regularization_power: int = 2,
        spatial_affinity_parameterization: str = "full",
        spatial_affinity_rank: int | None = None,
        embedding_mini_iterations: int = 1000,
        embedding_acceleration_trick: bool = True,
        embedding_step_size_multiplier: float = 1.0,
        downsampling_method: str = "grid",
        binning_downsample_rate: float = 0.2,
        chunks: int = 2,
        use_inplace_ops: bool = True,
        random_state: int = 0,
        verbose: int = 0,
    ):
        super().__init__()

        self.use_inplace_ops = use_inplace_ops
        self.verbose = verbose

        if (adata is None) == (dataset_path is None):
            raise ValueError("Specify exactly one of `adata` or `dataset_path`.")

        if K <= 1:
            raise ValueError("`K` must be an integer value greater than 1.")
        if spatial_affinity_parameterization not in {"full", "factorized"}:
            raise ValueError("spatial_affinity_parameterization must be 'full' or 'factorized'.")
        if spatial_affinity_parameterization == "factorized" and spatial_affinity_centering:
            raise ValueError("spatial_affinity_centering is not supported for factorized spatial affinities.")
        if spatial_affinity_rank is None:
            spatial_affinity_rank = K
        if not 1 <= spatial_affinity_rank <= K:
            raise ValueError(
                f"spatial_affinity_rank must satisfy 1 <= rank <= K; received {spatial_affinity_rank} for K={K}.",
            )

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
        self.spatial_affinity_parameterization = spatial_affinity_parameterization
        self.spatial_affinity_rank = spatial_affinity_rank
        self.M_constraint = M_constraint
        self.sigma_yx_inv_mode = sigma_yx_inv_mode
        self.pretrained = pretrained

        self.embedding_step_size_multiplier = embedding_step_size_multiplier
        self.embedding_mini_iterations = embedding_mini_iterations
        self.embedding_acceleration_trick = embedding_acceleration_trick

        self.hierarchical_levels = hierarchical_levels
        self.reloaded_hierarchy = reloaded_hierarchy
        self.downsampling_method = downsampling_method
        self.binning_downsample_rate = binning_downsample_rate
        self.chunks = chunks
        self.sample_key = sample_key
        self.dataset_path = None

        if dataset_path is not None:
            self.load_dataset(dataset_path)
        else:
            self.load_anndata(adata)

        self.replicate_names = list(self._adata.popari.sample_names)
        self.num_replicates = len(self.replicate_names)

        self._build_hierarchy(
            betas=betas,
            prior_x_modes=prior_x_modes,
            method=initialization_method,
            pretrained=pretrained,
            groups=groups,
            regularization_groups=regularization_groups,
        )

    @property
    def adata(self):
        """Unified AnnData at the finest hierarchy level."""

        if hasattr(self, "hierarchy"):
            return self.hierarchy[0].adata
        return self._adata

    @property
    def groups(self):
        return self.hierarchy[-1].groups

    @property
    def regularization_groups(self):
        return self.hierarchy[-1].regularization_groups

    def load_anndata(self, adata: ad.AnnData):
        """Load one unified Popari AnnData."""

        self._adata = adata
        if self.sample_key is not None:
            self._adata.uns[SAMPLE_KEY_KEY] = self.sample_key
        self._adata.uns[SCHEMA_VERSION_KEY] = SCHEMA_VERSION
        self._adata.popari.validate()
        self.sample_key = self._adata.popari.sample_key

    def load_dataset(self, dataset_path: Union[str, Path]):
        """Load dataset into Popari from saved .h5ad file.

        Args:
            dataset_path: path to input ST datasets, stored in .h5ad format

        """

        dataset_path = Path(dataset_path)

        self._adata = load_anndata(dataset_path)
        self.dataset_path = dataset_path
        self.sample_key = self._adata.popari.sample_key

    def _build_hierarchy(
        self,
        pretrained=False,
        betas: Optional[Sequence[float]] = None,
        prior_x_modes: Optional[Sequence[str]] = None,
        method: str = "svd",
        groups: dict | str | None = None,
        regularization_groups: dict | None = None,
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
            "groups": groups,
            "regularization_groups": regularization_groups,
            "sample_key": self.sample_key,
            "lambda_Sigma_x_inv": self.lambda_Sigma_x_inv,
            "lambda_Sigma_bar": self.lambda_Sigma_bar,
            "spatial_affinity_lr": self.spatial_affinity_lr,
            "spatial_affinity_tol": self.spatial_affinity_tol,
            "spatial_affinity_constraint": self.spatial_affinity_constraint,
            "spatial_affinity_centering": self.spatial_affinity_centering,
            "spatial_affinity_scaling": self.spatial_affinity_scaling,
            "spatial_affinity_regularization_power": self.spatial_affinity_regularization_power,
            "spatial_affinity_parameterization": self.spatial_affinity_parameterization,
            "spatial_affinity_rank": self.spatial_affinity_rank,
            "M_constraint": self.M_constraint,
            "sigma_yx_inv_mode": self.sigma_yx_inv_mode,
            "embedding_step_size_multiplier": self.embedding_step_size_multiplier,
            "embedding_mini_iterations": self.embedding_mini_iterations,
            "embedding_acceleration_trick": self.embedding_acceleration_trick,
        }

        bin_assignment_kwargs = {}
        if self.downsampling_method == "grid":
            bin_assignment_kwargs["chunks"] = self.chunks

        if self.pretrained:
            if self.reloaded_hierarchy is None:
                self.reloaded_hierarchy = {0: self._adata}
            self.hierarchy = Hierarchy.reconstruct(
                self.reloaded_hierarchy,
                **hierarchical_view_kwargs,
            )
        else:
            base_view = HierarchicalLevel(self.adata, level=0, **hierarchical_view_kwargs)
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

    def nll(self, level: int = 0, use_spatial: bool = False):
        """Compute the joint negative log pseudolikelihood at one hierarchy
        level."""

        with torch.no_grad():
            return self.forward(level=level, use_spatial=use_spatial).reshape(1).cpu().numpy()

    def forward(self, level: int = 0, use_spatial: bool = False):
        """Compute the joint negative log pseudolikelihood at one hierarchy
        level."""

        view = self.hierarchy[level]
        return view(use_spatial=use_spatial)

    def materialize_results(self, *, force: bool = False) -> dict[int, AnnData]:
        """Write learned tensor state into and return each hierarchy-level
        AnnData."""

        for level in range(self.hierarchical_levels):
            view = self.hierarchy[level]
            view.materialize_results(force=force)
            view.adata.uns["popari_hyperparameters"] = self._result_hyperparameters(view)
        return {level: self.hierarchy[level].adata for level in range(self.hierarchical_levels)}

    def _result_hyperparameters(self, view: HierarchicalLevel) -> dict:
        """Return constructor metadata needed to reload a materialized level."""

        return {
            "prior_x": {
                sample: view.prior_xs[index][0].cpu().detach().numpy()
                for index, sample in enumerate(view.replicate_names)
            },
            "K": self.K,
            "use_inplace_ops": self.use_inplace_ops,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "lambda_Sigma_x_inv": self.lambda_Sigma_x_inv,
            "lambda_Sigma_bar": self.lambda_Sigma_bar,
            "spatial_affinity_lr": self.spatial_affinity_lr,
            "spatial_affinity_tol": self.spatial_affinity_tol,
            "spatial_affinity_constraint": self.spatial_affinity_constraint,
            "spatial_affinity_centering": self.spatial_affinity_centering,
            "spatial_affinity_scaling": self.spatial_affinity_scaling,
            "spatial_affinity_regularization_power": self.spatial_affinity_regularization_power,
            "spatial_affinity_parameterization": self.spatial_affinity_parameterization,
            "spatial_affinity_rank": self.spatial_affinity_rank,
            "M_constraint": self.M_constraint,
            "sigma_yx_inv_mode": self.sigma_yx_inv_mode,
            "groups": view.groups,
            "regularization_groups": view.regularization_groups,
            "embedding_step_size_multiplier": self.embedding_step_size_multiplier,
            "embedding_mini_iterations": self.embedding_mini_iterations,
            "embedding_acceleration_trick": self.embedding_acceleration_trick,
        }

    def _reload_expression(self, raw_adata: AnnData):
        """Can be used to recover expression values for training model if saved
        with `ignore_raw_data=True`"""
        if not isinstance(raw_adata, AnnData):
            raise TypeError("raw_adata must be one unified AnnData object.")
        raw_adata = raw_adata.copy()
        raw_adata.uns[SAMPLE_KEY_KEY] = self.sample_key
        raw_adata.uns[SCHEMA_VERSION_KEY] = SCHEMA_VERSION
        raw_adata.popari.validate()

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

        high_resolution_view._recompute_observation_noise()

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

            low_res_view._recompute_observation_noise()


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
    if "groups" not in saved_hyperparameters:
        legacy_groups = {
            str(name): list(samples) for name, samples in saved_hyperparameters.pop("spatial_affinity_groups").items()
        }
        legacy_mode = saved_hyperparameters.pop("spatial_affinity_mode")
        if legacy_mode == "shared lookup":
            saved_hyperparameters["groups"] = legacy_groups
            saved_hyperparameters["regularization_groups"] = {}
        elif legacy_mode == "differential lookup":
            saved_hyperparameters["groups"] = "disjoint"
            saved_hyperparameters["regularization_groups"] = legacy_groups
        elif legacy_mode == "differential group lookup":
            saved_hyperparameters["groups"] = legacy_groups
            saved_hyperparameters["regularization_groups"] = {"_global": list(legacy_groups)}
        else:
            raise ValueError(f"Unsupported saved spatial-affinity mode: {legacy_mode!r}.")
    else:
        saved_hyperparameters["groups"] = {
            str(name): list(samples) for name, samples in saved_hyperparameters["groups"].items()
        }
        saved_hyperparameters["regularization_groups"] = {
            str(name): list(parameter_names)
            for name, parameter_names in saved_hyperparameters.get("regularization_groups", {}).items()
        }

    new_kwargs = saved_hyperparameters | popari_kwargs

    for noninitial_hyperparameter in [
        "prior_x",
        "metagene_groups",
        "metagene_tags",
        "metagene_mode",
        "lambda_M",
        "spatial_affinity_tags",
        "superresolution_lr",
    ]:
        new_kwargs.pop(noninitial_hyperparameter, None)

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

    pretrained_model.materialize_results()
    adata = pretrained_model.adata.copy()
    reloaded_hierarchy = {
        level: pretrained_model.hierarchy[level].adata.copy() for level in range(pretrained_model.hierarchical_levels)
    }

    return load_pretrained(
        adata,
        reloaded_hierarchy=reloaded_hierarchy,
        hierarchical_levels=pretrained_model.hierarchical_levels,
        groups="disjoint",
        regularization_groups={"_default": list(pretrained_model.replicate_names)},
        context=popari_context,
        lambda_Sigma_bar=lambda_Sigma_bar,
    )
