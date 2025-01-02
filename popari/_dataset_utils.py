from functools import partial, wraps
from typing import Callable, Optional, Sequence

import anndata as ad
import matplotlib.patches as patches
import mpl_toolkits.axisartist.angle_helper as angle_helper
import mpl_toolkits.axisartist.floating_axes as floating_axes
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce
import seaborn as sns
import squidpy as sq
from matplotlib import cm, colormaps
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.transforms import Affine2D
from mpl_toolkits.axisartist.grid_finder import DictFormatter, FixedLocator, MaxNLocator
from scipy.sparse import csr_array, csr_matrix, issparse
from scipy.stats import wilcoxon, zscore
from sklearn.metrics import adjusted_rand_score, confusion_matrix, precision_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from popari._binning_utils import chunked_downsample_on_grid, filter_gridpoints
from popari._popari_dataset import PopariDataset
from popari.util import (
    compute_neighborhood_enrichment,
    concatenate,
    get_metagene_signature,
    normalize_expression_by_threshold,
    run_gsea,
    smooth_labels,
    smooth_metagene_expression,
    unconcatenate,
)


def setup_squarish_axes(num_axes, **subplots_kwargs):
    """Create matplotlib subplots as squarely as possible."""

    height = int(np.sqrt(num_axes))
    width = num_axes // height
    height += width * height != num_axes

    constrained_layout = (
        True if "constrained_layout" not in subplots_kwargs else subplots_kwargs.pop("constrained_layout")
    )
    dpi = subplots_kwargs.pop("dpi", 300)
    sharex = subplots_kwargs.pop("sharex", True)
    sharey = subplots_kwargs.pop("sharey", True)

    fig, axes = plt.subplots(
        height,
        width,
        squeeze=False,
        constrained_layout=constrained_layout,
        dpi=dpi,
        sharex=sharex,
        sharey=sharey,
        **subplots_kwargs,
    )

    return fig, axes


def copy_annotations(original_dataset, updated_dataset, annotations: Optional[Sequence[str]] = None):
    """Copy updates from one AnnData to another.

    Note that this is a bit wasteful, as it copies every key in a given
    annotation layer.

    """
    if annotations is None:
        annotations = ()

    for annotation in annotations:
        original_annotation = getattr(original_dataset, annotation)
        updated_annotation = getattr(updated_dataset, annotation)

        for key in updated_annotation:
            original_annotation[key] = updated_annotation[key]


def for_model(function=None, *, return_outputs: bool = False):
    if function is None:
        return partial(for_model, return_outputs=return_outputs)

    @wraps(function)
    def model_wrapper(trained_model: "Popari", *args, **kwargs):
        level = kwargs.pop("level", 0)
        datasets = trained_model.hierarchy[level].datasets

        outputs = function(datasets, *args, **kwargs)

        return outputs if return_outputs else None

    return model_wrapper


def enable_joint(function=None, *, annotations: Optional[Sequence[str]] = None):
    """Decorator to extend functions that operate on lists of dataset to work on
    a joint, merged dataset.

    Args:
        function: function to decorate. Must accept a list of datasets as its first argument.

    """
    if function is None:
        return partial(enable_joint, annotations=annotations)

    @wraps(function)
    def joint_wrapper(datasets, *args, **kwargs):
        """
        Args:
            joint: if `True`, apply `function` to merged dataset.
        """
        joint = kwargs.pop("joint", False)
        if joint:
            original_datasets = datasets
            merged_dataset = concatenate(datasets)
            datasets = [merged_dataset]

        outputs = function(datasets, *args, **kwargs)

        if joint:
            unmerged_datasets = unconcatenate(merged_dataset)

            for unmerged_dataset, original_dataset in zip(unmerged_datasets, original_datasets):
                copy_annotations(original_dataset, unmerged_dataset, annotations=annotations)

            outputs = original_datasets, outputs

        return outputs

    return joint_wrapper


def broadcast(function):
    """Decorator to extend a function from a single dataset to multiple
    datasets.

    TODO: Doesn't work in the case that the function's return values are meaningful...

    """

    @wraps(function)
    def broadcast_wrapper(datasets, *args, **kwargs):
        """"""

        outputs = []
        for dataset in datasets:
            output = function(dataset, *args, **kwargs)
            outputs.append(output)

        if len(outputs) == 1:
            outputs = outputs[0]

        return outputs

    return broadcast_wrapper


def broadcast_plottable(function):
    """Decorator to extend a plottable function from a single dataset to
    multiple datasets.

    TODO: Doesn't work in the case that the function's return values are meaningful...

    """

    @wraps(function)
    def broadcast_wrapper(datasets, *args, **kwargs):
        """"""
        axes = kwargs.pop("axes", None)

        sharex = kwargs.pop("sharex", False)
        sharey = kwargs.pop("sharey", False)

        fig = None
        if axes is None:
            fig, axes = setup_squarish_axes(len(datasets), sharex=sharex, sharey=sharey)

        outputs = []
        for dataset, ax in zip(datasets, axes.flat):
            output = function(dataset, *args, ax=ax, **kwargs)
            outputs.append(output)

        if len(outputs) == 1:
            outputs = outputs[0]

        return fig, outputs

    return broadcast_wrapper


def _preprocess_embeddings(
    datasets: Sequence[PopariDataset],
    input_key="X",
    normalized_key="normalized_X",
):
    """Normalize embeddings per each cell.

    This step helps to make cell embeddings comparable, and facilitates
    downstream tasks like clustering.

    """
    for dataset in datasets:
        if "X" not in dataset.obsm:
            raise ValueError("Must initialize embeddings before normalizing them.")

        normalized_embeddings = zscore(dataset.obsm[input_key])
        nan_mask = np.isnan(normalized_embeddings)
        normalized_embeddings[nan_mask] = 0

        dataset.obsm[normalized_key] = normalized_embeddings
        sc.pp.neighbors(dataset, use_rep=normalized_key)

    return datasets


def _plot_metagene_embedding(
    datasets: Sequence[PopariDataset],
    metagene_index: int,
    axes: Optional[Sequence[Axes]] = None,
    **scatterplot_kwargs,
):
    r"""Plot a single metagene in-situ across all datasets.

    Args:
        datasets: list of datasets to plot
        metagene_index: the index of the metagene to plot.
        axes: A predefined set of matplotlib axes to plot on.

    """

    legend = scatterplot_kwargs.pop("legend", False)
    default_s = scatterplot_kwargs.pop("s", None)
    linewidth = scatterplot_kwargs.pop("linewidth", 0)
    palette = scatterplot_kwargs.pop("palette", "viridis")
    if axes is None:
        fig, axes = setup_squarish_axes(len(datasets), sharex=False, sharey=False)

    else:
        fig = axes.flat[0].get_figure()

    for index in range(len(datasets), axes.size):
        axes.flat[index].axis("off")

    for dataset, ax in zip(datasets, axes.flat):
        if default_s is None:
            s = round(20000 / len(dataset))
        else:
            s = default_s

        ax.set_aspect("equal", "box")
        ax.invert_yaxis()
        ax.set_xticks([], [])  # note you need two lists one for the positions and one for the labels
        ax.set_yticks([], [])  # same for y ticks
        dataset.plot_metagene_embedding(
            metagene_index,
            legend=legend,
            s=s,
            linewidth=linewidth,
            palette=palette,
            ax=ax,
            **scatterplot_kwargs,
        )

    fig.suptitle(f"Metagene {metagene_index}")

    return fig


@enable_joint(annotations=["obs", "uns", "obsp"])
def _leiden(
    datasets: Sequence[PopariDataset],
    resolution: float = 1.0,
    tolerance: float = 0.05,
    **kwargs,
):
    r"""Compute Leiden clustering for all datasets.

    Args:
        trained_model: the trained Popari model.
        joint: if `True`, jointly cluster the spots
        datasets: list of datasets to cluster
        use_rep: the key in the ``.obsm`` dataframe to ue as input to the Leiden clustering algorithm.
        resolution: the resolution to use for Leiden clustering. Higher values yield finer clusters..

    """
    # n_iterations = kwargs.pop("n_iterations", 2) # TODO: add these parameters back in after resolving issue
    _cluster.__wrapped__(
        datasets,
        resolution=resolution,
        method="leiden",
        tolerance=tolerance,
        # flavor="igraph",
        # n_iterations=n_iterations,
        **kwargs,
    )


@enable_joint(annotations=["obs", "uns", "obsp"])
def _cluster(
    datasets: Sequence[PopariDataset],
    use_rep="normalized_X",
    method: str = "leiden",
    n_neighbors: int = 20,
    target_clusters: Optional[int] = None,
    tolerance: float = 0.005,
    compute_neighbors: bool = True,
    verbose: bool = False,
    **kwargs,
):
    r"""Compute clustering for all datasets.

    Args:
        datasets: list of datasets to cluster
        use_rep: the key in the ``.obsm`` dataframe to ue as input to the Leiden clustering algorithm.
        resolution: the resolution to use for Leiden clustering. Higher values yield finer clusters.

    """

    clustering_function = getattr(sc.tl, method)

    random_state = kwargs.pop("random_state", 0)
    resolution = kwargs.pop("resolution", 1.0)
    key_added = kwargs.pop("key_added", method)

    for dataset in datasets:
        if compute_neighbors:
            sc.pp.neighbors(dataset, use_rep=use_rep, random_state=random_state, n_neighbors=n_neighbors)

        clustering_function(dataset, resolution=resolution, random_state=random_state, key_added=key_added, **kwargs)

        num_clusters = len(dataset.obs[key_added].unique())

        lower_bound = 0.1 * resolution
        upper_bound = 10 * resolution
        while target_clusters and num_clusters != target_clusters and np.abs(lower_bound - upper_bound) > tolerance:
            effective_resolution = (lower_bound * upper_bound) ** 0.5
            clustering_function(
                dataset,
                resolution=effective_resolution,
                random_state=random_state,
                key_added=key_added,
                **kwargs,
            )
            num_clusters = len(dataset.obs[key_added].unique())
            if num_clusters < target_clusters:
                lower_bound = effective_resolution
            elif num_clusters >= target_clusters:
                upper_bound = effective_resolution

            if verbose:
                print(f"Current number of clusters: {num_clusters}")
                print(f"Resolution: {effective_resolution}")

    return datasets


@enable_joint(annotations=["obsm", "varm", "uns"])
@broadcast
def _pca(dataset: PopariDataset, n_comps: int = 50, **pca_kwargs):
    r"""Compute PCA for all datasets.

    Args:
        datasets: list of datasets to process

    """
    dataset.X = csr_matrix(dataset.X)  # Hack because sparse PCA isn't implemented
    sc.pp.pca(dataset, n_comps=n_comps, **pca_kwargs)
    dataset.X = csr_array(dataset.X)


@enable_joint(annotations=["obsm"])
@broadcast
def _umap(dataset: PopariDataset, use_rep: str = "X", compute_neighbors: bool = True, n_neighbors: int = 20):
    r"""Compute UMAP for all datasets.

    Args:
        datasets: list of datasets to process

    """

    if compute_neighbors:
        sc.pp.neighbors(dataset, use_rep=use_rep, n_neighbors=n_neighbors)

    sc.tl.umap(dataset)


@enable_joint
@broadcast
def _plot_in_situ(dataset: Sequence[PopariDataset], axes=None, fig=None, color="leiden", **spatial_kwargs):
    r"""Plot a categorical label across all datasets in-situ.

    Extends AnnData's ``sc.pl.spatial`` function to plot labels/values across multiple replicates.

    Args:
        datasets: list of datasets to process
        color: the key in the ``.obs`` dataframe to plot.
        axes: A predefined set of matplotlib axes to plot on.

    """

    sharex = False if "sharex" not in spatial_kwargs else spatial_kwargs.pop("sharex")
    sharey = False if "sharey" not in spatial_kwargs else spatial_kwargs.pop("sharey")

    # fig = None
    if axes is None:
        fig, axes = setup_squarish_axes(1, sharex=sharex, sharey=sharey)

    edges_width = spatial_kwargs.pop("edges_width", 0.2)
    default_size = spatial_kwargs.pop("size", None)
    palette = spatial_kwargs.pop("palette", None)
    palette = sc.pl.palettes.godsnot_102 if not palette else plt.get_cmap(palette).colors
    palette = ListedColormap(palette)

    legend_fontsize = spatial_kwargs.pop("legend_fontsize", "xx-small")
    edgecolors = spatial_kwargs.pop("edgecolors", "none")
    connectivity_key = spatial_kwargs.pop("connectivity_key", "adjacency_matrix")
    shape = spatial_kwargs.pop("shape", None)

    neighbors_key = spatial_kwargs.pop("neighbors_key", "spatial_neighbors")

    size = 10000 / len(dataset)
    if default_size is not None:
        size *= default_size

    total_categories = len(dataset.obs[color].cat.categories)
    leiden_colorlist = np.array(palette.colors)[np.linspace(0, len(palette.colors) - 1, total_categories, dtype=int)]
    leiden_colors = dict(zip([str(category) for category in dataset.obs[color].cat.categories], leiden_colorlist))
    datasets = unconcatenate(dataset)

    for dataset, ax in zip(datasets, axes.flat):
        dataset_categories = dataset.obs[color].cat.categories
        palette = ListedColormap([color for category, color in leiden_colors.items() if category in dataset_categories])
        sq.pl.spatial_scatter(
            dataset,
            shape=shape,
            size=size,
            connectivity_key=connectivity_key,
            color=color,
            edges_width=edges_width,
            legend_fontsize=legend_fontsize,
            ax=ax,
            fig=fig,
            palette=palette,
            edgecolors=edgecolors,
            library_key="batch",
            **spatial_kwargs,
        )

    return fig


@enable_joint(annotations=[])
@broadcast_plottable
def _plot_umap(dataset: PopariDataset, color="leiden", ax=None, **kwargs):
    r"""Plot a categorical label across all datasets in-situ.

    Extends AnnData's ``sc.pl.spatial`` function to plot labels/values across multiple replicates.

    Args:
        datasets: list of datasets to process
        color: the key in the ``.obs`` dataframe to plot.
        axes: A predefined set of matplotlib axes to plot on.

    """

    edges_width = kwargs.pop("edges_width", 0.2)
    size = kwargs.pop("size", 0.04)
    edges = kwargs.pop("edges", sc.pl.palettes.godsnot_102)
    palette = kwargs.pop("palette", sc.pl.palettes.godsnot_102)
    legend_fontsize = kwargs.pop("legend_fontsize", "xx-small")
    neighbors_key = kwargs.pop("neighbors_key", "neighbors")

    sc.pl.umap(
        dataset,
        size=size,
        neighbors_key=neighbors_key,
        color=color,
        edges=edges,
        edges_width=edges_width,
        legend_fontsize=legend_fontsize,
        ax=ax,
        show=False,
        palette=palette,
        **kwargs,
    )


def _multireplicate_heatmap(
    datasets: Sequence[PopariDataset],
    title_font_size: Optional[int] = None,
    axes: Optional[Sequence[Axes]] = None,
    obsm: Optional[str] = None,
    obsp: Optional[str] = None,
    uns: Optional[str] = None,
    label_values: bool = False,
    label_font_size: int = None,
    nested: bool = True,
    mask: Optional[np.ndarray] = None,
    **heatmap_kwargs,
):
    r"""Plot 2D heatmap data across all datasets.

    Wrapper function to enable plotting of continuous 2D data across multiple replicates. Only
    one of ``obsm``, ``obsp`` or ``uns`` should be used.
    Args:
        datasets: list of datasets to process
        axes: A predefined set of matplotlib axes to plot on.
        obsm: the key in the ``.obsm`` dataframe to plot.
        obsp: the key in the ``.obsp`` dataframe to plot.
        uns: the key in the ``.uns`` dataframe to plot. Unstructured data must be 2D in shape.
        **heatmap_kwargs: arguments to pass to the `ax.imshow` call for each dataset

    """

    sharex = True if "sharex" not in heatmap_kwargs else heatmap_kwargs.pop("sharex", True)
    sharey = True if "sharey" not in heatmap_kwargs else heatmap_kwargs.pop("sharey", True)

    fig = None
    if axes is None:
        fig, axes = setup_squarish_axes(len(datasets), sharex=sharex, sharey=sharey)

    aspect = heatmap_kwargs.pop("aspect", 1)
    cmap = heatmap_kwargs.pop("cmap", "hot")

    images = []
    for dataset_index, ax in enumerate(axes.flat):
        if dataset_index >= len(datasets):
            ax.set_visible(False)
            continue

        dataset = datasets[dataset_index]
        key = None
        if obsm:
            image = dataset.obsm[obsm]
        if obsp:
            image = dataset.obsp[obsp]
        if uns:
            image = dataset.uns[uns]

        if nested:
            image = image[dataset.name]

        if mask is not None:
            image = np.ma.masked_where(mask, image)

        images.append(image)

    for dataset_index, (image, ax) in enumerate(zip(images, axes.flat)):
        dataset = datasets[dataset_index]
        im = ax.imshow(image, cmap=cmap, interpolation="nearest", aspect=aspect, **heatmap_kwargs)
        if title_font_size is not None:
            ax.set_title(dataset.name, fontsize=title_font_size)
        else:
            ax.set_title(dataset.name)

        if label_values:
            truncated_image = image.astype(int)
            for (j, i), label in np.ndenumerate(truncated_image):
                if mask is None or not mask[j, i]:
                    ax.text(i, j, label, ha="center", va="center", fontsize=label_font_size)

        num_rows, num_columns = image.shape
        ax.set_xticks(np.arange(num_columns).astype(int))
        ax.set_yticks(np.arange(num_rows).astype(int))

        plt.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)

    return fig


def _multigroup_heatmap(
    datasets: Sequence[PopariDataset],
    groups: dict,
    title_font_size: Optional[int] = None,
    axes: Optional[Sequence[Axes]] = None,
    key: Optional[str] = None,
    label_values: bool = False,
    label_font_size: int = None,
    **heatmap_kwargs,
):
    r"""Plot 2D heatmap data across all datasets.

    Wrapper function to enable plotting of continuous 2D data across multiple replicates. Only
    one of ``obsm``, ``obsp`` or ``uns`` should be used.

    Args:
        datasets: list of datasets to process
        axes: A predefined set of matplotlib axes to plot on.
        obsm: the key in the ``.obsm`` dataframe to plot.
        obsp: the key in the ``.obsp`` dataframe to plot.
        uns: the key in the ``.uns`` dataframe to plot. Unstructured data must be 2D in shape.
        **heatmap_kwargs: arguments to pass to the `ax.imshow` call for each dataset

    """

    sharex = True if "sharex" not in heatmap_kwargs else heatmap_kwargs.pop("sharex", True)
    sharey = True if "sharey" not in heatmap_kwargs else heatmap_kwargs.pop("sharey", True)

    fig = None
    if axes is None:
        fig, axes = setup_squarish_axes(len(groups), sharex=sharex, sharey=sharey)

    aspect = heatmap_kwargs.pop("aspect", 0.05)
    cmap = heatmap_kwargs.pop("cmap", "hot")

    for group_index, (ax, group_name) in enumerate(zip(axes.flat, groups)):
        first_dataset_name = groups[group_name][0]
        first_dataset = next(filter(lambda dataset: dataset.name == first_dataset_name, datasets))

        if group_index > len(groups):
            ax.set_visible(False)
            continue

        image = first_dataset.uns[key][group_name]

        im = ax.imshow(image, cmap=cmap, interpolation="nearest", aspect=aspect, **heatmap_kwargs)
        if title_font_size is not None:
            ax.set_title(group_name, fontsize=title_font_size)
        if label_values:
            truncated_image = image.astype(int)
            for (j, i), label in np.ndenumerate(truncated_image):
                ax.text(i, j, label, ha="center", va="center", fontsize=label_font_size)

        fig.colorbar(im, ax=ax, orientation="vertical")

    return fig


def _compute_empirical_correlations(
    datasets: Sequence[PopariDataset],
    scaling: float,
    feature: str = "X",
    output: str = "empirical_correlation",
):
    """Compute the empirical spatial correlation for a feature set across all
    datasets.

    Args:
        datasets: list of datasets to process
        feature: key in `.obsm` of feature set for which spatial correlation should be computed.
        output: key in `.uns` where output correlation matrices should be stored.

    """

    num_replicates = len(datasets)

    first_dataset = datasets[0]
    _, K = first_dataset.obsm[feature].shape
    empirical_correlations = np.zeros([num_replicates, K, K])
    for replicate, dataset in enumerate(datasets):
        adjacency_list = dataset.obsm["adjacency_list"]
        X = dataset.obsm[feature]
        Z = X / np.linalg.norm(X, axis=1, keepdims=True, ord=1)
        edges = np.array([(i, j) for i, e in enumerate(adjacency_list) for j in e])

        x = Z[edges[:, 0]]
        y = Z[edges[:, 1]]
        x = x - x.mean(axis=0, keepdims=True)
        y = y - y.mean(axis=0, keepdims=True)
        y_std = y.std(axis=0, keepdims=True)
        x_std = x.std(axis=0, keepdims=True)
        corr = (y / y_std).T @ (x / x_std) / len(x)
        empirical_correlations[replicate] = -corr

    # Convert nan values to 0
    empirical_correlations = np.nan_to_num(empirical_correlations)

    # Symmetrizing and zero-centering empirical_correlation
    empirical_correlations = (empirical_correlations + np.transpose(empirical_correlations, (0, 2, 1))) / 2
    empirical_correlations -= empirical_correlations.mean(axis=(1, 2), keepdims=True)
    empirical_correlations *= scaling

    for dataset, empirical_correlation in zip(datasets, empirical_correlations):
        all_correlations = {dataset.name: empirical_correlation}
        dataset.uns[output] = all_correlations

    return datasets


def _adjacency_permutation_test(
    dataset: PopariDataset,
    labels: str = "X",
    n_trials: int = 500,
    random_state: int = 0,
    pvalue_key: str = "pvalue",
):
    r"""Compute p-values for neighborhood enrichment.

    See
    `In silico tissue generation and power analysis for spatial omics <https://www.nature.com/articles/s41592-023-01766-6#Sec13>`_
    for details.

    """

    rng = np.random.default_rng(seed=random_state)
    original_labels = dataset.obsm[labels]
    adjacency_matrix = dataset.obsp["adjacency_matrix"]
    _, K = original_labels.shape

    original_enrichment = compute_neighborhood_enrichment(original_labels, adjacency_matrix)

    neighborhood_enrichments = np.zeros((n_trials, K, K), dtype=np.float64)
    for trial in range(n_trials):
        permuted_labels = rng.permutation(original_labels)
        neighborhood_enrichments[trial] = compute_neighborhood_enrichment(permuted_labels, adjacency_matrix)

    adjacency = (neighborhood_enrichments > original_enrichment[np.newaxis, :]).sum(axis=0)
    avoidance = (neighborhood_enrichments < original_enrichment[np.newaxis, :]).sum(axis=0)

    p_adjacency = (n_trials - adjacency + 1) / (n_trials + 1)
    p_avoidance = (n_trials - avoidance + 1) / (n_trials + 1)

    dataset.uns[f"adjacency_{pvalue_key}"] = p_adjacency
    dataset.uns[f"avoidance_{pvalue_key}"] = p_avoidance

    return dataset


def _adjacency_permutation_test(
    dataset: PopariDataset,
    labels: str = "X",
    n_trials: int = 100,
    random_state: int = 0,
    pvalue_key: str = "pvalue",
):
    r"""Compute p-values for neighborhood enrichment.

    See
    `In silico tissue generation and power analysis for spatial omics <https://www.nature.com/articles/s41592-023-01766-6#Sec13>`_
    for details.

    """

    rng = np.random.default_rng(seed=random_state)
    original_labels = dataset.obsm[labels]
    adjacency_matrix = dataset.obsp["adjacency_matrix"]
    _, K = original_labels.shape

    original_enrichment = compute_neighborhood_enrichment(original_labels, adjacency_matrix)

    neighborhood_enrichments = np.zeros((n_trials, K, K), dtype=np.float64)
    for trial in range(n_trials):
        permuted_labels = rng.permutation(original_labels)
        neighborhood_enrichments[trial] = compute_neighborhood_enrichment(permuted_labels, adjacency_matrix)

    adjacency = (neighborhood_enrichments > original_enrichment[np.newaxis, :]).sum(axis=0)
    avoidance = (neighborhood_enrichments < original_enrichment[np.newaxis, :]).sum(axis=0)

    p_adjacency = (n_trials - adjacency + 1) / (n_trials + 1)
    p_avoidance = (n_trials - avoidance + 1) / (n_trials + 1)

    dataset.uns[f"adjacency_{pvalue_key}"] = p_adjacency
    dataset.uns[f"avoidance_{pvalue_key}"] = p_avoidance

    return dataset


@enable_joint(annotations=["uns"])
@broadcast
def _compute_ari_score(dataset: PopariDataset, labels: str, predictions: str, ari_key: str = "ari"):
    r"""Compute adjusted Rand index (ARI) score  between a set of ground truth
    labels and an unsupervised clustering.

    Useful for assessing clustering validity. ARI score is computed per dataset.

    Args:
        dataset: dataset to process
        labels: the key in the ``.obs`` dataframe for the label data.
        predictions: the key in the ``.obs`` dataframe for the predictions data.
        ari_key: the key in the ``.uns`` dictionary where the ARI score will be stored.

    """

    ari = adjusted_rand_score(dataset.obs[labels], dataset.obs[predictions])
    dataset.uns[ari_key] = ari


@enable_joint(annotations=["uns"])
@broadcast
def _compute_silhouette_score(dataset: PopariDataset, labels: str, embeddings: str, silhouette_key: str = "silhouette"):
    r"""Compute silhouette score for a clustering based on Popari embeddings.

    Useful for assessing clustering validity. ARI score is computed per dataset.

    Args:
        dataset: dataset to process
        labels: the key in the ``.obs`` dataframe for the label data.
        predictions: the key in the ``.obs`` dataframe for the predictions data.
        ari_key: the key in the ``.uns`` dictionary where the ARI score will be stored.

    """

    silhouette = silhouette_score(dataset.obsm[embeddings], dataset.obs[labels])
    dataset.uns[silhouette_key] = silhouette


@broadcast
def _plot_all_embeddings(
    dataset: PopariDataset,
    embedding_key: str = "X",
    column_names: Optional[str] = None,
    **spatial_kwargs,
):
    r"""Plot all laerned metagenes in-situ across all replicates.

    Each replicate's metagenes are contained in a separate plot.

    Args:
        dataset: dataset to process
        embedding_key: the key in the ``.obsm`` dataframe for the cell/spot embeddings.
        column_names: a list of the suffixes for each latent feature. If ``None``, it is assumed
            that these suffixes are just the indices of the latent features.

    """

    _, K = dataset.obsm[f"{embedding_key}"].shape
    if column_names == None:
        # TODO: remove dependence on trained_model
        column_names = [f"{embedding_key}_{index}" for index in range(K)]

    edges_width = spatial_kwargs.pop("edges_width", 0.2)
    default_size = spatial_kwargs.pop("size", None)
    palette = spatial_kwargs.pop("palette", ListedColormap(sc.pl.palettes.godsnot_102))

    size = len(dataset) / 100
    if default_size is not None:
        size *= default_size

    axes = sq.pl.spatial_scatter(
        sq.pl.extract(dataset.copy(), embedding_key, prefix=f"{embedding_key}"),
        shape=None,
        color=column_names,
        edges_width=edges_width,
        connectivity_key="adjacency_matrix",
        size=size,
        wspace=0.2,
        ncols=2,
        **spatial_kwargs,
    )


@enable_joint(annotations=["uns"])
@broadcast
def _evaluate_classification_task(dataset: PopariDataset, embeddings: str, labels: str):
    """"""

    le = LabelEncoder()
    encoded_labels = le.fit_transform(dataset.obs[labels].astype(str))
    dataset_embeddings = dataset.obsm[embeddings]

    X_train, X_valid, y_train, y_valid = train_test_split(
        dataset_embeddings,
        encoded_labels,
        train_size=0.25,
        random_state=42,
        stratify=encoded_labels,
    )
    model = KNeighborsClassifier(n_neighbors=10)
    model.fit(X_train, y_train)

    df = []
    for split, X, y in [("train", X_train, y_train), ("validation", X_valid, y_valid)]:
        y_soft = model.predict_proba(X)
        y_hat = np.argmax(y_soft, 1)
        dataset.uns[f"microprecision_{split}"] = precision_score(y, y_hat, average="micro")
        dataset.uns[f"macroprecision_{split}"] = precision_score(y, y_hat, average="macro")

    return dataset


@enable_joint(annotations=["obs", "uns"])
@broadcast
def _compute_confusion_matrix(
    dataset: PopariDataset,
    labels: str,
    predictions: str,
    result_key: str = "confusion_matrix",
):
    r"""Compute confusion matrix for labels and predictions.

    Useful for visualizing clustering validity.

    Args:
        trained_model: the trained Popari model.
        labels: the key in the ``.obs`` dataframe for the label data.
        predictions: the key in the ``.obs`` dataframe for the predictions data.
        result_key: the key in the ``.uns`` dictionary where the reordered confusion matrix will be stored.

    """

    unique_labels = sorted(dataset.obs[labels].unique())
    unique_predictions = sorted(dataset.obs[predictions].unique())
    if len(unique_labels) != len(unique_predictions):
        raise ValueError("Number of unique labels and unique predictions must be equal.")

    encoded_labels = [unique_labels.index(label) for label in dataset.obs[labels].values]
    encoded_predictions = [unique_predictions.index(prediction) for prediction in dataset.obs[predictions].values]

    confusion_output = confusion_matrix(encoded_labels, encoded_predictions)

    permutation, index = get_optimal_permutation(confusion_output)
    dataset.obs[f"{labels}_inferred"] = [unique_labels[permutation[prediction]] for prediction in encoded_predictions]

    reordered_confusion = confusion_matrix(dataset.obs[labels], dataset.obs[f"{labels}_inferred"])[: len(unique_labels)]

    dataset.uns[result_key] = reordered_confusion


def get_optimal_permutation(confusion_output):
    """
    TODO: document
    maximum weight bipartite matching
    :param confusion_output:
    :return: confusion_output[perm, index], where index is sorted
    """

    num_label_classes, num_prediction_classes = confusion_output.shape

    bipartite_graph = nx.Graph()
    bipartite_graph.add_nodes_from([("label", i) for i in range(num_label_classes)], bipartite=0)
    bipartite_graph.add_nodes_from([("prediction", i) for i in range(num_prediction_classes)], bipartite=1)

    bipartite_graph.add_edges_from(
        [
            (("label", i), ("prediction", j), {"weight": confusion_output[i, j]})
            for i in range(num_label_classes)
            for j in range(num_prediction_classes)
        ],
    )

    assert nx.is_bipartite(bipartite_graph)
    matching = nx.max_weight_matching(bipartite_graph, maxcardinality=True)
    assert len({__ for _ in matching for __ in _}) == num_label_classes * 2

    matching = [sorted(match, key=lambda node_attributes: node_attributes[0]) for match in matching]

    matching = [tuple(index for (_, index) in match) for match in matching]
    matching = sorted(matching, key=lambda pair: pair[1])

    perm, index = tuple(map(np.array, zip(*matching)))

    return perm, index


@enable_joint(annotations=["uns"])
@broadcast
def _compute_columnwise_autocorrelation(
    dataset: PopariDataset,
    uns: str = "ground_truth_M",
    result_key: str = "ground_truth_M_correlation",
):
    """"""

    matrix = dataset.uns[uns][f"{dataset.name}"].T

    num_columns, _ = matrix.shape
    correlation_coefficient_matrix = np.corrcoef(matrix, matrix)[:num_columns, :num_columns]
    dataset.uns[result_key] = correlation_coefficient_matrix


@broadcast_plottable
def _plot_confusion_matrix(
    dataset: PopariDataset,
    labels: str,
    ax=None,
    confusion_matrix_key: str = "confusion_matrix",
):
    ordered_labels = sorted(dataset.obs[labels].unique())
    sns.heatmap(
        dataset.uns[confusion_matrix_key],
        xticklabels=ordered_labels,
        yticklabels=ordered_labels,
        annot=True,
        ax=ax,
        fmt="3d",
    )
    plt.show()


@enable_joint(annotations=["uns"])
@broadcast
def _compute_spatial_gene_correlation(
    dataset: PopariDataset,
    spatial_key: str = "Sigma_x_inv",
    metagene_key: str = "M",
    spatial_gene_correlation_key: str = "spatial_gene_correlation",
    neighbor_interactions_key: str = "neighbor_interactions",
):
    """Computes spatial gene correlation according to learned metagenes."""

    spatial_affinity_matrix = dataset.uns[spatial_key][f"{dataset.name}"]
    metagenes = dataset.uns[metagene_key][f"{dataset.name}"]

    neighbor_interactions = metagenes @ spatial_affinity_matrix
    spatial_gene_correlation = neighbor_interactions @ metagenes.T

    dataset.uns[spatial_gene_correlation_key] = spatial_gene_correlation
    dataset.uns[neighbor_interactions_key] = neighbor_interactions


def _metagene_neighbor_interactions(dataset: PopariDataset, interaction_key: str = "metagene_neighbor_interactions"):
    """Compute pairwise interactions between every cell in terms of learned
    metagene embeddings.

    Can be used to visualize the empirical spatial correlations between metagenes.

    Args:
        dataset:

    """
    embeddings = dataset.obsm["X"]
    X = embeddings
    adjacency_matrix = dataset.obsp["adjacency_matrix"].toarray()

    adjacency_list = dataset.obsm["adjacency_list"]
    num_cells, num_metagenes = embeddings.shape

    Z = X / np.linalg.norm(X, axis=1, keepdims=True, ord=1)
    edges = np.array([(i, j) for i, e in enumerate(adjacency_list) for j in e])

    x = Z[edges[:, 0]]
    y = Z[edges[:, 1]]

    pair_interactions = np.zeros((num_cells, num_cells, num_metagenes, num_metagenes))
    cell_i, cell_j = adjacency_matrix.nonzero()
    for i in range(num_metagenes):
        for j in range(i, num_metagenes):
            pair_interactions[cell_i, cell_j, i, j] = 1 - x[:, i] * y[:, j]

    dataset.obsp[interaction_key] = pair_interactions


def _pretty_plot_spatial_affinities(
    datasets,
    metagene_key: str = "M",
    metagene_label_key: str = "metagene_labels",
    spatial_affinity_key: str = "Sigma_x_inv",
):
    """Rotate and plot spatial affinities matrix.

    Args:
        datasets:

    """

    transform_skew = Affine2D().skew_deg(15, 15)
    transform_rotate = Affine2D().rotate_deg(-45)
    transform = transform_skew + transform_rotate

    height = len(datasets) // 2 + len(datasets) % 2
    width = 2
    fig = plt.figure(dpi=1200, figsize=(width, height))

    first_dataset = datasets[0]
    _, K = first_dataset.uns[metagene_key][first_dataset.name].shape

    def setup_axes(fig, rect, metagene_ticks):
        """Setup axes for rotated heatmap plot."""

        grid_locator = FixedLocator([v for v, s in metagene_ticks])
        tick_formatter = DictFormatter(dict(metagene_ticks))

        grid_helper = floating_axes.GridHelperCurveLinear(
            transform,
            extremes=(-0.5, K - 0.5, -0.5, K - 0.5),  # TODO: adjust to fit number of metagenes K
            grid_locator1=grid_locator,
            tick_formatter1=tick_formatter,
            grid_locator2=grid_locator,
            tick_formatter2=tick_formatter,
        )
        ax = floating_axes.FloatingSubplot(fig, *rect, grid_helper=grid_helper)
        fig.add_subplot(ax)

        ax.axis["left"].toggle(ticklabels=False, label=False, ticks=False)

        ax.axis["right"].toggle(ticklabels=True, label=True, ticks=False)
        ax.axis["top"].toggle(ticks=False)

        ax.axis["bottom"].toggle(ticks=False)

        ax.axis["left"].line.set_linewidth(0)
        ax.axis["top"].line.set_linewidth(0)

        ax.axis["bottom"].line.set_linewidth(0.1)
        ax.axis["right"].line.set_linewidth(0.1)

        ax.axis["right"].major_ticklabels.set_axis_direction("right")
        ax.axis["bottom"].major_ticklabels.set_axis_direction("left")

        aux_ax = ax.get_aux_axes(transform)

        grid_helper.grid_finder.grid_locator1._nbins = 4
        grid_helper.grid_finder.grid_locator2._nbins = 4

        ax.axis[:].major_ticklabels.set_fontsize(2)

        return ax, aux_ax

    axes = []
    for index, dataset in enumerate(datasets):
        metagene_labels = dataset.uns[metagene_label_key]

        metagene_ticks = [(k, f"{metagene_labels[k]}" if k in metagene_labels else "N/A") for k in range(K)]

        rect = (height, width, index + 1)
        ax, aux_ax = setup_axes(fig, rect, metagene_ticks)
        ax.set_title(dataset.name, fontsize=2, y=0.5)
        axes.append(aux_ax)

    mask = np.ones((K, K), dtype=np.bool)
    mask[np.triu_indices_from(mask)] = 0

    _multireplicate_heatmap(
        datasets,
        uns=spatial_affinity_key,
        cmap="bwr",
        label_values=False,
        label_font_size=1.5,
        vmin=-10,
        vmax=10,
        mask=mask,
        axes=np.array(axes),
    )
    for ax in axes:
        im = ax.images
        cb = im[-1].colorbar
        cb.remove()

    fig.subplots_adjust(hspace=-0.5, wspace=-0.25)


def _plot_sparsity(dataset: ad.AnnData):
    """Plot overall sparsity of dataset.

    Args:
        dataset: dataset for which to compute and plot data sparsity.

    """

    raw_data = dataset.X

    if issparse(raw_data):
        raw_data = raw_data.toarray()

    raw_data = raw_data.flatten()
    raw_data_clipped = raw_data
    raw_data_clipped = raw_data_clipped[
        (raw_data_clipped > 1e-6) & (raw_data_clipped < np.percentile(raw_data_clipped, 99.9))
    ]

    fig, axes = plt.subplots(1, 2, sharey=True, tight_layout=True)
    axes[0].hist(raw_data, bins=20)
    axes[0].set_title("Count histogram")

    heights, *_ = axes[1].hist(raw_data_clipped, bins=20)
    axes[1].set_title("Clipped count histogram")
    axes[1].set_ylim([0, max(heights)])

    sparsity = (raw_data == 0).sum() / raw_data.size
    fig.suptitle(f"Overall sparsity: {sparsity}")

    return fig


def _plot_cell_type_to_metagene(
    dataset,
    cell_type_de_genes: dict,
    rank_mode: str = "metagene",
    plot_type: str = "box",
    normalize: bool = False,
    figsize: Optional[tuple] = None,
    cell_types: Optional[list] = None,
    metagene_index: Optional[Sequence] = None,
    alternative_hypothesis: str = "greater",
    metagene_key: str = "M",
    cmap: str = "rainbow",
    **subplot_kwargs,
):
    """Plot correspondence between cell types and metagenes.

    Args:
        dataset: output dataset from Popari
        cell_type_de_genes: mapping from cell types to DEG names.
        rank_mode: whether to compute ranks across metagenes or across genes.
        plot_type: whether to use box plot
        normalize: whether to normalize genes by zscore (across metagenes)
            before computing significance
        p_values: whether or not to calculate p-values for each cell type association
        cell_types: only plot listed cell types. Default: ``None`` (plot all cell types)
        metagene_index: only use listed metagenes types. Default: ``None`` (plot all cell types)
        metagene_key: key under which metagene values are stored. Default: ``M``

    """
    if cell_types is None:
        cell_types = cell_type_de_genes.keys()

    metagenes = dataset.uns[metagene_key][dataset.name]

    if normalize:
        metagenes = zscore(metagenes, axis=1)

    if metagene_index is None:
        num_genes, K = metagenes.shape
        metagene_index = np.arange(K)
    else:
        num_genes, _ = metagenes.shape
        K = len(metagene_index)

    sorted_indices = np.argsort(metagenes, axis=int(rank_mode != "metagene"))

    sorted_indices = sorted_indices[:, metagene_index]

    if figsize is None:
        figsize = (K, len(cell_types))

    fig, axes = plt.subplots(len(cell_types), 1, figsize=figsize, sharex=True, dpi=300, **subplot_kwargs, squeeze=False)

    ordered_genes = dataset.var_names
    #     axes[0].set_title("Gene Rank")

    means = {}
    for index, (cell_type, ax) in enumerate(zip(cell_types, axes.flat)):
        de_genes = cell_type_de_genes[cell_type]
        num_de_genes = len(de_genes)

        de_gene_indices = ordered_genes.get_indexer(de_genes)
        filtered_indices = [index for index in de_gene_indices if (index > -1)]
        num_actual_de_genes = len(filtered_indices)
        rank_distributions = []

        colors = colormaps[cmap].resampled(len(metagene_index))

        for sorted_index in sorted_indices.T:
            de_gene_ranks = sorted_index[filtered_indices]

            rank_distributions.append(de_gene_ranks)

        cell_type_means = [np.median(distribution) for distribution in rank_distributions]
        means[cell_type] = cell_type_means

        reduction = np.argmax if alternative_hypothesis == "greater" else np.argmin
        best_index = reduction(cell_type_means)

        best_distribution = rank_distributions[best_index]
        p_values = []
        for other_index in range(K):
            if other_index == best_index:
                p_values.append(0)
                continue
            other_distribution = rank_distributions[other_index]
            difference_distribution = best_distribution - other_distribution
            p_value = wilcoxon(difference_distribution, alternative=alternative_hypothesis).pvalue
            p_values.append(p_value)

        #         print(p_values)
        is_significant = (np.array(p_values) < (0.05 // len(p_values))).all()

        ax.text(1.05, 1, f"n={num_de_genes}[{num_actual_de_genes}]", transform=ax.transAxes, fontsize=figsize[0] + 2)

        if plot_type == "violin":
            plot = ax.violinplot(rank_distributions, showextrema=True, showmedians=True)

            plot["bodies"][0].set_zorder(4)
        elif plot_type == "box":
            medianprops = dict(linewidth=2, color="brown")
            plot = ax.boxplot(rank_distributions, widths=0.8, patch_artist=True, medianprops=medianprops)
            for median in plot["medians"]:
                median.set_zorder(1)
                median.set_label("Median rank")

            for index, patch in enumerate(plot["boxes"]):
                color = "white"
                if index != best_index:
                    p_value = p_values[index]
                    color = colors(index)
                    text_color = "red" if p_value <= 0.05 else "black"
                    ax.text(
                        index / K,
                        1,
                        f"p={p_value:.0E}",
                        rotation=30,
                        color=text_color,
                        transform=ax.transAxes,
                        fontsize=figsize[0] + 2,
                    )

                patch.set_facecolor(color)
                patch.set_zorder(0)

        ymax = num_genes if rank_mode == "metagene" else K
        ax.hlines(
            y=ymax // 2,
            xmin=0.5,
            xmax=K + 0.5,
            linewidth=1.5,
            linestyle="--",
            color="k",
            zorder=2,
            label="Null hypothesis rank",
        )

        # Setting axis tick lines
        tick_interval = 1000
        top_tick = num_genes // tick_interval
        y_ticks = [1] + [(i + 1) * tick_interval for i in range(top_tick)]
        y_ticklabels = [y_tick if y_tick in (1, top_tick * tick_interval) else "" for y_tick in y_ticks]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticklabels)

        ax.tick_params(axis="x", bottom=False)
        ax.set_ylabel(cell_type)

        ax.yaxis.tick_right()
        for side, spine in ax.spines.items():
            if side != "right":
                spine.set_edgecolor("none")
            else:
                spine.set_bounds((1, num_genes))

    ax.set_xticks(np.arange(len(metagene_index)) + 1)
    ax.set_xticklabels([f"m{m}" for m in metagene_index])

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.1)

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = (sum(lol, []) for lol in zip(*lines_labels))

    # Create legend for null hypothesis and medians of boxplots
    filtered_lines = []
    filtered_labels = []
    for label in np.unique(labels):
        first_index = labels.index(label)
        filtered_labels.append(label)
        filtered_lines.append(lines[first_index])

    fig.legend(filtered_lines, filtered_labels, loc="upper left", bbox_to_anchor=(1, 1))

    fig.supxlabel("Metagene")
    fig.supylabel("Cell Type")

    return fig, means


def _plot_cell_type_to_metagene_difference(
    dataset,
    cell_type_de_genes: dict,
    first_metagene: int,
    second_metagene: int,
    rank_mode: str = "metagene",
    plot_type: str = "box",
    normalize: bool = False,
    figsize: Optional[tuple] = None,
    cell_types: Optional[list] = None,
    alternative_hypothesis: str = "greater",
    metagene_key: str = "M",
    cmap: str = "rainbow",
    **subplot_kwargs,
):
    """Plot correspondence between cell types and metagenes.

    Args:
        dataset: output dataset from Popari
        cell_type_de_genes: mapping from cell types to DEG names.
        rank_mode: whether to compute ranks across metagenes or across genes.
        plot_type: whether to use box plot
        normalize: whether to normalize genes by zscore (across metagenes)
            before computing significance
        p_values: whether or not to calculate p-values for each cell type association
        cell_types: only plot listed cell types. Default: ``None`` (plot all cell types)
        metagene_index: only use listed metagenes types. Default: ``None`` (plot all cell types)
        metagene_key: key under which metagene values are stored. Default: ``M``

    """
    if cell_types is None:
        cell_types = cell_type_de_genes.keys()

    metagenes = dataset.uns[metagene_key][dataset.name]

    if normalize:
        metagenes = zscore(metagenes, axis=1)

    num_genes, K = metagenes.shape

    sorted_indices = np.argsort(metagenes, axis=int(rank_mode != "metagene"))

    if figsize is None:
        figsize = (len(cell_types), 4)

    fig, ax = plt.subplots(figsize=figsize, sharex=True, dpi=300, **subplot_kwargs)

    ordered_genes = dataset.var_names
    #     axes[0].set_title("Gene Rank")

    means = {}
    difference_distributions = []
    p_values = []
    for index, cell_type in enumerate(cell_types):
        de_genes = cell_type_de_genes[cell_type]
        num_de_genes = len(de_genes)

        de_gene_indices = ordered_genes.get_indexer(de_genes)
        filtered_indices = [index for index in de_gene_indices if (index > -1)]
        num_actual_de_genes = len(filtered_indices)

        colors = colormaps[cmap].resampled(2)

        first_metagene_de_gene_ranks = sorted_indices[:, first_metagene][filtered_indices]
        second_metagene_de_gene_ranks = sorted_indices[:, second_metagene][filtered_indices]

        difference_distribution = first_metagene_de_gene_ranks - second_metagene_de_gene_ranks
        difference_distributions.append(difference_distribution)

        p_value = wilcoxon(difference_distribution, alternative=alternative_hypothesis).pvalue
        p_values.append(p_value)

    cell_type_means = [np.median(distribution) for distribution in difference_distributions]
    means[cell_type] = cell_type_means

    #         print(p_values)
    is_significant = (np.array(p_values) < (0.05 // len(p_values))).all()

    # ax.text(1.05, 1, f"n={num_de_genes}[{num_actual_de_genes}]", transform=ax.transAxes, fontsize=figsize[0] + 2)

    if plot_type == "violin":
        plot = ax.violinplot(rank_distributions, showextrema=True, showmedians=True)

        plot["bodies"][0].set_zorder(4)
    elif plot_type == "box":
        medianprops = dict(linewidth=2, color="brown")
        plot = ax.boxplot(difference_distributions, widths=0.8, patch_artist=True, medianprops=medianprops)
        for median in plot["medians"]:
            median.set_zorder(1)
            median.set_label("Median difference")

        for index, patch in enumerate(plot["boxes"]):
            p_value = p_values[index]
            color = colors(index)
            text_color = "red" if p_value <= 0.05 else "black"
            ax.text(
                index / len(cell_types),
                1,
                f"p={p_value:.0E}",
                rotation=30,
                color=text_color,
                transform=ax.transAxes,
                fontsize=figsize[0] + 2,
            )

            patch.set_facecolor(color)
            patch.set_zorder(0)

    ax.hlines(
        y=0,
        xmin=0.5,
        xmax=len(cell_types) + 1,
        linewidth=1.5,
        linestyle="--",
        color="k",
        zorder=2,
        label="Null hypothesis rank",
    )

    # Setting axis tick lines
    tick_interval = 1000
    top_tick = num_genes // tick_interval
    y_ticks = [i * tick_interval for i in range(-top_tick, top_tick + 1)]
    y_ticklabels = [
        y_tick if y_tick in (-top_tick * tick_interval, 0, top_tick * tick_interval) else "" for y_tick in y_ticks
    ]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticklabels)

    ax.tick_params(axis="x", bottom=False)
    ax.set_ylabel(f"Difference distribution between m{first_metagene} and m{second_metagene}")

    ax.yaxis.tick_right()
    for side, spine in ax.spines.items():
        if side != "right":
            spine.set_edgecolor("none")
        else:
            pass
            spine.set_bounds((-num_genes, num_genes))

    ax.set_xticks(np.arange(len(cell_types)) + 1)
    ax.set_xticklabels(cell_types, rotation=-30, ha="left")

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.1)

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = (sum(lol, []) for lol in zip(*lines_labels))

    # Create legend for null hypothesis and medians of boxplots
    filtered_lines = []
    filtered_labels = []
    for label in np.unique(labels):
        first_index = labels.index(label)
        filtered_labels.append(label)
        filtered_lines.append(lines[first_index])

    fig.legend(filtered_lines, filtered_labels, loc="upper left", bbox_to_anchor=(1, 1))

    ax.set_xlabel("Cell Type")

    return fig, means


def _compile_de_genes(
    dataset,
    de_category: str = "cell_type",
    filtered_deg_key: str = "t-test filtered",
    p_value_threshold: float = 1e-5,
    max_genes: int = 500,
    format="disk",
):
    """Compile DE genes.

    After running ``sc.tl.rank_genes_groups`` and ``sc.tl.filter_rank_genes_groups``, call this
    function in order to summarize and collect the DE genes identified before.

    Args:
        dataset:

    """
    all_de_genes = set()
    ranked_genes = dataset.uns[filtered_deg_key]["names"]

    # Record arrays are nasty, and will be removed in Scanpy 2.0, probably
    for cell_type in ranked_genes.dtype.names:
        index = ranked_genes[cell_type] == "nan"
        ranked_genes[cell_type][index] = np.nan

    adjusted_pvals = dataset.uns[filtered_deg_key]["pvals_adj"]
    cell_type_de_genes = {}
    for cell_type in dataset.obs[de_category].unique():
        cell_type_genes = ranked_genes[cell_type]
        cell_type_pvals = adjusted_pvals[cell_type]

        pval_filter = cell_type_pvals < p_value_threshold

        filtered_genes = cell_type_genes[(pval_filter) & (~pd.isna(cell_type_genes))][:max_genes]
        cell_type_de_genes[cell_type] = filtered_genes
        print(cell_type, len(filtered_genes))
        all_de_genes.update(filtered_genes)

    if format == "disk":
        for cell_type in ranked_genes.dtype.names:
            index = pd.isna(ranked_genes[cell_type])
            ranked_genes[cell_type][index] = "nan"
        # ranked_genes[cell_type][index] = ranked_genes[cell_type][index].astype(str)

    print(len(all_de_genes))

    return cell_type_de_genes, all_de_genes


def _call_de_genes(
    dataset,
    label_key: str,
    output_key: str,
    filter: bool = True,
    min_fold_change: float = 1,
    max_genes: int = 500,
):
    """Call differentially expressed genes using a t-test, and compile.

    Args:

    """
    filtered_deg_key = output_key

    sc.tl.rank_genes_groups(dataset, label_key, method="t-test", key_added=output_key, min_fold_change=min_fold_change)
    if filter:
        filtered_deg_key = f"{filtered_deg_key}_filtered"
        sc.tl.filter_rank_genes_groups(
            dataset,
            key=output_key,
            key_added=f"{output_key}_filtered",
            min_fold_change=min_fold_change,
        )

    cell_type_de_genes, all_de_genes = _compile_de_genes(
        dataset,
        label_key,
        max_genes=max_genes,
        filtered_deg_key=filtered_deg_key,
    )

    return cell_type_de_genes, all_de_genes


@enable_joint
@broadcast_plottable
def _plot_embeddings_to_label(
    dataset,
    ax=None,
    names: Optional[Sequence[str]] = None,
    embedding_key: str = "normalized_X",
    label_key: str = "leiden",
    **dotplot_kwargs,
):
    """Plot enrichment of embedding dimensions for a label category.

    Args:
        names: Names of embedding dimensions
        swap

    """

    X = dataset.obsm[embedding_key]
    num_spots, K = X.shape

    mock_dataset = ad.AnnData(
        X=dataset.obsm[embedding_key],
    )

    if not names:
        mock_dataset.var_names = [f"m{index}" for index in range(K)]
    else:
        mock_dataset.var_names = names

    swap_axes = dotplot_kwargs.pop("swap_axes", True)
    standard_scale = dotplot_kwargs.pop("standard_scale", "var")

    mock_dataset.obs = dataset.obs.copy()

    aggregated = sc.get.aggregate(
        mock_dataset,
        by=label_key,
        func=["sum", "count_nonzero"],
    )

    mean_in_expressed = aggregated.to_df(layer="sum") / aggregated.to_df(layer="count_nonzero")

    vmin = None
    vmax = None
    if standard_scale is None:
        vmax = np.max(np.abs(mean_in_expressed))
        vmin = -vmax

    dotplot = sc.pl.dotplot(
        mock_dataset,
        mock_dataset.var_names,
        groupby=label_key,
        dendrogram=False,
        ax=ax,
        standard_scale=standard_scale,
        swap_axes=swap_axes,
        return_fig=True,
        vmin=vmin,
        vmax=vmax,
        **dotplot_kwargs,
    )

    dotplot.add_totals()
    dotplot.show()

    return dotplot, mean_in_expressed


@enable_joint(annotations=["obsm"])
@broadcast
def _score_marker_expression(dataset, de_genes: dict[str, Sequence[str]], output_key="marker_expression"):
    """Given a mapping from cell types to marker genes, compute enrichment."""
    zscored_expression = zscore(dataset.X)

    marker_gene_expression = np.zeros((len(dataset), len(de_genes)))
    for index, (subtype, gene_list) in enumerate(de_genes.items()):
        gene_list_index = dataset.var_names.isin(gene_list)
        marker_gene_expression[:, index] = zscored_expression[:, gene_list_index].sum(axis=1)

    dataset.obsm[output_key] = marker_gene_expression


@enable_joint(annotations=["obsm"])
def _plot_clusters_to_categories(
    datasets,
    category_de_genes: dict[str, Sequence[str]],
    output_key="marker_expression",
    **dotplot_kwargs,
):
    """Plot clusters to category correspondence, based on category marker gene
    expression.

    Args:
        category_de_genes: mapping from category to marker genes for that category

    """
    _score_marker_expression.__wrapped__(datasets, category_de_genes, output_key=output_key)
    (cell_type_to_cluster_correspondence_fig, (_, mean_in_expressed)) = _plot_embeddings_to_label.__wrapped__(
        datasets,
        names=list(category_de_genes.keys()),
        standard_scale=None,
        cmap="bwr",
        embedding_key=output_key,
        title="Cell-type-to-cluster Correspondence",
        **dotplot_kwargs,
    )

    return cell_type_to_cluster_correspondence_fig


def _cluster_domains(
    datasets,
    target_domains: Optional[int] = None,
    skip_thresholding: bool = True,
    batch_correct: bool = False,
):
    """Discover domains from Popari embeddings."""

    normalized_key = "normalized_X"
    if not skip_thresholding:
        normalized_key = "normalized_thresholded_expression"
        for dataset in datasets:
            normalize_expression_by_threshold(dataset, thresholded_key="normalized_X")

    processed_key = normalized_key
    if batch_correct:
        merged_dataset = concatenate(datasets)
        sce.pp.scanorama_integrate(merged_dataset, "batch", basis=normalized_key, verbose=1)

        processed_key = "X_scanorama"
        for dataset in datasets:
            dataset.obsm[processed_key] = (
                merged_dataset[merged_dataset.obs["batch"] == dataset.name].obsm[processed_key].copy()
            )

    for dataset in datasets:
        smooth_metagene_expression(dataset, processed_key=processed_key)

    _cluster(
        datasets,
        verbose=True,
        use_rep="smoothed_expression",
        target_clusters=target_domains,
        n_neighbors=40,
        joint=True,
    )

    for dataset in datasets:
        smooth_labels(dataset, output_key="smoothed_domain")


def _metagene_gsea(
    dataset: PopariDataset,
    metagene_index: int,
    metagene_key: str = "M",
    sensitivity: float = 0.5,
    **gsea_kwargs,
):
    """Compute GSEA for the top genes from a particular metagene."""
    metagenes = dataset.uns[metagene_key][dataset.name]
    metagene = metagenes[:, metagene_index]

    gene_names = dataset.var_names

    signature = list(get_metagene_signature(metagene, gene_names, show_plot=False, sensitivity=sensitivity))

    mode = gsea_kwargs.pop("mode", "barplot")
    try:
        enrichment_result, ax = run_gsea(
            signature,
            f"Metagene {metagene_index}",
            gene_names,
            mode=mode,
        )

        fig = ax.figure
    except ValueError:
        raise ValueError("Didn't find any enriched GO terms.")

    return fig


def _plot_metagene_signature_enrichment(
    datasets,
    metagene_index,
    categories=None,
    category_key="domain",
    sensitivity=1,
    **subplot_kwargs,
):
    """Plot enrichment of metagene signature across categories."""

    first_dataset = datasets[0]
    M = list(first_dataset.uns["M"].values())[0]

    gene_names = first_dataset.var_names

    merged_dataset = concatenate(datasets)
    if categories is None:
        categories = merged_dataset.obs[category_key].unique()

    metagene_signature = get_metagene_signature(
        M[:, metagene_index],
        gene_names,
        sensitivity=sensitivity,
        type="upregulated",
        show_plot=False,
    )

    indices = gene_names.get_indexer(metagene_signature)

    num_columns = len(datasets) * len(categories)
    raw_genes = np.zeros((len(metagene_signature), num_columns))

    for category_index, category in enumerate(categories):
        for dataset_index, dataset in enumerate(datasets):

            first_subtype = dataset[dataset.obs[category_key] == category]
            raw_genes[:, dataset_index * len(categories) + category_index] = first_subtype.X[:, indices].mean(axis=0)

    vmax = np.abs(raw_genes).max()
    vmin = 0

    dpi = subplot_kwargs.pop("dpi", 200)
    figsize = subplot_kwargs.pop("figsize", (4, len(metagene_signature) / 5))
    fig, ax = plt.subplots(dpi=dpi, figsize=figsize, **subplot_kwargs)
    im = ax.imshow(raw_genes, cmap="Reds", vmax=vmax, vmin=vmin)

    for index in range(0, len(datasets)):
        ax.axvline((index) * len(categories) - 0.5, color="k")

    ax.grid(False)

    plt.colorbar(im, ax=ax)
    ax.set_yticks(np.arange(len(metagene_signature)))
    ax.set_yticklabels(metagene_signature)
    ax.set_xticks(np.arange(num_columns))
    ax.set_xticklabels([])
    ax.set_aspect("equal")
    colors = plt.get_cmap("tab20b").colors  # Colors for each tick

    for i, tick in enumerate(ax.get_xticks()):
        rect = patches.Rectangle(
            ((tick + 0.1) / num_columns, 0),
            0.04,
            -0.5 / len(metagene_signature),
            transform=ax.transAxes,
            color=colors[i % len(categories)],
        )
        fig.add_artist(rect)

    return fig, metagene_signature
