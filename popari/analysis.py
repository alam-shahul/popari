from functools import partial, wraps
from typing import Optional, Sequence

import numpy as np
import scanpy as sc
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from scipy.stats import pearsonr, spearmanr, wasserstein_distance

from popari._dataset_utils import (
    _cluster,
    _cluster_domains,
    _compute_ari_score,
    _compute_columnwise_autocorrelation,
    _compute_confusion_matrix,
    _compute_empirical_correlations,
    _compute_silhouette_score,
    _compute_spatial_gene_correlation,
    _evaluate_classification_task,
    _leiden,
    _metagene_gsea,
    _multigroup_heatmap,
    _pca,
    _preprocess_embeddings,
    _umap,
    for_model,
    setup_squarish_axes,
)
from popari.model import Popari
from popari.util import spatially_smooth_feature

preprocess_embeddings = for_model(_preprocess_embeddings)
cluster = for_model(_cluster)
leiden = for_model(_leiden)
pca = for_model(_pca)
umap = for_model(_umap)
compute_ari_scores = for_model(_compute_ari_score)
compute_silhouette_scores = for_model(_compute_silhouette_score)
evaluate_classification_task = for_model(_evaluate_classification_task)
compute_confusion_matrix = for_model(_compute_confusion_matrix)
compute_columnwise_autocorrelation = for_model(_compute_columnwise_autocorrelation)
compute_spatial_gene_correlation = for_model(_compute_spatial_gene_correlation)
cluster_domains = for_model(_cluster_domains)


# def leiden(
#     trained_model: Popari,
#     resolution: float = 1.0,
#     tolerance: float = 0.05,
#     **kwargs,
# ):
#     r"""Compute Leiden clustering for all datasets.
#
#     Args:
#         trained_model: the trained Popari model.
#         joint: if `True`, jointly cluster the spots
#         use_rep: the key in the ``.obsm`` dataframe to ue as input to the Leiden clustering algorithm.
#         resolution: the resolution to use for Leiden clustering. Higher values yield finer clusters..
#
#     """
#     cluster(
#         trained_model,
#         resolution=resolution,
#         tolerance=tolerance,
#         flavor="igraph",
#         n_iterations=2,
#         **kwargs,
#     )


def multigroup_heatmap(
    trained_model: Popari,
    title_font_size: Optional[int] = None,
    group_type: str = "metagene",
    axes: Optional[Sequence[Axes]] = None,
    key: Optional[str] = None,
    level=0,
    **heatmap_kwargs,
):
    r"""Plot 2D heatmap data across all datasets.

    Wrapper function to enable plotting of continuous 2D data across multiple replicates. Only
    one of ``obsm``, ``obsp`` or ``uns`` should be used.

    Args:
        trained_model: the trained Popari model.
        axes: A predefined set of matplotlib axes to plot on.
        obsm: the key in the ``.obsm`` dataframe to plot.
        obsp: the key in the ``.obsp`` dataframe to plot.
        uns: the key in the ``.uns`` dataframe to plot. Unstructured data must be 2D in shape.
        **heatmap_kwargs: arguments to pass to the `ax.imshow` call for each dataset

    """
    datasets = trained_model.hierarchy[level].datasets

    groups = trained_model.metagene_groups if group_type == "metagene" else trained_model.spatial_affinity_groups
    _multigroup_heatmap(datasets, title_font_size=title_font_size, groups=groups, axes=axes, key=key, **heatmap_kwargs)


def compute_empirical_correlations(
    trained_model: Popari,
    feature: str = "X",
    level=0,
    output: str = "empirical_correlation",
):
    """Compute the empirical spatial correlation for a feature set across all
    datasets.

    Args:
        trained_model: the trained Popari model.
        feature: key in `.obsm` of feature set for which spatial correlation should be computed.
        output: key in `.uns` where output correlation matrices should be stored.

    """

    datasets = trained_model.hierarchy[level].datasets

    scaling = trained_model.parameter_optimizer.spatial_affinity_state.scaling

    _compute_empirical_correlations(datasets, scaling, feature=feature, output=output)


def find_differential_genes(trained_model: Popari, level=0, top_gene_limit: int = 1):
    """Identify genes/features that distinguish differential metagenes within a
    group.

    This type of analysis is only valid for runs of Popari in which ``metagene_mode="differential"``
    was used.

    Args:
        trained_model: the trained Popari model.
        top_gene_limit: the number of top genes to mark as differential for each metagene in a dataset.

    Returns:
        The names of genes that are differentially expressed with respect to their group.

    """

    datasets = trained_model.hierarchy[level].datasets

    for dataset in datasets:
        if "M_bar" not in dataset.uns:
            raise ValueError("This model was not trained in differential metagene mode.")

    genes_of_interest = set()
    for dataset in datasets:
        for group_name in trained_model.metagene_tags[dataset.name]:
            image = dataset.uns["M"][dataset.name] - dataset.uns["M_bar"][group_name]
            top_genes_per_metagene = np.argpartition(np.abs(image), -top_gene_limit, axis=0)[-top_gene_limit:]
            dataset_top_genes = dataset.var_names[top_genes_per_metagene.flatten()]
            genes_of_interest.update(dataset_top_genes)

    return genes_of_interest


def plot_gene_activations(trained_model: Popari, gene_subset: Sequence[str], level=0):
    """Plot metagene activation heatmaps for target genes across all groups.

    This type of analysis is only valid for runs of Popari in which ``metagene_mode="differential"``
    was used.

    Args:
        trained_model: the trained Popari model.
        gene_subset: names of genes to plot for.

    Returns:
        The names of genes that are differentially expressed with respect to their group.

    """
    gene_indices = trained_model.datasets[0].var_names.get_indexer(gene_subset)
    images = np.zeros((len(gene_indices), trained_model.K, len(trained_model.metagene_groups)))
    for group_index, group_name in enumerate(trained_model.metagene_groups):
        M_bar_subset = trained_model.datasets[0].uns["M_bar"][group_name][gene_indices]
        images[:, :, group_index] = M_bar_subset

    fig, axes = setup_squarish_axes(len(gene_indices), figsize=(10, 10))
    for ax, image, gene in zip(axes.flat, images, gene_subset):
        aspect = 0.1
        im = ax.imshow(image, interpolation="nearest", aspect=aspect)
        ax.set_title(gene)
        colorbar = fig.colorbar(im, ax=ax, orientation="vertical")


def plot_gene_trajectories(
    trained_model: Popari,
    gene_subset: Sequence[str],
    covariate_values: Sequence[float],
    level=0,
    **subplots_kwargs,
):
    """Plot metagene activation lineplots for target genes across all groups.

    This type of analysis is only valid for runs of Popari in which ``metagene_mode="differential"``
    was used.

    Args:
        trained_model: the trained Popari model.
        gene_subset: names of genes to plot for.
        covariate_values: dependent vairable values against which gene trajectories will be plotted.

    Returns:
        The names of genes that are differentially expressed with respect to their group.

    """
    gene_indices = trained_model.datasets[0].var_names.get_indexer(gene_subset)
    images = np.zeros((len(gene_indices), trained_model.K, len(trained_model.metagene_groups)))
    for group_index, group_name in enumerate(trained_model.metagene_groups):
        M_bar_subset = trained_model.parameter_optimizer.metagene_state.M_bar[group_name][gene_indices]
        images[:, :, group_index] = M_bar_subset

    summed_weights = images.sum(axis=1)
    fig, axes = setup_squarish_axes(len(gene_indices), figsize=(10, 10))
    for ax, trend, gene in zip(axes.flat, summed_weights, gene_subset):
        aspect = 0.1
        r = np.corrcoef(covariate_values, y=trend)[0, 1]
        im = ax.plot(covariate_values, trend)
        ax.set_title(f"{gene}, R = {r:.2f}")


def normalized_affinity_trends(
    trained_model,
    timepoint_values: Sequence[float],
    time_unit="Days",
    normalize: bool = False,
    spatial_affinity_key: str = "Sigma_x_inv",
    n_best: int = 5,
    highlight_metric: str = "pearson",
    figsize: tuple = None,
    margin_size: float = 0.25,
    level=0,
):
    """Plot trends for every pair of affinities; highlight top trends.

    Args:
        trained_model: the trained Popari model.
        timepoint_values: x-values against which to plot trends
        time_unit: unit in which time is measured (used for x-axis label)

    """

    datasets = trained_model.hierarchy[level].datasets
    all_affinities = np.array([dataset.uns[spatial_affinity_key][dataset.name] for dataset in datasets])

    if normalize:
        for index in range(len(datasets), axes.size):
            prenormalization_affinity_std = np.std(all_affinities, axis=0, keepdims=True)
            prenormalization_timepoint_std = np.std(timepoint_values)
            all_affinities /= prenormalization_affinity_std
            timepoint_values /= prenormalization_timepoint_std

    timepoint_min = np.min(timepoint_values)
    timepoint_ptp = np.ptp(timepoint_values)
    timepoint_std = np.std(timepoint_values)

    affinity_min = np.min(all_affinities)
    affinity_ptp = np.ptp(all_affinities)
    affinity_std = np.std(all_affinities, axis=0, keepdims=True)

    pearson_correlations = {}
    variances = {}
    pearson_p_values = {}
    slopes = {}

    for i in range(trained_model.K):
        for j in range(i + 1):
            affinity_values = all_affinities[:, i, j]
            r, p_value = pearsonr(affinity_values, timepoint_values)
            variances[(i, j)] = np.var(affinity_values)
            pearson_correlations[(i, j)] = r
            pearson_p_values[(i, j)] = r
            slopes[(i, j)] = r * affinity_std[0, i, j] / timepoint_std

    if highlight_metric == "pearson":
        pairs, metric_values = zip(*pearson_correlations.items())
    elif highlight_metric == "variance":
        pairs, metric_values = zip(*variances.items())

    pairs = np.array(pairs)
    sorted_values = np.argsort(metric_values)
    best_values = sorted_values[-n_best:][::-1] if n_best > 0 else sorted_values[:-n_best]

    top_pairs = pairs[best_values].tolist()

    for dataset in datasets:
        dataset.uns["spatial_trends"] = {
            "top_pairs": top_pairs,
            "pearson_correlations": pearson_correlations,
            "variances": variances,
            "slopes": slopes,
        }

    return top_pairs, pearson_correlations, variances


def propagate_labels(trained_model, label_key: str, starting_level=None, smooth=False):
    """Propagate a label from the most binned layer to the least binned
    layer."""

    if starting_level is None:
        starting_level = trained_model.hierarchical_levels - 1

    for level in range(starting_level, 0, -1):
        view = trained_model.hierarchy[level]
        high_res_view = trained_model.hierarchy[level - 1]
        for dataset, high_res_dataset in zip(view.datasets, high_res_view.datasets):
            B = dataset.obsm[f"bin_assignments_{dataset.name}"]
            labels = dataset.obs[label_key]

            high_res_labels = np.array(labels) @ B.astype(int).toarray()
            if smooth:
                high_res_labels = spatially_smooth_feature(
                    high_res_labels,
                    high_res_dataset.obs["adjacency_list"],
                    max_smoothing_rounds=200,
                    smoothing_threshold=0.3,
                )

            high_res_dataset.obs[label_key] = list(high_res_labels)
            high_res_dataset.obs[label_key] = high_res_dataset.obs[label_key].astype("category")


def metagene_gsea(trained_model, metagene_index: int, level=0, **gsea_kwargs):
    """Run GSEA on metagenes from trained model."""

    first_dataset = trained_model.hierarchy[level].datasets[0]
    fig = _metagene_gsea(first_dataset, metagene_index, **gsea_kwargs)

    return fig
