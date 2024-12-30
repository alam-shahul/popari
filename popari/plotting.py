from functools import partial
from typing import Optional, Sequence

import numpy as np
import scanpy as sc
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap

from popari._dataset_utils import (
    _evaluate_classification_task,
    _multigroup_heatmap,
    _multireplicate_heatmap,
    _plot_all_embeddings,
    _plot_cell_type_to_metagene,
    _plot_cell_type_to_metagene_difference,
    _plot_clusters_to_categories,
    _plot_confusion_matrix,
    _plot_in_situ,
    _plot_metagene_embedding,
    _plot_metagene_signature_enrichment,
    _plot_umap,
    for_model,
    setup_squarish_axes,
)
from popari.model import Popari

in_situ = for_model(_plot_in_situ, return_outputs=True)
metagene_embedding = for_model(_plot_metagene_embedding, return_outputs=True)
confusion_matrix = for_model(_plot_confusion_matrix, return_outputs=True)
umap = for_model(_plot_umap, return_outputs=True)
multireplicate_heatmap = for_model(_multireplicate_heatmap, return_outputs=True)
clusters_to_categories = for_model(_plot_clusters_to_categories, return_outputs=True)
metagene_signature_enrichment = for_model(_plot_metagene_signature_enrichment, return_outputs=True)


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


def spatial_affinities(
    trained_model: Popari,
    title_font_size: Optional[int] = None,
    spatial_affinity_key: Optional[str] = "Sigma_x_inv",
    axes: Optional[Sequence[Axes]] = None,
    level=0,
    **heatmap_kwargs,
):
    r"""Plot Sigma_x_inv across all datasets.

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

    # Override following kwargs with
    cmap = heatmap_kwargs.pop("cmap") if "cmap" in heatmap_kwargs else "bwr"
    nested = heatmap_kwargs.pop("nested") if "nested" in heatmap_kwargs else True
    max_value = round(
        np.max(np.abs(np.array([dataset.uns[spatial_affinity_key][dataset.name] for dataset in datasets]))),
    )
    vmin = -max_value
    vmax = max_value

    _multireplicate_heatmap(
        datasets,
        title_font_size=title_font_size,
        axes=axes,
        uns=spatial_affinity_key,
        nested=nested,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        **heatmap_kwargs,
    )


def all_embeddings(
    trained_model: Popari,
    embedding_key: str = "X",
    column_names: Optional[str] = None,
    level=0,
    **spatial_kwargs,
):
    r"""Plot all learned metagenes in-situ across all replicates.

    Each replicate's metagenes are contained in a separate plot.

    Args:
        trained_model: the trained Popari model.
        embedding_key: the key in the ``.obsm`` dataframe for the cell/spot embeddings.
        column_names: a list of the suffixes for each latent feature. If ``None``, it is assumed
            that these suffixes are just the indices of the latent features.

    """

    datasets = trained_model.hierarchy[level].datasets

    first_dataset = datasets[0]
    _, K = first_dataset.obsm[f"{embedding_key}"].shape

    if column_names == None:
        column_names = [f"{embedding_key}_{index}" for index in range(K)]

    _plot_all_embeddings(datasets, embedding_key=embedding_key, column_names=column_names, **spatial_kwargs)


def cell_type_to_metagene(trained_model: Popari, cell_type_de_genes: dict, level=0, **correspondence_kwargs):
    r"""Plot distribution of gene ranks of marker genes within each metagene.

    Args:
        trained_model: the trained Popari model.
        cell_type_de_genes: dictionary mapping each cell type to a list of marker genes.

    Returns:
        mapping from each cell type to the median rank of its marker genes in each metagene

    """

    datasets = trained_model.hierarchy[level].datasets

    first_dataset = datasets[0]

    fig, medians = _plot_cell_type_to_metagene(first_dataset, cell_type_de_genes, **correspondence_kwargs)

    return medians, fig


def cell_type_to_metagene_difference(
    trained_model: Popari,
    cell_type_de_genes: dict,
    first_metagene: int,
    second_metagene: int,
    level=0,
    **correspondence_kwargs,
):
    r"""Plot distribution of gene ranks of marker genes within each metagene.

    Args:
        trained_model: the trained Popari model.
        cell_type_de_genes: dictionary mapping each cell type to a list of marker genes.

    Returns:
        mapping from each cell type to the median rank of its marker genes in each metagene

    """

    datasets = trained_model.hierarchy[level].datasets

    first_dataset = datasets[0]

    fig, medians = _plot_cell_type_to_metagene_difference(
        first_dataset,
        cell_type_de_genes,
        first_metagene,
        second_metagene,
        **correspondence_kwargs,
    )


def affinity_magnitude_vs_difference(
    trained_model,
    level=0,
    spatial_affinity_key: str = "Sigma_x_inv",
    spatial_affinity_bar_key: str = "spatial_affinity_bar",
    joint=False,
    figsize=(10, 10),
    n_best: int = 5,
):
    """Plot all pairwise affinities, in terms of absolute and relative
    magnitude.

    Args:
        trained_model: the trained Popari model.

    """
    datasets = trained_model.hierarchy[level].datasets
    group_suffix = f"level_{level}" if level > 0 else ""

    fig, axes = setup_squarish_axes(len(datasets), figsize=figsize)

    all_top_pairs = []

    for index in range(len(datasets), axes.size):
        axes.flat[index].axis("off")

    for ax, (index, dataset) in zip(axes.flat, enumerate(datasets)):
        dataset.uns["delta_Sigma"] = {
            dataset.name: dataset.uns[spatial_affinity_key][dataset.name]
            - dataset.uns[spatial_affinity_bar_key][f"_default_{group_suffix}"],
        }

        Sigma_x_inv = dataset.uns[spatial_affinity_key][dataset.name]
        delta_Sigma = dataset.uns["delta_Sigma"][dataset.name]
        pairs = {}
        for i in range(trained_model.K):
            for j in range(i + 1):
                pairs[(i, j)] = (delta_Sigma[(i, j)], Sigma_x_inv[(i, j)])

        indices, flat_pairs = zip(*pairs.items())

        magnitudes = np.linalg.norm(flat_pairs, axis=1)

        sorted_index = np.argsort(magnitudes)
        best_index = sorted_index[-n_best:][::-1]

        x, y = np.array(flat_pairs).T

        ax.scatter(x, y, s=1, color="#D3D3D3")

        num_top_points = abs(n_best)
        colors = sc.pl.palettes.godsnot_102[:num_top_points]

        best_indices = np.array(indices)[best_index]
        for (i, j), color in zip(best_indices, colors):
            x, y = pairs[(i, j)]
            ax.scatter(x, y, s=20, color=color, label=f"m{i} × m{j}")

        ax.set_title("Pairwise affinity scatter")
        ax.set_xlabel("Difference from average affinity")
        ax.set_ylabel("Pairwise affinity")
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

        all_top_pairs.append(best_indices)

    return fig, all_top_pairs


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

    first_dataset = datasets[0]
    spatial_trends = first_dataset.uns["spatial_trends"]

    all_affinities = np.array([dataset.uns[spatial_affinity_key][dataset.name] for dataset in datasets])

    if normalize:
        for index in range(len(datasets), axes.size):
            axes.flat[index].axis("off")
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

    if figsize is None:
        figsize = (10, 5)

    fig, ax = plt.subplots(dpi=300, figsize=figsize)

    ax.set_ylim([affinity_min - margin_size, affinity_min + affinity_ptp + margin_size])
    ax.set_xlim([timepoint_min - margin_size, timepoint_min + timepoint_ptp + margin_size])
    ax.set_xticks(timepoint_values)

    number_of_lines = abs(n_best)
    colors = cm.get_cmap("rainbow", number_of_lines)

    for i in range(trained_model.K):
        for j in range(i + 1):
            affinity_values = all_affinities[:, i, j]
            if [i, j] not in spatial_trends["top_pairs"]:
                #             if True:
                line = ax.plot(timepoint_values, affinity_values, color="#D3D3D3", linestyle="--", linewidth=0.5)

    for index, (i, j) in enumerate(spatial_trends["top_pairs"]):
        affinity_values = all_affinities[:, i, j]
        if highlight_metric == "pearson":
            r = spatial_trends["pearson_correlations"][(i, j)]
            slope = spatial_trends["slopes"][(i, j)]
            slope_display = f", slope={slope:.2f}" if not normalize else ""
            label = f"m{i} × m{j}, r={r:.2}{slope_display}"
        elif highlight_metric == "variance":
            variance = spatial_trends["variances"][(i, j)]
            label = f"m{i} × m{j}, σ={variance:.2f}"

        color = colors(index)
        ax.plot(timepoint_values, affinity_values, color=color, linestyle="-", linewidth=3, label=label, zorder=2)

    ax.set_title("Pairwise affinity trends")
    ax.set_xlabel(f"{time_unit}")
    ax.set_ylabel("Pairwise affinity")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 0))

    return fig
