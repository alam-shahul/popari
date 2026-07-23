from functools import partial
from typing import Optional, Sequence

import anndata as ad
import numpy as np
import scanpy as sc
import squidpy as sq
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap

from popari._dataset_utils import (
    _evaluate_classification_task,
    _matrix_heatmap,
    _matrix_heatmap_panel,
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
    _spatial_affinity_heatmap,
    for_model,
    setup_squarish_axes,
)
from popari.analysis_utils import metagene_pair_edge_values
from popari.model import Popari

in_situ = for_model(_plot_in_situ, return_outputs=True)
metagene_embedding = for_model(_plot_metagene_embedding, return_outputs=True)
confusion_matrix = for_model(_plot_confusion_matrix, return_outputs=True)
umap = for_model(_plot_umap, return_outputs=True)
multireplicate_heatmap = for_model(_multireplicate_heatmap, return_outputs=True)
spatial_affinity_heatmap = for_model(_spatial_affinity_heatmap, return_outputs=True)
clusters_to_categories = for_model(_plot_clusters_to_categories, return_outputs=True)
metagene_signature_enrichment = for_model(_plot_metagene_signature_enrichment, return_outputs=True)


def _highlight_cell(x, y, ax=None, **kwargs):
    """Draw a rectangle around a heatmap cell."""

    ax = ax or plt.gca()
    rectangle = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, **kwargs)
    ax.add_patch(rectangle)
    return rectangle


def matrix_heatmap(matrix, **heatmap_kwargs):
    """Plot a standalone matrix-like object as a heatmap.

    ``matrix`` may be a :class:`pandas.DataFrame` or array-like object. DataFrame
    indexes and columns are used as axis labels by default.

    """

    return _matrix_heatmap(matrix, **heatmap_kwargs)


def matrix_heatmap_panel(matrices, **heatmap_kwargs):
    """Plot a collection of standalone matrix-like objects as heatmaps.

    ``matrices`` may be a mapping from title to matrix, or a sequence of
    matrix-like objects.

    """

    return _matrix_heatmap_panel(matrices, **heatmap_kwargs)


def edge_interactions(
    dataset: ad.AnnData,
    first_metagene: int,
    second_metagene: int,
    *,
    mode: str = "affinity",
    color: Optional[str] = "cell_type",
    embedding_key: str = "X",
    affinity_key: str = "Sigma_x_inv",
    neighbor_key: str = "adjacency_matrix",
    rescale: bool = True,
    ax: Optional[Axes] = None,
    edge_cmap="Blues",
    edges_width: float = 1,
    size: Optional[float] = None,
    title: Optional[str] = None,
    **spatial_kwargs,
):
    """Plot metagene-pair edge interaction values in situ.

    Edge colors are computed by :func:`popari.analysis_utils.metagene_pair_edge_values`
    and are not stored in ``dataset``.

    """

    edges, edge_values = metagene_pair_edge_values(
        dataset,
        first_metagene,
        second_metagene,
        mode=mode,
        embedding_key=embedding_key,
        affinity_key=affinity_key,
        neighbor_key=neighbor_key,
        rescale=rescale,
    )

    if ax is None:
        dpi = spatial_kwargs.pop("dpi", 100)
        fig, ax = plt.subplots(dpi=dpi)
    else:
        fig = ax.get_figure()

    if isinstance(edge_cmap, str):
        edge_cmap = plt.get_cmap(edge_cmap)

    edges_kwargs = spatial_kwargs.pop("edges_kwargs", {})
    edges_kwargs = {
        "edgelist": edges,
        "edge_cmap": edge_cmap,
        **edges_kwargs,
    }

    if size is None:
        size = 5000 / dataset.n_obs
    if title is None:
        title = f"m{first_metagene} -> m{second_metagene} ({mode})"

    axes = sq.pl.spatial_scatter(
        dataset,
        shape=None,
        color=color,
        connectivity_key=neighbor_key,
        edges_width=edges_width,
        edges_color=edge_values,
        edges_kwargs=edges_kwargs,
        size=size,
        title=title,
        ax=ax,
        return_ax=True,
        **spatial_kwargs,
    )

    if isinstance(axes, np.ndarray):
        return axes.flat[0].get_figure()
    if isinstance(axes, list):
        return axes[0].get_figure()
    if axes is None:
        return fig
    return axes.get_figure()


def affinity_difference(
    dataset: ad.AnnData,
    comparison: str,
    reference: str,
    spatial_affinity_key: str = "Sigma_x_inv",
    ax: Optional[Axes] = None,
    cmap: str = "bwr_r",
    mask_upper: bool = True,
    label_values: bool = False,
    label_font_size: float = 8,
    metagene_labels: Optional[Sequence[str]] = None,
    **imshow_kwargs,
):
    """Plot the spatial affinity difference between two named datasets.

    Args:
        dataset: AnnData object containing named affinity matrices in ``.uns``.
        comparison: Dataset name for the positive term.
        reference: Reference dataset name for the negative term.
        spatial_affinity_key: Key in ``.uns`` containing named affinity matrices.
        ax: Existing matplotlib axis. If ``None``, a new figure and axis are created.
        cmap: Colormap for the centered heatmap.
        mask_upper: If ``True``, mask the upper triangle.
        label_values: If ``True``, label visible heatmap cells.
        label_font_size: Font size for heatmap cell labels.
        metagene_labels: Optional labels for x and y ticks.
        **imshow_kwargs: Additional keyword arguments passed to ``imshow``.

    """

    difference = dataset.popari.affinity_difference(
        comparison,
        reference,
        spatial_affinity_key=spatial_affinity_key,
    )

    if mask_upper:
        mask = np.zeros_like(difference, dtype=bool)
        mask[np.triu_indices_from(mask, k=1)] = True
        image = np.ma.masked_where(mask, difference)
    else:
        image = difference

    max_value = np.max(np.abs(difference))
    if max_value == 0:
        max_value = 1.0
    vmin = imshow_kwargs.pop("vmin", -max_value)
    vmax = imshow_kwargs.pop("vmax", max_value)

    if ax is None:
        dpi = imshow_kwargs.pop("dpi", 100)
        fig, ax = plt.subplots(dpi=dpi)
    else:
        fig = ax.get_figure()

    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", **imshow_kwargs)
    ax.set_title(f"{comparison} - {reference}")

    if metagene_labels is None:
        metagene_labels = [f"m{index}" for index in range(difference.shape[0])]
    ax.set_xticks(np.arange(difference.shape[1]), metagene_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(difference.shape[0]), metagene_labels)

    if label_values:
        for (row, column), value in np.ndenumerate(difference):
            if not mask_upper or not mask[row, column]:
                ax.text(column, row, f"{value:.2g}", ha="center", va="center", fontsize=label_font_size)

    plt.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    return fig


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
    return _multigroup_heatmap(
        datasets,
        title_font_size=title_font_size,
        groups=groups,
        axes=axes,
        key=key,
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

    return _plot_all_embeddings(datasets, embedding_key=embedding_key, column_names=column_names, **spatial_kwargs)


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

    return medians, fig


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
    group_suffix = f"_level_{level}" if level > 0 else ""

    fig, axes = setup_squarish_axes(len(datasets), figsize=figsize)

    all_top_pairs = []

    for index in range(len(datasets), axes.size):
        axes.flat[index].axis("off")

    for ax, (index, dataset) in zip(axes.flat, enumerate(datasets)):
        dataset.uns["delta_Sigma"] = {
            dataset.popari.name: dataset.uns[spatial_affinity_key][dataset.popari.name]
            - dataset.uns[spatial_affinity_bar_key][f"_default{group_suffix}"],
        }

        Sigma_x_inv = dataset.uns[spatial_affinity_key][dataset.popari.name]
        delta_Sigma = dataset.uns["delta_Sigma"][dataset.popari.name]
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

    all_affinities = np.array([dataset.uns[spatial_affinity_key][dataset.popari.name] for dataset in datasets])

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
