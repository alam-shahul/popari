"""Spatial and embedding plots for AnnData results."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import anndata as ad
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import squidpy as sq
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap

from popari._datasets import as_datasets, broadcast_plottable, enable_joint
from popari.analysis.gene_sets import order_columns_by_best_row
from popari.analysis.metrics import score_marker_expression
from popari.plotting.utils import setup_squarish_axes
from popari.util import concatenate, unconcatenate


def metagene_embedding(
    datasets: ad.AnnData | Sequence[ad.AnnData],
    metagene_index: int,
    axes: Sequence[Axes] | None = None,
    **scatterplot_kwargs,
):
    r"""Plot a single metagene in-situ across all datasets.

    Args:
        datasets: list of datasets to plot
        metagene_index: the index of the metagene to plot.
        axes: A predefined set of matplotlib axes to plot on.

    """

    datasets = as_datasets(datasets)
    legend = scatterplot_kwargs.pop("legend", False)
    default_s = scatterplot_kwargs.pop("s", None)
    linewidth = scatterplot_kwargs.pop("linewidth", 0)
    palette = scatterplot_kwargs.pop("palette", "viridis")
    dpi = scatterplot_kwargs.pop("dpi", 100)

    if axes is None:
        fig, axes = setup_squarish_axes(len(datasets), sharex=False, sharey=False, dpi=dpi)

    else:
        axes = np.asarray(axes, dtype=object)
        fig = axes.flat[0].get_figure()

    for index in range(len(datasets), axes.size):
        axes.flat[index].axis("off")

    for dataset, ax in zip(datasets, axes.flat):
        if default_s is None:
            s = round(10000 / len(dataset))
        else:
            s = default_s

        ax.set_aspect("equal", "box")
        ax.invert_yaxis()
        ax.set_xticks([], [])  # note you need two lists one for the positions and one for the labels
        ax.set_yticks([], [])  # same for y ticks
        points = dataset.obsm["spatial"]
        embedding_label = f"Metagene {metagene_index}"
        plot_data = pd.DataFrame(
            {
                "x": points[:, 0],
                "y": points[:, 1],
                embedding_label: dataset.popari.embedding[:, metagene_index],
            },
        )
        sns.scatterplot(
            data=plot_data,
            x="x",
            y="y",
            hue=embedding_label,
            legend=legend,
            s=s,
            linewidth=linewidth,
            palette=palette,
            ax=ax,
            **scatterplot_kwargs,
        )

    fig.suptitle(f"Metagene {metagene_index}")

    return fig


def in_situ(
    data: ad.AnnData | Sequence[ad.AnnData],
    axes=None,
    fig=None,
    color="leiden",
    figsize=None,
    **spatial_kwargs,
):
    r"""Plot an observation annotation across all datasets in situ.

    Extends AnnData's ``sc.pl.spatial`` function to plot labels/values across multiple replicates.

    Args:
        data: One AnnData object or a sequence of datasets.
        color: Key in ``obs`` containing categorical or continuous values.
        axes: A predefined set of matplotlib axes to plot on.
        fig: Figure containing ``axes``.
        figsize: Figure size used when creating axes internally.
        **spatial_kwargs: Additional arguments for Squidpy's spatial scatter plot.

    """

    datasets = as_datasets(data)
    spatial_kwargs.pop("joint", None)

    sharex = spatial_kwargs.pop("sharex", False)
    sharey = spatial_kwargs.pop("sharey", False)
    dpi = spatial_kwargs.pop("dpi", 100)

    if axes is None:
        fig, axes = setup_squarish_axes(
            len(datasets),
            sharex=sharex,
            sharey=sharey,
            dpi=dpi,
            figsize=figsize,
        )
    else:
        axes = np.asarray(axes, dtype=object)
        if fig is None:
            fig = axes.flat[0].get_figure()

    edges_width = spatial_kwargs.pop("edges_width", 0.2)
    default_size = spatial_kwargs.pop("size", None)
    palette = spatial_kwargs.pop("palette", None)

    legend_fontsize = spatial_kwargs.pop("legend_fontsize", "xx-small")
    edgecolors = spatial_kwargs.pop("edgecolors", "none")
    connectivity_key = spatial_kwargs.pop("connectivity_key", "adjacency_matrix")
    if not edges_width:
        connectivity_key = None
    shape = spatial_kwargs.pop("shape", None)
    library_key = spatial_kwargs.pop("library_key", "batch")
    spatial_kwargs.pop("neighbors_key", None)

    size = 5000 / sum(dataset.n_obs for dataset in datasets)
    if default_size is not None:
        size *= default_size

    categorical = isinstance(datasets[0].obs[color].dtype, pd.CategoricalDtype)
    category_colors = None
    if categorical and palette is None:
        categories = list(
            dict.fromkeys(category for dataset in datasets for category in dataset.obs[color].cat.categories),
        )
        colors = np.asarray(sc.pl.palettes.godsnot_102)
        color_indices = np.linspace(0, len(colors) - 1, len(categories), dtype=int)
        category_colors = dict(zip(categories, colors[color_indices]))
    if not categorical:
        values = np.concatenate([dataset.obs[color].to_numpy(dtype=float) for dataset in datasets])
        finite_values = values[np.isfinite(values)]
        if finite_values.size:
            spatial_kwargs.setdefault("vmin", finite_values.min())
            spatial_kwargs.setdefault("vmax", finite_values.max())

    for dataset, ax in zip(datasets, axes.flat):
        dataset_palette = palette
        if category_colors is not None:
            dataset_palette = ListedColormap(
                [category_colors[category] for category in dataset.obs[color].cat.categories],
            )

        sq.pl.spatial_scatter(
            dataset,
            shape=shape,
            size=size,
            connectivity_key=connectivity_key,
            color=color,
            edges_width=edges_width,
            legend_fontsize=legend_fontsize,
            ax=ax,
            title=dataset.popari.name,
            fig=fig,
            palette=dataset_palette,
            edgecolors=edgecolors,
            library_key=library_key,
            **spatial_kwargs,
        )

    return fig


@enable_joint
@broadcast_plottable
def umap(dataset: ad.AnnData, color="leiden", ax=None, **kwargs):
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


def all_embeddings(
    dataset: ad.AnnData | Sequence[ad.AnnData],
    embedding_key: str = "X",
    column_names: str | None = None,
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

    datasets = as_datasets(dataset)
    dataset = datasets[0] if len(datasets) == 1 else concatenate(datasets)
    if "ax" in spatial_kwargs:
        axes = spatial_kwargs["ax"]
        if isinstance(axes, np.flatiter):
            spatial_kwargs["ax"] = np.asarray(list(axes), dtype=object)

    _, K = dataset.obsm[f"{embedding_key}"].shape
    if column_names == None:
        column_names = [f"{embedding_key}_{index}" for index in range(K)]

    edges_width = spatial_kwargs.pop("edges_width", 0.2)
    connectivity_key = spatial_kwargs.pop(
        "connectivity_key",
        "adjacency_matrix" if "adjacency_matrix" in dataset.obsp else None,
    )
    default_size = spatial_kwargs.pop("size", None)
    palette = spatial_kwargs.pop("palette", ListedColormap(sc.pl.palettes.godsnot_102))

    size = len(dataset) / 100
    if default_size is not None:
        size *= default_size

    axes = sq.pl.spatial_scatter(
        sq.pl.extract(dataset, embedding_key, prefix=f"{embedding_key}"),
        shape=None,
        color=column_names,
        edges_width=edges_width,
        connectivity_key=connectivity_key,
        size=size,
        wspace=0.2,
        ncols=2,
        return_ax=True,
        **spatial_kwargs,
    )

    if isinstance(axes, np.ndarray):
        return axes.flat[0].get_figure()

    if isinstance(axes, list):
        return axes[0].get_figure()

    if axes is None:
        return plt.gcf()

    return axes.get_figure()


def embedding_label_dotplot(
    dataset,
    ax=None,
    names: Sequence[str] | None = None,
    embedding_key: str = "normalized_X",
    label_key: str = "leiden",
    add_totals: bool = False,
    **dotplot_kwargs,
):
    """Plot embedding activity and prevalence for each label category.

    Args:
        dataset: AnnData containing embeddings and categorical labels.
        ax: Existing axis on which to draw.
        names: Names of embedding dimensions.
        embedding_key: Key in ``obsm`` containing the embedding matrix.
        label_key: Categorical observation column used to group cells.
        add_totals: Whether to add category-size totals to the dot plot.
        **dotplot_kwargs: Additional arguments for :func:`scanpy.pl.dotplot`.

    Returns:
        Scanpy ``DotPlot`` object.

    """

    embeddings = dataset.obsm[embedding_key]
    num_features = embeddings.shape[1]
    mock_dataset = ad.AnnData(X=embeddings, obs=dataset.obs.copy())

    if not names:
        mock_dataset.var_names = [f"m{index}" for index in range(num_features)]
    else:
        mock_dataset.var_names = names

    swap_axes = dotplot_kwargs.pop("swap_axes", True)
    standard_scale = dotplot_kwargs.pop("standard_scale", "var")

    if standard_scale is None and "vmin" not in dotplot_kwargs and "vmax" not in dotplot_kwargs:
        aggregated = sc.get.aggregate(mock_dataset, by=label_key, func=["sum", "count_nonzero"])
        mean_in_expressed = aggregated.to_df(layer="sum") / aggregated.to_df(layer="count_nonzero")
        max_value = np.max(np.abs(mean_in_expressed))
        dotplot_kwargs.update(vmin=-max_value, vmax=max_value)

    dotplot = sc.pl.dotplot(
        mock_dataset,
        mock_dataset.var_names,
        groupby=label_key,
        dendrogram=False,
        ax=ax,
        standard_scale=standard_scale,
        swap_axes=swap_axes,
        return_fig=True,
        **dotplot_kwargs,
    )

    if add_totals:
        dotplot.add_totals()
    dotplot.show()
    return dotplot


def embedding_label_heatmap(
    data: ad.AnnData | Sequence[ad.AnnData],
    *,
    names: Sequence[str] | None = None,
    embedding_key: str = "normalized_X",
    label_key: str = "leiden",
    excluded_categories: Sequence[str] | None = None,
    title: str | None = None,
):
    """Plot normalized mean embedding activity for each categorical label."""

    datasets = as_datasets(data)
    dataset = datasets[0] if len(datasets) == 1 else concatenate(datasets)
    if excluded_categories is not None:
        dataset = dataset[~dataset.obs[label_key].isin(excluded_categories)]

    embeddings = dataset.obsm[embedding_key]
    num_features = embeddings.shape[1]
    mock_dataset = ad.AnnData(X=embeddings)
    mock_dataset.var_names = [f"m{index}" for index in range(num_features)] if names is None else names
    mock_dataset.obs = dataset.obs.copy()

    aggregated = sc.get.aggregate(mock_dataset, by=label_key, func=["sum", "count_nonzero"])
    mean_expression = aggregated.to_df(layer="sum") / aggregated.to_df(layer="count_nonzero")
    values = mean_expression.to_numpy()
    values -= values.min(axis=0)
    column_max = values.max(axis=0)
    values = np.divide(values, column_max, out=np.zeros_like(values), where=column_max != 0)
    order = order_columns_by_best_row(values)
    values = values[:, order]

    fig = plt.figure(figsize=(num_features * 0.3, len(mean_expression.index) * 0.5))
    grid_spec = gridspec.GridSpec(2, 1, height_ratios=[1, 0.05])
    ax = plt.subplot(grid_spec[0])
    colorbar_ax = plt.subplot(grid_spec[1])
    image = ax.pcolormesh(values, cmap="magma", edgecolor="k")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks(np.arange(num_features), [f"m{index}" for index in order], rotation=90)
    ax.set_yticks(np.arange(len(mean_expression.index)), mean_expression.index)
    if title is not None:
        ax.set_title(title)
    fig.colorbar(image, cax=colorbar_ax, orientation="horizontal")
    return fig, values


def clusters_to_categories(
    datasets: ad.AnnData | Sequence[ad.AnnData],
    category_de_genes: dict[str, Sequence[str]],
    output_key="marker_expression",
    **dotplot_kwargs,
):
    """Plot clusters to category correspondence, based on category marker gene
    expression.

    Args:
        category_de_genes: mapping from category to marker genes for that category

    """
    datasets = as_datasets(datasets)
    score_marker_expression(datasets, category_de_genes, output_key=output_key)
    merged_dataset = datasets[0] if len(datasets) == 1 else concatenate(datasets)
    dotplot = embedding_label_dotplot(
        merged_dataset,
        names=list(category_de_genes.keys()),
        standard_scale=None,
        cmap="bwr",
        embedding_key=output_key,
        title="Cell-type-to-cluster Correspondence",
        add_totals=True,
        **dotplot_kwargs,
    )

    return dotplot.fig
