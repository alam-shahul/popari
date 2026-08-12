"""Spatial and embedding plots for AnnData results."""

from __future__ import annotations

from collections.abc import Sequence

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
from matplotlib.lines import Line2D

from popari.analysis.gene_sets import order_columns_by_best_row
from popari.analysis.metrics import score_marker_expression
from popari.plotting._samples import resolve_samples
from popari.plotting.utils import setup_squarish_axes


def metagene_embedding(
    adata: ad.AnnData,
    metagene_index: int,
    *,
    samples: str | Sequence[str] | None = None,
    axes: Sequence[Axes] | None = None,
    figsize=None,
    **scatterplot_kwargs,
):
    r"""Plot a single metagene in situ across selected samples.

    Args:
        adata: Unified multisample AnnData containing spatial coordinates and embeddings.
        metagene_index: Index of the metagene to plot.
        samples: Sample names to plot. By default, plot every sample.
        axes: A predefined set of matplotlib axes to plot on.
        figsize: Size of the complete figure when creating axes.

    """

    sample_axis, selected_samples = resolve_samples(
        adata,
        samples=samples,
    )
    legend = scatterplot_kwargs.pop("legend", False)
    default_s = scatterplot_kwargs.pop("s", None)
    linewidth = scatterplot_kwargs.pop("linewidth", 0)
    palette = scatterplot_kwargs.pop("palette", "viridis")
    dpi = scatterplot_kwargs.pop("dpi", 100)

    if axes is None:
        fig, axes = setup_squarish_axes(
            len(selected_samples),
            sharex=False,
            sharey=False,
            dpi=dpi,
            figsize=figsize,
        )

    else:
        axes = np.asarray(axes, dtype=object)
        fig = axes.flat[0].get_figure()

    if axes.size < len(selected_samples):
        raise ValueError("axes must provide at least one axis per selected sample.")
    for index in range(len(selected_samples), axes.size):
        axes.flat[index].axis("off")

    selected_indices = np.concatenate([sample_axis.indices(sample) for sample in selected_samples])
    selected_values = np.asarray(adata.obsm["X"])[selected_indices, metagene_index]
    scatterplot_kwargs.setdefault("hue_norm", (selected_values.min(), selected_values.max()))

    for sample, ax in zip(selected_samples, axes.flat):
        dataset = adata[sample_axis.indices(sample)]
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
        ax.set_title(sample)

    fig.suptitle(f"Metagene {metagene_index}")

    return fig


def in_situ(
    adata: ad.AnnData,
    *,
    samples: str | Sequence[str] | None = None,
    color="leiden",
    figsize=None,
    **spatial_kwargs,
):
    r"""Plot an observation annotation across selected samples in situ.

    Extends Squidpy's spatial scatter plot by faceting a unified AnnData object
    over its sample axis.

    Args:
        adata: Unified multisample AnnData object.
        samples: Sample names to plot. By default, plot every sample.
        color: Key in ``obs`` containing categorical or continuous values.
        figsize: Size of the complete figure.
        **spatial_kwargs: Additional arguments for Squidpy's spatial scatter plot.

    """

    sample_axis, selected_samples = resolve_samples(
        adata,
        samples=samples,
    )
    spatial_kwargs.pop("joint", None)

    dpi = spatial_kwargs.pop("dpi", 100)
    fig, axes = setup_squarish_axes(
        len(selected_samples),
        sharex=spatial_kwargs.pop("sharex", False),
        sharey=spatial_kwargs.pop("sharey", False),
        dpi=dpi,
        figsize=figsize,
    )
    for index in range(len(selected_samples), axes.size):
        axes.flat[index].axis("off")

    edges_width = spatial_kwargs.pop("edges_width", 0.2)
    default_size = spatial_kwargs.pop("size", None)
    palette = spatial_kwargs.pop("palette", None)
    legend_fontsize = spatial_kwargs.pop("legend_fontsize", "xx-small")
    connectivity_key = spatial_kwargs.pop("connectivity_key", "adjacency_matrix")
    if not edges_width or connectivity_key not in adata.obsp:
        connectivity_key = None
    shape = spatial_kwargs.pop("shape", None)
    spatial_kwargs.pop("library_key", None)
    library_key = sample_axis.sample_key
    spatial_kwargs.pop("neighbors_key", None)
    spatial_kwargs.setdefault("edgecolors", "none")
    spatial_kwargs.setdefault("linewidths", 0)

    selected_indices = np.concatenate([sample_axis.indices(sample) for sample in selected_samples])
    size = 5000 / len(selected_indices)
    if default_size is not None:
        size *= default_size

    color_values = adata.obs[color]
    categorical = not pd.api.types.is_numeric_dtype(color_values.dtype)
    if categorical and palette is None:
        categories = (
            color_values.cat.categories
            if isinstance(color_values.dtype, pd.CategoricalDtype)
            else pd.Index(color_values.dropna().unique())
        )
        colors = np.asarray(sc.pl.palettes.godsnot_102)
        color_indices = np.linspace(0, len(colors) - 1, len(categories), dtype=int)
        palette = ListedColormap(colors[color_indices])
    if not categorical:
        values = adata.obs.iloc[selected_indices][color].to_numpy(dtype=float)
        finite_values = values[np.isfinite(values)]
        if finite_values.size:
            spatial_kwargs.setdefault("vmin", finite_values.min())
            spatial_kwargs.setdefault("vmax", finite_values.max())
    spatial_kwargs.setdefault("title", list(selected_samples))

    selected_axes = list(axes.flat[: len(selected_samples)])
    if len(selected_axes) == 1:
        selected_axes = selected_axes[0]

    # Squidpy subsets each library before assigning categorical colors. Keep
    # globally defined categories so a category missing from one sample does
    # not shift the colors of every subsequent category in the palette.
    with ad.settings.override(remove_unused_categories=False):
        sq.pl.spatial_scatter(
            adata,
            shape=shape,
            color=color,
            library_key=library_key,
            library_id=list(selected_samples),
            connectivity_key=connectivity_key,
            edges_width=edges_width,
            size=size,
            palette=palette,
            legend_fontsize=legend_fontsize,
            ax=selected_axes,
            fig=fig,
            **spatial_kwargs,
        )

    return fig


def umap(
    adata: ad.AnnData,
    color="leiden",
    *,
    samples: str | Sequence[str] | None = None,
    axes=None,
    figsize=None,
    **kwargs,
):
    r"""Plot a unified UMAP embedding faceted by sample.

    Args:
        adata: Unified multisample AnnData containing ``obsm["X_umap"]``.
        color: Observation annotation to plot.
        samples: Sample names to plot. By default, plot every sample.
        axes: A predefined set of matplotlib axes to plot on.
        figsize: Size of the complete figure when creating axes.
        **kwargs: Additional arguments for :func:`scanpy.pl.umap`.

    Returns:
        Figure and the axes used for the sample facets.

    """

    sample_axis, selected_samples = resolve_samples(
        adata,
        samples=samples,
    )
    sharex = kwargs.pop("sharex", True)
    sharey = kwargs.pop("sharey", True)
    dpi = kwargs.pop("dpi", 100)
    if axes is None:
        fig, axes = setup_squarish_axes(
            len(selected_samples),
            sharex=sharex,
            sharey=sharey,
            dpi=dpi,
            figsize=figsize,
        )
    else:
        axes = np.asarray(axes, dtype=object)
        fig = axes.flat[0].get_figure()
    if axes.size < len(selected_samples):
        raise ValueError("axes must provide at least one axis per selected sample.")
    for index in range(len(selected_samples), axes.size):
        axes.flat[index].axis("off")

    edges_width = kwargs.pop("edges_width", 0.2)
    size = kwargs.pop("size", 0.04)
    edges = kwargs.pop("edges", False)
    palette = kwargs.pop("palette", sc.pl.palettes.godsnot_102)
    legend_fontsize = kwargs.pop("legend_fontsize", "xx-small")
    legend_loc = kwargs.pop("legend_loc", "right margin")
    neighbors_key = kwargs.pop("neighbors_key", "neighbors")

    color_values = adata.obs[color]
    categorical = not pd.api.types.is_numeric_dtype(color_values.dtype)
    if categorical:
        categories = (
            color_values.cat.categories
            if isinstance(color_values.dtype, pd.CategoricalDtype)
            else pd.Index(color_values.dropna().unique())
        )
        if isinstance(palette, dict):
            palette = {category: palette[category] for category in categories}
        else:
            colors = sns.color_palette(palette, n_colors=len(categories))
            palette = dict(zip(categories, colors))

    for sample, ax in zip(selected_samples, axes.flat):
        sample_adata = adata[sample_axis.indices(sample)].copy()
        sc.pl.umap(
            sample_adata,
            size=size,
            neighbors_key=neighbors_key,
            color=color,
            edges=edges,
            edges_width=edges_width,
            legend_fontsize=legend_fontsize,
            legend_loc="none" if categorical else legend_loc,
            ax=ax,
            show=False,
            palette=palette,
            title=sample,
            **kwargs,
        )

    if categorical and legend_loc != "none":
        handles = [
            Line2D([], [], marker="o", linestyle="none", color=palette[category], label=str(category))
            for category in categories
        ]
        fig.legend(
            handles=handles,
            loc="outside right center",
            title=color,
            fontsize=legend_fontsize,
        )
    return fig, axes


def all_embeddings(
    adata: ad.AnnData,
    embedding_key: str = "X",
    column_names: Sequence[str] | None = None,
    *,
    samples: str | Sequence[str] | None = None,
    axes=None,
    **spatial_kwargs,
):
    r"""Plot every embedding dimension in situ across selected samples.

    Rows correspond to samples and columns correspond to embedding dimensions.

    Args:
        adata: Unified multisample AnnData object.
        embedding_key: Key in ``obsm`` containing cell or spot embeddings.
        column_names: Display names for embedding dimensions.
        samples: Sample names to plot. By default, plot every sample.
        axes: A predefined array of axes with one cell per sample and embedding.
        **spatial_kwargs: Additional arguments for Squidpy's spatial scatter plot.

    """

    sample_axis, selected_samples = resolve_samples(
        adata,
        samples=samples,
    )
    if axes is None:
        axes = spatial_kwargs.pop("ax", None)
    else:
        spatial_kwargs.pop("ax", None)
    if isinstance(axes, np.flatiter):
        axes = np.asarray(list(axes), dtype=object)

    _, K = adata.obsm[embedding_key].shape
    if column_names is None:
        column_names = [f"m_{index}" for index in range(K)]
    elif len(column_names) != K:
        raise ValueError(f"column_names must contain {K} labels.")

    edges_width = spatial_kwargs.pop("edges_width", 0.2)
    connectivity_key = spatial_kwargs.pop(
        "connectivity_key",
        "adjacency_matrix" if "adjacency_matrix" in adata.obsp else None,
    )
    if not edges_width:
        connectivity_key = None
    default_size = spatial_kwargs.pop("size", None)
    cmap = spatial_kwargs.pop("cmap", "viridis")
    dpi = spatial_kwargs.pop("dpi", 100)
    figsize = spatial_kwargs.pop("figsize", None)
    spatial_kwargs.setdefault("linewidths", 0)
    expected_axes = len(selected_samples) * K
    if axes is None:
        fig, axes = plt.subplots(
            len(selected_samples),
            K,
            squeeze=False,
            dpi=dpi,
            figsize=figsize,
        )
    else:
        axes = np.asarray(axes, dtype=object)
        if axes.size < expected_axes:
            raise ValueError(
                f"axes must provide at least {expected_axes} axes for "
                f"{len(selected_samples)} samples and {K} embeddings.",
            )
        fig = axes.flat[0].get_figure()
    axes = np.asarray(axes, dtype=object)

    selected_indices = np.concatenate([sample_axis.indices(sample) for sample in selected_samples])
    selected_embeddings = np.asarray(adata.obsm[embedding_key])[selected_indices]
    limits = [
        (np.nanmin(selected_embeddings[:, index]), np.nanmax(selected_embeddings[:, index])) for index in range(K)
    ]
    size = len(selected_indices) / 100
    if default_size is not None:
        size *= default_size

    for sample_index, sample in enumerate(selected_samples):
        dataset = adata[sample_axis.indices(sample)]
        extracted = sq.pl.extract(dataset, embedding_key, prefix="m")
        for feature_index, column_name in enumerate(column_names):
            ax = axes.flat[sample_index * K + feature_index]
            vmin, vmax = limits[feature_index]
            sq.pl.spatial_scatter(
                extracted,
                shape=None,
                color=column_name,
                edges_width=edges_width,
                connectivity_key=connectivity_key,
                size=size,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                library_key=sample_axis.sample_key,
                library_id=sample,
                title=column_name if sample_index == 0 else "",
                ax=ax,
                **spatial_kwargs,
            )
            ax.set_title(column_name if sample_index == 0 else "")
            ax.set_xlabel("")
            ax.set_ylabel(sample if feature_index == 0 else "")
            ax.set_xticks([])
            ax.set_yticks([])
    for index in range(expected_axes, axes.size):
        axes.flat[index].axis("off")
    return fig


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
    embedding_names = [f"m{index}" for index in range(num_features)] if not names else list(names)
    group_key = "_popari_label"
    while group_key in embedding_names:
        group_key = f"_{group_key}"
    mock_obs = dataset.obs[[label_key]].rename(columns={label_key: group_key}).copy()
    mock_dataset = ad.AnnData(X=embeddings, obs=mock_obs)
    mock_dataset.var_names = embedding_names

    swap_axes = dotplot_kwargs.pop("swap_axes", True)
    standard_scale = dotplot_kwargs.pop("standard_scale", "var")

    if standard_scale is None and "vmin" not in dotplot_kwargs and "vmax" not in dotplot_kwargs:
        aggregated = sc.get.aggregate(mock_dataset, by=group_key, func=["sum", "count_nonzero"])
        mean_in_expressed = aggregated.to_df(layer="sum") / aggregated.to_df(layer="count_nonzero")
        max_value = np.max(np.abs(mean_in_expressed))
        dotplot_kwargs.update(vmin=-max_value, vmax=max_value)

    dotplot = sc.pl.dotplot(
        mock_dataset,
        mock_dataset.var_names,
        groupby=group_key,
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
    dataset: ad.AnnData,
    *,
    names: Sequence[str] | None = None,
    embedding_key: str = "normalized_X",
    label_key: str = "leiden",
    excluded_categories: Sequence[str] | None = None,
    title: str | None = None,
):
    """Plot normalized mean embedding activity for each categorical label."""

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
    num_rows, num_columns = values.shape
    image = ax.pcolormesh(
        np.arange(num_columns + 1) - 0.5,
        np.arange(num_rows + 1) - 0.5,
        values,
        cmap="magma",
        edgecolors="black",
        linewidth=0.5,
        shading="flat",
    )
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, num_columns - 0.5)
    ax.set_ylim(num_rows - 0.5, -0.5)
    ax.grid(False)
    ax.set_xticks(np.arange(num_features), [f"m{index}" for index in order], rotation=90)
    ax.set_yticks(np.arange(len(mean_expression.index)), mean_expression.index)
    if title is not None:
        ax.set_title(title)
    fig.colorbar(image, cax=colorbar_ax, orientation="horizontal")
    return fig, values


def clusters_to_categories(
    dataset: ad.AnnData,
    category_de_genes: dict[str, Sequence[str]],
    output_key="marker_expression",
    **dotplot_kwargs,
):
    """Plot clusters to category correspondence, based on category marker gene
    expression.

    Args:
        category_de_genes: mapping from category to marker genes for that category

    """
    score_marker_expression(dataset, category_de_genes, output_key=output_key)
    dotplot = embedding_label_dotplot(
        dataset,
        names=list(category_de_genes.keys()),
        standard_scale=None,
        cmap="bwr",
        embedding_key=output_key,
        title="Cell-type-to-cluster Correspondence",
        add_totals=True,
        **dotplot_kwargs,
    )

    return dotplot.fig
