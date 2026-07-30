"""Diagnostic and biological-interpretation plots."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import anndata as ad
import numpy as np
from matplotlib import colormaps
from matplotlib import pyplot as plt
from matplotlib.transforms import Affine2D
from mpl_toolkits.axisartist import floating_axes
from mpl_toolkits.axisartist.grid_finder import DictFormatter, FixedLocator
from scipy.sparse import issparse
from scipy.stats import wilcoxon, zscore

from popari._datasets import as_datasets
from popari.plotting.heatmaps import multireplicate_heatmap


def pretty_spatial_affinities(
    datasets,
    metagene_key: str = "M",
    metagene_label_key: str = "metagene_labels",
    spatial_affinity_key: str = "Sigma_x_inv",
):
    """Rotate and plot spatial affinities matrix.

    Args:
        datasets:

    """

    datasets = as_datasets(datasets)
    transform_skew = Affine2D().skew_deg(15, 15)
    transform_rotate = Affine2D().rotate_deg(-45)
    transform = transform_skew + transform_rotate

    height = len(datasets) // 2 + len(datasets) % 2
    width = 2
    fig = plt.figure(dpi=1200, figsize=(width, height))

    first_dataset = datasets[0]
    _, K = first_dataset.uns[metagene_key][first_dataset.popari.name].shape

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
        ax.set_title(dataset.popari.name, fontsize=2, y=0.5)
        axes.append(aux_ax)

    mask = np.ones((K, K), dtype=bool)
    mask[np.triu_indices_from(mask)] = 0

    multireplicate_heatmap(
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


def sparsity(dataset: ad.AnnData):
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


def cell_type_to_metagene(
    dataset,
    cell_type_de_genes: dict,
    rank_mode: str = "metagene",
    plot_type: str = "box",
    normalize: bool = False,
    figsize: tuple | None = None,
    cell_types: list | None = None,
    metagene_index: Sequence | None = None,
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

    metagenes = dataset.uns[metagene_key][dataset.popari.name]

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


def cell_type_to_metagene_difference(
    dataset,
    cell_type_de_genes: dict,
    first_metagene: int,
    second_metagene: int,
    rank_mode: str = "metagene",
    plot_type: str = "box",
    normalize: bool = False,
    figsize: tuple | None = None,
    cell_types: list | None = None,
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

    metagenes = dataset.uns[metagene_key][dataset.popari.name]

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
