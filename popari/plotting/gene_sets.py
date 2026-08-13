"""Plots for gene-set analysis results."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from textwrap import fill
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import colormaps, gridspec
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from upsetplot import from_contents, plot

from popari.analysis.gene_sets import compute_gene_set_auroc, order_columns_by_best_row


def enrichment_barplot(
    results: pd.DataFrame,
    *,
    column: str = "Adjusted P-value",
    group: str | None = "Gene_set",
    title: str = "",
    cutoff: float = 0.05,
    top_term: int = 10,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    cmap: str = "magma",
    line_length: int = 30,
) -> Figure:
    """Plot significant Enrichr terms as faceted horizontal bars.

    Args:
        results: Result table returned by :func:`popari.tl.run_enrichr`.
        column: Adjusted p-value column used to rank terms.
        group: Optional result column used to create panels.
        title: Plot title.
        cutoff: Maximum value of ``column`` to display.
        top_term: Maximum number of terms to display per group.
        ax: Existing axes for an ungrouped or single-group plot.
        figsize: Figure size. By default, scale it to the number of panels.
        cmap: Colormap encoding the overlap ratio.
        line_length: Maximum wrapped GO-term label width.

    Returns:
        Figure containing the enrichment bar plot.

    Raises:
        ValueError: If no terms pass ``cutoff``, required columns are absent,
            or one axes is supplied for multiple groups.

    """

    if results.empty:
        raise ValueError("results must contain at least one enrichment term.")
    required_columns = {"Term", column}
    if group is not None:
        required_columns.add(group)
    missing_columns = required_columns - set(results.columns)
    if missing_columns:
        raise ValueError(f"results is missing required columns: {sorted(missing_columns)}.")

    significant = results.loc[results[column] <= cutoff].copy()
    if significant.empty:
        raise ValueError(f"No enrichment terms pass the cutoff of {cutoff}.")

    significant["negative_log_pvalue"] = -np.log(significant[column].clip(lower=np.finfo(float).tiny))
    if "Overlap Ratio" in significant:
        color_values = pd.to_numeric(significant["Overlap Ratio"])
        colorbar_label = "Gene overlap ratio"
        normalization = Normalize(vmin=0, vmax=1)
    elif "Overlap" in significant:
        overlap = significant["Overlap"].str.split("/", expand=True).astype(float)
        color_values = overlap[0] / overlap[1]
        colorbar_label = "Gene overlap ratio"
        normalization = Normalize(vmin=0, vmax=1)
    elif "Odds Ratio" in significant:
        color_values = pd.to_numeric(significant["Odds Ratio"])
        colorbar_label = "Odds ratio"
        normalization = Normalize(vmin=color_values.min(), vmax=color_values.max())
    else:
        color_values = pd.Series(1.0, index=significant.index)
        colorbar_label = None
        normalization = Normalize(vmin=0, vmax=1)
    significant["_color_value"] = color_values

    grouped_results = (
        [("Enrichment", significant)] if group is None else list(significant.groupby(group, sort=False, observed=True))
    )
    if ax is not None and len(grouped_results) != 1:
        raise ValueError("ax can only be supplied for an ungrouped or single-group plot.")

    if ax is None:
        if figsize is None:
            figsize = (4 * len(grouped_results), max(3, 0.45 * top_term + 1.5))
        figure, axes = plt.subplots(
            1,
            len(grouped_results),
            figsize=figsize,
            squeeze=False,
            layout="constrained",
        )
        axes = axes.ravel()
    else:
        figure = ax.figure
        axes = np.asarray([ax])

    mapper = ScalarMappable(norm=normalization, cmap=colormaps[cmap])
    significance_threshold = -np.log(cutoff)

    for axis, (group_name, group_results) in zip(axes, grouped_results):
        plotted = group_results.nsmallest(top_term, column).iloc[::-1]
        axis.barh(
            np.arange(len(plotted)),
            plotted["negative_log_pvalue"],
            color=mapper.to_rgba(plotted["_color_value"]),
        )
        axis.axvline(significance_threshold, color="red", linestyle="--", linewidth=1)
        axis.set_yticks(
            np.arange(len(plotted)),
            [fill(term, line_length) for term in plotted["Term"]],
        )
        axis.set_title(str(group_name))
        axis.set_xlabel(r"$-\log(\mathrm{adjusted}\ p)$")
        axis.set_ylabel("")
        axis.yaxis.grid(False)

    if colorbar_label is not None:
        figure.colorbar(
            mapper,
            ax=axes.tolist(),
            label=colorbar_label,
            fraction=0.04,
            pad=0.04,
        )
    if title:
        figure.suptitle(title)
    return figure


def enrichment_dotplot(
    results: pd.DataFrame,
    *,
    column: str = "Adjusted P-value",
    x: str | None = "Gene_set",
    y: str = "Term",
    title: str = "",
    cutoff: float = 0.05,
    top_term: int = 5,
    size: float = 2,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (3, 5),
    cmap: str = "viridis_r",
    xticklabels_rot: float | None = 45,
    yticklabels_rot: float | None = None,
    marker: str = "o",
    show_ring: bool = True,
) -> Figure:
    """Plot an Enrichr result table as a dot plot.

    Args:
        results: Result table returned by :func:`popari.tl.run_enrichr`.
        column: Enrichment statistic used to rank and size terms.
        x: Result column displayed along the x-axis.
        y: Result column displayed along the y-axis.
        title: Plot title.
        cutoff: Maximum value of ``column`` to display.
        top_term: Maximum number of terms to display.
        size: Dot-size scaling factor.
        ax: Existing axes. By default, create new axes.
        figsize: Figure size used when creating axes.
        cmap: Dot colormap.
        xticklabels_rot: Rotation of x-axis labels.
        yticklabels_rot: Rotation of y-axis labels.
        marker: Dot marker.
        show_ring: Whether to draw rings around dots.

    Returns:
        Figure containing the enrichment dot plot.

    Raises:
        ValueError: If ``results`` is empty.

    """

    if results.empty:
        raise ValueError("results must contain at least one enrichment term.")

    from gseapy import dotplot

    ax = dotplot(
        results,
        column=column,
        x=x,
        y=y,
        title=title,
        cutoff=cutoff,
        top_term=top_term,
        size=size,
        ax=ax,
        figsize=figsize,
        cmap=cmap,
        xticklabels_rot=xticklabels_rot,
        yticklabels_rot=yticklabels_rot,
        marker=marker,
        show_ring=show_ring,
    )
    return ax.figure


def gene_set_upset(
    gene_sets: Mapping[str, Iterable[str]],
    *,
    background: Iterable[str] | None = None,
    fig: Figure | None = None,
    figsize=None,
    dpi: int = 100,
    **upset_kwargs,
) -> Figure:
    """Plot intersections among named gene sets.

    Args:
        gene_sets: Mapping from display labels to gene iterables.
        background: Optional measured-gene universe used to filter every set.
        fig: Existing figure on which to draw. By default, create a new figure.
        figsize: Size used when creating a figure.
        dpi: Resolution used when creating a figure.
        **upset_kwargs: Additional arguments forwarded to
            :func:`upsetplot.plot`.

    Returns:
        Figure containing the UpSet plot.

    Raises:
        ValueError: If fewer than two sets are provided or no genes remain
            after background filtering.

    """

    if len(gene_sets) < 2:
        raise ValueError("gene_sets must contain at least two named gene sets.")

    universe = None if background is None else set(background)
    contents = {name: set(genes) if universe is None else set(genes) & universe for name, genes in gene_sets.items()}
    if not set().union(*contents.values()):
        raise ValueError("gene_sets must contain at least one gene after background filtering.")

    if fig is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            module=r"upsetplot\..*",
        )
        plot(from_contents(contents), fig=fig, **upset_kwargs)
    return fig


def plot_metagene_gene_set_aurocs(
    dataset,
    gene_sets,
    metagene_key: str = "M",
):
    """Plot metagene-by-gene-set AUROC scores."""

    num_gene_sets = len(gene_sets.columns)

    aurocs, _ = compute_gene_set_auroc(
        dataset,
        gene_sets,
        metagene_key=metagene_key,
    )
    num_metagenes = aurocs.shape[1]
    sorted_indices = order_columns_by_best_row(aurocs)
    aurocs = aurocs[:, sorted_indices]

    fig = plt.figure(figsize=(num_metagenes * 0.3, num_gene_sets * 0.5))
    grid_spec = gridspec.GridSpec(2, 1, height_ratios=[1, 0.05])
    ax = plt.subplot(grid_spec[0])
    cax = plt.subplot(grid_spec[1])

    row_edges = np.arange(num_gene_sets + 1) - 0.5
    column_edges = np.arange(num_metagenes + 1) - 0.5
    image = ax.pcolormesh(
        column_edges,
        row_edges,
        aurocs,
        vmin=1 - aurocs.max(),
        vmax=aurocs.max(),
        cmap="bwr",
        edgecolors="black",
        linewidth=0.5,
        shading="flat",
    )
    ax.grid(False)
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, num_metagenes - 0.5)
    ax.set_ylim(num_gene_sets - 0.5, -0.5)
    ax.set_yticks(np.arange(num_gene_sets), gene_sets.columns.values)
    ax.set_xticks(np.arange(num_metagenes), [f"m{k}" for k in sorted_indices], rotation=90)

    for label in ax.get_yticklabels():
        label.set_verticalalignment("top")
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("left")

    fig.colorbar(image, cax=cax, orientation="horizontal")
    return fig
