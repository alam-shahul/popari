"""Plots for differential metagenes and spatial-affinity trends."""

from __future__ import annotations

from collections.abc import Sequence

import anndata as ad
import numpy as np
import scanpy as sc
from matplotlib import colormaps
from matplotlib import pyplot as plt

from popari.analysis.trends import MatrixTrends
from popari.plotting._samples import resolve_samples
from popari.plotting.utils import setup_squarish_axes


def affinity_magnitude_vs_difference(
    adata: ad.AnnData,
    *,
    samples: str | Sequence[str] | None = None,
    spatial_affinity_key: str = "Sigma_x_inv",
    spatial_affinity_bar_key: str = "spatial_affinity_bar",
    figsize=(10, 10),
    n_best: int = 5,
):
    """Plot each sample's affinity against its deviation from the group mean."""

    _, selected_samples = resolve_samples(
        adata,
        samples=samples,
    )
    fig, axes = setup_squarish_axes(len(selected_samples), figsize=figsize)
    affinity_groups = adata.uns["popari_hyperparameters"]["spatial_affinity_groups"]
    all_top_pairs = []
    for ax, sample in zip(axes.flat, selected_samples):
        sample_groups = [group for group, members in affinity_groups.items() if sample in members]
        if len(sample_groups) != 1:
            raise ValueError(
                f"Sample {sample!r} must belong to exactly one spatial-affinity group; " f"found {sample_groups}.",
            )
        group_name = sample_groups[0]
        affinity = adata.uns[spatial_affinity_key][sample]
        difference = affinity - adata.uns[spatial_affinity_bar_key][group_name]
        pairs = [(first, second) for first in range(affinity.shape[0]) for second in range(first + 1)]
        values = np.asarray([(difference[pair], affinity[pair]) for pair in pairs])
        best_indices = np.argsort(np.linalg.norm(values, axis=1))[-n_best:][::-1]

        ax.scatter(values[:, 0], values[:, 1], s=1, color="#D3D3D3")
        for pair_index, color in zip(best_indices, sc.pl.palettes.godsnot_102):
            pair = pairs[pair_index]
            ax.scatter(*values[pair_index], s=20, color=color, label=f"m{pair[0]} x m{pair[1]}")
        ax.set_title(sample)
        ax.set_xlabel("Difference from average affinity")
        ax.set_ylabel("Pairwise affinity")
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        all_top_pairs.append(np.asarray(pairs)[best_indices])

    return fig, all_top_pairs


def normalized_affinity_trends(
    adata: ad.AnnData,
    timepoint_values: Sequence[float],
    *,
    samples: str | Sequence[str] | None = None,
    time_unit: str = "Days",
    normalize: bool = False,
    spatial_affinity_key: str = "Sigma_x_inv",
    n_best: int = 5,
    highlight_metric: str = "pearson",
    figsize: tuple | None = None,
    margin_size: float = 0.25,
):
    """Plot precomputed spatial-affinity trends over a covariate."""

    _, selected_samples = resolve_samples(
        adata,
        samples=samples,
    )
    affinities = np.asarray(
        [adata.uns[spatial_affinity_key][sample] for sample in selected_samples],
    )
    timepoint_values = np.asarray(timepoint_values, dtype=float)
    if len(timepoint_values) != len(selected_samples):
        raise ValueError("timepoint_values must contain one value per selected sample.")
    if normalize:
        affinity_std = affinities.std(axis=0, keepdims=True)
        affinities = np.divide(affinities, affinity_std, out=np.zeros_like(affinities), where=affinity_std != 0)
        if timepoint_values.std():
            timepoint_values = timepoint_values / timepoint_values.std()

    trends = adata.uns["spatial_trends"]
    fig, ax = plt.subplots(dpi=300, figsize=(10, 5) if figsize is None else figsize)
    ax.set_ylim(affinities.min() - margin_size, affinities.max() + margin_size)
    ax.set_xlim(timepoint_values.min() - margin_size, timepoint_values.max() + margin_size)
    ax.set_xticks(timepoint_values)

    top_pairs = [tuple(pair) for pair in trends["top_pairs"]]
    for first in range(affinities.shape[1]):
        for second in range(first + 1):
            if (first, second) not in top_pairs:
                ax.plot(timepoint_values, affinities[:, first, second], color="#D3D3D3", linestyle="--", linewidth=0.5)

    colors = colormaps["rainbow"].resampled(abs(n_best))
    for index, (first, second) in enumerate(top_pairs):
        if highlight_metric == "pearson":
            correlation = trends["pearson_correlations"][(first, second)]
            slope = trends["slopes"][(first, second)]
            slope_label = "" if normalize else f", slope={slope:.2f}"
            label = f"m{first} x m{second}, r={correlation:.2f}{slope_label}"
        else:
            variance = trends["variances"][(first, second)]
            label = f"m{first} x m{second}, variance={variance:.2f}"
        ax.plot(
            timepoint_values,
            affinities[:, first, second],
            color=colors(index),
            linewidth=3,
            label=label,
            zorder=2,
        )

    ax.set_title("Pairwise affinity trends")
    ax.set_xlabel(time_unit)
    ax.set_ylabel("Pairwise affinity")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 0))
    return fig


def matrix_trend_dotplot(
    trends: MatrixTrends,
    labels: Sequence[str],
    *,
    title: str | None = None,
    min_dot_size: float = 5,
    max_dot_size: float = 150,
    cmap: str = "seismic",
    colorbar_label: str = "Pearson correlation",
    figsize: tuple[float, float] = (8, 7),
    dpi: int = 300,
):
    """Plot trends in a symmetric matrix using color and dot size.

    Correlation controls color and absolute regression slope controls size. Only
    the lower triangle, including the diagonal, is displayed.

    """

    correlations = np.asarray(trends.correlations, dtype=float)
    slopes = np.asarray(trends.slopes, dtype=float)
    if correlations.ndim != 2 or correlations.shape[0] != correlations.shape[1]:
        raise ValueError("trend matrices must be square.")
    if slopes.shape != correlations.shape:
        raise ValueError("correlations and slopes must have the same shape.")
    if len(labels) != len(correlations):
        raise ValueError("labels must contain one name per matrix dimension.")
    if min_dot_size < 0 or max_dot_size < min_dot_size:
        raise ValueError("dot sizes must satisfy 0 <= min_dot_size <= max_dot_size.")

    triangle = np.tril(np.ones(correlations.shape, dtype=bool))
    triangle &= np.isfinite(correlations) & np.isfinite(slopes)
    rows, columns = np.where(triangle)
    if not len(rows):
        raise ValueError("the lower triangle contains no finite trends to plot.")

    correlation_limit = np.max(np.abs(correlations[rows, columns]))
    if correlation_limit == 0:
        correlation_limit = 1.0
    slope_limit = np.max(np.abs(slopes[rows, columns]))
    relative_slopes = np.divide(
        np.abs(slopes[rows, columns]),
        slope_limit,
        out=np.zeros(len(rows), dtype=float),
        where=slope_limit != 0,
    )
    sizes = min_dot_size + (max_dot_size - min_dot_size) * relative_slopes

    figure, axis = plt.subplots(figsize=figsize, dpi=dpi)
    points = axis.scatter(
        columns,
        rows,
        c=correlations[rows, columns],
        s=sizes,
        cmap=cmap,
        vmin=-correlation_limit,
        vmax=correlation_limit,
        edgecolors="black",
        linewidths=0.3,
    )
    for row, column in zip(rows, columns):
        axis.add_patch(
            plt.Rectangle(
                (column - 0.5, row - 0.5),
                1,
                1,
                fill=False,
                edgecolor="gray",
                linewidth=1,
            ),
        )

    axis.set_xticks(np.arange(len(labels)), labels, rotation=90)
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.set_xlim(-0.5, len(labels) - 0.5)
    axis.set_ylim(len(labels) - 0.5, -0.5)
    axis.set_title(title)
    axis.set_aspect("equal")
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.colorbar(points, ax=axis, label=colorbar_label)
    return figure
