"""Plots for differential metagenes and spatial-affinity trends."""

from __future__ import annotations

from collections.abc import Sequence

import anndata as ad
import numpy as np
import scanpy as sc
from matplotlib import cm
from matplotlib import pyplot as plt

from popari._datasets import as_datasets
from popari.plotting.utils import setup_squarish_axes


def _group_metagene_images(datasets, gene_subset):
    first_dataset = datasets[0]
    gene_indices = first_dataset.var_names.get_indexer(gene_subset)
    metagene_groups = first_dataset.uns["popari_hyperparameters"]["metagene_groups"]
    images = np.stack(
        [first_dataset.uns["M_bar"][group_name][gene_indices] for group_name in metagene_groups],
        axis=-1,
    )
    return images


def gene_activations(
    data: ad.AnnData | Sequence[ad.AnnData],
    gene_subset: Sequence[str],
):
    """Plot group-level metagene weights for selected genes."""

    datasets = as_datasets(data)
    images = _group_metagene_images(datasets, gene_subset)
    fig, axes = setup_squarish_axes(len(gene_subset), figsize=(10, 10))
    for ax, image, gene in zip(axes.flat, images, gene_subset):
        plotted = ax.imshow(image, interpolation="nearest", aspect=0.1)
        ax.set_title(gene)
        fig.colorbar(plotted, ax=ax, orientation="vertical")
    return fig


def gene_trajectories(
    data: ad.AnnData | Sequence[ad.AnnData],
    gene_subset: Sequence[str],
    covariate_values: Sequence[float],
):
    """Plot total metagene weight across differential groups."""

    datasets = as_datasets(data)
    trends = _group_metagene_images(datasets, gene_subset).sum(axis=1)
    fig, axes = setup_squarish_axes(len(gene_subset), figsize=(10, 10))
    for ax, trend, gene in zip(axes.flat, trends, gene_subset):
        correlation = np.corrcoef(covariate_values, trend)[0, 1]
        ax.plot(covariate_values, trend)
        ax.set_title(f"{gene}, R = {correlation:.2f}")
    return fig


def affinity_magnitude_vs_difference(
    data: ad.AnnData | Sequence[ad.AnnData],
    *,
    spatial_affinity_key: str = "Sigma_x_inv",
    spatial_affinity_bar_key: str = "spatial_affinity_bar",
    figsize=(10, 10),
    n_best: int = 5,
):
    """Plot absolute affinity against deviation from its group mean."""

    datasets = as_datasets(data)
    fig, axes = setup_squarish_axes(len(datasets), figsize=figsize)
    all_top_pairs = []
    for ax, dataset in zip(axes.flat, datasets):
        group_name = next(iter(dataset.uns[spatial_affinity_bar_key]))
        affinity = dataset.uns[spatial_affinity_key][dataset.popari.name]
        difference = affinity - dataset.uns[spatial_affinity_bar_key][group_name]
        pairs = [(first, second) for first in range(affinity.shape[0]) for second in range(first + 1)]
        values = np.asarray([(difference[pair], affinity[pair]) for pair in pairs])
        best_indices = np.argsort(np.linalg.norm(values, axis=1))[-n_best:][::-1]

        ax.scatter(values[:, 0], values[:, 1], s=1, color="#D3D3D3")
        for pair_index, color in zip(best_indices, sc.pl.palettes.godsnot_102):
            pair = pairs[pair_index]
            ax.scatter(*values[pair_index], s=20, color=color, label=f"m{pair[0]} x m{pair[1]}")
        ax.set_title(dataset.popari.name)
        ax.set_xlabel("Difference from average affinity")
        ax.set_ylabel("Pairwise affinity")
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        all_top_pairs.append(np.asarray(pairs)[best_indices])

    return fig, all_top_pairs


def normalized_affinity_trends(
    data: ad.AnnData | Sequence[ad.AnnData],
    timepoint_values: Sequence[float],
    *,
    time_unit: str = "Days",
    normalize: bool = False,
    spatial_affinity_key: str = "Sigma_x_inv",
    n_best: int = 5,
    highlight_metric: str = "pearson",
    figsize: tuple | None = None,
    margin_size: float = 0.25,
):
    """Plot precomputed spatial-affinity trends over a covariate."""

    datasets = as_datasets(data)
    affinities = np.asarray(
        [dataset.uns[spatial_affinity_key][dataset.popari.name] for dataset in datasets],
    )
    timepoint_values = np.asarray(timepoint_values, dtype=float)
    if normalize:
        affinity_std = affinities.std(axis=0, keepdims=True)
        affinities = np.divide(affinities, affinity_std, out=np.zeros_like(affinities), where=affinity_std != 0)
        if timepoint_values.std():
            timepoint_values = timepoint_values / timepoint_values.std()

    trends = datasets[0].uns["spatial_trends"]
    fig, ax = plt.subplots(dpi=300, figsize=(10, 5) if figsize is None else figsize)
    ax.set_ylim(affinities.min() - margin_size, affinities.max() + margin_size)
    ax.set_xlim(timepoint_values.min() - margin_size, timepoint_values.max() + margin_size)
    ax.set_xticks(timepoint_values)

    top_pairs = [tuple(pair) for pair in trends["top_pairs"]]
    for first in range(affinities.shape[1]):
        for second in range(first + 1):
            if (first, second) not in top_pairs:
                ax.plot(timepoint_values, affinities[:, first, second], color="#D3D3D3", linestyle="--", linewidth=0.5)

    colors = cm.get_cmap("rainbow", abs(n_best))
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
