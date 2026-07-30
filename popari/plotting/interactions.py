"""Plots for spatial-affinity and edge-interaction analyses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional

import anndata as ad
import numpy as np
import squidpy as sq
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from popari.analysis.interactions import EdgeInteractions
from popari.plotting._samples import resolve_samples
from popari.plotting.heatmaps import matrix_heatmap
from popari.plotting.utils import setup_squarish_axes


def _edge_score_values(
    interactions: EdgeInteractions,
    score: str,
    metagene_pair: tuple[int, int] | None,
) -> np.ndarray:
    if score == "total":
        if metagene_pair is not None:
            raise ValueError("metagene_pair must be omitted when score='total'.")
        return interactions.scores
    if score in {"affinity", "cooccurrence"}:
        if metagene_pair is None:
            raise ValueError(f"metagene_pair is required when score={score!r}.")
        return interactions.metagene_pair_scores(*metagene_pair, mode=score)
    raise ValueError("score must be 'total', 'affinity', or 'cooccurrence'.")


def edge_interactions(
    dataset: ad.AnnData,
    interactions: EdgeInteractions,
    *,
    score: str = "total",
    metagene_pair: tuple[int, int] | None = None,
    category_key: str | None = None,
    category_pair: tuple | None = None,
    directed: bool = False,
    color: str | None = "cell_type",
    neighbor_key: str = "adjacency_matrix",
    ax: Axes | None = None,
    edge_cmap="bwr",
    edge_vmin: float | None = None,
    edge_vmax: float | None = None,
    center_zero: bool = True,
    colorbar: bool = True,
    edges_width: float = 1,
    size: float | None = None,
    title: str | None = None,
    **spatial_kwargs,
):
    """Plot total or metagene-pair scores on selected spatial edges."""

    if not interactions.obs_names.equals(dataset.obs_names):
        missing_observations = interactions.obs_names.difference(dataset.obs_names)
        if len(missing_observations):
            raise ValueError(
                "interactions contains observations absent from dataset: " f"{missing_observations.tolist()}.",
            )
        dataset = dataset[interactions.obs_names]
    if not interactions.obs_names.equals(dataset.obs_names):
        raise ValueError("Could not align interactions to dataset observations.")
    edge_values = _edge_score_values(interactions, score, metagene_pair)

    edge_mask = np.ones(len(interactions.source), dtype=bool)
    if category_pair is not None:
        if category_key is None:
            raise ValueError("category_key is required when category_pair is provided.")
        edge_mask = interactions.category_mask(
            dataset.obs[category_key],
            category_pair,
            directed=directed,
        )
    edges = interactions.edges[edge_mask]
    edge_values = edge_values[edge_mask]
    if not len(edges):
        raise ValueError("No graph edges match the requested interaction selection.")

    if ax is None:
        fig, ax = plt.subplots(dpi=spatial_kwargs.pop("dpi", 100))
    else:
        fig = ax.get_figure()

    if center_zero and edge_vmin is None and edge_vmax is None:
        edge_limit = np.max(np.abs(edge_values))
        if edge_limit == 0:
            edge_limit = 1.0
        edge_vmin, edge_vmax = -edge_limit, edge_limit
    else:
        if edge_vmin is None:
            edge_vmin = float(np.min(edge_values))
        if edge_vmax is None:
            edge_vmax = float(np.max(edge_values))
    edge_cmap = plt.get_cmap(edge_cmap) if isinstance(edge_cmap, str) else edge_cmap
    edges_kwargs = {
        "edgelist": edges,
        "edge_cmap": edge_cmap,
        **spatial_kwargs.pop("edges_kwargs", {}),
    }
    if edge_vmin is not None:
        edges_kwargs["edge_vmin"] = edge_vmin
    if edge_vmax is not None:
        edges_kwargs["edge_vmax"] = edge_vmax
    if size is None:
        size = 5000 / dataset.n_obs
    if title is None:
        title = score
        if metagene_pair is not None:
            title = f"m{metagene_pair[0]} -> m{metagene_pair[1]} ({score})"
        if category_pair is not None:
            title = f"{category_pair[0]} - {category_pair[1]} ({title})"

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
        fig = axes.flat[0].get_figure()
    elif isinstance(axes, list):
        fig = axes[0].get_figure()
    elif axes is not None:
        fig = axes.get_figure()
    if colorbar:
        fig.colorbar(
            ScalarMappable(
                norm=Normalize(vmin=edge_vmin, vmax=edge_vmax),
                cmap=edge_cmap,
            ),
            ax=ax,
            label="Edge accordance score",
            fraction=0.046,
            pad=0.04,
        )
    return fig


def edge_interactions_panel(
    adata: ad.AnnData,
    interactions: Mapping[str, EdgeInteractions],
    *,
    samples: str | Sequence[str] | None = None,
    sample_key: str | None = None,
    score: str = "total",
    metagene_pair: tuple[int, int] | None = None,
    category_key: str | None = None,
    category_pair: tuple | None = None,
    directed: bool = False,
    center_zero: bool = True,
    edge_cmap="bwr",
    colorbar_label: str = "Edge accordance score",
    figsize=None,
    dpi: int = 300,
    **plot_kwargs,
):
    """Plot one edge-score selection across selected samples with a shared
    scale."""

    sample_axis, selected_samples = resolve_samples(
        adata,
        samples=samples,
        sample_key=sample_key,
        require_graph=True,
    )
    missing_results = [sample for sample in selected_samples if sample not in interactions]
    if missing_results:
        raise KeyError(f"Missing edge interactions for samples: {missing_results}.")

    selected_values = []
    for sample in selected_samples:
        dataset = adata[sample_axis.indices(sample)]
        result = interactions[sample]
        if not result.obs_names.equals(dataset.obs_names):
            raise ValueError(f"Edge interactions for {sample!r} are not aligned to that sample.")
        values = _edge_score_values(result, score, metagene_pair)
        if category_pair is not None:
            if category_key is None:
                raise ValueError("category_key is required when category_pair is provided.")
            mask = result.category_mask(
                dataset.obs[category_key],
                category_pair,
                directed=directed,
            )
            values = values[mask]
        selected_values.append(values)

    nonempty_values = [values for values in selected_values if len(values)]
    if not nonempty_values:
        raise ValueError("No graph edges match the requested interaction selection.")
    if center_zero:
        limit = max(np.abs(values).max() for values in nonempty_values)
        limit = 1.0 if limit == 0 else limit
        edge_vmin, edge_vmax = -limit, limit
    else:
        edge_vmin = min(values.min() for values in nonempty_values)
        edge_vmax = max(values.max() for values in nonempty_values)

    fig, axes = setup_squarish_axes(
        len(selected_samples),
        dpi=dpi,
        figsize=figsize,
        constrained_layout=False,
    )
    for sample, values, ax in zip(selected_samples, selected_values, axes.flat):
        result = interactions[sample]
        if not len(values):
            ax.set_visible(False)
            continue
        edge_interactions(
            adata,
            result,
            score=score,
            metagene_pair=metagene_pair,
            category_key=category_key,
            category_pair=category_pair,
            directed=directed,
            center_zero=center_zero,
            edge_cmap=edge_cmap,
            edge_vmin=edge_vmin,
            edge_vmax=edge_vmax,
            colorbar=False,
            title=sample,
            ax=ax,
            **plot_kwargs,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")

    for ax in axes.flat[len(selected_samples) :]:
        ax.set_visible(False)
    visible_axes = [ax for ax in axes.flat if ax.get_visible()]
    fig.colorbar(
        ScalarMappable(
            norm=Normalize(vmin=edge_vmin, vmax=edge_vmax),
            cmap=edge_cmap,
        ),
        ax=visible_axes,
        label=colorbar_label,
        fraction=0.02,
        pad=0.02,
    )
    return fig


def affinity_difference(
    dataset: ad.AnnData,
    comparison: str,
    reference: str,
    *,
    spatial_affinity_key: str = "Sigma_x_inv",
    ax: Axes | None = None,
    cmap: str = "bwr_r",
    mask_upper: bool = True,
    label_values: bool = False,
    label_font_size: float = 8,
    metagene_labels: Sequence[str] | None = None,
    **heatmap_kwargs,
):
    """Plot one named spatial-affinity matrix minus another."""

    difference = dataset.popari.affinity_difference(
        comparison,
        reference,
        spatial_affinity_key=spatial_affinity_key,
    )
    mask = None
    if mask_upper:
        mask = np.zeros_like(difference, dtype=bool)
        mask[np.triu_indices_from(mask, k=1)] = True
    if metagene_labels is None:
        metagene_labels = [f"m{index}" for index in range(difference.shape[0])]

    return matrix_heatmap(
        difference,
        ax=ax,
        title=f"{comparison} - {reference}",
        cmap=cmap,
        center_zero=True,
        mask=mask,
        label_values=label_values,
        label_font_size=label_font_size,
        xticklabels=metagene_labels,
        yticklabels=metagene_labels,
        **heatmap_kwargs,
    )
