"""Differential metagene and spatial-affinity trend analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import anndata as ad
import numpy as np
from scipy.stats import pearsonr

from popari._sample_axis import SampleAxis
from popari.analysis.clustering import smooth_labels


def normalized_affinity_trends(
    dataset: ad.AnnData,
    timepoint_values: Sequence[float],
    *,
    normalize: bool = False,
    spatial_affinity_key: str = "Sigma_x_inv",
    n_best: int = 5,
    highlight_metric: str = "pearson",
):
    """Compute temporal summaries for every lower-triangular affinity entry."""

    sample_axis = SampleAxis.from_anndata(
        dataset,
        sample_key=dataset.popari.sample_key,
    )
    timepoint_values = np.asarray(timepoint_values, dtype=float)
    if len(timepoint_values) != len(sample_axis):
        raise ValueError("timepoint_values must contain one value per sample.")

    affinities = np.asarray(
        [dataset.uns[spatial_affinity_key][sample] for sample in sample_axis.names],
    )
    if normalize:
        affinity_std = affinities.std(axis=0, keepdims=True)
        affinities = np.divide(affinities, affinity_std, out=np.zeros_like(affinities), where=affinity_std != 0)
        timepoint_std = timepoint_values.std()
        if timepoint_std:
            timepoint_values = timepoint_values / timepoint_std

    timepoint_std = timepoint_values.std()
    affinity_std = affinities.std(axis=0)
    pearson_correlations = {}
    variances = {}
    slopes = {}
    for first in range(affinities.shape[1]):
        for second in range(first + 1):
            values = affinities[:, first, second]
            correlation, _ = pearsonr(values, timepoint_values)
            pair = (first, second)
            pearson_correlations[pair] = correlation
            variances[pair] = np.var(values)
            slopes[pair] = correlation * affinity_std[first, second] / timepoint_std if timepoint_std else np.nan

    metrics = pearson_correlations if highlight_metric == "pearson" else variances
    if highlight_metric not in {"pearson", "variance"}:
        raise ValueError("highlight_metric must be 'pearson' or 'variance'.")
    sorted_pairs = sorted(metrics, key=metrics.get)
    top_pairs = sorted_pairs[-n_best:][::-1] if n_best > 0 else sorted_pairs[:-n_best]

    spatial_trends = {
        "top_pairs": top_pairs,
        "pearson_correlations": pearson_correlations,
        "variances": variances,
        "slopes": slopes,
    }
    dataset.uns["spatial_trends"] = spatial_trends

    return top_pairs, pearson_correlations, variances


def propagate_labels(
    hierarchy: Mapping[int, ad.AnnData],
    label_key: str,
    *,
    starting_level: int | None = None,
    smooth: bool = False,
) -> None:
    """Propagate coarse labels to finer hierarchy levels through bin
    assignments."""

    if starting_level is None:
        starting_level = max(hierarchy)

    for level in range(starting_level, 0, -1):
        dataset = hierarchy[level]
        finer_dataset = hierarchy[level - 1]
        bin_assignments = dataset.obsm["bin_assignments"]
        assignment_index = np.asarray(bin_assignments.argmax(axis=0)).squeeze()
        propagated_labels = dataset.obs[label_key].values[assignment_index]
        finer_dataset.obs[label_key] = propagated_labels
        finer_dataset.obs[label_key] = finer_dataset.obs[label_key].astype("category")
        if smooth:
            smooth_labels(
                finer_dataset,
                label_key=label_key,
                output_key=label_key,
                max_smoothing_rounds=200,
                smoothing_threshold=0.3,
            )
