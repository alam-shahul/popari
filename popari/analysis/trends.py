"""Differential metagene and spatial-affinity trend analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import anndata as ad
import numpy as np
from scipy.stats import pearsonr

from popari._datasets import as_datasets
from popari.util import spatially_smooth_feature


def find_differential_genes(
    data: ad.AnnData | Sequence[ad.AnnData],
    top_gene_limit: int = 1,
):
    """Return genes with the largest replicate-to-group metagene deviations."""

    datasets = as_datasets(data)
    genes_of_interest = set()
    for dataset in datasets:
        if "M_bar" not in dataset.uns:
            raise ValueError("The datasets were not trained in differential metagene mode.")

        metagene_tags = dataset.uns["popari_hyperparameters"]["metagene_tags"]
        for group_name in metagene_tags[dataset.popari.name]:
            difference = dataset.uns["M"][dataset.popari.name] - dataset.uns["M_bar"][group_name]
            top_indices = np.argpartition(np.abs(difference), -top_gene_limit, axis=0)[-top_gene_limit:]
            genes_of_interest.update(dataset.var_names[top_indices.ravel()])

    return genes_of_interest


def normalized_affinity_trends(
    data: ad.AnnData | Sequence[ad.AnnData],
    timepoint_values: Sequence[float],
    *,
    normalize: bool = False,
    spatial_affinity_key: str = "Sigma_x_inv",
    n_best: int = 5,
    highlight_metric: str = "pearson",
):
    """Compute temporal summaries for every lower-triangular affinity entry."""

    datasets = as_datasets(data)
    timepoint_values = np.asarray(timepoint_values, dtype=float)
    if len(timepoint_values) != len(datasets):
        raise ValueError("timepoint_values must contain one value per dataset.")

    affinities = np.asarray(
        [dataset.uns[spatial_affinity_key][dataset.popari.name] for dataset in datasets],
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
    for dataset in datasets:
        dataset.uns["spatial_trends"] = spatial_trends

    return top_pairs, pearson_correlations, variances


def propagate_labels(
    hierarchy: Mapping[int, Sequence[ad.AnnData]],
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
        datasets = hierarchy[level]
        finer_datasets = hierarchy[level - 1]
        for dataset, finer_dataset in zip(datasets, finer_datasets):
            bin_assignments = dataset.obsm[f"bin_assignments_{dataset.popari.name}"]
            assignment_index = np.asarray(bin_assignments.argmax(axis=0)).squeeze()
            propagated_labels = dataset.obs[label_key].values[assignment_index]
            if smooth:
                propagated_labels = spatially_smooth_feature(
                    propagated_labels,
                    finer_dataset.obsm["adjacency_list"],
                    max_smoothing_rounds=200,
                    smoothing_threshold=0.3,
                )
            finer_dataset.obs[label_key] = propagated_labels
            finer_dataset.obs[label_key] = finer_dataset.obs[label_key].astype("category")
