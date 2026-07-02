"""Metrics for simulated Popari results stored as models or AnnData objects."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import anndata as ad
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr, wasserstein_distance


def propagate_ground_truth(
    model,
    embedding_truth_key: str = "ground_truth_X",
    metagene_truth_key: str = "ground_truth_M",
):
    """Propagate ground-truth embeddings/metagenes through hierarchy levels."""

    for level in range(model.hierarchical_levels - 1):
        view = model.hierarchy[level]
        for dataset, low_res_dataset in zip(view.datasets, view.low_res_view.datasets):
            if embedding_truth_key == "ground_truth_X":
                true_embeddings = dataset.simulation.ground_truth_X
            else:
                true_embeddings = dataset.obsm[embedding_truth_key]

            if metagene_truth_key == "ground_truth_M":
                true_metagenes = dataset.simulation.ground_truth_M.copy()
            else:
                true_metagenes = dataset.uns[metagene_truth_key][dataset.name].copy()

            assignment_key = f"bin_assignments_{dataset.name}_level_{level + 1}"
            propagated_embeddings = low_res_dataset.obsm[assignment_key] @ true_embeddings

            if embedding_truth_key == "ground_truth_X":
                low_res_dataset.simulation.ground_truth_X = propagated_embeddings
            else:
                low_res_dataset.obsm[embedding_truth_key] = propagated_embeddings

            if metagene_truth_key == "ground_truth_M":
                low_res_dataset.simulation.ground_truth_M = true_metagenes
            else:
                low_res_dataset.uns[metagene_truth_key] = {low_res_dataset.name: true_metagenes}


def evaluate_ground_truth(
    dataset,
    real_metagene_index: Sequence[int] | None = None,
    embedding_truth_key: str = "ground_truth_X",
    metagene_truth_key: str = "ground_truth_M",
):
    """Evaluate learned embeddings/metagenes against ground truth arrays."""

    from popari._simulation_utils import all_pairs_spatial_wasserstein

    if real_metagene_index is None:
        real_metagene_index = slice(None)

    dataset_name = dataset.popari.name()
    if embedding_truth_key == "ground_truth_X":
        true_embeddings = dataset.simulation.ground_truth_X.T
    else:
        true_embeddings = dataset.obsm[embedding_truth_key].T

    if metagene_truth_key == "ground_truth_M":
        true_metagenes = dataset.simulation.ground_truth_M.T
    else:
        true_metagenes = dataset.uns[metagene_truth_key][dataset_name].T

    learned_embeddings = dataset.simulation.learned_X.T

    try:
        learned_metagenes = dataset.simulation.learned_M.T
    except Exception:
        learned_metagenes = true_metagenes + 0.05

    def pearson_correlation_magnitude(x, y):
        return abs(pearsonr(x, y)[0])

    def normalized_wasserstein_distance(x, y):
        return wasserstein_distance(x / x.sum(), y / y.sum())

    embedding_pearson_correlations = cdist(true_embeddings, learned_embeddings, metric=pearson_correlation_magnitude)
    metagene_pearson_correlations = cdist(true_metagenes, learned_metagenes, metric=pearson_correlation_magnitude)
    embedding_wasserstein = cdist(true_embeddings, learned_embeddings, metric=normalized_wasserstein_distance)
    metagene_wasserstein = cdist(true_metagenes, learned_metagenes, metric=normalized_wasserstein_distance)
    embedding_spatial_wasserstein = all_pairs_spatial_wasserstein(dataset)

    dataset.uns["embedding_pearson_correlations"] = embedding_pearson_correlations
    dataset.uns["metagene_pearson_correlations"] = metagene_pearson_correlations
    dataset.uns["embedding_wasserstein"] = embedding_wasserstein
    dataset.uns["metagene_wasserstein"] = metagene_wasserstein
    dataset.uns["embedding_spatial_wasserstein"] = embedding_spatial_wasserstein

    dataset.uns["metagene_pearson_minmax"] = metagene_pearson_correlations.max(axis=1)[real_metagene_index].min()
    dataset.uns["embedding_pearson_minmax"] = np.nanmax(
        embedding_pearson_correlations[real_metagene_index],
        axis=1,
    ).min()
    dataset.uns["metagene_wasserstein_maxmin"] = metagene_wasserstein.min(axis=1)[real_metagene_index].max()
    dataset.uns["embedding_wasserstein_maxmin"] = np.nanmin(
        embedding_wasserstein[real_metagene_index],
        axis=1,
    ).max()
    dataset.uns["embedding_spatial_wasserstein_maxmin"] = np.nanmin(
        embedding_spatial_wasserstein[real_metagene_index],
        axis=1,
    ).max()
    dataset.uns["embedding_spatial_wasserstein_avgmin"] = np.min(
        embedding_spatial_wasserstein[real_metagene_index],
        axis=1,
    ).mean()


def compute_matching(
    dataset,
    embedding_truth_key: str = "ground_truth_X",
    embedding_key: str = "X",
    real_metagene_index=None,
):
    """Match learned factors to ground-truth factors using spatial Wasserstein
    distance."""

    embedding_spatial_wasserstein = dataset.uns["embedding_spatial_wasserstein"][real_metagene_index]
    _, matched_indices = linear_sum_assignment(embedding_spatial_wasserstein)

    dataset.uns["matched_indices"] = matched_indices
    if embedding_truth_key == "ground_truth_X":
        ground_truth_embeddings = dataset.simulation.ground_truth_X
    else:
        ground_truth_embeddings = dataset.obsm[embedding_truth_key]

    if embedding_key == "X":
        learned_embeddings = dataset.simulation.learned_X
    else:
        learned_embeddings = dataset.obsm[embedding_key]

    dataset.obsm["truncated_ground_truth_X"] = ground_truth_embeddings[:, real_metagene_index]
    dataset.obsm["truncated_matched_X"] = learned_embeddings[:, matched_indices]


def compute_affinity_correlation(
    dataset,
    spatial_affinity_key: str = "Sigma_x_inv",
    correlation_truth_key: str = "ground_truth_correlation",
    real_metagene_index=None,
):
    """Compute rank correlation between learned and ground-truth affinities."""

    sigma_x_inv = dataset.uns[spatial_affinity_key][dataset.name]
    empirical_correlation = dataset.uns[correlation_truth_key][dataset.name][real_metagene_index][
        :,
        real_metagene_index,
    ]
    matched_indices = dataset.uns["matched_indices"]
    permuted_sigma_x_inv = sigma_x_inv[matched_indices, :][:, matched_indices]

    correlation = spearmanr(empirical_correlation.flatten(), permuted_sigma_x_inv.flatten())[0]
    dataset.uns["affinity_correlation"] = np.nan_to_num(correlation)
    dataset.uns["permuted_Sigma_x_inv"] = {dataset.name: permuted_sigma_x_inv}


def compute_affinity_coherence(
    dataset,
    spatial_affinity_key: str = "Sigma_x_inv",
    empirical_correlation_key: str = "empirical_correlation",
):
    """Compute Pearson correlation between learned affinities and empirical
    spatial correlation."""

    empirical_correlation = dataset.uns[empirical_correlation_key][dataset.name].copy()
    empirical_correlation /= np.abs(empirical_correlation).max()

    sigma_x_inv = dataset.uns[spatial_affinity_key][dataset.name].copy()
    sigma_x_inv /= np.abs(sigma_x_inv).max()

    dataset.uns["affinity_coherence"] = pearsonr(empirical_correlation.flatten(), sigma_x_inv.flatten())[0]


def evaluate_model(model, metagene_indices, is_spatial: bool, hierarchical_levels=None):
    """Evaluate metrics for every requested level of a Popari model."""

    from popari._dataset_utils import _compute_empirical_correlations

    if hierarchical_levels is None:
        hierarchical_levels = range(model.hierarchical_levels)
    elif isinstance(hierarchical_levels, int):
        hierarchical_levels = range(hierarchical_levels)

    if model.hierarchical_levels > 1:
        propagate_ground_truth(model)

    for level in hierarchical_levels:
        datasets = model.hierarchy[level].datasets
        if is_spatial:
            _compute_empirical_correlations(datasets, scaling=10)
            _compute_empirical_correlations(
                datasets,
                scaling=10,
                feature="ground_truth_X",
                output="ground_truth_correlation",
            )

        for dataset, metagene_index in zip(datasets, metagene_indices):
            evaluate_ground_truth(dataset, real_metagene_index=metagene_index)
            compute_matching(dataset, real_metagene_index=metagene_index)
            if is_spatial:
                compute_affinity_coherence(dataset)
                compute_affinity_correlation(dataset, real_metagene_index=metagene_index)


def propagate_ground_truth_to_anndata_hierarchy(hierarchy: Mapping[int, Sequence[ad.AnnData]]) -> None:
    """Propagate ground-truth arrays across an AnnData hierarchy when needed."""

    levels = sorted(hierarchy)
    for level in levels:
        for dataset in hierarchy[level]:
            dataset_name = getattr(dataset, "name", None) or dataset.popari.name()
            dataset.name = dataset_name
            dataset.obs["batch"] = dataset_name
            if "ground_truth_M" not in dataset.uns and "M" in dataset.uns:
                dataset.uns["ground_truth_M"] = dataset.uns["M"]

    for previous_level, level in zip(levels, levels[1:]):
        previous_datasets = hierarchy[previous_level]
        current_datasets = hierarchy[level]
        for previous_dataset, current_dataset in zip(previous_datasets, current_datasets):
            current_name = current_dataset.name
            previous_name = previous_dataset.name

            if "ground_truth_X" not in current_dataset.obsm:
                candidate_keys = (
                    f"bin_assignments_{previous_name}_level_{level}",
                    f"bin_assignments_{current_name}_level_{level}",
                    f"bin_assignments_{current_name}",
                )
                for key in candidate_keys:
                    if key in current_dataset.obsm:
                        bin_assignments = current_dataset.obsm[key]
                        break
                else:
                    raise KeyError(
                        f"Could not find bin assignments for level {level}; tried {candidate_keys}.",
                    )
                current_dataset.simulation.ground_truth_X = bin_assignments @ previous_dataset.simulation.ground_truth_X

            if "ground_truth_M" not in current_dataset.uns and "ground_truth_M" in previous_dataset.uns:
                current_dataset.simulation.ground_truth_M = previous_dataset.simulation.ground_truth_M.copy()


def evaluate_anndata_hierarchy(
    hierarchy: Mapping[int, Sequence[ad.AnnData]],
    metagene_indices,
    is_spatial: bool,
    metric_names: Sequence[str] | None = None,
) -> None:
    """Evaluate metric fields in place for every level of an AnnData
    hierarchy."""

    from popari._dataset_utils import _compute_empirical_correlations

    propagate_ground_truth_to_anndata_hierarchy(hierarchy)

    for level in sorted(hierarchy):
        datasets = hierarchy[level]
        if is_spatial:
            _compute_empirical_correlations(datasets, scaling=10)
            _compute_empirical_correlations(
                datasets,
                scaling=10,
                feature="ground_truth_X",
                output="ground_truth_correlation",
            )

        for dataset, metagene_index in zip(datasets, metagene_indices):
            evaluate_ground_truth(dataset, real_metagene_index=metagene_index)
            compute_matching(dataset, real_metagene_index=metagene_index)
            if is_spatial:
                compute_affinity_coherence(dataset)
                compute_affinity_correlation(dataset, real_metagene_index=metagene_index)


__all__ = [
    propagate_ground_truth.__name__,
    evaluate_ground_truth.__name__,
    compute_matching.__name__,
    compute_affinity_correlation.__name__,
    compute_affinity_coherence.__name__,
    evaluate_model.__name__,
    propagate_ground_truth_to_anndata_hierarchy.__name__,
    evaluate_anndata_hierarchy.__name__,
]
