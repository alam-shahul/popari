"""Clustering and low-dimensional representation tools."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce
from loguru import logger
from scipy.sparse import csr_array

from popari._graph import graph_neighbors
from popari._sample_axis import SampleAxis


def leiden(
    dataset: ad.AnnData,
    resolution: float = 1.0,
    tolerance: float = 0.05,
    **kwargs,
):
    r"""Compute Leiden clustering across the unified embedding.

    Args:
        dataset: Unified AnnData object to cluster.
        resolution: Leiden resolution. Higher values yield finer clusters.
        tolerance: Resolution-search tolerance when targeting a cluster count.
        **kwargs: Additional arguments passed to :func:`cluster`.

    """
    # n_iterations = kwargs.pop("n_iterations", 2) # TODO: add these parameters back in after resolving issue
    cluster(
        dataset,
        resolution=resolution,
        method="leiden",
        tolerance=tolerance,
        # flavor="igraph",
        # n_iterations=n_iterations,
        **kwargs,
    )


def cluster(
    dataset: ad.AnnData,
    use_rep="normalized_X",
    method: str = "leiden",
    n_neighbors: int = 20,
    target_clusters: int | None = None,
    tolerance: float = 0.005,
    compute_neighbors: bool = True,
    verbose: bool = False,
    **kwargs,
) -> None:
    r"""Compute clustering across a unified AnnData.

    Args:
        dataset: Unified AnnData object to cluster.
        use_rep: the key in the ``.obsm`` dataframe to ue as input to the Leiden clustering algorithm.
        resolution: the resolution to use for Leiden clustering. Higher values yield finer clusters.

    """

    clustering_function = getattr(sc.tl, method)

    random_state = kwargs.pop("random_state", 0)
    resolution = kwargs.pop("resolution", 1.0)
    key_added = kwargs.pop("key_added", method)

    if compute_neighbors:
        sc.pp.neighbors(dataset, use_rep=use_rep, random_state=random_state, n_neighbors=n_neighbors)

    clustering_function(dataset, resolution=resolution, random_state=random_state, key_added=key_added, **kwargs)

    num_clusters = len(dataset.obs[key_added].unique())

    lower_bound = 0.1 * resolution
    upper_bound = 10 * resolution
    while target_clusters and num_clusters != target_clusters and np.abs(lower_bound - upper_bound) > tolerance:
        effective_resolution = (lower_bound * upper_bound) ** 0.5
        clustering_function(
            dataset,
            resolution=effective_resolution,
            random_state=random_state,
            key_added=key_added,
            **kwargs,
        )
        num_clusters = len(dataset.obs[key_added].unique())
        if num_clusters < target_clusters:
            lower_bound = effective_resolution
        elif num_clusters >= target_clusters:
            upper_bound = effective_resolution

        if verbose:
            logger.info("Clustering resolution {:.3g} produced {} clusters", effective_resolution, num_clusters)


def umap(dataset: ad.AnnData, use_rep: str = "X", compute_neighbors: bool = True, n_neighbors: int = 20):
    r"""Compute UMAP across the unified embedding.

    Args:
        dataset: Unified AnnData object to process.

    """

    if compute_neighbors:
        sc.pp.neighbors(dataset, use_rep=use_rep, n_neighbors=n_neighbors)

    sc.tl.umap(dataset)


def cluster_domains(
    dataset: ad.AnnData,
    target_domains: int | None = None,
    skip_thresholding: bool = True,
    batch_correct: bool = False,
):
    """Discover spatial domains across a unified multisample embedding.

    Normalization and spatial smoothing are sample-local. Neighbor construction
    and Leiden clustering are joint so that domain labels are shared across
    samples. Final label smoothing is again restricted to each spatial graph.

    """

    sample_key = dataset.popari.sample_key
    sample_axis = SampleAxis.from_anndata(dataset, sample_key=sample_key)
    normalized_key = "normalized_X"
    if not skip_thresholding:
        normalized_key = "normalized_thresholded_expression"
        normalized_expression = np.asarray(dataset.obsm["normalized_X"])
        normalized_thresholded_expression = np.empty(normalized_expression.shape, dtype=float)
        for sample in sample_axis.names:
            indices = sample_axis.indices(sample)
            sample_dataset = dataset[indices].copy()
            normalized_thresholded_expression[indices] = normalize_expression_by_threshold(
                sample_dataset,
                thresholded_key="normalized_X",
            )
        dataset.obsm[normalized_key] = normalized_thresholded_expression

    processed_key = normalized_key
    if batch_correct:
        sce.pp.scanorama_integrate(dataset, sample_key, basis=normalized_key, verbose=1)
        processed_key = "X_scanorama"

    dataset.popari.validate_spatial_graph()
    smooth_metagene_expression(dataset, processed_key=processed_key)

    cluster(
        dataset,
        verbose=True,
        use_rep="smoothed_expression",
        target_clusters=target_domains,
        n_neighbors=40,
    )

    smooth_labels(dataset, output_key="smoothed_domain")


def normalize_expression_by_threshold(dataset, thresholded_key: str = "elbowed_X", threshold: float = 99.0):
    """Replacement for Z-score threshold."""

    thresholded_expression = dataset.obsm[thresholded_key]
    expression_threshold = np.percentile(thresholded_expression, threshold, axis=0)
    mask = thresholded_expression > expression_threshold

    total_entities = mask.sum(axis=0)
    total_expression = (expression_threshold * mask).sum(axis=0)

    normalized_thresholded_expression = thresholded_expression / total_expression

    dataset.obsm["normalized_thresholded_expression"] = normalized_thresholded_expression

    return normalized_thresholded_expression


def smooth_metagene_expression(
    dataset,
    processed_key: str = "normalized_thresholded_expression",
    adjacency_key: str = "adjacency_matrix",
):
    """"""
    processed_expression = dataset.obsm[processed_key]
    adjacency = csr_array(dataset.obsp[adjacency_key]).astype(bool).astype(float)
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1, 1)
    smoothed_expression = (processed_expression + adjacency @ processed_expression) / (degree + 1)

    dataset.obsm["smoothed_expression"] = smoothed_expression

    return smoothed_expression


def spatially_smooth_feature(labels, adjacency, max_smoothing_rounds=1, smoothing_threshold=0.5):
    """"""
    labels = np.asarray(labels)
    num_entities = len(labels)

    smoothed_labels = labels.copy()
    for _ in range(max_smoothing_rounds):
        new_labels = smoothed_labels.copy()
        for entity in np.arange(num_entities):
            current_cluster = smoothed_labels[entity]

            adjacencies = graph_neighbors(adjacency, entity)
            neighbor_labels = smoothed_labels[adjacencies]
            num_neighbors = len(neighbor_labels)
            if num_neighbors == 0:
                new_labels[entity] = current_cluster
                continue

            values, counts = np.unique(neighbor_labels, return_counts=True)

            max_index = np.argmax(counts)
            max_cluster = values[max_index]

            ratio = (counts[max_index] + (max_cluster == current_cluster)) / (num_neighbors + 1)
            if ratio >= smoothing_threshold:
                new_labels[entity] = max_cluster
            else:
                new_labels[entity] = current_cluster

        if np.all(smoothed_labels == new_labels):
            break

        smoothed_labels = new_labels

    return new_labels


def smooth_labels(
    dataset,
    label_key: str = "leiden",
    output_key: str = "smoothed_leiden",
    smoothing_threshold: float = 0.5,
    max_smoothing_rounds: int = 1,
    adjacency_key: str = "adjacency_matrix",
):
    """"""
    adjacency = csr_array(dataset.obsp[adjacency_key])

    labels = dataset.obs[label_key]
    dataset.obs[output_key] = pd.Categorical(
        spatially_smooth_feature(
            labels,
            adjacency,
            max_smoothing_rounds,
            smoothing_threshold,
        ),
    )

    return dataset.obs[output_key]
