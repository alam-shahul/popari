"""Clustering and low-dimensional representation tools."""

from __future__ import annotations

import anndata as ad
import numpy as np
import scanpy as sc
import scanpy.external as sce

from popari.schema import DEFAULT_SAMPLE_KEY
from popari.util import normalize_expression_by_threshold, smooth_labels, smooth_metagene_expression


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
            print(f"Current number of clusters: {num_clusters}")
            print(f"Resolution: {effective_resolution}")


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
    *,
    sample_key: str = DEFAULT_SAMPLE_KEY,
):
    """Discover spatial domains across a unified multisample embedding."""

    normalized_key = "normalized_X"
    if not skip_thresholding:
        normalized_key = "normalized_thresholded_expression"
        normalize_expression_by_threshold(dataset, thresholded_key="normalized_X")

    processed_key = normalized_key
    if batch_correct:
        sce.pp.scanorama_integrate(dataset, sample_key, basis=normalized_key, verbose=1)
        processed_key = "X_scanorama"

    smooth_metagene_expression(dataset, processed_key=processed_key)

    cluster(
        dataset,
        verbose=True,
        use_rep="smoothed_expression",
        target_clusters=target_domains,
        n_neighbors=40,
    )

    smooth_labels(dataset, output_key="smoothed_domain")
