"""Expression preprocessing operations."""

from __future__ import annotations

import anndata as ad
import scanpy as sc
from scipy.sparse import csr_matrix


def pca(dataset: ad.AnnData, n_comps: int = 50, **pca_kwargs):
    """Compute PCA once across the unified expression matrix.

    Args:
        dataset: AnnData object to process.
        n_comps: Number of principal components.
        **pca_kwargs: Additional arguments passed to :func:`scanpy.pp.pca`.

    """
    original_expression = dataset.X
    dataset.X = csr_matrix(dataset.X)
    try:
        sc.pp.pca(dataset, n_comps=n_comps, **pca_kwargs)
    finally:
        dataset.X = original_expression
