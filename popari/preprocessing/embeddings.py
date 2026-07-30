"""Expression preprocessing operations."""

from __future__ import annotations

import anndata as ad
import scanpy as sc
from scipy.sparse import csr_array, csr_matrix

from popari._datasets import broadcast, enable_joint


@enable_joint(annotations={"obsm": ["X_pca"], "varm": ["PCs"], "uns": ["pca"]})
@broadcast
def pca(dataset: ad.AnnData, n_comps: int = 50, **pca_kwargs):
    """Compute PCA on expression values.

    Args:
        dataset: AnnData object to process.
        n_comps: Number of principal components.
        **pca_kwargs: Additional arguments passed to :func:`scanpy.pp.pca`.

    """
    dataset.X = csr_matrix(dataset.X)  # Hack because sparse PCA isn't implemented
    sc.pp.pca(dataset, n_comps=n_comps, **pca_kwargs)
    dataset.X = csr_array(dataset.X)
