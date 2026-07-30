"""Spatial-graph preprocessing for AnnData objects."""

from __future__ import annotations

from collections.abc import Sequence

import anndata as ad
import awkward as ak
import numpy as np
import squidpy as sq
from scipy.sparse import csr_matrix

from popari._datasets import as_datasets


def remove_connectivity_artifacts(
    sparse_distance_matrix: csr_matrix,
    sparse_adjacency_matrix: csr_matrix,
    threshold: float = 94.5,
):
    """Remove graph edges above a percentile of observed edge distances."""

    dense_distances = sparse_distance_matrix.toarray()
    cutoff = np.percentile(sparse_distance_matrix.data, threshold)
    sparse_adjacency_matrix[dense_distances >= cutoff] = 0
    sparse_adjacency_matrix.eliminate_zeros()
    return sparse_adjacency_matrix


def compute_spatial_neighbors(
    data: ad.AnnData | Sequence[ad.AnnData],
    threshold: float = 94.5,
):
    """Construct Delaunay spatial graphs and Popari adjacency annotations."""

    datasets = as_datasets(data)
    for dataset in datasets:
        sq.gr.spatial_neighbors(dataset, coord_type="generic", delaunay=True)
        cutoff = np.percentile(dataset.obsp["spatial_distances"].data, threshold)
        sq.gr.spatial_neighbors(
            dataset,
            coord_type="generic",
            delaunay=True,
            radius=[0, cutoff],
        )

        adjacency_matrix = dataset.obsp["spatial_connectivities"]
        dataset.obsp["adjacency_matrix"] = adjacency_matrix
        adjacency_list = [[] for _ in range(dataset.n_obs)]
        for source, target in zip(*adjacency_matrix.nonzero()):
            adjacency_list[source].append(target)
        dataset.obsm["adjacency_list"] = ak.Array(adjacency_list)

    return datasets
