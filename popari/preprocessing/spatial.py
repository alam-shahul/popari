"""Spatial-graph preprocessing for AnnData objects."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import squidpy as sq
from scipy.sparse import csr_matrix

from popari._sample_axis import SampleAxis
from popari.schema import DEFAULT_SAMPLE_KEY, SAMPLE_KEY_KEY
from popari.util import convert_adjacency_matrix_to_awkward_array


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
    dataset: ad.AnnData,
    threshold: float = 94.5,
    *,
    sample_key: str = DEFAULT_SAMPLE_KEY,
) -> None:
    """Construct a block-diagonal Delaunay graph for a multisample AnnData.

    Spatial neighbors are inferred independently for every sample and then
    scattered into matrices aligned to the original observation order.

    """

    if sample_key not in dataset.obs:
        try:
            sample_name = dataset.popari.name
        except ValueError:
            sample_name = "sample_0"
        dataset.obs[sample_key] = pd.Categorical([sample_name] * dataset.n_obs)
    else:
        dataset.obs[sample_key] = pd.Categorical(dataset.obs[sample_key].astype(str))
        dataset.obs[sample_key] = dataset.obs[sample_key].cat.remove_unused_categories()
    dataset.uns[SAMPLE_KEY_KEY] = sample_key

    sample_axis = SampleAxis.from_anndata(dataset, sample_key=sample_key)
    graph_keys = ("spatial_distances", "spatial_connectivities")
    graph_blocks = {key: [] for key in graph_keys}

    for sample in sample_axis.names:
        indices = sample_axis.indices(sample)
        sample_dataset = dataset[indices].copy()
        sq.gr.spatial_neighbors(sample_dataset, coord_type="generic", delaunay=True)
        distances = sample_dataset.obsp["spatial_distances"]
        if distances.nnz == 0:
            cutoff = 0
        else:
            cutoff = np.percentile(distances.data, threshold)
        sq.gr.spatial_neighbors(
            sample_dataset,
            coord_type="generic",
            delaunay=True,
            radius=[0, cutoff],
        )
        for key in graph_keys:
            graph = sample_dataset.obsp[key].tocoo()
            graph_blocks[key].append(
                csr_matrix(
                    (graph.data, (indices[graph.row], indices[graph.col])),
                    shape=(dataset.n_obs, dataset.n_obs),
                ),
            )

    for key, blocks in graph_blocks.items():
        dataset.obsp[key] = sum(blocks[1:], start=blocks[0]) if blocks else csr_matrix((dataset.n_obs, dataset.n_obs))
    dataset.obsp["adjacency_matrix"] = dataset.obsp["spatial_connectivities"].copy()
    dataset.obsm["adjacency_list"] = convert_adjacency_matrix_to_awkward_array(
        dataset.obsp["adjacency_matrix"],
    )
    dataset.popari.validate_spatial_graph()
