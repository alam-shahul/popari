"""Postprocessing for learned Popari embeddings."""

from __future__ import annotations

from copy import deepcopy

import anndata as ad
import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix
from scipy.stats import zscore

from popari._sample_axis import SampleAxis


def postprocess_embeddings(
    dataset: ad.AnnData,
    input_key: str = "X",
    normalized_key: str = "normalized_X",
) -> None:
    """Standardize embeddings and build sample-local neighbor graphs."""

    if input_key not in dataset.obsm:
        raise ValueError(f"Missing embeddings in obsm[{input_key!r}].")

    sample_axis = SampleAxis.from_anndata(dataset, sample_key=dataset.popari.sample_key)
    embeddings = np.asarray(dataset.obsm[input_key])
    normalized_embeddings = np.empty(embeddings.shape, dtype=float)
    for sample in sample_axis.names:
        indices = sample_axis.indices(sample)
        normalized_embeddings[indices] = np.nan_to_num(zscore(embeddings[indices], axis=0))
    dataset.obsm[normalized_key] = normalized_embeddings

    graph_keys = ("distances", "connectivities")
    graph_blocks = {key: [] for key in graph_keys}
    neighbors_metadata = None
    for sample in sample_axis.names:
        indices = sample_axis.indices(sample)
        sample_dataset = dataset[indices].copy()
        sc.pp.neighbors(sample_dataset, use_rep=normalized_key)
        if neighbors_metadata is None:
            neighbors_metadata = deepcopy(sample_dataset.uns["neighbors"])
        for key in graph_keys:
            graph = sample_dataset.obsp[key].tocoo()
            graph_blocks[key].append(
                csr_matrix(
                    (graph.data, (indices[graph.row], indices[graph.col])),
                    shape=(dataset.n_obs, dataset.n_obs),
                ),
            )

    for key, blocks in graph_blocks.items():
        dataset.obsp[key] = sum(blocks[1:], start=blocks[0]) if blocks else csr_matrix((dataset.n_obs,) * 2)
    dataset.uns["neighbors"] = neighbors_metadata
