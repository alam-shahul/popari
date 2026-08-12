"""Postprocessing for learned Popari embeddings."""

from __future__ import annotations

from copy import deepcopy
from typing import Literal

import anndata as ad
import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix
from scipy.stats import zscore

from popari._sample_axis import SampleAxis


def compute_metagene_proportions(
    dataset: ad.AnnData,
    embedding_key: str = "X",
    key_added: str = "metagene_proportions",
) -> None:
    """Store each observation's relative metagene activity.

    Popari embeddings are nonnegative, so dividing each row by its sum gives the
    compositional contribution of each metagene. Zero rows remain zero.

    """

    if embedding_key not in dataset.obsm:
        raise KeyError(f"Missing embeddings in obsm[{embedding_key!r}].")

    embeddings = np.asarray(dataset.obsm[embedding_key], dtype=float)
    if embeddings.ndim != 2:
        raise ValueError(f"obsm[{embedding_key!r}] must be a two-dimensional matrix.")
    if not np.isfinite(embeddings).all():
        raise ValueError(f"obsm[{embedding_key!r}] contains non-finite values.")
    if np.any(embeddings < 0):
        raise ValueError(f"obsm[{embedding_key!r}] must be nonnegative to compute proportions.")

    totals = embeddings.sum(axis=1, keepdims=True)
    dataset.obsm[key_added] = np.divide(
        embeddings,
        totals,
        out=np.zeros_like(embeddings),
        where=totals != 0,
    )


def postprocess_embeddings(
    dataset: ad.AnnData,
    input_key: str = "X",
    normalized_key: str = "normalized_X",
    *,
    mode: Literal["sample", "joint"] = "sample",
    neighbors_key: str | None = None,
) -> None:
    """Standardize embeddings and build a downstream neighbor graph.

    ``mode="sample"`` reproduces processing a sequence of individual samples:
    normalization and neighbor construction are performed independently within
    each sample. ``mode="joint"`` treats all observations as one population.
    Named neighbor graphs follow Scanpy's ``key_added`` storage convention.

    """

    if input_key not in dataset.obsm:
        raise ValueError(f"Missing embeddings in obsm[{input_key!r}].")
    if mode not in {"sample", "joint"}:
        raise ValueError("mode must be 'sample' or 'joint'.")

    embeddings = np.asarray(dataset.obsm[input_key])
    if mode == "joint":
        dataset.obsm[normalized_key] = np.nan_to_num(zscore(embeddings, axis=0))
        neighbors_kwargs = {"key_added": neighbors_key} if neighbors_key is not None else {}
        sc.pp.neighbors(dataset, use_rep=normalized_key, **neighbors_kwargs)
        return

    sample_axis = SampleAxis.from_anndata(dataset, sample_key=dataset.popari.sample_key)
    normalized_embeddings = np.empty(embeddings.shape, dtype=float)
    for sample in sample_axis.names:
        indices = sample_axis.indices(sample)
        normalized_embeddings[indices] = np.nan_to_num(zscore(embeddings[indices], axis=0))
    dataset.obsm[normalized_key] = normalized_embeddings

    metadata_key = neighbors_key or "neighbors"
    graph_keys = (
        ("distances", "connectivities")
        if neighbors_key is None
        else (f"{neighbors_key}_distances", f"{neighbors_key}_connectivities")
    )
    graph_blocks = {key: [] for key in graph_keys}
    neighbors_metadata = None
    for sample in sample_axis.names:
        indices = sample_axis.indices(sample)
        sample_dataset = dataset[indices].copy()
        neighbors_kwargs = {"key_added": neighbors_key} if neighbors_key is not None else {}
        sc.pp.neighbors(sample_dataset, use_rep=normalized_key, **neighbors_kwargs)
        if neighbors_metadata is None:
            neighbors_metadata = deepcopy(sample_dataset.uns[metadata_key])
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
    dataset.uns[metadata_key] = neighbors_metadata
