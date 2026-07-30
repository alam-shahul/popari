"""Postprocessing for learned Popari embeddings."""

from __future__ import annotations

from collections.abc import Sequence

import anndata as ad
import numpy as np
import scanpy as sc
from scipy.stats import zscore

from popari._datasets import as_datasets


def postprocess_embeddings(
    data: ad.AnnData | Sequence[ad.AnnData],
    input_key: str = "X",
    normalized_key: str = "normalized_X",
) -> None:
    """Standardize embedding dimensions and build downstream neighbor graphs."""

    datasets = as_datasets(data)
    for dataset in datasets:
        if input_key not in dataset.obsm:
            raise ValueError(f"Missing embeddings in obsm[{input_key!r}].")

        normalized_embeddings = np.nan_to_num(zscore(dataset.obsm[input_key]))
        dataset.obsm[normalized_key] = normalized_embeddings
        sc.pp.neighbors(dataset, use_rep=normalized_key)
