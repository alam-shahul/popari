"""Dataset-level similarity helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from popari._sample_axis import SampleAxis


def calculate_dataset_similarity_matrix(
    dataset,
    similarity_function=pearsonr,
    feature_key: str = "X",
    *,
    sample_key: str | None = None,
):
    """Compute a pairwise similarity matrix for sample-level average
    embeddings."""

    sample_axis = SampleAxis.from_anndata(
        dataset,
        sample_key=sample_key or dataset.popari.sample_key,
        adjacency_key=None,
    )
    embeddings = np.asarray(dataset.obsm[feature_key])
    feature_vectors = [embeddings[sample_axis.indices(sample)].mean(axis=0) for sample in sample_axis.names]
    num_samples = len(feature_vectors)
    similarity_matrix = np.zeros((num_samples, num_samples))
    for i, first in enumerate(feature_vectors):
        for j, second in enumerate(feature_vectors):
            similarity_matrix[i, j] = similarity_function(first, second)[0]
    return pd.DataFrame(
        similarity_matrix,
        index=sample_axis.names,
        columns=sample_axis.names,
    )


__all__ = [
    calculate_dataset_similarity_matrix.__name__,
]
