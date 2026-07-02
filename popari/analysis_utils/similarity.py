"""Dataset-level similarity helpers."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr


def calculate_dataset_similarity_matrix(datasets, similarity_function=pearsonr, feature_key: str = "X"):
    """Compute a pairwise similarity matrix for dataset-level average
    embeddings."""

    feature_vectors = [np.asarray(dataset.obsm[feature_key]).mean(axis=0) for dataset in datasets]
    num_datasets = len(feature_vectors)
    similarity_matrix = np.zeros((num_datasets, num_datasets))
    for i, first in enumerate(feature_vectors):
        for j, second in enumerate(feature_vectors):
            similarity_matrix[i, j] = similarity_function(first, second)[0]
    return similarity_matrix


__all__ = [
    calculate_dataset_similarity_matrix.__name__,
]
