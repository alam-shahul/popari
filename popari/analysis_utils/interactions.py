"""Category interaction summaries from embeddings and affinity matrices."""

from __future__ import annotations

from typing import Optional

import numpy as np


def average_category_interaction(embeddings, aligned_embeddings, category_mask, other_category_mask, adjacency_matrix):
    """Impute a category's interaction with neighboring cells for each
    metagene."""

    adjacency_matrix = adjacency_matrix[category_mask][:, other_category_mask]
    edge_count = np.array(adjacency_matrix.sum(axis=1)).sum()
    neighbor_sum = adjacency_matrix @ embeddings[other_category_mask]
    average_interaction = np.sum(aligned_embeddings[category_mask] * neighbor_sum, axis=1, keepdims=True)
    return average_interaction, edge_count


def compute_category_interaction(
    affinity_datasets,
    embedding_datasets=None,
    category_key=None,
    categories=None,
    rescale: bool = True,
    embedding_key: str = "X",
    affinity_key: str = "Sigma_x_inv",
    reference_dataset_index: int | None = None,
    reference_dataset_group=None,
    level: int = 0,
    neighbor_key: str = "adjacency_matrix",
    output_key: str = "aligned",
):
    """Compute cell-category interaction scores from embeddings and spatial
    affinities."""

    model_mode = embedding_datasets is None and hasattr(affinity_datasets, "hierarchy")
    if model_mode:
        model = affinity_datasets
        embedding_datasets = model.hierarchy[level].datasets
        affinity_datasets = embedding_datasets
    elif embedding_datasets is None:
        raise TypeError("embedding_datasets must be provided unless the first argument is a Popari model.")

    if category_key is None:
        raise TypeError("category_key must be provided.")

    if categories is None:
        categories = sorted({category for dataset in embedding_datasets for category in dataset.obs[category_key]})

    first_affinity_matrix = next(iter(affinity_datasets[0].uns[affinity_key].values()))
    reference_affinity_matrix = np.zeros_like(first_affinity_matrix)
    if reference_dataset_index is not None:
        reference_affinity_matrix = next(iter(affinity_datasets[reference_dataset_index].uns[affinity_key].values()))
    elif reference_dataset_group is not None:
        raise NotImplementedError("reference_dataset_group requires merged affinity groups and is not available here.")

    num_categories = len(categories)
    for affinity_dataset, embedding_dataset in zip(affinity_datasets, embedding_datasets):
        labels = embedding_dataset.obs[category_key]
        label_interactions = np.zeros((num_categories, num_categories))
        edge_frequencies = np.zeros((num_categories, num_categories))

        embeddings = embedding_dataset.obsm[embedding_key]
        size_factor = (
            np.linalg.norm(embeddings, axis=1, ord=1, keepdims=True) if rescale else np.ones((len(embeddings), 1))
        )
        scaled_embeddings = embeddings / size_factor

        adjacency_matrix = embedding_dataset.obsp[neighbor_key]
        affinity_matrix = next(iter(affinity_dataset.uns[affinity_key].values()))
        aligned_embeddings = -scaled_embeddings @ (affinity_matrix - reference_affinity_matrix)
        embedding_dataset.obsm[f"{output_key}_{embedding_key}"] = aligned_embeddings

        for other_category_index, other_category in enumerate(categories):
            if model_mode:
                embedding_dataset.obs[f"{other_category}_interaction"] = 0.0
            for category_index, category in enumerate(categories):
                category_mask = labels == category
                other_mask = labels == other_category
                if other_mask.sum() > 0 and category_mask.sum() > 0:
                    category_interaction, edge_count = average_category_interaction(
                        scaled_embeddings,
                        aligned_embeddings,
                        category_mask,
                        other_mask,
                        adjacency_matrix,
                    )
                    if model_mode:
                        embedding_dataset.obs.loc[category_mask, f"{other_category}_interaction"] = (
                            category_interaction.ravel()
                        )
                    label_interactions[category_index, other_category_index] = np.sum(category_interaction)
                    edge_frequencies[category_index, other_category_index] = edge_count

        label_interactions /= edge_frequencies + 1e-6
        embedding_dataset.uns[f"{category_key}_interaction"] = label_interactions
        embedding_dataset.uns[f"{category_key}_edge_frequencies"] = edge_frequencies
        embedding_dataset.uns[f"{category_key}_categories"] = categories

    return categories


__all__ = [
    average_category_interaction.__name__,
    compute_category_interaction.__name__,
]
