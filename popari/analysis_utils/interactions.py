"""Interaction summaries from embeddings, spatial affinities, and graph
edges."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from sklearn.metrics import average_precision_score, roc_auc_score

from popari._dataset_utils import _compute_empirical_correlations
from popari.util import convert_adjacency_matrix_to_awkward_array

SPATIAL_COLOCALIZATION_OUTPUTS = {
    "empirical": "empirical_correlation",
    "pearson": "pearson_correlation",
    "cosine": "cosine_similarity",
    "hotspot": "hotspot_local_correlation",
}


@dataclass(frozen=True)
class EdgeInteractions:
    """Affinity-weighted interaction scores on directed graph edges."""

    source: np.ndarray
    target: np.ndarray
    source_embeddings: np.ndarray
    target_embeddings: np.ndarray
    aligned_source_embeddings: np.ndarray
    scores: np.ndarray
    metagene_pair_scores: np.ndarray


def _affinity_matrix(dataset, affinity_key: str, reference_affinity=None):
    affinity_by_name = dataset.uns[affinity_key]
    if reference_affinity is None:
        reference_affinity = 0
    return next(iter(affinity_by_name.values())) - reference_affinity


def _scaled_embeddings(dataset, embedding_key: str, rescale: bool):
    embeddings = np.asarray(dataset.obsm[embedding_key])
    if not rescale:
        return embeddings

    size_factor = np.linalg.norm(embeddings, axis=1, ord=1, keepdims=True)
    return np.divide(embeddings, size_factor, out=np.zeros_like(embeddings, dtype=float), where=size_factor != 0)


def _as_dataset_sequence(datasets):
    if isinstance(datasets, (list, tuple)):
        return datasets
    return (datasets,)


def _graph_edges(dataset, neighbor_key: str):
    if neighbor_key in dataset.obsp:
        adjacency_matrix = dataset.obsp[neighbor_key]
        if hasattr(adjacency_matrix, "tocoo"):
            adjacency_matrix = adjacency_matrix.tocoo()
            source = np.asarray(adjacency_matrix.row)
            target = np.asarray(adjacency_matrix.col)
            weight = np.asarray(adjacency_matrix.data, dtype=float)
        else:
            source, target = np.nonzero(np.asarray(adjacency_matrix))
            weight = np.asarray(adjacency_matrix[source, target], dtype=float)
    elif "adjacency_list" in dataset.obsm:
        source = []
        target = []
        for cell_index, neighbors in enumerate(dataset.obsm["adjacency_list"]):
            for neighbor_index in neighbors:
                source.append(cell_index)
                target.append(neighbor_index)
        source = np.asarray(source, dtype=int)
        target = np.asarray(target, dtype=int)
        weight = np.ones(len(source), dtype=float)
    else:
        raise KeyError(f"Expected `{neighbor_key}` in `.obsp` or `adjacency_list` in `.obsm`.")

    non_self_edges = source != target
    return source[non_self_edges], target[non_self_edges], weight[non_self_edges]


def _postprocess_colocalization_matrix(matrix, *, symmetrize: bool, zero_center: bool, scaling: float):
    matrix = np.nan_to_num(matrix)
    if symmetrize:
        matrix = (matrix + matrix.T) / 2
    if zero_center:
        matrix = matrix - matrix.mean()
    return matrix * scaling


def compute_spatial_colocalization(
    datasets,
    method: str,
    *,
    feature: str = "X",
    output: str | None = None,
    neighbor_key: str = "adjacency_matrix",
    symmetrize: bool = True,
    zero_center: bool = True,
    scaling: float = 1,
):
    """Compute one post-hoc factor co-localization matrix on spatial graph
    edges.

    Results are stored as ``dataset.uns[output][dataset.popari.name]`` for each
    dataset, matching Popari spatial-affinity storage.

    """

    method = method.lower()
    if method not in SPATIAL_COLOCALIZATION_OUTPUTS:
        supported_methods = ", ".join(sorted(SPATIAL_COLOCALIZATION_OUTPUTS))
        raise ValueError(f"method must be one of: {supported_methods}.")
    if output is None:
        output = SPATIAL_COLOCALIZATION_OUTPUTS[method]

    if method == "empirical":
        if not symmetrize or not zero_center:
            raise ValueError("method='empirical' uses the legacy helper, which always symmetrizes and zero-centers.")
        for dataset in _as_dataset_sequence(datasets):
            if "adjacency_list" not in dataset.obsm and neighbor_key in dataset.obsp:
                dataset.obsm["adjacency_list"] = convert_adjacency_matrix_to_awkward_array(dataset.obsp[neighbor_key])
        return _compute_empirical_correlations(
            tuple(_as_dataset_sequence(datasets)),
            scaling=scaling,
            feature=feature,
            output=output,
        )

    for dataset in _as_dataset_sequence(datasets):
        embeddings = np.asarray(dataset.obsm[feature])
        source, target, weight = _graph_edges(dataset, neighbor_key)
        source_embeddings = embeddings[source]
        target_embeddings = embeddings[target]
        total_weight = weight.sum()
        if total_weight == 0:
            matrix = np.zeros((embeddings.shape[1], embeddings.shape[1]))
        elif method == "pearson":
            source_mean = np.average(source_embeddings, axis=0, weights=weight)
            target_mean = np.average(target_embeddings, axis=0, weights=weight)
            centered_source = source_embeddings - source_mean
            centered_target = target_embeddings - target_mean
            covariance = centered_source.T @ (centered_target * weight[:, None]) / total_weight
            source_std = np.sqrt(np.average(centered_source**2, axis=0, weights=weight))
            target_std = np.sqrt(np.average(centered_target**2, axis=0, weights=weight))
            matrix = np.divide(
                covariance,
                source_std[:, None] * target_std[None, :],
                out=np.zeros_like(covariance),
                where=(source_std[:, None] != 0) & (target_std[None, :] != 0),
            )
        elif method == "cosine":
            dot_product = source_embeddings.T @ (target_embeddings * weight[:, None])
            source_norm = np.sqrt((source_embeddings**2).T @ weight)
            target_norm = np.sqrt((target_embeddings**2).T @ weight)
            matrix = np.divide(
                dot_product,
                source_norm[:, None] * target_norm[None, :],
                out=np.zeros_like(dot_product),
                where=(source_norm[:, None] != 0) & (target_norm[None, :] != 0),
            )
        else:
            matrix = (
                source_embeddings.T @ (target_embeddings * weight[:, None])
                + target_embeddings.T @ (source_embeddings * weight[:, None])
            ) / total_weight

        dataset.uns[output] = {
            dataset.popari.name: _postprocess_colocalization_matrix(
                matrix,
                symmetrize=symmetrize,
                zero_center=zero_center,
                scaling=scaling,
            ),
        }

    return datasets


def compute_posthoc_colocalization(
    datasets,
    *,
    methods=("empirical", "pearson", "cosine", "hotspot"),
    feature: str = "X",
    neighbor_key: str = "adjacency_matrix",
    symmetrize: bool = True,
    zero_center: bool = True,
    scaling: float = 1,
    outputs: dict[str, str] | None = None,
):
    """Compute multiple post-hoc factor co-localization baselines.

    This is the public AnnData-level API for reviewer-style baselines. It works
    for Popari embeddings and external topic models such as STAMP as long as the
    features are stored in ``.obsm[feature]`` and the spatial graph is available
    in ``.obsp[neighbor_key]`` or ``.obsm["adjacency_list"]``.

    """

    outputs = {} if outputs is None else dict(outputs)
    for method in methods:
        compute_spatial_colocalization(
            datasets,
            method,
            feature=feature,
            output=outputs.get(method),
            neighbor_key=neighbor_key,
            symmetrize=symmetrize,
            zero_center=zero_center,
            scaling=scaling,
        )
    return datasets


def compute_edge_interactions(
    dataset,
    *,
    affinity_dataset=None,
    embedding_key: str = "X",
    affinity_key: str = "Sigma_x_inv",
    neighbor_key: str = "adjacency_matrix",
    rescale: bool = True,
    reference_affinity=None,
) -> EdgeInteractions:
    """Compute affinity-weighted interaction scores on every graph edge.

    The canonical edge score is ``-z_i @ Sigma @ z_j``, where ``z`` is the
    optionally L1-normalized embedding. The returned edge order follows
    ``dataset.obsp[neighbor_key].nonzero()`` exactly.

    """

    if affinity_dataset is None:
        affinity_dataset = dataset

    scaled_embeddings = _scaled_embeddings(dataset, embedding_key, rescale)
    affinity_matrix = _affinity_matrix(affinity_dataset, affinity_key, reference_affinity)
    aligned_embeddings = -scaled_embeddings @ affinity_matrix

    adjacency_matrix = dataset.obsp[neighbor_key]
    source, target = adjacency_matrix.nonzero()
    source_embeddings = scaled_embeddings[source]
    target_embeddings = scaled_embeddings[target]
    aligned_source_embeddings = aligned_embeddings[source]

    metagene_pair_scores = -source_embeddings[:, :, None] * affinity_matrix[None, :, :] * target_embeddings[:, None, :]
    scores = np.einsum("ij,ij->i", aligned_source_embeddings, target_embeddings)

    return EdgeInteractions(
        source=np.asarray(source),
        target=np.asarray(target),
        source_embeddings=source_embeddings,
        target_embeddings=target_embeddings,
        aligned_source_embeddings=aligned_source_embeddings,
        scores=scores,
        metagene_pair_scores=metagene_pair_scores,
    )


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

        edge_interactions = compute_edge_interactions(
            embedding_dataset,
            affinity_dataset=affinity_dataset,
            embedding_key=embedding_key,
            affinity_key=affinity_key,
            neighbor_key=neighbor_key,
            rescale=rescale,
            reference_affinity=reference_affinity_matrix,
        )
        scaled_embeddings = _scaled_embeddings(embedding_dataset, embedding_key, rescale)
        affinity_matrix = _affinity_matrix(affinity_dataset, affinity_key, reference_affinity_matrix)
        embedding_dataset.obsm[f"{output_key}_{embedding_key}"] = -scaled_embeddings @ affinity_matrix

        for other_category_index, other_category in enumerate(categories):
            if model_mode:
                embedding_dataset.obs[f"{other_category}_interaction"] = 0.0
            for category_index, category in enumerate(categories):
                category_mask = np.asarray(labels == category)
                other_mask = np.asarray(labels == other_category)
                edge_mask = category_mask[edge_interactions.source] & other_mask[edge_interactions.target]
                edge_count = edge_mask.sum()
                if edge_count > 0:
                    source_cells = edge_interactions.source[edge_mask]
                    category_interaction = np.zeros(category_mask.sum())
                    source_positions = np.searchsorted(np.flatnonzero(category_mask), source_cells)
                    np.add.at(category_interaction, source_positions, edge_interactions.scores[edge_mask])
                    if model_mode:
                        embedding_dataset.obs.loc[category_mask, f"{other_category}_interaction"] = (
                            category_interaction.ravel()
                        )
                    label_interactions[category_index, other_category_index] = np.sum(category_interaction)
                    edge_frequencies[category_index, other_category_index] = edge_count

        label_interactions = np.divide(
            label_interactions,
            edge_frequencies,
            out=np.zeros_like(label_interactions),
            where=edge_frequencies != 0,
        )
        embedding_dataset.uns[f"{category_key}_interaction"] = label_interactions
        embedding_dataset.uns[f"{category_key}_edge_frequencies"] = edge_frequencies
        embedding_dataset.uns[f"{category_key}_categories"] = categories

    return categories


def compute_category_edge_rates(
    dataset,
    *,
    categories,
    category_key: str = "cell_type",
    neighbor_key: str = "adjacency_matrix",
) -> pd.DataFrame:
    """Compute source-normalized category edge rates.

    Entry ``(a, b)`` is ``P(neighbor category = b | source category = a)``
    over directed graph edges from ``dataset.obsp[neighbor_key]``.

    """

    labels = dataset.obs[category_key].astype(str).to_numpy()
    categories = [str(category) for category in categories]
    category_to_index = {category: index for index, category in enumerate(categories)}

    source, target = dataset.obsp[neighbor_key].nonzero()
    edge_counts = np.zeros((len(categories), len(categories)), dtype=float)
    outgoing_counts = np.zeros(len(categories), dtype=float)

    for source_cell, target_cell in zip(source, target):
        source_label = labels[source_cell]
        target_label = labels[target_cell]
        if source_label not in category_to_index or target_label not in category_to_index:
            continue
        source_index = category_to_index[source_label]
        target_index = category_to_index[target_label]
        edge_counts[source_index, target_index] += 1
        outgoing_counts[source_index] += 1

    edge_rates = np.divide(
        edge_counts,
        outgoing_counts[:, None],
        out=np.zeros_like(edge_counts),
        where=outgoing_counts[:, None] != 0,
    )
    return pd.DataFrame(edge_rates, index=categories, columns=categories)


def match_categories_to_factors(
    datasets,
    category_key: str,
    *,
    embedding_key: str = "X",
    categories=None,
    rescale: bool = True,
    return_association: bool = False,
):
    """Match observed categories to embedding factors by mean factor activity.

    This is intended for simulations where a ground-truth category should map to
    one dominant learned factor. The assignment maximizes the summed category-
    by-factor association using the Hungarian algorithm.

    """

    if not isinstance(datasets, (list, tuple)):
        datasets = [datasets]

    if categories is None:
        categories = sorted({str(label) for dataset in datasets for label in dataset.obs[category_key]})
    else:
        categories = [str(category) for category in categories]

    num_factors = datasets[0].obsm[embedding_key].shape[1]
    if len(categories) > num_factors:
        raise ValueError(
            f"Cannot match {len(categories)} categories to {num_factors} factors. "
            "Use fewer categories or more embedding factors.",
        )

    association = np.zeros((len(categories), num_factors), dtype=float)
    category_counts = np.zeros(len(categories), dtype=float)
    category_to_index = {category: index for index, category in enumerate(categories)}

    for dataset in datasets:
        embeddings = _scaled_embeddings(dataset, embedding_key, rescale)
        labels = dataset.obs[category_key].astype(str).to_numpy()
        for category, category_index in category_to_index.items():
            mask = labels == category
            if not mask.any():
                continue
            association[category_index] += embeddings[mask].sum(axis=0)
            category_counts[category_index] += mask.sum()

    association = np.divide(
        association,
        category_counts[:, None],
        out=np.zeros_like(association),
        where=category_counts[:, None] != 0,
    )
    association_frame = pd.DataFrame(
        association,
        index=categories,
        columns=[f"factor_{index}" for index in range(num_factors)],
    )

    category_indices, factor_indices = linear_sum_assignment(-association)
    mapping = pd.Series(
        factor_indices,
        index=[categories[index] for index in category_indices],
        name="factor",
        dtype=int,
    )

    if return_association:
        return mapping, association_frame
    return mapping


def compute_pair_edge_classification_scores(
    dataset,
    category_to_factor,
    *,
    category_key: str = "cell_type",
    embedding_key: str = "X",
    affinity_key: str = "Sigma_x_inv",
    pairs=None,
    neighbor_key: str = "adjacency_matrix",
    rescale: bool = True,
) -> pd.DataFrame:
    """Score whether metagene-pair edge scores identify category-pair edges.

    For category pair ``(a, b)``, the positive class is every directed graph
    edge whose source has category ``a`` and target has category ``b``. The
    prediction score is the edge-level affinity contribution from the matched
    factor pair ``(factor(a), factor(b))``.

    """

    category_to_factor = pd.Series(category_to_factor)
    category_to_factor.index = category_to_factor.index.astype(str)
    if pairs is None:
        categories = category_to_factor.index.astype(str).tolist()
        pairs = [(source, target) for source in categories for target in categories]
    else:
        pairs = [(str(source), str(target)) for source, target in pairs]

    edge_interactions = compute_edge_interactions(
        dataset,
        embedding_key=embedding_key,
        affinity_key=affinity_key,
        neighbor_key=neighbor_key,
        rescale=rescale,
    )
    labels = dataset.obs[category_key].astype(str).to_numpy()

    rows = []
    for source_category, target_category in pairs:
        source_factor = int(category_to_factor.loc[source_category])
        target_factor = int(category_to_factor.loc[target_category])
        positives = (labels[edge_interactions.source] == source_category) & (
            labels[edge_interactions.target] == target_category
        )
        scores = edge_interactions.metagene_pair_scores[:, source_factor, target_factor]

        if positives.any() and (~positives).any():
            auroc = roc_auc_score(positives, scores)
            auprc = average_precision_score(positives, scores)
        else:
            auroc = np.nan
            auprc = np.nan

        rows.append(
            {
                "source_category": source_category,
                "target_category": target_category,
                "source_factor": source_factor,
                "target_factor": target_factor,
                "auroc": auroc,
                "auprc": auprc,
                "num_positive_edges": int(positives.sum()),
                "num_edges": int(len(positives)),
                "num_negative_edges": int((~positives).sum()),
                "auroc_weight": int(positives.sum() * (~positives).sum()),
                "auprc_weight": int(positives.sum()),
            },
        )

    return pd.DataFrame(rows)


def frequency_weighted_interaction_matrix(
    interaction_matrix: pd.DataFrame,
    edge_rate_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Weight conditional interaction scores by source-normalized edge rates."""

    interaction_matrix = interaction_matrix.copy()
    interaction_matrix.index = interaction_matrix.index.astype(str)
    interaction_matrix.columns = interaction_matrix.columns.astype(str)
    return interaction_matrix * edge_rate_matrix.loc[interaction_matrix.index, interaction_matrix.columns]


def summarize_matrix_correlations(
    reference_matrix: pd.DataFrame,
    query_matrices: dict[str, pd.DataFrame],
    correlation_fns: dict[str, object],
) -> pd.DataFrame:
    """Summarize all-entry and off-diagonal correlations for matrices."""

    rows = []
    for method_name, query_matrix in query_matrices.items():
        aligned_reference = reference_matrix.loc[query_matrix.index, query_matrix.columns]
        reference_values = aligned_reference.to_numpy()
        query_values = query_matrix.to_numpy()
        row = {"method": method_name}
        for correlation_name, correlation_fn in correlation_fns.items():
            for suffix, off_diagonal in [("all", False), ("off_diagonal", True)]:
                if off_diagonal:
                    mask = ~np.eye(aligned_reference.shape[0], dtype=bool)
                    flattened_reference = reference_values[mask]
                    flattened_query = query_values[mask]
                else:
                    flattened_reference = reference_values.ravel()
                    flattened_query = query_values.ravel()

                correlation, pvalue = correlation_fn(flattened_reference, flattened_query)
                row[f"{correlation_name}_{suffix}"] = correlation
                row[f"{correlation_name}_pvalue_{suffix}"] = pvalue
        rows.append(row)

    return pd.DataFrame(rows)


def compute_cell_average_interaction(
    model,
    *,
    affinity_level: int = 0,
    embedding_level: int = 0,
    rescale: bool = True,
    embedding_key: str = "X",
    affinity_key: str = "Sigma_x_inv",
    neighbor_key: str = "adjacency_matrix",
    output_key: str = "aligned",
):
    """Store each cell's degree-normalized average neighbor interaction."""

    affinity_datasets = model.hierarchy[affinity_level].datasets
    embedding_datasets = model.hierarchy[embedding_level].datasets
    for affinity_dataset, embedding_dataset in zip(affinity_datasets, embedding_datasets):
        edge_interactions = compute_edge_interactions(
            embedding_dataset,
            affinity_dataset=affinity_dataset,
            embedding_key=embedding_key,
            affinity_key=affinity_key,
            neighbor_key=neighbor_key,
            rescale=rescale,
        )
        scaled_embeddings = _scaled_embeddings(embedding_dataset, embedding_key, rescale)
        affinity_matrix = _affinity_matrix(affinity_dataset, affinity_key)
        embedding_dataset.obsm[f"{output_key}_{embedding_key}"] = -scaled_embeddings @ affinity_matrix

        degree = np.asarray(embedding_dataset.obsp[neighbor_key].sum(axis=1)).reshape(-1)
        interaction_sum = np.zeros(embedding_dataset.n_obs)
        np.add.at(interaction_sum, edge_interactions.source, edge_interactions.scores)
        embedding_dataset.obs[f"{output_key}_{embedding_key}_average_interaction"] = np.divide(
            interaction_sum,
            degree,
            out=np.zeros_like(interaction_sum, dtype=float),
            where=degree != 0,
        )


def compute_cell_type_pair_interaction(
    embedding_dataset,
    affinity_dataset,
    cell_type_pair: tuple,
    cell_type_key: str,
    *,
    rescale: bool = True,
    embedding_key: str = "X",
    affinity_key: str = "Sigma_x_inv",
    neighbor_key: str = "adjacency_matrix",
    interaction_key: str = "interaction",
):
    """Store edge interactions for one cell-type pair as a sparse cell
    matrix."""

    edge_interactions = compute_edge_interactions(
        embedding_dataset,
        affinity_dataset=affinity_dataset,
        embedding_key=embedding_key,
        affinity_key=affinity_key,
        neighbor_key=neighbor_key,
        rescale=rescale,
    )
    labels = embedding_dataset.obs[cell_type_key]
    first_cell_type, second_cell_type = cell_type_pair
    first_mask = np.asarray(labels == first_cell_type)
    second_mask = np.asarray(labels == second_cell_type)

    edge_mask = (first_mask[edge_interactions.source] & second_mask[edge_interactions.target]) | (
        second_mask[edge_interactions.source] & first_mask[edge_interactions.target]
    )
    interaction_matrix = csr_matrix(
        (
            edge_interactions.scores[edge_mask],
            (edge_interactions.source[edge_mask], edge_interactions.target[edge_mask]),
        ),
        shape=(embedding_dataset.n_obs, embedding_dataset.n_obs),
    )
    first_label, second_label = cell_type_pair
    embedding_dataset.obsp[f"{first_label}_{second_label}_{interaction_key}"] = interaction_matrix
    return interaction_matrix


def compute_metagene_pair_interaction(
    dataset,
    *,
    category_key: str | None = None,
    embedding_key: str = "X",
    affinity_key: str = "Sigma_x_inv",
    neighbor_key: str = "adjacency_matrix",
    rescale: bool = True,
    output_key: str = "metagene_pair_interaction",
):
    """Store affinity-weighted metagene-pair interactions averaged over
    edges."""

    edge_interactions = compute_edge_interactions(
        dataset,
        embedding_key=embedding_key,
        affinity_key=affinity_key,
        neighbor_key=neighbor_key,
        rescale=rescale,
    )

    if category_key is None:
        dataset.uns[output_key] = edge_interactions.metagene_pair_scores.mean(axis=0)
        dataset.uns[f"{output_key}_edge_count"] = len(edge_interactions.source)
        return dataset.uns[output_key]

    labels = dataset.obs[category_key]
    categories = sorted(labels.unique())
    num_metagenes = dataset.obsm[embedding_key].shape[1]
    interaction = np.zeros((len(categories), len(categories), num_metagenes, num_metagenes))
    edge_counts = np.zeros((len(categories), len(categories)))
    for source_index, source_category in enumerate(categories):
        source_mask = np.asarray(labels == source_category)[edge_interactions.source]
        for target_index, target_category in enumerate(categories):
            edge_mask = source_mask & np.asarray(labels == target_category)[edge_interactions.target]
            edge_counts[source_index, target_index] = edge_mask.sum()
            if edge_counts[source_index, target_index] > 0:
                interaction[source_index, target_index] = edge_interactions.metagene_pair_scores[edge_mask].mean(axis=0)

    dataset.uns[output_key] = interaction
    dataset.uns[f"{output_key}_edge_frequencies"] = edge_counts
    dataset.uns[f"{output_key}_categories"] = categories
    return interaction


def metagene_pair_edge_values(
    dataset,
    first_metagene: int,
    second_metagene: int,
    *,
    mode: str = "affinity",
    embedding_key: str = "X",
    affinity_key: str = "Sigma_x_inv",
    neighbor_key: str = "adjacency_matrix",
    rescale: bool = True,
):
    """Return graph edges and edge colors for one metagene pair.

    ``mode="affinity"`` returns Popari affinity-weighted interactions
    ``-z_i[k] * Sigma[k, l] * z_j[l]``. ``mode="cooccurrence"`` returns the
    older exploratory score ``1 - z_i[k] * z_j[l]``.

    """

    edge_interactions = compute_edge_interactions(
        dataset,
        embedding_key=embedding_key,
        affinity_key=affinity_key,
        neighbor_key=neighbor_key,
        rescale=rescale,
    )
    edges = np.column_stack([edge_interactions.source, edge_interactions.target])
    if mode == "affinity":
        values = edge_interactions.metagene_pair_scores[:, first_metagene, second_metagene]
    elif mode == "cooccurrence":
        values = (
            1
            - edge_interactions.source_embeddings[:, first_metagene]
            * edge_interactions.target_embeddings[:, second_metagene]
        )
    else:
        raise ValueError("mode must be either 'affinity' or 'cooccurrence'.")

    return edges, values


__all__ = [
    EdgeInteractions.__name__,
    compute_edge_interactions.__name__,
    average_category_interaction.__name__,
    compute_category_interaction.__name__,
    compute_category_edge_rates.__name__,
    match_categories_to_factors.__name__,
    compute_pair_edge_classification_scores.__name__,
    compute_spatial_colocalization.__name__,
    compute_posthoc_colocalization.__name__,
    frequency_weighted_interaction_matrix.__name__,
    summarize_matrix_correlations.__name__,
    compute_cell_average_interaction.__name__,
    compute_cell_type_pair_interaction.__name__,
    compute_metagene_pair_interaction.__name__,
    metagene_pair_edge_values.__name__,
]
