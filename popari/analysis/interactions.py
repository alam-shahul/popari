"""Interaction summaries from embeddings, spatial affinities, and graph
edges."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score, roc_auc_score

from popari._sample_axis import SampleAxis
from popari.analysis.metrics import compute_empirical_correlations

SPATIAL_COLOCALIZATION_OUTPUTS = {
    "empirical": "empirical_correlation",
    "pearson": "pearson_correlation",
    "cosine": "cosine_similarity",
    "hotspot": "hotspot_local_correlation",
}


@dataclass(frozen=True)
class EdgeInteractions:
    """Lazy affinity-weighted interaction analysis on directed graph edges."""

    source: np.ndarray
    target: np.ndarray
    edge_weights: np.ndarray
    embeddings: np.ndarray
    affinity: np.ndarray
    obs_names: pd.Index

    @property
    def edges(self) -> np.ndarray:
        """Return graph edges as an ``(E, 2)`` integer array."""

        return np.column_stack([self.source, self.target])

    @property
    def scores(self) -> np.ndarray:
        """Return total scores ``-z_i @ affinity @ z_j`` for every edge."""

        aligned_source = -self.embeddings[self.source] @ self.affinity
        return np.einsum("ij,ij->i", aligned_source, self.embeddings[self.target])

    def metagene_pair_scores(
        self,
        first_metagene: int,
        second_metagene: int,
        *,
        mode: str = "affinity",
    ) -> np.ndarray:
        """Return one metagene-pair contribution for every edge."""

        source_values = self.embeddings[self.source, first_metagene]
        target_values = self.embeddings[self.target, second_metagene]
        if mode == "affinity":
            return -source_values * self.affinity[first_metagene, second_metagene] * target_values
        if mode == "cooccurrence":
            return 1 - source_values * target_values
        raise ValueError("mode must be either 'affinity' or 'cooccurrence'.")

    def mean_metagene_pair_scores(self, edge_mask=None) -> np.ndarray:
        """Return the mean contribution of every metagene pair."""

        if edge_mask is None:
            edge_mask = np.ones(len(self.source), dtype=bool)
        source = self.embeddings[self.source[edge_mask]]
        target = self.embeddings[self.target[edge_mask]]
        if len(source) == 0:
            return np.zeros_like(self.affinity, dtype=float)
        return -self.affinity * (source.T @ target / len(source))

    def category_mask(
        self,
        labels,
        category_pair: tuple,
        *,
        directed: bool = True,
    ) -> np.ndarray:
        """Select edges joining a requested category pair."""

        labels = self._aligned_labels(labels)
        first_category, second_category = category_pair
        forward = (labels[self.source] == first_category) & (labels[self.target] == second_category)
        if directed or first_category == second_category:
            return forward
        reverse = (labels[self.source] == second_category) & (labels[self.target] == first_category)
        return forward | reverse

    def category_summary(self, labels, *, categories=None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return directed category-pair mean scores and edge counts."""

        labels = self._aligned_labels(labels)
        if categories is None:
            categories = list(pd.unique(labels))
        scores = np.zeros((len(categories), len(categories)))
        counts = np.zeros((len(categories), len(categories)), dtype=int)
        edge_scores = self.scores
        for source_index, source_category in enumerate(categories):
            for target_index, target_category in enumerate(categories):
                mask = self.category_mask(labels, (source_category, target_category))
                counts[source_index, target_index] = mask.sum()
                if counts[source_index, target_index]:
                    scores[source_index, target_index] = edge_scores[mask].mean()
        return (
            pd.DataFrame(scores, index=categories, columns=categories),
            pd.DataFrame(counts, index=categories, columns=categories),
        )

    def cell_summary(self, *, reduction: str = "mean") -> pd.Series:
        """Aggregate outgoing edge scores for each cell."""

        score_sums = np.zeros(len(self.obs_names))
        edge_counts = np.zeros(len(self.obs_names), dtype=int)
        np.add.at(score_sums, self.source, self.scores)
        np.add.at(edge_counts, self.source, 1)
        if reduction == "sum":
            values = score_sums
        elif reduction == "mean":
            values = np.divide(
                score_sums,
                edge_counts,
                out=np.zeros_like(score_sums),
                where=edge_counts != 0,
            )
        else:
            raise ValueError("reduction must be 'mean' or 'sum'.")
        return pd.Series(values, index=self.obs_names, name=f"edge_score_{reduction}")

    def cell_category_summary(
        self,
        labels,
        *,
        source_category,
        target_categories,
        reduction: str = "mean",
    ) -> pd.DataFrame:
        """Aggregate outgoing edge scores by source cell and target category.

        Args:
            labels: Category labels aligned to the analyzed observations.
            source_category: Category whose cells define the output rows.
            target_categories: Neighbor categories to include as columns.
            reduction: Edge-score reduction, either ``"mean"`` or ``"sum"``.

        Returns:
            Source cells by target categories. Missing category-specific
            neighborhoods are represented by ``NaN``.

        """

        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'.")

        labels = self._aligned_labels(labels)
        target_categories = list(dict.fromkeys(target_categories))
        source_cells = np.flatnonzero(labels == source_category)
        summary = pd.DataFrame(
            np.nan,
            index=self.obs_names[source_cells],
            columns=target_categories,
            dtype=float,
        )

        for target_category in target_categories:
            edge_mask = (labels[self.source] == source_category) & (labels[self.target] == target_category)
            selected_sources = self.source[edge_mask]
            score_sums = np.zeros(len(self.obs_names))
            edge_counts = np.zeros(len(self.obs_names), dtype=int)
            np.add.at(score_sums, selected_sources, self.scores[edge_mask])
            np.add.at(edge_counts, selected_sources, 1)

            observed_sources = source_cells[edge_counts[source_cells] > 0]
            if reduction == "mean":
                values = score_sums[observed_sources] / edge_counts[observed_sources]
            else:
                values = score_sums[observed_sources]
            summary.loc[self.obs_names[observed_sources], target_category] = values

        return summary

    def to_frame(self, labels=None) -> pd.DataFrame:
        """Return a tidy edge table, optionally including endpoint labels."""

        frame = pd.DataFrame(
            {
                "source": self.source,
                "target": self.target,
                "source_obs": self.obs_names[self.source],
                "target_obs": self.obs_names[self.target],
                "edge_weight": self.edge_weights,
                "score": self.scores,
            },
        )
        if labels is not None:
            labels = self._aligned_labels(labels)
            frame["source_category"] = labels[self.source]
            frame["target_category"] = labels[self.target]
        return frame

    def _aligned_labels(self, labels) -> np.ndarray:
        if isinstance(labels, pd.Series):
            missing = self.obs_names.difference(labels.index)
            if len(missing):
                raise ValueError("labels are missing observations present in the edge interactions.")
            labels = labels.reindex(self.obs_names).to_numpy()
        else:
            labels = np.asarray(labels)
        if len(labels) != len(self.obs_names):
            raise ValueError("labels must contain one value per observation.")
        return labels


def _scaled_embeddings(dataset, embedding_key: str, rescale: bool):
    embeddings = np.asarray(dataset.obsm[embedding_key])
    if not rescale:
        return embeddings

    size_factor = np.linalg.norm(embeddings, axis=1, ord=1, keepdims=True)
    return np.divide(embeddings, size_factor, out=np.zeros_like(embeddings, dtype=float), where=size_factor != 0)


def _sample_axis(dataset, sample_key: str | None = None) -> SampleAxis:
    return SampleAxis.from_anndata(
        dataset,
        sample_key=sample_key or dataset.popari.sample_key,
    )


def _resolve_sample(sample_axis: SampleAxis, sample: str | None) -> str:
    if sample is not None:
        sample_axis.position(sample)
        return str(sample)
    if len(sample_axis) == 1:
        return sample_axis.names[0]
    raise ValueError(
        f"sample must be specified for a multisample AnnData; expected one of {sample_axis.names}.",
    )


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
    dataset,
    method: str,
    *,
    feature: str = "X",
    output: str | None = None,
    neighbor_key: str = "adjacency_matrix",
    symmetrize: bool = True,
    zero_center: bool = True,
    scaling: float = 1,
    sample_key: str | None = None,
) -> None:
    """Compute one post-hoc factor co-localization matrix on spatial graph
    edges.

    Results are stored as ``dataset.uns[output][sample]``, matching Popari
    spatial-affinity storage.

    """

    method = method.lower()
    if method not in SPATIAL_COLOCALIZATION_OUTPUTS:
        supported_methods = ", ".join(sorted(SPATIAL_COLOCALIZATION_OUTPUTS))
        raise ValueError(f"method must be one of: {supported_methods}.")
    if output is None:
        output = SPATIAL_COLOCALIZATION_OUTPUTS[method]

    if method == "empirical":
        compute_empirical_correlations(
            dataset,
            scaling=scaling,
            feature=feature,
            output=output,
            sample_key=sample_key,
            neighbor_key=neighbor_key,
        )
        return

    sample_axis = _sample_axis(dataset, sample_key)
    matrices = {}
    for sample in sample_axis.names:
        sample_dataset = dataset[sample_axis.indices(sample)]
        embeddings = np.asarray(sample_dataset.obsm[feature])
        source, target, weight = _graph_edges(sample_dataset, neighbor_key)
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

        matrices[sample] = _postprocess_colocalization_matrix(
            matrix,
            symmetrize=symmetrize,
            zero_center=zero_center,
            scaling=scaling,
        )
    dataset.uns[output] = matrices


def compute_posthoc_colocalization(
    dataset,
    *,
    methods=("empirical", "pearson", "cosine", "hotspot"),
    feature: str = "X",
    neighbor_key: str = "adjacency_matrix",
    symmetrize: bool = True,
    zero_center: bool = True,
    scaling: float = 1,
    outputs: dict[str, str] | None = None,
    sample_key: str | None = None,
) -> None:
    """Compute multiple post-hoc factor co-localization baselines.

    This is the public AnnData-level API for reviewer-style baselines. It works
    for Popari embeddings and external topic models such as STAMP as long as the
    features are stored in ``.obsm[feature]`` and the spatial graph is available
    in ``.obsp[neighbor_key]`` or ``.obsm["adjacency_list"]``.

    """

    outputs = {} if outputs is None else dict(outputs)
    for method in methods:
        compute_spatial_colocalization(
            dataset,
            method,
            feature=feature,
            output=outputs.get(method),
            neighbor_key=neighbor_key,
            symmetrize=symmetrize,
            zero_center=zero_center,
            scaling=scaling,
            sample_key=sample_key,
        )


def compute_edge_interactions(
    dataset,
    *,
    sample: str | None = None,
    affinity=None,
    embedding_key: str = "X",
    neighbor_key: str = "adjacency_matrix",
    rescale: bool = True,
    sample_key: str | None = None,
) -> EdgeInteractions:
    """Compute affinity-weighted interaction scores on every graph edge.

    The canonical edge score is ``-z_i @ Sigma @ z_j``, where ``z`` is the
    optionally L1-normalized embedding. Self-edges are excluded.

    """

    sample_axis = _sample_axis(dataset, sample_key)
    sample = _resolve_sample(sample_axis, sample)
    sample_dataset = dataset[sample_axis.indices(sample)]
    scaled_embeddings = _scaled_embeddings(sample_dataset, embedding_key, rescale)
    if affinity is None:
        affinity = dataset.popari.spatial_affinity_for(sample)
    affinity = np.asarray(affinity)
    expected_shape = (scaled_embeddings.shape[1], scaled_embeddings.shape[1])
    if affinity.shape != expected_shape:
        raise ValueError(
            f"Affinity matrix has shape {affinity.shape}; expected {expected_shape} "
            f"for embeddings with {scaled_embeddings.shape[1]} factors.",
        )
    source, target, edge_weights = _graph_edges(sample_dataset, neighbor_key)

    return EdgeInteractions(
        source=np.asarray(source),
        target=np.asarray(target),
        edge_weights=np.asarray(edge_weights),
        embeddings=np.asarray(scaled_embeddings).copy(),
        affinity=affinity.copy(),
        obs_names=sample_dataset.obs_names.copy(),
    )


def compute_differential_edge_interactions(
    dataset,
    *,
    comparison: str,
    reference: str,
    sample: str | None = None,
    embedding_key: str = "X",
    affinity_key: str = "Sigma_x_inv",
    neighbor_key: str = "adjacency_matrix",
    rescale: bool = True,
    sample_key: str | None = None,
) -> EdgeInteractions:
    """Compute one sample's edge scores from a named affinity contrast."""

    affinity = dataset.popari.affinity_difference(
        comparison,
        reference,
        spatial_affinity_key=affinity_key,
    )
    return compute_edge_interactions(
        dataset,
        sample=sample,
        affinity=affinity,
        embedding_key=embedding_key,
        neighbor_key=neighbor_key,
        rescale=rescale,
        sample_key=sample_key,
    )


def average_category_interaction(embeddings, aligned_embeddings, category_mask, other_category_mask, adjacency_matrix):
    """Impute a category's interaction with neighboring cells for each
    metagene."""

    adjacency_matrix = adjacency_matrix[category_mask][:, other_category_mask]
    edge_count = np.array(adjacency_matrix.sum(axis=1)).sum()
    neighbor_sum = adjacency_matrix @ embeddings[other_category_mask]
    average_interaction = np.sum(aligned_embeddings[category_mask] * neighbor_sum, axis=1, keepdims=True)
    return average_interaction, edge_count


def compute_category_edge_rates(
    dataset,
    *,
    categories,
    sample: str | None = None,
    category_key: str = "cell_type",
    neighbor_key: str = "adjacency_matrix",
    sample_key: str | None = None,
) -> pd.DataFrame:
    """Compute source-normalized category edge rates.

    Entry ``(a, b)`` is ``P(neighbor category = b | source category = a)``
    over directed graph edges from ``dataset.obsp[neighbor_key]``.

    """

    if sample is not None:
        sample_axis = _sample_axis(dataset, sample_key)
        dataset = dataset[sample_axis.indices(sample)]

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
    dataset,
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

    if categories is None:
        categories = sorted(dataset.obs[category_key].dropna().astype(str).unique())
    else:
        categories = [str(category) for category in categories]

    num_factors = dataset.obsm[embedding_key].shape[1]
    if len(categories) > num_factors:
        raise ValueError(
            f"Cannot match {len(categories)} categories to {num_factors} factors. "
            "Use fewer categories or more embedding factors.",
        )

    association = np.zeros((len(categories), num_factors), dtype=float)
    category_counts = np.zeros(len(categories), dtype=float)
    category_to_index = {category: index for index, category in enumerate(categories)}

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
    sample: str | None = None,
    category_key: str = "cell_type",
    embedding_key: str = "X",
    affinity_key: str = "Sigma_x_inv",
    pairs=None,
    neighbor_key: str = "adjacency_matrix",
    rescale: bool = True,
    sample_key: str | None = None,
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

    sample_axis = _sample_axis(dataset, sample_key)
    sample = _resolve_sample(sample_axis, sample)
    sample_dataset = dataset[sample_axis.indices(sample)]
    edge_interactions = compute_edge_interactions(
        dataset,
        sample=sample,
        affinity=np.asarray(dataset.uns[affinity_key][sample]),
        embedding_key=embedding_key,
        neighbor_key=neighbor_key,
        rescale=rescale,
        sample_key=sample_key,
    )
    labels = sample_dataset.obs[category_key].astype(str).to_numpy()

    rows = []
    for source_category, target_category in pairs:
        source_factor = int(category_to_factor.loc[source_category])
        target_factor = int(category_to_factor.loc[target_category])
        positives = (labels[edge_interactions.source] == source_category) & (
            labels[edge_interactions.target] == target_category
        )
        scores = edge_interactions.metagene_pair_scores(source_factor, target_factor)

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
    dataset,
    *,
    rescale: bool = True,
    embedding_key: str = "X",
    affinity_key: str = "Sigma_x_inv",
    neighbor_key: str = "adjacency_matrix",
    output_key: str = "aligned",
    sample_key: str | None = None,
) -> None:
    """Store sample-specific aligned embeddings and mean edge accordance."""

    sample_axis = _sample_axis(dataset, sample_key)
    aligned_embeddings = np.zeros_like(np.asarray(dataset.obsm[embedding_key]), dtype=float)
    average_interactions = np.zeros(dataset.n_obs, dtype=float)
    for sample in sample_axis.names:
        indices = sample_axis.indices(sample)
        sample_dataset = dataset[indices]
        affinity = np.asarray(dataset.uns[affinity_key][sample])
        edge_interactions = compute_edge_interactions(
            dataset,
            sample=sample,
            affinity=affinity,
            embedding_key=embedding_key,
            neighbor_key=neighbor_key,
            rescale=rescale,
            sample_key=sample_key,
        )
        scaled_embeddings = _scaled_embeddings(sample_dataset, embedding_key, rescale)
        aligned_embeddings[indices] = -scaled_embeddings @ affinity

        degree = np.asarray(sample_dataset.obsp[neighbor_key].sum(axis=1)).reshape(-1)
        interaction_sum = np.zeros(sample_dataset.n_obs)
        np.add.at(interaction_sum, edge_interactions.source, edge_interactions.scores)
        average_interactions[indices] = np.divide(
            interaction_sum,
            degree,
            out=np.zeros_like(interaction_sum, dtype=float),
            where=degree != 0,
        )
    dataset.obsm[f"{output_key}_{embedding_key}"] = aligned_embeddings
    dataset.obs[f"{output_key}_{embedding_key}_average_interaction"] = average_interactions


def compute_metagene_pair_interaction(
    dataset,
    *,
    sample: str | None = None,
    category_key: str | None = None,
    embedding_key: str = "X",
    affinity_key: str = "Sigma_x_inv",
    neighbor_key: str = "adjacency_matrix",
    rescale: bool = True,
    output_key: str = "metagene_pair_interaction",
    sample_key: str | None = None,
):
    """Store affinity-weighted metagene-pair interactions averaged over
    edges."""

    sample_axis = _sample_axis(dataset, sample_key)
    samples = sample_axis.names if sample is None else (_resolve_sample(sample_axis, sample),)
    interactions_by_sample = {}
    edge_counts_by_sample = {}
    categories = None
    if category_key is not None:
        categories = sorted(dataset.obs[category_key].dropna().unique())

    for selected_sample in samples:
        indices = sample_axis.indices(selected_sample)
        sample_dataset = dataset[indices]
        edge_interactions = compute_edge_interactions(
            dataset,
            sample=selected_sample,
            affinity=np.asarray(dataset.uns[affinity_key][selected_sample]),
            embedding_key=embedding_key,
            neighbor_key=neighbor_key,
            rescale=rescale,
            sample_key=sample_key,
        )

        if category_key is None:
            interactions_by_sample[selected_sample] = edge_interactions.mean_metagene_pair_scores()
            edge_counts_by_sample[selected_sample] = len(edge_interactions.source)
            continue

        labels = sample_dataset.obs[category_key]
        num_metagenes = dataset.obsm[embedding_key].shape[1]
        interaction = np.zeros((len(categories), len(categories), num_metagenes, num_metagenes))
        edge_counts = np.zeros((len(categories), len(categories)))
        for source_index, source_category in enumerate(categories):
            source_mask = np.asarray(labels == source_category)[edge_interactions.source]
            for target_index, target_category in enumerate(categories):
                edge_mask = source_mask & np.asarray(labels == target_category)[edge_interactions.target]
                edge_counts[source_index, target_index] = edge_mask.sum()
                if edge_counts[source_index, target_index] > 0:
                    interaction[source_index, target_index] = edge_interactions.mean_metagene_pair_scores(edge_mask)
        interactions_by_sample[selected_sample] = interaction
        edge_counts_by_sample[selected_sample] = edge_counts

    dataset.uns[output_key] = interactions_by_sample
    dataset.uns[f"{output_key}_edge_frequencies"] = edge_counts_by_sample
    if categories is not None:
        dataset.uns[f"{output_key}_categories"] = categories
    return interactions_by_sample


__all__ = [
    EdgeInteractions.__name__,
    compute_edge_interactions.__name__,
    compute_differential_edge_interactions.__name__,
    average_category_interaction.__name__,
    compute_category_edge_rates.__name__,
    match_categories_to_factors.__name__,
    compute_pair_edge_classification_scores.__name__,
    compute_spatial_colocalization.__name__,
    compute_posthoc_colocalization.__name__,
    frequency_weighted_interaction_matrix.__name__,
    summarize_matrix_correlations.__name__,
    compute_cell_average_interaction.__name__,
    compute_metagene_pair_interaction.__name__,
]
