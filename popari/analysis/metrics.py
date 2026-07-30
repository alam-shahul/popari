"""Metrics and summaries for Popari AnnData results."""

from __future__ import annotations

from collections.abc import Sequence

import anndata as ad
import networkx as nx
import numpy as np
from scipy.sparse import issparse
from scipy.stats import zscore
from sklearn.metrics import adjusted_rand_score, confusion_matrix, precision_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from popari._sample_axis import SampleAxis
from popari.schema import _validate_spatial_graph
from popari.util import compute_neighborhood_enrichment


def compute_empirical_correlations(
    dataset: ad.AnnData,
    scaling: float = 10,
    feature: str = "X",
    output: str = "empirical_correlation",
    *,
    neighbor_key: str = "adjacency_matrix",
) -> None:
    """Compute one empirical spatial-correlation matrix per sample.

    Args:
        dataset: Unified multisample AnnData.
        feature: key in `.obsm` of feature set for which spatial correlation should be computed.
        output: key in `.uns` where output correlation matrices should be stored.

    """

    sample_axis = SampleAxis.from_anndata(
        dataset,
        sample_key=dataset.popari.sample_key,
    )
    _validate_spatial_graph(dataset, sample_axis, adjacency_key=neighbor_key)
    embeddings = np.asarray(dataset.obsm[feature])
    num_factors = embeddings.shape[1]
    correlations = {}
    for sample in sample_axis.names:
        indices = sample_axis.indices(sample)
        sample_embeddings = embeddings[indices]
        norms = np.linalg.norm(sample_embeddings, axis=1, keepdims=True, ord=1)
        normalized = np.divide(
            sample_embeddings,
            norms,
            out=np.zeros_like(sample_embeddings, dtype=float),
            where=norms != 0,
        )
        adjacency = dataset.obsp[neighbor_key][indices][:, indices]
        source, target = adjacency.nonzero()
        if len(source) == 0:
            correlation = np.zeros((num_factors, num_factors))
        else:
            x = normalized[source]
            y = normalized[target]
            x = x - x.mean(axis=0, keepdims=True)
            y = y - y.mean(axis=0, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                correlation = -(y / y.std(axis=0, keepdims=True)).T @ (x / x.std(axis=0, keepdims=True)) / len(x)
            correlation = np.nan_to_num(correlation)
            correlation = (correlation + correlation.T) / 2
            correlation -= correlation.mean()
            correlation *= scaling
        correlations[sample] = correlation
    dataset.uns[output] = correlations


def adjacency_permutation_test(
    dataset: ad.AnnData,
    labels: str = "X",
    n_trials: int = 100,
    random_state: int = 0,
    pvalue_key: str = "pvalue",
) -> None:
    r"""Compute p-values for neighborhood enrichment.

    See
    `In silico tissue generation and power analysis for spatial omics <https://www.nature.com/articles/s41592-023-01766-6#Sec13>`_
    for details.

    """

    rng = np.random.default_rng(seed=random_state)
    sample_axis = SampleAxis.from_anndata(
        dataset,
        sample_key=dataset.popari.sample_key,
    )
    dataset.popari.validate_spatial_graph()
    all_labels = np.asarray(dataset.obsm[labels])
    adjacency_pvalues = {}
    avoidance_pvalues = {}
    for sample in sample_axis.names:
        indices = sample_axis.indices(sample)
        original_labels = all_labels[indices]
        adjacency_matrix = dataset.obsp["adjacency_matrix"][indices][:, indices]
        _, num_factors = original_labels.shape
        original_enrichment = compute_neighborhood_enrichment(original_labels, adjacency_matrix)
        neighborhood_enrichments = np.zeros((n_trials, num_factors, num_factors), dtype=np.float64)
        for trial in range(n_trials):
            permuted_labels = rng.permutation(original_labels)
            neighborhood_enrichments[trial] = compute_neighborhood_enrichment(
                permuted_labels,
                adjacency_matrix,
            )

        adjacency = (neighborhood_enrichments > original_enrichment[np.newaxis, :]).sum(axis=0)
        avoidance = (neighborhood_enrichments < original_enrichment[np.newaxis, :]).sum(axis=0)
        adjacency_pvalues[sample] = (n_trials - adjacency + 1) / (n_trials + 1)
        avoidance_pvalues[sample] = (n_trials - avoidance + 1) / (n_trials + 1)

    dataset.uns[f"adjacency_{pvalue_key}"] = adjacency_pvalues
    dataset.uns[f"avoidance_{pvalue_key}"] = avoidance_pvalues


def compute_ari_scores(dataset: ad.AnnData, labels: str, predictions: str, ari_key: str = "ari"):
    r"""Compute adjusted Rand index (ARI) score  between a set of ground truth
    labels and an unsupervised clustering.

    Useful for assessing clustering validity. ARI score is computed per dataset.

    Args:
        dataset: dataset to process
        labels: the key in the ``.obs`` dataframe for the label data.
        predictions: the key in the ``.obs`` dataframe for the predictions data.
        ari_key: the key in the ``.uns`` dictionary where the ARI score will be stored.

    """

    ari = adjusted_rand_score(dataset.obs[labels], dataset.obs[predictions])
    dataset.uns[ari_key] = ari


def compute_silhouette_scores(dataset: ad.AnnData, labels: str, embeddings: str, silhouette_key: str = "silhouette"):
    r"""Compute silhouette score for a clustering based on Popari embeddings.

    Useful for assessing clustering validity. ARI score is computed per dataset.

    Args:
        dataset: dataset to process
        labels: the key in the ``.obs`` dataframe for the label data.
        predictions: the key in the ``.obs`` dataframe for the predictions data.
        ari_key: the key in the ``.uns`` dictionary where the ARI score will be stored.

    """

    silhouette = silhouette_score(dataset.obsm[embeddings], dataset.obs[labels])
    dataset.uns[silhouette_key] = silhouette


def evaluate_classification_task(dataset: ad.AnnData, embeddings: str, labels: str) -> None:
    """"""

    le = LabelEncoder()
    encoded_labels = le.fit_transform(dataset.obs[labels].astype(str))
    dataset_embeddings = dataset.obsm[embeddings]

    X_train, X_valid, y_train, y_valid = train_test_split(
        dataset_embeddings,
        encoded_labels,
        train_size=0.25,
        random_state=42,
        stratify=encoded_labels,
    )
    model = KNeighborsClassifier(n_neighbors=10)
    model.fit(X_train, y_train)

    for split, X, y in [("train", X_train, y_train), ("validation", X_valid, y_valid)]:
        y_soft = model.predict_proba(X)
        y_hat = np.argmax(y_soft, 1)
        dataset.uns[f"microprecision_{split}"] = precision_score(y, y_hat, average="micro")
        dataset.uns[f"macroprecision_{split}"] = precision_score(y, y_hat, average="macro")


def compute_confusion_matrix(
    dataset: ad.AnnData,
    labels: str,
    predictions: str,
    result_key: str = "confusion_matrix",
):
    r"""Compute confusion matrix for labels and predictions.

    Useful for visualizing clustering validity.

    Args:
        dataset: AnnData object containing labels and predictions.
        labels: the key in the ``.obs`` dataframe for the label data.
        predictions: the key in the ``.obs`` dataframe for the predictions data.
        result_key: the key in the ``.uns`` dictionary where the reordered confusion matrix will be stored.

    """

    unique_labels = sorted(dataset.obs[labels].unique())
    unique_predictions = sorted(dataset.obs[predictions].unique())
    if len(unique_labels) != len(unique_predictions):
        raise ValueError("Number of unique labels and unique predictions must be equal.")

    encoded_labels = [unique_labels.index(label) for label in dataset.obs[labels].values]
    encoded_predictions = [unique_predictions.index(prediction) for prediction in dataset.obs[predictions].values]

    confusion_output = confusion_matrix(encoded_labels, encoded_predictions)

    permutation, index = get_optimal_permutation(confusion_output)
    dataset.obs[f"{labels}_inferred"] = [unique_labels[permutation[prediction]] for prediction in encoded_predictions]

    reordered_confusion = confusion_matrix(dataset.obs[labels], dataset.obs[f"{labels}_inferred"])[: len(unique_labels)]

    dataset.uns[result_key] = reordered_confusion


def get_optimal_permutation(confusion_output):
    """
    TODO: document
    maximum weight bipartite matching
    :param confusion_output:
    :return: confusion_output[perm, index], where index is sorted
    """

    num_label_classes, num_prediction_classes = confusion_output.shape

    bipartite_graph = nx.Graph()
    bipartite_graph.add_nodes_from([("label", i) for i in range(num_label_classes)], bipartite=0)
    bipartite_graph.add_nodes_from([("prediction", i) for i in range(num_prediction_classes)], bipartite=1)

    bipartite_graph.add_edges_from(
        [
            (("label", i), ("prediction", j), {"weight": confusion_output[i, j]})
            for i in range(num_label_classes)
            for j in range(num_prediction_classes)
        ],
    )

    assert nx.is_bipartite(bipartite_graph)
    matching = nx.max_weight_matching(bipartite_graph, maxcardinality=True)
    assert len({__ for _ in matching for __ in _}) == num_label_classes * 2

    matching = [sorted(match, key=lambda node_attributes: node_attributes[0]) for match in matching]

    matching = [tuple(index for (_, index) in match) for match in matching]
    matching = sorted(matching, key=lambda pair: pair[1])

    perm, index = tuple(map(np.array, zip(*matching)))

    return perm, index


def compute_columnwise_autocorrelation(
    dataset: ad.AnnData,
    uns: str = "ground_truth_M",
    result_key: str = "ground_truth_M_correlation",
):
    """"""

    matrix = np.asarray(dataset.uns[uns]).T
    num_columns, _ = matrix.shape
    dataset.uns[result_key] = np.corrcoef(matrix, matrix)[:num_columns, :num_columns]


def compute_spatial_gene_correlation(
    dataset: ad.AnnData,
    spatial_key: str = "Sigma_x_inv",
    metagene_key: str = "M",
    spatial_gene_correlation_key: str = "spatial_gene_correlation",
    neighbor_interactions_key: str = "neighbor_interactions",
):
    """Computes spatial gene correlation according to learned metagenes."""

    spatial_gene_correlations = {}
    neighbor_interactions = {}
    metagenes = np.asarray(dataset.uns[metagene_key])
    for sample in dataset.popari.sample_names:
        spatial_affinity_matrix = np.asarray(dataset.uns[spatial_key][sample])
        sample_neighbor_interactions = metagenes @ spatial_affinity_matrix
        neighbor_interactions[sample] = sample_neighbor_interactions
        spatial_gene_correlations[sample] = sample_neighbor_interactions @ metagenes.T

    dataset.uns[spatial_gene_correlation_key] = spatial_gene_correlations
    dataset.uns[neighbor_interactions_key] = neighbor_interactions


def metagene_neighbor_interactions(dataset: ad.AnnData, interaction_key: str = "metagene_neighbor_interactions"):
    """Compute pairwise interactions between every cell in terms of learned
    metagene embeddings.

    Can be used to visualize the empirical spatial correlations between metagenes.

    Args:
        dataset:

    """
    embeddings = dataset.obsm["X"]
    X = embeddings
    adjacency_matrix = dataset.obsp["adjacency_matrix"].toarray()

    adjacency_list = dataset.obsm["adjacency_list"]
    num_cells, num_metagenes = embeddings.shape

    Z = X / np.linalg.norm(X, axis=1, keepdims=True, ord=1)
    edges = np.array([(i, j) for i, e in enumerate(adjacency_list) for j in e])

    x = Z[edges[:, 0]]
    y = Z[edges[:, 1]]

    pair_interactions = np.zeros((num_cells, num_cells, num_metagenes, num_metagenes))
    cell_i, cell_j = adjacency_matrix.nonzero()
    for i in range(num_metagenes):
        for j in range(i, num_metagenes):
            pair_interactions[cell_i, cell_j, i, j] = 1 - x[:, i] * y[:, j]

    dataset.obsp[interaction_key] = pair_interactions


def score_marker_expression(dataset, de_genes: dict[str, Sequence[str]], output_key="marker_expression"):
    """Given a mapping from cell types to marker genes, compute enrichment."""
    data = dataset.X if not issparse(dataset.X) else dataset.X.todense()
    zscored_expression = zscore(data)

    marker_gene_expression = np.zeros((len(dataset), len(de_genes)))
    for index, (subtype, gene_list) in enumerate(de_genes.items()):
        gene_list_index = dataset.var_names.isin(gene_list)
        marker_gene_expression[:, index] = zscored_expression[:, gene_list_index].sum(axis=1)

    dataset.obsm[output_key] = marker_gene_expression
