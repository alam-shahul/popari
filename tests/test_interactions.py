from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from popari.analysis_utils import (
    compute_category_edge_rates,
    compute_category_interaction,
    compute_cell_average_interaction,
    compute_cell_type_pair_interaction,
    compute_edge_interactions,
    compute_metagene_pair_interaction,
    compute_pair_edge_classification_scores,
    compute_posthoc_colocalization,
    compute_spatial_colocalization,
    frequency_weighted_interaction_matrix,
    match_categories_to_factors,
    metagene_pair_edge_values,
    summarize_matrix_correlations,
)


def _interaction_dataset():
    dataset = ad.AnnData(X=np.ones((3, 2)))
    dataset.obsm["X"] = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
    )
    dataset.obs["cell_type"] = ["A", "B", "A"]
    dataset.obsp["adjacency_matrix"] = csr_matrix(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ],
    )
    dataset.uns["Sigma_x_inv"] = {"replicate_0": np.diag([2.0, 3.0])}
    return dataset


def test_compute_edge_interactions_uses_affinity_weighted_edge_scores():
    dataset = _interaction_dataset()

    interactions = compute_edge_interactions(dataset, rescale=False)

    assert interactions.source.tolist() == [0, 1, 2]
    assert interactions.target.tolist() == [1, 2, 0]
    assert interactions.scores.tolist() == pytest.approx([0.0, -3.0, -2.0])
    assert interactions.metagene_pair_scores.shape == (3, 2, 2)
    np.testing.assert_allclose(
        interactions.metagene_pair_scores[1],
        np.array(
            [
                [0.0, 0.0],
                [0.0, -3.0],
            ],
        ),
    )


def test_compute_category_interaction_normalizes_by_category_pair_edges():
    dataset = _interaction_dataset()

    categories = compute_category_interaction([dataset], [dataset], "cell_type", rescale=False)

    assert categories == ["A", "B"]
    assert dataset.uns["cell_type_edge_frequencies"].tolist() == [[1.0, 1.0], [1.0, 0.0]]
    np.testing.assert_allclose(
        dataset.uns["cell_type_interaction"],
        np.array(
            [
                [-2.0, 0.0],
                [-3.0, 0.0],
            ],
        ),
    )


def test_compute_category_edge_rates_source_normalizes_edges():
    dataset = _interaction_dataset()

    edge_rates = compute_category_edge_rates(dataset, categories=["A", "B"])

    expected = pd.DataFrame(
        [[0.5, 0.5], [1.0, 0.0]],
        index=["A", "B"],
        columns=["A", "B"],
    )
    pd.testing.assert_frame_equal(edge_rates, expected)


def test_frequency_weighted_interaction_matrix_multiplies_by_edge_rate():
    interaction = pd.DataFrame(
        [[2.0, 4.0], [6.0, 8.0]],
        index=["A", "B"],
        columns=["A", "B"],
    )
    edge_rates = pd.DataFrame(
        [[0.25, 0.75], [1.0, 0.0]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    weighted = frequency_weighted_interaction_matrix(interaction, edge_rates)

    expected = pd.DataFrame(
        [[0.5, 3.0], [6.0, 0.0]],
        index=["A", "B"],
        columns=["A", "B"],
    )
    pd.testing.assert_frame_equal(weighted, expected)


def _colocalization_dataset():
    dataset = ad.AnnData(X=np.ones((3, 2)))
    dataset.popari.name = "replicate_0"
    dataset.obsm["X"] = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 1.0],
        ],
    )
    dataset.obsp["adjacency_matrix"] = csr_matrix(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
            [3.0, 0.0, 0.0],
        ],
    )
    return dataset


def test_compute_spatial_colocalization_stores_symmetric_centered_matrices_by_default():
    for method in ["empirical", "pearson", "cosine", "hotspot"]:
        dataset = _colocalization_dataset()

        compute_spatial_colocalization(dataset, method)

        output_key = {
            "empirical": "empirical_correlation",
            "pearson": "pearson_correlation",
            "cosine": "cosine_similarity",
            "hotspot": "hotspot_local_correlation",
        }[method]
        matrix = dataset.uns[output_key][dataset.popari.name]
        assert matrix.shape == (2, 2)
        assert np.isfinite(matrix).all()
        np.testing.assert_allclose(matrix, matrix.T)
        assert matrix.mean() == pytest.approx(0.0)


def test_compute_spatial_colocalization_can_preserve_raw_directed_values():
    dataset = _colocalization_dataset()

    compute_spatial_colocalization(
        dataset,
        "cosine",
        output="raw_cosine",
        symmetrize=False,
        zero_center=False,
    )

    matrix = dataset.uns["raw_cosine"][dataset.popari.name]
    assert not np.allclose(matrix, matrix.T)
    assert matrix.mean() != pytest.approx(0.0)


def test_compute_spatial_colocalization_rejects_unknown_methods():
    dataset = _colocalization_dataset()

    with pytest.raises(ValueError, match="method must be one of"):
        compute_spatial_colocalization(dataset, "invalid")


def test_compute_posthoc_colocalization_computes_all_reviewer_baselines():
    dataset = _colocalization_dataset()

    compute_posthoc_colocalization(dataset)

    for output_key in [
        "empirical_correlation",
        "pearson_correlation",
        "cosine_similarity",
        "hotspot_local_correlation",
    ]:
        matrix = dataset.uns[output_key][dataset.popari.name]
        assert matrix.shape == (2, 2)
        assert np.isfinite(matrix).all()


def test_match_categories_to_factors_uses_mean_factor_activity():
    dataset = ad.AnnData(X=np.ones((4, 3)))
    dataset.obsm["X"] = np.array(
        [
            [9.0, 1.0, 0.0],
            [8.0, 2.0, 0.0],
            [1.0, 7.0, 0.0],
            [2.0, 8.0, 0.0],
        ],
    )
    dataset.obs["cell_type"] = ["A", "A", "B", "B"]

    mapping, association = match_categories_to_factors(
        dataset,
        "cell_type",
        rescale=False,
        return_association=True,
    )

    assert mapping.to_dict() == {"A": 0, "B": 1}
    assert association.loc["A", "factor_0"] > association.loc["A", "factor_1"]
    assert association.loc["B", "factor_1"] > association.loc["B", "factor_0"]


def test_compute_pair_edge_classification_scores_uses_matched_factor_pair_scores():
    dataset = ad.AnnData(X=np.ones((4, 2)))
    dataset.obsm["X"] = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
    )
    dataset.obs["cell_type"] = ["A", "B", "A", "B"]
    dataset.obsp["adjacency_matrix"] = csr_matrix(
        [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
        ],
    )
    dataset.uns["Sigma_x_inv"] = {"replicate_0": np.array([[0.0, -5.0], [0.0, 0.0]])}

    scores = compute_pair_edge_classification_scores(
        dataset,
        {"A": 0, "B": 1},
        pairs=[("A", "B")],
        rescale=False,
    )

    assert scores.loc[0, "source_factor"] == 0
    assert scores.loc[0, "target_factor"] == 1
    assert scores.loc[0, "num_positive_edges"] == 3
    assert scores.loc[0, "num_edges"] == 6
    assert scores.loc[0, "num_negative_edges"] == 3
    assert scores.loc[0, "auroc_weight"] == 9
    assert scores.loc[0, "auprc_weight"] == 3
    assert scores.loc[0, "auroc"] == pytest.approx(1.0)
    assert scores.loc[0, "auprc"] == pytest.approx(1.0)


def test_compute_pair_edge_classification_scores_returns_nan_without_positive_edges():
    dataset = _interaction_dataset()

    scores = compute_pair_edge_classification_scores(
        dataset,
        {"A": 0, "B": 1, "C": 0},
        pairs=[("C", "B")],
        rescale=False,
    )

    assert scores.loc[0, "num_positive_edges"] == 0
    assert np.isnan(scores.loc[0, "auroc"])
    assert np.isnan(scores.loc[0, "auprc"])


def test_compute_cell_average_interaction_stores_degree_normalized_scores():
    dataset = _interaction_dataset()
    level = SimpleNamespace(datasets=[dataset])
    model = SimpleNamespace(hierarchy={0: level})

    compute_cell_average_interaction(model, rescale=False)

    assert dataset.obs["aligned_X_average_interaction"].to_numpy().tolist() == pytest.approx([0.0, -3.0, -2.0])


def test_compute_metagene_pair_interaction_can_summarize_by_category_pair():
    dataset = _interaction_dataset()

    interactions = compute_metagene_pair_interaction(dataset, category_key="cell_type", rescale=False)

    assert interactions.shape == (2, 2, 2, 2)
    assert dataset.uns["metagene_pair_interaction_edge_frequencies"].tolist() == [[1.0, 1.0], [1.0, 0.0]]
    np.testing.assert_allclose(
        interactions[1, 0],
        np.array(
            [
                [0.0, 0.0],
                [0.0, -3.0],
            ],
        ),
    )


def test_compute_cell_type_pair_interaction_replaces_notebook_helper():
    dataset = _interaction_dataset()

    interaction = compute_cell_type_pair_interaction(dataset, dataset, ("A", "B"), "cell_type", rescale=False)

    assert interaction.shape == (3, 3)
    assert dataset.obsp["A_B_interaction"] is interaction
    np.testing.assert_allclose(
        interaction.toarray(),
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, -3.0],
                [0.0, 0.0, 0.0],
            ],
        ),
    )


def test_metagene_pair_edge_values_returns_plot_ready_vectors():
    dataset = _interaction_dataset()

    edges, values = metagene_pair_edge_values(dataset, 1, 1, mode="affinity", rescale=False)

    np.testing.assert_array_equal(edges, np.array([[0, 1], [1, 2], [2, 0]]))
    np.testing.assert_allclose(values, np.array([0.0, -3.0, -0.0]))

    _, cooccurrence_values = metagene_pair_edge_values(dataset, 1, 1, mode="cooccurrence", rescale=False)
    np.testing.assert_allclose(cooccurrence_values, np.array([1.0, 0.0, 1.0]))

    with pytest.raises(ValueError, match="mode must be"):
        metagene_pair_edge_values(dataset, 1, 1, mode="invalid", rescale=False)


def test_summarize_matrix_correlations_uses_all_and_off_diagonal_entries():
    reference = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=["A", "B"],
        columns=["A", "B"],
    )
    query = pd.DataFrame(
        [[2.0, 4.0], [6.0, 8.0]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    summary = summarize_matrix_correlations(
        reference,
        {"query": query},
        correlation_fns={"dummy": lambda x, y: (float(np.dot(x, y)), 0.5)},
    )

    assert summary.loc[0, "method"] == "query"
    assert summary.loc[0, "dummy_all"] == pytest.approx(60.0)
    assert summary.loc[0, "dummy_pvalue_all"] == pytest.approx(0.5)
    assert summary.loc[0, "dummy_off_diagonal"] == pytest.approx(26.0)
    assert summary.loc[0, "dummy_pvalue_off_diagonal"] == pytest.approx(0.5)
