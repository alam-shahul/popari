import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from popari.analysis import (
    compute_category_edge_rates,
    compute_cell_average_interaction,
    compute_differential_edge_interactions,
    compute_edge_interactions,
    compute_metagene_pair_interaction,
    compute_pair_edge_classification_scores,
    compute_posthoc_colocalization,
    compute_spatial_colocalization,
    frequency_weighted_interaction_matrix,
    match_categories_to_factors,
    summarize_matrix_correlations,
)


def _interaction_dataset():
    dataset = ad.AnnData(X=np.ones((3, 2)))
    dataset.popari.name = "replicate_0"
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
    assert interactions.edge_weights.tolist() == [1.0, 1.0, 1.0]
    assert interactions.scores.tolist() == pytest.approx([0.0, -3.0, -2.0])
    assert "metagene_pair_scores" not in interactions.__dataclass_fields__
    assert not hasattr(interactions, "source_embeddings")
    assert not hasattr(interactions, "target_embeddings")
    assert interactions.metagene_pair_scores(1, 1).tolist() == pytest.approx([0.0, -3.0, 0.0])
    reconstructed_scores = sum(
        interactions.metagene_pair_scores(first, second) for first in range(2) for second in range(2)
    )
    np.testing.assert_allclose(reconstructed_scores, interactions.scores)


def test_compute_edge_interactions_excludes_self_edges():
    dataset = _interaction_dataset()
    dataset.obsp["adjacency_matrix"][0, 0] = 1

    interactions = compute_edge_interactions(dataset)

    assert not np.any(interactions.source == interactions.target)


def test_compute_edge_interactions_accepts_explicit_affinity():
    dataset = _interaction_dataset()
    affinity = np.diag([5.0, 7.0])

    interactions = compute_edge_interactions(dataset, affinity=affinity, rescale=False)

    np.testing.assert_array_equal(interactions.affinity, affinity)
    assert interactions.scores.tolist() == pytest.approx([0.0, -7.0, -5.0])


def test_compute_edge_interactions_rejects_incompatible_affinity_shape():
    dataset = _interaction_dataset()

    with pytest.raises(ValueError, match=r"expected \(2, 2\)"):
        compute_edge_interactions(dataset, affinity=np.eye(3))


def test_compute_differential_edge_interactions_uses_named_contrast():
    dataset = _interaction_dataset()
    affinity_dataset = ad.AnnData(X=np.ones((1, 1)))
    affinity_dataset.uns["average_Sigma_x_inv"] = {
        "comparison": np.diag([5.0, 7.0]),
        "reference": np.diag([2.0, 3.0]),
    }

    interactions = compute_differential_edge_interactions(
        dataset,
        affinity_dataset=affinity_dataset,
        comparison="comparison",
        reference="reference",
        affinity_key="average_Sigma_x_inv",
        rescale=False,
    )

    np.testing.assert_array_equal(interactions.affinity, np.diag([3.0, 4.0]))
    assert interactions.scores.tolist() == pytest.approx([0.0, -4.0, -3.0])


def test_edge_interactions_support_category_and_cell_summaries():
    dataset = _interaction_dataset()
    interactions = compute_edge_interactions(dataset, rescale=False)

    directed = interactions.category_mask(dataset.obs["cell_type"], ("A", "B"))
    undirected = interactions.category_mask(dataset.obs["cell_type"], ("A", "B"), directed=False)
    np.testing.assert_array_equal(directed, [True, False, False])
    np.testing.assert_array_equal(undirected, [True, True, False])

    scores, counts = interactions.category_summary(dataset.obs["cell_type"], categories=["A", "B"])
    pd.testing.assert_frame_equal(
        counts,
        pd.DataFrame([[1, 1], [1, 0]], index=["A", "B"], columns=["A", "B"]),
    )
    pd.testing.assert_frame_equal(
        scores,
        pd.DataFrame([[-2.0, 0.0], [-3.0, 0.0]], index=["A", "B"], columns=["A", "B"]),
    )
    assert interactions.cell_summary().tolist() == pytest.approx([0.0, -3.0, -2.0])
    cell_category_scores = interactions.cell_category_summary(
        dataset.obs["cell_type"].iloc[::-1],
        source_category="A",
        target_categories=["A", "B"],
    )
    assert cell_category_scores.index.tolist() == ["0", "2"]
    assert np.isnan(cell_category_scores.loc["0", "A"])
    assert cell_category_scores.loc["0", "B"] == 0.0
    assert cell_category_scores.loc["2", "A"] == -2.0
    assert np.isnan(cell_category_scores.loc["2", "B"])
    assert interactions.to_frame(dataset.obs["cell_type"]).columns.tolist() == [
        "source",
        "target",
        "source_obs",
        "target_obs",
        "edge_weight",
        "score",
        "source_category",
        "target_category",
    ]

    with pytest.raises(ValueError, match="reduction must be"):
        interactions.cell_category_summary(
            dataset.obs["cell_type"],
            source_category="A",
            target_categories=["B"],
            reduction="median",
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
    dataset.popari.name = "replicate_0"
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
    compute_cell_average_interaction(dataset, rescale=False)

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


def test_edge_interactions_returns_plot_ready_metagene_pair_vectors():
    dataset = _interaction_dataset()
    interactions = compute_edge_interactions(dataset, rescale=False)

    np.testing.assert_array_equal(interactions.edges, np.array([[0, 1], [1, 2], [2, 0]]))
    values = interactions.metagene_pair_scores(1, 1, mode="affinity")
    np.testing.assert_allclose(values, np.array([0.0, -3.0, -0.0]))

    cooccurrence_values = interactions.metagene_pair_scores(1, 1, mode="cooccurrence")
    np.testing.assert_allclose(cooccurrence_values, np.array([1.0, 0.0, 1.0]))

    with pytest.raises(ValueError, match="mode must be"):
        interactions.metagene_pair_scores(1, 1, mode="invalid")


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
