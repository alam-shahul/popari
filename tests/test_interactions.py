from types import SimpleNamespace

import anndata as ad
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from popari.analysis_utils import (
    compute_category_interaction,
    compute_cell_average_interaction,
    compute_cell_type_pair_interaction,
    compute_edge_interactions,
    compute_metagene_pair_interaction,
    metagene_pair_edge_values,
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
