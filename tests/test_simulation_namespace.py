import anndata as ad
import numpy as np

import popari  # noqa: F401


def make_dataset(name: str = "replicate"):
    dataset = ad.AnnData(X=np.ones((2, 3)))
    dataset.popari.name = name
    return dataset


def test_ground_truth_x_getter_and_setter():
    dataset = make_dataset()
    ground_truth_X = np.array([[1.0, 0.0], [0.0, 1.0]])

    dataset.simulation.ground_truth_X = ground_truth_X

    np.testing.assert_array_equal(dataset.obsm["ground_truth_X"], ground_truth_X)
    np.testing.assert_array_equal(dataset.simulation.ground_truth_X, ground_truth_X)


def test_ground_truth_m_getter_and_setter_use_shared_matrix():
    dataset = make_dataset("sample_a")
    ground_truth_M = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])

    dataset.simulation.ground_truth_M = ground_truth_M

    np.testing.assert_array_equal(dataset.uns["ground_truth_M"], ground_truth_M)
    np.testing.assert_array_equal(dataset.simulation.ground_truth_M, ground_truth_M)


def test_learned_x_getter_and_setter():
    dataset = make_dataset()
    learned_X = np.array([[0.25, 0.75], [0.75, 0.25]])

    dataset.simulation.learned_X = learned_X

    np.testing.assert_array_equal(dataset.obsm["X"], learned_X)
    np.testing.assert_array_equal(dataset.simulation.learned_X, learned_X)


def test_learned_m_getter_and_setter_use_shared_matrix():
    dataset = make_dataset("sample_b")
    learned_M = np.array([[0.2, 0.8], [0.4, 0.6], [0.6, 0.4]])

    dataset.simulation.learned_M = learned_M

    np.testing.assert_array_equal(dataset.uns["M"], learned_M)
    np.testing.assert_array_equal(dataset.simulation.learned_M, learned_M)


def test_ground_truth_expression_uses_ground_truth_embeddings_and_metagenes():
    dataset = make_dataset()
    dataset.simulation.ground_truth_X = np.array([[1.0, 0.0], [0.25, 0.75]])
    dataset.simulation.ground_truth_M = np.array([[2.0, 0.0], [0.0, 4.0], [1.0, 1.0]])

    np.testing.assert_allclose(
        dataset.simulation.ground_truth_expression,
        dataset.simulation.ground_truth_X @ dataset.simulation.ground_truth_M.T,
    )


def test_shared_metagene_access_does_not_require_dataset_name():
    dataset = ad.AnnData(X=np.ones((2, 3)))
    ground_truth_M = np.ones((3, 1))
    dataset.uns["ground_truth_M"] = ground_truth_M

    np.testing.assert_array_equal(dataset.simulation.ground_truth_M, ground_truth_M)
