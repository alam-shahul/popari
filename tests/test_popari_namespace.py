import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_array

import popari  # noqa: F401


@pytest.mark.parametrize(
    "batch",
    [
        ["old", "old"],
        ["replicate_a", "replicate_b"],
    ],
)
def test_popari_name_property_getter_and_setter_preserves_batch(batch):
    dataset = ad.AnnData(X=np.ones((2, 3)))
    dataset.obs["batch"] = batch

    dataset.popari.name = "sample_a"

    assert dataset.popari.name == "sample_a"
    assert dataset.uns["dataset_name"] == "sample_a"
    assert dataset.obs["batch"].tolist() == batch


def test_popari_name_property_requires_dataset_name():
    dataset = ad.AnnData(X=np.ones((2, 3)))

    with pytest.raises(ValueError, match="Dataset name is not set"):
        _ = dataset.popari.name


def test_spatial_affinity_property_getter_and_setter():
    dataset = ad.AnnData(X=np.ones((2, 3)))
    dataset.popari.name = "sample_a"
    spatial_affinity = np.array([[1.0, 2.0], [2.0, 3.0]])

    dataset.popari.spatial_affinity = spatial_affinity

    np.testing.assert_array_equal(dataset.popari.spatial_affinity, spatial_affinity)
    np.testing.assert_array_equal(dataset.uns["Sigma_x_inv"]["sample_a"], spatial_affinity)


def test_learned_result_properties_use_canonical_anndata_keys():
    dataset = ad.AnnData(X=np.ones((2, 3)))
    dataset.popari.name = "sample_a"
    embedding = np.arange(4).reshape(2, 2)
    metagenes = np.arange(6).reshape(3, 2)
    adjacency_matrix = np.eye(2)
    adjacency_list = np.array([[1], [0]])
    hyperparameters = {"K": 2}

    dataset.popari.embedding = embedding
    dataset.popari.metagenes = metagenes
    dataset.popari.adjacency_matrix = adjacency_matrix
    dataset.popari.adjacency_list = adjacency_list
    dataset.popari.hyperparameters = hyperparameters

    np.testing.assert_array_equal(dataset.popari.embedding, embedding)
    np.testing.assert_array_equal(dataset.popari.metagenes, metagenes)
    np.testing.assert_array_equal(dataset.popari.adjacency_matrix, adjacency_matrix)
    np.testing.assert_array_equal(dataset.popari.adjacency_list, adjacency_list)
    assert dataset.popari.hyperparameters == hyperparameters


def test_affinity_difference_subtracts_named_matrices():
    dataset = ad.AnnData(X=np.ones((2, 3)))
    dataset.uns["Sigma_x_inv"] = {
        "dataset_1": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "dataset_2": np.array([[5.0, 7.0], [11.0, 13.0]]),
    }

    difference = dataset.popari.affinity_difference(
        comparison="dataset_2",
        reference="dataset_1",
    )

    np.testing.assert_array_equal(difference, np.array([[4.0, 5.0], [8.0, 9.0]]))


def test_affinity_difference_supports_non_default_key():
    dataset = ad.AnnData(X=np.ones((2, 3)))
    dataset.uns["average_Sigma_x_inv"] = {
        "dataset_1": np.array([[1.0, 1.0], [1.0, 1.0]]),
        "dataset_2": np.array([[3.0, 5.0], [7.0, 9.0]]),
    }

    difference = dataset.popari.affinity_difference(
        "dataset_2",
        "dataset_1",
        spatial_affinity_key="average_Sigma_x_inv",
    )

    np.testing.assert_array_equal(difference, np.array([[2.0, 4.0], [6.0, 8.0]]))


def test_affinity_difference_requires_affinity_key():
    dataset = ad.AnnData(X=np.ones((2, 3)))

    with pytest.raises(KeyError, match="Missing spatial affinity key"):
        dataset.popari.affinity_difference("dataset_2", "dataset_1")


def test_affinity_difference_requires_named_matrices():
    dataset = ad.AnnData(X=np.ones((2, 3)))
    dataset.uns["Sigma_x_inv"] = {
        "dataset_1": np.eye(2),
    }

    with pytest.raises(KeyError, match="dataset_2"):
        dataset.popari.affinity_difference("dataset_2", "dataset_1")


def test_multisample_namespace_accessors_use_named_parameters():
    dataset = ad.AnnData(X=np.ones((4, 3)))
    dataset.obs_names = [f"cell_{index}" for index in range(4)]
    dataset.var_names = [f"gene_{index}" for index in range(3)]
    dataset.obs["batch"] = pd.Categorical(
        ["sample_b", "sample_a", "sample_b", "sample_a"],
        categories=["sample_a", "sample_b"],
    )
    dataset.obsp["adjacency_matrix"] = csr_array(
        (np.ones(4), ([0, 1, 2, 3], [2, 3, 0, 1])),
        shape=(4, 4),
    )
    dataset.uns["M"] = {"sample_a": np.ones((3, 2)), "sample_b": np.full((3, 2), 2)}
    dataset.uns["Sigma_x_inv"] = {
        "sample_a": np.eye(2),
        "sample_b": np.full((2, 2), 3),
    }

    assert dataset.popari.sample_names == ("sample_a", "sample_b")
    np.testing.assert_array_equal(dataset.popari.sample_indices("sample_a"), [1, 3])
    np.testing.assert_array_equal(dataset.popari.sample_mask("sample_b"), [True, False, True, False])
    np.testing.assert_array_equal(dataset.popari.metagenes_for("sample_b"), np.full((3, 2), 2))
    np.testing.assert_array_equal(dataset.popari.spatial_affinity_for("sample_a"), np.eye(2))

    with pytest.raises(ValueError, match="multiple samples"):
        _ = dataset.popari.metagenes


def test_sample_axis_accessors_do_not_require_a_spatial_graph():
    dataset = ad.AnnData(
        X=np.ones((3, 1)),
        obs=pd.DataFrame(
            {"batch": pd.Categorical(["first", "second", "first"])},
            index=["cell_0", "cell_1", "cell_2"],
        ),
    )

    assert dataset.popari.sample_names == ("first", "second")
    np.testing.assert_array_equal(dataset.popari.sample_indices("first"), [0, 2])
