import anndata as ad
import numpy as np
import pytest

import popari  # noqa: F401


def test_popari_name_property_getter_and_setter_updates_batch():
    dataset = ad.AnnData(X=np.ones((2, 3)))
    dataset.obs["batch"] = "old"

    dataset.popari.name = "sample_a"

    assert dataset.popari.name == "sample_a"
    assert dataset.uns["dataset_name"] == "sample_a"
    assert dataset.obs["batch"].tolist() == ["sample_a", "sample_a"]


def test_popari_name_property_requires_dataset_name():
    dataset = ad.AnnData(X=np.ones((2, 3)))

    with pytest.raises(ValueError, match="Dataset name is not set"):
        _ = dataset.popari.name


def test_affinity_difference_subtracts_named_matrices():
    dataset = ad.AnnData(X=np.ones((2, 3)))
    dataset.uns["Sigma_x_inv"] = {
        "dataset_1": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "dataset_2": np.array([[5.0, 7.0], [11.0, 13.0]]),
    }

    difference = dataset.popari.affinity_difference("dataset_2", "dataset_1")

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
