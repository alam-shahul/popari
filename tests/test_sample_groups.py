import anndata as ad
import numpy as np
import pandas as pd
import pytest

from popari import pp, tl


@pytest.fixture
def multisample_dataset():
    dataset = ad.AnnData(
        X=np.arange(12).reshape(4, 3),
        obs=pd.DataFrame(
            {"batch": ["sample_a", "sample_a", "sample_b", "sample_b"]},
            index=[f"cell_{index}" for index in range(4)],
        ),
    )
    dataset.uns["M"] = np.ones((3, 2))
    dataset.uns["Sigma_x_inv"] = {
        "sample_a": np.eye(2),
        "sample_b": np.full((2, 2), 3),
    }
    return dataset


def test_subset_samples_copies_groups_and_filters_parameters(multisample_dataset):
    subsets = pp.subset_samples(
        multisample_dataset,
        {
            "first": ["sample_a"],
            "both": ["sample_a", "sample_b"],
        },
    )

    assert subsets["first"].n_obs == 2
    assert subsets["first"].popari.name == "first"
    assert subsets["first"].obs["batch"].unique().tolist() == ["sample_a"]
    np.testing.assert_array_equal(subsets["first"].uns["M"], multisample_dataset.uns["M"])
    assert set(subsets["first"].uns["Sigma_x_inv"]) == {"sample_a"}
    assert subsets["both"].n_obs == 4
    np.testing.assert_array_equal(subsets["both"].uns["M"], multisample_dataset.uns["M"])
    assert "dataset_name" not in multisample_dataset.uns


@pytest.mark.parametrize(
    ("sample_groups", "message"),
    [
        ({"empty": []}, "is empty"),
        ({"unknown": ["sample_c"]}, "unknown samples"),
    ],
)
def test_subset_samples_rejects_invalid_groups(multisample_dataset, sample_groups, message):
    with pytest.raises(ValueError, match=message):
        pp.subset_samples(multisample_dataset, sample_groups)


def test_subset_samples_requires_sample_key(multisample_dataset):
    del multisample_dataset.obs["batch"]

    with pytest.raises(KeyError, match="Sample key"):
        pp.subset_samples(multisample_dataset, {"first": ["sample_a"]})


def test_aggregate_sample_matrices_supports_mean_and_median(multisample_dataset):
    sample_groups = {
        "first": ["sample_a"],
        "both": ["sample_a", "sample_b"],
    }

    means = tl.aggregate_sample_matrices(multisample_dataset, sample_groups, "Sigma_x_inv")
    medians = tl.aggregate_sample_matrices(
        multisample_dataset,
        sample_groups,
        "Sigma_x_inv",
        reduction="median",
    )

    np.testing.assert_array_equal(means["first"], np.eye(2))
    np.testing.assert_array_equal(means["both"], (np.eye(2) + 3) / 2)
    np.testing.assert_array_equal(medians["both"], (np.eye(2) + 3) / 2)


def test_aggregate_sample_matrices_rejects_missing_and_inconsistent_matrices(multisample_dataset):
    with pytest.raises(KeyError, match="sample_c"):
        tl.aggregate_sample_matrices(
            multisample_dataset,
            {"invalid": ["sample_a", "sample_c"]},
            "Sigma_x_inv",
        )

    multisample_dataset.uns["Sigma_x_inv"]["sample_b"] = np.ones((3, 3))
    with pytest.raises(ValueError, match="inconsistent shapes"):
        tl.aggregate_sample_matrices(
            multisample_dataset,
            {"invalid": ["sample_a", "sample_b"]},
            "Sigma_x_inv",
        )
