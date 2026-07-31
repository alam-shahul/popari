import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_array

from popari._sample_axis import SampleAxis


def _canonical_adata(labels=("sample_b", "sample_a", "sample_b", "sample_a")):
    adata = ad.AnnData(X=np.ones((len(labels), 3)))
    adata.obs_names = [f"cell_{index}" for index in range(len(labels))]
    adata.var_names = [f"gene_{index}" for index in range(3)]
    adata.obs["batch"] = pd.Categorical(labels, categories=["sample_a", "sample_b"])
    rows = np.array([0, 1, 2, 3])
    columns = np.array([2, 3, 0, 1])
    adata.obsp["adjacency_matrix"] = csr_array(
        (np.ones(len(rows)), (rows, columns)),
        shape=(len(labels), len(labels)),
    )
    return adata


def test_sample_axis_uses_categorical_order_and_indexes_observations():
    axis = SampleAxis.from_anndata(_canonical_adata())

    assert axis.names == ("sample_a", "sample_b")
    assert axis.position("sample_b") == 1
    np.testing.assert_array_equal(axis.codes, [1, 0, 1, 0])
    np.testing.assert_array_equal(axis.indices("sample_a"), [1, 3])
    np.testing.assert_array_equal(axis.mask("sample_b"), [True, False, True, False])


def test_sample_axis_rejects_unknown_samples():
    axis = SampleAxis.from_anndata(_canonical_adata())

    with pytest.raises(KeyError, match="Unknown sample"):
        axis.indices("missing")


def test_sample_axis_can_index_samples_before_graph_construction():
    adata = _canonical_adata()
    del adata.obsp["adjacency_matrix"]

    axis = SampleAxis.from_anndata(adata)

    assert axis.names == ("sample_a", "sample_b")
    np.testing.assert_array_equal(axis.indices("sample_a"), [1, 3])


@pytest.mark.parametrize(
    ("mutation", "error", "message"),
    [
        (lambda adata: adata.obs.__delitem__("batch"), KeyError, "Missing sample column"),
        (
            lambda adata: adata.obs.__setitem__("batch", adata.obs["batch"].astype(str)),
            TypeError,
            "categorical dtype",
        ),
        (
            lambda adata: adata.obs.__setitem__(
                "batch",
                pd.Categorical(["sample_a", None, "sample_b", "sample_a"]),
            ),
            ValueError,
            "missing sample labels",
        ),
        (
            lambda adata: adata.obs.__setitem__(
                "batch",
                adata.obs["batch"].cat.add_categories(["unused"]),
            ),
            ValueError,
            "unused categories",
        ),
        (lambda adata: setattr(adata, "obs_names", ["a", "a", "b", "c"]), ValueError, "Observation names"),
        (lambda adata: setattr(adata, "var_names", ["a", "a", "b"]), ValueError, "Variable names"),
    ],
)
def test_sample_axis_validates_schema(mutation, error, message):
    adata = _canonical_adata()
    mutation(adata)

    with pytest.raises(error, match=message):
        SampleAxis.from_anndata(adata)


def test_namespace_rejects_missing_spatial_graph():
    adata = _canonical_adata()
    del adata.obsp["adjacency_matrix"]

    with pytest.raises(KeyError, match="Missing spatial graph"):
        adata.popari.validate_spatial_graph()


def test_namespace_rejects_cross_sample_edges():
    adata = _canonical_adata()
    adata.obsp["adjacency_matrix"] = csr_array(
        (np.ones(2), ([0, 1], [1, 0])),
        shape=(adata.n_obs, adata.n_obs),
    )

    with pytest.raises(ValueError, match="cross-sample edges"):
        adata.popari.validate_spatial_graph()
