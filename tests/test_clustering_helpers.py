import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_array

from popari.analysis.clustering import smooth_labels, spatially_smooth_feature


@pytest.mark.baseline
def test_spatial_smoothing_propagates_across_rounds():
    labels = np.array(["A", "B", "A", "A"])
    adjacency = csr_array(
        ([1] * 6, ([0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2])),
        shape=(4, 4),
    )

    one_round = spatially_smooth_feature(labels, adjacency, max_smoothing_rounds=1)
    default = spatially_smooth_feature(labels, adjacency)
    converged = spatially_smooth_feature(labels, adjacency, max_smoothing_rounds=10)

    assert np.array_equal(one_round, ["B", "A", "A", "A"])
    assert np.array_equal(default, one_round)
    assert np.array_equal(converged, ["A", "A", "A", "A"])


@pytest.mark.baseline
def test_smooth_labels_stores_categorical_annotation():
    dataset = ad.AnnData(np.ones((3, 1)))
    dataset.obs["leiden"] = pd.Categorical(["A", "A", "B"])
    dataset.obsp["adjacency_matrix"] = csr_array(
        ([1] * 4, ([0, 1, 1, 2], [1, 0, 2, 1])),
        shape=(3, 3),
    )

    smooth_labels(dataset)

    assert isinstance(dataset.obs["smoothed_leiden"].dtype, pd.CategoricalDtype)
