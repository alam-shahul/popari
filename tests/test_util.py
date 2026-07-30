import anndata as ad
import awkward as ak
import numpy as np
import pandas as pd
import pytest
import torch

from popari.util import project2simplex, project2simplex_, smooth_labels, spatially_smooth_feature


@pytest.mark.baseline
def test_project2simplex_projects_columns_to_simplex():
    projection_input = torch.tensor(
        [[0.2, -1.0, 3.0], [1.2, 0.5, -2.0], [0.8, 2.5, 1.0]],
        dtype=torch.float32,
    )

    projection_output = project2simplex(projection_input, dim=0)

    assert torch.all(projection_output >= 0)
    assert torch.allclose(projection_output.sum(dim=0), torch.ones(3), atol=1e-4)


@pytest.mark.baseline
def test_project2simplex_inplace_matches_functional():
    projection_input = torch.tensor(
        [[0.2, -1.0, 3.0], [1.2, 0.5, -2.0], [0.8, 2.5, 1.0]],
        dtype=torch.float32,
    )

    projected = project2simplex(projection_input, dim=1)
    projected_inplace = project2simplex_(projection_input, dim=1)

    assert torch.allclose(projected, projected_inplace, atol=1e-5)


@pytest.mark.baseline
def test_spatial_smoothing_propagates_across_rounds():
    labels = np.array(["A", "B", "A", "A"])
    adjacency_list = np.asarray([[1], [0, 2], [1, 3], [2]], dtype=object)

    one_round = spatially_smooth_feature(labels, adjacency_list, max_smoothing_rounds=1)
    default = spatially_smooth_feature(labels, adjacency_list)
    converged = spatially_smooth_feature(labels, adjacency_list, max_smoothing_rounds=10)

    assert np.array_equal(one_round, ["B", "A", "A", "A"])
    assert np.array_equal(default, one_round)
    assert np.array_equal(converged, ["A", "A", "A", "A"])


@pytest.mark.baseline
def test_smooth_labels_stores_categorical_annotation():
    dataset = ad.AnnData(np.ones((3, 1)))
    dataset.obs["leiden"] = pd.Categorical(["A", "A", "B"])
    dataset.obsm["adjacency_list"] = ak.Array([[1], [0, 2], [1]])

    smooth_labels(dataset)

    assert isinstance(dataset.obs["smoothed_leiden"].dtype, pd.CategoricalDtype)
