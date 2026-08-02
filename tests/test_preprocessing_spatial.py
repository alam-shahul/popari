import anndata as ad
import numpy as np
import pandas as pd

from popari import pp
from popari._sample_axis import SampleAxis


def test_compute_spatial_neighbors_builds_one_block_diagonal_graph():
    coordinates = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ],
    )
    dataset = ad.AnnData(
        X=np.ones((8, 2)),
        obs=pd.DataFrame(
            {"sample": ["a", "b"] * 4},
            index=[f"cell_{index}" for index in range(8)],
        ),
    )
    dataset.var_names = ["gene_0", "gene_1"]
    dataset.obsm["spatial"] = coordinates

    result = pp.compute_spatial_neighbors(dataset, sample_key="sample")

    assert result is None
    assert isinstance(dataset.obs["sample"].dtype, pd.CategoricalDtype)
    axis = SampleAxis.from_anndata(dataset, sample_key="sample")
    source, target = dataset.obsp["adjacency_matrix"].nonzero()
    assert np.all(axis.codes[source] == axis.codes[target])
    assert dataset.obsp["adjacency_matrix"].nnz > 0
    assert len(dataset.obsm["adjacency_list"]) == dataset.n_obs
