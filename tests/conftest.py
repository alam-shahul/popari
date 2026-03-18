from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import torch
from pytest import fixture
from scipy.sparse import csr_array

from popari._hierarchical_view import HierarchicalView, Hierarchy
from popari._popari_dataset import PopariNamespace
from popari.model import Popari


@dataclass
class MockPopari(Popari):
    K: int
    datasets: list[ad.AnnData]
    verbose: bool = True

    def __post_init__(self):
        hierarchical_view_kwargs = {
            "K": self.K,
            "verbose": self.verbose,
        }
        self.base_view = MockHierarchicalView(self.datasets, level=0)

        self.hierarchy = MockHierarchy(
            downsampling_method="grid",
            base_view=self.base_view,
        )


@dataclass
class MockHierarchicalView(HierarchicalView):
    datasets: list[ad.AnnData]
    level: int


@dataclass
class MockHierarchy(Hierarchy):
    downsampling_method: str
    base_view: MockHierarchicalView


@fixture(scope="module")
def test_datapath():
    return Path("tests/test_data/synthetic_dataset")


@fixture(scope="module")
def context():
    context = {
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "dtype": torch.float64,
    }
    return context


@fixture(scope="module")
def float32_context():
    context = {
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "dtype": torch.float32,
    }
    return context


@fixture(scope="module")
def shared_model(test_datapath, context):
    obj = Popari(
        K=10,
        lambda_Sigma_x_inv=1e-3,
        metagene_mode="shared",
        torch_context=context,
        initial_context=context,
        initialization_method="svd",
        dataset_path=test_datapath / "all_data.h5",
        verbose=1,
    )

    return obj


@fixture
def rng():
    seed = 42
    generator = np.random.default_rng(seed)
    return generator


@fixture
def mock_datasets(rng):
    # num_rows = 20
    # num_cols = 10
    # x, y = np.meshgrid(np.arange(num_rows), np.arange(num_cols))
    # num_cells = num_rows * num_cols
    # coordinates =

    num_cells = 200
    num_genes = 100
    num_replicates = 2

    datasets = []
    for index in range(num_replicates):
        dataset = ad.AnnData(
            X=rng.random((num_cells, num_genes)),
        )
        dataset.X = csr_array(dataset.X)
        dataset.obs["batch"] = str(index)
        dataset.obsm["spatial"] = rng.random((num_cells, 2))

        dataset.popari.name()
        dataset.popari.compute_spatial_neighbors()
        datasets.append(dataset)

    return datasets


@fixture
def shared_mock_model(mock_datasets):
    obj = Popari(
        K=10,
        datasets=mock_datasets,
        replicate_names=list(range(len(mock_datasets))),
        verbose=1,
    )

    return obj
