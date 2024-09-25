from pathlib import Path

import numpy as np
import pytest
import scanpy as sc
import squidpy as sq
import torch

from popari import pl, tl
from popari._dataset_utils import _spatial_binning
from popari.io import load_anndata, save_anndata
from popari.model import Popari, load_trained_model


@pytest.fixture(scope="module")
def test_datapath():
    return Path("tests/test_data/synthetic_dataset")


@pytest.fixture(scope="module")
def trained_model(test_datapath):
    replicate_names = [0, 1]
    trained_model = load_trained_model(test_datapath / "trained_4_iterations.h5ad")

    return trained_model


def test_binning(trained_model, test_datapath):
    binned_datasets = []
    for index, dataset in enumerate(trained_model.datasets):
        binned_dataset = _spatial_binning(dataset, chunks=4, downsample_rate=0.5)
        print(f"{dataset.name=}")
        print(f"{binned_dataset.name=}")
        if not (test_datapath / f"binned_dataset_{index}.h5ad").exists():
            binned_dataset.write_h5ad(test_datapath / f"binned_dataset_{index}.h5ad")

        print(binned_dataset.obsm)
        bin_assignments_key = f"bin_assignments_{binned_dataset.name}"
        saved_dataset = sc.read_h5ad(test_datapath / f"binned_dataset_{index}.h5ad")
        print(saved_dataset.obsm)
        assert np.allclose(
            binned_dataset.obsm[bin_assignments_key].toarray(),
            saved_dataset.obsm[bin_assignments_key].toarray(),
        )


@pytest.fixture(scope="module")
def hierarchical_model():
    path2dataset = Path("tests/test_data/synthetic_dataset")
    replicate_names = [0, 1]
    obj = Popari(
        K=10,
        lambda_Sigma_x_inv=1e-3,
        torch_context=dict(device="cuda:0", dtype=torch.float64),
        initial_context=dict(device="cuda:0", dtype=torch.float64),
        initialization_method="svd",
        spatial_affinity_mode="differential lookup",
        dataset_path=path2dataset / "all_data.h5",
        replicate_names=replicate_names,
        hierarchical_levels=2,
        binning_downsample_rate=0.5,
        superresolution_lr=1e-2,
        verbose=4,
    )

    return obj


@pytest.fixture(scope="module")
def leiden_initialized_model():
    path2dataset = Path("tests/test_data/synthetic_dataset")
    replicate_names = [0, 1]
    obj = Popari(
        K=10,
        lambda_Sigma_x_inv=1e-3,
        torch_context=dict(device="cuda:0", dtype=torch.float64),
        initial_context=dict(device="cuda:0", dtype=torch.float64),
        spatial_affinity_mode="differential lookup",
        dataset_path=path2dataset / "all_data.h5",
        replicate_names=replicate_names,
        hierarchical_levels=2,
        binning_downsample_rate=0.5,
        superresolution_lr=1e-2,
        verbose=4,
    )

    return obj


@pytest.fixture(scope="module")
def coarser_model():
    path2dataset = Path("tests/test_data/synthetic_dataset")
    replicate_names = [0, 1]
    obj = Popari(
        K=10,
        lambda_Sigma_x_inv=1e-3,
        torch_context=dict(device="cuda:0", dtype=torch.float64),
        initial_context=dict(device="cuda:0", dtype=torch.float64),
        initialization_method="svd",
        spatial_affinity_mode="differential lookup",
        dataset_path=path2dataset / "all_data.h5",
        replicate_names=replicate_names,
        hierarchical_levels=3,
        binning_downsample_rate=0.2,
        superresolution_lr=1e-2,
        verbose=4,
    )


def test_hierarchical_leiden_initialization(hierarchical_model):
    pass


def test_hierarchical_svd_initialization(leiden_initialized_model):
    pass


def test_coarser_initialization(coarser_model):
    pass


def test_nll_hierarchical(hierarchical_model):
    nll = hierarchical_model.nll()
    level_0_nll = hierarchical_model.nll(level=0)


def test_superresolution(hierarchical_model):
    path2dataset = Path("tests/test_data/synthetic_dataset")
    for iteration in range(1, 2):
        print(f"-----  Iteration {iteration} -----")
        hierarchical_model.estimate_parameters()
        nll_metagenes = hierarchical_model.base_view.parameter_optimizer.nll_metagenes()
        nll_spatial_affinities = hierarchical_model.base_view.parameter_optimizer.nll_spatial_affinities()
        nll_sigma_yx = hierarchical_model.base_view.parameter_optimizer.nll_sigma_yx()
        print(f"Metagene loss: {nll_metagenes}")
        print(f"Spatial affinity loss: {nll_spatial_affinities}")
        print(f"Sigma_yx loss: {nll_sigma_yx}")
        hierarchical_model.estimate_weights()
        nll_embeddings = hierarchical_model.base_view.embedding_optimizer.nll_embeddings()
        print(f"Embedding loss: {nll_embeddings}")
        print(f"Overall loss: {hierarchical_model.base_view.nll()}")

    hierarchical_model.superresolve(n_epochs=10)

    hierarchical_model.superresolve(n_epochs=10, tol=1e-8)
    hierarchical_model.set_superresolution_lr(new_lr=1e-1)
    hierarchical_model.superresolve(n_epochs=10, tol=1e-8)

    hierarchical_model.superresolve(n_epochs=10, tol=1e-8, use_manual_gradients=True)
    hierarchical_model.nll(level=1, use_spatial=True)
    hierarchical_model.nll(level=0, use_spatial=True)

    # TODO: add check for superresolution results
    hierarchical_model.save_results(path2dataset / "superresolved_results", ignore_raw_data=False)
    hierarchical_model.nll(level=0, use_spatial=True)


def test_hierarchical_load():
    path2dataset = Path("tests/test_data/synthetic_dataset")
    reloaded_model = load_trained_model(
        path2dataset / "superresolved_results",
        context=dict(device="cuda:0", dtype=torch.float64),
    )
    reloaded_model.superresolve(n_epochs=10, tol=1e-8)
