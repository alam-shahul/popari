import pytest

import torch

import squidpy as sq

from popari import pl, tl

from popari.io import load_anndata, save_anndata
from popari.model import Popari, load_trained_model
from popari._dataset_utils import _spatial_binning

from pathlib import Path

@pytest.fixture(scope="module")
def trained_model():
    path2dataset = Path('tests/test_data/synthetic_500_100_20_15_0_0_i4')
    replicate_names=[0, 1]
    trained_model = load_trained_model(path2dataset / "trained_4_iterations.h5ad")

    return trained_model

def test_binning(trained_model):
    binned_datasets = []
    for dataset in trained_model.datasets:
        binned_dataset = _spatial_binning(dataset, chunks=4, downsample_rate=0.5)
        binned_datasets.append(binned_dataset)

@pytest.fixture(scope="module")
def hierarchical_model():
    path2dataset = Path('tests/test_data/synthetic_500_100_20_15_0_0_i4')
    replicate_names=[0, 1]
    obj = Popari(
        K=10, lambda_Sigma_x_inv=1e-3,
        metagene_mode="shared",
        torch_context=dict(device='cuda:0', dtype=torch.float64),
        initial_context=dict(device='cuda:0', dtype=torch.float64),
        initialization_method="svd",
        spatial_affinity_mode="differential lookup",
        dataset_path=path2dataset / "all_data.h5",
        replicate_names=replicate_names,
        hierarchical_levels=3,
        binning_downsample_rate=0.5,
        superresolution_lr=1e-2,
        verbose=4
    )

    return obj

@pytest.fixture(scope="module")
def coarser_model():
    path2dataset = Path('tests/test_data/synthetic_500_100_20_15_0_0_i4')
    replicate_names=[0, 1]
    obj = Popari(
        K=10, lambda_Sigma_x_inv=1e-3,
        metagene_mode="shared",
        torch_context=dict(device='cuda:0', dtype=torch.float64),
        initial_context=dict(device='cuda:0', dtype=torch.float64),
        initialization_method="svd",
        spatial_affinity_mode="differential lookup",
        dataset_path=path2dataset / "all_data.h5",
        replicate_names=replicate_names,
        hierarchical_levels=2,
        binning_downsample_rate=0.2,
        superresolution_lr=1e-2,
        verbose=4
    )
def test_hierarchical_initialization(hierarchical_model):
    pass

def test_coarser_initialization(coarser_model):
    pass

def test_nll_hierarchical(hierarchical_model):
    nll = hierarchical_model.nll()
    level_0_nll = hierarchical_model.nll(level=0)

def test_superresolution(hierarchical_model):
    path2dataset = Path('tests/test_data/synthetic_500_100_20_15_0_0_i4')
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

    hierarchical_model.superresolve()
    
    hierarchical_model.superresolve(n_epochs=100, tol=1e-8)
    hierarchical_model.set_superresolution_lr(new_lr=1e-1)
    hierarchical_model.superresolve(n_epochs=100, tol=1e-8)

    hierarchical_model.save_results(path2dataset / "superresolved_results.h5ad")

    reloaded_model = load_trained_model(path2dataset / "superresolved_results.h5ad")
