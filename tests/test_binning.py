from pathlib import Path

import numpy as np
import pytest
import scanpy as sc
import torch

from popari._binning_utils import GridDownsampler, PartitionDownsampler
from popari.io import load_anndata
from popari.model import Popari, load_trained_model


@pytest.fixture(scope="module")
def test_datapath():
    return Path("tests/test_data/synthetic_dataset")


@pytest.fixture(scope="module")
def context():
    context = {
        "device": "cuda:0",
        "dtype": torch.float64,
    }
    return context


@pytest.fixture(scope="module")
def trained_model(test_datapath):
    trained_model = load_trained_model(test_datapath / "trained_4_iterations.h5ad")

    return trained_model


def test_grid_binning(trained_model, test_datapath):
    binned_datasets = []

    downsampler = GridDownsampler()
    for index, dataset in enumerate(trained_model.datasets):
        binned_dataset_name = f"{dataset.name}_level_0"
        bin_assignments_key = f"bin_assignments_{binned_dataset_name}"
        binned_dataset, _ = downsampler.downsample(
            dataset,
            bin_assignments_key=bin_assignments_key,
            chunks=4,
            downsample_rate=0.5,
        )
        if not (test_datapath / f"grid_binned_dataset_{index}.h5ad").exists():
            binned_dataset.write_h5ad(test_datapath / f"grid_binned_dataset_{index}.h5ad")

        saved_dataset = sc.read_h5ad(test_datapath / f"grid_binned_dataset_{index}.h5ad")
        assert np.allclose(
            binned_dataset.obsm[bin_assignments_key].toarray(),
            saved_dataset.obsm[bin_assignments_key].toarray(),
        )


def test_partition_binning(trained_model, test_datapath):
    binned_datasets = []

    downsampler = PartitionDownsampler()
    for index, dataset in enumerate(trained_model.datasets):
        binned_dataset_name = f"{dataset.name}_level_0"
        bin_assignments_key = f"bin_assignments_{binned_dataset_name}"
        binned_dataset, _ = downsampler.downsample(
            dataset,
            bin_assignments_key=bin_assignments_key,
            downsample_rate=0.5,
            adjacency_list_key="adjacency_list",
        )
        if not (test_datapath / f"partition_binned_dataset_{index}.h5ad").exists():
            binned_dataset.write_h5ad(test_datapath / f"partition_binned_dataset_{index}.h5ad")

        saved_dataset = sc.read_h5ad(test_datapath / f"partition_binned_dataset_{index}.h5ad")
        assert np.allclose(
            binned_dataset.obsm[bin_assignments_key].toarray(),
            saved_dataset.obsm[bin_assignments_key].toarray(),
        )


@pytest.fixture(scope="module")
def hierarchical_model(test_datapath, context):
    replicate_names = ["0", "1"]
    hierarchical_parameters = {
        "K": 10,
        "lambda_Sigma_x_inv": 1e-3,
        "torch_context": context,
        "initial_context": context,
        "initialization_method": "svd",
        "spatial_affinity_mode": "differential lookup",
        "dataset_path": test_datapath / "all_data.h5",
        "replicate_names": replicate_names,
        "hierarchical_levels": 2,
        "binning_downsample_rate": 0.5,
        "superresolution_lr": 1e-2,
        "verbose": 4,
    }

    obj = Popari(**hierarchical_parameters)

    # TODO: add test for tl.propagate_labels

    return obj


#
#
# @pytest.fixture(scope="module")
# def leiden_initialized_model(test_datapath):
#     replicate_names = [0, 1]
#     obj = Popari(
#         K=10,
#         lambda_Sigma_x_inv=1e-3,
#         torch_context=context,
#         initial_context=context,
#         spatial_affinity_mode="differential lookup",
#         dataset_path=test_datapath / "all_data.h5",
#         replicate_names=replicate_names,
#         hierarchical_levels=2,
#         binning_downsample_rate=0.5,
#         superresolution_lr=1e-2,
#         verbose=4,
#     )
#
#     return obj
#
#
@pytest.fixture(scope="module")
def coarser_model(test_datapath, context):
    replicate_names = [0, 1]
    _ = Popari(
        K=10,
        lambda_Sigma_x_inv=1e-3,
        torch_context=context,
        initial_context=context,
        initialization_method="svd",
        spatial_affinity_mode="differential lookup",
        dataset_path=test_datapath / "all_data.h5",
        replicate_names=replicate_names,
        hierarchical_levels=3,
        binning_downsample_rate=0.2,
        superresolution_lr=1e-2,
        verbose=4,
    )


# def test_hierarchical_svd_initialization(hierarchical_model):
#     pass
#
#
# def test_hierarchical_leiden_initialization(leiden_initialized_model):
#     pass
#
#
def test_coarser_initialization(coarser_model):
    pass


#
#
# def test_nll_hierarchical(hierarchical_model):
#     nll = hierarchical_model.nll()
#     level_0_nll = hierarchical_model.nll(level=0)


@pytest.fixture(scope="module")
def superresolved_model(hierarchical_model, test_datapath):
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

    if not (test_datapath / "outputs" / "superresolved_results").is_dir():
        # if True:
        hierarchical_model.save_results(test_datapath / "outputs" / "superresolved_results", ignore_raw_data=False)

    if not (test_datapath / "outputs" / "untrainable_superresolved_results").is_dir():
        hierarchical_model.save_results(
            test_datapath / "outputs" / "untrainable_superresolved_results",
            ignore_raw_data=True,
        )

    return hierarchical_model


@pytest.fixture(scope="module")
def loaded_model(superresolved_model, test_datapath, context):
    reloaded_model = load_trained_model(
        test_datapath / "outputs" / "superresolved_results",
        context=context,
    )

    return reloaded_model


@pytest.fixture(scope="module")
def untrainable_model(superresolved_model, test_datapath, context):
    reloaded_model = load_trained_model(
        test_datapath / "outputs" / "untrainable_superresolved_results",
        context=context,
        verbose=6,
    )

    raw_datasets, _ = load_anndata(test_datapath / "all_data.h5")
    reloaded_model._reload_expression(raw_datasets)

    return reloaded_model


def test_hierarchical_load(loaded_model):
    pass
    # Try superresolution on reloaded model
    for level in range(loaded_model.hierarchical_levels - 2, -1, -1):
        view = loaded_model.hierarchy[level]

        for dataset in view.datasets:
            assert dataset.X.sum() > 0

        loss = view._superresolve_embeddings(n_epochs=10, tol=1e-8)
        assert not np.any(np.isnan(loss))


def test_untrainable_superresolve(superresolved_model, untrainable_model):
    pass
    # Try superresolution on reloaded model
    for level in range(untrainable_model.hierarchical_levels - 2, -1, -1):
        view = untrainable_model.hierarchy[level]

        for dataset in view.datasets:
            assert dataset.X.sum() > 0

        loss = view._superresolve_embeddings(n_epochs=10, tol=1e-8)
        assert not np.any(np.isnan(loss))


def test_superresolution(superresolved_model, loaded_model, test_datapath):
    for dataset_index, (dataset, loaded_dataset) in enumerate(
        zip(superresolved_model.hierarchy[0].datasets, loaded_model.hierarchy[0].datasets),
    ):
        print((dataset.obsm["X"] - loaded_dataset.obsm["X"]).max())
        assert np.allclose(dataset.obsm["X"], loaded_dataset.obsm["X"])

    # loaded_model.superresolve(n_epochs=10, tol=1e-8)

    superresolved_model.nll(level=0, use_spatial=True)
