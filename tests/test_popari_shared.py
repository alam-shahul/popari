from pathlib import Path

import numpy as np
import pytest
import torch

from popari import tl
from popari.model import Popari


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
def popari_with_neighbors(test_datapath, context):
    obj = Popari(
        K=10,
        lambda_Sigma_x_inv=1e-3,
        metagene_mode="shared",
        torch_context=context,
        initial_context=context,
        initialization_method="svd",
        dataset_path=test_datapath / "all_data.h5",
        verbose=4,
    )

    for iteration in range(1, 5):
        print(f"-----  Iteration {iteration} -----")
        obj.estimate_parameters()
        nll_metagenes = obj.base_view.parameter_optimizer.nll_metagenes()
        nll_spatial_affinities = obj.base_view.parameter_optimizer.nll_spatial_affinities()
        nll_sigma_yx = obj.base_view.parameter_optimizer.nll_sigma_yx()
        print(f"Metagene loss: {nll_metagenes}")
        print(f"Spatial affinity loss: {nll_spatial_affinities}")
        print(f"Sigma_yx loss: {nll_sigma_yx}")
        obj.estimate_weights()
        nll_embeddings = obj.base_view.embedding_optimizer.nll_embeddings()
        print(f"Embedding loss: {nll_embeddings}")
        print(f"Overall loss: {obj.base_view.nll()}")

    for dataset in obj.datasets:
        dataset.uns["multigroup_heatmap"] = {
            group_name: np.arange(4).reshape((2, 2)) for group_name in obj.metagene_groups
        }

    if not (test_datapath / "trained_4_iterations.h5ad").exists():
        obj.save_results(test_datapath / "trained_4_iterations.h5ad")

    return obj


@pytest.fixture(scope="module")
def popari_with_leiden_initialization(context, test_datapath):
    replicate_names = [0, 1]
    _ = Popari(
        K=10,
        lambda_Sigma_x_inv=1e-3,
        metagene_mode="shared",
        torch_context=context,
        initial_context=context,
        dataset_path=test_datapath / "all_data.h5",
        replicate_names=replicate_names,
        verbose=4,
    )


def test_leiden_initialization(popari_with_leiden_initialization):
    pass


def test_Sigma_x_inv(popari_with_neighbors, test_datapath):
    Sigma_x_inv = (
        list(popari_with_neighbors.parameter_optimizer.spatial_affinity_state.values())[0].cpu().detach().numpy()
    )
    # np.save("outputs/Sigma_x_inv_shared.npy", Sigma_x_inv)
    test_Sigma_x_inv = np.load(test_datapath / "outputs/Sigma_x_inv_shared.npy")
    assert np.allclose(test_Sigma_x_inv, Sigma_x_inv)


def test_M(popari_with_neighbors, test_datapath):
    M_bar = popari_with_neighbors.parameter_optimizer.metagene_state.metagenes.detach().cpu().numpy()
    # np.save(test_datapath / "outputs/M_bar_shared.npy", M_bar)
    test_M = np.load(test_datapath / "outputs/M_bar_shared.npy")
    assert np.allclose(test_M, M_bar)


def test_X_0(popari_with_neighbors, test_datapath):
    X_0 = popari_with_neighbors.embedding_optimizer.embedding_state["0"].detach().cpu().numpy()
    # np.save(test_datapath / "outputs/X_0_shared.npy", X_0)
    test_X_0 = np.load(test_datapath / "outputs/X_0_shared.npy")
    assert np.allclose(test_X_0, X_0)


def test_louvain_clustering(popari_with_neighbors):
    tl.preprocess_embeddings(popari_with_neighbors, joint=True)
    tl.leiden(popari_with_neighbors, joint=True, target_clusters=8)
    tl.compute_ari_scores(popari_with_neighbors, labels="cell_type", predictions="leiden")
    tl.compute_silhouette_scores(popari_with_neighbors, labels="cell_type", embeddings="normalized_X")
    tl.evaluate_classification_task(popari_with_neighbors, labels="cell_type", embeddings="normalized_X", joint=False)
    tl.evaluate_classification_task(popari_with_neighbors, labels="cell_type", embeddings="normalized_X", joint=True)

    expected_aris = [0.7638266995561214, 0.8126115509883821]
    for expected_ari, dataset in zip(expected_aris, popari_with_neighbors.datasets):
        assert expected_ari == pytest.approx(dataset.uns["ari"])

    expected_silhouettes = [0.30699036262154916, 0.34466040612221843]
    for expected_silhouette, dataset in zip(expected_silhouettes, popari_with_neighbors.datasets):
        print(f"Silhouette score: {dataset.uns['silhouette']}")
        assert expected_silhouette == pytest.approx(dataset.uns["silhouette"])
