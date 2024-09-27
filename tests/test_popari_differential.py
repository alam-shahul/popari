import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm, trange

from popari import tl
from popari.model import Popari
from popari.util import clustering_louvain_nclust


@pytest.fixture(scope="module")
def test_datapath():
    return Path("tests/test_data/synthetic_dataset")


@pytest.fixture(scope="module")
def popari_with_neighbors(test_datapath):
    replicate_names = [0, 1]
    obj = Popari(
        K=10,
        lambda_Sigma_x_inv=1e-5,
        metagene_mode="differential",
        lambda_M=0.5,
        torch_context=dict(device="cuda:0", dtype=torch.float64),
        initial_context=dict(device="cuda:0", dtype=torch.float64),
        dataset_path=test_datapath / "all_data.h5",
        replicate_names=replicate_names,
        verbose=2,
    )

    for iteration in range(1, 5):
        obj.estimate_parameters()
        obj.estimate_weights()

    obj.save_results(test_datapath / "trained_differential_metagenes_4_iterations.h5ad")

    return obj


def test_Sigma_x_inv(popari_with_neighbors, test_datapath):
    Sigma_x_inv = (
        list(popari_with_neighbors.parameter_optimizer.spatial_affinity_state.values())[0].detach().cpu().numpy()
    )
    np.save(test_datapath / "outputs/Sigma_x_inv_differential.npy", Sigma_x_inv)
    test_Sigma_x_inv = np.load(test_datapath / "outputs/Sigma_x_inv_differential.npy")
    assert np.allclose(test_Sigma_x_inv, Sigma_x_inv)


def test_M(popari_with_neighbors, test_datapath):
    first_group_name = list(popari_with_neighbors.metagene_groups.keys())[0]
    M_bar = popari_with_neighbors.parameter_optimizer.metagene_state.M_bar[first_group_name].detach().cpu().numpy()
    np.save(test_datapath / "outputs/M_bar_differential.npy", M_bar)
    test_M = np.load(test_datapath / "outputs/M_bar_differential.npy")
    assert np.allclose(test_M, M_bar)


def test_X_0(popari_with_neighbors, test_datapath):
    X_0 = popari_with_neighbors.embedding_optimizer.embedding_state["0"].detach().cpu().numpy()
    np.save(test_datapath / "outputs/X_0_differential.npy", X_0)
    test_X_0 = np.load(test_datapath / "outputs/X_0_differential.npy")
    assert np.allclose(test_X_0, X_0)


def test_louvain_clustering(popari_with_neighbors):
    print([dataset.obs["cell_type"] for dataset in popari_with_neighbors.datasets])
    tl.preprocess_embeddings(popari_with_neighbors, joint=True)
    tl.leiden(popari_with_neighbors, joint=True, target_clusters=8)
    tl.compute_ari_scores(popari_with_neighbors, labels="cell_type", predictions="leiden")
    tl.compute_silhouette_scores(popari_with_neighbors, labels="cell_type", embeddings="normalized_X")
    tl.evaluate_classification_task(popari_with_neighbors, labels="cell_type", embeddings="normalized_X", joint=False)
    tl.evaluate_classification_task(popari_with_neighbors, labels="cell_type", embeddings="normalized_X", joint=True)

    expected_aris = [0.6743909365208949, 0.7620951295366563]
    for expected_ari, dataset in zip(expected_aris, popari_with_neighbors.datasets):
        assert expected_ari == pytest.approx(dataset.uns["ari"])

    expected_silhouettes = [0.3575846057115737, 0.48696553363312534]
    for expected_silhouette, dataset in zip(expected_silhouettes, popari_with_neighbors.datasets):
        print(f"Silhouette score: {dataset.uns['silhouette']}")
        assert expected_silhouette == pytest.approx(dataset.uns["silhouette"])
