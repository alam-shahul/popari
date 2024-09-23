import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm, trange

from popari.model import Popari
from popari.util import clustering_louvain_nclust


@pytest.fixture(scope="module")
def popari_with_neighbors():
    data_dir = Path("tests/test_data/multigroup")
    K = 11
    lambda_Sigma_x_inv = 1e-4  # Spatial affinity regularization hyperparameter
    torch_context = dict(
        device="cuda:0",
        dtype=torch.float64,
    )  # Context for PyTorch tensor instantiation
    replicate_names = ["top", "bottom", "central"]
    spatial_affinity_groups = {
        "vertical_gradient": ["top", "bottom"],
        "central": ["central"],
    }

    model = Popari(
        K=K,
        lambda_Sigma_x_inv=lambda_Sigma_x_inv,
        dataset_path=data_dir / "example.h5ad",
        replicate_names=replicate_names,
        spatial_affinity_groups=spatial_affinity_groups,
        spatial_affinity_mode="differential lookup",
        lambda_Sigma_bar=1,
        torch_context=torch_context,
        initial_context=torch_context,
    )

    for iteration in range(1, 5):
        model.estimate_parameters()
        model.estimate_weights()

    return model


def test_Sigma_x_inv(popari_with_neighbors):
    Sigma_x_inv = (
        list(popari_with_neighbors.parameter_optimizer.spatial_affinity_state.values())[
            0
        ]
        .cpu()
        .detach()
        .numpy()
    )
    # np.save("tests/test_data/multigroup/outputs/Sigma_x_inv_spatial_differential.npy", Sigma_x_inv)
    test_Sigma_x_inv = np.load(
        "tests/test_data/multigroup/outputs/Sigma_x_inv_spatial_differential.npy",
    )
    assert np.allclose(test_Sigma_x_inv, Sigma_x_inv)


def test_M(popari_with_neighbors):
    M_bar = (
        popari_with_neighbors.parameter_optimizer.metagene_state.metagenes.detach()
        .cpu()
        .numpy()
    )
    # np.save("tests/test_data/multigroup/outputs/M_bar_spatial_differential.npy", M_bar)
    test_M = np.load(
        "tests/test_data/multigroup/outputs/M_bar_spatial_differential.npy",
    )
    assert np.allclose(test_M, M_bar)


def test_X_0(popari_with_neighbors):
    X_0 = (
        popari_with_neighbors.embedding_optimizer.embedding_state["top"]
        .detach()
        .cpu()
        .numpy()
    )
    # np.save("tests/test_data/multigroup/outputs/X_0_spatial_differential.npy", X_0)
    test_X_0 = np.load(
        "tests/test_data/multigroup/outputs/X_0_spatial_differential.npy",
    )
    assert np.allclose(test_X_0, X_0)
