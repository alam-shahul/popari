from pathlib import Path

import numpy as np
import pytest
import torch

from popari import tl
from popari.model import Popari
from popari.train import Trainer, TrainParameters


@pytest.fixture(scope="module")
def popari_with_neighbors(test_datapath, context, shared_model):
    obj = shared_model

    iterations = 4
    train_parameters = TrainParameters(
        nmf_iterations=0,
        iterations=iterations,
        savepath=(test_datapath / f"trained_{iterations}_iterations.h5ad"),
    )

    trainer = Trainer(
        parameters=train_parameters,
        model=obj,
        verbose=True,
        batch_effect_correction=True,
    )

    trainer.train()

    for dataset in obj.datasets:
        dataset.uns["multigroup_heatmap"] = {
            group_name: np.arange(4).reshape((2, 2)) for group_name in obj.metagene_groups
        }

    if not (test_datapath / "trained_4_iterations.h5ad").exists():
        obj.save_results(test_datapath / "trained_4_iterations.h5ad")

    trainer.save_results()

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
        # dataset_path=test_datapath / "processed_dataset.h5ad",
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
    # np.save(test_datapath / "outputs/Sigma_x_inv_shared_batch.npy", Sigma_x_inv)
    test_Sigma_x_inv = np.load(test_datapath / "outputs/Sigma_x_inv_shared_batch.npy")
    assert np.allclose(test_Sigma_x_inv, Sigma_x_inv, atol=1e-2)


def test_M(popari_with_neighbors, test_datapath):
    M_bar = popari_with_neighbors.parameter_optimizer.metagene_state.metagenes.detach().cpu().numpy()
    # np.save(test_datapath / "outputs/M_bar_shared_batch.npy", M_bar)
    test_M = np.load(test_datapath / "outputs/M_bar_shared_batch.npy")
    assert np.allclose(test_M, M_bar, atol=1e-2)


def test_X_0(popari_with_neighbors, test_datapath):
    # X_0 = popari_with_neighbors.embedding_optimizer.embedding_state["progenitor_0"].detach().cpu().numpy()
    X_0 = popari_with_neighbors.embedding_optimizer.embedding_state["0"].detach().cpu().numpy()
    # np.save(test_datapath / "outputs/X_0_shared_batch.npy", X_0)
    test_X_0 = np.load(test_datapath / "outputs/X_0_shared_batch.npy")
    assert np.allclose(test_X_0, X_0, atol=1e-3)


def test_B(popari_with_neighbors, test_datapath):
    B = np.array(
        [
            b.detach().cpu().numpy()
            for b in popari_with_neighbors.batch_effect_optimizer.batch_effect_state.batch_effects
        ],
    )
    # np.save(test_datapath / "outputs/B_shared_batch.npy", B)
    test_B = np.load(test_datapath / "outputs/B_shared_batch.npy")
    assert np.allclose(test_B, B, atol=1e-3)


def test_sigma_yx(popari_with_neighbors, test_datapath):
    sigma_yx = popari_with_neighbors.parameter_optimizer.sigma_yxs
    # np.save(test_datapath / "outputs/sigma_yx_shared_batch.npy", sigma_yx)
    test_sigma_yx = np.load(test_datapath / "outputs/sigma_yx_shared_batch.npy")
    assert np.allclose(test_sigma_yx, sigma_yx, atol=1e-3)
