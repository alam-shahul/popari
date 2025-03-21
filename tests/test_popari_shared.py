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
    )

    trainer.train()

    for dataset in obj.datasets:
        dataset.uns["multigroup_heatmap"] = {
            group_name: np.arange(4).reshape((2, 2)) for group_name in obj.metagene_groups
        }

    # if not (test_datapath / "trained_4_iterations.h5ad").exists():
    #     obj.save_results(test_datapath / "trained_4_iterations.h5ad")

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
    # np.save(test_datapath / "outputs/Sigma_x_inv_shared.npy", Sigma_x_inv)
    test_Sigma_x_inv = np.load(test_datapath / "outputs/Sigma_x_inv_shared.npy")
    assert np.allclose(test_Sigma_x_inv, Sigma_x_inv, atol=1e-2)


def test_M(popari_with_neighbors, test_datapath):
    M_bar = popari_with_neighbors.parameter_optimizer.metagene_state.metagenes.detach().cpu().numpy()
    # np.save(test_datapath / "outputs/M_bar_shared.npy", M_bar)
    test_M = np.load(test_datapath / "outputs/M_bar_shared.npy")
    assert np.allclose(test_M, M_bar, atol=1e-2)


def test_X_0(popari_with_neighbors, test_datapath):
    X_0 = popari_with_neighbors.embedding_optimizer.embedding_state["0"].detach().cpu().numpy()
    # np.save(test_datapath / "outputs/X_0_shared.npy", X_0)
    test_X_0 = np.load(test_datapath / "outputs/X_0_shared.npy")
    assert np.allclose(test_X_0, X_0, atol=1e-3)


def test_louvain_clustering(popari_with_neighbors):
    tl.preprocess_embeddings(popari_with_neighbors)
    tl.leiden(popari_with_neighbors, joint=True, target_clusters=8)
    tl.compute_ari_scores(popari_with_neighbors, labels="cell_type", predictions="leiden")
    tl.compute_silhouette_scores(popari_with_neighbors, labels="cell_type", embeddings="normalized_X")
    tl.evaluate_classification_task(popari_with_neighbors, labels="cell_type", embeddings="normalized_X", joint=False)
    tl.evaluate_classification_task(popari_with_neighbors, labels="cell_type", embeddings="normalized_X", joint=True)

    expected_aris = [0.7999987761039686, 0.8330125889278888]
    for expected_ari, dataset in zip(expected_aris, popari_with_neighbors.datasets):
        print(f"ARI score: {dataset.uns['ari']}")
        assert expected_ari == pytest.approx(dataset.uns["ari"], abs=1e-3)

    expected_silhouettes = [0.3065469455513892, 0.34429891570227095]
    for expected_silhouette, dataset in zip(expected_silhouettes, popari_with_neighbors.datasets):
        print(f"Silhouette score: {dataset.uns['silhouette']}")
        assert expected_silhouette == pytest.approx(dataset.uns["silhouette"], abs=1e-3)
