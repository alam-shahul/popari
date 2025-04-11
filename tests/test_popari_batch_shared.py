from pathlib import Path

import numpy as np
import pytest
import torch

from popari import tl
from popari.model import Popari
from popari.train import BatchBlendTrainer, TrainParameters


@pytest.fixture(scope="module")
def popari_with_neighbors(dataset_path, context, shared_model):
    obj = shared_model

    iterations = 1
    train_parameters = TrainParameters(
        nmf_iterations=0,
        iterations=iterations,
        savepath=(dataset_path / f"trained_{iterations}_iterations_batch.h5ad"),
    )

    trainer = BatchBlendTrainer(
        parameters=train_parameters,
        model=obj,
        verbose=True,
    )

    trainer.train()

    for dataset in obj.datasets:
        dataset.uns["multigroup_heatmap"] = {
            group_name: np.arange(4).reshape((2, 2)) for group_name in obj.metagene_groups
        }

    # if not (dataset_path / "trained_4_iterations.h5ad").exists():
    #     obj.save_results(dataset_path / "trained_4_iterations.h5ad")

    trainer.save_results()

    return obj


@pytest.fixture(scope="module")
def popari_with_leiden_initialization(context, mock_datasets):
    replicate_names = ["mock_1", "mock_2"]
    _ = Popari(
        K=2,
        lambda_Sigma_x_inv=1e-3,
        metagene_mode="shared",
        torch_context=context,
        initial_context=context,
        datasets=mock_datasets,
        replicate_names=replicate_names,
        verbose=4,
        batch_effect_correction=True,
    )


def test_leiden_initialization(popari_with_leiden_initialization):
    pass


def test_Sigma_x_inv(popari_with_neighbors):
    Sigma_x_inv = (
        list(popari_with_neighbors.parameter_optimizer.spatial_affinity_state.values())[0].cpu().detach().numpy()
    )


def test_M(popari_with_neighbors):
    M_bar = popari_with_neighbors.parameter_optimizer.metagene_state.metagenes.detach().cpu().numpy()


def test_X_0(popari_with_neighbors):
    X_0 = popari_with_neighbors.embedding_optimizer.embedding_state["mock_1"].detach().cpu().numpy()


def test_louvain_clustering(popari_with_neighbors):
    tl.preprocess_embeddings(popari_with_neighbors)
    tl.leiden(popari_with_neighbors, joint=True, target_clusters=8)
    tl.compute_ari_scores(popari_with_neighbors, labels="cell_type", predictions="leiden")
    tl.compute_silhouette_scores(popari_with_neighbors, labels="cell_type", embeddings="normalized_X")
    tl.evaluate_classification_task(
        popari_with_neighbors,
        labels="cell_type",
        embeddings="normalized_X",
        n_neighbors=5,
        joint=False,
    )
    tl.evaluate_classification_task(
        popari_with_neighbors,
        labels="cell_type",
        embeddings="normalized_X",
        n_neighbors=5,
        joint=True,
    )

    expected_aris = [0.1798389126604581, 0.06944444444444445]
    for expected_ari, dataset in zip(expected_aris, popari_with_neighbors.datasets):
        print(f"ARI score: {dataset.uns['ari']}")
        assert expected_ari == pytest.approx(dataset.uns["ari"], abs=1e-3)

    expected_silhouettes = [-0.011309038681332075, -0.06082095978987505]
    for expected_silhouette, dataset in zip(expected_silhouettes, popari_with_neighbors.datasets):
        print(f"Silhouette score: {dataset.uns['silhouette']}")
        assert expected_silhouette == pytest.approx(dataset.uns["silhouette"], abs=1e-3)
