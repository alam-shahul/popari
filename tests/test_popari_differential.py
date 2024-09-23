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
    path2dataset = Path("tests/test_data/synthetic_dataset")
    replicate_names = [0, 1]
    obj = Popari(
        K=10,
        lambda_Sigma_x_inv=1e-5,
        metagene_mode="differential",
        lambda_M=0.5,
        torch_context=dict(device="cuda:0", dtype=torch.float64),
        initial_context=dict(device="cuda:0", dtype=torch.float64),
        dataset_path=path2dataset / "all_data.h5",
        replicate_names=replicate_names,
        verbose=2,
    )

    for iteration in range(1, 5):
        obj.estimate_parameters()
        obj.estimate_weights()

    obj.save_results(path2dataset / "trained_differential_metagenes_4_iterations.h5ad")

    return obj


def test_Sigma_x_inv(popari_with_neighbors):
    Sigma_x_inv = (
        list(popari_with_neighbors.parameter_optimizer.spatial_affinity_state.values())[0].detach().cpu().numpy()
    )
    test_Sigma_x_inv = np.load("tests/test_data/synthetic_dataset/outputs/Sigma_x_inv_differential.npy")
    assert np.allclose(test_Sigma_x_inv, Sigma_x_inv)


def test_M(popari_with_neighbors):
    first_group_name = list(popari_with_neighbors.metagene_groups.keys())[0]
    M_bar = popari_with_neighbors.parameter_optimizer.metagene_state.M_bar[first_group_name].detach().cpu().numpy()
    test_M = np.load("tests/test_data/synthetic_dataset/outputs/M_bar_differential.npy")
    assert np.allclose(test_M, M_bar)


def test_X_0(popari_with_neighbors):
    X_0 = popari_with_neighbors.embedding_optimizer.embedding_state["0"].detach().cpu().numpy()
    test_X_0 = np.load("tests/test_data/synthetic_dataset/outputs/X_0_differential.npy")
    assert np.allclose(test_X_0, X_0)


def test_louvain_clustering(popari_with_neighbors):
    df_meta = []
    path2dataset = Path("tests/test_data/synthetic_dataset")
    repli_list = [0, 1]
    expected_aris = [0.6327315501616527, 0.6780724762927413]
    expected_silhouettes = [0.42346037571358214, 0.423733593575608]

    print([dataset.obs["cell_type"] for dataset in popari_with_neighbors.datasets])
    tl.preprocess_embeddings(popari_with_neighbors, joint=True)
    tl.leiden(popari_with_neighbors, joint=True, target_clusters=8)
    tl.compute_ari_scores(popari_with_neighbors, labels="cell_type", predictions="leiden")
    tl.compute_silhouette_scores(popari_with_neighbors, labels="cell_type", embeddings="normalized_X")
    tl.evaluate_classification_task(popari_with_neighbors, labels="cell_type", embeddings="normalized_X", joint=False)
    tl.evaluate_classification_task(popari_with_neighbors, labels="cell_type", embeddings="normalized_X", joint=True)

    for index, (r, X) in enumerate(popari_with_neighbors.embedding_optimizer.embedding_state.items()):
        #     df = pd.read_csv(path2dataset / 'files' / f'meta_{r}.csv')
        df = pd.read_csv(path2dataset / "files" / f"celltypes_{r}.txt", header=None)
        df.columns = ["cell type"]
        df["repli"] = r
        df["cell type"] = pd.Categorical(df["cell type"], categories=np.unique(df["cell type"]))
        df_meta.append(df)

        x = StandardScaler().fit_transform(X.cpu().numpy())

        y = clustering_louvain_nclust(
            x.copy(),
            8,
            kwargs_neighbors=dict(n_neighbors=10),
            kwargs_clustering=dict(),
            resolution_boundaries=(0.1, 1.0),
        )

        df["label Popari"] = y
        ari = adjusted_rand_score(*df[["cell type", "label Popari"]].values.T)
        print(ari)
        assert expected_aris[index] == pytest.approx(ari)

        silhouette = silhouette_score(x, df["cell type"])
        print(silhouette)
        assert expected_silhouettes[index] == pytest.approx(silhouette)


if __name__ == "__main__":
    test_Sigma_x_inv(example_popari_run)
    test_M()
    test_X_0()

    test_louvain_clustering()
