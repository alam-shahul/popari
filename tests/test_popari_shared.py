from popari.model import Popari
from popari.util import clustering_louvain_nclust

import os
import pandas as pd
import pytest
import torch
from tqdm.auto import tqdm, trange
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

@pytest.fixture(scope="module")
def popari_with_neighbors():
    path2dataset = Path('tests/test_data/synthetic_500_100_20_15_0_0_i4')
    replicate_names=[0, 1]
    obj = Popari(
        K=10, lambda_Sigma_x_inv=1e-3,
        metagene_mode="shared",
        torch_context=dict(device='cuda:0', dtype=torch.float64),
        initial_context=dict(device='cuda:0', dtype=torch.float64),
        initialization_method="svd",
        dataset_path=path2dataset / "all_data.h5",
        replicate_names=replicate_names,
        verbose=4
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
        dataset.uns["multigroup_heatmap"] = {group_name: np.arange(4).reshape((2, 2)) for group_name in obj.metagene_groups}

    obj.save_results(path2dataset / 'trained_4_iterations.h5ad')
    return obj

@pytest.fixture(scope="module")
def popari_with_leiden_initialization():
    path2dataset = Path('tests/test_data/synthetic_500_100_20_15_0_0_i4')
    replicate_names=[0, 1]
    obj = Popari(
        K=10, lambda_Sigma_x_inv=1e-3,
        metagene_mode="shared",
        torch_context=dict(device='cuda:0', dtype=torch.float64),
        initial_context=dict(device='cuda:0', dtype=torch.float64),
        dataset_path=path2dataset / "all_data.h5",
        replicate_names=replicate_names,
        verbose=4
    )

def test_leiden_initialization(popari_with_leiden_initialization):
    pass

def test_Sigma_x_inv(popari_with_neighbors):
    Sigma_x_inv = list(popari_with_neighbors.parameter_optimizer.spatial_affinity_state.values())[0].cpu().detach().numpy()
    # np.save("tests/test_data/synthetic_500_100_20_15_0_0_i4/outputs/Sigma_x_inv_shared.npy", Sigma_x_inv)
    test_Sigma_x_inv = np.load("tests/test_data/synthetic_500_100_20_15_0_0_i4/outputs/Sigma_x_inv_shared.npy")
    assert np.allclose(test_Sigma_x_inv, Sigma_x_inv)

def test_M(popari_with_neighbors):
    M_bar = popari_with_neighbors.parameter_optimizer.metagene_state.metagenes.detach().cpu().numpy()
    # np.save("tests/test_data/synthetic_500_100_20_15_0_0_i4/outputs/M_bar_shared.npy", M_bar)
    test_M = np.load("tests/test_data/synthetic_500_100_20_15_0_0_i4/outputs/M_bar_shared.npy")
    assert np.allclose(test_M, M_bar)

def test_X_0(popari_with_neighbors):
    X_0 = popari_with_neighbors.embedding_optimizer.embedding_state["0"].detach().cpu().numpy()
    # np.save("tests/test_data/synthetic_500_100_20_15_0_0_i4/outputs/X_0_shared.npy", X_0)
    test_X_0 = np.load("tests/test_data/synthetic_500_100_20_15_0_0_i4/outputs/X_0_shared.npy")
    assert np.allclose(test_X_0, X_0)

def test_louvain_clustering(popari_with_neighbors):
    df_meta = []
    path2dataset = Path('tests/test_data/synthetic_500_100_20_15_0_0_i4')
    repli_list = [0, 1]
    expected_aris = [0.8153973232569635, 0.8493627279334213]
    expected_silhouettes = [0.3065145002352882, 0.3437466653435347]
    
    for index, (r, X) in enumerate(popari_with_neighbors.embedding_optimizer.embedding_state.items()):
    #     df = pd.read_csv(path2dataset / 'files' / f'meta_{r}.csv')
        df = pd.read_csv(path2dataset / 'files' / f'celltypes_{r}.txt', header=None)
        df.columns = ['cell type']
        df['repli'] = r
        df['cell type'] = pd.Categorical(df['cell type'], categories=np.unique(df['cell type']))
        df_meta.append(df)

        x = StandardScaler().fit_transform(X.cpu().numpy())
        
        y = clustering_louvain_nclust(
            x.copy(), 8,
            kwargs_neighbors=dict(n_neighbors=10),
            kwargs_clustering=dict(),
            resolution_boundaries=(.1, 1.),
        )
        
        df['label Popari'] = y
        ari = adjusted_rand_score(*df[['cell type', 'label Popari']].values.T)
        print(ari)
        assert expected_aris[index] == pytest.approx(ari)
            
        silhouette = silhouette_score(x, df['cell type'])
        print(silhouette)
        assert expected_silhouettes[index]  == pytest.approx(silhouette)

    # df_meta = pd.concat(df_meta, axis=0).reset_index(drop=True)
    # df_meta['cell type'] = pd.Categorical(df_meta['cell type'], categories=np.unique(df_meta['cell type']))

    # Xs = [X.cpu().numpy() for X in popari_with_neighbors.Xs]

    # x = np.concatenate(Xs, axis=0)
    # x = StandardScaler().fit_transform(x)
    # 
    # y = clustering_louvain_nclust(
    #     x.copy(), 8,
    #     kwargs_neighbors=dict(n_neighbors=10),
    #     kwargs_clustering=dict(),
    #     resolution_boundaries=(.1, 1.),
    # )
    # 
    # df_meta['label Popari'] = y
    # ari = adjusted_rand_score(*df_meta[['cell type', 'label Popari']].values.T)
    # print(ari)
    # assert 0.3731545260146673 == pytest.approx(ari)
    #     
    # silhouette = silhouette_score(x, df_meta['cell type'])
    # print(silhouette)
    # assert 0.029621144756674767  == pytest.approx(silhouette)
# def test_project2simplex():
#     project2simplex(x, dim=0)

if __name__ == "__main__":
    test_Sigma_x_inv(example_popari_run)
    test_M()
    test_X_0()
    
    test_louvain_clustering()

