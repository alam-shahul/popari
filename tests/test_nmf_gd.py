from model import SpiceMixPlus
from util import clustering_louvain_nclust

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
def spicemix_without_neighbors():
    path2dataset = Path('../tests/test_data/synthetic_500_100_20_15_0_0_i4')
    obj = SpiceMixPlus(
        K=10, lambda_Sigma_x_inv=1e-5,
        repli_list=[0],
        context=dict(device='cuda:0', dtype=torch.float32),
        context_Y=dict(dtype=torch.float32, device='cuda:0'),
    )   
    obj.load_dataset(path2dataset)
    obj.initialize(
    #     method='kmeans',
        method='svd',
    )   
    obj.initialize_Sigma_x_inv()
    for iiter in trange(5):
        obj.estimate_weights(iiter=iiter, use_spatial=[False]*obj.num_repli)
        obj.estimate_parameters(iiter=iiter, use_spatial=[False]*obj.num_repli)

    return obj

def test_Sigma_x_inv_no_neighbors(spicemix_without_neighbors):
    Sigma_x_inv = spicemix_without_neighbors.Sigma_x_inv.detach().cpu().numpy()
    test_Sigma_x_inv = np.load("../tests/test_data/synthetic_500_100_20_15_0_0_i4/outputs/Sigma_x_inv_no_neighbors.npy")
    assert np.allclose(test_Sigma_x_inv, Sigma_x_inv)
    
def test_M_no_neighbors(spicemix_without_neighbors):
    M = spicemix_without_neighbors.M.detach().cpu().numpy()
    test_M = np.load("../tests/test_data/synthetic_500_100_20_15_0_0_i4/outputs/M_no_neighbors.npy")
    assert np.allclose(test_M, M)
    
def test_X_0_no_neighbors(spicemix_without_neighbors):
    X_0 = spicemix_without_neighbors.Xs[0].detach().cpu().numpy()
    test_X_0 = np.load("../tests/test_data/synthetic_500_100_20_15_0_0_i4/outputs/X_0_no_neighbors.npy")
    assert np.allclose(test_X_0, X_0)
    
def test_louvain_clustering_no_neighbors(spicemix_without_neighbors):
    df_meta = []
    path2dataset = Path('../tests/test_data/synthetic_500_100_20_15_0_0_i4')
    repli_list = [0]
    for r in repli_list:
    #     df = pd.read_csv(path2dataset / 'files' / f'meta_{r}.csv')
        df = pd.read_csv(path2dataset / 'files' / f'celltypes_{r}.txt', header=None)
        df.columns = ['cell type']
        df['repli'] = r
        df_meta.append(df)
    df_meta = pd.concat(df_meta, axis=0).reset_index(drop=True)
    df_meta['cell type'] = pd.Categorical(df_meta['cell type'], categories=np.unique(df_meta['cell type']))

    Xs = [X.cpu().numpy() for X in spicemix_without_neighbors.Xs]

    x = np.concatenate(Xs, axis=0)
    x = StandardScaler().fit_transform(x)
    
    y = clustering_louvain_nclust(
        x.copy(), 8,
        kwargs_neighbors=dict(n_neighbors=10),
        kwargs_clustering=dict(),
        resolution_boundaries=(.1, 1.),
    )
    
    df_meta['label SpiceMixPlus'] = y
    ari = adjusted_rand_score(*df_meta[['cell type', 'label SpiceMixPlus']].values.T)
    print(ari)
    assert 0.4907662186723056 == pytest.approx(ari)
        
    silhouette = silhouette_score(x, df_meta['cell type'])
    print(silhouette)
    assert 0.22784024477005005 == pytest.approx(silhouette)
