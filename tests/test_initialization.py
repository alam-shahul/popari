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
def random_initialization_0():
    path2dataset = Path('tests/test_data/synthetic_500_100_20_15_0_0_i4')
    replicate_names=[0, 1]
    obj = Popari(
        K=10, lambda_Sigma_x_inv=1e-3,
        metagene_mode="shared",
        torch_context=dict(device='cuda:0', dtype=torch.float64),
        initial_context=dict(device='cuda:0', dtype=torch.float64),
        dataset_path=path2dataset / "all_data.h5",
        replicate_names=replicate_names,
        random_state=0,
        verbose=4
    )

    return obj

@pytest.fixture(scope="module")
def random_initialization_1():
    path2dataset = Path('tests/test_data/synthetic_500_100_20_15_0_0_i4')
    replicate_names=[0, 1]
    obj = Popari(
        K=10, lambda_Sigma_x_inv=1e-3,
        metagene_mode="shared",
        torch_context=dict(device='cuda:0', dtype=torch.float64),
        initial_context=dict(device='cuda:0', dtype=torch.float64),
        dataset_path=path2dataset / "all_data.h5",
        replicate_names=replicate_names,
        random_state=1,
        verbose=4
    )

    return obj

@pytest.fixture(scope="module")
def hierarchical_initialization_1():
    path2dataset = Path('../data/synthetic_cortex/final_hierarchical/num_replicates_1/synthetic_1296_100_sigY-scale-2.0_sigX-scale-1.0_noise-metagene-parameter_4.0/random_state_0/processed_dataset.h5ad')
    obj = Popari(
        K=11, lambda_Sigma_x_inv=1e-4,
        spatial_affinity_mode="shared lookup",
        binning_downsample_rate=0.2,
        hierarchical_levels=2,
        lambda_Sigma_bar=0.0,
        torch_context=dict(device='cuda:0', dtype=torch.float64),
        initial_context=dict(device='cuda:0', dtype=torch.float64),
        dataset_path=path2dataset,
        random_state=1,
        verbose=4
    )

    return obj

def test_random_initialization(random_initialization_0, random_initialization_1):
    for index in range(2):
        dataset_0 = random_initialization_0.datasets[index]
        dataset_1 = random_initialization_1.datasets[index]

        assert not np.allclose(dataset_0.uns["M"][dataset_0.name], dataset_1.uns["M"][dataset_1.name])
        assert not np.allclose(dataset_0.obsm["X"], dataset_1.obsm["X"])
        assert not np.allclose(dataset_0.uns["Sigma_x_inv"][dataset_0.name], dataset_1.uns["Sigma_x_inv"][dataset_1.name])

def test_hierarchical_initialization(hierarchical_initialization_1):
    pass
