from pathlib import Path

import numpy as np
import anndata as ad
import h5py
import pytest
import torch

from popari import tl
from popari.model import Popari


@pytest.fixture(scope="module")
def popari_with_neighbors(test_datapath, context):
    replicate_names = [0, 1]
    obj = Popari(
        K=10,
        lambda_Sigma_x_inv=1e-5,
        metagene_mode="differential",
        lambda_M=0.5,
        torch_context=context,
        initial_context=context,
        dataset_path=test_datapath / "all_data.h5", # Shape  (100,100)
        replicate_names=replicate_names,
        verbose=2,
    )
    return obj


def test_embedding_optimization(popari_with_neighbors, test_datapath):
    #Check the embedding state after updates 
    initial_data_0 = popari_with_neighbors.embedding_optimizer.embedding_state["0"].detach().cpu().numpy()
    initial_data_1 = popari_with_neighbors.embedding_optimizer.embedding_state["1"].detach().cpu().numpy()
    for iteration in range(1, 5):
        popari_with_neighbors.estimate_parameters()
        np.allclose(initial_data_0, popari_with_neighbors.embedding_optimizer.embedding_state["0"].detach().cpu().numpy(), atol=1e-3)
        np.allclose(initial_data_1, popari_with_neighbors.embedding_optimizer.embedding_state["1"].detach().cpu().numpy(), atol=1e-3)
        popari_with_neighbors.estimate_weights()
    

def test_model_parameters_optimization(popari_with_neighbors, test_datapath):p
    assert True

