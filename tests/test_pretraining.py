from pathlib import Path

import pytest
import squidpy as sq
import torch

from popari import pl, tl
from popari._dataset_utils import _spatial_binning
from popari.io import load_anndata, save_anndata
from popari.model import Popari, from_pretrained, load_trained_model


@pytest.fixture(scope="module")
def hierarchical_model():
    path2dataset = Path("tests/test_data/synthetic_dataset")
    replicate_names = [0, 1]
    obj = Popari(
        K=10,
        lambda_Sigma_x_inv=1e-3,
        metagene_mode="shared",
        torch_context=dict(device="cuda:0", dtype=torch.float64),
        initial_context=dict(device="cuda:0", dtype=torch.float64),
        initialization_method="svd",
        dataset_path=path2dataset / "all_data.h5",
        replicate_names=replicate_names,
        hierarchical_levels=3,
        superresolution_lr=1e-2,
        verbose=4,
    )

    return obj


def test_differential_initialization(hierarchical_model):
    lambda_Sigma_bar = 1e-3
    popari_context = {"dtype": torch.float64, "device": "cuda:2"}
    popari_test = from_pretrained(hierarchical_model, popari_context=popari_context, lambda_Sigma_bar=lambda_Sigma_bar)
