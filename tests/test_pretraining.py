from pathlib import Path

import pytest
import torch

from popari.model import Popari, from_pretrained


@pytest.fixture(scope="module")
def context():
    context = {
        "device": "cuda:0",
        "dtype": torch.float64,
    }
    return context


@pytest.fixture(scope="module")
def hierarchical_model(context):
    path2dataset = Path("tests/test_data/synthetic_dataset")
    replicate_names = [0, 1]
    obj = Popari(
        K=10,
        lambda_Sigma_x_inv=1e-3,
        metagene_mode="shared",
        torch_context=context,
        initial_context=context,
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
    popari_context = {"dtype": torch.float32, "device": "cuda:0"}
    _ = from_pretrained(hierarchical_model, popari_context=popari_context, lambda_Sigma_bar=lambda_Sigma_bar)
