from pathlib import Path

import pytest
import torch

from popari.model import Popari


@pytest.fixture(scope="module")
def test_datapath():
    #return Path("tests/test_data/synthetic_dataset")
    return Path("tests/test_data/raehash_dataset")


@pytest.fixture(scope="module")
def context():
    context = {
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "dtype": torch.float64,
    }
    return context


@pytest.fixture(scope="module")
def float32_context():
    context = {
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "dtype": torch.float32,
    }
    return context


@pytest.fixture(scope="module")
def shared_model(test_datapath, context):
    obj = Popari(
        K=10,
        lambda_Sigma_x_inv=1e-3,
        metagene_mode="shared",
        torch_context=context,
        initial_context=context,
        initialization_method="svd",
        dataset_path=test_datapath / "all_data.h5",
        verbose=1,
    )

    return obj
