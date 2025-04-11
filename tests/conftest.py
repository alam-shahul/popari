import itertools
from pathlib import Path

import anndata as ad
import numpy as np
import pytest
import scanpy as sc
import torch

from popari.components import PopariDataset
from popari.model import Popari


@pytest.fixture(scope="module")
def test_datapath():
    return Path("tests/test_data/synthetic_dataset")


@pytest.fixture(scope="module")
def dataset_path():
    path2dataset = Path("tests/test_data/synthetic_dataset")

    return path2dataset


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
def shared_model(mock_datasets, context):
    obj = Popari(
        K=10,
        lambda_Sigma_x_inv=1e-3,
        metagene_mode="shared",
        torch_context=context,
        initial_context=context,
        initialization_method="svd",
        datasets=mock_datasets,
        replicate_names=["mock_1", "mock_2"],
        verbose=1,
    )

    return obj


@pytest.fixture(scope="module")
def shared_model_batch_effect_correction(mock_datasets, context):
    obj = Popari(
        K=10,
        lambda_Sigma_x_inv=1e-3,
        metagene_mode="shared",
        torch_context=context,
        initial_context=context,
        initialization_method="svd",
        datasets=mock_datasets,
        replicate_names=["mock_1", "mock_2"],
        batch_effect_correction=True,
        verbose=1,
    )

    return obj


@pytest.fixture(scope="module")
def shared_model_batch_effect_correction_with_prior(mock_datasets, context):
    obj = Popari(
        K=10,
        lambda_Sigma_x_inv=1e-3,
        metagene_mode="shared",
        torch_context=context,
        initial_context=context,
        initialization_method="svd",
        datasets=mock_datasets,
        replicate_names=["mock_1", "mock_2"],
        prior_x_modes=["exponential shared fixed", "exponential shared fixed"],
        batch_effect_correction=True,
        verbose=1,
    )

    return obj


def get_mock_dataset(
    gene_expression,
    cell_types,
    num_cells: int = 50,
    num_genes: int = 20,
    num_rows: int = 10,
):
    coordinates = np.zeros((num_cells, 2))

    num_cols = num_cells // num_rows
    coordinates[:, 0] = np.tile(np.arange(num_rows), num_cols)
    coordinates[:, 1] = np.repeat(np.arange(num_cols), num_rows)

    dataset = ad.AnnData(
        X=gene_expression,
    )

    dataset.obsm["spatial"] = coordinates
    dataset.obs["cell_type"] = cell_types

    # dataset = PopariDataset(dataset, "mock")

    return dataset


@pytest.fixture(scope="module")
def mock_datasets():
    num_genes = 20
    num_cells_1 = 50
    num_cells_2 = 25

    random_state = 0
    rng = np.random.default_rng(random_state)

    gene_expression_1 = np.zeros((num_cells_1, num_genes))
    gene_expression_1[00:10] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    gene_expression_1[10:20] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    gene_expression_1[20:30] = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    gene_expression_1[30:40] = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    gene_expression_1[40:50] = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

    gene_expression_1 += rng.normal(size=gene_expression_1.shape)

    cell_types_1 = list(
        itertools.chain.from_iterable(
            [
                [f"C0" for _ in range(10)],
                [f"C1" for _ in range(10)],
                [f"C2" for _ in range(10)],
                [f"C3" for _ in range(10)],
                [f"C4" for _ in range(10)],
            ],
        ),
    )

    mock_dataset_1 = get_mock_dataset(
        gene_expression_1,
        cell_types_1,
        num_cells=num_cells_1,
        num_genes=num_genes,
        num_rows=10,
    )

    gene_expression_2 = np.zeros((num_cells_2, num_genes))
    gene_expression_2[0:5] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    gene_expression_2[5:10] = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    gene_expression_2[10:15] = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    gene_expression_2[15:20] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    gene_expression_2[20:25] = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

    gene_expression_2 += rng.normal(size=gene_expression_2.shape)

    cell_types_2 = list(
        itertools.chain.from_iterable(
            [
                [f"C0" for _ in range(5)],
                [f"C3" for _ in range(5)],
                [f"C2" for _ in range(5)],
                [f"C1" for _ in range(5)],
                [f"C4" for _ in range(5)],
            ],
        ),
    )

    mock_dataset_2 = get_mock_dataset(
        gene_expression_2,
        cell_types_2,
        num_cells=num_cells_2,
        num_genes=num_genes,
        num_rows=5,
    )

    return [mock_dataset_1, mock_dataset_2]
