from pathlib import Path

import numpy as np
import pytest

from popari.io import load_anndata, save_anndata
from popari.model import load_trained_model


@pytest.fixture(scope="module")
def dataset_path():
    return Path("tests/test_data/synthetic_dataset")


@pytest.fixture(scope="module")
def trained_model(dataset_path):
    trained_model = load_trained_model(dataset_path / "trained_4_iterations.h5ad")

    return trained_model


@pytest.fixture
def delete_mock_data(dataset_path):
    yield
    (dataset_path / "mock_results.h5ad").unlink()
    (dataset_path / "mock_results_ignore_raw_data.h5ad").unlink()


@pytest.fixture(scope="module")
def trained_differential_model(dataset_path):
    dataset_path = Path("tests/test_data/synthetic_dataset")
    trained_model = load_trained_model(dataset_path / "trained_differential_metagenes_4_iterations.h5ad")

    print(trained_model.datasets[0].uns)

    return trained_model


@pytest.fixture(scope="module")
def differential_from_shared(dataset_path):
    trained_model = load_trained_model(
        dataset_path / "trained_4_iterations.h5ad",
        metagene_mode="differential",
        spatial_affinity_mode="differential lookup",
    )

    return trained_model


def test_load_trained_model(trained_model):
    pass


def test_load_differential_from_shared(differential_from_shared):
    assert differential_from_shared.metagene_mode == "differential"
    assert differential_from_shared.spatial_affinity_mode == "differential lookup"


def test_save_anndata(trained_model, dataset_path, delete_mock_data):
    dataset_path = Path("tests/test_data/synthetic_dataset")
    save_anndata(dataset_path / "mock_results.h5ad", trained_model.datasets)
    save_anndata(dataset_path / "mock_results_ignore_raw_data.h5ad", trained_model.datasets, ignore_raw_data=True)


def test_load_anndata(dataset_path):
    load_anndata(dataset_path / "trained_4_iterations.h5ad")


def test_load_hierarchical_model(dataset_path):
    load_trained_model(dataset_path / "outputs" / "superresolved_results")


def test_nll(trained_model):
    nll = trained_model.nll(level=0)

    expected_nll = 0

    assert nll == expected_nll  # TODO: why is this returning nan?
