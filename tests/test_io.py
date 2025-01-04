from pathlib import Path

import numpy as np
import pytest

from popari.io import load_anndata, save_anndata
from popari.model import load_trained_model


@pytest.fixture(scope="module")
def trained_model(test_datapath):
    trained_model = load_trained_model(test_datapath / "trained_4_iterations.h5ad")

    return trained_model


@pytest.fixture
def delete_mock_data(test_datapath):
    yield
    (test_datapath / "mock_results.h5ad").unlink()
    (test_datapath / "mock_results_ignore_raw_data.h5ad").unlink()


@pytest.fixture(scope="module")
def trained_differential_model(test_datapath):
    trained_model = load_trained_model(test_datapath / "trained_differential_metagenes_4_iterations.h5ad")

    return trained_model


@pytest.fixture(scope="module")
def differential_from_shared(test_datapath):
    trained_model = load_trained_model(
        test_datapath / "trained_4_iterations.h5ad",
        metagene_mode="differential",
        spatial_affinity_mode="differential lookup",
    )

    return trained_model


def test_load_trained_model(trained_model):
    pass


def test_load_differential_from_shared(differential_from_shared):
    assert differential_from_shared.metagene_mode == "differential"
    assert differential_from_shared.spatial_affinity_mode == "differential lookup"


def test_save_anndata(trained_model, test_datapath, delete_mock_data):
    save_anndata(test_datapath / "mock_results.h5ad", trained_model.datasets)
    save_anndata(test_datapath / "mock_results_ignore_raw_data.h5ad", trained_model.datasets, ignore_raw_data=True)


def test_load_anndata(test_datapath):
    load_anndata(test_datapath / "trained_4_iterations.h5ad")


def test_load_hierarchical_model(test_datapath):
    load_trained_model(test_datapath / "outputs" / "superresolved_results")


def test_nll(trained_model):
    nll = trained_model.nll(level=0)[0]

    expected_nll = -266943.2440865346

    assert nll == pytest.approx(expected_nll)  # TODO: why is this returning nan?
