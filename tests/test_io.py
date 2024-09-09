import pytest
import squidpy as sq

from popari import pl, tl

from popari.io import load_anndata, save_anndata
from popari.model import Popari, load_trained_model

from pathlib import Path

@pytest.fixture(scope="module")
def trained_model():
    path2dataset = Path('tests/test_data/synthetic_500_100_20_15_0_0_i4')
    replicate_names=[0, 1]
    trained_model = load_trained_model(path2dataset / "trained_4_iterations.h5ad")

    return trained_model

@pytest.fixture(scope="module")
def trained_differential_model():
    path2dataset = Path('tests/test_data/synthetic_500_100_20_15_0_0_i4')
    replicate_names=[0, 1]
    trained_model = load_trained_model(path2dataset / "trained_differential_metagenes_4_iterations.h5ad")

    return trained_model

@pytest.fixture(scope="module")
def differential_from_shared():
    path2dataset = Path('tests/test_data/synthetic_500_100_20_15_0_0_i4')
    trained_model = load_trained_model(path2dataset / "trained_4_iterations.h5ad", metagene_mode="differential", spatial_affinity_mode="differential lookup")

    return trained_model

def test_load_trained_model(trained_model):
    pass

def test_load_differential_from_shared(differential_from_shared):
    assert differential_from_shared.metagene_mode == "differential"
    assert differential_from_shared.spatial_affinity_mode == "differential lookup"

def test_save_anndata(trained_model):
    replicate_names = [dataset.name for dataset in trained_model.datasets]
    path2dataset = Path('tests/test_data/synthetic_500_100_20_15_0_0_i4')
    save_anndata(path2dataset/ "mock_results.h5ad", trained_model.datasets)
    save_anndata(path2dataset/ "mock_results_ignore_raw_data.h5ad", trained_model.datasets, ignore_raw_data=True)

def test_load_anndata():
    replicate_names=[0, 1]
    path2dataset = Path('tests/test_data/synthetic_500_100_20_15_0_0_i4')
    load_anndata(path2dataset/ "trained_4_iterations.h5ad")

def test_load_hierarchical_model():
    replicate_names=[0, 1]
    path2dataset = Path('tests/test_data/synthetic_500_100_20_15_0_0_i4')
    load_trained_model(path2dataset/ "superresolved_results")

def test_nll(trained_model):
    nll = trained_model.nll()
