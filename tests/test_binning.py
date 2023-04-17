import pytest
import squidpy as sq

from popari import pl, tl

from popari.io import load_anndata, save_anndata
from popari.model import Popari, load_trained_model
from popari._dataset_utils import _spatial_binning

from pathlib import Path

@pytest.fixture(scope="module")
def trained_model():
    path2dataset = Path('tests/test_data/synthetic_500_100_20_15_0_0_i4')
    replicate_names=[0, 1]
    trained_model = load_trained_model(path2dataset / "trained_4_iterations.h5ad", replicate_names)
    trained_model = load_trained_model(path2dataset / "trained_4_iterations.h5ad")

    return trained_model

def test_load_trained_model(trained_model):
    binned_datasets = []
    for dataset in trained_model.datasets:
        binned_dataset = _spatial_binning(dataset, chunks=4, downsample_rate=0.5)
        binned_datasets.append(binned_dataset)
