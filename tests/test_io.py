import pytest
from spicemix.io import load_anndata, save_anndata
from spicemix.model import SpiceMixPlus, load_trained_model
from spicemix.analysis import plot_metagene_embedding, leiden, plot_in_situ, multireplicate_heatmap, compute_ari_scores, plot_ari_scores, plot_all_metagene_embeddings
from pathlib import Path

@pytest.fixture(scope="module")
def trained_model():
    path2dataset = Path('tests/test_data/synthetic_500_100_20_15_0_0_i4')
    replicate_names=[0, 1]
    trained_model = load_trained_model(path2dataset / "trained_4_iterations.h5ad", replicate_names)

    return trained_model

def test_load_trained_model(trained_model):
    pass

def test_save_anndata(trained_model):
    replicate_names = [dataset.name for dataset in trained_model.datasets]
    path2dataset = Path('tests/test_data/synthetic_500_100_20_15_0_0_i4')
    save_anndata(path2dataset/ "mock_results.h5ad", trained_model.datasets, replicate_names)

def test_analysis_functions(trained_model):
    trained_model.embedding_optimizer.embedding_state.normalize()

    leiden(trained_model)
    plot_in_situ(trained_model)
    
    compute_ari_scores(trained_model, labels="cell_type", predictions="leiden")
    plot_ari_scores(trained_model)
    multireplicate_heatmap(trained_model, uns="Sigma_x_inv")
    plot_all_metagene_embeddings(trained_model, embedding_key="normalized_X")
