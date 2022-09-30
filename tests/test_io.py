import pytest
import squidpy as sq

from spicemix.io import load_anndata, save_anndata
from spicemix.model import SpiceMixPlus, load_trained_model
from spicemix.analysis import plot_metagene_embedding, leiden, plot_in_situ, multireplicate_heatmap, multigroup_heatmap, compute_ari_scores, plot_all_metagene_embeddings, compute_empirical_correlations
from pathlib import Path

@pytest.fixture(scope="module")
def trained_model():
    path2dataset = Path('tests/test_data/synthetic_500_100_20_15_0_0_i4')
    replicate_names=[0, 1]
    trained_model = load_trained_model(path2dataset / "trained_4_iterations.h5ad", replicate_names)
    trained_model = load_trained_model(path2dataset / "trained_4_iterations.h5ad")

    return trained_model

def test_load_trained_model(trained_model):
    pass

def test_save_anndata(trained_model):
    replicate_names = [dataset.name for dataset in trained_model.datasets]
    path2dataset = Path('tests/test_data/synthetic_500_100_20_15_0_0_i4')
    save_anndata(path2dataset/ "mock_results.h5ad", trained_model.datasets, replicate_names)

def test_load_anndata(trained_model):
    replicate_names=[0, 1]
    path2dataset = Path('tests/test_data/synthetic_500_100_20_15_0_0_i4')
    load_anndata(path2dataset/ "trained_4_iterations.h5ad", replicate_names)
    load_anndata(path2dataset/ "trained_4_iterations.h5ad")

def test_analysis_functions(trained_model):
    trained_model.embedding_optimizer.embedding_state.normalize()

    expected_aris = [0.8719074554517243, 0.8732437089486653]
    leiden(trained_model, joint=True, target_clusters=8)
    compute_ari_scores(trained_model, labels="cell_type", predictions="leiden")

    for expected_ari, dataset in zip(expected_aris, trained_model.datasets):
        print(f"ARI score: {dataset.uns['ari']}")
        dataset.uns["spatial_neighbors"] = {
            "connectivities_key": "adjacency_matrix",
            "distances_key": "spatial_distances"
        }
        assert expected_ari == pytest.approx(dataset.uns["ari"])
        sq.gr.spatial_neighbors(dataset, key_added="spatial")

    plot_in_situ(trained_model)
    

    multireplicate_heatmap(trained_model, uns="Sigma_x_inv")
    plot_all_metagene_embeddings(trained_model, embedding_key="normalized_X")
    compute_empirical_correlations(trained_model, output="empirical_correlation")
    multireplicate_heatmap(trained_model, uns="empirical_correlation")

    multigroup_heatmap(trained_model, key="multigroup_heatmap")
