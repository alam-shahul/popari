import pytest
import squidpy as sq

from popari.io import load_anndata, save_anndata
from popari.model import SpiceMixPlus, load_trained_model
from popari.analysis import preprocess_embeddings, plot_metagene_embedding, leiden, plot_in_situ, multireplicate_heatmap, \
     multigroup_heatmap, compute_ari_scores, compute_silhouette_scores, plot_all_metagene_embeddings, \
     compute_empirical_correlations, find_differential_genes, plot_gene_activations, plot_gene_trajectories, \
     evaluate_classification_task, compute_confusion_matrix, plot_confusion_matrix, compute_columnwise_autocorrelation

from pathlib import Path

@pytest.fixture(scope="module")
def trained_model():
    path2dataset = Path('tests/test_data/synthetic_500_100_20_15_0_0_i4')
    replicate_names=[0, 1]
    trained_model = load_trained_model(path2dataset / "trained_4_iterations.h5ad", replicate_names)
    trained_model = load_trained_model(path2dataset / "trained_4_iterations.h5ad")

    return trained_model

@pytest.fixture(scope="module")
def trained_differential_model():
    path2dataset = Path('tests/test_data/synthetic_500_100_20_15_0_0_i4')
    replicate_names=[0, 1]
    trained_model = load_trained_model(path2dataset / "trained_differential_metagenes_4_iterations.h5ad", replicate_names)
    trained_model = load_trained_model(path2dataset / "trained_differential_metagenes_4_iterations.h5ad")

    return trained_model

def test_load_trained_model(trained_model):
    pass

def test_save_anndata(trained_model):
    replicate_names = [dataset.name for dataset in trained_model.datasets]
    path2dataset = Path('tests/test_data/synthetic_500_100_20_15_0_0_i4')
    save_anndata(path2dataset/ "mock_results.h5ad", trained_model.datasets, replicate_names)
    save_anndata(path2dataset/ "mock_results_ignore_raw_data.h5ad", trained_model.datasets, replicate_names, ignore_raw_data=True)

def test_load_anndata(trained_model):
    replicate_names=[0, 1]
    path2dataset = Path('tests/test_data/synthetic_500_100_20_15_0_0_i4')
    load_anndata(path2dataset/ "trained_4_iterations.h5ad", replicate_names)
    load_anndata(path2dataset/ "trained_4_iterations.h5ad")

def test_analysis_functions(trained_model, trained_differential_model):
    preprocess_embeddings(trained_model)

    differential_genes = find_differential_genes(trained_differential_model)
    print(differential_genes)

    covariate_values = [0]
    plot_gene_trajectories(trained_differential_model, differential_genes, covariate_values)
    plot_gene_activations(trained_differential_model, differential_genes)

    expected_aris = [0.8629513626120968, 0.8770125653681966]
    leiden(trained_model, joint=True, target_clusters=8)
    compute_ari_scores(trained_model, labels="cell_type", predictions="leiden")
    compute_silhouette_scores(trained_model, labels="cell_type", embeddings="normalized_X")
    evaluate_classification_task(trained_model, labels="cell_type", embeddings="normalized_X", joint=False)
    evaluate_classification_task(trained_model, labels="cell_type", embeddings="normalized_X", joint=True)

    compute_confusion_matrix(trained_model, labels="cell_type", predictions="leiden")
    plot_confusion_matrix(trained_model, labels="cell_type")
    compute_columnwise_autocorrelation(trained_model, uns="M")

    for expected_ari, dataset in zip(expected_aris, trained_model.datasets):
        print(f"ARI score: {dataset.uns['ari']}")
        print(f"Silhouette score: {dataset.uns['silhouette']}")
        print(f"Train micro-precision: {dataset.uns['microprecision_train']}")
        print(f"Validation micro-precision: {dataset.uns['microprecision_validation']}")
        print(f"Train macro-precision: {dataset.uns['macroprecision_train']}")
        print(f"Validation macro-precision: {dataset.uns['macroprecision_validation']}")
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
