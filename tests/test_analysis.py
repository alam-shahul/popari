import pytest
import squidpy as sq

from popari import pl, tl

from popari.io import load_anndata, save_anndata
from popari._simulation_utils import all_pairs_spatial_wasserstein
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

def test_analysis_functions(trained_model, trained_differential_model):
    tl.preprocess_embeddings(trained_model)

    differential_genes = tl.find_differential_genes(trained_differential_model)

    covariate_values = [0]
    tl.plot_gene_trajectories(trained_differential_model, differential_genes, covariate_values)
    tl.plot_gene_activations(trained_differential_model, differential_genes)

    tl.pca(trained_model, joint=True)

    expected_aris = [0.7405537444360839, 0.7694805741910644]
    tl.leiden(trained_model, joint=True, target_clusters=8)
    tl.compute_ari_scores(trained_model, labels="cell_type", predictions="leiden")
    tl.compute_silhouette_scores(trained_model, labels="cell_type", embeddings="normalized_X")
    tl.evaluate_classification_task(trained_model, labels="cell_type", embeddings="normalized_X", joint=False)
    tl.evaluate_classification_task(trained_model, labels="cell_type", embeddings="normalized_X", joint=True)

    tl.compute_confusion_matrix(trained_model, labels="cell_type", predictions="leiden")
    pl.confusion_matrix(trained_model, labels="cell_type")
    tl.compute_columnwise_autocorrelation(trained_model, uns="M")

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

    pl.in_situ(trained_model)

    tl.umap(trained_model)
    pl.umap(trained_model, color="leiden")

    pl.multireplicate_heatmap(trained_model, uns="Sigma_x_inv")
    pl.multireplicate_heatmap(trained_model, uns="Sigma_x_inv", label_values=True)
    pl.spatial_affinities(trained_model, label_values=True)
    pl.metagene_embedding(trained_model, 0)
    pl.all_embeddings(trained_model, embedding_key="normalized_X")
    tl.compute_empirical_correlations(trained_model, output="empirical_correlation")
    pl.multireplicate_heatmap(trained_model, uns="empirical_correlation")
    tl.compute_spatial_correlation(trained_model)

    pl.multigroup_heatmap(trained_model, key="multigroup_heatmap")

    print(trained_model.datasets[0].var_names)
    pl.cell_type_to_metagene(trained_model, {"type_1": ['0'], "type_2": ['1']})
    pl.cell_type_to_metagene_difference(trained_model, {"type_1": ['0'], "type_2": ['1']}, 0, 1)

    tl.normalized_affinity_trends(trained_differential_model, timepoint_values=[0, 1])
    pl.normalized_affinity_trends(trained_differential_model, timepoint_values=[0, 1])
    distances = _all_pairs_spatial_wasserstein(trained_model.datasets[0], embeddings_truth_key='X', embeddings_pred_key='X')
    print(distances)
