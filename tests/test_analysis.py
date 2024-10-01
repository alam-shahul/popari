from pathlib import Path

import numpy as np
import pytest

from popari import pl, tl
from popari.model import load_trained_model
from popari.util import concatenate


@pytest.fixture(scope="module")
def dataset_path():
    path2dataset = Path("tests/test_data/synthetic_dataset")

    return path2dataset


@pytest.fixture(scope="module")
def trained_model(dataset_path):
    trained_model = load_trained_model(dataset_path / "trained_4_iterations.h5ad")

    return trained_model


@pytest.fixture(scope="module")
def trained_differential_model(dataset_path):
    trained_model = load_trained_model(dataset_path / "trained_differential_metagenes_4_iterations.h5ad")

    return trained_model


def test_differential_analysis(trained_differential_model):
    differential_genes = tl.find_differential_genes(trained_differential_model)

    covariate_values = [0]
    tl.plot_gene_trajectories(trained_differential_model, differential_genes, covariate_values)
    tl.plot_gene_activations(trained_differential_model, differential_genes)

    tl.normalized_affinity_trends(trained_differential_model, timepoint_values=[0, 1])
    pl.normalized_affinity_trends(trained_differential_model, timepoint_values=[0, 1])


@pytest.fixture(scope="module")
def preprocessed_model(trained_model):
    tl.preprocess_embeddings(trained_model, joint=True)
    tl.preprocess_embeddings(trained_model, joint=False)

    return trained_model


def test_preprocess_model(preprocessed_model): ...


def test_pca(preprocessed_model, dataset_path):
    tl.pca(preprocessed_model, joint=False)
    merged_dataset = concatenate(preprocessed_model.datasets)
    disjoint_pca = merged_dataset.obsm["X_pca"]
    if not (dataset_path / "pca_disjoint.npy").exists():
        np.save(dataset_path / "pca_disjoint.npy", disjoint_pca)

    saved_pca = np.load(dataset_path / "pca_disjoint.npy")
    assert np.allclose(disjoint_pca, saved_pca)

    tl.pca(preprocessed_model, joint=True)
    merged_dataset = concatenate(preprocessed_model.datasets)
    joint_pca = merged_dataset.obsm["X_pca"]
    if not (dataset_path / "pca_joint.npy").exists():
        np.save(dataset_path / "pca_joint.npy", joint_pca)

    saved_pca = np.load(dataset_path / "pca_joint.npy")
    assert np.allclose(joint_pca, saved_pca)


@pytest.fixture(scope="module")
def clustered_model(preprocessed_model):
    tl.leiden(preprocessed_model, joint=True, target_clusters=8, verbose=True)

    return preprocessed_model


def test_leiden_clustering(clustered_model): ...


def test_ari_score(clustered_model):
    expected_aris = [0.7638266995561214, 0.8126115509883821]
    tl.compute_ari_scores(clustered_model, labels="cell_type", predictions="leiden")

    for expected_ari, dataset in zip(expected_aris, clustered_model.datasets):
        print(f"ARI score: {dataset.uns['ari']}")
        assert expected_ari == pytest.approx(dataset.uns["ari"])


def test_silhouette_score(clustered_model):
    expected_silhouettes = [0.30699036262154916, 0.34466040612221843]
    tl.compute_silhouette_scores(clustered_model, labels="cell_type", embeddings="normalized_X")

    for expected_silhouette, dataset in zip(expected_silhouettes, clustered_model.datasets):
        print(f"Silhouette score: {dataset.uns['silhouette']}")
        assert expected_silhouette == pytest.approx(dataset.uns["silhouette"])


def test_classification_task_disjoint(clustered_model):
    expected_microprecisions = [0.9013333333333333, 0.88]
    expected_macroprecisions = [0.9171604437229437, 0.9097823912614673]
    tl.evaluate_classification_task(clustered_model, labels="cell_type", embeddings="normalized_X", joint=False)

    for expected_microprecision, expected_macroprecision, dataset in zip(
        expected_microprecisions,
        expected_macroprecisions,
        clustered_model.datasets,
    ):
        print(f"Validation microprecision: {dataset.uns['microprecision_validation']}")
        print(f"Validation macroprecision: {dataset.uns['macroprecision_validation']}")
        assert pytest.approx(dataset.uns["microprecision_validation"]) == expected_microprecision
        assert pytest.approx(dataset.uns["macroprecision_validation"]) == expected_macroprecision


def test_classification_task_joint(clustered_model):
    expected_microprecisions = [0.9226666666666666, 0.9226666666666666]
    expected_macroprecisions = [0.9333166458237094, 0.9333166458237094]
    tl.evaluate_classification_task(clustered_model, labels="cell_type", embeddings="normalized_X", joint=True)

    for expected_microprecision, expected_macroprecision, dataset in zip(
        expected_microprecisions,
        expected_macroprecisions,
        clustered_model.datasets,
    ):
        print(f"Validation microprecision: {dataset.uns['microprecision_validation']}")
        print(f"Validation macroprecision: {dataset.uns['macroprecision_validation']}")
        assert pytest.approx(dataset.uns["microprecision_validation"]) == expected_microprecision
        assert pytest.approx(dataset.uns["macroprecision_validation"]) == expected_macroprecision


def test_confusion_matrix(clustered_model):
    try:
        tl.compute_confusion_matrix(clustered_model, labels="cell_type", predictions="leiden")
        pl.confusion_matrix(clustered_model, labels="cell_type")
    except Exception as e:
        assert type(e) is ValueError

    try:
        tl.compute_confusion_matrix(clustered_model, labels="cell_type", predictions="leiden", joint=True)
        pl.confusion_matrix(clustered_model, labels="cell_type")
    except Exception as e:
        assert type(e) is ValueError


def test_columnwise_autocorrelation(clustered_model):
    tl.compute_columnwise_autocorrelation(clustered_model, uns="M")


def test_plot_in_situ(clustered_model):
    pl.in_situ(clustered_model)


def test_umap(clustered_model):
    tl.umap(clustered_model)
    pl.umap(clustered_model, color="leiden")
    pl.umap(clustered_model, color="leiden", joint=True)


def test_multireplicate_heatmap(clustered_model):
    pl.multireplicate_heatmap(clustered_model, uns="Sigma_x_inv")
    pl.multireplicate_heatmap(clustered_model, uns="Sigma_x_inv", label_values=True)
    pl.spatial_affinities(clustered_model, label_values=True)


def test_plot_metagene_embedding(clustered_model):
    pl.metagene_embedding(clustered_model, 0)


def test_plot_all_embeddings(clustered_model):
    pl.all_embeddings(clustered_model, embedding_key="normalized_X")


def test_empirical_correlations(clustered_model):
    tl.compute_empirical_correlations(clustered_model, output="empirical_correlation")
    pl.multireplicate_heatmap(clustered_model, uns="empirical_correlation")


def test_multigroup_heatmap(clustered_model):
    pl.multigroup_heatmap(clustered_model, key="multigroup_heatmap")


def test_plot_metagene_to_cell_type(clustered_model):
    pl.cell_type_to_metagene(clustered_model, {"type_1": ["0"], "type_2": ["1"]})
    pl.cell_type_to_metagene_difference(clustered_model, {"type_1": ["0"], "type_2": ["1"]}, 0, 1)


def test_spatial_gene_correlation(clustered_model):
    tl.compute_spatial_gene_correlation(clustered_model)
