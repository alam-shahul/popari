import numpy as np
import pytest

from popari import tl


def _fit_model(model, n_steps: int = 2):
    for _ in range(n_steps):
        model.estimate_parameters()
        model.estimate_weights()
    return model


@pytest.mark.baseline
def test_preprocess_and_pca(preprocessed_shared_model, shared_model_expected_metrics):
    model = preprocessed_shared_model
    metrics = shared_model_expected_metrics

    for dataset in model.datasets:
        assert "normalized_X" in dataset.obsm
        assert "X_pca" in dataset.obsm
        assert np.isfinite(dataset.obsm["normalized_X"]).all()
        assert np.isfinite(dataset.obsm["X_pca"]).all()
    assert np.linalg.norm(model.datasets[0].obsm["X_pca"]) == pytest.approx(metrics["pca_norms"][0], abs=1e-6)
    assert np.linalg.norm(model.datasets[1].obsm["X_pca"]) == pytest.approx(metrics["pca_norms"][1], abs=1e-6)


@pytest.mark.expensive
def test_clustering_metrics_and_classification(clustered_shared_model, shared_model_expected_metrics):
    model = clustered_shared_model
    metrics = shared_model_expected_metrics
    try:
        tl.compute_confusion_matrix(model, labels="cell_type", predictions="leiden", joint=True)
    except ValueError:
        pass

    for dataset in model.datasets:
        assert "leiden" in dataset.obs
        assert np.isfinite(dataset.uns["ari"])
        assert np.isfinite(dataset.uns["silhouette"])
        assert 0 <= dataset.uns["microprecision_validation"] <= 1
        assert 0 <= dataset.uns["macroprecision_validation"] <= 1
    assert model.datasets[0].uns["ari"] == pytest.approx(metrics["ari"][0], abs=1e-9)
    assert model.datasets[1].uns["ari"] == pytest.approx(metrics["ari"][1], abs=1e-9)
    assert model.datasets[0].uns["silhouette"] == pytest.approx(metrics["silhouette"][0], abs=1e-9)
    assert model.datasets[1].uns["silhouette"] == pytest.approx(metrics["silhouette"][1], abs=1e-9)
    assert model.datasets[0].uns["microprecision_validation"] == pytest.approx(
        metrics["microprecision_validation"][0],
        abs=1e-9,
    )
    assert model.datasets[1].uns["microprecision_validation"] == pytest.approx(
        metrics["microprecision_validation"][1],
        abs=1e-9,
    )
    assert model.datasets[0].uns["macroprecision_validation"] == pytest.approx(
        metrics["macroprecision_validation"][0],
        abs=1e-9,
    )
    assert model.datasets[1].uns["macroprecision_validation"] == pytest.approx(
        metrics["macroprecision_validation"][1],
        abs=1e-9,
    )


@pytest.mark.expensive
def test_embedding_and_spatial_summaries(analyzed_shared_model):
    model = analyzed_shared_model

    for dataset in model.datasets:
        assert "empirical_correlation" in dataset.uns
        assert "spatial_gene_correlation" in dataset.uns
        assert "neighbor_interactions" in dataset.uns
        assert "domain" in dataset.obs
        empirical = dataset.uns["empirical_correlation"][dataset.popari.name]
        assert empirical.shape[0] == empirical.shape[1] == model.K
        assert np.allclose(empirical, empirical.T)


@pytest.mark.gpu
@pytest.mark.expensive
def test_differential_analysis_helpers(differential_model_factory, gpu_context):
    model = _fit_model(
        differential_model_factory(torch_context=gpu_context, initial_context=gpu_context),
        n_steps=1,
    )

    genes = tl.find_differential_genes(model, top_gene_limit=2)
    assert genes

    tl.plot_gene_trajectories(model, list(genes)[:2], covariate_values=list(range(len(model.metagene_groups))))
    tl.plot_gene_activations(model, list(genes)[:2])
    top_pairs, correlations, variances = tl.normalized_affinity_trends(
        model,
        timepoint_values=list(range(len(model.datasets))),
    )

    assert top_pairs
    assert correlations
    assert variances
