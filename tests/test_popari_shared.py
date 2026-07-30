import numpy as np
import pytest

from popari import tl


@pytest.mark.gpu
@pytest.mark.baseline
def test_shared_parameter_updates_are_finite(shared_model_factory, gpu_context):
    model = shared_model_factory(torch_context=gpu_context, initial_context=gpu_context)
    initial_m = model.parameter_optimizer.metagene_state.metagenes.detach().cpu().numpy().copy()
    initial_sigma = model.parameter_optimizer.sigma_yxs.detach().cpu().numpy().copy()

    model.estimate_parameters()

    updated_m = model.parameter_optimizer.metagene_state.metagenes.detach().cpu().numpy()
    updated_sigma = model.parameter_optimizer.sigma_yxs.detach().cpu().numpy()
    assert np.isfinite(updated_m).all()
    assert np.isfinite(updated_sigma).all()
    assert not np.allclose(initial_m, updated_m)
    assert not np.allclose(initial_sigma, updated_sigma)


@pytest.mark.gpu
@pytest.mark.baseline
def test_shared_embedding_updates_preserve_nonnegativity(shared_model_factory, gpu_context):
    model = shared_model_factory(torch_context=gpu_context, initial_context=gpu_context)
    name = model.datasets[0].popari.name
    initial_x = model.embedding_optimizer.embedding_state[name].detach().cpu().numpy().copy()

    model.estimate_parameters()
    model.estimate_weights()

    updated_x = model.embedding_optimizer.embedding_state[name].detach().cpu().numpy()
    assert np.isfinite(updated_x).all()
    assert np.all(updated_x >= 0)
    assert not np.allclose(initial_x, updated_x)


@pytest.mark.baseline
def test_shared_nll_components_are_numerically_stable(trained_shared_model, shared_model_expected_metrics):
    model = trained_shared_model
    metrics = shared_model_expected_metrics

    assert model.nll(level=0)[0] == pytest.approx(metrics["nll"], abs=1e-6)
    assert model.parameter_optimizer.sigma_yxs[0].item() == pytest.approx(metrics["sigma_yx"][0], abs=1e-6)
    assert model.parameter_optimizer.sigma_yxs[1].item() == pytest.approx(metrics["sigma_yx"][1], abs=1e-6)
    assert model.parameter_optimizer.metagene_state.metagenes.detach().cpu().numpy().sum() == pytest.approx(
        metrics["metagene_sum"],
    )
    assert model.embedding_optimizer.embedding_state["0"].detach().cpu().numpy().sum() == pytest.approx(
        metrics["embedding_sum_0"],
        abs=1e-6,
    )
    assert model.parameter_optimizer.spatial_affinity["0"].detach().cpu().numpy().sum() == pytest.approx(
        metrics["spatial_affinity_sum_0"],
        abs=1e-6,
    )


@pytest.mark.expensive
def test_shared_analysis_pipeline_sets_expected_annotations(clustered_shared_model, shared_model_expected_metrics):
    model = clustered_shared_model
    metrics = shared_model_expected_metrics

    for dataset in model.datasets:
        assert "normalized_X" in dataset.obsm
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
def test_shared_confusion_matrix_requires_aligned_categories(clustered_shared_model):
    model = clustered_shared_model

    try:
        tl.compute_confusion_matrix(model.datasets, labels="cell_type", predictions="leiden", joint=True)
    except ValueError:
        return

    for dataset in model.datasets:
        assert "confusion_matrix" in dataset.uns
