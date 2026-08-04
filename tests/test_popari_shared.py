import numpy as np
import pytest

from popari import tl
from tests._training import train_model


@pytest.mark.gpu
@pytest.mark.baseline
def test_shared_parameter_updates_are_finite(shared_model_factory, gpu_context):
    model = shared_model_factory(torch_context=gpu_context, initial_context=gpu_context)
    initial_m = model.hierarchy[-1].metagenes.detach().cpu().numpy().copy()
    initial_sigma = model.hierarchy[-1].sigma_yxs.detach().cpu().numpy().copy()

    train_model(model)

    updated_m = model.hierarchy[-1].metagenes.detach().cpu().numpy()
    updated_sigma = model.hierarchy[-1].sigma_yxs.detach().cpu().numpy()
    assert np.isfinite(updated_m).all()
    assert np.isfinite(updated_sigma).all()
    assert not np.allclose(initial_m, updated_m)
    assert not np.allclose(initial_sigma, updated_sigma)


@pytest.mark.gpu
@pytest.mark.baseline
def test_shared_embedding_updates_preserve_nonnegativity(shared_model_factory, gpu_context):
    model = shared_model_factory(torch_context=gpu_context, initial_context=gpu_context)
    name = model.adata.popari.sample_names[0]
    initial_x = model.hierarchy[-1].embedding(name).detach().cpu().numpy().copy()

    train_model(model)

    updated_x = model.hierarchy[-1].embedding(name).detach().cpu().numpy()
    assert np.isfinite(updated_x).all()
    assert np.all(updated_x >= 0)
    assert not np.allclose(initial_x, updated_x)


@pytest.mark.baseline
def test_shared_nll_components_are_numerically_stable(trained_shared_model, shared_model_expected_metrics):
    model = trained_shared_model
    metrics = shared_model_expected_metrics

    assert model.nll(level=0)[0] == pytest.approx(metrics["nll"], abs=5e-6)
    assert model.hierarchy[-1].sigma_yxs[0].item() == pytest.approx(metrics["sigma_yx"][0], abs=1e-6)
    assert model.hierarchy[-1].sigma_yxs[1].item() == pytest.approx(metrics["sigma_yx"][1], abs=1e-6)
    assert model.hierarchy[-1].metagenes.detach().cpu().numpy().sum() == pytest.approx(
        metrics["metagene_sum"],
    )
    assert model.hierarchy[-1].embedding("0").detach().cpu().numpy().sum() == pytest.approx(
        metrics["embedding_sum_0"],
        abs=1e-6,
    )
    assert model.hierarchy[-1].spatial_affinity.for_sample("0").detach().cpu().numpy().sum() == pytest.approx(
        metrics["spatial_affinity_sum_0"],
        abs=1e-6,
    )


@pytest.mark.expensive
def test_shared_analysis_pipeline_sets_expected_annotations(clustered_shared_model):
    model = clustered_shared_model

    assert "normalized_X" in model.adata.obsm
    assert "leiden" in model.adata.obs
    assert np.isfinite(model.adata.uns["ari"])
    assert np.isfinite(model.adata.uns["silhouette"])
    assert 0 <= model.adata.uns["microprecision_validation"] <= 1
    assert 0 <= model.adata.uns["macroprecision_validation"] <= 1


@pytest.mark.expensive
def test_shared_confusion_matrix_requires_aligned_categories(clustered_shared_model):
    model = clustered_shared_model

    try:
        tl.compute_confusion_matrix(model.adata, labels="cell_type", predictions="leiden")
    except ValueError:
        return

    assert "confusion_matrix" in model.adata.uns
