import numpy as np
import pytest


@pytest.mark.baseline
def test_sigma_yx_update_matches_dataset_count(shared_model_factory):
    model = shared_model_factory()

    model.parameter_optimizer.update_sigma_yx()

    assert model.parameter_optimizer.sigma_yxs.shape == (len(model.datasets),)
    assert np.all(np.isfinite(model.parameter_optimizer.sigma_yxs))
    assert np.all(model.parameter_optimizer.sigma_yxs > 0)


@pytest.mark.baseline
def test_reinitialize_spatial_affinities_rebuilds_optimizer_state(shared_model_factory):
    model = shared_model_factory()

    model.parameter_optimizer.reinitialize_spatial_affinities()

    assert model.parameter_optimizer.spatial_affinity_state.optimizers
    assert set(model.parameter_optimizer.spatial_affinity_state.optimizers) == set(model.spatial_affinity_groups)


@pytest.mark.baseline
def test_metagene_update_preserves_simplex_constraint(shared_model_factory):
    model = shared_model_factory()
    model.parameter_optimizer.update_metagenes(simplex_projection_mode="exact")

    metagenes = model.parameter_optimizer.metagene_state.metagenes.detach().cpu().numpy()
    column_sums = metagenes.sum(axis=1)
    assert np.allclose(column_sums, 1.0, atol=1e-3)
    assert np.all(metagenes >= 0)


@pytest.mark.baseline
def test_embedding_update_without_neighbors_is_finite(shared_model_factory):
    model = shared_model_factory()
    model.parameter_optimizer.update_sigma_yx()

    model.embedding_optimizer.update_embeddings(use_neighbors=False)

    for dataset in model.datasets:
        embedding = model.embedding_optimizer.embedding_state[dataset.name].detach().cpu().numpy()
        assert np.isfinite(embedding).all()
        assert np.all(embedding >= 0)


@pytest.mark.baseline
def test_spatial_affinity_update_is_symmetric(shared_model_factory):
    model = shared_model_factory()
    model.parameter_optimizer.update_sigma_yx()
    model.parameter_optimizer.update_spatial_affinity()

    for dataset in model.datasets:
        affinity = model.parameter_optimizer.spatial_affinity_state[dataset.name].detach().cpu().numpy()
        assert np.allclose(affinity, affinity.T, atol=1e-6)
