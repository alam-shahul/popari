import numpy as np
import pytest
import torch

from popari.util import project_M


def _shared_group_and_mask(model):
    group_name, group_replicates = next(iter(model.metagene_groups.items()))
    replicate_mask = np.array([dataset.name in group_replicates for dataset in model.datasets], dtype=bool)
    first_dataset_name = group_replicates[0]
    return group_name, group_replicates, replicate_mask, first_dataset_name


@pytest.mark.baseline
def test_sigma_yx_update_matches_manual_residual_sum(shared_model_factory):
    model = shared_model_factory()
    model.parameter_optimizer.update_sigma_yx()

    manual_squared_loss = 0.0
    for y, dataset in zip(model.Ys, model.datasets):
        residual = torch.addmm(
            y.to_dense(),
            model.embedding_optimizer.embedding_state[dataset.name],
            model.parameter_optimizer.metagene_state[dataset.name].T,
            alpha=-1,
        )
        manual_squared_loss += torch.linalg.norm(residual, ord="fro").item() ** 2

    assert model.parameter_optimizer.sigma_yxs.shape == (len(model.datasets),)
    assert np.all(np.isfinite(model.parameter_optimizer.sigma_yxs))
    assert np.all(model.parameter_optimizer.sigma_yxs > 0)
    assert model.parameter_optimizer.nll_sigma_yx() == pytest.approx(manual_squared_loss, abs=1e-6)


@pytest.mark.baseline
def test_scale_metagenes_preserves_reconstruction_and_simplex_constraint(shared_model_factory):
    model = shared_model_factory()
    dataset = model.datasets[0]
    name = dataset.name

    original_metagenes = model.parameter_optimizer.metagene_state[name].clone()
    original_embedding = model.embedding_optimizer.embedding_state[name].clone()

    model.parameter_optimizer.metagene_state[name].mul_(3.0)
    model.parameter_optimizer.scale_metagenes()

    scaled_metagenes = model.parameter_optimizer.metagene_state[name]
    scaled_embedding = model.embedding_optimizer.embedding_state[name]

    assert torch.allclose(
        scaled_metagenes.sum(dim=0),
        torch.ones(scaled_metagenes.shape[1], device=scaled_metagenes.device, dtype=scaled_metagenes.dtype),
        atol=1e-6,
    )
    assert torch.allclose(scaled_metagenes, original_metagenes, atol=1e-6)
    assert torch.allclose(scaled_embedding, 3.0 * original_embedding, atol=1e-6)


@pytest.mark.baseline
def test_direct_estimate_m_reduces_group_loss(shared_model_factory):
    model = shared_model_factory()
    model.parameter_optimizer.update_sigma_yx()
    _, _, replicate_mask, first_dataset_name = _shared_group_and_mask(model)

    initial_m = model.parameter_optimizer.metagene_state[first_dataset_name].clone()
    initial_loss = model.parameter_optimizer.nll_M(initial_m, replicate_mask)

    updated_m = model.parameter_optimizer.estimate_M(
        initial_m.clone(),
        replicate_mask,
        n_epochs=200,
        simplex_projection_mode="exact",
    )
    updated_loss = model.parameter_optimizer.nll_M(updated_m, replicate_mask)

    assert np.isfinite(initial_loss)
    assert np.isfinite(updated_loss)
    assert updated_loss <= initial_loss + 1e-6
    assert torch.all(updated_m >= 0)
    assert torch.allclose(
        updated_m.sum(dim=0),
        torch.ones(updated_m.shape[1], device=updated_m.device, dtype=updated_m.dtype),
        atol=1e-5,
    )


@pytest.mark.baseline
def test_update_metagenes_shared_changes_values_and_preserves_simplex(shared_model_factory):
    model = shared_model_factory()
    before = model.parameter_optimizer.metagene_state.metagenes.clone()
    model.parameter_optimizer.update_sigma_yx()

    model.parameter_optimizer.update_metagenes(simplex_projection_mode="exact")

    after = model.parameter_optimizer.metagene_state.metagenes
    assert not torch.allclose(before, after)
    assert torch.all(after >= 0)
    assert torch.allclose(
        after.sum(dim=1),
        torch.ones((after.shape[0], after.shape[2]), device=after.device, dtype=after.dtype),
        atol=1e-5,
    )


@pytest.mark.baseline
def test_nll_metagenes_matches_groupwise_sum(shared_model_factory):
    model = shared_model_factory()
    model.parameter_optimizer.update_sigma_yx()

    manual_loss = 0.0
    for _, group_replicates in model.metagene_groups.items():
        first_dataset_name = group_replicates[0]
        replicate_mask = np.array([dataset.name in group_replicates for dataset in model.datasets], dtype=bool)
        m = model.parameter_optimizer.metagene_state[first_dataset_name]
        manual_loss += model.parameter_optimizer.nll_M(m, replicate_mask)

    assert model.parameter_optimizer.nll_metagenes().item() == pytest.approx(manual_loss, abs=1e-6)


@pytest.mark.baseline
def test_direct_estimate_sigma_x_inv_reduces_group_loss(shared_model_factory):
    model = shared_model_factory()
    model.parameter_optimizer.update_sigma_yx()
    group_name, group_replicates, replicate_mask, first_dataset_name = _shared_group_and_mask(model)

    sigma_x_inv = model.parameter_optimizer.spatial_affinity_state[first_dataset_name].clone()
    initial_loss = model.parameter_optimizer.nll_Sigma_x_inv(sigma_x_inv, replicate_mask)
    updated_sigma_x_inv, _ = model.parameter_optimizer.estimate_Sigma_x_inv(
        sigma_x_inv.clone(),
        replicate_mask,
        model.parameter_optimizer.spatial_affinity_state.optimizers[group_name],
        n_epochs=100,
        check_frequency=10,
        tol=1e-4,
    )
    updated_loss = model.parameter_optimizer.nll_Sigma_x_inv(updated_sigma_x_inv, replicate_mask)

    assert torch.isfinite(initial_loss)
    assert torch.isfinite(updated_loss)
    assert updated_loss.item() <= initial_loss.item() + 1e-4
    assert torch.allclose(updated_sigma_x_inv, updated_sigma_x_inv.T, atol=1e-6)


@pytest.mark.baseline
def test_reinitialize_spatial_affinities_rebuilds_optimizer_state(shared_model_factory):
    model = shared_model_factory()

    model.parameter_optimizer.reinitialize_spatial_affinities()

    assert model.parameter_optimizer.spatial_affinity_state.optimizers
    assert set(model.parameter_optimizer.spatial_affinity_state.optimizers) == set(model.spatial_affinity_groups)


@pytest.mark.baseline
def test_nll_spatial_affinities_matches_groupwise_sum(shared_model_factory):
    model = shared_model_factory()
    model.parameter_optimizer.update_sigma_yx()

    manual_loss = 0.0
    for _, group_replicates in model.spatial_affinity_groups.items():
        first_dataset_name = group_replicates[0]
        replicate_mask = np.array([dataset.name in group_replicates for dataset in model.datasets], dtype=bool)
        sigma_x_inv = model.parameter_optimizer.spatial_affinity_state[first_dataset_name]
        manual_loss += model.parameter_optimizer.nll_Sigma_x_inv(sigma_x_inv, replicate_mask).item()

    assert model.parameter_optimizer.nll_spatial_affinities().item() == pytest.approx(manual_loss, abs=1e-6)


@pytest.mark.baseline
def test_spatial_affinity_update_is_symmetric(shared_model_factory):
    model = shared_model_factory()
    model.parameter_optimizer.update_sigma_yx()
    model.parameter_optimizer.update_spatial_affinity()

    for dataset in model.datasets:
        affinity = model.parameter_optimizer.spatial_affinity_state[dataset.name].detach().cpu().numpy()
        assert np.allclose(affinity, affinity.T, atol=1e-6)


@pytest.mark.baseline
def test_update_spatial_affinity_differential_reaverages_group_bars(differential_model_factory):
    model = differential_model_factory()
    model.parameter_optimizer.update_sigma_yx()

    model.parameter_optimizer.update_spatial_affinity()

    for group_name, group_replicates in model.spatial_affinity_groups.items():
        expected = sum(model.parameter_optimizer.spatial_affinity_state[name] for name in group_replicates) / len(
            group_replicates,
        )
        assert torch.allclose(
            model.parameter_optimizer.spatial_affinity_state.spatial_affinity_bar[group_name],
            expected,
            atol=1e-6,
        )


@pytest.mark.baseline
def test_nll_embeddings_matches_sum_without_neighbors(shared_model_factory):
    model = shared_model_factory()
    model.parameter_optimizer.update_sigma_yx()

    manual_loss = 0.0
    for dataset_index, dataset in enumerate(model.datasets):
        manual_loss += model.embedding_optimizer.nll_weight_wonbr(
            model.Ys[dataset_index].to(model.embedding_optimizer.context["device"]),
            model.parameter_optimizer.metagene_state[dataset.name].to(model.embedding_optimizer.context["device"]),
            model.embedding_optimizer.embedding_state[dataset.name].to(model.embedding_optimizer.context["device"]),
            model.parameter_optimizer.sigma_yxs[dataset_index],
            model.parameter_optimizer.prior_x_modes[dataset_index],
            model.parameter_optimizer.prior_xs[dataset_index],
            dataset,
        )

    assert model.embedding_optimizer.nll_embeddings(use_neighbors=False).item() == pytest.approx(
        manual_loss.item(),
        abs=1e-6,
    )


@pytest.mark.baseline
def test_direct_estimate_weight_wonbr_reduces_loss(shared_model_factory):
    model = shared_model_factory()
    model.parameter_optimizer.update_sigma_yx()
    dataset_index = 0
    dataset = model.datasets[dataset_index]
    y = model.Ys[dataset_index].to(model.embedding_optimizer.context["device"])
    m = model.parameter_optimizer.metagene_state[dataset.name].to(model.embedding_optimizer.context["device"])
    x = model.embedding_optimizer.embedding_state[dataset.name].clone().to(model.embedding_optimizer.context["device"])
    sigma_yx = model.parameter_optimizer.sigma_yxs[dataset_index]
    prior_x_mode = model.parameter_optimizer.prior_x_modes[dataset_index]
    prior_x = model.parameter_optimizer.prior_xs[dataset_index]

    initial_loss = model.embedding_optimizer.nll_weight_wonbr(y, m, x, sigma_yx, prior_x_mode, prior_x, dataset)
    updated_loss, updated_x = model.embedding_optimizer.estimate_weight_wonbr(
        y,
        m,
        x,
        sigma_yx,
        prior_x_mode,
        prior_x,
        dataset,
    )

    assert np.isfinite(updated_loss)
    assert updated_loss <= initial_loss.item() + 1e-6
    assert torch.all(updated_x >= 0)


@pytest.mark.baseline
def test_update_embeddings_without_neighbors_changes_values(shared_model_factory):
    model = shared_model_factory()
    model.parameter_optimizer.update_sigma_yx()
    before = {
        dataset.name: model.embedding_optimizer.embedding_state[dataset.name].clone() for dataset in model.datasets
    }

    model.embedding_optimizer.update_embeddings(use_neighbors=False)

    for dataset in model.datasets:
        after = model.embedding_optimizer.embedding_state[dataset.name]
        assert not torch.allclose(before[dataset.name], after)
        assert torch.all(after >= 0)
        assert torch.isfinite(after).all()


@pytest.mark.baseline
def test_nll_embeddings_matches_sum_with_neighbors(shared_model_factory):
    model = shared_model_factory()
    model.parameter_optimizer.update_sigma_yx()

    manual_loss = 0.0
    for dataset_index, dataset in enumerate(model.datasets):
        manual_loss += model.embedding_optimizer.nll_weight_wnbr(
            model.Ys[dataset_index].to(model.embedding_optimizer.context["device"]),
            model.parameter_optimizer.metagene_state[dataset.name].to(model.embedding_optimizer.context["device"]),
            model.embedding_optimizer.embedding_state[dataset.name].to(model.embedding_optimizer.context["device"]),
            model.parameter_optimizer.sigma_yxs[dataset_index],
            model.parameter_optimizer.prior_x_modes[dataset_index],
            model.parameter_optimizer.prior_xs[dataset_index],
            dataset,
        )

    assert model.embedding_optimizer.nll_embeddings(use_neighbors=True).item() == pytest.approx(
        manual_loss,
        abs=1e-6,
    )


@pytest.mark.baseline
def test_direct_estimate_weight_wnbr_reduces_loss(shared_model_factory):
    model = shared_model_factory()
    model.parameter_optimizer.update_sigma_yx()
    dataset_index = 0
    dataset = model.datasets[dataset_index]
    y = model.Ys[dataset_index].to(model.embedding_optimizer.context["device"])
    m = model.parameter_optimizer.metagene_state[dataset.name].to(model.embedding_optimizer.context["device"])
    x = model.embedding_optimizer.embedding_state[dataset.name].clone().to(model.embedding_optimizer.context["device"])
    sigma_yx = model.parameter_optimizer.sigma_yxs[dataset_index]
    prior_x_mode = model.parameter_optimizer.prior_x_modes[dataset_index]
    prior_x = model.parameter_optimizer.prior_xs[dataset_index]

    initial_loss = model.embedding_optimizer.nll_weight_wnbr(y, m, x, sigma_yx, prior_x_mode, prior_x, dataset)
    updated_loss, updated_x = model.embedding_optimizer.estimate_weight_wnbr(
        y,
        m,
        x,
        sigma_yx,
        prior_x_mode,
        prior_x,
        dataset,
        tol=1e-4,
    )

    assert np.isfinite(updated_loss)
    assert updated_loss <= initial_loss + 1e-4
    assert torch.all(updated_x >= 0)


@pytest.mark.baseline
def test_update_embeddings_with_neighbors_changes_values(shared_model_factory):
    model = shared_model_factory()
    model.parameter_optimizer.update_sigma_yx()
    before = {
        dataset.name: model.embedding_optimizer.embedding_state[dataset.name].clone() for dataset in model.datasets
    }

    model.embedding_optimizer.update_embeddings(use_neighbors=True)

    for dataset in model.datasets:
        after = model.embedding_optimizer.embedding_state[dataset.name]
        assert not torch.allclose(before[dataset.name], after)
        assert torch.all(after >= 0)
        assert torch.isfinite(after).all()


@pytest.mark.baseline
def test_differential_reaverage_updates_group_averages(differential_model_factory):
    model = differential_model_factory()

    dataset_names = [dataset.name for dataset in model.datasets]
    for offset, dataset_name in enumerate(dataset_names, start=1):
        model.parameter_optimizer.metagene_state[dataset_name].fill_(float(offset))
        model.parameter_optimizer.spatial_affinity_state[dataset_name].fill_(float(offset))

    model.parameter_optimizer.metagene_state.reaverage()
    model.parameter_optimizer.spatial_affinity_state.reaverage()

    for group_name, group_replicates in model.metagene_groups.items():
        average = sum(model.parameter_optimizer.metagene_state[name] for name in group_replicates) / len(
            group_replicates,
        )
        expected = project_M(average, model.parameter_optimizer.M_constraint)
        assert torch.allclose(model.parameter_optimizer.metagene_state.M_bar[group_name], expected, atol=1e-6)

    for group_name, group_replicates in model.spatial_affinity_groups.items():
        expected = sum(model.parameter_optimizer.spatial_affinity_state[name] for name in group_replicates) / len(
            group_replicates,
        )
        assert torch.allclose(
            model.parameter_optimizer.spatial_affinity_state.spatial_affinity_bar[group_name],
            expected,
            atol=1e-6,
        )


@pytest.mark.baseline
def test_update_metagenes_differential_reaverages_group_bars(differential_model_factory):
    model = differential_model_factory()
    model.parameter_optimizer.update_sigma_yx()

    model.parameter_optimizer.update_metagenes(simplex_projection_mode="exact")

    for dataset in model.datasets:
        metagenes = model.parameter_optimizer.metagene_state[dataset.name]
        assert torch.all(metagenes >= 0)
        assert torch.allclose(
            metagenes.sum(dim=0),
            torch.ones(metagenes.shape[1], device=metagenes.device, dtype=metagenes.dtype),
            atol=1e-5,
        )

    for group_name, group_replicates in model.metagene_groups.items():
        average = sum(model.parameter_optimizer.metagene_state[name] for name in group_replicates) / len(
            group_replicates,
        )
        expected = project_M(average, model.parameter_optimizer.M_constraint)
        assert torch.allclose(model.parameter_optimizer.metagene_state.M_bar[group_name], expected, atol=1e-6)
