import numpy as np
import pytest
import torch

from popari.optim.embedding import _embedding_loss, estimate_weight_wnbr, estimate_weight_wonbr
from popari.optim.metagene import _metagene_loss, estimate_metagenes
from popari.optim.simplex_integral import integrate_of_exponential_over_simplex
from popari.optim.spatial_affinity import _spatial_affinity_loss, _spatial_affinity_losses, estimate_spatial_affinities
from popari.train import Trainer

pytestmark = [pytest.mark.baseline, pytest.mark.cheap]


def _spatial_affinity_optimizer(level):
    return torch.optim.Adam(
        [level.spatial_affinity.values[name] for name in level.spatial_affinity.parameter_names],
        lr=level.spatial_affinity_lr,
        betas=(0.5, 0.9),
    )


def _trainer(model):
    return Trainer(model, iterations=0)


def _sigma_yxs_numpy(model):
    return model.hierarchy[-1].sigma_yxs.detach().cpu().numpy()


def _shared_affinity_group_and_mask(model):
    group_name, group_replicates = next(iter(model.spatial_affinity_groups.items()))
    replicate_mask = np.array([sample in group_replicates for sample in model.replicate_names], dtype=bool)
    first_dataset_name = group_replicates[0]
    return group_name, group_replicates, replicate_mask, first_dataset_name


def _embedding_objective(level, sample_index, embedding, *, use_neighbors):
    sample = level.sample_names[sample_index]
    expression = level.Ys[sample_index].to(level.context["device"])
    metagenes = level.metagenes.to(level.context["device"])
    sigma_yx = level.sigma_yxs[sample_index]
    quadratic_factor = metagenes.T @ metagenes / sigma_yx.square()
    linear_factor = expression @ metagenes / sigma_yx.square()
    constant = torch.square(expression).sum() / sigma_yx.square()
    kwargs = {}
    if use_neighbors:
        global_normalized = level.embeddings / torch.linalg.norm(level.embeddings, dim=1, ord=1, keepdim=True)
        global_normalized = global_normalized.clone()
        normalized = embedding / torch.linalg.norm(embedding, dim=1, ord=1, keepdim=True)
        global_normalized.index_copy_(0, level.sample_indices[sample], normalized)
        kwargs = {
            "neighbor_embeddings": torch.index_select(
                level.adjacency_matrix,
                0,
                level.sample_indices[sample],
            )
            @ global_normalized,
            "spatial_affinity": level.spatial_affinity.for_sample(sample),
        }
    return _embedding_loss(
        embedding,
        quadratic_factor,
        linear_factor,
        constant,
        level.prior_x_modes[sample_index],
        level.prior_xs[sample_index],
        **kwargs,
    )


def _metagene_objective(level, metagenes, replicate_mask):
    selected = np.flatnonzero(replicate_mask)
    weights = level.betas[selected]
    weights = weights / weights.sum()
    loss = metagenes.new_zeros(())
    for weight, sample_index in zip(weights, selected):
        sample = level.sample_names[sample_index]
        residual = level.Ys[sample_index].to_dense() - level.embedding(sample) @ metagenes.T
        loss += weight * residual.square().sum() / (2 * level.sigma_yxs[sample_index].square())
    return loss


def _spatial_objective(level, spatial_affinity, replicate_mask):
    selected = np.flatnonzero(replicate_mask)
    samples = [level.sample_names[index] for index in selected]
    weights = level.betas[selected]
    weights = weights / weights.sum()
    normalized = level.embeddings / torch.linalg.norm(level.embeddings, dim=1, ord=1, keepdim=True)
    neighbor_sums = level.adjacency_matrix @ normalized
    linear_factor = torch.zeros_like(spatial_affinity)
    sample_neighbor_sums = []
    weighted_edge_count = spatial_affinity.new_zeros(())
    edge_counts = np.diff(level.adjacency.indptr)
    for sample, weight in zip(samples, weights):
        indices = level.sample_indices[sample]
        sample_normalized = normalized.index_select(0, indices)
        sample_neighbors = neighbor_sums.index_select(0, indices)
        linear_factor.addmm_(sample_normalized.T, sample_neighbors, alpha=weight)
        sample_neighbor_sums.append(sample_neighbors)
        weighted_edge_count += weight * edge_counts[level.sample_axis.indices(sample)].sum()
    return _spatial_affinity_loss(
        spatial_affinity,
        linear_factor,
        sample_neighbor_sums,
        weights,
        weighted_edge_count,
        regularization_strength=level.lambda_Sigma_x_inv,
        regularization_power=level.spatial_affinity_regularization_power,
    )


def _direct_joint_pseudolikelihood(level, *, use_spatial):
    loss = level.metagenes.new_zeros(())
    normalized = level.embeddings / level.embeddings.norm(dim=1, p=1, keepdim=True)
    neighbor_sums = level.adjacency_matrix @ normalized if use_spatial else None
    edge_counts = np.diff(level.adjacency.indptr)

    for sample_index, sample in enumerate(level.sample_names):
        expression = level.Ys[sample_index].to_dense()
        embedding = level.embedding(sample)
        sigma_yx = level.sigma_yxs[sample_index]
        residual = expression - embedding @ level.metagenes.T
        num_observations, num_genes = expression.shape
        loss += residual.square().sum() / (2 * sigma_yx.square())
        loss += num_observations * num_genes / 2 * torch.log(2 * torch.pi * sigma_yx.square())

        prior_x_mode = level.prior_x_modes[sample_index]
        if prior_x_mode == "exponential shared fixed":
            rate = level.prior_xs[sample_index][0][0]
            loss += rate * embedding.norm(dim=1, p=1).sum()
            loss -= num_observations * level.K * torch.log(rate)
            if use_spatial:
                loss += num_observations * torch.lgamma(rate.new_tensor(float(level.K)))

        if use_spatial:
            indices = level.sample_indices[sample]
            sample_normalized = normalized.index_select(0, indices)
            sample_neighbors = neighbor_sums.index_select(0, indices)
            affinity = level.spatial_affinity.for_sample(sample)
            eta = sample_neighbors @ affinity
            loss += (eta * sample_normalized).sum()
            loss += integrate_of_exponential_over_simplex(eta).sum()

    if not use_spatial:
        return loss

    if level.spatial_affinity_mode == "shared lookup":
        for group, samples in level.spatial_affinity_groups.items():
            affinity = level.spatial_affinity.values[group]
            edge_count = sum(int(edge_counts[level.sample_axis.indices(sample)].sum()) for sample in samples)
            loss += (
                edge_count
                * level.lambda_Sigma_x_inv
                * affinity.abs().pow(level.spatial_affinity_regularization_power).sum()
                / 2
            )
    else:
        group_means = level.spatial_affinity.group_means()
        for sample in level.sample_names:
            affinity = level.spatial_affinity.for_sample(sample)
            edge_count = int(edge_counts[level.sample_axis.indices(sample)].sum())
            regularization = (
                level.lambda_Sigma_x_inv * affinity.abs().pow(level.spatial_affinity_regularization_power).sum()
            )
            sample_groups = level.spatial_affinity_tags[sample]
            regularization += sum(
                level.lambda_Sigma_bar * (group_means[group] - affinity).square().sum() / len(sample_groups)
                for group in sample_groups
            )
            loss += edge_count * regularization / 2

    return loss


def test_embedding_loss_matches_direct_formula_and_is_differentiable():
    embedding = torch.tensor([[0.3, 0.7], [0.6, 0.4]], dtype=torch.float64, requires_grad=True)
    quadratic = torch.tensor([[2.0, 0.2], [0.2, 1.5]], dtype=torch.float64)
    linear = torch.tensor([[0.5, 0.8], [0.4, 0.6]], dtype=torch.float64)
    constant = torch.tensor(3.0, dtype=torch.float64)
    prior = (torch.tensor([0.4, 0.4], dtype=torch.float64),)
    neighbors = torch.tensor([[0.2, 0.8], [0.9, 0.1]], dtype=torch.float64)
    affinity = torch.tensor([[0.5, -0.2], [-0.2, 0.3]], dtype=torch.float64)

    loss = _embedding_loss(
        embedding,
        quadratic,
        linear,
        constant,
        "exponential shared fixed",
        prior,
        neighbor_embeddings=neighbors,
        spatial_affinity=affinity,
    )
    normalized = embedding / embedding.norm(dim=1, p=1, keepdim=True)
    expected = (
        ((embedding @ quadratic) * embedding).sum() / 2
        - (embedding * linear).sum()
        + constant / 2
        + prior[0][0] * embedding.norm(dim=1, p=1).sum()
        + (neighbors @ affinity).mul(normalized).sum() / 2
    )

    assert torch.allclose(loss, expected)
    gradient = torch.autograd.grad(loss, embedding)[0]
    assert gradient.shape == embedding.shape
    assert torch.isfinite(gradient).all()


def test_metagene_loss_matches_direct_formula_and_is_differentiable():
    metagenes = torch.tensor([[0.2, 0.8], [0.7, 0.3]], dtype=torch.float64, requires_grad=True)
    quadratic = torch.tensor([[1.5, 0.1], [0.1, 2.0]], dtype=torch.float64)
    linear = torch.tensor([[0.4, 0.6], [0.8, 0.2]], dtype=torch.float64)
    constant = torch.tensor(4.0, dtype=torch.float64)

    loss = _metagene_loss(metagenes, quadratic, linear, constant)
    expected = (((metagenes @ quadratic) * metagenes).sum() - 2 * (linear * metagenes).sum() + constant) / 2

    assert torch.allclose(loss, expected)
    gradient = torch.autograd.grad(loss, metagenes)[0]
    assert gradient.shape == metagenes.shape
    assert torch.isfinite(gradient).all()


def test_spatial_affinity_loss_matches_direct_formula_and_is_differentiable():
    affinity = torch.tensor([[-0.5, 0.2], [0.2, -0.1]], dtype=torch.float64, requires_grad=True)
    linear = torch.tensor([[0.3, -0.2], [-0.2, 0.4]], dtype=torch.float64)
    neighbor_sums = [
        torch.tensor([[0.7, 0.3], [0.2, 0.8]], dtype=torch.float64),
        torch.tensor([[0.4, 0.6]], dtype=torch.float64),
    ]
    weights = torch.tensor([0.6, 0.4], dtype=torch.float64)
    edge_count = torch.tensor(5.0, dtype=torch.float64)
    group_means = [torch.zeros_like(affinity)]

    loss = _spatial_affinity_loss(
        affinity,
        linear,
        neighbor_sums,
        weights,
        edge_count,
        regularization_strength=0.3,
        regularization_power=1,
        group_means=group_means,
        group_regularization_strength=0.2,
    )
    partition = sum(
        weight * integrate_of_exponential_over_simplex(neighbors @ affinity).sum()
        for neighbors, weight in zip(neighbor_sums, weights)
    )
    regularization = 0.3 * affinity.abs().sum() + 0.2 * affinity.square().sum()
    expected = (affinity.flatten() @ linear.flatten() + regularization * edge_count / 2 + partition) / edge_count

    assert torch.allclose(loss, expected)
    gradient = torch.autograd.grad(loss, affinity)[0]
    assert gradient.shape == affinity.shape
    assert torch.isfinite(gradient).all()


def test_batched_spatial_affinity_losses_match_scalar_losses_and_gradients():
    torch.manual_seed(0)
    affinities = torch.randn((2, 3, 3), dtype=torch.float64, requires_grad=True)
    linear_factors = torch.randn((2, 3, 3), dtype=torch.float64)
    neighbor_sums = torch.randn((2, 3, 3), dtype=torch.float64)
    observation_mask = torch.tensor([[True, True, False], [True, True, True]])
    observation_weights = observation_mask.to(torch.float64)
    edge_counts = torch.tensor([4.0, 6.0], dtype=torch.float64)
    group_means = torch.randn((2, 3, 3), dtype=torch.float64)
    group_membership = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64)

    batched = _spatial_affinity_losses(
        affinities,
        linear_factors,
        neighbor_sums,
        observation_weights,
        observation_mask,
        edge_counts,
        regularization_strength=0.2,
        regularization_power=2,
        group_means=group_means,
        group_membership=group_membership,
        group_regularization_strength=0.3,
    )
    batched.sum().backward()
    batched_gradient = affinities.grad.detach().clone()

    affinities.grad = None
    scalar = torch.stack(
        [
            _spatial_affinity_loss(
                affinities[parameter],
                linear_factors[parameter],
                [neighbor_sums[parameter, observation_mask[parameter]]],
                torch.ones(1, dtype=torch.float64),
                edge_counts[parameter],
                regularization_strength=0.2,
                regularization_power=2,
                group_means=[
                    group_means[group]
                    for group in torch.nonzero(group_membership[parameter], as_tuple=False).flatten().tolist()
                ],
                group_regularization_strength=0.3,
            )
            for parameter in range(2)
        ],
    )
    scalar.sum().backward()

    torch.testing.assert_close(batched, scalar)
    torch.testing.assert_close(batched_gradient, affinities.grad)


@pytest.mark.parametrize("use_spatial", [False, True])
def test_shared_forward_matches_direct_joint_pseudolikelihood(shared_model_factory, use_spatial):
    model = shared_model_factory(
        prior_x_modes=["exponential shared fixed"] * 2,
        spatial_affinity_regularization_power=1,
    )
    level = model.hierarchy[-1]

    expected = _direct_joint_pseudolikelihood(level, use_spatial=use_spatial)

    assert torch.allclose(level(use_spatial=use_spatial), expected)


def test_differential_forward_matches_direct_joint_pseudolikelihood(differential_model_factory):
    model = differential_model_factory()
    level = model.hierarchy[-1]

    expected = _direct_joint_pseudolikelihood(level, use_spatial=True)

    assert torch.allclose(level(use_spatial=True), expected)


def test_sigma_yx_update_matches_manual_residual_sum(shared_model_factory):
    model = shared_model_factory()
    level = model.hierarchy[-1]
    level._recompute_observation_noise()

    squared_losses = []
    for y, sample in zip(level.Ys, model.replicate_names):
        residual = torch.addmm(
            y.to_dense(),
            level.embedding(sample),
            level.metagenes.T,
            alpha=-1,
        )
        squared_losses.append(torch.linalg.norm(residual, ord="fro").square())

    squared_losses = torch.stack(squared_losses)
    sizes = torch.as_tensor([expression.numel() for expression in level.Ys], **level.context)
    if level.sigma_yx_inv_mode == "separate":
        expected = torch.sqrt(squared_losses / sizes)
    else:
        expected = torch.sqrt(torch.dot(level.betas, squared_losses) / torch.dot(level.betas, sizes)).expand_as(
            squared_losses,
        )

    sigma_yxs = _sigma_yxs_numpy(model)
    assert sigma_yxs.shape == (len(model.replicate_names),)
    assert np.all(np.isfinite(sigma_yxs))
    assert np.all(sigma_yxs > 0)
    assert np.allclose(sigma_yxs, expected.detach().cpu().numpy(), atol=1e-6)


def test_scale_metagenes_preserves_reconstruction_and_simplex_constraint(shared_model_factory):
    model = shared_model_factory()
    name = model.replicate_names[0]

    original_metagenes = model.hierarchy[-1].metagenes.clone()
    original_embedding = model.hierarchy[-1].embedding(name).clone()

    with torch.no_grad():
        model.hierarchy[-1].metagenes.mul_(3.0)
    model.hierarchy[-1]._normalize_factorization()

    scaled_metagenes = model.hierarchy[-1].metagenes
    scaled_embedding = model.hierarchy[-1].embedding(name)

    assert torch.allclose(
        scaled_metagenes.sum(dim=0),
        torch.ones(scaled_metagenes.shape[1], device=scaled_metagenes.device, dtype=scaled_metagenes.dtype),
        atol=1e-6,
    )
    assert torch.allclose(scaled_metagenes, original_metagenes, atol=1e-6)
    assert torch.allclose(scaled_embedding, 3.0 * original_embedding, atol=1e-6)


def test_direct_estimate_m_reduces_group_loss(shared_model_factory):
    model = shared_model_factory()
    model.hierarchy[-1]._recompute_observation_noise()
    replicate_mask = np.ones(len(model.replicate_names), dtype=bool)

    initial_m = model.hierarchy[-1].metagenes.clone()
    initial_loss = _metagene_objective(model.hierarchy[-1], initial_m, replicate_mask)

    updated_m, returned_loss = estimate_metagenes(
        model.hierarchy[-1],
        initial_m.clone(),
        replicate_mask,
        n_epochs=200,
        simplex_projection_mode="exact",
    )
    updated_loss = _metagene_objective(model.hierarchy[-1], updated_m, replicate_mask)

    assert torch.isfinite(initial_loss)
    assert torch.isfinite(updated_loss)
    assert returned_loss == pytest.approx(updated_loss.item(), abs=1e-6)
    assert updated_loss <= initial_loss + 1e-6
    assert torch.all(updated_m >= 0)
    assert torch.allclose(
        updated_m.sum(dim=0),
        torch.ones(updated_m.shape[1], device=updated_m.device, dtype=updated_m.dtype),
        atol=1e-5,
    )


def test_update_metagenes_shared_changes_values_and_preserves_simplex(shared_model_factory):
    model = shared_model_factory()
    before = model.hierarchy[-1].metagenes.clone()
    model.hierarchy[-1]._recompute_observation_noise()

    metrics = _trainer(model)._update_parameters(
        update_spatial_affinities=False,
        simplex_projection_mode="exact",
    )

    after = model.hierarchy[-1].metagenes
    assert not torch.allclose(before, after)
    assert np.isfinite(metrics["metagene_loss"])
    assert torch.all(after >= 0)
    assert torch.allclose(
        after.sum(dim=0),
        torch.ones(after.shape[1], device=after.device, dtype=after.dtype),
        atol=1e-5,
    )


def test_direct_estimate_sigma_x_inv_reduces_group_loss(shared_model_factory):
    model = shared_model_factory()
    model.hierarchy[-1]._recompute_observation_noise()
    group_name, group_replicates, replicate_mask, first_dataset_name = _shared_affinity_group_and_mask(model)

    sigma_x_inv = model.hierarchy[-1].spatial_affinity.for_sample(first_dataset_name)
    initial_loss = _spatial_objective(model.hierarchy[-1], sigma_x_inv, replicate_mask)
    estimate_spatial_affinities(
        model.hierarchy[-1],
        _spatial_affinity_optimizer(model.hierarchy[-1]),
        n_epochs=100,
        check_frequency=10,
        tol=1e-4,
    )
    updated_sigma_x_inv = model.hierarchy[-1].spatial_affinity.for_sample(first_dataset_name)
    updated_loss = _spatial_objective(model.hierarchy[-1], updated_sigma_x_inv, replicate_mask)

    assert torch.isfinite(initial_loss)
    assert torch.isfinite(updated_loss)
    assert updated_loss.item() <= initial_loss.item() + 1e-4
    assert torch.allclose(updated_sigma_x_inv, updated_sigma_x_inv.T, atol=1e-6)


def test_spatial_affinity_update_tracks_the_best_pre_step_state(shared_model_factory):
    model = shared_model_factory()
    level = model.hierarchy[-1]
    group_name, _, replicate_mask, sample = _shared_affinity_group_and_mask(model)
    affinity = level.spatial_affinity.for_sample(sample)
    initial_affinity = affinity.detach().clone()

    estimate_spatial_affinities(
        level,
        _spatial_affinity_optimizer(level),
        n_epochs=1,
    )

    torch.testing.assert_close(affinity, initial_affinity)


def test_spatial_affinity_update_rejects_nonfinite_initial_affinity(shared_model_factory):
    model = shared_model_factory()
    level = model.hierarchy[-1]
    group_name, _, replicate_mask, sample = _shared_affinity_group_and_mask(model)
    affinity = level.spatial_affinity.for_sample(sample)
    with torch.no_grad():
        affinity[0, 0] = torch.nan

    with pytest.raises(FloatingPointError, match="before optimization"):
        estimate_spatial_affinities(
            level,
            _spatial_affinity_optimizer(level),
            n_epochs=1,
        )


def test_spatial_affinity_update_rejects_nonfinite_fixed_neighbor_sums(shared_model_factory):
    model = shared_model_factory()
    level = model.hierarchy[-1]
    group_name, _, replicate_mask, sample = _shared_affinity_group_and_mask(model)
    with torch.no_grad():
        level.embeddings[0, 0] = torch.nan

    with pytest.raises(FloatingPointError, match="Neighbor sums"):
        estimate_spatial_affinities(
            level,
            _spatial_affinity_optimizer(level),
            n_epochs=1,
        )


def test_spatial_affinity_update_rejects_subsampling(shared_model_factory):
    model = shared_model_factory()
    level = model.hierarchy[-1]

    with pytest.raises(NotImplementedError, match="subsampling"):
        estimate_spatial_affinities(
            level,
            _spatial_affinity_optimizer(level),
            subsample_rate=0.5,
            n_epochs=1,
        )


def test_reinitialize_spatial_affinities_allows_fresh_optimizer_state(shared_model_factory):
    model = shared_model_factory()

    model.hierarchy[-1]._initialize_spatial_affinities()

    optimizer = _spatial_affinity_optimizer(model.hierarchy[-1])
    assert len(optimizer.param_groups[0]["params"]) == len(model.hierarchy[-1].spatial_affinity.parameter_names)


def test_spatial_affinity_update_is_symmetric(shared_model_factory):
    model = shared_model_factory()
    model.hierarchy[-1]._recompute_observation_noise()
    _trainer(model)._update_spatial_affinities(
        model.hierarchy[-1],
        _spatial_affinity_optimizer(model.hierarchy[-1]),
        differentiate=True,
    )

    for sample in model.replicate_names:
        affinity = model.hierarchy[-1].spatial_affinity.for_sample(sample).detach().cpu().numpy()
        assert np.allclose(affinity, affinity.T, atol=1e-6)


def test_update_spatial_affinity_differential_derives_group_means(differential_model_factory):
    model = differential_model_factory()
    model.hierarchy[-1]._recompute_observation_noise()

    _trainer(model)._update_spatial_affinities(
        model.hierarchy[-1],
        _spatial_affinity_optimizer(model.hierarchy[-1]),
        differentiate=True,
    )

    group_means = model.hierarchy[-1].spatial_affinity.group_means()
    for group_name, group_replicates in model.spatial_affinity_groups.items():
        expected = sum(model.hierarchy[-1].spatial_affinity.for_sample(name) for name in group_replicates) / len(
            group_replicates,
        )
        assert torch.allclose(
            group_means[group_name],
            expected,
            atol=1e-6,
        )


def test_differential_affinity_update_uses_one_frozen_group_mean_snapshot(differential_model_factory, monkeypatch):
    model = differential_model_factory()
    level = model.hierarchy[-1]
    original_group_means = level.spatial_affinity.group_means
    calls = 0

    def counted_group_means():
        nonlocal calls
        calls += 1
        return original_group_means()

    monkeypatch.setattr(level.spatial_affinity, "group_means", counted_group_means)
    _trainer(model)._update_spatial_affinities(
        level,
        _spatial_affinity_optimizer(level),
        differentiate=True,
        n_epochs=2,
    )
    assert calls == 1


def test_direct_estimate_weight_wonbr_reduces_loss(shared_model_factory):
    model = shared_model_factory()
    model.hierarchy[-1]._recompute_observation_noise()
    dataset_index = 0
    sample = model.replicate_names[dataset_index]
    y = model.hierarchy[-1].Ys[dataset_index].to(model.hierarchy[-1].context["device"])
    m = model.hierarchy[-1].metagenes.to(model.hierarchy[-1].context["device"])
    x = model.hierarchy[-1].embedding(sample).clone().to(model.hierarchy[-1].context["device"])
    sigma_yx = model.hierarchy[-1].sigma_yxs[dataset_index].item()
    prior_x_mode = model.hierarchy[-1].prior_x_modes[dataset_index]
    prior_x = model.hierarchy[-1].prior_xs[dataset_index]

    initial_loss = _embedding_objective(model.hierarchy[-1], dataset_index, x, use_neighbors=False)
    updated_loss, updated_x = estimate_weight_wonbr(
        model.hierarchy[-1],
        y,
        m,
        x,
        sigma_yx,
        prior_x_mode,
        prior_x,
    )

    assert np.isfinite(updated_loss)
    assert updated_loss <= initial_loss.item() + 1e-6
    assert torch.all(updated_x >= 0)


def test_update_embeddings_without_neighbors_changes_values(shared_model_factory):
    model = shared_model_factory()
    model.hierarchy[-1]._recompute_observation_noise()
    before = {sample: model.hierarchy[-1].embedding(sample).clone() for sample in model.replicate_names}

    _trainer(model)._update_embeddings(use_neighbors=False)

    for sample in model.replicate_names:
        after = model.hierarchy[-1].embedding(sample)
        assert not torch.allclose(before[sample], after)
        assert torch.all(after >= 0)
        assert torch.isfinite(after).all()


def test_direct_estimate_weight_wnbr_reduces_loss(shared_model_factory):
    model = shared_model_factory()
    model.hierarchy[-1]._recompute_observation_noise()
    dataset_index = 0
    sample = model.replicate_names[dataset_index]
    y = model.hierarchy[-1].Ys[dataset_index].to(model.hierarchy[-1].context["device"])
    m = model.hierarchy[-1].metagenes.to(model.hierarchy[-1].context["device"])
    x = model.hierarchy[-1].embedding(sample).clone().to(model.hierarchy[-1].context["device"])
    sigma_yx = model.hierarchy[-1].sigma_yxs[dataset_index].item()
    prior_x_mode = model.hierarchy[-1].prior_x_modes[dataset_index]
    prior_x = model.hierarchy[-1].prior_xs[dataset_index]

    initial_loss = _embedding_objective(model.hierarchy[-1], dataset_index, x, use_neighbors=True)
    updated_loss, updated_x = estimate_weight_wnbr(
        model.hierarchy[-1],
        y,
        m,
        x,
        sigma_yx,
        prior_x_mode,
        prior_x,
        sample,
        tol=1e-4,
    )

    assert np.isfinite(updated_loss)
    assert updated_loss <= initial_loss + 1e-4
    assert torch.all(updated_x >= 0)


def test_update_embeddings_with_neighbors_changes_values(shared_model_factory):
    model = shared_model_factory()
    model.hierarchy[-1]._recompute_observation_noise()
    before = {sample: model.hierarchy[-1].embedding(sample).clone() for sample in model.replicate_names}

    _trainer(model)._update_embeddings(use_neighbors=True)

    for sample in model.replicate_names:
        after = model.hierarchy[-1].embedding(sample)
        assert not torch.allclose(before[sample], after)
        assert torch.all(after >= 0)
        assert torch.isfinite(after).all()


def test_differential_affinity_group_means_reflect_current_values(differential_model_factory):
    model = differential_model_factory()

    dataset_names = model.replicate_names
    for offset, dataset_name in enumerate(dataset_names, start=1):
        with torch.no_grad():
            model.hierarchy[-1].spatial_affinity.for_sample(dataset_name).fill_(float(offset))

    group_means = model.hierarchy[-1].spatial_affinity.group_means()
    for group_name, group_replicates in model.spatial_affinity_groups.items():
        expected = sum(model.hierarchy[-1].spatial_affinity.for_sample(name) for name in group_replicates) / len(
            group_replicates,
        )
        assert torch.allclose(
            group_means[group_name],
            expected,
            atol=1e-6,
        )


def test_differential_affinity_group_means_support_overlapping_groups(
    adata_factory,
    differential_model_factory,
):
    sample_names = ["left", "center", "right"]
    model = differential_model_factory(
        adata=adata_factory(num_replicates=3, replicate_names=sample_names),
        spatial_affinity_groups={
            "left_pair": ["left", "center"],
            "right_pair": ["center", "right"],
        },
    )
    state = model.hierarchy[-1].spatial_affinity
    for value, sample in enumerate(sample_names, start=1):
        with torch.no_grad():
            state.for_sample(sample).fill_(value)

    means = state.group_means()

    assert state.tags["center"] == ["left_pair", "right_pair"]
    assert torch.all(means["left_pair"] == 1.5)
    assert torch.all(means["right_pair"] == 2.5)


def test_update_metagenes_with_differential_affinities_preserves_simplex(differential_model_factory):
    model = differential_model_factory()
    model.hierarchy[-1]._recompute_observation_noise()

    _trainer(model)._update_parameters(
        update_spatial_affinities=False,
        simplex_projection_mode="exact",
    )

    metagenes = model.hierarchy[-1].metagenes
    assert torch.all(metagenes >= 0)
    assert torch.allclose(
        metagenes.sum(dim=0),
        torch.ones(metagenes.shape[1], device=metagenes.device, dtype=metagenes.dtype),
        atol=1e-5,
    )
