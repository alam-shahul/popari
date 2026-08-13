import numpy as np
import pytest
import torch

from popari.io import save_anndata, save_anndata_hierarchy
from popari.model import load_trained_model
from popari.optim.spatial_affinity import SpatialAffinityState
from popari.schema import SPATIAL_FACTOR_EMBEDDING_KEY
from tests._training import train_model


def test_factorized_affinity_validates_rank(shared_model_factory):
    with pytest.raises(ValueError, match="1 <= rank <= K"):
        shared_model_factory(spatial_affinity_parameterization="factorized", spatial_affinity_rank=0)
    with pytest.raises(ValueError, match="1 <= rank <= K"):
        shared_model_factory(spatial_affinity_parameterization="factorized", spatial_affinity_rank=4)


def test_factorized_affinity_has_shared_transform_and_sample_interactions(differential_model_factory):
    model = differential_model_factory(spatial_affinity_parameterization="factorized", spatial_affinity_rank=2)
    state = model.hierarchy[0].spatial_affinity

    assert state.transform.shape == (2, 3)
    assert torch.all(state.transform >= 0)
    torch.testing.assert_close(state.transform.sum(dim=1), torch.ones(2, dtype=state.transform.dtype))
    assert len(state.interactions) == len(model.replicate_names)
    for sample in model.replicate_names:
        transform = state.transform_for_sample(sample)
        interaction = state.interaction_for_sample(sample)
        assert transform is state.transform
        torch.testing.assert_close(interaction, interaction.T)
        torch.testing.assert_close(state.for_sample(sample), transform.T @ interaction @ transform)


def test_factorized_affinity_gradients_reach_shared_transform_and_every_interaction(context):
    state = SpatialAffinityState(
        K=3,
        sample_names=("first", "second"),
        groups={"all": ["first", "second"]},
        mode="differential lookup",
        context=context,
        parameterization="factorized",
        rank=2,
    )
    with torch.no_grad():
        state.transform.copy_(torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]], **context))
        state.interactions["first"].copy_(torch.tensor([[1.0, 0.25], [0.25, -0.5]], **context))
        state.interactions["second"].copy_(torch.tensor([[0.5, -0.1], [-0.1, 0.75]], **context))

    loss = sum(state.for_parameter(name).square().sum() for name in state.parameter_names)
    loss.backward()

    assert state.transform.grad is not None
    assert torch.isfinite(state.transform.grad).all()
    assert torch.count_nonzero(state.transform.grad)
    for interaction in state.interactions.values():
        assert interaction.grad is not None
        assert torch.isfinite(interaction.grad).all()
        assert torch.count_nonzero(interaction.grad)


def test_factorized_affinity_projection_constrains_transforms_and_effective_scale(context):
    state = SpatialAffinityState(
        K=3,
        sample_names=("sample",),
        groups={"sample": ["sample"]},
        mode="differential lookup",
        context=context,
        parameterization="factorized",
        rank=2,
    )
    with torch.no_grad():
        state.transform.copy_(torch.tensor([[2.0, -1.0, 0.5], [0.2, 0.3, 0.5]], **context))
        state.interactions["sample"].copy_(torch.tensor([[4.0, 2.0], [-1.0, -3.0]], **context))

    state.project_(constraint="clamp", scaling=0.25)
    transform = state.transform_for_sample("sample")
    assert torch.all(transform >= 0)
    torch.testing.assert_close(transform.sum(dim=1), torch.ones(2, dtype=transform.dtype))
    interaction = state.interaction_for_sample("sample")
    torch.testing.assert_close(interaction, interaction.T)
    assert state.for_sample("sample").abs().max() <= 0.25
    with pytest.raises(ValueError, match="Centering is not supported"):
        state.project_(center=True)


def test_shared_factorized_affinity_uses_shared_transform_and_interaction(shared_model_factory):
    model = shared_model_factory(spatial_affinity_parameterization="factorized", spatial_affinity_rank=2)
    state = model.hierarchy[0].spatial_affinity

    assert len(state.interactions) == 1
    assert {state.parameter_for_sample(sample) for sample in model.replicate_names} == {"_default"}
    first, second = model.replicate_names
    assert state.transform_for_sample(first) is state.transform_for_sample(second)
    torch.testing.assert_close(state.for_sample(first), state.for_sample(second))


def test_factorized_affinity_training_preserves_constraints(differential_model_factory):
    model = differential_model_factory(spatial_affinity_parameterization="factorized", spatial_affinity_rank=2)
    train_model(model, spatial_affinity_epochs=2)
    state = model.hierarchy[0].spatial_affinity

    assert torch.isfinite(state.transform).all()
    assert torch.all(state.transform >= 0)
    torch.testing.assert_close(state.transform.sum(dim=1), torch.ones(2, dtype=state.transform.dtype))
    for sample in model.replicate_names:
        interaction = state.interaction_for_sample(sample)
        torch.testing.assert_close(interaction, interaction.T)
        assert torch.isfinite(state.for_sample(sample)).all()


def test_factorized_affinity_roundtrip(differential_model_factory, tmp_path):
    model = differential_model_factory(spatial_affinity_parameterization="factorized", spatial_affinity_rank=2)
    original = model.hierarchy[0].spatial_affinity
    filepath = tmp_path / "factorized_model.h5ad"

    save_anndata(filepath, model.materialize_results()[0])
    expected = np.empty((model.adata.n_obs, 2))
    normalized = model.hierarchy[0].embeddings / torch.linalg.norm(
        model.hierarchy[0].embeddings,
        dim=1,
        ord=1,
        keepdim=True,
    )
    for sample in model.replicate_names:
        indices = model.adata.popari.sample_indices(sample)
        expected[indices] = (
            (normalized[model.hierarchy[0].sample_indices[sample]] @ original.transform_for_sample(sample).T)
            .detach()
            .cpu()
            .numpy()
        )
    np.testing.assert_allclose(model.adata.obsm[SPATIAL_FACTOR_EMBEDDING_KEY], expected)

    restored = load_trained_model(filepath).hierarchy[0].spatial_affinity
    torch.testing.assert_close(restored.transform, original.transform)
    for sample in model.replicate_names:
        torch.testing.assert_close(restored.interaction_for_sample(sample), original.interaction_for_sample(sample))
        torch.testing.assert_close(restored.for_sample(sample), original.for_sample(sample))


def test_factorized_affinity_hierarchy_roundtrip(hierarchical_model_factory, tmp_path):
    model = hierarchical_model_factory(spatial_affinity_parameterization="factorized", spatial_affinity_rank=2)
    result_directory = tmp_path / "factorized_hierarchy"
    save_anndata_hierarchy(result_directory, model.materialize_results())
    reloaded = load_trained_model(result_directory)

    for level_index in range(model.hierarchical_levels):
        original = model.hierarchy[level_index].spatial_affinity
        restored = reloaded.hierarchy[level_index].spatial_affinity
        torch.testing.assert_close(restored.transform, original.transform)
        for sample in model.replicate_names:
            torch.testing.assert_close(restored.interaction_for_sample(sample), original.interaction_for_sample(sample))
            torch.testing.assert_close(restored.for_sample(sample), original.for_sample(sample))
