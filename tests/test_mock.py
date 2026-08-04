import inspect

import pytest
import torch
import torch.nn as nn

from popari.model import Popari, load_pretrained
from popari.train import Trainer

pytestmark = [pytest.mark.baseline, pytest.mark.cheap]


def test_popari_constructor_exposes_only_unified_inputs():
    parameters = inspect.signature(Popari).parameters

    assert "adata" in parameters
    assert "dataset_path" in parameters
    assert "datasets" not in parameters
    assert "replicate_names" not in parameters


def test_popari_init(shared_mock_model, mock_datasets):
    shared_mock_model.materialize_results()
    assert shared_mock_model.adata.n_obs == sum(dataset.n_obs for dataset in mock_datasets)
    assert shared_mock_model.adata.obsm["X"].shape == (
        shared_mock_model.adata.n_obs,
        shared_mock_model.K,
    )
    assert shared_mock_model.hierarchy[-1].level == 0
    assert shared_mock_model.hierarchy[0] is shared_mock_model.hierarchy[-1]


def test_popari_uses_configured_sample_key(shared_model_factory, adata_factory):
    model = shared_model_factory(
        adata=adata_factory(replicate_names=["alpha", "beta"], sample_key="library"),
        sample_key="library",
    )

    assert model.sample_key == "library"
    assert model.replicate_names == ["alpha", "beta"]
    assert model.adata.obs["library"].cat.categories.tolist() == ["alpha", "beta"]


@pytest.mark.gpu
def test_popari_registers_parameters_and_state_dict_roundtrip(shared_model_factory, gpu_context):
    model = shared_model_factory(torch_context=gpu_context, initial_context=gpu_context)
    level = model.hierarchy[-1]
    level._recompute_observation_noise()
    trainer = Trainer(model, iterations=0)
    trainer._update_parameters(update_spatial_affinities=False)
    trainer._update_embeddings()

    assert isinstance(model, nn.Module)
    parameter_names = dict(model.named_parameters())
    buffer_names = dict(model.named_buffers())

    assert any(name.endswith("metagenes") for name in parameter_names)
    assert any(name.endswith("embeddings") for name in parameter_names)
    assert any("spatial_affinity.values" in name for name in parameter_names)
    assert any(name.endswith("sigma_yxs") for name in buffer_names)
    assert any(name.endswith("betas") for name in buffer_names)

    reloaded_model = shared_model_factory(torch_context=gpu_context, initial_context=gpu_context)
    reloaded_model.load_state_dict(model.state_dict())

    assert reloaded_model.hierarchy[-1].sigma_yxs.detach().cpu().numpy() == pytest.approx(
        model.hierarchy[-1].sigma_yxs.detach().cpu().numpy(),
        abs=1e-9,
    )
    assert reloaded_model.hierarchy[-1].metagenes.detach().cpu().numpy() == pytest.approx(
        model.hierarchy[-1].metagenes.detach().cpu().numpy(),
        abs=1e-9,
    )
    assert reloaded_model.hierarchy[-1].embedding("0").detach().cpu().numpy() == pytest.approx(
        model.hierarchy[-1].embedding("0").detach().cpu().numpy(),
        abs=1e-9,
    )


def test_module_to_updates_global_adjacency_buffer(shared_model_factory):
    model = shared_model_factory()
    adjacency_before = model.hierarchy[-1].adjacency_matrix
    assert adjacency_before.is_sparse
    assert not adjacency_before.requires_grad

    model.to(dtype=torch.float32)

    adjacency_after = model.hierarchy[-1].adjacency_matrix
    assert not adjacency_after.requires_grad
    assert adjacency_after.dtype == torch.float32


def test_load_pretrained_preserves_global_adjacency_buffer(shared_model_factory, context):
    trained_model = shared_model_factory(torch_context=context, initial_context=context)
    trained_model.materialize_results()
    adata = trained_model.adata.copy()
    reloaded_model = load_pretrained(
        adata,
        context=context,
        reloaded_hierarchy={0: adata},
    )

    adjacency_matrix = reloaded_model.hierarchy[-1].adjacency_matrix
    assert adjacency_matrix.is_sparse
    assert not adjacency_matrix.requires_grad


def test_state_dict_reload_rejects_reordered_datasets(shared_model_factory, adata_factory):
    model = shared_model_factory(adata=adata_factory(replicate_names=["alpha", "beta"]))
    state_dict = model.state_dict()

    reversed_adata = adata_factory(replicate_names=["beta", "alpha"])
    mismatched_model = shared_model_factory(adata=reversed_adata)

    with pytest.raises(RuntimeError, match="samples"):
        mismatched_model.load_state_dict(state_dict)
