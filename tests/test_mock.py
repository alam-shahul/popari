import pytest
import torch
import torch.nn as nn

from popari.model import load_pretrained

pytestmark = [pytest.mark.baseline, pytest.mark.cheap]


def test_popari_init(shared_mock_model, mock_datasets):
    assert len(shared_mock_model.datasets) == len(mock_datasets)
    assert shared_mock_model.base_view.level == 0
    assert shared_mock_model.hierarchy[0] is shared_mock_model.base_view
    assert shared_mock_model.parameter_optimizer is shared_mock_model.base_view.parameter_optimizer
    assert shared_mock_model.embedding_optimizer is shared_mock_model.base_view.embedding_optimizer


def test_popari_registers_parameters_and_state_dict_roundtrip(shared_model_factory):
    model = shared_model_factory()
    model.estimate_parameters()
    model.estimate_weights()

    assert isinstance(model, nn.Module)
    parameter_names = dict(model.named_parameters())
    buffer_names = dict(model.named_buffers())

    assert any(name.endswith("metagene_state.metagenes") for name in parameter_names)
    assert any("embedding_state.embedding_dict" in name for name in parameter_names)
    assert any("spatial_affinity.spatial_affinity_dict" in name for name in parameter_names)
    assert any(name.endswith("parameter_optimizer.sigma_yxs") for name in buffer_names)
    assert any(name.endswith("betas") for name in buffer_names)

    reloaded_model = shared_model_factory()
    reloaded_model.load_state_dict(model.state_dict())

    assert reloaded_model.parameter_optimizer.sigma_yxs.detach().cpu().numpy() == pytest.approx(
        model.parameter_optimizer.sigma_yxs.detach().cpu().numpy(),
        abs=1e-9,
    )
    assert reloaded_model.parameter_optimizer.metagene_state.metagenes.detach().cpu().numpy() == pytest.approx(
        model.parameter_optimizer.metagene_state.metagenes.detach().cpu().numpy(),
        abs=1e-9,
    )
    assert reloaded_model.embedding_optimizer.embedding_state["0"].detach().cpu().numpy() == pytest.approx(
        model.embedding_optimizer.embedding_state["0"].detach().cpu().numpy(),
        abs=1e-9,
    )


def test_module_to_updates_adjacency_parameters(shared_model_factory):
    model = shared_model_factory()
    dataset_name = model.datasets[0].popari.name()
    adjacency_before = model.parameter_optimizer.adjacency_matrices[dataset_name]
    assert isinstance(adjacency_before, nn.Parameter)
    assert not adjacency_before.requires_grad

    model.to(dtype=torch.float32)

    adjacency_after = model.parameter_optimizer.adjacency_matrices[dataset_name]
    assert isinstance(adjacency_after, nn.Parameter)
    assert not adjacency_after.requires_grad
    assert adjacency_after.dtype == torch.float32


def test_load_pretrained_preserves_adjacency_parameters(shared_model_factory, context):
    trained_model = shared_model_factory(torch_context=context, initial_context=context)
    trained_model.synchronize_datasets()
    datasets = [dataset.copy().popari.ensure_name(dataset.popari.name()) for dataset in trained_model.datasets]
    reloaded_model = load_pretrained(
        datasets,
        [dataset.popari.name() for dataset in datasets],
        context=context,
        reloaded_hierarchy={0: datasets},
    )

    dataset_name = reloaded_model.datasets[0].popari.name()
    adjacency_matrix = reloaded_model.parameter_optimizer.adjacency_matrices[dataset_name]
    assert isinstance(adjacency_matrix, nn.Parameter)
    assert not adjacency_matrix.requires_grad


def test_state_dict_reload_rejects_reordered_datasets(shared_model_factory, dataset_factory):
    model = shared_model_factory(replicate_names=["alpha", "beta"])
    state_dict = model.state_dict()

    reversed_datasets = list(reversed(dataset_factory(replicate_names=["alpha", "beta"])))
    mismatched_model = shared_model_factory(datasets=reversed_datasets)

    with pytest.raises(RuntimeError, match="datasets"):
        mismatched_model.load_state_dict(state_dict)
