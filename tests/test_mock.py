import pytest


@pytest.mark.baseline
def test_popari_init(shared_mock_model, mock_datasets):
    assert len(shared_mock_model.datasets) == len(mock_datasets)
    assert shared_mock_model.base_view.level == 0
    assert shared_mock_model.hierarchy[0] is shared_mock_model.base_view
    assert shared_mock_model.parameter_optimizer is shared_mock_model.active_view.parameter_optimizer
    assert shared_mock_model.embedding_optimizer is shared_mock_model.active_view.embedding_optimizer
