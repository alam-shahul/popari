from unittest.mock import MagicMock, patch

from popari.model import Popari


def test_popari_init(shared_mock_model, mock_datasets):
    # mock_hierarchy = MagicMock()
    # mock_hierarchical_view = MagicMock()

    # mock_hierarchy.__getitem__.side_effect = lambda level: print(10)
    # mock_hierarchical_view.datasets = mock_datasets

    # MockHierarchy.return_value = mock_hierarchy
    # MockHierarchicalView.return_value = mock_hierarchical_view

    assert len(shared_mock_model.datasets) == 2
