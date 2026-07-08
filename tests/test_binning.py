import numpy as np
import pytest

from popari._binning_utils import GridDownsampler, PartitionDownsampler
from popari.io import load_anndata
from popari.model import load_trained_model


@pytest.mark.baseline
def test_grid_binning_produces_valid_assignments(shared_model_factory, dataset_factory):
    model = shared_model_factory(datasets=dataset_factory(num_cells=64))
    downsampler = GridDownsampler()

    for dataset in model.datasets:
        binned_name = f"{dataset.popari.name}_level_0"
        key = f"bin_assignments_{binned_name}"
        binned_dataset, _ = downsampler.downsample(
            dataset,
            bin_assignments_key=key,
            chunks=2,
            downsample_rate=0.5,
        )

        assignments = binned_dataset.obsm[key].toarray()
        assert len(binned_dataset) < len(dataset)
        assert assignments.shape[1] == len(dataset)
        assert np.all(assignments.sum(axis=0) == 1)


@pytest.mark.baseline
def test_partition_binning_produces_valid_assignments(shared_model_factory, dataset_factory):
    model = shared_model_factory(datasets=dataset_factory(num_cells=64))
    downsampler = PartitionDownsampler()

    for dataset in model.datasets:
        binned_name = f"{dataset.popari.name}_level_0"
        key = f"bin_assignments_{binned_name}"
        binned_dataset, _ = downsampler.downsample(
            dataset,
            bin_assignments_key=key,
            downsample_rate=0.5,
            adjacency_list_key="adjacency_list",
        )

        assignments = binned_dataset.obsm[key].toarray()
        assert len(binned_dataset) < len(dataset)
        assert assignments.shape[1] == len(dataset)
        assert np.all(assignments.sum(axis=0) == 1)


@pytest.mark.expensive
def test_hierarchical_superresolution_is_finite(hierarchical_model_factory):
    model = hierarchical_model_factory(hierarchical_levels=2)
    model.estimate_parameters()
    model.estimate_weights()

    model.superresolve(n_epochs=2, tol=1e-6)

    for level in range(model.hierarchical_levels):
        assert np.isfinite(model.nll(level=level)).all()


@pytest.mark.expensive
def test_hierarchical_save_load_and_reload_expression(hierarchical_model_factory, tmp_path):
    model = hierarchical_model_factory(hierarchical_levels=2)
    raw_datasets = [dataset.copy() for dataset in model.hierarchy[0].datasets]

    trainable_path = tmp_path / "superresolved_results"
    untrainable_path = tmp_path / "untrainable_results"

    model.save_results(trainable_path, ignore_raw_data=False)
    model.save_results(untrainable_path, ignore_raw_data=True)

    reloaded_trainable = load_trained_model(trainable_path)
    reloaded_untrainable = load_trained_model(untrainable_path)
    reloaded_untrainable._reload_expression(raw_datasets)

    for level in range(model.hierarchical_levels):
        for original, restored in zip(model.hierarchy[level].datasets, reloaded_trainable.hierarchy[level].datasets):
            assert original.shape == restored.shape

    for dataset in reloaded_untrainable.hierarchy[0].datasets:
        assert dataset.X.sum() > 0


@pytest.mark.expensive
def test_load_anndata_roundtrip_for_saved_hierarchy(hierarchical_model_factory, tmp_path):
    model = hierarchical_model_factory(hierarchical_levels=2)
    filepath = tmp_path / "hierarchy_results"
    model.save_results(filepath, ignore_raw_data=False)

    datasets, replicate_names = load_anndata(filepath / "level_0.h5ad")
    assert len(datasets) == len(replicate_names)
