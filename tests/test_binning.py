import numpy as np
import pytest

from popari._binning_utils import GridDownsampler, PartitionDownsampler
from popari.io import load_anndata
from popari.model import load_trained_model


@pytest.mark.baseline
def test_grid_binning_produces_valid_assignments(shared_model_factory, adata_factory):
    model = shared_model_factory(adata=adata_factory(num_cells=64))
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
def test_partition_binning_produces_valid_assignments(shared_model_factory, adata_factory):
    model = shared_model_factory(adata=adata_factory(num_cells=64))
    downsampler = PartitionDownsampler()

    for dataset in model.datasets:
        binned_name = f"{dataset.popari.name}_level_0"
        key = f"bin_assignments_{binned_name}"
        binned_dataset, _ = downsampler.downsample(
            dataset,
            bin_assignments_key=key,
            downsample_rate=0.5,
        )

        assignments = binned_dataset.obsm[key].toarray()
        assert len(binned_dataset) < len(dataset)
        assert assignments.shape[1] == len(dataset)
        assert np.all(assignments.sum(axis=0) == 1)


@pytest.mark.gpu
@pytest.mark.expensive
def test_hierarchical_superresolution_is_finite(hierarchical_model_factory, gpu_context):
    model = hierarchical_model_factory(
        hierarchical_levels=2,
        torch_context=gpu_context,
        initial_context=gpu_context,
    )
    model.estimate_parameters()
    model.estimate_weights()

    model.superresolve(n_epochs=2, tol=1e-6, use_manual_gradients=False)

    for level in range(model.hierarchical_levels):
        assert np.isfinite(model.nll(level=level)).all()


@pytest.mark.expensive
def test_hierarchical_save_load_and_reload_expression(hierarchical_model_factory, tmp_path):
    model = hierarchical_model_factory(hierarchical_levels=2)
    raw_adata = model.adata.copy()

    trainable_path = tmp_path / "superresolved_results"
    untrainable_path = tmp_path / "untrainable_results"

    model.save_results(trainable_path, ignore_raw_data=False)
    model.save_results(untrainable_path, ignore_raw_data=True)

    reloaded_trainable = load_trained_model(trainable_path)
    reloaded_untrainable = load_trained_model(untrainable_path)
    reloaded_untrainable._reload_expression(raw_adata)

    for level in range(model.hierarchical_levels):
        assert model.hierarchy[level].adata.shape == reloaded_trainable.hierarchy[level].adata.shape

    assert reloaded_untrainable.adata.X.sum() > 0


@pytest.mark.expensive
def test_load_anndata_roundtrip_for_saved_hierarchy(hierarchical_model_factory, tmp_path):
    model = hierarchical_model_factory(hierarchical_levels=2)
    filepath = tmp_path / "hierarchy_results"
    model.save_results(filepath, ignore_raw_data=False)

    reloaded = load_anndata(filepath / "level_0.h5ad")

    assert reloaded.shape == model.hierarchy[0].adata.shape
    assert reloaded.popari.sample_names == tuple(model.replicate_names)
