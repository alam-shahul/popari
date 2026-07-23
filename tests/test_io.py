import anndata as ad
import numpy as np
import pytest

from popari.io import load_anndata, save_anndata, unmerge_anndata
from popari.model import load_trained_model


def test_unmerge_anndata_allows_missing_adjacency_matrix():
    merged_dataset = ad.concat(
        [ad.AnnData(X=np.ones((2, 2))), ad.AnnData(X=np.ones((3, 2)))],
        label="batch",
        keys=["replicate_0", "replicate_1"],
    )

    datasets, replicate_names = unmerge_anndata(merged_dataset)

    assert replicate_names == ["replicate_0", "replicate_1"]
    assert [dataset.popari.name for dataset in datasets] == replicate_names
    for dataset in datasets:
        assert "adjacency_matrix" not in dataset.obsp
        assert "adjacency_list" not in dataset.obsm


@pytest.mark.baseline
def test_save_and_load_anndata_roundtrip(shared_model_factory, tmp_path):
    model = shared_model_factory()
    filepath = tmp_path / "results.h5ad"

    save_anndata(filepath, model.datasets)
    datasets, replicate_names = load_anndata(filepath)

    assert replicate_names == model.replicate_names
    assert len(datasets) == len(model.datasets)
    for original, reloaded in zip(model.datasets, datasets):
        assert reloaded.popari.name == original.popari.name
        assert reloaded.shape == original.shape
        assert np.allclose(reloaded.obsm["X"], original.obsm["X"])
        assert np.allclose(
            reloaded.uns["M"][reloaded.popari.name],
            original.uns["M"][original.popari.name],
        )
        assert np.allclose(
            reloaded.uns["Sigma_x_inv"][reloaded.popari.name],
            original.uns["Sigma_x_inv"][original.popari.name],
        )


@pytest.mark.baseline
def test_save_anndata_ignore_raw_data(shared_model_factory, tmp_path):
    model = shared_model_factory()
    filepath = tmp_path / "results_ignore_raw.h5ad"

    datasets = save_anndata(filepath, model.datasets, ignore_raw_data=True)

    assert filepath.exists()
    assert datasets.X.nnz == 0


@pytest.mark.baseline
def test_load_trained_model_roundtrip(shared_model_factory, tmp_path):
    model = shared_model_factory()
    filepath = tmp_path / "trained_model.h5ad"

    model.save_results(filepath, ignore_raw_data=False)
    reloaded = load_trained_model(filepath)

    assert reloaded.replicate_names == model.replicate_names
    assert reloaded.metagene_mode == model.metagene_mode
    assert reloaded.spatial_affinity_mode == model.spatial_affinity_mode

    for original, restored in zip(model.datasets, reloaded.datasets):
        assert np.allclose(restored.obsm["X"], original.obsm["X"])
        assert np.allclose(restored.uns["M"][restored.popari.name], original.uns["M"][original.popari.name])

    assert np.isfinite(reloaded.nll(level=0)).all()


@pytest.mark.baseline
def test_load_differential_from_shared_file(shared_model_factory, tmp_path):
    model = shared_model_factory()
    filepath = tmp_path / "shared_model.h5ad"
    model.save_results(filepath, ignore_raw_data=False)

    differential = load_trained_model(
        filepath,
        metagene_mode="differential",
        spatial_affinity_mode="differential lookup",
    )

    assert differential.metagene_mode == "differential"
    assert differential.spatial_affinity_mode == "differential lookup"


@pytest.mark.gpu
@pytest.mark.expensive
def test_hierarchical_save_and_load_roundtrip(hierarchical_model_factory, gpu_context, tmp_path):
    model = hierarchical_model_factory(
        hierarchical_levels=2,
        torch_context=gpu_context,
        initial_context=gpu_context,
    )
    model.estimate_parameters()
    model.estimate_weights()
    model.superresolve(n_epochs=2, tol=1e-6)

    filepath = tmp_path / "hierarchical_results"
    model.save_results(filepath, ignore_raw_data=False)
    reloaded = load_trained_model(filepath)

    assert reloaded.hierarchical_levels == model.hierarchical_levels
    for level in range(model.hierarchical_levels):
        for original, restored in zip(model.hierarchy[level].datasets, reloaded.hierarchy[level].datasets):
            assert original.shape == restored.shape
            assert np.allclose(original.obsm["X"], restored.obsm["X"])


@pytest.mark.expensive
def test_reload_expression_restores_trainability(hierarchical_model_factory, tmp_path):
    model = hierarchical_model_factory(hierarchical_levels=2)
    raw_datasets = [dataset.copy() for dataset in model.hierarchy[0].datasets]

    filepath = tmp_path / "hierarchical_untrainable"
    model.save_results(filepath, ignore_raw_data=True)
    reloaded = load_trained_model(filepath)

    reloaded._reload_expression(raw_datasets)

    for dataset in reloaded.hierarchy[0].datasets:
        assert dataset.X.sum() > 0
