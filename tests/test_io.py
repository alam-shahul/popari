import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_array

from popari.io import (
    convert_legacy_anndata,
    load_anndata,
    load_anndata_hierarchy,
    merge_anndata,
    normalize_anndata_hierarchy,
    save_anndata,
    unmerge_anndata,
)
from popari.model import load_trained_model


def test_convert_legacy_anndata_requires_adjacency_matrix():
    merged_dataset = ad.concat(
        [ad.AnnData(X=np.ones((2, 2))), ad.AnnData(X=np.ones((3, 2)))],
        label="batch",
        keys=["replicate_0", "replicate_1"],
    )

    with pytest.raises(KeyError, match="Missing spatial graph"):
        convert_legacy_anndata(merged_dataset)


def test_convert_legacy_anndata_reconstructs_sample_graphs():
    merged_dataset = ad.concat(
        [ad.AnnData(X=np.ones((2, 2))), ad.AnnData(X=np.ones((3, 2)))],
        label="batch",
        keys=["replicate_0", "replicate_1"],
    )
    merged_dataset.uns["adjacency_matrix"] = {
        "replicate_0": np.array([[0, 1], [1, 0]]),
        "replicate_1": np.eye(3),
    }

    canonical = convert_legacy_anndata(merged_dataset)

    assert canonical.popari.sample_names == ("replicate_0", "replicate_1")
    assert canonical.obs_names.is_unique
    assert canonical.uns["popari_schema_version"] == 1
    assert canonical.obsp["adjacency_matrix"].shape == (5, 5)
    assert canonical.obsp["adjacency_matrix"][:2, 2:].nnz == 0


@pytest.mark.baseline
def test_save_and_load_anndata_roundtrip(shared_model_factory, tmp_path):
    model = shared_model_factory()
    filepath = tmp_path / "results.h5ad"
    canonical = model.adata.copy()

    save_anndata(filepath, canonical)
    reloaded = load_anndata(filepath)
    datasets, replicate_names = unmerge_anndata(reloaded)

    assert replicate_names == model.replicate_names
    assert reloaded.popari.sample_names == tuple(model.replicate_names)
    assert len(datasets) == len(model.replicate_names)
    for sample, sample_adata in zip(model.replicate_names, datasets):
        indices = model.adata.popari.sample_indices(sample)
        assert sample_adata.popari.name == sample
        assert sample_adata.shape == (len(indices), model.adata.n_vars)
        assert np.allclose(sample_adata.obsm["X"], model.adata.obsm["X"][indices])
        assert np.allclose(
            sample_adata.uns["M"][sample],
            model.adata.uns["M"][sample],
        )
        assert np.allclose(
            sample_adata.uns["Sigma_x_inv"][sample],
            model.adata.uns["Sigma_x_inv"][sample],
        )


@pytest.mark.baseline
def test_save_anndata_ignore_raw_data(shared_model_factory, tmp_path):
    model = shared_model_factory()
    filepath = tmp_path / "results_ignore_raw.h5ad"

    canonical = save_anndata(
        filepath,
        model.adata,
        ignore_raw_data=True,
    )

    assert filepath.exists()
    assert canonical.X.nnz == 0


def test_load_anndata_hierarchy_returns_one_anndata_per_level(shared_model_factory, tmp_path):
    model = shared_model_factory()
    result_path = tmp_path / "hierarchy"
    result_path.mkdir()
    canonical = model.adata
    save_anndata(result_path / "level_0.h5ad", canonical)
    save_anndata(result_path / "level_1.h5ad", canonical)

    hierarchy = load_anndata_hierarchy(result_path)

    assert tuple(hierarchy) == (0, 1)
    assert all(isinstance(dataset, ad.AnnData) for dataset in hierarchy.values())
    assert hierarchy[0].popari.sample_names == tuple(model.replicate_names)


def test_normalize_anndata_hierarchy_removes_legacy_level_suffixes(shared_model_factory):
    model = shared_model_factory()
    fine = model.adata.copy()
    coarse = model.adata.copy()
    renames = {sample: f"{sample}_level_1" for sample in model.replicate_names}
    coarse.obs["batch"] = pd.Categorical(
        coarse.obs["batch"].astype(str).map(renames),
        categories=renames.values(),
        ordered=True,
    )
    for key in ("M", "Sigma_x_inv", "sigma_yx"):
        coarse.uns[key] = {renames[sample]: value for sample, value in coarse.uns[key].items()}

    hierarchy = normalize_anndata_hierarchy({0: fine, 1: coarse})

    assert hierarchy[1].popari.sample_names == tuple(model.replicate_names)
    assert tuple(hierarchy[1].uns["M"]) == tuple(model.replicate_names)
    assert tuple(hierarchy[1].uns["Sigma_x_inv"]) == tuple(model.replicate_names)


def test_save_and_load_supports_custom_sample_key(tmp_path):
    dataset = ad.AnnData(X=np.ones((4, 2)))
    dataset.obs_names = [f"cell_{index}" for index in range(4)]
    dataset.var_names = ["gene_0", "gene_1"]
    dataset.obs["library"] = pd.Categorical(
        ["sample_a", "sample_a", "sample_b", "sample_b"],
        categories=["sample_b", "sample_a"],
    )
    dataset.obsp["adjacency_matrix"] = csr_array(
        (np.ones(4), ([0, 1, 2, 3], [1, 0, 3, 2])),
        shape=(4, 4),
    )
    filepath = tmp_path / "custom_sample_key.h5ad"

    save_anndata(filepath, dataset, sample_key="library")
    reloaded = load_anndata(filepath)

    assert reloaded.popari.sample_key == "library"
    assert reloaded.popari.sample_names == ("sample_b", "sample_a")


@pytest.mark.baseline
def test_load_trained_model_roundtrip(shared_model_factory, tmp_path):
    model = shared_model_factory()
    filepath = tmp_path / "trained_model.h5ad"

    model.save_results(filepath, ignore_raw_data=False)
    reloaded = load_trained_model(filepath)

    assert reloaded.replicate_names == model.replicate_names
    assert reloaded.metagene_mode == model.metagene_mode
    assert reloaded.spatial_affinity_mode == model.spatial_affinity_mode

    assert np.allclose(reloaded.adata.obsm["X"], model.adata.obsm["X"])
    for sample in model.replicate_names:
        assert np.allclose(
            reloaded.adata.uns["M"][sample],
            model.adata.uns["M"][sample],
        )

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
        original = model.hierarchy[level].adata
        restored = reloaded.hierarchy[level].adata
        assert original.shape == restored.shape
        assert np.allclose(original.obsm["X"], restored.obsm["X"])


@pytest.mark.expensive
def test_reload_expression_restores_trainability(hierarchical_model_factory, tmp_path):
    model = hierarchical_model_factory(hierarchical_levels=2)
    raw_adata = model.hierarchy[0].adata.copy()

    filepath = tmp_path / "hierarchical_untrainable"
    model.save_results(filepath, ignore_raw_data=True)
    reloaded = load_trained_model(filepath)

    reloaded._reload_expression(raw_adata)

    assert reloaded.hierarchy[0].adata.X.sum() > 0
