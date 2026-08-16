import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.sparse import csr_array

from popari.io import load_anndata, load_anndata_hierarchy, save_anndata, save_anndata_hierarchy
from popari.legacy_io import convert_legacy_anndata, normalize_anndata_hierarchy
from popari.model import load_trained_model
from popari.schema import validate_anndata_hierarchy
from scripts.migrate_popari_artifact import migrate_artifact
from tests._training import train_model


def test_convert_legacy_anndata_requires_adjacency_matrix():
    merged_dataset = ad.concat(
        [ad.AnnData(X=np.ones((2, 2))), ad.AnnData(X=np.ones((3, 2)))],
        label="batch",
        keys=["replicate_0", "replicate_1"],
    )

    with pytest.raises(KeyError, match="Missing spatial graph"):
        convert_legacy_anndata(merged_dataset)


def test_load_anndata_rejects_legacy_artifact(tmp_path):
    dataset = ad.AnnData(X=np.ones((2, 2)))
    dataset.obs["batch"] = pd.Categorical(["sample", "sample"])
    dataset.obsp["adjacency_matrix"] = csr_array((2, 2))
    path = tmp_path / "legacy.h5ad"
    dataset.write_h5ad(path)

    with pytest.raises(ValueError, match="migrate_popari_artifact.py"):
        load_anndata(path)


def test_migrate_artifact_writes_canonical_h5ad(tmp_path):
    legacy = ad.AnnData(X=np.ones((2, 2)))
    legacy.obs_names = ["cell_0", "cell_1"]
    legacy.obs["batch"] = pd.Categorical(["sample", "sample"])
    legacy.uns["adjacency_matrix"] = {"sample": np.array([[0, 1], [1, 0]])}
    source = tmp_path / "legacy.h5ad"
    destination = tmp_path / "canonical.h5ad"
    legacy.write_h5ad(source)

    migrate_artifact(source, destination)

    migrated = load_anndata(destination)
    assert migrated.popari.sample_names == ("sample",)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        migrate_artifact(source, destination)


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
    assert canonical.uns["popari_schema_version"] == 2
    assert canonical.obsp["adjacency_matrix"].shape == (5, 5)
    assert canonical.obsp["adjacency_matrix"][:2, 2:].nnz == 0


def test_convert_legacy_anndata_collapses_shared_metagene_mappings():
    dataset = ad.AnnData(X=np.ones((4, 3)))
    dataset.obs["batch"] = pd.Categorical(["first", "first", "second", "second"])
    dataset.obsp["adjacency_matrix"] = csr_array((4, 4))
    metagenes = np.arange(6).reshape(3, 2)
    dataset.uns["M"] = {"first": metagenes, "second": metagenes.copy()}
    dataset.uns["ground_truth_M"] = {"first": metagenes, "second": metagenes.copy()}

    canonical = convert_legacy_anndata(dataset)

    np.testing.assert_array_equal(canonical.uns["M"], metagenes)
    np.testing.assert_array_equal(canonical.uns["ground_truth_M"], metagenes)


def test_convert_legacy_anndata_rejects_differential_metagenes():
    dataset = ad.AnnData(X=np.ones((4, 3)))
    dataset.obs["batch"] = pd.Categorical(["first", "first", "second", "second"])
    dataset.obsp["adjacency_matrix"] = csr_array((4, 4))
    dataset.uns["M"] = {
        "first": np.ones((3, 2)),
        "second": np.full((3, 2), 2),
    }

    with pytest.raises(ValueError, match="Differential metagenes are no longer supported"):
        convert_legacy_anndata(dataset)


def test_convert_legacy_anndata_reconstructs_duplicate_observation_names():
    merged_dataset = ad.AnnData(X=np.ones((4, 2)))
    merged_dataset.obs_names = ["Astro", "Astro", "Astro", "Oligo"]
    merged_dataset.obs["batch"] = pd.Categorical(
        ["replicate_0", "replicate_0", "replicate_1", "replicate_1"],
        categories=["replicate_0", "replicate_1"],
        ordered=True,
    )
    merged_dataset.uns["adjacency_matrix"] = {
        "replicate_0": np.array([[0, 1], [1, 0]]),
        "replicate_1": np.array([[0, 1], [1, 0]]),
    }

    canonical = convert_legacy_anndata(merged_dataset)

    assert canonical.obs_names.tolist() == [
        "replicate_0:0",
        "replicate_0:1",
        "replicate_1:0",
        "replicate_1:1",
    ]
    assert canonical.obs["_legacy_obs_name"].tolist() == [
        "Astro",
        "Astro",
        "Astro",
        "Oligo",
    ]
    assert canonical.obs["batch"].astype(str).tolist() == [
        "replicate_0",
        "replicate_0",
        "replicate_1",
        "replicate_1",
    ]


@pytest.mark.baseline
def test_save_and_load_anndata_roundtrip(shared_model_factory, tmp_path):
    model = shared_model_factory()
    model.materialize_results()
    filepath = tmp_path / "results.h5ad"
    canonical = model.adata.copy()

    save_anndata(filepath, canonical)
    reloaded = load_anndata(filepath)

    assert reloaded.popari.sample_names == tuple(model.replicate_names)
    assert reloaded.shape == model.adata.shape
    assert reloaded.obs_names.equals(model.adata.obs_names)
    assert np.allclose(reloaded.obsm["X"], model.adata.obsm["X"])
    assert np.allclose(reloaded.uns["M"], model.adata.uns["M"])
    for sample in model.replicate_names:
        assert np.array_equal(
            reloaded.popari.sample_indices(sample),
            model.adata.popari.sample_indices(sample),
        )
        assert np.allclose(
            reloaded.uns["Sigma_x_inv"][sample],
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
    coarse = canonical.copy()
    coarse.obsm["bin_assignments"] = csr_array(np.eye(canonical.n_obs))
    save_anndata(result_path / "level_0.h5ad", canonical)
    save_anndata(result_path / "level_1.h5ad", coarse)

    hierarchy = load_anndata_hierarchy(result_path)

    assert tuple(hierarchy) == (0, 1)
    assert all(isinstance(dataset, ad.AnnData) for dataset in hierarchy.values())
    assert hierarchy[0].popari.sample_names == tuple(model.replicate_names)


def test_normalize_anndata_hierarchy_removes_legacy_level_suffixes(shared_model_factory):
    model = shared_model_factory()
    model.materialize_results()
    fine = model.adata.copy()
    coarse = model.adata.copy()
    renames = {sample: f"{sample}_level_1" for sample in model.replicate_names}
    coarse.obs["batch"] = pd.Categorical(
        coarse.obs["batch"].astype(str).map(renames),
        categories=renames.values(),
        ordered=True,
    )
    for key in ("Sigma_x_inv", "sigma_yx"):
        coarse.uns[key] = {renames[sample]: value for sample, value in coarse.uns[key].items()}

    hierarchy = normalize_anndata_hierarchy({0: fine, 1: coarse})

    assert hierarchy[1].popari.sample_names == tuple(model.replicate_names)
    np.testing.assert_array_equal(hierarchy[1].uns["M"], model.adata.uns["M"])
    assert tuple(hierarchy[1].uns["Sigma_x_inv"]) == tuple(model.replicate_names)


def test_normalize_anndata_hierarchy_combines_legacy_bin_assignments():
    fine = ad.AnnData(X=np.ones((5, 2)))
    fine.obs_names = [f"fine_{index}" for index in range(5)]
    fine.obs["batch"] = pd.Categorical(
        ["sample_a", "sample_a", "sample_b", "sample_b", "sample_b"],
        categories=["sample_a", "sample_b"],
        ordered=True,
    )
    fine.uns["adjacency_matrix"] = {
        "sample_a": np.eye(2),
        "sample_b": np.eye(3),
    }
    fine = convert_legacy_anndata(fine)

    coarse = ad.AnnData(X=np.ones((3, 2)))
    coarse.obs_names = [f"coarse_{index}" for index in range(3)]
    coarse.obs["batch"] = pd.Categorical(
        ["sample_a_level_1", "sample_b_level_1", "sample_b_level_1"],
        categories=["sample_a_level_1", "sample_b_level_1"],
        ordered=True,
    )
    coarse.uns["adjacency_matrix"] = {
        "sample_a_level_1": np.eye(1),
        "sample_b_level_1": np.eye(2),
    }
    coarse.obsm["bin_assignments_sample_a_level_1"] = csr_array(
        [[1, 1], [0, 0], [0, 0]],
    )
    coarse.obsm["bin_assignments_sample_b_level_1"] = csr_array(
        [[0, 0, 0], [1, 1, 0], [0, 0, 1]],
    )
    coarse = convert_legacy_anndata(coarse)

    hierarchy = normalize_anndata_hierarchy({0: fine, 1: coarse})

    assert hierarchy[1].popari.sample_names == ("sample_a", "sample_b")
    assert list(hierarchy[1].obsm) == ["bin_assignments"]
    assert np.array_equal(
        hierarchy[1].obsm["bin_assignments"].toarray(),
        np.array(
            [
                [1, 1, 0, 0, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 0, 1],
            ],
        ),
    )


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


def test_materialized_default_affinity_group_is_serializable(shared_model_factory, tmp_path):
    model = shared_model_factory()
    filepath = tmp_path / "materialized.h5ad"

    save_anndata(filepath, model.materialize_results()[0])
    reloaded = load_anndata(filepath)

    assert list(reloaded.uns["popari_hyperparameters"]["groups"]) == ["_default"]
    assert reloaded.uns["popari_hyperparameters"]["regularization_groups"] == {}


def test_hierarchy_validation_requires_coarse_bin_assignments(hierarchical_model_factory):
    model = hierarchical_model_factory(hierarchical_levels=2)
    hierarchy = {level: model.hierarchy[level].adata.copy() for level in range(2)}
    del hierarchy[1].obsm["bin_assignments"]

    with pytest.raises(ValueError, match=r"missing obsm\['bin_assignments'\]"):
        validate_anndata_hierarchy(hierarchy)


@pytest.mark.baseline
def test_load_trained_model_roundtrip(shared_model_factory, tmp_path):
    model = shared_model_factory()
    filepath = tmp_path / "trained_model.h5ad"

    save_anndata(filepath, model.materialize_results()[0])
    reloaded = load_trained_model(filepath)

    assert reloaded.replicate_names == model.replicate_names
    assert reloaded.groups == model.groups
    assert reloaded.regularization_groups == model.regularization_groups

    assert np.allclose(reloaded.adata.obsm["X"], model.adata.obsm["X"])
    assert np.allclose(reloaded.adata.uns["M"], model.adata.uns["M"])

    assert np.isfinite(reloaded.nll(level=0)).all()


@pytest.mark.baseline
def test_load_differential_affinities_from_shared_file(shared_model_factory, tmp_path):
    model = shared_model_factory()
    filepath = tmp_path / "shared_model.h5ad"
    save_anndata(filepath, model.materialize_results()[0])

    differential = load_trained_model(
        filepath,
        groups="disjoint",
        regularization_groups={"_default": list(model.replicate_names)},
    )

    assert differential.groups == {sample: [sample] for sample in differential.replicate_names}
    assert differential.regularization_groups == {"_default": list(differential.replicate_names)}


def test_differential_group_affinity_roundtrip(adata_factory, differential_model_factory, tmp_path):
    samples = ["early_1", "early_2", "late_1", "late_2"]
    groups = {"early": samples[:2], "late": samples[2:]}
    model = differential_model_factory(
        adata=adata_factory(num_replicates=4, replicate_names=samples),
        groups=groups,
        regularization_groups={"all": list(groups)},
    )
    state = model.hierarchy[0].spatial_affinity
    with torch.no_grad():
        state.for_parameter("early").fill_(1)
        state.for_parameter("late").fill_(2)
    filepath = tmp_path / "differential_group.h5ad"
    save_anndata(filepath, model.materialize_results()[0])

    reloaded = load_trained_model(filepath)
    restored = reloaded.hierarchy[0].spatial_affinity

    assert reloaded.groups == groups
    assert reloaded.regularization_groups == {"all": list(groups)}
    assert restored.for_sample("early_1") is restored.for_sample("early_2")
    assert restored.for_sample("late_1") is restored.for_sample("late_2")
    torch.testing.assert_close(restored.for_parameter("early"), state.for_parameter("early"))
    torch.testing.assert_close(restored.for_parameter("late"), state.for_parameter("late"))


@pytest.mark.parametrize(
    ("legacy_mode", "groups", "regularization_groups", "legacy_groups"),
    [
        ("shared lookup", None, None, {"_default": ["0", "1"]}),
        ("differential lookup", "disjoint", {"cohort": ["0", "1"]}, {"cohort": ["0", "1"]}),
        (
            "differential group lookup",
            {"first": ["0"], "second": ["1"]},
            {"all": ["first", "second"]},
            {"first": ["0"], "second": ["1"]},
        ),
    ],
)
def test_load_trained_model_translates_legacy_affinity_metadata(
    shared_model_factory,
    tmp_path,
    legacy_mode,
    groups,
    regularization_groups,
    legacy_groups,
):
    model = shared_model_factory(groups=groups, regularization_groups=regularization_groups)
    adata = model.materialize_results()[0].copy()
    hyperparameters = adata.uns["popari_hyperparameters"]
    hyperparameters["spatial_affinity_mode"] = legacy_mode
    hyperparameters["spatial_affinity_groups"] = legacy_groups
    hyperparameters.pop("groups")
    hyperparameters.pop("regularization_groups")
    filepath = tmp_path / f"{legacy_mode.replace(' ', '_')}.h5ad"
    save_anndata(filepath, adata)

    reloaded = load_trained_model(filepath)

    assert reloaded.groups == model.groups
    assert list(reloaded.regularization_groups.values()) == list(model.regularization_groups.values())


@pytest.mark.gpu
@pytest.mark.expensive
def test_hierarchical_save_and_load_roundtrip(hierarchical_model_factory, gpu_context, tmp_path):
    model = hierarchical_model_factory(
        hierarchical_levels=2,
        torch_context=gpu_context,
        initial_context=gpu_context,
    )
    trainer = train_model(model)
    trainer.superresolve(n_epochs=2, tol=1e-6)

    filepath = tmp_path / "hierarchical_results"
    save_anndata_hierarchy(filepath, model.materialize_results())
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
    save_anndata_hierarchy(filepath, model.materialize_results(), ignore_raw_data=True)
    reloaded = load_trained_model(filepath)

    reloaded._reload_expression(raw_adata)

    assert reloaded.hierarchy[0].adata.X.sum() > 0


def test_reload_expression_rejects_sample_sequences(hierarchical_model_factory):
    model = hierarchical_model_factory(hierarchical_levels=2)

    with pytest.raises(TypeError, match="one unified AnnData"):
        model._reload_expression([model.adata.copy()])
