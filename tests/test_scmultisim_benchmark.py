from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.sparse import csr_array

from popari.schema import DATASET_NAME_KEY, SAMPLE_KEY_KEY, SCHEMA_VERSION, SCHEMA_VERSION_KEY
from scripts.scmultisim_benchmark import (
    ALPHA_MEANS,
    DIFF_CIF_FRACTIONS,
    NEGATIVE_POSTHOC_COLOCALIZATION_KEYS,
    POSTHOC_COLOCALIZATION_KEYS,
    add_posthoc_colocalizations,
    alpha_mean_variant,
    benchmark_experiment_names,
    benchmark_replicate_names,
    benchmark_variant,
    diff_cif_fraction_from_experiment,
    diffusion_matrix_slope,
    experiment_dir,
    generate_raw_scmultisim_outputs,
    ground_truth_category_accordance_matrices,
    load_benchmark_metrics,
    load_raw_scmultisim_outputs,
    method_specs,
    projected_category_accordance_matrices,
    raw_replicate_paths,
    scale_aligned_nrmse,
    scmultisim_diff_cif_experiment_name,
    spatial_affinity_groups,
    weighted_mean,
)


def _diffusion_config(tmp_path, *, cohort=0, replicates_per_time=1):
    return OmegaConf.create(
        {
            "benchmark_root": str(tmp_path),
            "experiment": {
                "name": "spatial_diffusion_no_cci_phyla5_multisample",
                "options": {"cif_sigma": 0.1},
                "spatial_diffusion": {
                    "enabled": True,
                    "times": [0.0, 1.0, 2.0, 4.0, 8.0],
                    "replicates_per_time": replicates_per_time,
                    "cohort": cohort,
                },
            },
        },
    )


def _diffusion_adata(times):
    samples = [f"sample_{index}" for index in range(len(times))]
    adata = ad.AnnData(X=np.ones((len(times), 2)))
    adata.obs["batch"] = pd.Categorical(samples, categories=samples, ordered=True)
    adata.obs["diffusion_time"] = np.asarray(times, dtype=float)
    adata.obsp["adjacency_matrix"] = csr_array((len(times), len(times)))
    adata.uns[DATASET_NAME_KEY] = "multisample"
    adata.uns[SAMPLE_KEY_KEY] = "batch"
    adata.uns[SCHEMA_VERSION_KEY] = SCHEMA_VERSION
    return adata


def _category_projection_adata():
    adata = ad.AnnData(X=np.ones((4, 2)))
    adata.obs["batch"] = pd.Categorical(["sample"] * 4)
    adata.obs["cell_type"] = pd.Categorical(["A", "A", "B", "B"])
    adata.obsm["X"] = np.array(
        [
            [3.0, 1.0],
            [1.0, 1.0],
            [1.0, 3.0],
            [1.0, 1.0],
        ],
    )
    adata.obsp["adjacency_matrix"] = csr_array(
        [
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ],
    )
    adata.uns["affinity"] = {"sample": np.diag([-2.0, -1.0])}
    adata.uns[DATASET_NAME_KEY] = "multisample"
    adata.uns[SAMPLE_KEY_KEY] = "batch"
    adata.uns[SCHEMA_VERSION_KEY] = SCHEMA_VERSION
    return adata


def test_scmultisim_diff_cif_experiment_names_roundtrip():
    experiment_names = benchmark_experiment_names()

    assert len(experiment_names) == len(DIFF_CIF_FRACTIONS) == 10
    for fraction, experiment_name in zip(DIFF_CIF_FRACTIONS, experiment_names, strict=True):
        assert scmultisim_diff_cif_experiment_name(fraction) == experiment_name
        assert diff_cif_fraction_from_experiment(experiment_name) == fraction


def test_scmultisim_benchmark_paths_use_experiment_name(tmp_path):
    cfg = OmegaConf.create(
        {
            "benchmark_root": str(tmp_path),
            "experiment": {"name": scmultisim_diff_cif_experiment_name(0.5)},
        },
    )

    case_dir = experiment_dir(cfg)
    paths = raw_replicate_paths(case_dir)

    assert case_dir == tmp_path / cfg.experiment.name
    assert paths["spatial_mix_45"] == case_dir / "raw" / "spatial_mix_45.h5ad"
    assert paths["spatial_mix_89"] == case_dir / "raw" / "spatial_mix_89.h5ad"


def test_observation_noise_uses_alpha_mean_variant_directory(tmp_path):
    cfg = OmegaConf.create(
        {
            "benchmark_root": str(tmp_path),
            "experiment": {"name": scmultisim_diff_cif_experiment_name(0.5)},
            "observation_noise": {"enabled": True, "alpha_mean": 0.05},
        },
    )

    assert [alpha_mean_variant(value) for value in ALPHA_MEANS] == [
        "alpha_mean_0_02",
        "alpha_mean_0_05",
        "alpha_mean_0_1",
        "alpha_mean_0_2",
        "alpha_mean_0_5",
    ]
    assert benchmark_variant(cfg) == "alpha_mean_0_05"
    assert experiment_dir(cfg) == tmp_path / cfg.experiment.name / "alpha_mean_0_05"


def test_diffusion_experiment_expands_timepoints_and_uses_cohort_directory(tmp_path):
    cfg = _diffusion_config(tmp_path, cohort=2)

    assert benchmark_replicate_names(cfg) == (
        "diffusion_0_rep1",
        "diffusion_1_rep1",
        "diffusion_2_rep1",
        "diffusion_3_rep1",
        "diffusion_4_rep1",
    )
    assert benchmark_variant(cfg) == "pilot/cohort_2"
    assert experiment_dir(cfg) == tmp_path / cfg.experiment.name / "pilot" / "cohort_2"


def test_diffusion_groups_all_samples_into_one_trajectory():
    adata = _diffusion_adata([0.0, 0.075, 0.075, 0.15])

    assert spatial_affinity_groups(adata) == {
        "diffusion_trajectory": ["sample_0", "sample_1", "sample_2", "sample_3"],
    }


def test_diffusion_matrix_slope_uses_timepoint_means():
    times = [0.0, 0.075, 0.075, 0.15]
    adata = _diffusion_adata(times)
    intercept = np.array([[2.0, -1.0], [-1.0, 3.0]])
    expected_slope = np.array([[-4.0, 2.0], [2.0, -1.0]])
    offsets = [0.0, 0.2, -0.2, 0.0]
    matrices = {
        sample: pd.DataFrame(intercept + time * expected_slope + offset)
        for sample, time, offset in zip(adata.popari.sample_names, times, offsets, strict=True)
    }

    slope = diffusion_matrix_slope(adata, matrices)

    np.testing.assert_allclose(slope, expected_slope)


def test_projected_category_accordance_uses_normalized_category_means():
    adata = _category_projection_adata()

    result = projected_category_accordance_matrices(adata, "affinity", categories=("A", "B"))["sample"]

    category_means = np.array([[0.625, 0.375], [0.375, 0.625]])
    expected = -category_means @ adata.uns["affinity"]["sample"] @ category_means.T
    expected = (expected + expected.T) / 2
    expected -= expected.mean()
    np.testing.assert_allclose(result, expected)


def test_ground_truth_category_accordance_is_centered_and_symmetric():
    adata = _category_projection_adata()

    result = ground_truth_category_accordance_matrices(adata, categories=("A", "B"))["sample"]

    np.testing.assert_allclose(result, result.T)
    np.testing.assert_allclose(result.to_numpy().mean(), 0, atol=1e-15)


def test_scale_aligned_nrmse_ignores_positive_scale_and_offset():
    reference = pd.DataFrame([[2.0, -1.0], [-1.0, 0.0]])
    query = 4 * reference + 7

    assert scale_aligned_nrmse(reference, query) < 1e-12
    assert scale_aligned_nrmse(reference, -query) == 1.0


def test_load_benchmark_metrics_concatenates_available_method_tables(tmp_path):
    experiment_name = scmultisim_diff_cif_experiment_name(1.0)
    popari_path = tmp_path / experiment_name / "popari" / "metrics.csv"
    stamp_path = tmp_path / experiment_name / "stamp" / "metrics.csv"
    popari_path.parent.mkdir(parents=True)
    stamp_path.parent.mkdir(parents=True)
    pd.DataFrame({"method": ["Popari"], "weighted_auroc": [0.8]}).to_csv(popari_path, index=False)
    pd.DataFrame({"method": ["STAMP"], "weighted_auroc": [0.7]}).to_csv(stamp_path, index=False)

    metrics = load_benchmark_metrics(Path(tmp_path), [experiment_name])

    assert metrics["method"].tolist() == ["Popari", "STAMP"]
    assert metrics["weighted_auroc"].tolist() == [0.8, 0.7]


def test_load_benchmark_metrics_reads_selected_variants(tmp_path):
    experiment_name = scmultisim_diff_cif_experiment_name(0.5)
    variants = (alpha_mean_variant(0.02), alpha_mean_variant(0.5))
    for alpha_mean, variant in zip((0.02, 0.5), variants, strict=True):
        path = tmp_path / experiment_name / variant / "popari" / "metrics.csv"
        path.parent.mkdir(parents=True)
        pd.DataFrame({"method": ["Popari"], "alpha_mean": [alpha_mean]}).to_csv(path, index=False)

    metrics = load_benchmark_metrics(tmp_path, [experiment_name], variants)

    assert metrics["alpha_mean"].tolist() == [0.02, 0.5]


def test_weighted_mean_ignores_nonfinite_and_zero_weight_entries():
    values = np.array([1.0, np.nan, 3.0, 10.0])
    weights = np.array([1.0, 1.0, 3.0, 0.0])

    assert weighted_mean(values, weights) == 2.5


def test_load_raw_outputs_requires_generation_stage(tmp_path):
    cfg = OmegaConf.create(
        {
            "benchmark_root": str(tmp_path),
            "experiment": {"name": scmultisim_diff_cif_experiment_name(0.5)},
        },
    )

    with np.testing.assert_raises_regex(FileNotFoundError, "generate_scmultisim_benchmark.py"):
        load_raw_scmultisim_outputs(cfg)


def test_generate_raw_outputs_skips_complete_results_unless_overwriting(tmp_path, monkeypatch):
    cfg = OmegaConf.create(
        {
            "benchmark_root": str(tmp_path),
            "experiment": {"name": scmultisim_diff_cif_experiment_name(0.5)},
            "overwrite": False,
        },
    )
    calls = []

    def simulate(_cfg):
        calls.append(_cfg)
        datasets = []
        for sample in ("spatial_mix_45", "spatial_mix_89"):
            dataset = ad.AnnData(X=np.ones((2, 2)))
            dataset.uns["sample"] = {"name": sample}
            datasets.append(dataset)
        return datasets

    monkeypatch.setattr("scmultisim.experiment.simulate_datasets_from_config", simulate)

    paths = generate_raw_scmultisim_outputs(cfg)
    assert all(path.exists() for path in paths)
    assert len(calls) == 1

    generate_raw_scmultisim_outputs(cfg)
    assert len(calls) == 1

    cfg.overwrite = True
    generate_raw_scmultisim_outputs(cfg)
    assert len(calls) == 2


def test_generate_raw_outputs_promotes_observed_counts_and_preserves_truth(tmp_path, monkeypatch):
    cfg = OmegaConf.create(
        {
            "benchmark_root": str(tmp_path),
            "experiment": {"name": scmultisim_diff_cif_experiment_name(0.5)},
            "observation_noise": {"enabled": True, "alpha_mean": 0.05},
        },
    )

    def simulate(_cfg):
        datasets = []
        for sample in ("spatial_mix_45", "spatial_mix_89"):
            dataset = ad.AnnData(X=np.full((2, 2), 10))
            dataset.layers["counts_obs"] = np.full((2, 2), 2)
            dataset.uns["sample"] = {"name": sample}
            dataset.uns["observation_noise"] = {"alpha_mean": 0.05}
            datasets.append(dataset)
        return datasets

    monkeypatch.setattr("scmultisim.experiment.simulate_datasets_from_config", simulate)

    paths = generate_raw_scmultisim_outputs(cfg)
    result = ad.read_h5ad(paths[0])

    np.testing.assert_array_equal(result.X, np.full((2, 2), 2))
    np.testing.assert_array_equal(result.layers["true_counts"], np.full((2, 2), 10))
    assert "counts_obs" not in result.layers


def test_posthoc_colocalizations_preserve_correlations_and_add_affinities():
    adata = ad.AnnData(X=np.ones((4, 2)))
    adata.obs["batch"] = pd.Categorical(["sample_0", "sample_0", "sample_1", "sample_1"])
    adata.obsm["X"] = np.array(
        [
            [0.9, 0.1],
            [0.7, 0.3],
            [0.2, 0.8],
            [0.4, 0.6],
        ],
    )
    adata.obsp["adjacency_matrix"] = csr_array(
        [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
    )
    adata.uns[DATASET_NAME_KEY] = "multisample"
    adata.uns[SAMPLE_KEY_KEY] = "batch"
    adata.uns[SCHEMA_VERSION_KEY] = SCHEMA_VERSION

    add_posthoc_colocalizations(adata, kind="stamp")

    assert "stamp_empirical_correlation" in adata.uns
    for method, correlation_key in POSTHOC_COLOCALIZATION_KEYS.items():
        affinity_key = NEGATIVE_POSTHOC_COLOCALIZATION_KEYS[method]
        for sample in adata.obs["batch"].cat.categories:
            np.testing.assert_allclose(
                adata.uns[affinity_key][sample],
                -adata.uns[correlation_key][sample],
            )


def test_method_specs_use_negative_posthoc_affinities():
    for kind in ("popari", "stamp"):
        affinity_keys = {key for _, key in method_specs(kind)}
        assert "pearson_correlation" not in affinity_keys
        assert "cosine_similarity" not in affinity_keys
        assert "hotspot_local_correlation" not in affinity_keys
        assert set(NEGATIVE_POSTHOC_COLOCALIZATION_KEYS.values()) <= affinity_keys
