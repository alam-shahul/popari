from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from scripts.scmultisim_benchmark import (
    DIFF_CIF_FRACTIONS,
    benchmark_experiment_names,
    diff_cif_fraction_from_experiment,
    experiment_dir,
    load_benchmark_metrics,
    raw_replicate_paths,
    scmultisim_diff_cif_experiment_name,
    weighted_mean,
)


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


def test_weighted_mean_ignores_nonfinite_and_zero_weight_entries():
    values = np.array([1.0, np.nan, 3.0, 10.0])
    weights = np.array([1.0, 1.0, 3.0, 0.0])

    assert weighted_mean(values, weights) == 2.5
