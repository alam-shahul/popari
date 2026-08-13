import numpy as np
import pandas as pd

from scripts.benchmark_spatial_convergence import Schedule, run_benchmark, stopping_iteration, summarize_stopping_rules


def test_stopping_iteration_handles_negative_nll():
    nll = np.array([-100.0, -110.0, -110.001, -110.002, -110.003])

    assert stopping_iteration(nll, tolerance=1e-3, patience=2, minimum_iterations=1) == 4


def test_stopping_summary_selects_fast_quality_preserving_schedule():
    records = []
    for seed in (0, 1):
        for schedule, seconds in (("fast", 1.0), ("reference", 2.0)):
            for iteration, nll in enumerate((-100.0, -110.0, -110.001), start=1):
                records.append(
                    {
                        "seed": seed,
                        "schedule": schedule,
                        "outer_iteration": iteration,
                        "elapsed_seconds": seconds * iteration,
                        "nll": nll,
                        "metagene_recovery": 0.9,
                        "embedding_recovery": 0.9,
                        "affinity_recovery": 0.9,
                    },
                )

    trajectories = pd.DataFrame(records).replace({"reference": Schedule(1000, 2e-3).name})
    summary, recommendation = summarize_stopping_rules(
        trajectories,
        reference_schedule=Schedule(1000, 2e-3),
    )

    assert not summary.empty
    assert recommendation["schedule"] == "fast"


def test_tiny_cpu_benchmark_writes_results(tmp_path):
    payload = run_benchmark(
        tmp_path,
        device="cpu",
        seeds=(0,),
        schedules=(Schedule(1, 2e-3),),
        grid_size=4,
        num_genes=12,
        nmf_iterations=0,
        outer_iterations=1,
    )

    assert payload["recommendation"]["schedule"] == "epochs=1,tol=0.002"
    for filename in (
        "trajectories.csv",
        "stopping_rules.csv",
        "summary.json",
        "nll_by_iteration.png",
        "nll_by_time.png",
    ):
        assert (tmp_path / filename).is_file()
