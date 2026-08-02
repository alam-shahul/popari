import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from popari.train import Trainer, TrainParameters


class FakeConfig(dict):
    def update(self, values, allow_val_change=False):
        super().update(values)


class FakeRun:
    def __init__(self, run_id="run-id"):
        self.id = run_id
        self.config = FakeConfig()
        self.history = []
        self.artifacts = []
        self.finish_calls = []

    def log(self, values, step):
        self.history.append((step, values))

    def log_artifact(self, artifact, aliases):
        self.artifacts.append((artifact, aliases))

    def finish(self, exit_code=0):
        self.finish_calls.append(exit_code)


class FakeArtifact:
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.files = []
        self.directories = []

    def add_file(self, path):
        self.files.append(path)

    def add_dir(self, path):
        self.directories.append(path)


class FakeParameterOptimizer:
    def __init__(self):
        self.reinitializations = 0

    def reinitialize_spatial_affinities(self):
        self.reinitializations += 1


class FakeModel:
    K = 3
    lambda_Sigma_bar = 0.1
    lambda_Sigma_x_inv = 0.2
    hierarchical_levels = 1
    downsampling_method = "partition"
    binning_downsample_rate = 0.5
    spatial_affinity_mode = "differential lookup"
    random_state = 0
    superresolution_lr = 0.1
    adata = SimpleNamespace(n_obs=12)
    replicate_names = ("sample_0", "sample_1")
    context = {"device": "cpu", "dtype": "float64"}

    def __init__(self):
        self.parameter_optimizer = FakeParameterOptimizer()
        self.parameter_updates = []
        self.weight_updates = []
        self.synchronizations = 0
        self.nll_calls = []

    def estimate_parameters(self, **kwargs):
        self.parameter_updates.append(kwargs)

    def estimate_weights(self, **kwargs):
        self.weight_updates.append(kwargs)

    def synchronize_datasets(self):
        self.synchronizations += 1

    def nll(self, use_spatial=False):
        self.nll_calls.append(use_spatial)
        return 2.0 if use_spatial else 1.0

    def save_results(self, path, **kwargs):
        result_path = Path(path).with_suffix(".h5ad")
        result_path.touch()
        return result_path


class FakeHierarchicalModel(FakeModel):
    hierarchical_levels = 2

    def save_results(self, path, **kwargs):
        result_path = Path(path).with_suffix("")
        result_path.mkdir()
        for level in range(self.hierarchical_levels):
            (result_path / f"level_{level}.h5ad").touch()
        return result_path


def fake_wandb(monkeypatch, *, active_run=None):
    created_runs = []

    def init(**kwargs):
        run = FakeRun()
        run.config.update(kwargs.get("config", {}))
        created_runs.append((run, kwargs))
        return run

    module = SimpleNamespace(run=active_run, init=init, Artifact=FakeArtifact)
    monkeypatch.setitem(sys.modules, "wandb", module)
    return module, created_runs


@pytest.mark.gpu
def test_trainer_runs_and_saves(shared_model_factory, gpu_context, tmp_path):
    model = shared_model_factory(
        torch_context=gpu_context,
        initial_context=gpu_context,
        verbose=1,
    )
    savepath = tmp_path / "trainer_test.h5ad"
    trainer = Trainer(
        parameters=TrainParameters(
            nmf_iterations=0,
            iterations=1,
            savepath=savepath,
        ),
        model=model,
        verbose=1,
    )

    trainer.train()
    result_path = trainer.save_results()

    assert result_path == savepath
    assert savepath.exists()


def test_trainer_logs_wandb_metrics_and_final_artifact(
    tmp_path,
    monkeypatch,
):
    _, created_runs = fake_wandb(monkeypatch)
    model = FakeModel()
    savepath = tmp_path / "wandb_trainer_test.h5ad"
    parameters = TrainParameters(
        nmf_iterations=0,
        spatial_preiterations=1,
        iterations=1,
        savepath=savepath,
        synchronization_frequency=1,
    )

    with Trainer(
        parameters,
        model,
        use_wandb=True,
        wandb_kwargs={
            "project": "test-project",
            "mode": "disabled",
            "config": {
                "training": {
                    "spatial_preiterations": 1,
                    "iterations": 1,
                },
            },
        },
    ) as trainer:
        trainer.train()
        first_result = trainer.save_results()
        trainer.save_results()

    run, init_kwargs = created_runs[0]
    assert first_result == savepath
    assert init_kwargs["project"] == "test-project"
    assert run.config == {
        "training": {
            "spatial_preiterations": 1,
            "iterations": 1,
        },
    }
    assert [step for step, _ in run.history] == sorted(step for step, _ in run.history)
    assert {key for _, values in run.history for key in values} == {
        "nll",
        "nll_spatial_preiteration",
        "nll_spatial",
    }
    assert len(run.artifacts) == 1
    artifact, aliases = run.artifacts[0]
    assert artifact.name == "popari-model-run-id-output"
    assert artifact.type == "popari-model"
    assert artifact.files == [str(savepath)]
    assert aliases == ["latest"]
    assert run.finish_calls == [0]
    assert model.parameter_updates == [
        {"differentiate_spatial_affinities": False, "synchronize": True},
        {"synchronize": True},
    ]


def test_trainer_reuses_active_wandb_run(tmp_path, monkeypatch):
    active_run = FakeRun("active-run")
    _, created_runs = fake_wandb(monkeypatch, active_run=active_run)
    model = FakeModel()

    trainer = Trainer(
        TrainParameters(nmf_iterations=0, iterations=0, savepath=tmp_path / "unused.h5ad"),
        model,
        use_wandb=True,
        wandb_kwargs={"config": {"model": {"K": model.K}}},
    )
    trainer.finish()

    assert not created_runs
    assert active_run.config == {"model": {"K": model.K}}
    assert not active_run.finish_calls


@pytest.mark.parametrize(
    ("verbose", "progress_disabled"),
    [(0, True), (1, False)],
)
def test_trainer_macro_progress_starts_at_verbose_one(
    verbose,
    progress_disabled,
    tmp_path,
    monkeypatch,
):
    progress_calls = []

    class EmptyProgress:
        def __iter__(self):
            return iter(())

    def record_progress(*args, **kwargs):
        progress_calls.append(kwargs)
        return EmptyProgress()

    monkeypatch.setattr("popari.train.trange", record_progress)
    trainer = Trainer(
        TrainParameters(
            nmf_iterations=0,
            spatial_preiterations=0,
            iterations=0,
            savepath=tmp_path / "unused.h5ad",
        ),
        FakeModel(),
        verbose=verbose,
    )

    trainer.train()

    assert [call["desc"] for call in progress_calls] == [
        "NMF",
        "Spatial pretraining",
        "Spatial training",
    ]
    assert [call["disable"] for call in progress_calls] == [progress_disabled] * 3


def test_trainer_does_not_generate_wandb_config(tmp_path, monkeypatch):
    _, created_runs = fake_wandb(monkeypatch)
    trainer = Trainer(
        TrainParameters(nmf_iterations=0, iterations=0, savepath=tmp_path / "unused.h5ad"),
        FakeModel(),
        use_wandb=True,
        wandb_kwargs={"project": "test-project"},
    )

    trainer.finish()

    run, init_kwargs = created_runs[0]
    assert "config" not in init_kwargs
    assert not run.config


def test_trainer_logs_hierarchical_model_directory(tmp_path, monkeypatch):
    _, created_runs = fake_wandb(monkeypatch)
    model = FakeHierarchicalModel()
    trainer = Trainer(
        TrainParameters(nmf_iterations=0, iterations=0, savepath=tmp_path / "hierarchy"),
        model,
        use_wandb=True,
    )

    result_path = trainer.save_results()

    run, _ = created_runs[0]
    artifact, _ = run.artifacts[0]
    assert result_path == tmp_path / "hierarchy"
    assert artifact.directories == [str(result_path)]
    assert not artifact.files


def test_local_trainer_does_not_compute_tracking_metrics(tmp_path):
    model = FakeModel()
    trainer = Trainer(
        TrainParameters(
            nmf_iterations=1,
            spatial_preiterations=1,
            iterations=1,
            savepath=tmp_path / "unused.h5ad",
            synchronization_frequency=1,
        ),
        model,
    )

    trainer.train()

    assert not model.nll_calls
