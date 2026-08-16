import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from popari.train import Trainer


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


class FakeLevel:
    spatial_affinity_lr = 0.01

    def __init__(self, model):
        self.model = model
        self.sigma_yxs = torch.ones(2)
        parameter = torch.nn.Parameter(torch.zeros(1))
        self.spatial_affinity = SimpleNamespace(
            parameter_names=("default",),
            values={"default": parameter},
            parameters=lambda: iter((parameter,)),
        )

    def _initialize_spatial_affinities(self):
        self.model.reinitializations += 1

    def _recompute_observation_noise(self):
        self.model.noise_updates += 1

    def mark_dirty(self):
        pass


class FakeModel:
    K = 3
    hierarchical_levels = 1
    adata = SimpleNamespace(n_obs=12)
    replicate_names = ("sample_0", "sample_1")
    context = {"device": "cpu", "dtype": "float64"}
    dataset_path = Path("/input.h5ad")

    def __init__(self):
        self.parameter_updates = []
        self.weight_updates = []
        self.reinitializations = 0
        self.noise_updates = 0
        self.nll_calls = []
        self.hierarchy = [FakeLevel(self)]

    def state_dict(self):
        return {}

    def materialize_results(self):
        raise AssertionError("Training must not materialize analysis results.")

    def nll(self, level=0, use_spatial=False):
        self.nll_calls.append(use_spatial)
        return 2.0 if use_spatial else 1.0


def fake_wandb(monkeypatch, *, active_run=None):
    created_runs = []

    def init(**kwargs):
        run = FakeRun()
        run.config.update(kwargs.get("config", {}))
        created_runs.append((run, kwargs))
        return run

    module = SimpleNamespace(run=active_run, init=init)
    monkeypatch.setitem(sys.modules, "wandb", module)
    return module, created_runs


def mock_trainer_updates(monkeypatch):
    def update_parameters(trainer, **kwargs):
        trainer.model.parameter_updates.append(kwargs)
        return {"spatial_affinity_loss": 1.0, "metagene_loss": 1.0, "sigma_yx_mean": 1.0}

    def update_embeddings(trainer, **kwargs):
        trainer.model.weight_updates.append(kwargs)
        return {"embedding_loss": 1.0}

    monkeypatch.setattr(Trainer, "_update_parameters", update_parameters)
    monkeypatch.setattr(Trainer, "_update_embeddings", update_embeddings)


@pytest.mark.gpu
def test_trainer_runs(shared_model_factory, gpu_context):
    model = shared_model_factory(torch_context=gpu_context, initial_context=gpu_context, verbose=1)
    trainer = Trainer(model, iterations=1, spatial_affinity_epochs=1, verbose=1)

    trainer.train()
    hierarchy = model.materialize_results()

    assert trainer.global_step == 1
    assert trainer.training_completed
    assert tuple(hierarchy) == (0,)
    assert hierarchy[0].obsm["X"].shape == (model.adata.n_obs, model.K)


def test_trainer_logs_wandb_metrics(monkeypatch):
    mock_trainer_updates(monkeypatch)
    _, created_runs = fake_wandb(monkeypatch)
    model = FakeModel()
    with Trainer(
        model,
        spatial_preiterations=1,
        iterations=1,
        use_wandb=True,
        wandb_kwargs={"project": "test-project", "mode": "disabled", "config": {"model": {"K": 3}}},
    ) as trainer:
        trainer.train()

    run, init_kwargs = created_runs[0]
    assert init_kwargs["project"] == "test-project"
    assert [step for step, _ in run.history] == sorted(step for step, _ in run.history)
    assert {key for _, values in run.history for key in values} == {
        "nll",
        "nll_spatial_preiteration",
        "nll_spatial",
        "embedding_loss",
        "metagene_loss",
        "spatial_affinity_loss",
        "sigma_yx_mean",
    }
    assert not run.artifacts
    assert run.finish_calls == [0]
    assert model.parameter_updates == [
        {
            "differentiate_spatial_affinities": False,
            "spatial_affinity_epochs": 1000,
        },
        {"spatial_affinity_epochs": 1000},
    ]
    assert model.weight_updates == [{}, {}]


def test_trainer_reuses_active_wandb_run(monkeypatch):
    active_run = FakeRun("active-run")
    _, created_runs = fake_wandb(monkeypatch, active_run=active_run)

    trainer = Trainer(
        FakeModel(),
        iterations=0,
        use_wandb=True,
        wandb_kwargs={"config": {"model": {"K": 3}}},
    )
    trainer.finish()

    assert not created_runs
    assert active_run.config == {"model": {"K": 3}}
    assert not active_run.finish_calls


def test_completed_trainer_does_not_train_or_log_again(monkeypatch):
    mock_trainer_updates(monkeypatch)
    _, created_runs = fake_wandb(monkeypatch)
    model = FakeModel()
    trainer = Trainer(
        model,
        iterations=1,
        use_wandb=True,
        wandb_kwargs={"project": "test-project", "mode": "disabled"},
    )

    trainer.train()
    run, _ = created_runs[0]
    history = list(run.history)

    trainer.train()

    assert len(model.parameter_updates) == 1
    assert len(model.weight_updates) == 1
    assert run.history == history


@pytest.mark.parametrize(("verbose", "progress_disabled"), [(0, True), (1, False)])
def test_trainer_macro_progress_starts_at_verbose_one(verbose, progress_disabled, monkeypatch):
    progress_calls = []

    class EmptyProgress:
        def __iter__(self):
            return iter(())

    def record_progress(*args, **kwargs):
        progress_calls.append(kwargs)
        return EmptyProgress()

    monkeypatch.setattr("popari.train.trange", record_progress)
    Trainer(FakeModel(), iterations=0, verbose=verbose).train()

    assert [call["desc"] for call in progress_calls] == ["NMF", "Spatial pretraining", "Spatial training"]
    assert [call["disable"] for call in progress_calls] == [progress_disabled] * 3


def test_trainer_does_not_generate_wandb_config(monkeypatch):
    _, created_runs = fake_wandb(monkeypatch)
    trainer = Trainer(FakeModel(), iterations=0, use_wandb=True, wandb_kwargs={"project": "test-project"})

    trainer.finish()

    run, init_kwargs = created_runs[0]
    assert "config" not in init_kwargs
    assert not run.config


def test_local_trainer_does_not_compute_tracking_metrics(monkeypatch):
    mock_trainer_updates(monkeypatch)
    model = FakeModel()
    Trainer(model, nmf_iterations=1, spatial_preiterations=1, iterations=1).train()

    assert not model.nll_calls


def test_verbose_trainer_reports_phase_diagnostics(monkeypatch):
    mock_trainer_updates(monkeypatch)
    progress_bars = []
    log_calls = []

    class Progress:
        def __init__(self, iterations):
            self.iterations = iterations
            self.postfixes = []

        def __iter__(self):
            return iter(range(self.iterations))

        def set_postfix(self, **kwargs):
            self.postfixes.append(kwargs)

    def progress(iterations, **kwargs):
        progress_bar = Progress(iterations)
        progress_bars.append(progress_bar)
        return progress_bar

    monkeypatch.setattr("popari.train.trange", progress)
    monkeypatch.setattr("popari.train.logger.info", lambda message, *args: log_calls.append((message, args)))
    model = FakeModel()

    Trainer(model, nmf_iterations=1, spatial_preiterations=1, iterations=1, verbose=1).train()

    assert model.nll_calls == [False, True, True]
    assert [bar.postfixes for bar in progress_bars] == [
        [{"nll": pytest.approx(1.0)}],
        [{"nll": pytest.approx(2.0)}],
        [{"nll": pytest.approx(2.0)}],
    ]
    iteration_logs = [arguments for message, arguments in log_calls if "(step {})" in message]
    assert [arguments[:4] for arguments in iteration_logs] == [
        ("NMF", 1, 1, 1),
        ("Spatial pretraining", 1, 1, 2),
        ("Spatial training", 1, 1, 3),
    ]
    assert "nll=1.000e+00" in iteration_logs[0][4]
    assert "spatial_affinity_loss=1.000e+00" in iteration_logs[1][4]
    assert "metagene_loss=1.000e+00" in iteration_logs[2][4]
    assert "embedding_loss=1.000e+00" in iteration_logs[2][4]
    assert "sigma_yx_mean=1.000e+00" in iteration_logs[2][4]


@pytest.mark.parametrize("nmf_iterations", [0, 1])
def test_trainer_initializes_spatial_optimizer_at_phase_boundary(monkeypatch, nmf_iterations):
    mock_trainer_updates(monkeypatch)
    model = FakeModel()
    trainer = Trainer(model, nmf_iterations=nmf_iterations, iterations=1)

    assert trainer.spatial_affinity_optimizer is None

    trainer.train()

    assert len(trainer.spatial_affinity_optimizer.param_groups[0]["params"]) == 1
    assert model.reinitializations == int(nmf_iterations > 0)


def test_trainer_reuses_spatial_optimizer_across_phases(monkeypatch):
    optimizer_ids = []

    def update_parameters(trainer, **kwargs):
        optimizer_ids.append(id(trainer.spatial_affinity_optimizer))
        return {"spatial_affinity_loss": 1.0, "metagene_loss": 1.0, "sigma_yx_mean": 1.0}

    monkeypatch.setattr(Trainer, "_update_parameters", update_parameters)
    monkeypatch.setattr(Trainer, "_update_embeddings", lambda trainer, **kwargs: {"embedding_loss": 1.0})
    trainer = Trainer(FakeModel(), spatial_preiterations=1, iterations=1)

    trainer.train()

    assert len(optimizer_ids) == 2
    assert len(set(optimizer_ids)) == 1


def test_parameter_update_requires_spatial_optimizer():
    trainer = Trainer(FakeModel(), iterations=0)

    with pytest.raises(RuntimeError, match="must be initialized"):
        trainer._update_parameters()


@pytest.mark.parametrize("use_manual_gradients", [False, True])
def test_superresolve_respects_epoch_count_and_preserves_adam_state(
    monkeypatch,
    hierarchical_model_factory,
    use_manual_gradients,
):
    model = hierarchical_model_factory(hierarchical_levels=2)
    trainer = Trainer(model, iterations=0)
    fine_level, coarse_level = model.hierarchy
    optimizer_ids = []
    original_step = torch.optim.Adam.step

    def record_step(optimizer, *args, **kwargs):
        optimizer_ids.append(id(optimizer))
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", record_step)
    trainer._superresolve_stage(
        fine_level,
        coarse_level,
        learning_rate=0.1,
        n_epochs=3,
        tol=-1,
        check_frequency=2,
        use_manual_gradients=use_manual_gradients,
    )

    assert len(optimizer_ids) == 3 * fine_level.num_replicates
    assert len(set(optimizer_ids)) == fine_level.num_replicates


def test_hierarchical_training_superresolves_once(monkeypatch):
    model = FakeModel()
    model.hierarchical_levels = 2
    model.hierarchy.append(FakeLevel(model))
    trainer = Trainer(model, iterations=0)
    calls = []

    def superresolve():
        calls.append(True)
        trainer.superresolution_completed = True

    monkeypatch.setattr(trainer, "superresolve", superresolve)

    trainer.train()
    trainer.train()

    assert calls == [True]
    assert trainer.training_completed


def test_flat_training_does_not_superresolve(monkeypatch):
    trainer = Trainer(FakeModel(), iterations=0)
    superresolve = Mock()
    monkeypatch.setattr(trainer, "superresolve", superresolve)

    trainer.train()

    superresolve.assert_not_called()


@pytest.mark.parametrize("learning_rates", [(), (0,), (-0.1,)])
def test_superresolve_rejects_invalid_learning_rates(learning_rates):
    trainer = Trainer(FakeModel(), iterations=0)

    with pytest.raises(ValueError, match="positive"):
        trainer.superresolve(learning_rates=learning_rates)


def test_superresolve_applies_both_stages_per_level(monkeypatch):
    events = []

    class Level:
        num_replicates = 1
        sample_names = ("sample",)
        spatial_affinity_lr = 0.01

        def __init__(self, level):
            self.level = level
            self.metagenes = torch.nn.Parameter(torch.ones((1, 1)))
            affinity = torch.nn.Parameter(torch.zeros((1, 1)))
            self.spatial_affinity = SimpleNamespace(
                parameter_names=("sample",),
                values={"sample": affinity},
                parameters=lambda: iter((affinity,)),
            )

        def embedding(self, sample):
            return torch.ones((1, 1))

        def _initialize_spatial_affinities(self):
            events.append((self.level, "initialize"))

        def mark_dirty(self):
            events.append((self.level, "dirty"))

    levels = [Level(0), Level(1), Level(2)]
    model = SimpleNamespace(hierarchy=levels, hierarchical_levels=3)
    trainer = Trainer(model, iterations=0)

    def stage(level, coarse_level, **kwargs):
        events.append((level.level, "stage", kwargs))

    def update_spatial_affinities(level, *args, **kwargs):
        events.append((level.level, "affinity"))
        return np.asarray([1.0])

    monkeypatch.setattr(trainer, "_superresolve_stage", stage)
    monkeypatch.setattr(trainer, "_update_spatial_affinities", update_spatial_affinities)

    trainer.superresolve()

    assert events == [
        (1, "dirty"),
        (1, "stage", {"learning_rate": 1e-1, "n_epochs": 10_000, "tol": 1e-6, "use_manual_gradients": False}),
        (1, "stage", {"learning_rate": 1e-2, "n_epochs": 10_000, "tol": 1e-6, "use_manual_gradients": False}),
        (1, "initialize"),
        (1, "affinity"),
        (1, "dirty"),
        (0, "dirty"),
        (0, "stage", {"learning_rate": 1e-1, "n_epochs": 10_000, "tol": 1e-6, "use_manual_gradients": False}),
        (0, "stage", {"learning_rate": 1e-2, "n_epochs": 10_000, "tol": 1e-6, "use_manual_gradients": False}),
        (0, "initialize"),
        (0, "affinity"),
        (0, "dirty"),
    ]
    assert trainer.superresolution_completed
    assert trainer.spatial_affinity_optimizer is None
