from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts import train as train_script

CONFIG_DIRECTORY = Path(__file__).parents[1] / "configs"
SWEEP_PATH = Path(__file__).parents[1] / "sweeps" / "popari.yaml"


def compose_train_config(*overrides):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIRECTORY)):
        return compose(config_name="train", overrides=list(overrides))


class FakeTrainer:
    def __init__(self):
        self.trained = False

    def __enter__(self):
        return self

    def __exit__(self, error_type, value, traceback):
        return None

    def train(self):
        self.trained = True


def mock_training(monkeypatch):
    model = Mock()
    model.materialize_results.return_value = {0: Mock()}
    model_constructor = Mock(return_value=model)
    trainer = FakeTrainer()
    trainer.wandb_run = None
    save_results = Mock()
    log_results = Mock()

    def construct_trainer(constructed_model, **kwargs):
        trainer.model = constructed_model
        trainer.kwargs = kwargs
        trainer.wandb_run = Mock(id="run-id") if kwargs.get("use_wandb") else None
        return trainer

    trainer_constructor = Mock(side_effect=construct_trainer)
    monkeypatch.setattr(train_script, "Popari", model_constructor)
    monkeypatch.setattr(train_script, "Trainer", trainer_constructor)
    monkeypatch.setattr(train_script, "save_anndata_hierarchy", save_results)
    monkeypatch.setattr(train_script, "log_popari_results", log_results)
    return model_constructor, trainer_constructor, trainer, save_results, log_results


def test_hydra_config_composes_training_overrides():
    config = compose_train_config(
        "data.dataset_path=/path/to/input.h5ad",
        "model.K=10",
        "model.spatial_affinity_mode=differential lookup",
        "training.iterations=20",
        "tracking.enabled=true",
    )

    assert config.data.dataset_path == "/path/to/input.h5ad"
    assert config.model.K == 10
    assert config.model.spatial_affinity_mode == "differential lookup"
    assert config.training.iterations == 20
    assert config.tracking.enabled
    assert config.tracking.entity == "popari"
    assert config.tracking.project == "revisions"


def test_train_from_config_builds_local_training_run(tmp_path, monkeypatch):
    model_constructor, trainer_constructor, trainer, save_results, log_results = mock_training(monkeypatch)
    config = compose_train_config(
        "data.dataset_path=/path/to/input.h5ad",
        "model.K=10",
        "training.iterations=2",
        f"output_dir={tmp_path}",
    )
    original_config = OmegaConf.to_container(config, resolve=True)

    result_path = train_script.train_from_config(config)

    model_kwargs = model_constructor.call_args.kwargs
    (constructed_model,) = trainer_constructor.call_args.args
    trainer_kwargs = trainer_constructor.call_args.kwargs
    assert model_kwargs["K"] == 10
    assert model_kwargs["dataset_path"] == Path("/path/to/input.h5ad")
    assert OmegaConf.to_container(config, resolve=True) == original_config
    assert model_kwargs["initial_context"] == {"device": "cuda", "dtype": torch.float64}
    assert model_kwargs["torch_context"] == {"device": "cuda", "dtype": torch.float64}
    assert "spatial_affinity_groups" not in model_kwargs
    assert constructed_model is model_constructor.return_value
    assert trainer_kwargs["iterations"] == 2
    assert trainer.trained
    assert trainer_kwargs["use_wandb"] is False
    assert trainer_kwargs["wandb_kwargs"] is None
    model_constructor.return_value.materialize_results.assert_called_once_with()
    save_results.assert_called_once_with(
        result_path,
        {0: model_constructor.return_value.materialize_results.return_value[0]},
    )
    log_results.assert_not_called()
    assert result_path.parent == tmp_path
    assert not list(result_path.rglob("*.pt"))


def test_train_from_config_configures_wandb_tracking(tmp_path, monkeypatch):
    _, trainer_constructor, trainer, save_results, log_results = mock_training(monkeypatch)
    config = compose_train_config(
        "data.dataset_path=/path/to/input.h5ad",
        "model.K=10",
        f"output_dir={tmp_path}",
        "tracking.enabled=true",
    )

    result_path = train_script.train_from_config(config)

    trainer_kwargs = trainer_constructor.call_args.kwargs
    wandb_kwargs = trainer_kwargs["wandb_kwargs"]
    assert trainer_kwargs["use_wandb"] is True
    assert wandb_kwargs["entity"] == "popari"
    assert wandb_kwargs["project"] == "revisions"
    assert "enabled" not in wandb_kwargs
    assert wandb_kwargs["config"]["model"]["K"] == 10
    assert wandb_kwargs["config"]["tracking"]["enabled"] is True
    assert wandb_kwargs["config"]["config_uuid"] == result_path.name
    assert wandb_kwargs["config"]["result_directory"] == str(result_path)
    assert "result_path" not in wandb_kwargs["config"]
    save_results.assert_called_once()
    log_results.assert_called_once()
    assert result_path.parent == tmp_path


def test_result_config_hash_excludes_tracking_verbosity_and_output_directory():
    first = compose_train_config(
        "data.dataset_path=/path/to/input.h5ad",
        "model.K=10",
        "verbose=0",
        "output_dir=first",
        "tracking.enabled=false",
    )
    second = compose_train_config(
        "data.dataset_path=/path/to/input.h5ad",
        "model.K=10",
        "verbose=2",
        "output_dir=second",
        "tracking.enabled=true",
    )

    first_result = train_script.select_minimal_config(first, train_script.RESULT_CONFIG_KEYS)
    second_result = train_script.select_minimal_config(second, train_script.RESULT_CONFIG_KEYS)

    assert train_script.hash_config(first_result) == train_script.hash_config(second_result)


def test_result_config_hash_changes_with_training_configuration():
    first = compose_train_config("data.dataset_path=/path/to/input.h5ad", "model.K=10")
    second = compose_train_config("data.dataset_path=/path/to/input.h5ad", "model.K=11")

    first_result = train_script.select_minimal_config(first, train_script.RESULT_CONFIG_KEYS)
    second_result = train_script.select_minimal_config(second, train_script.RESULT_CONFIG_KEYS)

    assert train_script.hash_config(first_result) != train_script.hash_config(second_result)


def test_train_from_config_rejects_existing_result_directory(tmp_path, monkeypatch):
    mock_training(monkeypatch)
    config = compose_train_config(
        "data.dataset_path=/path/to/input.h5ad",
        "model.K=10",
        f"output_dir={tmp_path}",
    )
    train_script.train_from_config(config)

    with pytest.raises(FileExistsError, match="Result directory already exists"):
        train_script.train_from_config(config)


def test_train_from_config_rejects_relative_dataset_path(tmp_path, monkeypatch):
    model_constructor, _, _, _, _ = mock_training(monkeypatch)
    config = compose_train_config(
        "data.dataset_path=input.h5ad",
        "model.K=10",
        f"output_dir={tmp_path}",
    )

    with pytest.raises(ValueError, match="data.dataset_path must be absolute"):
        train_script.train_from_config(config)

    model_constructor.assert_not_called()


def test_wandb_sweep_uses_hydra_overrides():
    sweep = yaml.safe_load(SWEEP_PATH.read_text())
    config = compose_train_config("data.dataset_path=/path/to/input.h5ad", "model.K=10")

    assert sweep["entity"] == "popari"
    assert sweep["project"] == "revisions"
    assert sweep["metric"] == {"name": "nll_spatial", "goal": "minimize"}
    assert "model.K" in sweep["parameters"]
    assert "model.lambda_Sigma_x_inv" in sweep["parameters"]
    assert "${args_no_hyphens}" in sweep["command"]
    assert "tracking.enabled=true" in sweep["command"]
    for parameter_path in sweep["parameters"]:
        assert OmegaConf.select(config, parameter_path) is not None, f"Unknown sweep parameter: {parameter_path}"
