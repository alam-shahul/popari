import hashlib
import sys
import uuid
from pathlib import Path

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from popari.io import save_anndata_hierarchy
from popari.model import Popari
from popari.schema import SCHEMA_VERSION
from popari.train import Trainer
from popari.wandb_util import log_popari_results

RESULT_CONFIG_KEYS = (
    "data",
    "model",
    "training",
    "dtype",
    "initial_device",
    "torch_device",
)


def configure_logging(verbose: int) -> None:
    """Configure canonical training logs without disrupting tqdm bars."""

    level = "WARNING" if verbose == 0 else "DEBUG" if verbose >= 3 else "INFO"
    logger.remove()
    logger.add(
        lambda message: tqdm.write(message, end="", file=sys.stderr),
        level=level,
        colorize=sys.stderr.isatty(),
    )


def select_minimal_config(config: DictConfig, keys: tuple[str, ...]) -> DictConfig:
    """Select the configuration values that determine a trained result."""

    selected = OmegaConf.create({})
    for key in keys:
        OmegaConf.update(selected, key, OmegaConf.select(config, key), merge=True)
    return selected


def hash_config(config: DictConfig) -> str:
    """Return a deterministic UUID for a resolved configuration."""

    serialized = OmegaConf.to_yaml(config, resolve=True, sort_keys=True)
    digest = hashlib.md5(serialized.encode(), usedforsecurity=False).hexdigest()
    return str(uuid.UUID(digest))


def train_from_config(config: DictConfig) -> Path:
    """Train Popari from a resolved Hydra configuration."""

    # Resolve model arguments.
    model_kwargs = OmegaConf.to_container(config.model, resolve=True)
    model_kwargs = {key: value for key, value in model_kwargs.items() if value is not None}
    dataset_path = Path(config.data.dataset_path)
    if not dataset_path.is_absolute():
        raise ValueError("data.dataset_path must be absolute.")
    model_kwargs["dataset_path"] = dataset_path

    dtype = getattr(torch, config.dtype, None)
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise ValueError("dtype must name a floating-point PyTorch dtype.")

    model_kwargs["initial_context"] = {
        "device": config.initial_device,
        "dtype": dtype,
    }
    model_kwargs["torch_context"] = {
        "device": config.torch_device,
        "dtype": dtype,
    }
    model_kwargs["verbose"] = config.verbose

    # Derive and reserve the result directory from the effective configuration.
    result_config = select_minimal_config(config, RESULT_CONFIG_KEYS)
    config_uuid = hash_config(result_config)
    result_directory = Path(config.output_dir).resolve() / config_uuid
    try:
        result_directory.mkdir(parents=True)
    except FileExistsError as error:
        raise FileExistsError(f"Result directory already exists: {result_directory}") from error

    resolved_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    resolved_config["config_uuid"] = config_uuid
    resolved_config["result_directory"] = str(result_directory)
    OmegaConf.save(resolved_config, result_directory / "config.yaml")

    # Resolve W&B arguments.
    wandb_kwargs = None
    if config.tracking.enabled:
        wandb_kwargs = OmegaConf.to_container(config.tracking, resolve=True)
        wandb_kwargs.pop("enabled")
        wandb_kwargs.pop("upload_results")
        wandb_kwargs = {key: value for key, value in wandb_kwargs.items() if value is not None}
        wandb_kwargs["config"] = OmegaConf.to_container(resolved_config, resolve=True)

    model = Popari(**model_kwargs)
    with Trainer(
        model,
        nmf_iterations=config.training.nmf_iterations,
        spatial_preiterations=config.training.spatial_preiterations,
        iterations=config.training.iterations,
        verbose=config.verbose,
        use_wandb=config.tracking.enabled,
        wandb_kwargs=wandb_kwargs,
    ) as trainer:
        trainer.train()
        hierarchy = model.materialize_results()
        save_anndata_hierarchy(result_directory, hierarchy)
        if trainer.wandb_run is not None:
            trainer.wandb_run.summary["config_uuid"] = config_uuid
            trainer.wandb_run.summary["result_directory"] = str(result_directory)
            if config.tracking.upload_results:
                log_popari_results(
                    trainer.wandb_run,
                    result_directory,
                    metadata={
                        "schema_version": SCHEMA_VERSION,
                        "hierarchical_levels": len(hierarchy),
                        "config_uuid": config_uuid,
                    },
                )
        return result_directory


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(config: DictConfig) -> None:
    """Hydra entry point for one Popari training run."""

    configure_logging(config.verbose)
    train_from_config(config)


if __name__ == "__main__":
    main()
