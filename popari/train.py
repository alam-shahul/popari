from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger
from tqdm.auto import trange

from popari.model import Popari


@dataclass
class TrainParameters:
    """Iteration and persistence settings for Popari training."""

    nmf_iterations: int
    iterations: int
    savepath: Path
    spatial_preiterations: int = 0
    synchronization_frequency: int = field(default=10, kw_only=True)


class Trainer:
    """Train Popari locally with optional Weights & Biases tracking."""

    def __init__(
        self,
        parameters: TrainParameters,
        model: Popari,
        verbose: int = 0,
        *,
        use_wandb: bool = False,
        wandb_kwargs: dict[str, Any] | None = None,
    ):
        if parameters.synchronization_frequency < 1:
            raise ValueError("synchronization_frequency must be positive.")

        self.model = model
        self.parameters = parameters
        self.verbose = verbose
        self.nmf_iterations = 0
        self.spatial_preiterations = 0
        self.iterations = 0
        self.global_step = 0
        self.wandb_run = None
        self._owns_wandb_run = False
        self._wandb_artifact_logged = False

        if use_wandb:
            self._initialize_wandb(wandb_kwargs or {})

    def _initialize_wandb(self, wandb_kwargs: dict[str, Any]) -> None:
        try:
            import wandb
        except ImportError as error:
            raise ImportError(
                "W&B tracking requires the optional dependency; install Popari with `pip install popari[wandb]`.",
            ) from error

        if wandb.run is None:
            self.wandb_run = wandb.init(**wandb_kwargs)
            self._owns_wandb_run = True
        else:
            self.wandb_run = wandb.run
            config = wandb_kwargs.get("config")
            if config is not None:
                self.wandb_run.config.update(config, allow_val_change=True)

    def _log(self, values: dict[str, Any]) -> None:
        if self.wandb_run is not None:
            self.wandb_run.log(values, step=self.global_step)

    def _should_synchronize(self, iteration: int) -> bool:
        return not (iteration % self.parameters.synchronization_frequency)

    def train(self) -> None:
        if self.verbose >= 1:
            logger.info(
                "Training Popari: observations={}, samples={}, K={}, hierarchy_levels={}, device={}, dtype={}",
                self.model.adata.n_obs,
                len(self.model.replicate_names),
                self.model.K,
                self.model.hierarchical_levels,
                self.model.context["device"],
                self.model.context["dtype"],
            )
        if self.wandb_run is not None:
            self._log({"nll": self.model.nll()})

        nmf_progress_bar = trange(
            self.parameters.nmf_iterations,
            desc="NMF",
            leave=True,
            disable=self.verbose < 1,
            dynamic_ncols=True,
            mininterval=1,
        )
        for _ in nmf_progress_bar:
            synchronize = self._should_synchronize(self.nmf_iterations)
            self.model.estimate_parameters(update_spatial_affinities=False, synchronize=synchronize)
            self.model.estimate_weights(use_neighbors=False, synchronize=synchronize)
            self.nmf_iterations += 1
            self.global_step += 1
            if self.wandb_run is not None and (synchronize or self.nmf_iterations == self.parameters.nmf_iterations):
                self._log({"nll": self.model.nll()})

        has_spatial_training = self.parameters.spatial_preiterations > 0 or self.parameters.iterations > 0
        if self.parameters.nmf_iterations > 0 and has_spatial_training:
            self.model.parameter_optimizer.reinitialize_spatial_affinities()
            self.model.synchronize_datasets()

        spatial_preprogress_bar = trange(
            self.parameters.spatial_preiterations,
            desc="Spatial pretraining",
            leave=True,
            disable=self.verbose < 1,
            dynamic_ncols=True,
            mininterval=1,
        )
        for _ in spatial_preprogress_bar:
            synchronize = self._should_synchronize(self.spatial_preiterations)
            self.model.estimate_parameters(
                differentiate_spatial_affinities=False,
                synchronize=synchronize,
            )
            self.model.estimate_weights(synchronize=synchronize)
            self.spatial_preiterations += 1
            self.global_step += 1
            if self.wandb_run is not None and (
                synchronize or self.spatial_preiterations == self.parameters.spatial_preiterations
            ):
                self._log({"nll_spatial_preiteration": self.model.nll(use_spatial=True)})

        progress_bar = trange(
            self.parameters.iterations,
            desc="Spatial training",
            leave=True,
            disable=self.verbose < 1,
            dynamic_ncols=True,
            mininterval=1,
        )
        for _ in progress_bar:
            synchronize = self._should_synchronize(self.iterations)
            self.model.estimate_parameters(synchronize=synchronize)
            self.model.estimate_weights(synchronize=synchronize)
            self.iterations += 1
            self.global_step += 1
            if self.wandb_run is not None and (synchronize or self.iterations == self.parameters.iterations):
                self._log({"nll_spatial": self.model.nll(use_spatial=True)})

        if self.verbose >= 1:
            logger.info("Finished Popari training after {} outer iterations", self.global_step)

    def save_results(self, savepath: str | Path | None = None, **kwargs) -> Path:
        if savepath is None:
            savepath = self.parameters.savepath

        result_path = self.model.save_results(savepath, **kwargs)
        if self.wandb_run is not None and not self._wandb_artifact_logged:
            import wandb

            artifact = wandb.Artifact(
                f"popari-model-{self.wandb_run.id}-output",
                type="popari-model",
            )
            if result_path.is_dir():
                artifact.add_dir(str(result_path))
            else:
                artifact.add_file(str(result_path))
            self.wandb_run.log_artifact(artifact, aliases=["latest"])
            self._wandb_artifact_logged = True
        return result_path

    def superresolve(self, **kwargs) -> None:
        new_lr = kwargs.pop("new_lr", self.model.superresolution_lr)
        target_level = kwargs.pop("target_level", None)
        self.model.set_superresolution_lr(new_lr, target_level)
        self.model.superresolve(**kwargs)

    def finish(self, exit_code: int = 0) -> None:
        if self._owns_wandb_run and self.wandb_run is not None:
            self.wandb_run.finish(exit_code=exit_code)
            self._owns_wandb_run = False

    def __enter__(self) -> "Trainer":
        return self

    def __exit__(self, error_type, value, traceback) -> None:
        self.finish(exit_code=1 if error_type is not None else 0)
