from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from loguru import logger
from scipy.sparse import csr_array
from tqdm.auto import trange

from popari._sparse import convert_numpy_to_pytorch_sparse_coo
from popari.model import Popari
from popari.optim.embedding import estimate_weight_wnbr, estimate_weight_wonbr
from popari.optim.metagene import estimate_metagenes
from popari.optim.spatial_affinity import estimate_spatial_affinity
from popari.schema import BIN_ASSIGNMENTS_KEY


class Trainer:
    """Train Popari locally with optional Weights & Biases tracking."""

    def __init__(
        self,
        model: Popari,
        *,
        nmf_iterations: int = 0,
        spatial_preiterations: int = 0,
        iterations: int = 1,
        spatial_affinity_epochs: int = 1000,
        verbose: int = 0,
        use_wandb: bool = False,
        wandb_kwargs: dict[str, Any] | None = None,
    ):
        self.model = model
        self.nmf_iterations = nmf_iterations
        self.spatial_preiterations = spatial_preiterations
        self.iterations = iterations
        self.spatial_affinity_epochs = spatial_affinity_epochs
        self.verbose = verbose
        self.global_step = 0
        self.training_completed = False
        self.superresolution_completed = False
        self.wandb_run = None
        self._owns_wandb_run = False
        self.spatial_affinity_optimizers = None

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

    def _report_iteration(
        self,
        progress_bar,
        *,
        phase: str,
        iteration: int,
        total: int,
        metrics: dict[str, float],
        use_spatial: bool,
    ) -> None:
        """Report one completed outer iteration."""

        if self.verbose < 1:
            return
        nll = np.asarray(
            self.model.nll(
                level=self.model.hierarchical_levels - 1,
                use_spatial=use_spatial,
            ),
        ).item()
        progress_bar.set_postfix(nll=nll)
        diagnostics = {"nll": nll, **metrics}
        formatted = " | ".join(f"{name}={value:.3e}" for name, value in diagnostics.items())
        logger.info(
            "{} {}/{} (step {}) | {}",
            phase,
            iteration + 1,
            total,
            self.global_step,
            formatted,
        )

    def _create_spatial_affinity_optimizers(self, level) -> dict[str, torch.optim.Adam]:
        state = level.spatial_affinity
        return {
            name: torch.optim.Adam(
                [state.values[name]],
                lr=level.spatial_affinity_lr,
                betas=(0.5, 0.9),
            )
            for name in state.parameter_names
        }

    @property
    def training_level(self):
        """Return the coarsest level optimized by alternating training."""

        return self.model.hierarchy[-1]

    def _update_embeddings(self, *, use_neighbors: bool = True) -> dict[str, float]:
        level = self.training_level
        if self.verbose >= 2:
            logger.info("Updating embeddings")

        losses = []
        global_embedding = level.embeddings
        global_normalized = global_embedding / torch.linalg.norm(global_embedding, dim=1, ord=1, keepdim=True)
        for sample_index, sample in enumerate(level.sample_names):
            arguments = (
                level,
                level.Ys[sample_index].to(level.context["device"]),
                level.metagenes.to(level.context["device"]),
                level.embedding(sample).to(level.context["device"]),
                level.sigma_yxs[sample_index],
                level.prior_x_modes[sample_index],
                level.prior_xs[sample_index],
            )
            if use_neighbors:
                loss, embedding = estimate_weight_wnbr(*arguments, sample, global_normalized)
            else:
                loss, embedding = estimate_weight_wonbr(*arguments)

            level.set_embedding(sample, embedding)
            normalized = embedding / torch.linalg.norm(embedding, dim=1, ord=1, keepdim=True)
            global_normalized.index_copy_(0, level.sample_indices[sample], normalized)
            losses.append(loss)

        level.mark_dirty()
        return {"embedding_loss": float(np.mean(losses))}

    def _update_spatial_affinities(
        self,
        level,
        optimizers,
        *,
        differentiate: bool,
        **optimization_kwargs,
    ) -> np.ndarray:
        losses = []
        if level.spatial_affinity_mode == "shared lookup":
            for group_name, samples in level.spatial_affinity_groups.items():
                sample_mask = [sample in samples for sample in level.sample_names]
                representative = samples[0]
                affinity, loss = estimate_spatial_affinity(
                    level,
                    level.spatial_affinity.for_sample(representative).to(level.context["device"]),
                    sample_mask,
                    optimizers[group_name],
                    tol=level.spatial_affinity_tol,
                    **optimization_kwargs,
                )
                with torch.no_grad():
                    level.spatial_affinity.set_sample_(representative, affinity)
                losses.append(float(loss))
        elif level.spatial_affinity_mode == "differential lookup":
            group_mean_snapshot = level.spatial_affinity.group_means() if differentiate else None
            for sample_index, sample in enumerate(level.sample_names):
                group_means = None
                if group_mean_snapshot is not None:
                    group_means = [group_mean_snapshot[group] for group in level.spatial_affinity_tags[sample]]
                sample_mask = [index == sample_index for index in range(len(level.sample_names))]
                affinity, loss = estimate_spatial_affinity(
                    level,
                    level.spatial_affinity.for_sample(sample).to(level.context["device"]),
                    sample_mask,
                    optimizers[sample],
                    Sigma_x_inv_bar=group_means,
                    tol=level.spatial_affinity_tol,
                    **optimization_kwargs,
                )
                with torch.no_grad():
                    level.spatial_affinity.set_sample_(sample, affinity)
                losses.append(float(loss))
        else:
            raise NotImplementedError(f"Unsupported spatial-affinity mode: {level.spatial_affinity_mode!r}")

        return np.asarray(losses, dtype=float)

    def _update_parameters(
        self,
        *,
        update_spatial_affinities: bool = True,
        differentiate_spatial_affinities: bool = True,
        simplex_projection_mode: str = "exact",
        edge_subsample_rate: float | None = None,
        spatial_affinity_epochs: int = 1000,
    ) -> dict[str, float]:
        level = self.training_level
        metrics = {}
        if update_spatial_affinities:
            if self.spatial_affinity_optimizers is None:
                raise RuntimeError("Spatial-affinity optimizers must be initialized before parameter updates.")
            if self.verbose >= 2:
                logger.info("Updating spatial affinities")
            losses = self._update_spatial_affinities(
                level,
                self.spatial_affinity_optimizers,
                differentiate=differentiate_spatial_affinities,
                subsample_rate=edge_subsample_rate,
                n_epochs=spatial_affinity_epochs,
            )
            metrics["spatial_affinity_loss"] = float(np.mean(losses))
        if self.verbose >= 2:
            logger.info("Updating metagenes")
        metagenes, metagene_loss = estimate_metagenes(
            level,
            level.metagenes,
            [True] * len(level.sample_names),
            simplex_projection_mode=simplex_projection_mode,
        )
        with torch.no_grad():
            level.metagenes.copy_(metagenes)
        metrics["metagene_loss"] = float(metagene_loss)
        if self.verbose >= 2:
            logger.info("Updating observation noise")
        level._recompute_observation_noise()
        metrics["sigma_yx_mean"] = float(level.sigma_yxs.mean())
        level.mark_dirty()
        return metrics

    def train(self) -> None:
        if self.training_completed:
            return

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
            self.nmf_iterations,
            desc="NMF",
            leave=True,
            disable=self.verbose < 1,
            dynamic_ncols=True,
            mininterval=1,
        )
        for iteration in nmf_progress_bar:
            self.superresolution_completed = False
            metrics = self._update_parameters(update_spatial_affinities=False)
            metrics.update(self._update_embeddings(use_neighbors=False))
            self.global_step += 1
            self._log(metrics)
            self._report_iteration(
                nmf_progress_bar,
                phase="NMF",
                iteration=iteration,
                total=self.nmf_iterations,
                metrics=metrics,
                use_spatial=False,
            )
        if self.wandb_run is not None and self.nmf_iterations:
            self._log({"nll": self.model.nll(level=self.model.hierarchical_levels - 1)})

        has_spatial_training = self.spatial_preiterations > 0 or self.iterations > 0
        if has_spatial_training:
            if self.nmf_iterations > 0:
                self.training_level._initialize_spatial_affinities()
                self.training_level.mark_dirty()
            self.spatial_affinity_optimizers = self._create_spatial_affinity_optimizers(self.training_level)

        spatial_preprogress_bar = trange(
            self.spatial_preiterations,
            desc="Spatial pretraining",
            leave=True,
            disable=self.verbose < 1,
            dynamic_ncols=True,
            mininterval=1,
        )
        for iteration in spatial_preprogress_bar:
            self.superresolution_completed = False
            metrics = self._update_parameters(
                differentiate_spatial_affinities=False,
                spatial_affinity_epochs=self.spatial_affinity_epochs,
            )
            metrics.update(self._update_embeddings())
            self.global_step += 1
            self._log(metrics)
            self._report_iteration(
                spatial_preprogress_bar,
                phase="Spatial pretraining",
                iteration=iteration,
                total=self.spatial_preiterations,
                metrics=metrics,
                use_spatial=True,
            )
        if self.wandb_run is not None and self.spatial_preiterations:
            self._log(
                {
                    "nll_spatial_preiteration": self.model.nll(
                        level=self.model.hierarchical_levels - 1,
                        use_spatial=True,
                    ),
                },
            )

        progress_bar = trange(
            self.iterations,
            desc="Spatial training",
            leave=True,
            disable=self.verbose < 1,
            dynamic_ncols=True,
            mininterval=1,
        )
        for iteration in progress_bar:
            self.superresolution_completed = False
            metrics = self._update_parameters(spatial_affinity_epochs=self.spatial_affinity_epochs)
            metrics.update(self._update_embeddings())
            self.global_step += 1
            self._log(metrics)
            self._report_iteration(
                progress_bar,
                phase="Spatial training",
                iteration=iteration,
                total=self.iterations,
                metrics=metrics,
                use_spatial=True,
            )
        if self.wandb_run is not None and self.iterations:
            self._log(
                {
                    "nll_spatial": self.model.nll(
                        level=self.model.hierarchical_levels - 1,
                        use_spatial=True,
                    ),
                },
            )

        if self.model.hierarchical_levels > 1 and not self.superresolution_completed:
            self.superresolve()

        self.training_completed = True
        if self.verbose >= 1:
            logger.info("Finished Popari training after {} outer iterations", self.global_step)

    def _superresolve_stage(
        self,
        level,
        coarse_level,
        *,
        learning_rate: float,
        n_epochs: int,
        tol: float,
        use_manual_gradients: bool,
        check_frequency: int = 100,
    ) -> None:
        """Optimize one finer level for one superresolution stage."""

        assignments = csr_array(coarse_level.adata.obsm[BIN_ASSIGNMENTS_KEY])
        sample_states = {}
        for dataset_index, sample in enumerate(level.sample_names):
            sample_indices = level.sample_axis.indices(sample)
            coarse_indices = coarse_level.sample_axis.indices(sample)
            embedding = level.embedding(sample).clone().detach().requires_grad_(True)
            coarse_embedding = coarse_level.embedding(sample).detach().to(**level.context)
            expression = level.Ys[dataset_index].to(level.context["device"])
            if expression.sum() == 0:
                raise ValueError(
                    "It seems like you are trying to superresolve a hierarchical level with all zero expression "
                    "values. Reload a finalized AnnData result containing expression before superresolution.",
                )
            bin_assignments = assignments[coarse_indices][:, sample_indices]
            transposed_assignments = convert_numpy_to_pytorch_sparse_coo(
                bin_assignments.T.tocoo(),
                context=level.context,
            )
            sample_states[sample] = {
                "embedding": embedding,
                "expression": expression,
                "BTB": convert_numpy_to_pytorch_sparse_coo(
                    (bin_assignments.T @ bin_assignments).tocoo(),
                    context=level.context,
                ),
                "BTX_B": transposed_assignments @ coarse_embedding,
                "X_Bnorm": torch.linalg.norm(coarse_embedding, ord="fro").square().item(),
                "optimizer": torch.optim.Adam([embedding], lr=learning_rate, betas=(0.5, 0.9)),
            }

        progress = trange(
            0,
            n_epochs,
            check_frequency,
            desc=f"Superresolution level {level.level}",
            leave=True,
            disable=self.verbose < 1,
            dynamic_ncols=True,
            mininterval=1,
        )
        previous_losses = np.full(level.num_replicates, np.inf)
        for epoch_start in progress:
            level._recompute_observation_noise()
            block_epochs = min(check_frequency, n_epochs - epoch_start)
            losses = np.zeros(level.num_replicates)
            converged = np.zeros(level.num_replicates, dtype=bool)

            for dataset_index, sample in enumerate(level.sample_names):
                state = sample_states[sample]
                expression = state["expression"]
                embedding = state["embedding"]
                optimizer = state["optimizer"]
                sigma_yx = level.sigma_yxs[dataset_index]
                metagenes = level.metagenes
                MTM = (metagenes.T @ metagenes / sigma_yx.square()).detach()
                YM = (expression @ metagenes / sigma_yx.square()).detach()
                linear_term_gradient = YM + state["BTX_B"]
                if level.prior_x_modes[dataset_index] == "exponential shared fixed":
                    linear_term_gradient = linear_term_gradient - level.prior_xs[dataset_index][0][None]
                linear_term_gradient = linear_term_gradient.detach()
                Ynorm = (torch.square(expression).sum() / sigma_yx.square()).detach()

                loss = np.nan
                for _ in range(block_epochs):
                    previous_embedding = embedding.detach().clone()
                    optimizer.zero_grad()
                    quadratic_term_gradient = embedding @ MTM + state["BTB"] @ embedding
                    objective = (
                        (quadratic_term_gradient * embedding).sum() / 2
                        - (linear_term_gradient * embedding).sum()
                        + Ynorm / 2
                        + state["X_Bnorm"] / 2
                    )
                    if use_manual_gradients:
                        embedding.grad = quadratic_term_gradient - linear_term_gradient
                    else:
                        objective.backward()
                    optimizer.step()
                    with torch.no_grad():
                        embedding.clamp_(min=1e-10)

                    loss = objective.detach().item()
                    relative_change = (
                        torch.abs(
                            (previous_embedding - embedding) / torch.linalg.norm(embedding, dim=1, ord=1, keepdim=True),
                        )
                        .max()
                        .item()
                    )
                    if relative_change < tol:
                        converged[dataset_index] = True
                        break

                level.set_embedding(sample, embedding.detach())
                losses[dataset_index] = loss

            deltas = (previous_losses - losses) / losses
            progress.set_postfix(loss=np.mean(losses), delta=np.mean(deltas))
            previous_losses = losses
            if converged.all():
                break

        level.mark_dirty()

    def superresolve(
        self,
        *,
        n_epochs: int = 10_000,
        tol: float = 1e-6,
        learning_rates: Sequence[float] = (1e-1, 1e-2),
        use_manual_gradients: bool = False,
    ) -> None:
        """Apply the manuscript's two-stage hierarchical superresolution
        schedule."""

        if not learning_rates or any(learning_rate <= 0 for learning_rate in learning_rates):
            raise ValueError("learning_rates must contain at least one positive value.")

        for level_index in range(self.model.hierarchical_levels - 2, -1, -1):
            level = self.model.hierarchy[level_index]
            coarse_level = self.model.hierarchy[level_index + 1]
            if self.verbose >= 1:
                logger.info("Superresolving hierarchy level {}", level_index)
            with torch.no_grad():
                level.metagenes.copy_(coarse_level.metagenes)
            level.mark_dirty()

            for learning_rate in learning_rates:
                self._superresolve_stage(
                    level,
                    coarse_level,
                    learning_rate=learning_rate,
                    n_epochs=n_epochs,
                    tol=tol,
                    use_manual_gradients=use_manual_gradients,
                )

            level._initialize_spatial_affinities()
            spatial_affinity_optimizers = self._create_spatial_affinity_optimizers(level)
            self._update_spatial_affinities(
                level,
                spatial_affinity_optimizers,
                differentiate=True,
                subsample_rate=None,
            )
            level.mark_dirty()
        self.superresolution_completed = True

    def finish(self, exit_code: int = 0) -> None:
        if self._owns_wandb_run and self.wandb_run is not None:
            self.wandb_run.finish(exit_code=exit_code)
            self._owns_wandb_run = False

    def __enter__(self) -> "Trainer":
        return self

    def __exit__(self, error_type, value, traceback) -> None:
        self.finish(exit_code=1 if error_type is not None else 0)
