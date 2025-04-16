import logging
import time
from typing import Sequence

import numpy as np
import torch
from tqdm.auto import tqdm, trange

from popari._batch_optimizer_util import BatchEffectLoss
from popari._popari_dataset import PopariDataset
from popari.util import (
    IndependentSet,
    NesterovGD,
    get_datetime,
    project2simplex,
    project2simplex_,
    project_M,
    project_M_,
    sample_graph_iid,
)


class BatchEffectOptimizer:
    """Optimizer and state for Popari batch effect."""

    def __init__(
        self,
        K,
        Ys,
        datasets,
        initial_context=None,
        context=None,
        use_inplace_ops=False,
        batch_step_size_multiplier=1.0,
        batch_mini_iterations=1000,
        batch_tol=1e-5,
        verbose=0,
        batch_effect_correction=False,
    ):
        self.verbose = verbose
        self.use_inplace_ops = use_inplace_ops
        self.datasets = datasets
        self.hierarchical = False
        self.K = K
        self.Ys = Ys
        self.initial_context = initial_context if initial_context else {"device": "cpu", "dtype": torch.float32}
        self.context = context if context else {"device": "cpu", "dtype": torch.float32}
        self.adjacency_lists = {dataset.name: dataset.obsm["adjacency_list"] for dataset in self.datasets}
        self.batch_step_size_multiplier = batch_step_size_multiplier
        self.batch_mini_iterations = batch_mini_iterations
        self.batch_tol = batch_tol
        self._adjacency_matrices = {}

        if self.verbose:
            print(f"{get_datetime()} Initializing BatchEffectState")
        self.batch_effect_state = BatchEffectState(K, self.datasets, context=self.context)

    @property
    def adjacency_matrices(self):
        return self._adjacency_matrices

    @adjacency_matrices.setter
    def adjacency_matrices(self, val):
        self._adjacency_matrices = val

    def link(self, embedding_optimizer, parameter_optimizer):
        self.parameter_optimizer = parameter_optimizer
        self.embedding_optimizer = embedding_optimizer

    def update_batch_effects(self):
        """Update batch effects for all datasets."""
        logging.info(f"{get_datetime()} Updating batch effects")

        loss_list = []
        for dataset_index, dataset in enumerate(self.datasets):
            sigma_yx = self.parameter_optimizer.sigma_yxs[dataset_index]
            Y = self.Ys[dataset_index].to(self.context["device"])
            X = self.embedding_optimizer.embedding_state[dataset.name].to(self.context["device"])
            M = self.parameter_optimizer.metagene_state[dataset.name].to(self.context["device"])
            B = self.batch_effect_state[dataset.name].to(self.context["device"])

            loss, updated_B = self.estimate_batch_effect(Y, M, X, B, sigma_yx, dataset)
            self.batch_effect_state[dataset.name][:] = updated_B
            loss_list.append(loss)

        return loss_list

    def nll_batch_effects(self):
        """Compute negative log likelihood for batch effects."""
        with torch.no_grad():
            loss_batch_effects = torch.zeros(1, **self.context)
            for dataset_index, dataset in enumerate(self.datasets):
                sigma_yx = self.parameter_optimizer.sigma_yxs[dataset_index]
                Y = self.Ys[dataset_index].to(self.context["device"])
                X = self.embedding_optimizer.embedding_state[dataset.name].to(self.context["device"])
                M = self.parameter_optimizer.metagene_state[dataset.name].to(self.context["device"])
                B = self.batch_effect_state[dataset.name].to(self.context["device"])

                loss = self.nll_batch_effect(Y, M, X, B, sigma_yx)
                loss_batch_effects += loss

        return loss_batch_effects.cpu().numpy()

    @torch.no_grad()
    def nll_batch_effect(self, Y, M, X, B, sigma_yx):
        """Compute negative log likelihood for a single dataset's batch
        effect."""

        batch_effect_loss = BatchEffectLoss()
        loss = batch_effect_loss.compute_loss(B, M, X, Y, sigma_yx)
        return loss

    @torch.no_grad()
    def estimate_batch_effect(self, Y, M, X, B, sigma_yx, dataset, update_alg="nesterov"):
        """Estimate batch effect for a single dataset using Nesterov's
        Accelerated Gradient."""
        print("B", B)
        G, K = M.size()

        B = B.clone()

        batch_effect_loss = BatchEffectLoss(Y, M, X, sigma_yx)

        # hessian = compute_hessian_batch(M, sigma_yx)
        hessian = batch_effect_loss.compute_hessian()
        max_eigenvalue = torch.linalg.eigvalsh(hessian).max().item()
        base_step_size = self.batch_step_size_multiplier / max_eigenvalue

        if self.verbose > 1:
            print(f"Computed step size: {base_step_size} based on max eigenvalue: {max_eigenvalue}")

        loss_prev = float("inf")
        loss = float("inf")

        pbar = trange(
            self.batch_mini_iterations,
            disable=not self.verbose,
            desc="Updating batch effect with Nesterov's AG",
        )

        if update_alg == "nesterov":
            optimizer = NesterovGD(B.clone(), base_step_size)

            for epoch in pbar:
                loss, grad = batch_effect_loss(B)

                B_prev = B.clone()
                B = optimizer.step(grad)

                B.clip_(min=1e-5)
                # print(f"{B_prev = }")

                # Check convergence
                dB = (B_prev - B).abs().max().item()
                dloss = loss_prev - loss

                # Update descriptions
                pbar.set_description(f"Updating batch effect: loss = {loss:.1e}, δloss = {dloss:.1e}, δB = {dB:.1e}")

                loss_prev = loss

                # Check for convergence
                if dB < self.batch_tol:
                    break

                optimizer.set_parameters(B)

        elif update_alg == "gd":
            for epoch in pbar:
                loss, grad = batch_effect_loss(B, M, X, Y, sigma_yx)

                B_prev = B.clone()

                B = B - base_step_size * grad  # TODO: use Adam optimizer here

                B.clip_(min=1e-5)
                # Check convergence
                dB = (B_prev - B).abs().max().item()
                dloss = loss_prev - loss

                # Update descriptions
                pbar.set_description(
                    f"Updating batch effect: loss = {loss:.1e}, δloss = {dloss:.1e}, δB = {dB:.1e}",
                )

                loss_prev = loss

                # Check for convergence
                if dB < self.batch_tol:
                    break

        # Compute final loss
        final_loss = batch_effect_loss.compute_loss(B)
        return final_loss, B


class BatchEffectState(dict):
    """State to store batch effect parameters during Popari optimization."""

    def __init__(self, K: int, datasets: Sequence[PopariDataset], initial_context=None, context=None):
        self.datasets = datasets
        self.K = K
        self.initial_context = initial_context if initial_context else {"device": "cpu", "dtype": torch.float32}
        self.context = context if context else {"device": "cpu", "dtype": torch.float32}
        super().__init__()

        self.batch_effects = []

        for dataset in self.datasets:
            # Create batch effect tensor with shape [K] (one per metagene)
            batch_effect = torch.zeros(K, **self.context)
            self.__setitem__(dataset.name, batch_effect)
            self.batch_effects.append(batch_effect)
