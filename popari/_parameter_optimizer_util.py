import awkward as ak
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm, trange

from popari._popari_dataset import PopariDataset
from popari.sample_for_integral import integrate_of_exponential_over_simplex
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


class ComputeLossNllM(nn.Module):
    def __init__(self, metagene_mode, M_bar, lambda_M):
        super().__init__()
        self.metagene_mode = metagene_mode
        self.M_bar = M_bar
        self.lambda_M = lambda_M

    def forward(
        self,
        M,
        quadratic_factor,
        differential_regularization_quadratic_factor,
        linear_term,
        differential_regularization_linear_term,
        constant,
    ):
        """Compute loss for NLL of metagene parameters.

        Args:
            M: metagene parameters
            quadratic_factor: quadratic term factor
            differential_regularization_quadratic_factor: quadratic term for regularization
            linear_term: linear term
            differential_regularization_linear_term: linear term for regularization
            constant: constant term

        Returns:
            Loss value

        """
        quadratic_factor_grad = M @ (quadratic_factor + differential_regularization_quadratic_factor)
        loss = (quadratic_factor_grad * M).sum()
        linear_term_grad = linear_term + differential_regularization_linear_term
        loss -= 2 * (linear_term_grad * M).sum()

        loss += constant

        if self.metagene_mode == "differential" and self.M_bar is not None:
            differential_regularization_term = (M @ differential_regularization_quadratic_factor * M).sum() - 2 * (
                differential_regularization_linear_term * M
            ).sum()
            group_weighting = 1 / len(self.M_bar)
            for group_M_bar in self.M_bar:
                differential_regularization_term += group_weighting * self.lambda_M * (group_M_bar * group_M_bar).sum()

        loss /= 2

        return loss.item()


class EstimateMNAG(nn.Module):
    def __init__(
        self,
        M_bar,
        lambda_M: float,
        M_constraint: str,
        use_inplace_ops: bool,
        verbose: str,
        n_epochs: int,
        tol: float,
        metagene_mode: str,
        simplex_projection_mode: str,
        quadratic_factor,
        differential_regularization_quadratic_factor,
        linear_factor,
        differential_regularization_linear_factor,
        constant,
    ):
        super().__init__()
        self.metagene_mode = metagene_mode
        self.M_bar = M_bar
        self.lambda_M = lambda_M
        self.M_constraint = M_constraint
        self.use_inplace_ops = use_inplace_ops
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.tol = tol
        self.simplex_projection_mode = simplex_projection_mode
        self.quadratic_factor = quadratic_factor
        self.differential_regularization_quadratic_factor = differential_regularization_quadratic_factor
        self.linear_factor = linear_factor
        self.differential_regularization_linear_factor = differential_regularization_linear_factor
        self.constant = constant

    def get_loss_and_gradient(
        self,
        M,
        batch_effects,
    ):
        """Compute loss and gradient for metagene optimization.

        Args:
            M: metagene parameters
            batch_effects: batch effect parameters

        Returns:
            Tuple of (loss, gradient)

        """

        quadratic_factor_grad = M @ (self.quadratic_factor + self.differential_regularization_quadratic_factor)
        loss = (quadratic_factor_grad * M).sum()
        verbose_description = ""
        if self.verbose > 2:
            verbose_description += f"M quadratic term: {loss:.1e}"

        linear_term_grad = self.linear_factor + self.differential_regularization_linear_factor
        loss -= 2 * (linear_term_grad * M).sum()
        grad = quadratic_factor_grad - linear_term_grad

        if not all(torch.all(tensor == 0) for tensor in batch_effects):
            loss += -0.5 * torch.sum(torch.log(torch.linalg.eigvalsh(M.T @ M) + 1e-10))
            grad += -M @ torch.inverse(M.T @ M + 1e-10 * torch.eye(M.shape[1], device=M.device))

        loss += self.constant

        if self.metagene_mode == "differential" and self.M_bar is not None:
            differential_regularization_term = (M @ self.differential_regularization_quadratic_factor * M).sum() - 2 * (
                self.differential_regularization_linear_factor * M
            ).sum()
            group_weighting = 1 / len(self.M_bar)
            for group_M_bar in self.M_bar:
                differential_regularization_term += group_weighting * self.lambda_M * (group_M_bar * group_M_bar).sum()

        if self.verbose > 2:
            if self.metagene_mode == "differential":
                verbose_description += f"M differential regularization term: {differential_regularization_term}"

        loss /= 2
        loss = loss.item()

        if self.M_constraint == "simplex":
            grad.sub_(grad.sum(0, keepdim=True))

        return loss, grad

    def forward(
        self,
        M,
        batch_effects,
    ):
        """Estimate metagenes using Nesterov accelerated gradient descent.

        Args:
            M: metagene parameters
            batch_effects: batch effect parameters

        Returns:
            Updated metagene parameters

        """
        loss, grad = self.get_loss_and_gradient(
            M,
            batch_effects,
        )
        if self.verbose > 1:
            print(f"M NAG Initial Loss: {loss}")

        step_size = 1 / torch.linalg.eigvalsh(self.quadratic_factor).max().item()
        loss_prev = np.inf

        optimizer = NesterovGD(M.clone(), step_size)
        verbose_bar = tqdm(disable=not (self.verbose > 2), bar_format="{desc}{postfix}")
        progress_bar = trange(self.n_epochs, leave=True, disable=not self.verbose, desc="Updating M", miniters=1000)
        for epoch in progress_bar:
            loss_prev = loss
            M_prev = M.clone()

            # Update M
            loss, grad = self.get_loss_and_gradient(
                M,
                batch_effects,
            )
            M = optimizer.step(grad)
            if self.simplex_projection_mode == "exact":
                if self.use_inplace_ops:
                    M = project_M_(M, self.M_constraint)
                else:
                    M = project_M(M, self.M_constraint)
            elif self.simplex_projection_mode == "approximate":
                raise NotImplementedError()

            optimizer.set_parameters(M)

            dloss = loss_prev - loss
            dM = (M_prev - M).abs().max().item()
            stop_criterion = dM < self.tol and epoch > 5
            assert not np.isnan(loss)
            if epoch % 5 == 0 or stop_criterion:
                description = f"Updating M: loss = {loss:.1e}, " f"%δloss = {dloss / loss:.1e}, " f"δM = {dM:.1e}"
                progress_bar.set_description(description)
            if stop_criterion:
                break

        verbose_bar.close()
        progress_bar.close()

        loss, _ = self.get_loss_and_gradient(M, batch_effects)

        if self.verbose > 1:
            print(f"M NAG Final Loss: {loss}")

        return M
