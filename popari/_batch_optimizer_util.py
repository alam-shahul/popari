import torch
import torch.nn as nn


class BatchEffectLoss(nn.Module):
    def __init__(self, Y, M, X, sigma_yx, prior_batch_mode, prior_batch):
        super().__init__()

        self.MTM = M.T @ M
        self.YM = Y @ M
        self.X = X

        self.MT_diff_sum = (self.YM - self.X @ self.MTM).sum(dim=0)
        self.sigma_yx = sigma_yx
        self.prior_batch_mode = prior_batch_mode
        self.prior_batch = prior_batch

    def forward(self, B):
        """Calculate function value and gradient for batch effect optimization.
        Args:
            B: `K`-dimensional batch effect vector
            M: `G x K`-dimensional metagene matrix
            X: `N x K`-dimensional embedding matrix
            sigma_yx: variance of cell expression reconstruction distribution

        Returns:
            batch effect optimization loss and gradient
        """
        numerator = self.MTM @ B - self.MT_diff_sum
        if self.prior_batch_mode == "exponential":
            numerator += (self.prior_batch[0][None] * (self.sigma_yx**2) * (self.MTM @ B)).squeeze()
        grad = (numerator) / (self.sigma_yx**2)
        loss = self.compute_loss(B)

        return loss, grad

    def compute_loss(self, B):
        """Compute loss for the batch effect optimization.

        Args:
            B: `K`-dimensional batch effect vector
            M: `G x K`-dimensional metagene matrix
            X: `N x K`-dimensional embedding matrix
            Y: `N x G`-dimensional expression matrix
            sigma_yx: variance of cell expression reconstruction distribution

        Returns:
            batch effect optimization loss

        """
        term1 = B @ self.MTM @ B
        term2 = (self.YM @ B) - (self.X @ self.MTM @ B)
        numerator = term1 + 2 * term2.sum(dim=0)
        if self.prior_batch_mode == "exponential":
            numerator += 2 * torch.sum(self.prior_batch[0][None] * (self.sigma_yx**2) * (B @ self.MTM @ B))
        loss = (numerator) / (2 * self.sigma_yx**2)
        return loss.item()

    def compute_hessian(self):
        """Calculate Hessian for batch effect optimization.

        Hessian is `M^T M / (sigma_yx^2)`

        Args:
            M: `G x K`-dimensional metagene matrix
            sigma_yx: variance of cell expression reconstruction distribution

        Returns:
            Hessian matrix for batch effect optimization

        """
        numerator = self.MTM
        if self.prior_batch_mode == "exponential":
            numerator += self.prior_batch[0][None] * (self.sigma_yx**2) * self.MTM
        return numerator / (self.sigma_yx**2)
