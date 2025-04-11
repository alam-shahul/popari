import torch
import torch.nn as nn


class BatchEffectLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, B, M, X, Y, sigma_yx):
        """Calculate function value and gradient for batch effect optimization.
        Args:
            B: `K`-dimensional batch effect vector
            M: `G x K`-dimensional metagene matrix
            X: `N x K`-dimensional embedding matrix
            Y: `N x G`-dimensional expression matrix
            sigma_yx: variance of cell expression reconstruction distribution

        Returns:
            batch effect optimization loss and gradient
        """
        MTM = M.T @ M
        YM = Y.to(M.device) @ M
        MT_diff_sum = (YM - X @ MTM).sum(dim=0)
        grad = (MTM @ B - MT_diff_sum) / (sigma_yx**2)
        loss = self.compute_loss(B, M, X, Y, sigma_yx)

        return loss, grad

    def compute_loss(self, B, M, X, Y, sigma_yx):
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
        MB = M @ B
        MTM = M.T @ M
        term1 = MB.T @ MB
        term2 = (Y @ MB) - (X @ MTM @ B)
        loss = (term1 + 2 * term2.sum(dim=0)) / (2 * sigma_yx**2)
        return loss.item()

    def compute_hessian(self, M, sigma_yx):
        """Calculate Hessian for batch effect optimization.

        Hessian is `M^T M / (sigma_yx^2)`

        Args:
            M: `G x K`-dimensional metagene matrix
            sigma_yx: variance of cell expression reconstruction distribution

        Returns:
            Hessian matrix for batch effect optimization

        """
        return M.T @ M / (sigma_yx**2)
