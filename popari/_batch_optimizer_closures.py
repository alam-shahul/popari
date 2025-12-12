import torch


def compute_loss_batch(B, M, X, Y, sigma_yx):
    """Compute loss for the batch effect optimization."""

    # Combine X and B for calculation
    MB = M @ B
    YM = Y.to(M.device) @ M
    MTM = M.T @ M

    term1 = MB.T @ MB
    term2 = (YM @ B) - (X @ MTM @ B)

    loss = (term1 + 2 * term2.sum(dim=0)) / (2 * sigma_yx**2)

    return loss.item()


def calc_func_grad_batch(B, M, X, Y, sigma_yx):
    """Calculate function value and gradient for batch effect optimization."""

    MTM = M.T @ M
    YM = Y.to(M.device) @ M
    MT_diff_sum = (YM - X @ MTM).sum(dim=0)

    g = (MTM @ B - MT_diff_sum) / (sigma_yx**2)
    return compute_loss_batch(B, M, X, Y, sigma_yx), g


def compute_hessian_batch(M, sigma_yx):
    """Calculate Hessian for batch effect optimization.

    Hessian is M^T M / (sigma_yx^2)

    """
    return M.T @ M / (sigma_yx**2)
