import torch


def integrate_of_exponential_over_simplex(eta, eps=1e-30):
    """Return the log integral of ``exp(-eta @ x)`` over the unit simplex.

    The closed-form expression used here assumes that entries within each row
    of ``eta`` are numerically distinct.

    """

    num_factors = eta.shape[1]
    diagonal = torch.eye(num_factors, dtype=torch.bool, device=eta.device).unsqueeze(0)
    difference = eta[:, None, :] - eta[:, :, None]
    difference = difference.masked_fill(diagonal, 1)
    signs = difference.sign().prod(dim=-1)
    log_denominator = difference.abs().add(eps).log().masked_fill(diagonal, 0).sum(dim=-1)
    log_abs = -eta - log_denominator

    # signed logsumexp
    maxes, _ = log_abs.max(axis=1, keepdim=True)
    ret = log_abs.sub(maxes).exp()

    ret = ret.mul(signs).sum(axis=-1)
    ret = ret.clip(min=eps)

    squeezed_maxes = maxes.squeeze(axis=-1)
    ret = ret.log().add(squeezed_maxes)

    return ret
