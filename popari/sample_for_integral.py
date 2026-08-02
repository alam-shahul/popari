import torch


def integrate_of_exponential_over_simplex(eta, eps=1e-30):
    assert torch.isfinite(eta).all()
    N, K = eta.shape
    log_abs = torch.empty_like(eta)
    signs = torch.empty_like(log_abs)
    for k in range(K):
        difference = eta - eta[:, [k]]
        difference[:, k] = 1
        difference_sign = difference.sign()
        signs[:, k] = difference_sign.prod(axis=1)
        # t = t.abs().clip(min=1e10).log()
        difference = difference.abs().add(eps).log()
        difference[:, k] = eta[:, k]

        log_abs[:, k] = difference.sum(axis=1)
    assert torch.isfinite(log_abs).all()
    log_abs.neg_()

    # signed logsumexp
    maxes, _ = log_abs.max(axis=1, keepdim=True)
    ret = log_abs.sub(maxes).exp()
    assert torch.isfinite(ret).all()

    ret = ret.mul(signs).sum(axis=-1)
    ret = ret.clip(min=eps)
    assert (ret > 0).all(), ret.min().item()

    squeezed_maxes = maxes.squeeze(axis=-1)
    ret = ret.log().add(squeezed_maxes)

    return ret
