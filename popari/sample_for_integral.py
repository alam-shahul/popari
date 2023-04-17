import sys, os, time
from multiprocessing import Pool

import numpy as np
import torch
import scipy

def project2simplex(y, dim=0, zero_threshold=1e-10):
    """
    # https://math.stackexchange.com/questions/2402504/orthogonal-projection-onto-the-unit-simplex
    find a scalar mu such that || (y-mu)_+ ||_1 = 1

    Currently uses Newton's method to optimize || y - mu ||^2

    TODO: try implementing it this way instead: https://arxiv.org/pdf/1101.6081.pdf

    Args:
        y: vector to be projected to unit simplex
    """
    
    num_components = y.shape[dim]
    
    mu = (y.sum(dim=dim, keepdim=True) - 1) / num_components
    previous_derivative = derivative = None
    for _ in range(num_components):
        difference = y - mu
        derivative = -(difference > zero_threshold).sum(dim=dim, keepdim=True).to(y.dtype)
        assert -derivative.min() > 0, difference.clip(min=0).sum(dim).min()
        if previous_derivative is not None and (derivative == previous_derivative).all():
            break
        objective_value = torch.clip(difference, min=zero_threshold).sum(dim=dim, keepdim=True) - 1
        newton_update = objective_value / derivative
        mu -= newton_update
        previous_derivative = derivative
    assert (derivative == previous_derivative).all()
    
    assert not torch.isnan(y).any(), y

    y = (y - mu).clip(min=zero_threshold)
    assert not torch.isnan(y).any(), (mu, derivative)

    assert y.sum(dim=dim).sub_(1).abs_().max() < 1e-3, y.sum(dim=dim).sub_(1).abs_().max()
    
    return y


def test_project2simplex():
    x = torch.rand(100, 100)
    x = x * 3 - 1
    project2simplex(x, dim=0)

def test_project2simplex_basic():
    x = torch.tensor([1/2, 1/4, 1/4])
    assert x.allclose(project2simplex(x.clone(), dim=0))

def integrate_of_exponential_over_simplex(eta, eps=1e-15):
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

if __name__ == "__main__":
    test_project2simplex_basic()
