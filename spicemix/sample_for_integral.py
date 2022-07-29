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
    #print(mu)
    if torch.isnan(mu).any():
        np.save("../problematic_y.npy", y.detach().cpu().numpy())

    y = (y - mu).clip(min=zero_threshold)
    assert not torch.isnan(y).any(), (mu, derivative)
    # print(y.sum(dim=dim).sub_(1))

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
	A = torch.empty_like(eta)
	signs = torch.empty_like(A)
	for k in range(K):
		t = eta - eta[:, [k]]
		assert torch.isfinite(t).all()
		t[:, k] = 1
		tsign = t.sign()
		signs[:, k] = tsign.prod(-1)
		# t = t.abs().clip(min=1e-10).log()
		t = t.abs().add(eps).log()
		assert torch.isfinite(t).all()
		t[:, k] = -eta[:, k]
		A[:, k] = t.sum(-1).neg()
	assert torch.isfinite(A).all()
	# signed logsumexp
	o = A.max(-1, keepdim=True)[0]
	ret = A.sub(o).exp()
	assert torch.isfinite(ret).all()
	ret = ret.mul(signs).sum(-1)
	ret = ret.clip(min=eps)
	assert (ret > 0).all(), ret.min().item()
	ret = ret.log().add(o.squeeze(-1))
	return ret

if __name__ == "__main__":
    test_project2simplex_basic()
