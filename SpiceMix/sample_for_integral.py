import sys, os, time
from multiprocessing import Pool

import numpy as np
import torch
import scipy
from scipy.stats import truncnorm, multivariate_normal, mvn
from scipy.special import erf, loggamma


def project2simplex(x, dim, eps=1e-5):
	"""
	# https://math.stackexchange.com/questions/2402504/orthogonal-projection-onto-the-unit-simplex
	find a scalar u such that || (x-u)_+ ||_1 = 1
	"""
	u = x.sum(dim=dim, keepdim=True).sub_(1).div_(x.shape[dim])
	g_prev, g = None, None
	for i in range(x.shape[dim]):
		t = x.sub(u)
		g = (t > eps).sum(dim, keepdim=True).to(x.dtype)
		assert g.min() > 0, t.clip(min=0).sum(dim).min()
		if g_prev is not None and (g == g_prev).all(): break
		f = t.clip_(min=eps).sum(dim, keepdim=True).sub_(1)
		u.addcdiv_(f, g)
		g_prev = g
	assert (g == g_prev).all()
	x.sub_(u).clip_(min=eps)
	assert x.sum(dim=dim).sub_(1).abs_().max() < 1e-4, x.sum(dim=dim).sub_(1).abs_().max()
	return x


def test_project2simplex():
	x = torch.rand(100, 100)
	x = x * 3 - 1
	project2simplex(x, dim=0)


def integrate_of_exponential_over_simplex(eta):
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
		t = t.abs().clip(min=1e-10).log()
		assert torch.isfinite(t).all()
		t[:, k] = eta[:, k]
		A[:, k] = t.sum(-1).neg()
	assert torch.isfinite(A).all()
	# signed logsumexp
	o = A.max(-1, keepdim=True)[0]
	ret = A.sub(o).exp()
	assert torch.isfinite(ret).all()
	ret = ret.mul(signs).sum(-1)
	ret = ret.clip(min=1e-10)
	assert (ret > 0).all(), ret.min().item()
	ret = ret.log().add(o.squeeze(-1))
	return ret
