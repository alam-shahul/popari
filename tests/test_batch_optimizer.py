import os

import anndata as ad
import h5py
import numpy as np
import pytest
import torch

from popari._batch_optimizer_util import BatchEffectLoss


@pytest.fixture(scope="function")
def batch_effect_module():
    batch_effect_loss = BatchEffectLoss()

    return batch_effect_loss


def test_compute_loss(batch_effect_module):
    B = torch.tensor([[1.0, 0.5], [0.5, 1.0]], requires_grad=True)
    M = torch.tensor([[0.8, 0.2], [0.3, 0.7]], requires_grad=True)
    X = torch.tensor([[0.9, 0.1], [0.4, 0.6]], requires_grad=True)
    Y = torch.tensor([[0.7, 0.3], [0.2, 0.8]], requires_grad=True)
    sigma_yx = 0.1

    loss = batch_effect_module.compute_loss(B, M, X, Y, sigma_yx)
    print("loss", loss)


def test_forward(batch_effect_module):
    B = torch.tensor([[1.0, 0.5], [0.5, 1.0]], requires_grad=True)
    M = torch.tensor([[0.8, 0.2], [0.3, 0.7]], requires_grad=True)
    X = torch.tensor([[0.9, 0.1], [0.4, 0.6]], requires_grad=True)
    Y = torch.tensor([[0.7, 0.3], [0.2, 0.8]], requires_grad=True)
    sigma_yx = 0.1

    _, grad = batch_effect_module(B, M, X, Y, sigma_yx)
    print("grad", grad)
    assert grad.shape == B.shape


def test_compute_hessian(batch_effect_module):
    M = torch.tensor([[0.8, 0.2], [0.3, 0.7]], requires_grad=True)
    sigma_yx = 0.1

    expected_hessian = torch.tensor([[73.0, 37.0], [37.0, 53.0]])

    hessian = batch_effect_module.compute_hessian(M, sigma_yx)
    K = M.shape[1]
    assert hessian.shape == (K, K)
    assert torch.allclose(hessian, hessian.T, rtol=1e-4)
    assert torch.allclose(hessian, expected_hessian, rtol=1e-4)
