import os

import anndata as ad
import h5py
import numpy as np
import pytest
import torch

from popari._batch_optimizer_util import BatchEffectLoss


@pytest.fixture(scope="function")
def batch_effect_loss():
    M = torch.tensor([[0.8, 0.2], [0.3, 0.7]], requires_grad=True)
    X = torch.tensor([[0.9, 0.1], [0.4, 0.6], [0.5, 0.5]], requires_grad=True)
    Y = torch.tensor([[0.7, 0.3], [0.2, 0.8], [0.4, 0.6]], requires_grad=True)
    sigma_yx = 0.1

    batch_effect_loss = BatchEffectLoss(Y, M, X, sigma_yx)

    return batch_effect_loss


def test_compute_loss(batch_effect_loss):
    B = torch.tensor([1.0, 0.5], requires_grad=True)
    loss = batch_effect_loss.compute_loss(B)
    expected_loss = 48.22499
    assert abs(loss - expected_loss) < 1e-4


def test_forward(batch_effect_loss):
    B = torch.tensor([1.0, 0.5], requires_grad=True)

    _, grad = batch_effect_loss(B)
    expected_grad = torch.tensor([112.3000, 48.7000])
    assert torch.allclose(grad, expected_grad, rtol=1e-4)
    assert grad.shape == B.shape


def test_compute_hessian(batch_effect_loss):
    expected_hessian = torch.tensor([[73.0, 37.0], [37.0, 53.0]])

    hessian = batch_effect_loss.compute_hessian()
    K = batch_effect_loss.MTM.shape[1]
    assert hessian.shape == (K, K)
    assert torch.allclose(hessian, hessian.T, rtol=1e-4)
    assert torch.allclose(hessian, expected_hessian, rtol=1e-4)
