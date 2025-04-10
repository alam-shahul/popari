import os

import anndata as ad
import h5py
import numpy as np
import pytest
import torch

from popari._parameter_optimizer_util import EstimateMNAG


def test_compute_loss_nll_M_closure():
    M = torch.tensor([[0.3, 0.7], [0.6, 0.4]], dtype=torch.float32)
    quadratic_factor = torch.tensor([[1.0, 0.2], [0.2, 1.0]], dtype=torch.float32)
    differential_regularization_quadratic_factor = torch.tensor([[0.1, 0.0], [0.0, 0.1]], dtype=torch.float32)
    linear_term = torch.tensor([[0.8, 0.6], [0.4, 0.9]], dtype=torch.float32)
    differential_regularization_linear_term = torch.tensor([[0.05, 0.05], [0.05, 0.05]], dtype=torch.float32)
    constant = 1.0
    metagene_mode = None
    M_bar = None
    lambda_M = 0.1

    # loss = compute_loss_nll_M_closure(
    #     M,
    #     quadratic_factor,
    #     differential_regularization_quadratic_factor,
    #     linear_term,
    #     differential_regularization_linear_term,
    #     constant,
    #     metagene_mode,
    #     M_bar,
    #     lambda_M,
    # )
    # quad_grad = M @ (quadratic_factor + differential_regularization_quadratic_factor)
    # expected_loss = (quad_grad * M).sum()
    # lin_grad = linear_term + differential_regularization_linear_term
    # expected_loss -= 2 * (lin_grad * M).sum()
    # expected_loss += constant
    # expected_loss /= 2
    # expected_loss = expected_loss.item()

    compute_loss_nll_M = ComputeLossNllM(metagene_mode, M_bar, lambda_M)
    loss = compute_loss_nll_M.forward(
        M,
        quadratic_factor,
        differential_regularization_quadratic_factor,
        linear_term,
        differential_regularization_linear_term,
        constant,
    )

    expected_loss = -0.165
    assert abs(loss - expected_loss) < 1e-4


@pytest.fixture(scope="function")
def metagene_loss_nag():
    verbose = 0
    quadratic_factor = torch.tensor([[1.0, 0.2], [0.2, 1.0]], dtype=torch.float32)
    differential_regularization_quadratic_factor = torch.tensor([[0.1, 0.0], [0.0, 0.1]], dtype=torch.float32)
    linear_factor = torch.tensor([[0.8, 0.6], [0.4, 0.9]], dtype=torch.float32)
    differential_regularization_linear_factor = torch.tensor([[0.05, 0.05], [0.05, 0.05]], dtype=torch.float32)
    constant = 1.0
    metagene_mode = None
    M_bar = None
    lambda_M = 0.1
    simplex_projection_mode = "exact"
    use_inplace_ops = False
    M_constraint = "simplex"
    tol = 1e-4
    n_epochs = 10000

    metagene_loss = EstimateMNAG(
        M_bar=M_bar,
        lambda_M=lambda_M,
        M_constraint=M_constraint,
        use_inplace_ops=use_inplace_ops,
        verbose=verbose,
        n_epochs=n_epochs,
        metagene_mode=metagene_mode,
        tol=tol,
        simplex_projection_mode=simplex_projection_mode,
        quadratic_factor=quadratic_factor,
        differential_regularization_quadratic_factor=differential_regularization_quadratic_factor,
        linear_factor=linear_factor,
        differential_regularization_linear_factor=differential_regularization_linear_factor,
        constant=constant,
    )

    return metagene_loss


def test_get_loss_and_gradient(metagene_loss_nag):
    M = torch.tensor([[0.3, 0.7], [0.6, 0.4]], dtype=torch.float32)
    batch_effects = [torch.tensor([0.0, 0.0], dtype=torch.float32), torch.tensor([0.0, 0.0], dtype=torch.float32)]

    loss, grad = metagene_loss_nag.get_loss_and_gradient(M, batch_effects)

    expected_loss = -0.165
    expected_grad = torch.tensor([[-0.29, 0.39], [0.38, -0.18]], dtype=torch.float32)

    assert abs(loss - expected_loss) < 1e-4
    assert torch.allclose(grad, expected_grad, rtol=1e-4)


def test_get_loss_and_gradient_with_batch_effect(metagene_loss_nag):
    M = torch.tensor([[0.3, 0.7], [0.6, 0.4]], dtype=torch.float32)
    batch_effects = [torch.tensor([0.3, 0.7], dtype=torch.float32), torch.tensor([0.2, 0.8], dtype=torch.float32)]

    loss, grad = metagene_loss_nag.get_loss_and_gradient(M, batch_effects)

    expected_loss = 0.436986
    expected_grad = torch.tensor([[2.0433, -0.61], [-0.9533, 1.82]], dtype=torch.float32)
    assert abs(loss - expected_loss) < 1e-4
    assert torch.allclose(grad, expected_grad, rtol=1e-4)


def test_estimate_M_nag_closure(metagene_loss_nag):
    M = torch.tensor([[0.3, 0.7], [0.6, 0.4]], dtype=torch.float32)
    batch_effects = [torch.tensor([0.0, 0.0], dtype=torch.float32), torch.tensor([0.0, 0.0], dtype=torch.float32)]
    M_new = metagene_loss_nag(M, batch_effects)

    assert M_new.shape == M.shape
    assert torch.all(M_new >= 0)
    assert torch.allclose(M_new.sum(dim=0), torch.tensor([1.0, 1.0]), rtol=1e-4)
