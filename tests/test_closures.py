import os

import anndata as ad
import h5py
import numpy as np
import pytest
import torch

from popari._embedding_optimizer_closures import *
from popari._parameter_optimizer_closures import *


class MockProgressBar:
    def __init__(self, iterable=None):
        self.iterable = iterable or range(100)

    def __iter__(self):
        return iter(self.iterable)

    def set_description(self, desc):
        pass

    def close(self):
        pass


class MockIndependentSet:
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)


######################### Test Embedding Optimizer #########################


def test_multiplicative_update_wonbr_closure():
    X_prev = torch.tensor([[0.5, 0.5], [0.3, 0.7]], dtype=torch.float32)
    MTM = torch.tensor([[1.0, 0.2], [0.2, 1.0]], dtype=torch.float32)
    clipped_X = torch.tensor([[0.5, 0.5], [0.3, 0.7]], dtype=torch.float32)
    YM = torch.tensor([[0.8, 0.6], [0.4, 0.9]], dtype=torch.float32)
    Ynorm = 2.0
    prior_x_mode = "exponential shared fixed"
    prior_x = [torch.tensor([0.1, 0.1], dtype=torch.float32)]
    loss_prev = 10.0

    X_new, loss = multiplicative_update_wonbr_closure(
        X_prev,
        MTM,
        clipped_X,
        YM,
        Ynorm,
        prior_x_mode,
        prior_x,
        loss_prev,
    )

    expected_denominator = torch.tensor([[0.6, 0.6], [0.44, 0.76]], dtype=torch.float32)
    if prior_x_mode == "exponential shared fixed":
        expected_denominator += prior_x[0][None]
    expected_X = X_prev * (YM / expected_denominator)
    expected_X = torch.clip(expected_X, min=1e-10)

    assert torch.allclose(X_new, expected_X, rtol=1e-4)
    assert loss < loss_prev


def test_gradient_update_wonbr_closure():
    X = torch.tensor([[0.5, 0.5], [0.3, 0.7]], dtype=torch.float32)
    MTM = torch.tensor([[1.0, 0.2], [0.2, 1.0]], dtype=torch.float32)
    YM = torch.tensor([[0.8, 0.6], [0.4, 0.9]], dtype=torch.float32)
    prior_x_mode = "exponential shared fixed"
    prior_x = [torch.tensor([0.1, 0.1], dtype=torch.float32)]
    Ynorm = 2.0
    step_size = 0.1

    X_new, loss = gradient_update_wonbr_closure(X, MTM, YM, prior_x_mode, prior_x, Ynorm, step_size)

    expected_quadratic = torch.tensor([[0.6, 0.6], [0.44, 0.76]], dtype=torch.float32)
    expected_linear = YM.clone()
    if prior_x_mode == "exponential shared fixed":
        expected_linear = expected_linear - prior_x[0][None]

    expected_gradient = expected_quadratic - expected_linear
    expected_X = X - 0.1 * expected_gradient
    expected_X = torch.clip(expected_X, min=1e-10)

    assert torch.allclose(X_new, expected_X, rtol=1e-4)
    assert isinstance(loss, float)


def test_get_update_s_wnbr_closure():
    S = torch.tensor([[0.5], [0.7]], dtype=torch.float32)
    YM = torch.tensor([[0.8, 0.6], [0.4, 0.9]], dtype=torch.float32)
    MTM = torch.tensor([[1.0, 0.2], [0.2, 1.0]], dtype=torch.float32)
    prior_x = [torch.tensor([0.1], dtype=torch.float32)]
    prior_x_mode = "exponential shared fixed"
    Z = torch.tensor([[0.6, 0.4], [0.3, 0.7]], dtype=torch.float32)

    S_copy = S.clone()

    get_update_s_wnbr_closure(S, YM, MTM, prior_x, prior_x_mode, Z)

    expected_numerator = (YM * Z).sum(axis=1, keepdim=True) - prior_x[0][0] / 2
    expected_denominator = ((Z @ MTM) * Z).sum(axis=1, keepdim=True)
    expected_S = expected_numerator / expected_denominator
    expected_S.clip_(min=1e-5)

    assert torch.allclose(S, expected_S, rtol=1e-4)
    assert not torch.equal(S, S_copy)


def test_calc_func_grad():
    Z_batch = torch.tensor([[0.6, 0.4], [0.3, 0.7]], dtype=torch.float32)
    S_batch = torch.tensor([[0.5], [0.7]], dtype=torch.float32)
    quad = torch.tensor([[1.0, 0.2], [0.2, 1.0]], dtype=torch.float32)
    linear = torch.tensor([[0.8, 0.6], [0.4, 0.9]], dtype=torch.float32)

    f, g = calc_func_grad(Z_batch, S_batch, quad, linear)

    expected_t = (Z_batch @ quad) * (S_batch**2)
    expected_f = (expected_t * Z_batch).sum() / 2 - (linear * Z_batch).sum()
    expected_g = expected_t - linear
    row_sums = expected_g.sum(1, keepdim=True)
    expected_g = expected_g - row_sums

    assert abs(f - expected_f.item()) < 1e-4
    assert torch.allclose(g, expected_g, rtol=1e-4)


def test_update_z_gd_wnbr_closure():
    Z = torch.tensor([[0.6, 0.4], [0.3, 0.7]], dtype=torch.float32)
    base_step_size = 0.1
    S = torch.tensor([[0.5], [0.7]], dtype=torch.float32)
    N = 2
    E_adjacency_list = [torch.tensor([0]), torch.tensor([1])]
    device = "cpu"
    MTM = torch.tensor([[1.0, 0.2], [0.2, 1.0]], dtype=torch.float32)
    YM = torch.tensor([[0.8, 0.6], [0.4, 0.9]], dtype=torch.float32)
    adjacency_matrix = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    Sigma_x_inv = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    use_inplace_ops = False
    tol = 1e-5

    # Mock IndependentSet to control iteration
    import test

    test.IndependentSet = MockIndependentSet

    Z_new = update_z_gd_wnbr_closure(
        Z,
        base_step_size,
        S,
        N,
        E_adjacency_list,
        device,
        MTM,
        YM,
        adjacency_matrix,
        Sigma_x_inv,
        use_inplace_ops,
        tol,
    )

    assert Z_new.shape == Z.shape
    assert torch.all(Z_new >= 0)
    assert torch.allclose(Z_new.sum(dim=1), torch.tensor([1.0, 1.0]), rtol=1e-4)


def test_update_z_gd_nesterov_wnbr_closure():
    Z = torch.tensor([[0.6, 0.4], [0.3, 0.7]], dtype=torch.float32)
    N = 2
    S = torch.tensor([[0.5], [0.7]], dtype=torch.float32)
    MTM = torch.tensor([[1.0, 0.2], [0.2, 1.0]], dtype=torch.float32)
    YM = torch.tensor([[0.8, 0.6], [0.4, 0.9]], dtype=torch.float32)
    adjacency_matrix = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    Sigma_x_inv = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    E_adjacency_list = [torch.tensor([0]), torch.tensor([1])]
    device = "cpu"
    base_step_size = 0.1
    verbose = 0
    embedding_acceleration_trick = False
    update_s = lambda: None  # Mock function
    use_inplace_ops = False
    tol = 1e-4

    # Mock IndependentSet and trange to control iteration
    import test

    test.IndependentSet = MockIndependentSet
    test.trange = lambda *args, **kwargs: MockProgressBar(range(1))

    Z_new = update_z_gd_nesterov_wnbr_closure(
        Z,
        N,
        S,
        MTM,
        YM,
        adjacency_matrix,
        Sigma_x_inv,
        E_adjacency_list,
        device,
        base_step_size,
        verbose,
        embedding_acceleration_trick,
        update_s,
        use_inplace_ops,
        tol,
    )

    assert Z_new.shape == Z.shape
    assert torch.all(Z_new >= 0)
    assert torch.allclose(Z_new.sum(dim=1), torch.tensor([1.0, 1.0]), rtol=1e-4)


def test_compute_loss_wnbr_closure():
    Z = torch.tensor([[0.6, 0.4], [0.3, 0.7]], dtype=torch.float32)
    S = torch.tensor([[0.5], [0.7]], dtype=torch.float32)
    MTM = torch.tensor([[1.0, 0.2], [0.2, 1.0]], dtype=torch.float32)
    YM = torch.tensor([[0.8, 0.6], [0.4, 0.9]], dtype=torch.float32)
    Ynorm = 2.0
    prior_x_mode = "exponential shared fixed"
    prior_x = [torch.tensor([0.1], dtype=torch.float32)]
    Sigma_x_inv = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    adjacency_matrix = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)

    loss = compute_loss_wnbr_closure(
        Z,
        S,
        MTM,
        YM,
        Ynorm,
        prior_x_mode,
        prior_x,
        Sigma_x_inv,
        adjacency_matrix,
    )

    X = Z * S
    expected_loss = ((X @ MTM) * X).sum() / 2 - (X * YM).sum() + Ynorm / 2
    expected_loss += prior_x[0][0] * S.sum()
    expected_loss += ((adjacency_matrix @ Z) @ Sigma_x_inv).mul(Z).sum() / 2

    assert abs(loss - expected_loss.item()) < 1e-4


######################### Test Parameter Optimizer #########################


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

    loss = compute_loss_nll_M_closure(
        M,
        quadratic_factor,
        differential_regularization_quadratic_factor,
        linear_term,
        differential_regularization_linear_term,
        constant,
        metagene_mode,
        M_bar,
        lambda_M,
    )

    quad_grad = M @ (quadratic_factor + differential_regularization_quadratic_factor)
    expected_loss = (quad_grad * M).sum()
    lin_grad = linear_term + differential_regularization_linear_term
    expected_loss -= 2 * (lin_grad * M).sum()
    expected_loss += constant
    expected_loss /= 2

    assert abs(loss - expected_loss.item()) < 1e-4


def test_compute_loss_and_gradient():
    M = torch.tensor([[0.3, 0.7], [0.6, 0.4]], dtype=torch.float32)
    quadratic_factor = torch.tensor([[1.0, 0.2], [0.2, 1.0]], dtype=torch.float32)
    differential_regularization_quadratic_factor = torch.tensor([[0.1, 0.0], [0.0, 0.1]], dtype=torch.float32)
    verbose = 0
    linear_factor = torch.tensor([[0.8, 0.6], [0.4, 0.9]], dtype=torch.float32)
    differential_regularization_linear_factor = torch.tensor([[0.05, 0.05], [0.05, 0.05]], dtype=torch.float32)
    constant = 1.0
    metagene_mode = None
    M_bar = None
    lambda_M = 0.1
    M_constraint = "simplex"

    loss, grad = compute_loss_and_gradient(
        M,
        quadratic_factor,
        differential_regularization_quadratic_factor,
        verbose,
        linear_factor,
        differential_regularization_linear_factor,
        constant,
        metagene_mode,
        M_bar,
        lambda_M,
        M_constraint,
    )

    quad_grad = M @ (quadratic_factor + differential_regularization_quadratic_factor)
    expected_loss = (quad_grad * M).sum()
    lin_grad = linear_factor + differential_regularization_linear_factor
    expected_loss -= 2 * (lin_grad * M).sum()
    expected_loss += constant
    expected_loss /= 2

    expected_grad = quad_grad - lin_grad
    if M_constraint == "simplex":
        expected_grad -= expected_grad.sum(0, keepdim=True)

    assert abs(loss - expected_loss.item()) < 1e-4
    assert torch.allclose(grad, expected_grad, rtol=1e-4)


def test_estimate_M_nag_closure():
    M = torch.tensor([[0.3, 0.7], [0.6, 0.4]], dtype=torch.float32)
    verbose = 0
    quadratic_factor = torch.tensor([[1.0, 0.2], [0.2, 1.0]], dtype=torch.float32)
    differential_regularization_quadratic_factor = torch.tensor([[0.1, 0.0], [0.0, 0.1]], dtype=torch.float32)
    linear_factor = torch.tensor([[0.8, 0.6], [0.4, 0.9]], dtype=torch.float32)
    differential_regularization_linear_factor = torch.tensor([[0.05, 0.05], [0.05, 0.05]], dtype=torch.float32)
    constant = 1.0
    metagene_mode = None
    M_bar = None
    lambda_M = 0.1
    progress_bar = MockProgressBar(range(5))
    simplex_projection_mode = "exact"
    use_inplace_ops = False
    M_constraint = "simplex"
    tol = 1e-4
    verbose_bar = MockProgressBar()

    M_new = estimate_M_nag_closure(
        M,
        verbose,
        quadratic_factor,
        differential_regularization_quadratic_factor,
        linear_factor,
        differential_regularization_linear_factor,
        constant,
        metagene_mode,
        M_bar,
        lambda_M,
        progress_bar,
        simplex_projection_mode,
        use_inplace_ops,
        M_constraint,
        tol,
        verbose_bar,
    )
    assert M_new.shape == M.shape
    assert torch.all(M_new >= 0)
    if M_constraint == "simplex":
        assert torch.allclose(M_new.sum(dim=0), torch.tensor([1.0, 1.0]), rtol=1e-4)
