import os

import anndata as ad
import h5py
import numpy as np
import pytest
import torch

from popari._batch_optimizer_closures import *
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

    # expected_denominator = torch.tensor([[0.6, 0.6], [0.44, 0.76]], dtype=torch.float32) + prior_x[0][None]
    # expected_X = X_prev * (YM / expected_denominator)
    # expected_X = torch.clip(expected_X, min=1e-10)
    expected_X = torch.tensor([[0.5714, 0.4286], [0.2222, 0.7326]], dtype=torch.float32)
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

    # expected_quadratic = torch.tensor([[0.6, 0.6], [0.44, 0.76]], dtype=torch.float32)
    # expected_linear = YM.clone() - prior_x[0][None]
    # expected_gradient = expected_quadratic - expected_linear
    # expected_X = X - 0.1 * expected_gradient
    # expected_X = torch.clip(expected_X, min=1e-10)
    expected_X = torch.tensor([[0.5100, 0.4900], [0.2860, 0.7040]], dtype=torch.float32)
    assert torch.allclose(X_new, expected_X, rtol=1e-4)
    assert isinstance(loss, float)


def test_get_update_s_wnbr_closure():
    S = torch.tensor([[0.5], [0.7]], dtype=torch.float32)
    YM = torch.tensor([[0.8, 0.6], [0.4, 0.9]], dtype=torch.float32)
    MTM = torch.tensor([[1.0, 0.2], [0.2, 1.0]], dtype=torch.float32)
    prior_x = [torch.tensor([0.1], dtype=torch.float32)]
    prior_x_mode = "exponential shared fixed"
    Z = torch.tensor([[0.6, 0.4], [0.3, 0.7]], dtype=torch.float32)
    B = torch.tensor([0.0, 0.0], dtype=torch.float32)

    S_copy = S.clone()

    get_update_s_wnbr_closure(S, YM, MTM, prior_x, prior_x_mode, Z, B)

    # expected_numerator = (YM * Z).sum(axis=1, keepdim=True) - prior_x[0][0] / 2
    # expected_denominator = ((Z @ MTM) * Z).sum(axis=1, keepdim=True)
    # expected_S = expected_numerator / expected_denominator
    # expected_S.clip_(min=1e-5)
    expected_S = torch.tensor([[1.0877], [1.0542]], dtype=torch.float32)
    assert torch.allclose(S, expected_S, rtol=1e-4)
    assert not torch.equal(S, S_copy)


def test_get_update_s_wnbr_with_batch_effect():
    S = torch.tensor([[0.5], [0.7]], dtype=torch.float32)
    YM = torch.tensor([[0.8, 0.6], [0.4, 0.9]], dtype=torch.float32)
    MTM = torch.tensor([[1.0, 0.2], [0.2, 1.0]], dtype=torch.float32)
    prior_x = [torch.tensor([0.1], dtype=torch.float32)]
    prior_x_mode = "exponential shared fixed"
    Z = torch.tensor([[0.6, 0.4], [0.3, 0.7]], dtype=torch.float32)
    B = torch.tensor([0.3, 0.7], dtype=torch.float32)

    S_copy = S.clone()

    get_update_s_wnbr_closure(S, YM, MTM, prior_x, prior_x_mode, Z, B)

    # With batch effect, the formula changes to include B term
    expected_numerator = (YM * Z - ((Z @ MTM) * B)).sum(axis=1, keepdim=True) - prior_x[0][0] / 2
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

    # expected_t = (Z_batch @ quad) * (S_batch**2)
    # expected_f = ((expected_t * Z_batch).sum() / 2 - (linear * Z_batch).sum()).item()
    # expected_g = expected_t - linear
    # row_sums = expected_g.sum(1, keepdim=True)
    # expected_g = expected_g - row_sums
    expected_f = -1.23031
    expected_g = torch.tensor([[0.4700, 0.6300], [0.5276, 0.1844]], dtype=torch.float32)
    assert abs(f - expected_f) < 1e-4
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
    B = torch.tensor([0.0, 0.0], dtype=torch.float32)

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
        B,
    )

    X = Z * S
    expected_loss = ((X @ MTM) * X).sum() / 2 - (X * YM).sum() + Ynorm / 2
    expected_loss += prior_x[0][0] * S.sum()
    expected_loss += ((adjacency_matrix @ Z) @ Sigma_x_inv).mul(Z).sum() / 2

    assert abs(loss - expected_loss.item()) < 1e-4


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
    B = torch.zeros(MTM.shape[0])

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
        B,
    )

    # XB = Z * S + B
    # expected_loss = ((XB @ MTM) * XB).sum() / 2 - (XB * YM).sum() + Ynorm / 2
    # expected_loss += prior_x[0][0] * S.sum()
    # expected_loss += ((adjacency_matrix @ Z) @ Sigma_x_inv).mul(Z).sum() / 2
    # expected_loss = expected_loss.item()
    expected_loss = 0.93467
    assert abs(loss - expected_loss) < 1e-4


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

    # quad_grad = M @ (quadratic_factor + differential_regularization_quadratic_factor)
    # expected_loss = (quad_grad * M).sum()
    # lin_grad = linear_term + differential_regularization_linear_term
    # expected_loss -= 2 * (lin_grad * M).sum()
    # expected_loss += constant
    # expected_loss /= 2
    # expected_loss = expected_loss.item()
    expected_loss = -0.165
    assert abs(loss - expected_loss) < 1e-4


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
    batch_effects = [torch.tensor([0.0, 0.0], dtype=torch.float32), torch.tensor([0.0, 0.0], dtype=torch.float32)]

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
        batch_effects,
    )

    # quad_grad = M @ (quadratic_factor + differential_regularization_quadratic_factor)
    # expected_loss = (quad_grad * M).sum()
    # lin_grad = linear_factor + differential_regularization_linear_factor
    # expected_loss -= 2 * (lin_grad * M).sum()
    # expected_loss += constant
    # expected_loss /= 2
    # expected_loss = expected_loss.item()

    # expected_grad = quad_grad - lin_grad
    # expected_grad -= expected_grad.sum(0, keepdim=True)
    expected_loss = -0.165
    expected_grad = torch.tensor([[-0.29, 0.39], [0.38, -0.18]], dtype=torch.float32)

    assert abs(loss - expected_loss) < 1e-4
    assert torch.allclose(grad, expected_grad, rtol=1e-4)


def test_compute_loss_and_gradient_with_batch_effect():
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
    batch_effects = [torch.tensor([0.3, 0.7], dtype=torch.float32), torch.tensor([0.2, 0.8], dtype=torch.float32)]

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
        batch_effects,
    )

    # quad_grad = M @ (quadratic_factor + differential_regularization_quadratic_factor)
    # expected_loss = (quad_grad * M).sum()
    # lin_grad = linear_factor + differential_regularization_linear_factor
    # expected_loss -= 2 * (lin_grad * M).sum()
    # expected_loss += constant

    # Add determinant term (log det term)
    # log_det_term = -0.5 * torch.sum(torch.log(torch.linalg.eigvalsh(M.T @ M) + 1e-10))
    # expected_loss += log_det_term
    # expected_loss /= 2
    # expected_loss = expected_loss.item()

    # expected_grad = quad_grad - lin_grad
    # det_grad = -M @ torch.inverse(M.T @ M + 1e-10 * torch.eye(M.shape[1], device=M.device))
    # expected_grad += det_grad
    # expected_grad -= expected_grad.sum(0, keepdim=True)

    expected_loss = 0.436986
    expected_grad = torch.tensor([[2.0433, -0.61], [-0.9533, 1.82]], dtype=torch.float32)
    assert abs(loss - expected_loss) < 1e-4
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
    batch_effects = [torch.tensor([0.0, 0.0], dtype=torch.float32), torch.tensor([0.0, 0.0], dtype=torch.float32)]

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
        batch_effects,
    )
    assert M_new.shape == M.shape
    assert torch.all(M_new >= 0)
    assert torch.allclose(M_new.sum(dim=0), torch.tensor([1.0, 1.0]), rtol=1e-4)


######################### Test Batch Effect Optimizer #########################


def test_compute_loss_batch():
    B = torch.tensor([0.3, 0.7], dtype=torch.float32)
    M = torch.tensor([[1.0, 0.2], [0.2, 1.0]], dtype=torch.float32)
    X = torch.tensor([[0.5, 0.5], [0.3, 0.7]], dtype=torch.float32)
    Y = torch.tensor([[0.8, 0.6], [0.4, 0.9]], dtype=torch.float32)
    sigma_yx = 1.0

    loss = compute_loss_batch(B, M, X, Y, sigma_yx)

    # MB = M @ B
    # YM = Y @ M
    # MTM = M.T @ M
    # term1 = MB.T @ MB
    # term2 = (YM @ B) - (X @ MTM @ B)
    # expected_loss = (term1 + 2 * term2.sum(dim=0)) / (2 * sigma_yx**2)
    # expected_loss = expected_loss.item()
    expected_loss = 0.5624

    assert abs(loss - expected_loss) < 1e-4
    assert isinstance(loss, float)


def test_calc_func_grad_batch():
    B = torch.tensor([0.3, 0.7], dtype=torch.float32)
    M = torch.tensor([[1.0, 0.2], [0.2, 1.0]], dtype=torch.float32)
    X = torch.tensor([[0.5, 0.5], [0.3, 0.7]], dtype=torch.float32)
    Y = torch.tensor([[0.8, 0.6], [0.4, 0.9]], dtype=torch.float32)
    sigma_yx = 1.0

    loss, grad = calc_func_grad_batch(B, M, X, Y, sigma_yx)

    expected_grad = torch.tensor([0.4040, 0.6760], dtype=torch.float32)
    expected_loss = 0.5624

    assert abs(loss - expected_loss) < 1e-4
    assert torch.allclose(grad, expected_grad, rtol=1e-4)


def test_compute_hessian_batch():
    M = torch.tensor([[1.0, 0.2], [0.2, 1.0]], dtype=torch.float32)
    sigma_yx = 1.0

    hessian = compute_hessian_batch(M, sigma_yx)

    # expected_hessian = M.T @ M / (sigma_yx**2)
    expected_hessian = torch.tensor([[1.04, 0.4], [0.4, 1.04]], dtype=torch.float32)

    assert torch.allclose(hessian, expected_hessian, rtol=1e-4)
