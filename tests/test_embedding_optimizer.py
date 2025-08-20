import os

import anndata as ad
import h5py
import numpy as np
import pytest
import torch

from popari._embedding_optimizer_util import (
    BatchEffectEmbeddingLossWithNeighborsNesterov,
    EmbeddingLossNoNeighborsGD,
    EmbeddingLossWithNeighborsNesterov,
)
from popari.util import convert_adjacency_matrix_to_awkward_array

# def test_multiplicative_update_wonbr_closure():
#     X_prev = torch.tensor([[0.5, 0.5], [0.3, 0.7]], dtype=torch.float32)
#     MTM = torch.tensor([[1.0, 0.2], [0.2, 1.0]], dtype=torch.float32)
#     YM = torch.tensor([[0.8, 0.6], [0.4, 0.9]], dtype=torch.float32)
#     Ynorm = 2.0
#     prior_x_mode = "exponential shared fixed"
#     prior_x = [torch.tensor([0.1, 0.1], dtype=torch.float32)]
#     loss_prev = 10.0
#     step_size = 1
#
#     no_neighbors_gd_loss = EmbeddingLossNoNeighborsGD(
#         MTM,
#         YM,
#         Ynorm,
#         prior_x_mode,
#         prior_x,
#         step_size,
#     )  # TODO: replace with multiplicative updates
#     X_new, loss = no_neighbors_gd_loss(X_prev)
#
#     expected_X = torch.tensor([[0.6000, 0.4000], [0.1600, 0.7400]], dtype=torch.float32)
#     expected_loss = 0.3819999694824219
#
#     assert torch.allclose(X_new, expected_X, rtol=1e-4)
#     assert loss < loss_prev
#     assert abs(loss - expected_loss) < 1e-4


@pytest.fixture(scope="function")
def embedding_loss_no_neighbors_gd():
    MTM = torch.tensor([[1.0, 0.2], [0.2, 1.0]], dtype=torch.float32)
    YM = torch.tensor([[0.8, 0.6], [0.4, 0.9]], dtype=torch.float32)
    prior_x_mode = "exponential shared fixed"
    prior_x = [torch.tensor([0.1, 0.1], dtype=torch.float32)]
    Ynorm = 2.0
    step_size = 0.1

    embedding_loss = EmbeddingLossNoNeighborsGD(MTM, YM, Ynorm, prior_x_mode, prior_x, step_size)

    return embedding_loss


def test_embedding_update_no_neighbors_integration(embedding_loss_no_neighbors_gd):
    X = torch.tensor([[0.5, 0.5], [0.3, 0.7]], dtype=torch.float32)
    X_new, loss = embedding_loss_no_neighbors_gd(X)

    expected_X = torch.tensor([[0.5100, 0.4900], [0.2860, 0.7040]], dtype=torch.float32)
    assert torch.allclose(X_new, expected_X, rtol=1e-4)
    assert isinstance(loss, float)


@pytest.fixture(scope="function")
def embedding_loss_with_neighbors_nesterov():
    Z = torch.tensor([[0.6, 0.4], [0.3, 0.7]], dtype=torch.float32)
    S = torch.tensor([[0.5], [0.7]], dtype=torch.float32)
    MTM = torch.tensor([[1.0, 0.2], [0.2, 1.0]], dtype=torch.float32)
    YM = torch.tensor([[0.8, 0.6], [0.4, 0.9]], dtype=torch.float32)
    Ynorm = 2.0
    prior_x_mode = "exponential shared fixed"
    prior_x = [torch.tensor([0.1], dtype=torch.float32)]
    Sigma_x_inv = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    adjacency_matrix = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    adjacency_list = convert_adjacency_matrix_to_awkward_array(adjacency_matrix)
    B = torch.zeros(MTM.shape[0])

    embedding_loss = EmbeddingLossWithNeighborsNesterov(
        Z=Z.clone(),
        S=S.clone(),
        MTM=MTM,
        YM=YM,
        Ynorm=Ynorm,
        adjacency_matrix=adjacency_matrix,
        prior_x_mode=prior_x_mode,
        prior_x=prior_x,
        Sigma_x_inv=Sigma_x_inv,
        E_adjacency_list=adjacency_list,
        device="cpu",
        base_step_size=0.1,
        verbose=0,
        embedding_acceleration_trick=True,
        use_inplace_ops=True,
        embedding_mini_iterations=1,
        tol=1e-5,
    )

    return embedding_loss


def test_embedding_loss_nesterov_compute_loss(embedding_loss_with_neighbors_nesterov):
    loss = embedding_loss_with_neighbors_nesterov.compute_loss()

    expected_loss = 0.93467
    assert abs(loss - expected_loss) < 1e-4


def test_embedding_loss_nesterov_update_z(embedding_loss_with_neighbors_nesterov):
    Z = embedding_loss_with_neighbors_nesterov.Z.clone()
    Z_new = embedding_loss_with_neighbors_nesterov.update_z(Z)

    assert Z_new.shape == Z.shape
    assert torch.all(Z_new >= 0)
    assert torch.allclose(Z_new.sum(dim=1), torch.tensor([1.0, 1.0]), rtol=1e-4)
    assert torch.allclose(Z_new, Z, rtol=1e-4)  # For some reason, this test case doesn't change


def test_embedding_loss_nesterov_update_s(embedding_loss_with_neighbors_nesterov):
    loss = embedding_loss_with_neighbors_nesterov
    S_copy = loss.S.clone()

    loss.update_s()

    # With batch effect, the formula changes to include B term
    expected_numerator = (loss.YM * loss.Z).sum(axis=1, keepdim=True) - loss.prior_x[0][
        0
    ] / 2  # TODO: change these to be actual constants
    expected_denominator = ((loss.Z @ loss.MTM) * loss.Z).sum(axis=1, keepdim=True)
    expected_S = expected_numerator / expected_denominator
    expected_S.clip_(min=1e-5)

    assert torch.allclose(loss.S, expected_S, rtol=1e-4)
    assert not torch.equal(loss.S, S_copy)


def test_embedding_loss_nesterov_get_batch_loss_and_grad(embedding_loss_with_neighbors_nesterov):
    Z_batch = torch.tensor([[0.6, 0.4], [0.3, 0.7]], dtype=torch.float32)
    S_batch = torch.tensor([[0.5], [0.7]], dtype=torch.float32)
    quad = torch.tensor([[1.0, 0.2], [0.2, 1.0]], dtype=torch.float32)
    linear = torch.tensor([[0.8, 0.6], [0.4, 0.9]], dtype=torch.float32)

    # f, g = get_batch_loss_and_grad(Z_batch, S_batch, quad, linear)
    # expected_t = (Z_batch @ quad) * (S_batch**2)
    # expected_f = ((expected_t * Z_batch).sum() / 2 - (linear * Z_batch).sum()).item()
    # expected_g = expected_t - linear
    # row_sums = expected_g.sum(1, keepdim=True)
    # expected_g = expected_g - row_sums

    f, g = embedding_loss_with_neighbors_nesterov.get_batch_loss_and_grad(Z_batch, S_batch, quad, linear)

    expected_f = -1.23031
    expected_g = torch.tensor([[0.4700, 0.6300], [0.5276, 0.1844]], dtype=torch.float32)
    assert abs(f - expected_f) < 1e-4
    assert torch.allclose(g, expected_g, rtol=1e-4)


@pytest.fixture(scope="function")
def batch_embedding_loss_with_neighbors_nesterov():
    Z = torch.tensor([[0.6, 0.4], [0.3, 0.7]], dtype=torch.float32)
    S = torch.tensor([[0.5], [0.7]], dtype=torch.float32)
    MTM = torch.tensor([[1.0, 0.2], [0.2, 1.0]], dtype=torch.float32)
    YM = torch.tensor([[0.8, 0.6], [0.4, 0.9]], dtype=torch.float32)
    Ynorm = 2.0
    prior_x_mode = "exponential shared fixed"
    prior_x = [torch.tensor([0.1], dtype=torch.float32)]
    Sigma_x_inv = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    adjacency_matrix = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    adjacency_list = convert_adjacency_matrix_to_awkward_array(adjacency_matrix)
    B = torch.zeros(MTM.shape[0])

    embedding_loss = BatchEffectEmbeddingLossWithNeighborsNesterov(
        Z=Z.clone(),
        S=S.clone(),
        B=B.clone(),
        MTM=MTM,
        YM=YM,
        Ynorm=Ynorm,
        adjacency_matrix=adjacency_matrix,
        prior_x_mode=prior_x_mode,
        prior_x=prior_x,
        Sigma_x_inv=Sigma_x_inv,
        E_adjacency_list=adjacency_list,
        device="cpu",
        base_step_size=0.1,
        verbose=0,
        embedding_acceleration_trick=True,
        use_inplace_ops=True,
        embedding_mini_iterations=1,
        tol=1e-5,
    )

    return embedding_loss


def test_batch_embedding_loss_nesterov_compute_loss(batch_embedding_loss_with_neighbors_nesterov):
    B = torch.tensor([0.3, 0.7], dtype=torch.float32)
    loss = batch_embedding_loss_with_neighbors_nesterov.compute_loss()

    expected_loss = 0.934679
    assert abs(loss - expected_loss) < 1e-4


def test_batch_embedding_loss_nesterov_update_s(batch_embedding_loss_with_neighbors_nesterov):
    loss = batch_embedding_loss_with_neighbors_nesterov
    S_copy = loss.S.clone()
    B = torch.tensor([0.3, 0.7], dtype=torch.float32)
    loss.update_s()

    expected_S = torch.tensor([[1.0877], [1.0542]])
    assert torch.allclose(loss.S, expected_S, rtol=1e-4)
    assert not torch.equal(loss.S, S_copy)
