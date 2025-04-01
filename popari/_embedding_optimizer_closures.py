import logging
import time
from typing import Sequence

import numpy as np
import torch
from tqdm.auto import tqdm, trange

from popari._popari_dataset import PopariDataset
from popari.util import (
    IndependentSet,
    NesterovGD,
    convert_numpy_to_pytorch_sparse_coo,
    get_datetime,
    project2simplex,
    project2simplex_,
    project_M,
    project_M_,
    sample_graph_iid,
)

######################### Estimate Weight WONBR Closure Functions #########################


def multiplicative_update_wonbr_closure(X_prev, MTM, clipped_X, YM, Ynorm, prior_x_mode, prior_x, loss_prev):
    def multiplicative_update(X_prev):
        """TODO:UNTESTED."""
        X = torch.clip(X_prev, min=1e-10)
        loss = ((X @ MTM) * X).sum() / 2 - clipped_X.view(-1) @ YM.view(-1) + Ynorm / 2
        numerator = YM
        denominator = X @ MTM
        if prior_x_mode == "exponential shared fixed":
            # see sklearn.decomposition.NMF
            loss += (X @ prior_x[0]).sum()
            denominator += prior_x[0][None]
        else:
            raise NotImplementedError

        loss = loss.item()
        assert loss <= loss_prev * (1 + 1e-4), (loss_prev, loss, (loss_prev - loss) / loss)
        multiplicative_factor = numerator / denominator
        X *= multiplicative_factor
        torch.clip(X, min=1e-10)

        return X, loss

    return multiplicative_update(X_prev)


def gradient_update_wonbr_closure(X, MTM, YM, prior_x_mode, prior_x, Ynorm, step_size):
    def gradient_update(X):
        """TODO:UNTESTED."""
        quadratic_term_gradient = X @ MTM
        linear_term_gradient = YM
        if prior_x_mode == "exponential shared fixed":
            linear_term_gradient = linear_term_gradient - prior_x[0][None]
        elif not prior_x_mode:
            pass
        else:
            raise NotImplementedError
        loss = (quadratic_term_gradient * X).sum().item() / 2 - (linear_term_gradient * X).sum().item() + Ynorm / 2
        gradient = quadratic_term_gradient - linear_term_gradient
        X = X.sub(gradient, alpha=step_size)
        X = torch.clip(X, min=1e-10)

        return X, loss

    return gradient_update(X)


######################### Estimate Weight WNBR Closure Functions #########################


def get_update_s_wnbr_closure(S, YM, MTM, prior_x, prior_x_mode, Z, B):
    def update_s():
        # S[:] = (YM * Z).sum(axis=1, keepdim=True)
        S[:] = (YM * Z - ((Z @ MTM) * B)).sum(axis=1, keepdim=True)
        if prior_x_mode == "exponential shared fixed":
            # TODO: why divide by two?
            S.sub_(prior_x[0][0] / 2)
        elif not prior_x_mode:
            pass
        else:
            raise NotImplementedError

        denominator = ((Z @ MTM) * Z).sum(axis=1, keepdim=True)
        S.div_(denominator)
        S.clip_(min=1e-5)
        return

    return update_s()


"""
def calc_func_grad_wnbr_closure(Z_batch, S_batch, quad, linear):
    def calc_func_grad(Z_batch, S_batch, quad, linear):
        t = (Z_batch @ quad).mul_(S_batch**2)
        f = (t * Z_batch).sum() / 2
        g = t
        t = linear
        f -= (t * Z_batch).sum()
        g -= t
        g.sub_(g.sum(1, keepdim=True))

        return f.item(), g
    return calc_func_grad(Z_batch, S_batch, quad, linear)
"""


def calc_func_grad(Z_batch, S_batch, quad, linear):
    t = (Z_batch @ quad).mul_(S_batch**2)
    f = (t * Z_batch).sum() / 2
    g = t
    t = linear
    f -= (t * Z_batch).sum()
    g -= t
    g.sub_(g.sum(1, keepdim=True))

    return f.item(), g


def update_z_gd_wnbr_closure(
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
):
    def update_z_gd(Z):
        step_size = base_step_size / S.square()
        pbar = tqdm(range(N), leave=False, disable=True)
        for idx in IndependentSet(E_adjacency_list, device=device, batch_size=128):
            step_size_scale = 1
            quad_batch = MTM
            linear_batch = YM[idx] * S[idx] - torch.index_select(adjacency_matrix, 0, idx) @ Z @ Sigma_x_inv
            Z_batch = Z[idx].contiguous()
            S_batch = S[idx].contiguous()
            step_size_batch = step_size[idx].contiguous()
            func, grad = calc_func_grad(Z_batch, S_batch, quad_batch, linear_batch)
            while True:
                Z_batch_new = Z_batch - step_size_batch * step_size_scale * grad
                if use_inplace_ops:
                    Z_batch_new = project2simplex_(Z_batch_new, dim=1)
                else:
                    Z_batch_new = project2simplex(Z_batch_new, dim=1)
                dZ = Z_batch_new.sub(Z_batch).abs().max().item()
                func_new, grad_new = calc_func_grad(Z_batch_new, S_batch, quad_batch, linear_batch)
                if func_new < func:
                    Z_batch = Z_batch_new
                    func = func_new
                    grad = grad_new
                    step_size_scale *= 1.1
                    continue
                else:
                    step_size_scale *= 0.5
                if dZ < tol or step_size_scale < 0.5:
                    break
            assert step_size_scale > 0.1
            Z[idx] = Z_batch
            pbar.set_description(f"Updating Z w/ nbrs via line search: lr={step_size_scale:.1e}")
            pbar.update(len(idx))
        pbar.close()
        return Z

    return update_z_gd(Z)


def update_z_gd_nesterov_wnbr_closure(
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
):
    def update_z_gd_nesterov(Z):
        pbar = trange(N, leave=False, disable=True, desc="Updating Z w/ nbrs via Nesterov GD")
        func, grad = calc_func_grad(Z, S, MTM, YM * S - adjacency_matrix @ Z @ Sigma_x_inv / 2)
        for idx in IndependentSet(E_adjacency_list, device=device, batch_size=1024):
            quad_batch = MTM
            linear_batch_spatial = -torch.index_select(adjacency_matrix, 0, idx) @ Z @ Sigma_x_inv
            Z_batch = Z[idx].contiguous()
            S_batch = S[idx].contiguous()
            optimizer = NesterovGD(Z_batch, base_step_size / S_batch.square())
            ppbar = trange(100, leave=False, disable=not (verbose > 3))
            for i_iter in ppbar:
                if embedding_acceleration_trick:
                    update_s  # TODO: update S_batch directly
                S_batch = S[idx].contiguous()
                linear_batch = linear_batch_spatial + YM[idx] * S_batch
                if i_iter == 0:
                    func, grad = calc_func_grad(Z_batch, S_batch, quad_batch, linear_batch)
                    func, grad = calc_func_grad(Z, S, MTM, YM * S - adjacency_matrix @ Z @ Sigma_x_inv / 2)
                NesterovGD.step_size = base_step_size / S_batch.square()  # TM: I think this converges as s converges
                func, grad = calc_func_grad(Z_batch, S_batch, quad_batch, linear_batch)
                # grad_limit = torch.quantile(torch.abs(grad), 0.9)
                # max_before = torch.max(torch.abs(grad))
                # grad.clamp_(min=-grad_limit, max=grad_limit)
                # max_after = torch.max(torch.abs(grad))
                Z_batch_prev = Z_batch.clone()
                Z_batch = optimizer.step(grad)
                # if max(torch.linalg.norm(Z_batch_copy, ord=1, axis=1)) > 1000:
                #     print(f"L1 norm of Z_batch_copy: {torch.linalg.norm(Z_batch_copy, ord=1, axis=1)}")
                #     print(f"Max L1 norm of Z_batch_copy: {max(torch.linalg.norm(Z_batch_copy, ord=1, axis=1))}")
                #     print(f"L2 norm of Z_batch_copy: {torch.linalg.norm(Z_batch_copy, ord=2, axis=1)}")
                #     print(f"Max L2 norm of Z_batch_copy: {max(torch.linalg.norm(Z_batch_copy, ord=2, axis=1))}")
                #     print(f"Grad: {grad}")
                #     print(f"Grad max before: {max_before}")
                #     print(f"grad limit:{grad_limit}")
                #     print(f"Grad max after: {max_after}")

                if use_inplace_ops:
                    Z_batch = project2simplex_(Z_batch, dim=1)
                else:
                    Z_batch = project2simplex(Z_batch, dim=1)

                optimizer.set_parameters(Z_batch)

                dZ = (Z_batch_prev - Z_batch).abs().max().item()
                Z[idx] = Z_batch
                description = f"func={func:.1e}, dZ={dZ:.1e}"
                ppbar.set_description(description)
                if dZ < tol:
                    break
            ppbar.close()

            Z[idx] = Z_batch
            func, grad = calc_func_grad(Z_batch, S_batch, quad_batch, linear_batch)
            func, grad = calc_func_grad(Z, S, MTM, YM * S - adjacency_matrix @ Z @ Sigma_x_inv / 2)
            pbar.update(len(idx))
        pbar.close()
        func, grad = calc_func_grad(Z, S, MTM, YM * S - adjacency_matrix @ Z @ Sigma_x_inv / 2)
        return Z

    return update_z_gd_nesterov(Z)


def compute_loss_wnbr_closure(Z, S, MTM, YM, Ynorm, prior_x_mode, prior_x, Sigma_x_inv, adjacency_matrix, B):
    def compute_loss():
        XB = Z * S + B
        loss = ((XB @ MTM) * XB).sum() / 2 - (XB * YM).sum() + Ynorm / 2
        if prior_x_mode == "exponential shared fixed":
            loss += prior_x[0][0] * S.sum()
        elif not prior_x_mode:
            pass
        else:
            raise NotImplementedError

        if Sigma_x_inv is not None:
            loss += ((adjacency_matrix @ Z) @ Sigma_x_inv).mul(Z).sum() / 2
        loss = loss.item()
        # assert loss <= loss_prev, (loss_prev, loss)
        return loss

    return compute_loss()
