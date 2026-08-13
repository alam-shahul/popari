import time

import numpy as np
import torch
from loguru import logger
from tqdm.auto import tqdm, trange

from popari.optim.batching import IndependentSet
from popari.optim.nesterov import NesterovGD
from popari.optim.projection import project2simplex, project2simplex_


def _embedding_loss(
    embedding,
    quadratic_factor,
    linear_factor,
    constant,
    prior_x_mode,
    prior_x,
    *,
    neighbor_embeddings=None,
    spatial_affinity=None,
):
    """Return the conditional embedding objective as a scalar tensor."""

    loss = ((embedding @ quadratic_factor) * embedding).sum() / 2
    loss -= (embedding * linear_factor).sum()
    loss += constant / 2

    magnitude = torch.linalg.norm(embedding, dim=1, ord=1, keepdim=True)
    if prior_x_mode == "exponential shared fixed":
        loss += prior_x[0][0] * magnitude.sum()
    elif prior_x_mode is not None:
        raise NotImplementedError(f"Unsupported embedding prior: {prior_x_mode!r}")

    if neighbor_embeddings is not None:
        if spatial_affinity is None:
            raise ValueError("spatial_affinity is required with neighbor_embeddings.")
        normalized = embedding / magnitude
        loss += (neighbor_embeddings @ spatial_affinity).mul(normalized).sum() / 2

    return loss


@torch.no_grad()
def estimate_weight_wonbr(
    level,
    Y,
    M,
    X,
    sigma_yx,
    prior_x_mode,
    prior_x,
    n_epochs=1000,
    tol=1e-6,
    update_alg="gd",
):
    """Estimate weights without spatial information - equivalent to vanilla NMF.

    Optimizes the follwing objective with respect to hidden state X:

    min 1/2σ^2 || Y - X MT ||_2^2 + lam || X ||_1
    grad = X MT M / σ^2 - Y MT / σ^2 + lam

    TODO: use (projected) Nesterov GD. not urgent

    Args:
        Y (torch.Tensor):
    """

    # Precomputing quantities
    MTM = M.T @ M / (sigma_yx**2)
    YM = Y @ M / (sigma_yx**2)
    Ynorm = torch.square(Y).sum() / (sigma_yx**2)
    step_size = 1 / torch.linalg.eigvalsh(MTM).max().item()
    loss_prev, loss = np.inf, np.nan

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
        loss = _embedding_loss(X, MTM, YM, Ynorm, prior_x_mode, prior_x).item()
        gradient = quadratic_term_gradient - linear_term_gradient
        X = X.sub(gradient, alpha=step_size)
        X = torch.clip(X, min=1e-10)

        return X, loss

    progress_bar = trange(
        n_epochs,
        desc="Embedding optimization",
        leave=False,
        disable=level.verbose < 2,
        dynamic_ncols=True,
        mininterval=1,
    )
    for epoch in progress_bar:
        X_prev = X.clone()
        if update_alg == "mu":
            # TODO: it seems like loss might not always decrease...
            X.clip_(min=1e-10)
            loss = _embedding_loss(X, MTM, YM, Ynorm, prior_x_mode, prior_x)
            numerator = YM
            denominator = X @ MTM
            if prior_x_mode == "exponential shared fixed":
                # see sklearn.decomposition.NMF
                denominator.add_(prior_x[0][None])
            elif not prior_x_mode:
                pass
            else:
                raise NotImplementedError

            loss = loss.item()
            assert loss <= loss_prev * (1 + 1e-4), (loss_prev, loss, (loss_prev - loss) / loss)
            multiplicative_factor = numerator / denominator
            X.mul_(multiplicative_factor).clip_(min=1e-10)

            # X, loss = multiplicative_update(X_prev)
        elif update_alg == "gd":
            X, loss = gradient_update(X)
        else:
            raise NotImplementedError

        dX = torch.abs((X_prev - X) / torch.linalg.norm(X, dim=1, ord=1, keepdim=True)).max().item()
        do_stop = dX < tol
        progress_bar.set_description(
            f"Updating weight w/o nbrs: loss = {loss:.1e} "
            f"%δloss = {(loss_prev - loss) / loss:.1e} "
            f"%δX = {dX:.1e}",
        )
        loss_prev = loss
        if do_stop:
            break
    progress_bar.close()
    return loss, X


@torch.no_grad()
def estimate_weight_wnbr(
    level,
    Y,
    M,
    X,
    sigma_yx,
    prior_x_mode,
    prior_x,
    sample,
    global_Z=None,
    tol=1e-5,
    update_alg="nesterov",
):
    """Estimate updated weights taking neighbor-neighbor interactions into
    account.

    The optimization for all variables
    min 1/2σ^2 || Y - diag(S) Z MT ||_2^2 + lam || S ||_1 + sum_{ij in E} ziT Σx-1 zj

    for s_i
    min 1/2σ^2 || y - M z s ||_2^2 + lam s
    s* = max(0, ( yT M z / σ^2 - lam ) / ( zT MT M z / σ^2) )

    for Z
    min 1/2σ^2 || Y - diag(S) Z MT ||_2^2 + sum_{ij in E} ziT Σx-1 zj
    grad_i = MT M z s^2 / σ^2 - MT y s / σ^2 + sum_{j in Ei} Σx-1 zj

    TODO: Try projected Newton's method.
    TM: Inverse is precomputed once, and projection is cheap. Not sure if it works theoretically

    """
    # Precomputing quantities
    MTM = M.T @ M / (sigma_yx**2)
    YM = Y.to(M.device) @ M / (sigma_yx**2)
    Ynorm = torch.square(Y).sum() / (sigma_yx**2)
    base_step_size = level.embedding_step_size_multiplier / torch.linalg.eigvalsh(MTM).max().item()
    S = torch.linalg.norm(X, dim=1, ord=1, keepdim=True)

    if level.verbose >= 3:
        logger.debug("Embedding magnitude range: {:.3e} to {:.3e}", S.min().item(), S.max().item())

    Z = X / S
    N = len(Z)

    sample_indices = level.sample_indices[sample]
    sample_indices_numpy = level.sample_axis.indices(sample)
    sample_adjacency = level.adjacency[sample_indices_numpy][:, sample_indices_numpy]
    if global_Z is None:
        global_embedding = level.embeddings
        global_Z = global_embedding / torch.linalg.norm(global_embedding, dim=1, ord=1, keepdim=True)
    global_Z = global_Z.clone()
    global_Z.index_copy_(0, sample_indices, Z)
    adjacency_matrix = level.adjacency_matrix
    Sigma_x_inv = level.spatial_affinity.for_sample(sample).to(level.context["device"])

    def update_s():
        S[:] = (YM * Z).sum(axis=1, keepdim=True)
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

    def calc_func_grad(Z_batch, S_batch, quad, linear):
        t = (Z_batch @ quad).mul_(S_batch**2)
        f = (t * Z_batch).sum() / 2
        g = t
        t = linear
        f -= (t * Z_batch).sum()
        g -= t
        g.sub_(g.sum(1, keepdim=True))

        return f.item(), g

    def update_z_gd(Z):
        step_size = base_step_size / S.square()
        pbar = tqdm(range(N), leave=False, disable=True)
        for idx in IndependentSet(
            sample_adjacency,
            device=level.context["device"],
            batch_size=128,
        ):
            global_idx = sample_indices.index_select(0, idx)
            step_size_scale = 1
            quad_batch = MTM
            linear_batch = (
                YM[idx] * S[idx] - torch.index_select(adjacency_matrix, 0, global_idx) @ global_Z @ Sigma_x_inv
            )
            Z_batch = Z[idx].contiguous()
            S_batch = S[idx].contiguous()
            step_size_batch = step_size[idx].contiguous()
            func, grad = calc_func_grad(Z_batch, S_batch, quad_batch, linear_batch)
            while True:
                Z_batch_new = Z_batch - step_size_batch * step_size_scale * grad
                if level.use_inplace_ops:
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
            global_Z.index_copy_(0, global_idx, Z_batch)
            pbar.set_description(f"Updating Z w/ nbrs via line search: lr={step_size_scale:.1e}")
            pbar.update(len(idx))
        pbar.close()

        return Z

    def update_z_gd_nesterov(Z):
        pbar = trange(N, leave=False, disable=True, desc="Updating Z w/ nbrs via Nesterov GD")

        neighbor_Z = torch.index_select(adjacency_matrix, 0, sample_indices) @ global_Z
        func, grad = calc_func_grad(Z, S, MTM, YM * S - neighbor_Z @ Sigma_x_inv / 2)
        for idx in IndependentSet(
            sample_adjacency,
            device=level.context["device"],
            batch_size=1024,
        ):
            global_idx = sample_indices.index_select(0, idx)
            quad_batch = MTM
            linear_batch_spatial = -torch.index_select(adjacency_matrix, 0, global_idx) @ global_Z @ Sigma_x_inv
            Z_batch = Z[idx].contiguous()
            S_batch = S[idx].contiguous()

            optimizer = NesterovGD(Z_batch, base_step_size / S_batch.square())
            ppbar = trange(100, leave=False, disable=level.verbose < 3, dynamic_ncols=True, mininterval=1)
            for i_iter in ppbar:
                if level.embedding_acceleration_trick:
                    update_s()  # TODO: update S_batch directly
                S_batch = S[idx].contiguous()
                linear_batch = linear_batch_spatial + YM[idx] * S_batch
                if i_iter == 0:
                    func, grad = calc_func_grad(Z_batch, S_batch, quad_batch, linear_batch)
                    neighbor_Z = torch.index_select(adjacency_matrix, 0, sample_indices) @ global_Z
                    func, grad = calc_func_grad(Z, S, MTM, YM * S - neighbor_Z @ Sigma_x_inv / 2)
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

                if level.use_inplace_ops:
                    Z_batch = project2simplex_(Z_batch, dim=1)
                else:
                    Z_batch = project2simplex(Z_batch, dim=1)

                optimizer.set_parameters(Z_batch)

                dZ = (Z_batch_prev - Z_batch).abs().max().item()
                Z[idx] = Z_batch
                global_Z.index_copy_(0, global_idx, Z_batch)
                description = f"func={func:.1e}, dZ={dZ:.1e}"
                ppbar.set_description(description)
                if dZ < tol:
                    break
            ppbar.close()

            Z[idx] = Z_batch
            global_Z.index_copy_(0, global_idx, Z_batch)
            func, grad = calc_func_grad(Z_batch, S_batch, quad_batch, linear_batch)
            neighbor_Z = torch.index_select(adjacency_matrix, 0, sample_indices) @ global_Z
            func, grad = calc_func_grad(Z, S, MTM, YM * S - neighbor_Z @ Sigma_x_inv / 2)
            pbar.update(len(idx))
        pbar.close()
        neighbor_Z = torch.index_select(adjacency_matrix, 0, sample_indices) @ global_Z
        func, grad = calc_func_grad(Z, S, MTM, YM * S - neighbor_Z @ Sigma_x_inv / 2)

        return Z

    def compute_loss():
        X = Z * S
        neighbor_Z = torch.index_select(adjacency_matrix, 0, sample_indices) @ global_Z
        return _embedding_loss(
            X,
            MTM,
            YM,
            Ynorm,
            prior_x_mode,
            prior_x,
            neighbor_embeddings=neighbor_Z,
            spatial_affinity=Sigma_x_inv,
        ).item()

    # TM: consider combine compute_loss and update_z to remove a call to torch.sparse.mm
    # TM: the above idea is not practical if we update only a subset of nodes each time

    loss = np.inf
    pbar = trange(
        level.embedding_mini_iterations,
        desc="Spatial embedding optimization",
        leave=False,
        disable=level.verbose < 2,
        dynamic_ncols=True,
        mininterval=1,
    )

    for epoch in pbar:
        update_s()
        Z_prev = Z.clone().detach()
        # We may use Nesterov first and then vanilla GD in later iterations
        # update_z_mu(Z)
        # update_z_gd(Z)
        if update_alg == "gd":
            Z = update_z_gd(Z)
        elif update_alg == "nesterov":
            Z = update_z_gd_nesterov(Z)

        loss_prev = loss
        loss = compute_loss()
        dloss = loss_prev - loss
        dZ = (Z_prev - Z).abs().max().item()
        pbar.set_description(
            f"Updating weight w/ neighbors: loss = {loss:.1e} " f"δloss = {dloss:.1e} " f"δZ = {dZ:.1e}",
        )
        if dZ < tol:
            break

    X_final = Z * S
    return loss, X_final
