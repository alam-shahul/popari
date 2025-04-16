from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm, trange

from popari.util import IndependentSet, NesterovGD, project2simplex, project2simplex_

# def multiplicative_update_wonbr_closure(X_prev, MTM, YM, Ynorm, prior_x_mode, prior_x, loss_prev):
#     def multiplicative_update(X_prev):
#         """TODO:UNTESTED."""
#         X = torch.clip(X_prev, min=1e-10)
#         loss = ((X @ MTM) * X).sum() / 2 - X.view(-1) @ YM.view(-1) + Ynorm / 2
#         numerator = YM
#         denominator = X @ MTM
#         if prior_x_mode == "exponential shared fixed":
#             # see sklearn.decomposition.NMF
#             loss += (X @ prior_x[0]).sum()
#             denominator += prior_x[0][None]
#         else:
#             raise NotImplementedError
#
#         loss = loss.item()
#         assert loss <= loss_prev * (1 + 1e-4), (loss_prev, loss, (loss_prev - loss) / loss)
#         multiplicative_factor = numerator / denominator
#         X *= multiplicative_factor
#         torch.clip(X, min=1e-10)
#
#         return X, loss
#
#     return multiplicative_update(X_prev)


class EmbeddingLossNoNeighbors(nn.Module):
    def __init__(self, MTM, YM, Ynorm, prior_x_mode, prior_x, step_size):
        super().__init__()

        self.MTM = MTM
        self.YM = YM
        self.Ynorm = Ynorm
        self.prior_x_mode = prior_x_mode
        self.prior_x = prior_x
        self.step_size = step_size


class EmbeddingLossNoNeighborsGD(EmbeddingLossNoNeighbors):
    def forward(self, X):
        """TODO:UNTESTED."""
        quadratic_term_gradient = X @ self.MTM
        linear_term_gradient = self.YM
        if self.prior_x_mode == "exponential shared fixed":
            linear_term_gradient = linear_term_gradient - self.prior_x[0][None]
        elif not self.prior_x_mode:
            pass
        else:
            raise NotImplementedError

        loss = (quadratic_term_gradient * X).sum().item() / 2 - (linear_term_gradient * X).sum().item() + self.Ynorm / 2
        gradient = quadratic_term_gradient - linear_term_gradient
        X = X.sub(gradient, alpha=self.step_size)
        X = torch.clip(X, min=1e-10)

        return X, loss


class EmbeddingLossTripletLossNoNeighborsGD(EmbeddingLossNoNeighbors):
    def __init__(self, MTM, YM, Ynorm, prior_x_mode, prior_x, step_size, average_across_samples):
        super().__init__(MTM, YM, Ynorm, prior_x_mode, prior_x, step_size)

        self.average_across_samples = average_across_samples

    def forward(self, X):
        """TODO:UNTESTED."""
        quadratic_term_gradient = X @ self.MTM
        linear_term_gradient = self.YM
        if self.prior_x_mode == "exponential shared fixed":
            linear_term_gradient = linear_term_gradient - self.prior_x[0][None]
        elif self.prior_x_mode == "cross_dataset_average":
            sign_of_prior = torch.sign(X - self.average_across_samples)
            # print("sign of prior", linear_term_gradient.shape, sign_of_prior.shape)
            # linear_term_gradient = linear_term_gradient - (self.prior_x[0][None] * sign_of_prior)
            linear_term_gradient = linear_term_gradient - (
                (self.prior_x[0][None] * sign_of_prior) + self.prior_x[0][None]
            )
        elif not self.prior_x_mode:
            pass
        else:
            raise NotImplementedError

        loss = (quadratic_term_gradient * X).sum().item() / 2 - (linear_term_gradient * X).sum().item() + self.Ynorm / 2
        gradient = quadratic_term_gradient - linear_term_gradient
        X = X.sub(gradient, alpha=self.step_size)
        X = torch.clip(X, min=1e-10)

        return X, loss


class EmbeddingLossNoNeighborsMU(EmbeddingLossNoNeighbors):
    #  TODO: complete
    pass


class EmbeddingLossWithNeighbors(nn.Module, ABC):
    def __init__(
        self,
        Z,
        S,
        MTM,
        YM,
        Ynorm,
        adjacency_matrix,
        prior_x_mode,
        prior_x,
        Sigma_x_inv,
        E_adjacency_list,
        device,
        base_step_size,
        verbose,
        embedding_acceleration_trick,
        use_inplace_ops,
        embedding_mini_iterations,
        tol,
    ):
        super().__init__()

        self.Z = Z
        self.S = S
        self.MTM = MTM
        self.YM = YM
        self.Ynorm = Ynorm
        self.prior_x_mode = prior_x_mode
        self.prior_x = prior_x
        self.adjacency_matrix = adjacency_matrix
        self.Sigma_x_inv = Sigma_x_inv
        self.E_adjacency_list = E_adjacency_list
        self.device = device
        self.base_step_size = base_step_size
        self.verbose = verbose
        self.embedding_acceleration_trick = embedding_acceleration_trick
        self.use_inplace_ops = use_inplace_ops
        self.embedding_mini_iterations = embedding_mini_iterations
        self.tol = tol

        self.N = len(Z)

    def update_s(self):
        # S[:] = (YM * Z).sum(axis=1, keepdim=True)
        self.S[:] = (self.YM * self.Z).sum(
            axis=1,
            keepdim=True,
        )  #  TODO: there used to be a B multiplying self.MTM, add that back eventually

        if self.prior_x_mode == "exponential shared fixed":
            # TODO: why divide by two?
            self.S.sub_(self.prior_x[0][0] / 2)
        elif not self.prior_x_mode:
            pass
        else:
            raise NotImplementedError

        denominator = ((self.Z @ self.MTM) * self.Z).sum(axis=1, keepdim=True)
        self.S.div_(denominator)
        self.S.clip_(min=1e-5)

    def get_batch_loss_and_grad(self, Z_batch, S_batch, quad, linear):
        t = (Z_batch @ quad).mul_(S_batch**2)
        f = (t * Z_batch).sum() / 2
        g = t
        t = linear
        f -= (t * Z_batch).sum()
        g -= t
        g.sub_(g.sum(1, keepdim=True))

        return f.item(), g

    def compute_loss(self):
        X = self.Z * self.S  # TODO removed batch effect, add back in
        loss = ((X @ self.MTM) * X).sum() / 2 - (X * self.YM).sum() + self.Ynorm / 2
        if self.prior_x_mode == "exponential shared fixed":
            loss += self.prior_x[0][0] * self.S.sum()
        elif not self.prior_x_mode:
            pass
        else:
            raise NotImplementedError

        if self.Sigma_x_inv is not None:
            loss += ((self.adjacency_matrix @ self.Z) @ self.Sigma_x_inv).mul(self.Z).sum() / 2
        loss = loss.item()
        # assert loss <= loss_prev, (loss_prev, loss)

        return loss

    def forward(self):
        loss = np.inf
        pbar = trange(self.embedding_mini_iterations, disable=not self.verbose, desc="Updating weight w/ neighbors")

        for epoch in pbar:
            self.update_s()
            Z_prev = self.Z.clone().detach()
            # We may use Nesterov first and then vanilla GD in later iterations
            # update_z_mu(Z)
            # update_z_gd(Z)
            self.Z = self.update_z(self.Z)

            loss_prev = loss
            loss = self.compute_loss()
            dloss = loss_prev - loss
            dZ = (Z_prev - self.Z).abs().max().item()
            pbar.set_description(
                f"Updating weight w/ neighbors: loss = {loss:.1e} " f"δloss = {dloss:.1e} " f"δZ = {dZ:.1e}",
            )
            if dZ < self.tol:
                break

        X_final = self.Z * self.S
        return loss, X_final

    @abstractmethod
    def update_z(self, Z):
        pass


class EmbeddingLossWithNeighborsNesterov(EmbeddingLossWithNeighbors):
    def update_z(self, Z):
        pbar = trange(self.N, leave=False, disable=True, desc="Updating Z w/ nbrs via Nesterov GD")
        func, grad = self.get_batch_loss_and_grad(
            Z,
            self.S,
            self.MTM,
            self.YM * self.S - self.adjacency_matrix @ Z @ self.Sigma_x_inv / 2,
        )
        for idx in IndependentSet(self.E_adjacency_list, device=self.device, batch_size=1024):
            quad_batch = self.MTM
            linear_batch_spatial = -torch.index_select(self.adjacency_matrix, 0, idx) @ Z @ self.Sigma_x_inv
            Z_batch = Z[idx].contiguous()
            S_batch = self.S[idx].contiguous()
            optimizer = NesterovGD(Z_batch, self.base_step_size / S_batch.square())
            ppbar = trange(100, leave=False, disable=not (self.verbose > 3))
            for i_iter in ppbar:
                if self.embedding_acceleration_trick:
                    self.update_s()  # TODO: update S_batch directly
                S_batch = self.S[idx].contiguous()
                linear_batch = linear_batch_spatial + self.YM[idx] * S_batch
                if i_iter == 0:
                    func, grad = self.get_batch_loss_and_grad(
                        Z_batch,
                        S_batch,
                        quad_batch,
                        linear_batch,
                    )  # TODO: if we remove this line does it still run?
                    func, grad = self.get_batch_loss_and_grad(
                        Z,
                        self.S,
                        self.MTM,
                        self.YM * self.S - self.adjacency_matrix @ Z @ self.Sigma_x_inv / 2,
                    )

                NesterovGD.step_size = (
                    self.base_step_size / S_batch.square()
                )  # TM: I think this converges as s converges
                func, grad = self.get_batch_loss_and_grad(Z_batch, S_batch, quad_batch, linear_batch)

                Z_batch_prev = Z_batch.clone()
                Z_batch = optimizer.step(grad)

                if self.use_inplace_ops:
                    Z_batch = project2simplex_(Z_batch, dim=1)
                else:
                    Z_batch = project2simplex(Z_batch, dim=1)

                optimizer.set_parameters(Z_batch)

                dZ = (Z_batch_prev - Z_batch).abs().max().item()
                Z[idx] = Z_batch
                description = f"func={func:.1e}, dZ={dZ:.1e}"
                ppbar.set_description(description)
                if dZ < self.tol:
                    break
            ppbar.close()

            Z[idx] = Z_batch
            func, grad = self.get_batch_loss_and_grad(Z_batch, S_batch, quad_batch, linear_batch)
            func, grad = self.get_batch_loss_and_grad(
                Z,
                self.S,
                self.MTM,
                self.YM * self.S - self.adjacency_matrix @ Z @ self.Sigma_x_inv / 2,
            )
            pbar.update(len(idx))

        pbar.close()
        func, grad = self.get_batch_loss_and_grad(
            Z,
            self.S,
            self.MTM,
            self.YM * self.S - self.adjacency_matrix @ Z @ self.Sigma_x_inv / 2,
        )

        return Z


# def get_update_s_wnbr_closure(S, YM, MTM, prior_x, prior_x_mode, Z, B):
#     def update_s():
#         # S[:] = (YM * Z).sum(axis=1, keepdim=True)
#         S[:] = (YM * Z - ((Z @ MTM) * B)).sum(axis=1, keepdim=True)
#         if prior_x_mode == "exponential shared fixed":
#             # TODO: why divide by two?
#             S.sub_(prior_x[0][0] / 2)
#         elif not prior_x_mode:
#             pass
#         else:
#             raise NotImplementedError
#
#         denominator = ((Z @ MTM) * Z).sum(axis=1, keepdim=True)
#         S.div_(denominator)
#         S.clip_(min=1e-5)
#         return
#
#     return update_s()


class BatchEffectEmbeddingLossWithNeighborsNesterov(EmbeddingLossWithNeighborsNesterov):
    def __init__(
        self,
        Z,
        S,
        B,
        MTM,
        YM,
        Ynorm,
        adjacency_matrix,
        prior_x_mode,
        prior_x,
        Sigma_x_inv,
        E_adjacency_list,
        device,
        base_step_size,
        verbose,
        embedding_acceleration_trick,
        use_inplace_ops,
        embedding_mini_iterations,
        tol,
    ):
        super().__init__(
            Z,
            S,
            MTM,
            YM,
            Ynorm,
            adjacency_matrix,
            prior_x_mode,
            prior_x,
            Sigma_x_inv,
            E_adjacency_list,
            device,
            base_step_size,
            verbose,
            embedding_acceleration_trick,
            use_inplace_ops,
            embedding_mini_iterations,
            tol,
        )

        self.B = B

    def update_s(self):
        self.S[:] = (self.YM * self.Z - ((self.Z @ self.MTM) * self.B)).sum(axis=1, keepdim=True)

        if self.prior_x_mode == "exponential shared fixed":
            # TODO: why divide by two?
            self.S.sub_(self.prior_x[0][0] / 2)
        elif not self.prior_x_mode:
            pass
        else:
            raise NotImplementedError

        denominator = ((self.Z @ self.MTM) * self.Z).sum(axis=1, keepdim=True)
        self.S.div_(denominator)
        self.S.clip_(min=1e-5)

    def compute_loss(self):
        XB = self.Z * self.S + self.B
        loss = ((XB @ self.MTM) * XB).sum() / 2 - (XB * self.YM).sum() + self.Ynorm / 2
        if self.prior_x_mode == "exponential shared fixed":
            loss += self.prior_x[0][0] * self.S.sum()
        elif not self.prior_x_mode:
            pass
        else:
            raise NotImplementedError

        if self.Sigma_x_inv is not None:
            loss += ((self.adjacency_matrix @ self.Z) @ self.Sigma_x_inv).mul(self.Z).sum() / 2
        loss = loss.item()
        # assert loss <= loss_prev, (loss_prev, loss)

        return loss

    def update_z(self, Z):
        _, K = Z.size()

        pbar = trange(self.N, leave=False, disable=True, desc="Updating Z w/ nbrs via Nesterov GD")
        func, grad = self.get_batch_loss_and_grad(
            Z,
            self.S,
            self.MTM,
            self.YM * self.S - self.adjacency_matrix @ Z @ self.Sigma_x_inv / 2,
        )
        for idx in IndependentSet(self.E_adjacency_list, device=self.device, batch_size=1024):
            quad_batch = self.MTM
            linear_batch_spatial = -torch.index_select(self.adjacency_matrix, 0, idx) @ Z @ self.Sigma_x_inv
            Z_batch = Z[idx].contiguous()
            S_batch = self.S[idx].contiguous()
            optimizer = NesterovGD(Z_batch, self.base_step_size / S_batch.square())
            ppbar = trange(100, leave=False, disable=not (self.verbose > 3))
            for i_iter in ppbar:
                if self.embedding_acceleration_trick:
                    self.update_s()  # TODO: update S_batch directly
                S_batch = self.S[idx].contiguous()
                linear_batch = linear_batch_spatial + self.YM[idx] * S_batch
                if i_iter == 0:
                    func, grad = self.get_batch_loss_and_grad(
                        Z_batch,
                        S_batch,
                        quad_batch,
                        linear_batch,
                    )  # TODO: if we remove this line does it still run?
                    func, grad = self.get_batch_loss_and_grad(
                        Z,
                        self.S,
                        self.MTM,
                        self.YM * self.S - self.adjacency_matrix @ Z @ self.Sigma_x_inv / 2,
                    )

                NesterovGD.step_size = (
                    self.base_step_size / S_batch.square()
                )  # TM: I think this converges as s converges
                func, grad = self.get_batch_loss_and_grad(Z_batch, S_batch, quad_batch, linear_batch)

                Z_batch_prev = Z_batch.clone()
                Z_batch = optimizer.step(grad)

                if self.use_inplace_ops:
                    Z_batch = project2simplex_(Z_batch, dim=1)
                else:
                    Z_batch = project2simplex(Z_batch, dim=1)

                optimizer.set_parameters(Z_batch)

                dZ = (Z_batch_prev - Z_batch).abs().max().item()
                Z[idx] = Z_batch
                description = f"func={func:.1e}, dZ={dZ:.1e}"
                ppbar.set_description(description)
                if dZ < self.tol:
                    break
            ppbar.close()

            Z[idx] = Z_batch
            func, grad = self.get_batch_loss_and_grad(Z_batch, S_batch, quad_batch, linear_batch)
            func, grad = self.get_batch_loss_and_grad(
                Z,
                self.S,
                self.MTM,
                self.YM * self.S - self.adjacency_matrix @ Z @ self.Sigma_x_inv / 2,
            )
            pbar.update(len(idx))

        pbar.close()
        func, grad = self.get_batch_loss_and_grad(
            Z,
            self.S,
            self.MTM,
            self.YM * self.S - self.adjacency_matrix @ Z @ self.Sigma_x_inv / 2,
        )

        return Z


class BatchEffectTripletLossEmbeddingLossWithNeighborsNesterov(BatchEffectEmbeddingLossWithNeighborsNesterov):
    def __init__(
        self,
        Z,
        S,
        B,
        MTM,
        YM,
        Ynorm,
        adjacency_matrix,
        prior_x_mode,
        prior_x,
        average_across_samples,
        Sigma_x_inv,
        E_adjacency_list,
        device,
        base_step_size,
        verbose,
        embedding_acceleration_trick,
        use_inplace_ops,
        embedding_mini_iterations,
        tol,
    ):
        super().__init__(
            Z,
            S,
            B,
            MTM,
            YM,
            Ynorm,
            adjacency_matrix,
            prior_x_mode,
            prior_x,
            Sigma_x_inv,
            E_adjacency_list,
            device,
            base_step_size,
            verbose,
            embedding_acceleration_trick,
            use_inplace_ops,
            embedding_mini_iterations,
            tol,
        )
        self.average_across_samples = average_across_samples

    def update_s(self):
        # S[:] = (YM * Z).sum(axis=1, keepdim=True)
        self.S[:] = (self.YM * self.Z - ((self.Z @ self.MTM) * self.B)).sum(axis=1, keepdim=True)

        if self.prior_x_mode == "exponential shared fixed":
            # TODO: why divide by two?
            self.S.sub_(self.prior_x[0][0] / 2)
        elif self.prior_x_mode == "cross_dataset_average":
            # sign_of_prior = (torch.sign(self.S * self.Z - self.average_across_samples) * self.Z).sum(
            #     axis=1,
            #     keepdim=True,
            # )
            # # self.S.sub_(self.prior_x[0][0] * sign_of_prior / 2)
            # self.S.sub_((self.prior_x[0][0] * sign_of_prior + self.prior_x[0][0]) / 2)
            pass
        elif not self.prior_x_mode:
            pass
        else:
            raise NotImplementedError

        denominator = ((self.Z @ self.MTM) * self.Z).sum(axis=1, keepdim=True)
        self.S.div_(denominator)
        self.S.clip_(min=1e-5)

    def compute_loss(self):
        XB = self.Z * self.S + self.B
        loss = ((XB @ self.MTM) * XB).sum() / 2 - (XB * self.YM).sum() + self.Ynorm / 2
        if self.prior_x_mode == "exponential shared fixed":
            loss += self.prior_x[0][0] * self.S.sum()
        elif self.prior_x_mode == "cross_dataset_average":
            # loss += self.prior_x[0][0] * torch.abs(self.S - self.average_across_samples).sum()
            # loss += (
            #     self.prior_x[0][0] * torch.abs(self.S - self.average_across_samples).sum()
            #     + self.prior_x[0][0] * self.S.sum()
            # )
            pass
        elif not self.prior_x_mode:
            pass
        else:
            raise NotImplementedError

        if self.Sigma_x_inv is not None:
            loss += ((self.adjacency_matrix @ self.Z) @ self.Sigma_x_inv).mul(self.Z).sum() / 2
        loss = loss.item()
        # assert loss <= loss_prev, (loss_prev, loss)

        return loss
