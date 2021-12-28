import sys, logging, time, gc, os
from tqdm.auto import tqdm, trange

import numpy as np
import torch

from sample_for_integral import project2simplex


@torch.no_grad()
def estimate_weight_wonbr(
		Y, M, X, sigma_yx, prior_x_mode, prior_x, context, n_epochs=100, tol=1e-5, update_alg='mu'):
	"""
	min 1/2σ^2 || Y - X MT ||_2^2 + lam || X ||_1
	grad = X MT M / σ^2 - Y MT / σ^2 + lam
	"""
	MTM = M.T @ M / (sigma_yx ** 2)
	YM = Y.to(M.device) @ M / (sigma_yx ** 2)
	Ynorm = torch.linalg.norm(Y, ord='fro').item() ** 2 / (sigma_yx ** 2)
	loss_prev, loss = np.inf, np.nan
	pbar = trange(n_epochs, leave=False, disable=True)
	for i_epoch in pbar:
		if update_alg == 'mu':
			loss = ((X @ MTM) * X).sum() / 2 - X.view(-1) @ YM.view(-1) + Ynorm / 2
			numerator = YM
			denominator = X @ MTM
			if prior_x_mode == 'exponential shared fixed':
				# see sklearn.decomposition.NMF
				loss += (X @ prior_x[0]).sum()
				denominator.add_(prior_x[0][None])
			else:
				raise NotImplementedError

			loss = loss.item()
			assert loss <= loss_prev * (1 + 1e-4), (loss_prev, loss, (loss_prev - loss) / loss)
			mul_fac = numerator / denominator
			pbar.set_description(
				f'Updating weight w/o neighbors: loss = {loss:.1e}, '
				f'%δloss = {(loss_prev - loss) / loss:.1e}, '
				f'update = {mul_fac.min().item():.1e} ~ {mul_fac.max().item():.1e}'
			)

			loss_prev = loss
			if mul_fac.sub(1).abs().max() < tol: break
			X.mul_(mul_fac).clip_(min=1e-10)
		else:
			raise NotImplementedError
	pbar.close()
	return loss


@torch.no_grad()
def estimate_weight_wnbr(
		Y, M, X, sigma_yx, Sigma_x_inv, E, prior_x_mode, prior_x, context, n_epochs=100, tol=1e-4, update_alg='gd',
):
	"""
	The optimization for all variables
	min 1/2σ^2 || Y - diag(S) Z MT ||_2^2 + lam || S ||_1 + sum_{ij in E} ziT Σx-1 zj

	for s_i
	min 1/2σ^2 || y - M z s ||_2^2 + lam s
	s* = max(0, ( yT M z / σ^2 - lam ) / ( zT MT M z / σ^2) )

	for Z
	min 1/2σ^2 || Y - diag(S) Z MT ||_2^2 + sum_{ij in E} ziT Σx-1 zj
	grad_i = MT M z s^2 / σ^2 - MT y s / σ^2 + sum_{j in Ei} Σx-1 zj
	"""
	MTM = M.T @ M / (sigma_yx**2)
	YM = Y.to(M.device) @ M / (sigma_yx ** 2)
	Ynorm = torch.linalg.norm(Y, ord='fro').item() ** 2 / (sigma_yx**2)
	step_size_base = 1 / torch.linalg.eigvalsh(MTM).max()
	S = torch.linalg.norm(X, dim=1, ord=1, keepdim=True)
	Z = X / S
	N = len(Z)

	E_adj_list = np.array(E, dtype=object)

	def get_adj_mat(adj_list):
		edges = [(i, j) for i, e in enumerate(adj_list) for j in e]
		adj_mat = torch.sparse_coo_tensor(np.array(edges).T, np.ones(len(edges)), size=[len(adj_list), N], **context)
		return adj_mat

	E_adj_mat = get_adj_mat(E_adj_list)

	def update_s():
		S[:] = (YM * Z).sum(1, keepdim=True)
		if prior_x_mode == 'exponential shared fixed':
			S.sub_(prior_x[0][0])
		else:
			raise NotImplementedError
		S.div_(((Z @ MTM) * Z).sum(1, keepdim=True))
		S.clip_(min=1e-5)

	def update_z_mu(Z):
		indices = np.random.permutation(len(S))
		# bs = len(indices) // 50
		bs = 1
		thr = 1e3
		for i in range(0, len(indices), bs):
			idx = indices[i: i+bs]
			adj_mat = get_adj_mat(E_adj_list[idx])
			numerator = YM[idx] * S[idx]
			denominator = (Z[idx] @ MTM).mul_(S[idx] ** 2)
			t = (adj_mat @ Z) @ Sigma_x_inv
			# not sure if this works
			numerator -= t.clip(max=0)
			denominator += t.clip(min=0)
			mul_fac = numerator / denominator
			mul_fac.clip_(min=1/thr, max=thr)
			# Z.mul_(mul_fac)
			Z[idx] *= mul_fac
			project2simplex(Z, dim=1)

	def update_z_gd(Z):
		def calc_func_grad(Z, idx):
			# grad = (Z @ MTM).mul_(S ** 2)
			# grad.addcmul_(YM, S, value=-1)
			# grad.addmm_(E @ Z, Sigma_x_inv)
			# grad.sub_(grad.sum(1, keepdim=True))
			t = (Z[idx] @ MTM).mul_(S[idx] ** 2)
			f = (t * Z[idx]).sum().item() / 2
			g = t
			t = YM[idx] * S[idx]
			f -= (t * Z[idx]).sum().item()
			g -= t
			t = get_adj_mat(E_adj_list[idx]) @ Z @ Sigma_x_inv
			f += (t * Z[idx]).sum().item()
			g += t
			# f += Ynorm / 2
			return f, g
		step_size = step_size_base / S.square()
		bs = 50
		indices_remaining = set(range(N))
		# indices = np.random.permutation(len(S))
		# pbar = trange(0, N, bs, leave=False)
		pbar = tqdm(range(N), leave=False)
		# for i in pbar:
		while len(indices_remaining) > 0:
			step_size_scale = 1
			# idx = indices[i: i+bs]
			indices_candidates = np.random.choice(list(indices_remaining), size=min(bs, len(indices_remaining)), replace=False)
			indices = []
			indices_exclude = set()
			for i in indices_candidates:
				if i in indices_exclude:
					continue
				else:
					indices.append(i)
					indices_exclude |= set(E_adj_list[i])
			indices_remaining -= set(indices)
			idx = list(indices)
			func, grad = calc_func_grad(Z, idx)
			while True:
				# Z_new = Z.sub(grad, alpha=step_size * step_size_scale)
				Z_new = Z.clone()
				Z_new[idx] -= (step_size[idx] * step_size_scale) * grad
				Z_new[idx] = project2simplex(Z_new[idx], dim=1)
				dZ = Z_new[idx].sub(Z[idx]).abs().max().item()
				func_new, grad_new = calc_func_grad(Z_new, idx)
				# print(
				# 	func, func_new, func - func_new,
				# 	step_size_scale,
				# 	(grad * step_size * step_size_scale).abs().max().item(),
				# 	dZ,
				# )
				if func_new < func:
					Z[:] = Z_new
					func = func_new
					grad = grad_new
					step_size_scale *= 2
					continue
				else:
					step_size_scale *= .5
				if dZ < 1e-4 or step_size_scale < .5: break
			assert step_size_scale > .1
			pbar.set_description(f'Updating Z w/ nbrs via line search: lr={step_size_scale:.1e}')
			pbar.update(len(idx))
		pbar.close()

	def calc_loss(loss_prev):
		X = Z * S
		loss = ((X @ MTM) * X).sum() / 2 - (X * YM).sum() + Ynorm / 2
		if prior_x_mode == 'exponential shared fixed':
			loss += prior_x[0][0] * S.sum()
		else:
			raise NotImplementedError
		loss += ((E_adj_mat @ Z) @ Sigma_x_inv).mul(Z).sum()
		loss = loss.item()
		# assert loss <= loss_prev, (loss_prev, loss)
		return loss, loss_prev - loss

	# consider combine calc_loss and update_z to remove a call to torch.sparse.mm

	loss = np.inf
	pbar = trange(n_epochs)
	Z_prev = Z.clone().detach()
	for i_epoch in pbar:
		update_s()
		# update_z_mu(Z)
		update_z_gd(Z)
		loss, dloss = calc_loss(loss)
		dZ = (Z_prev - Z).abs().max().item()
		pbar.set_description(
			f'Updating weight w/ neighbors: loss = {loss:.1e}, '
			f'δloss = {dloss:.1e}, '
			f'δZ = {dZ:.1e}'
		)
		if dZ < tol: break
		Z_prev[:] = Z

	X[:] = Z * S
	return loss
