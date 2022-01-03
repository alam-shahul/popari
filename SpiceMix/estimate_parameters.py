import sys, time, itertools, logging
from multiprocessing import Pool, Process
from tqdm.auto import tqdm, trange

import torch
import numpy as np
from scipy.special import loggamma

from sample_for_integral import integrate_of_exponential_over_simplex, project2simplex


@torch.no_grad()
def estimate_M(Ys, Xs, M, betas, context, n_epochs=100, tol=1e-5, update_alg='gd'):
	"""
	min || Y - X MT ||_2^2 / 2
	s.t. || Mk ||_1 = 1
	grad = M XT X - YT X
	"""
	K = M.shape[1]
	XTX = torch.zeros([K, K], **context)
	YTX = torch.zeros_like(M)
	Ynorm = np.dot([torch.linalg.norm(Y, ord='fro') ** 2 for Y in Ys], betas)
	for Y, X, beta in zip(Ys, Xs, betas):
		XTX.addmm_(X.T, X, alpha=beta)
		YTX.addmm_(Y.T.to(X.device), X, alpha=beta)

	loss_prev, loss = np.inf, np.nan
	pbar = trange(n_epochs, leave=True, disable=False, desc='Updating M')
	M_prev = M.clone().detach()
	step_size = 1 / torch.linalg.eigvalsh(XTX).max().item()
	for i_epoch in pbar:
		loss = ((M @ XTX) * M).sum() / 2 - (M * YTX).sum() + Ynorm / 2
		loss = loss.item()
		assert loss <= loss_prev * (1 + 1e-4), (loss_prev, loss)
		M_prev[:] = M

		if update_alg == 'mu':
			numerator = YTX
			denominator = M @ XTX
			mul_fac = numerator / denominator
			M.mul_(mul_fac)
			mul_fac.clip_(max=10)
		elif update_alg == 'gd':
			grad = torch.addmm(YTX, M, XTX, alpha=1, beta=-1)
			grad.sub_(grad.sum(0, keepdim=True))
			M.add_(grad, alpha=-step_size)
		else:
			raise NotImplementedError
		project2simplex(M, dim=0)
		loss_prev = loss
		dM = M_prev.sub(M).abs_().max().item()
		pbar.set_description(
			f'Updating M: loss = {loss:.1e}, '
			f'%δloss = {(loss_prev - loss) / loss:.1e}, '
			f'δM = {dM:.1e}'
		)
		if dM < tol and i_epoch > 5: break
	pbar.close()
	return loss


def estimate_Sigma_x_inv(Xs, Sigma_x_inv, Es, use_spatial, lambda_Sigma_x_inv, betas, optimizer, context, n_epochs=10000):
	if not any(sum(map(len, E)) > 0 and u for E, u in zip(Es, use_spatial)): return
	linear_term = torch.zeros_like(Sigma_x_inv).requires_grad_(False)
	# Zs = [torch.tensor(X / np.linalg.norm(X, axis=1, ord=1, keepdims=True), **context) for X in Xs]
	Zs = [X / torch.linalg.norm(X, axis=1, ord=1, keepdim=True) for X in Xs]
	nus = [] # sum of neighbors' z
	num_edges = 0
	for Z, E, u, beta in zip(Zs, Es, use_spatial, betas):
		E = [(i, j) for i, e in enumerate(E) for j in e]
		E = torch.sparse_coo_tensor(np.array(E).T, np.ones(len(E)), size=[len(Z)]*2, **context)
		if u:
			nu = E @ Z
			linear_term.addmm_(Z.T, nu, alpha=beta)
		else:
			nu = None
		nus.append(nu)
		num_edges += beta * sum(map(len, E))
		del Z, E
	linear_term = (linear_term + linear_term.T) / 2

	assumption_str = 'mean-field'

	history = []

	if optimizer is not None:
		loss_prev, loss = np.inf, np.nan
		pbar = trange(n_epochs, desc='Updating Σx-1')
		Sigma_x_inv_best, loss_best, i_epoch_best = None, np.inf, -1
		dSigma_x_inv = np.inf
		early_stop_iepoch = 0
		for i_epoch in pbar:
			optimizer.zero_grad()

			loss = Sigma_x_inv.view(-1) @ linear_term.view(-1)
			loss = loss + lambda_Sigma_x_inv * Sigma_x_inv.pow(2).sum() * num_edges / 2
			for Z, nu, beta in zip(Zs, nus, betas):
				if nu is None: continue
				eta = nu @ Sigma_x_inv
				logZ = integrate_of_exponential_over_simplex(eta)
				loss = loss + beta * logZ.sum()
			loss = loss / num_edges
			# if i_epoch == 0: print(loss.item())

			if loss < loss_best:
				Sigma_x_inv_best = Sigma_x_inv.clone().detach()
				loss_best = loss.item()
				i_epoch_best = i_epoch

			history.append((Sigma_x_inv.detach().cpu().numpy(), loss.item()))

			with torch.no_grad():
				Sigma_x_inv_prev = Sigma_x_inv.clone().detach()
			loss.backward()
			Sigma_x_inv.grad = (Sigma_x_inv.grad + Sigma_x_inv.grad.T) / 2
			optimizer.step()
			with torch.no_grad():
				Sigma_x_inv -= Sigma_x_inv.mean()

			loss = loss.item()
			dloss = loss_prev - loss
			loss_prev = loss

			with torch.no_grad():
				dSigma_x_inv = Sigma_x_inv_prev.sub(Sigma_x_inv).abs().max().item()
			pbar.set_description(
				f'Updating Σx-1: loss = {dloss:.1e} -> {loss:.1e} '
				f'δΣx-1 = {dSigma_x_inv:.1e} '
				f'Σx-1 range = {Sigma_x_inv.min().item():.1e} ~ {Sigma_x_inv.max().item():.1e}'
			)

			if Sigma_x_inv.grad.abs().max() < 1e-4 and dSigma_x_inv < 1e-4:
				early_stop_iepoch += 1
			else:
				early_stop_iepoch = 0
			if early_stop_iepoch >= 10 or i_epoch > i_epoch_best + 100:
				break

		with torch.no_grad():
			Sigma_x_inv[:] = Sigma_x_inv_best
	else:
		Sigma_x_inv_storage = Sigma_x_inv

		def calc_func_grad(Sigma_x_inv):
			if Sigma_x_inv.grad is None: Sigma_x_inv.grad = torch.zeros_like(Sigma_x_inv)
			else: Sigma_x_inv.grad.zero_()
			loss = Sigma_x_inv.view(-1) @ linear_term.view(-1)
			loss = loss + lambda_Sigma_x_inv * Sigma_x_inv.pow(2).sum() * num_edges / 2
			for Z, nu, beta in zip(Zs, nus, betas):
				if nu is None: continue
				eta = nu @ Sigma_x_inv
				logZ = integrate_of_exponential_over_simplex(eta)
				loss = loss + beta * logZ.sum()
			return loss

		pbar = trange(n_epochs)
		step_size = 1e-1
		step_size_update = 2
		dloss = np.inf
		loss = calc_func_grad(Sigma_x_inv)
		loss.backward()
		loss = loss.item()
		for i_epoch in pbar:
			with torch.no_grad():
				Sigma_x_inv_new = Sigma_x_inv.add(Sigma_x_inv.grad, alpha=-step_size).requires_grad_(True)
				Sigma_x_inv_new.sub_(Sigma_x_inv_new.mean())
			loss_new = calc_func_grad(Sigma_x_inv_new)
			with torch.no_grad():
				if loss_new.item() < loss:
					loss_new.backward()
					loss_new = loss_new.item()
					dloss = loss - loss_new
					loss = loss_new
					dSigma_x_inv = Sigma_x_inv.sub(Sigma_x_inv_new).abs().max().item()
					Sigma_x_inv = Sigma_x_inv_new
					step_size *= step_size_update
				else:
					step_size /= step_size_update
				pbar.set_description(
					f'Updating Σx-1: loss = {dloss:.1e} -> {loss:.1e}, '
					f'δΣx-1 = {dSigma_x_inv:.1e} '
					f'lr = {step_size:.1e} '
					f'Σx-1 range = {Sigma_x_inv.min().item():.1e} ~ {Sigma_x_inv.max().item():.1e}'
				)
				if (Sigma_x_inv.grad * step_size).abs().max() < 1e-3:
					dloss = np.nan
					dSigma_x_inv = np.nan
					break

		with torch.no_grad():
			Sigma_x_inv_storage[:] = Sigma_x_inv

	return history
