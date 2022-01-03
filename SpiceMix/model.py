import sys, time, itertools, resource, logging, h5py, os
from pathlib import Path
import multiprocessing
from multiprocessing import Pool
from util import print_datetime, openH5File, encode4h5, parse_suffix

import numpy as np
import torch

from load_data import load_expression, load_edges, load_genelist
from initialization import initialize_kmeans, initialize_Sigma_x_inv, initialize_svd
from estimate_weights import estimate_weight_wonbr, estimate_weight_wnbr
from estimate_parameters import estimate_M, estimate_Sigma_x_inv


class SpiceMixPlus:
	"""SpiceMix optimization model.
	Provides state and functions to fit spatial transcriptomics data using the NMF-HMRF model. Can support multiple
	fields-of-view (FOVs).
	Attributes:
		device: device to use for PyTorch operations
		num_processes: number of parallel processes to use for optimizing weights (should be <= #FOVs)
		replicate_names: names of replicates/FOVs in input dataset
		TODO: finish docstring
	"""
	def __init__(
			self,
			K, lambda_Sigma_x_inv, repli_list, betas=None, prior_x_modes=None,
			path2result=None, context=None, context_Y=None,
	):
		if context is None: context = dict(device='cpu', dtype=torch.float32)
		if context_Y is None: context_Y = context
		self.context = context
		self.context_Y = context_Y
		self.repli_list = repli_list
		self.num_repli = len(self.repli_list)

		self.K = K
		self.lambda_Sigma_x_inv = lambda_Sigma_x_inv
		if betas is None: betas = np.full(self.num_repli, 1/self.num_repli)
		else: betas = np.array(betas, copy=False) / sum(betas)
		self.betas = betas
		if prior_x_modes is None: prior_x_modes = ['exponential shared fixed'] * self.num_repli
		self.prior_x_modes = prior_x_modes
		self.M_constraint = 'sum2one'
		self.X_constraint = 'none'
		self.dropout_mode = 'raw'
		self.sigma_yx_inv_mode = 'average'
		self.pairwise_potential_mode = 'normalized'

		if path2result is not None:
			self.path2result = path2result
			logging.info(f'result file = {self.result_filename}')
		else:
			self.result_filename = None
		# self.save_hyperparameters()

		self.Ys = self.Es = self.Es_isempty = self.genes = self.Ns = self.Gs = self.GG = None
		self.Sigma_x_inv = self.M = self.Xs = self.sigma_yxs = None
		self.optimizer_Sigma_x_inv = None
		self.prior_xs = None

	def load_dataset(self, path2dataset, neighbor_suffix=None, expression_suffix=None):
		path2dataset = Path(path2dataset)
		neighbor_suffix = parse_suffix(neighbor_suffix)
		expression_suffix = parse_suffix(expression_suffix)

		self.Ys = []
		for i in self.repli_list:
			for s in ['pkl', 'txt', 'tsv', 'pickle']:
				path2file = path2dataset / 'files' / f'expression_{i}{expression_suffix}.{s}'
				if not path2file.exists(): continue
				self.Ys.append(load_expression(path2file))
				break
		assert len(self.Ys) == len(self.repli_list)
		self.Ns, self.Gs = zip(*map(np.shape, self.Ys))
		self.GG = max(self.Gs)
		self.Es = [
			load_edges(path2dataset / 'files' / f'neighborhood_{i}{neighbor_suffix}.txt', N)
			for i, N in zip(self.repli_list, self.Ns)
		]
		self.Es_isempty = [sum(map(len, E)) == 0 for E in self.Es]
		self.genes = [
			load_genelist(path2dataset / 'files' / f'genes_{i}{expression_suffix}.txt')
			for i in self.repli_list
		]
		self.Ys = [G / self.GG * self.K * YT / YT.sum(1).mean() for YT, G in zip(self.Ys, self.Gs)]
		self.Ys = [torch.tensor(Y, **self.context_Y).pin_memory() for Y in self.Ys]
		# TODO: save Ys in cpu and calculate matrix multiplications (e.g., MT Y) using mini-batch.

	# def resume(self, iiter):

	def initialize(self, method='kmeans', random_state=0):
		if method == 'kmeans':
			self.M, self.Xs = initialize_kmeans(
				self.K, self.Ys,
				kwargs_kmeans=dict(random_state=random_state),
				context=self.context,
			)
		elif method == 'svd':
			self.M, self.Xs = initialize_svd(self.K, self.Ys, context=self.context)
		else:
			raise NotImplementedError
		scale_fac = self.M.sum(0, keepdim=True)
		self.M.div_(scale_fac)
		for X in self.Xs: X.mul_(scale_fac)
		del scale_fac
		if all(_ == 'exponential shared fixed' for _ in self.prior_x_modes):
			self.prior_xs = [(torch.ones(self.K, **self.context),) for _ in range(self.num_repli)]
		else:
			raise NotImplementedError
		self.estimate_sigma_yx()
		self.save_weights(iiter=0)
		self.save_parameters(iiter=0)

	def initialize_Sigma_x_inv(self):
		self.Sigma_x_inv = initialize_Sigma_x_inv(self.K, self.Xs, self.Es, self.betas, self.context)
		self.Sigma_x_inv.sub_(self.Sigma_x_inv.mean())
		self.Sigma_x_inv.requires_grad_(True)
		self.optimizer_Sigma_x_inv = torch.optim.Adam(
			[self.Sigma_x_inv],
			lr=1e-1,
			betas=(.5, .9),
		)
		# self.optimizer_Sigma_x_inv = torch.optim.SGD([self.Sigma_x_inv], lr=1e-2)
		# self.optimizer_Sigma_x_inv = None

	def estimate_sigma_yx(self):
		# d = np.array(np.linalg.norm(Y - X @ self.M.T, ord='fro') for Y, X in zip(self.Ys, self.Xs))
		d = np.array([
			torch.linalg.norm(torch.addmm(Y.to(X.device), X, self.M.T, alpha=-1), ord='fro').item() ** 2
			for Y, X in zip(self.Ys, self.Xs)
		])
		sizes = np.array([np.prod(Y.shape) for Y in self.Ys])
		if self.sigma_yx_inv_mode == 'separate':
			self.sigma_yxs = np.sqrt(d / sizes)
		elif self.sigma_yx_inv_mode == 'average':
			sigma_yx = np.sqrt(np.dot(self.betas, d) / np.dot(self.betas, sizes))
			self.sigma_yxs = np.full(self.num_repli, float(sigma_yx))
		else:
			raise NotImplementedError

	def estimate_weights(self, iiter, use_spatial):
		logging.info(f'{print_datetime()}Updating latent states')
		assert len(use_spatial) == self.num_repli

		assert self.X_constraint == 'none'
		assert self.pairwise_potential_mode == 'normalized'

		loss_list = []
		for i, (Y, X, sigma_yx, E, prior_x_mode, prior_x) in enumerate(zip(
				self.Ys, self.Xs, self.sigma_yxs, self.Es, self.prior_x_modes, self.prior_xs)):
			if self.Es_isempty[i] or not use_spatial[i]:
				loss = estimate_weight_wonbr(
					Y, self.M, X, sigma_yx, prior_x_mode, prior_x, context=self.context)
			else:
				loss = estimate_weight_wnbr(
					Y, self.M, X, sigma_yx, self.Sigma_x_inv, E, prior_x_mode, prior_x, context=self.context)
			loss_list.append(loss)

		self.save_weights(iiter=iiter)

		return loss_list

	def estimate_parameters(self, iiter, use_spatial, update_Sigma_x_inv=True):
		logging.info(f'{print_datetime()}Updating model parameters')

		if update_Sigma_x_inv:
			history = estimate_Sigma_x_inv(
				self.Xs, self.Sigma_x_inv, self.Es, use_spatial, self.lambda_Sigma_x_inv,
				self.betas, self.optimizer_Sigma_x_inv, self.context)
		else:
			history = []
		self.estimate_sigma_yx()
		estimate_M(self.Ys, self.Xs, self.M, self.betas, self.context)

		self.save_parameters(iiter=iiter)

		return history

	def skip_saving(self, iiter):
		return iiter % 10 != 0

	def save_hyperparameters(self):
		if self.result_filename is None: return

		with h5py.File(self.result_filename, 'w') as f:
			f['hyperparameters/repli_list'] = [_.encode('utf-8') for _ in self.repli_list]
			for k in ['prior_x_modes']:
				for repli, v in zip(self.repli_list, getattr(self, k)):
					f[f'hyperparameters/{k}/{repli}'] = encode4h5(v)
			for k in ['lambda_Sigma_x_inv', 'betas', 'K']:
				f[f'hyperparameters/{k}'] = encode4h5(getattr(self, k))

	def save_weights(self, iiter):
		if self.result_filename is None: return
		if self.skip_saving(iiter): return

		f = openH5File(self.result_filename)
		if f is None: return

		for repli, XT in zip(self.repli_list, self.Xs):
			f[f'latent_states/XT/{repli}/{iiter}'] = XT

		f.close()

	def save_parameters(self, iiter):
		if self.result_filename is None: return
		if self.skip_saving(iiter): return

		f = openH5File(self.result_filename)
		if f is None: return

		for k in ['M', 'Sigma_x_inv']:
			f[f'parameters/{k}/{iiter}'] = getattr(self, k)

		for k in ['sigma_yx_invs']:
			for repli, v in zip(self.repli_list, getattr(self, k)):
				f[f'parameters/{k}/{repli}/{iiter}'] = v

		for k in ['prior_xs']:
			for repli, v in zip(self.repli_list, getattr(self, k)):
				f[f'parameters/{k}/{repli}/{iiter}'] = np.array(v[1:])

		f.close()

	def save_progress(self, iiter):
		if self.result_filename is None: return

		f = openH5File(self.result_filename)
		if f is None: return

		f[f'progress/Q/{iiter}'] = self.Q

		f.close()
