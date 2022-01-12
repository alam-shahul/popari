import sys, time, itertools, resource, logging, h5py, os, pickle
from pathlib import Path
import multiprocessing
from multiprocessing import Pool
from util import print_datetime, openH5File, encode4h5, parse_suffix

import numpy as np, pandas as pd
import torch

from load_data import load_expression, load_edges, load_genelist
from initialization import initialize_kmeans, initialize_Sigma_x_inv, initialize_svd
from estimate_weights import estimate_weight_wonbr, estimate_weight_wnbr, \
	estimate_weight_wnbr_phenotype, estimate_weight_wnbr_phenotype_v2
from estimate_parameters import estimate_M, estimate_Sigma_x_inv, estimate_phenotype_predictor


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
		self.M_constraint = 'simplex'
		# self.M_constraint = 'unit sphere'
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

		self.Ys = self.meta = self.Es = self.Es_isempty = self.genes = self.Ns = self.Gs = self.GG = None
		self.Sigma_x_inv = self.Xs = self.sigma_yxs = None
		self.M = None
		self.Ms = self.M_base = None
		self.lambda_M = 0
		self.Zs = self.Ss = self.Z_optimizers = None
		self.optimizer_Sigma_x_inv = None
		self.prior_xs = None
		self.phenotypes = self.phenotype_predictors = None
		self.meta_repli = None

	def load_dataset(self, path2dataset, neighbor_suffix=None, expression_suffix=None):
		path2dataset = Path(path2dataset)
		neighbor_suffix = parse_suffix(neighbor_suffix)
		expression_suffix = parse_suffix(expression_suffix)

		self.Ys = []
		for r in self.repli_list:
			for s in ['pkl', 'txt', 'tsv', 'pickle']:
				path2file = path2dataset / 'files' / f'expression_{r}{expression_suffix}.{s}'
				if not path2file.exists(): continue
				self.Ys.append(load_expression(path2file))
				break
		assert len(self.Ys) == len(self.repli_list)
		self.Ns, self.Gs = zip(*map(np.shape, self.Ys))
		self.GG = max(self.Gs)
		self.genes =  [
			load_genelist(path2dataset / 'files' / f'genes_{r}{expression_suffix}.txt')
			for r in self.repli_list
		]
		self.Ys = [G / self.GG * self.K * Y / Y.sum(1).mean() for Y, G in zip(self.Ys, self.Gs)]
		self.Ys = [
			torch.tensor(Y, **self.context_Y).pin_memory()
			if self.context_Y.get('device', 'cpu') == 'cpu' else
			torch.tensor(Y, **self.context_Y)
			for Y in self.Ys
		]
		# TODO: save Ys in cpu and calculate matrix multiplications (e.g., MT Y) using mini-batch.

		self.Es = [
			load_edges(path2dataset / 'files' / f'neighborhood_{i}{neighbor_suffix}.txt', N)
			for i, N in zip(self.repli_list, self.Ns)
		]
		self.Es_isempty = [sum(map(len, E)) == 0 for E in self.Es]

		df_all = []
		for r in self.repli_list:
			path2file = path2dataset / 'files' / f'meta_{r}.pkl'
			if os.path.exists(path2file):
				with open(path2file, 'rb') as f: df_all.append(pickle.load(f))
				continue
			path2file = path2dataset / 'files' / f'meta_{r}.csv'
			if os.path.exists(path2file):
				df_all.append(pd.read_csv(path2file))
				continue
			path2file = path2dataset / 'files' / f'celltypes_{r}.txt'
			if os.path.exists(path2file):
				df = pd.read_csv(path2file, header=None)
				df.columns = ['cell type']
				# df['repli'] = r
				df_all.append(df)
				continue
			raise FileNotFoundError(r)
		assert len(df_all) == len(self.repli_list)
		for r, df in zip(self.repli_list, df_all):
			df['repli'] = r
		df_all = pd.concat(df_all)
		self.meta = df_all

		self.phenotypes = [{} for i in range(self.num_repli)]
		self.phenotype_predictors = None

	def register_phenotype(self, phenotype2predictor):
		"""
		:param phenotype2predictor: a dictionary. Each key is a column name in self.meta. The corresponding value is a
		3-element tuple that contains:
			1) a predictor, an instance of torch.nn.Module
			2) an optimizer
			3) a loss function
		:return:
		"""
		self.phenotype_predictors = phenotype2predictor
		for predictor, optimizer, loss_fn in phenotype2predictor.values():
			for param in predictor.parameters(): param.requires_grad_(False)
		self.phenotypes = [None] * len(self.repli_list)
		for repli, df in self.meta.groupby('repli'):
			phenotype = {}
			for key in phenotype2predictor.keys():
				p = torch.tensor(df[key].values, device=self.context.get('device', 'cpu'))
				phenotype[key] = p
			self.phenotypes[self.repli_list.index(repli)] = phenotype

	def register_meta_repli(self, df_meta_repli):
		self.meta_repli = df_meta_repli

	# def resume(self, iiter):

	def get_M(self, repli):
		key = 'metagene group'
		if self.meta_repli is None or key not in self.meta_repli.columns:
			key = 'shared'
		else:
			key = self.meta_repli.loc[repli, key]
		if self.Ms is None: return key
		else: return self.Ms[key]

	def initialize(self, method='kmeans', random_state=0):
		# if method == 'kmeans':
		# 	self.M, self.Xs = initialize_kmeans(
		# 		self.K, self.Ys,
		# 		kwargs_kmeans=dict(random_state=random_state),
		# 		context=self.context,
		# 	)
		# elif method == 'svd':
		# 	self.M, self.Xs = initialize_svd(self.K, self.Ys, context=self.context)
		# else:
		# 	raise NotImplementedError

		if self.phenotype_predictors is not None:
			key = next(iter(self.phenotype_predictors.keys()))
			set_of_labels = np.unique(sum([p[key].cpu().numpy().tolist() for p in self.phenotypes if p[key] is not None], []))
			L = len(set_of_labels)
			M = torch.zeros([self.GG, L], **self.context)
			n = torch.zeros([L], **self.context)
			Xs = [torch.zeros([N, L], **self.context) for N in self.Ns]
			is_valid = lambda Y, phenotype: Y.shape[1] == M.shape[0] and phenotype[key] is not None
			for phenotype, Y, X in zip(self.phenotypes, self.Ys, Xs):
				if not is_valid(Y, phenotype): continue
				for i, c in enumerate(set_of_labels):
					mask = phenotype[key] == c
					M[:, i] += Y[mask].sum(0)
					n[i] += mask.sum()
					X[mask, i] = 1
			M.div_(n)

			Ys = []
			for phenotype, Y in zip(self.phenotypes, self.Ys):
				if not is_valid(Y, phenotype): continue
				Y = Y.clone()
				for i, c in enumerate(set_of_labels):
					mask = phenotype[key] == c
					Y[mask] -= M[:, i]
				Ys.append(Y)
			M_res, Xs_res = initialize_svd(self.K - L, Ys, context=self.context)
			self.M = torch.cat([M, M_res], dim=1)
			self.Xs = []
			Xs_res_iter = iter(Xs_res)
			for phenotype, Y, X in zip(self.phenotypes, self.Ys, Xs):
				if is_valid(Y, phenotype):
					self.Xs.append(torch.cat([X, next(Xs_res_iter)], dim=1))
				else:
					self.Xs.append(torch.linalg.lstsq(self.M, Y.T)[0].clip_(min=0).T.contiguous())
		elif method == 'kmeans':
			self.M, self.Xs = initialize_kmeans(
				self.K, self.Ys,
				kwargs_kmeans=dict(random_state=random_state),
				context=self.context,
			)
		elif method == 'svd':
			self.M, self.Xs = initialize_svd(
				self.K, self.Ys, context=self.context,
				M_nonneg=self.M_constraint == 'simplex', X_nonneg=True,
			)
		else:
			raise NotImplementedError

		if self.M_constraint == 'simplex':
			scale_fac = torch.linalg.norm(self.M, axis=0, ord=1, keepdim=True)
		elif self.M_constraint == 'unit sphere':
			scale_fac = torch.linalg.norm(self.M, axis=0, ord=2, keepdim=True)
		else:
			raise NotImplementedError
		self.M.div_(scale_fac)
		for X in self.Xs: X.mul_(scale_fac)
		del scale_fac
		keys = set(self.get_M(r) for r in self.repli_list)
		self.M_base = self.M
		self.Ms = {key: self.M_base.clone() for key in keys}

		if all(_ == 'exponential shared fixed' for _ in self.prior_x_modes):
			self.prior_xs = [(torch.ones(self.K, **self.context),) for _ in range(self.num_repli)]
		else:
			raise NotImplementedError

		self.estimate_sigma_yx()
		self.estimate_phenotype_predictor()
		#
		# self.Zs = []
		# self.Ss = []
		# for X in self.Xs:
		# 	S = torch.linalg.norm(X, ord=1, dim=1, keepdim=True)
		# 	self.Ss.append(S)
		# 	self.Zs.append(X / S)
		# self.Z_optimizers = [
		# 	torch.optim.Adam(
		# 		[Z],
		# 		lr=1e-3,
		# 		betas=(.3, .9),
		# 	)
		# 	for Z in self.Zs
		# ]

		self.save_weights(iiter=0)
		self.save_parameters(iiter=0)

	def initialize_Sigma_x_inv(self):
		self.Sigma_x_inv = initialize_Sigma_x_inv(self.K, self.Xs, self.Es, self.betas, self.context)
		self.Sigma_x_inv.sub_(self.Sigma_x_inv.mean())
		# self.Sigma_x_inv.requires_grad_(True)
		self.optimizer_Sigma_x_inv = torch.optim.Adam(
			[self.Sigma_x_inv],
			lr=1e-1,
			betas=(.5, .9),
		)
		# self.optimizer_Sigma_x_inv = torch.optim.SGD([self.Sigma_x_inv], lr=1e-2)
		# self.optimizer_Sigma_x_inv = None

	def estimate_sigma_yx(self):
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
		for i, (Y, X, sigma_yx, E, phenotype, prior_x_mode, prior_x) in enumerate(zip(
				self.Ys, self.Xs, self.sigma_yxs, self.Es, self.phenotypes, self.prior_x_modes, self.prior_xs)):
			valid_keys = [k for k, v in phenotype.items() if v is not None]
			if len(valid_keys) > 0:
				loss = estimate_weight_wnbr_phenotype(
					Y, self.M, X, sigma_yx, self.Sigma_x_inv, E, prior_x_mode, prior_x,
					{k: phenotype[k] for k in valid_keys}, {k: self.phenotype_predictors[k] for k in valid_keys},
					context=self.context,
				)

				# S = self.Ss[i]
				# Z = self.Zs[i]
				# S[:] = torch.linalg.norm(X, ord=1, dim=1, keepdim=True)
				# Z[:] = X / S
				#
				# loss = estimate_weight_wnbr_phenotype_v2(
				# 	Y, self.M, Z, S, sigma_yx, self.Sigma_x_inv, E, prior_x_mode, prior_x,
				# 	self.Z_optimizers[i],
				# 	{k: phenotype[k] for k in valid_keys}, {k: self.phenotype_predictors[k] for k in valid_keys},
				# 	context=self.context,
				# )
				#
				# X[:] = Z * S
			elif self.Es_isempty[i] or not use_spatial[i]:
				loss = estimate_weight_wonbr(
					Y, self.M, X, sigma_yx, prior_x_mode, prior_x, context=self.context)
				# loss = estimate_weight_wnbr(
				# 	Y, self.M, X, sigma_yx, self.Sigma_x_inv, E, prior_x_mode, prior_x, context=self.context)

			else:
				loss = estimate_weight_wnbr(
					Y, self.M, X, sigma_yx, self.Sigma_x_inv, E, prior_x_mode, prior_x, context=self.context)

				# S = self.Ss[i]
				# Z = self.Zs[i]
				# S[:] = torch.linalg.norm(X, ord=1, dim=1, keepdim=True)
				# Z[:] = X / S
				#
				# loss = estimate_weight_wnbr_phenotype_v2(
				# 	Y, self.M, Z, S, sigma_yx, self.Sigma_x_inv, E, prior_x_mode, prior_x,
				# 	self.Z_optimizers[i],
				# 	{k: phenotype[k] for k in valid_keys}, {k: self.phenotype_predictors[k] for k in valid_keys},
				# 	context=self.context,
				# )
				#
				# X[:] = Z * S
			loss_list.append(loss)

		self.save_weights(iiter=iiter)

		return loss_list

	def estimate_phenotype_predictor(self):
		if self.phenotype_predictors is None: return
		for phenotype_name, predictor_tuple in self.phenotype_predictors.items():
			input_target_list = []
			for X, phenotype in zip(self.Xs, self.phenotypes):
				if phenotype[phenotype_name] is None: continue
				X = X / torch.linalg.norm(X, ord=1, dim=1, keepdim=True)
				input_target_list.append((X, phenotype[phenotype_name]))
			estimate_phenotype_predictor(*zip(*input_target_list), phenotype_name, *predictor_tuple)

	def estimate_M(self):
		for group, M in self.Ms.items():
			is_valid = [self.get_M(repli) is M for repli in self.repli_list]
			Ys, Xs, sigma_yxs, betas = zip(*itertools.compress(zip(
				self.Ys, self.Xs, self.sigma_yxs, self.betas), is_valid))
			betas = np.array(betas)
			betas /= sum(betas) # not sure if we should do this
			estimate_M(
				# self.Ys, self.Xs, self.M, self.sigma_yxs, self.betas,
				Ys, Xs, M, np.array(sigma_yxs), betas,
				M_base=self.M_base, lambda_M=self.lambda_M,
				M_constraint=self.M_constraint, context=self.context,
			)

		self.M_base.zero_()
		for group, M in self.Ms.items():
			self.M_base.add_(M)
		self.M_base.div_(len(self.Ms))

	def estimate_parameters(self, iiter, use_spatial, update_Sigma_x_inv=True):
		logging.info(f'{print_datetime()}Updating model parameters')

		if update_Sigma_x_inv:
			history = estimate_Sigma_x_inv(
				self.Xs, self.Sigma_x_inv, self.Es, use_spatial, self.lambda_Sigma_x_inv,
				self.betas, self.optimizer_Sigma_x_inv, self.context)
		else:
			history = []
		self.estimate_M()
		self.estimate_sigma_yx()
		self.estimate_phenotype_predictor()

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
