import sys, logging, time, gc, os, itertools
from multiprocessing import Pool
from util import print_datetime
import torch

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import TruncatedSVD


def initialize_kmeans(K, Ys, kwargs_kmeans, context):
	assert 'random_state' in kwargs_kmeans
	Ns, Gs = zip(*[Y.shape for Y in Ys])
	GG = max(Gs)
	repli_valid = np.array(Gs) == GG
	Ys = [Y.cpu().numpy() for Y in Ys]
	Y_cat = np.concatenate(list(itertools.compress(Ys, repli_valid)), axis=0)
	kmeans = KMeans(n_clusters=K, **kwargs_kmeans).fit(Y_cat)
	M = kmeans.cluster_centers_.T
	Xs = []
	for is_valid, N, Y in zip(repli_valid, Ns, Ys):
		if is_valid:
			label = kmeans.predict(Y)
			X = np.full([N, K], 1e-10)
			X[(range(N), label)] = 1
		else:
			X = np.full([N, K], 1/K)
		Xs.append(X)
	M = torch.tensor(M, **context)
	Xs = [torch.tensor(X, **context) for X in Xs]
	return M, Xs


def initialize_svd(K, Ys, context):
	Ns, Gs = zip(*[Y.shape for Y in Ys])
	GG = max(Gs)
	repli_valid = np.array(Gs) == GG
	Ys = [Y.cpu().numpy() for Y in Ys]
	Y_cat = np.concatenate(list(itertools.compress(Ys, repli_valid)), axis=0)
	svd = TruncatedSVD(K).fit(Y_cat)
	M = svd.components_.T
	norm_p = np.linalg.norm(np.clip(M, a_min=0, a_max=None), axis=0, ord=1, keepdims=True)
	norm_n = np.linalg.norm(np.clip(M, a_min=None, a_max=0), axis=0, ord=1, keepdims=True)
	sign = np.where(norm_p > norm_n, 1., -1.)
	M = np.clip(M * sign, a_min=1e-10, a_max=None)
	Xs = []
	for is_valid, N, Y in zip(repli_valid, Ns, Ys):
		if is_valid:
			X = svd.transform(Y)
			X = np.clip(X * sign, a_min=1e-10, a_max=None)
		else:
			X = np.full([N, K], 1/K)
		Xs.append(X)
	M = torch.tensor(M, **context)
	Xs = [torch.tensor(X, **context) for X in Xs]
	return M, Xs


def initialize_Sigma_x_inv(K, Xs, Es, betas, context):
	Sigma_x_inv = torch.zeros([K, K], **context)
	for X, E, beta in zip(Xs, Es, betas):
		Z = X / torch.linalg.norm(X, dim=1, keepdim=True, ord=1)
		E = np.array([(i, j) for i, e in enumerate(E) for j in e])
		# E = torch.sparse_coo_tensor(E.T, np.ones(len(E)), size=[len(X)]*2, **context)
		# cov = (E @ X).T @ X
		# var = X.std(0) # this is not the variance
		x = Z[E[:, 0]]
		y = Z[E[:, 1]]
		x = x - x.mean(0, keepdim=True)
		y = y - y.mean(0, keepdim=True)
		corr = (y / y.std(0, keepdim=True)).T @ (x / x.std(0, keepdim=True)) / len(x)
		Sigma_x_inv.add_(corr, alpha=-beta)
	Sigma_x_inv = (Sigma_x_inv + Sigma_x_inv.T) / 2
	Sigma_x_inv -= Sigma_x_inv.mean()
	Sigma_x_inv *= 10
	return Sigma_x_inv
