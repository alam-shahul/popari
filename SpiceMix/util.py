import os, time, pickle, sys, datetime, h5py, logging
from collections import Iterable
from tqdm.auto import tqdm, trange

import numpy as np, pandas as pd
from anndata import AnnData
import scanpy as sc


def calc_modularity(A, label, resolution=1):
	A = A.tocoo()
	n = A.shape[0]
	Asum = A.data.sum()
	score = A.data[label[A.row] == label[A.col]].sum() / Asum

	idx = np.argsort(label)
	label = label[idx]
	k = np.array(A.sum(0)).ravel() / Asum
	k = k[idx]
	idx = np.concatenate([[0], np.nonzero(label[:-1] != label[1:])[0] + 1, [len(label)]])
	score -= sum(k[i:j].sum() ** 2 for i, j in zip(idx[:-1], idx[1:])) * resolution
	return score


def clustering_louvain(X, *, kwargs_neighbors, kwargs_clustering, num_rs=100, method='louvain'):
	adata = AnnData(X)
	sc.pp.neighbors(adata, use_rep='X', **kwargs_neighbors)
	best = {'score': np.nan}
	resolution = kwargs_clustering.get('resolution', 1)
	pbar = trange(num_rs, desc=f'Louvain clustering: res={resolution:.2e}')
	for rs in pbar:
		getattr(sc.tl, method)(adata, **kwargs_clustering, random_state=rs)
		cluster = np.array(list(adata.obs[method]))
		score = calc_modularity(
			adata.obsp['connectivities'], cluster, resolution=resolution)
		if not best['score'] >= score: best.update({'score': score, 'cluster': cluster.copy(), 'rs': rs})
		pbar.set_description(
			f'Louvain clustering: res={resolution:.2e}; '
			f"best: score = {best['score']} rs = {best['rs']} # of clusters = {len(set(best['cluster']))}"
		)
	y = best['cluster']
	y = pd.Categorical(y, categories=np.unique(y))
	return y


def clustering_louvain_nclust(
		X, n_clust_target, *, kwargs_neighbors, kwargs_clustering,
		resolution_boundaries=None,
		resolution_init=1, resolution_update=2,
		num_rs=100, method='louvain',
):
	adata = AnnData(X)
	sc.pp.neighbors(adata, use_rep='X', **kwargs_neighbors)
	kwargs_clustering = kwargs_clustering.copy()
	y = None

	def do_clustering(res):
		y = clustering_louvain(
			X,
			kwargs_neighbors=kwargs_neighbors,
			kwargs_clustering=dict(**kwargs_clustering, **dict(resolution=res)),
			method=method,
			num_rs=num_rs,
		)
		n_clust = len(set(y))
		return y, n_clust

	lb = rb = None
	if resolution_boundaries is not None:
		lb, rb = resolution_boundaries
	else:
		res = resolution_init
		y, n_clust = do_clustering(res)
		if n_clust > n_clust_target:
			while n_clust > n_clust_target and res > 1e-2:
				rb = res
				res /= resolution_update
				y, n_clust = do_clustering(res)
			lb = res
		elif n_clust < n_clust_target:
			while n_clust < n_clust_target:
				lb = res
				res *= resolution_update
				y, n_clust = do_clustering(res)
			rb = res
		if n_clust == n_clust_target: lb = rb = res

	while rb - lb > .01:
		mid = (lb * rb) ** .5
		y = clustering_louvain(
			X,
			kwargs_neighbors=kwargs_neighbors,
			kwargs_clustering=dict(**kwargs_clustering, **dict(resolution=mid)),
			method=method,
			num_rs=num_rs,
		)
		n_clust = len(set(y))
		print(
			f'binary search for resolution: lb={lb:.2f}\trb={rb:.2f}\tmid={mid:.2f}\tn_clust={n_clust}',
			# '{:.2f}'.format(adjusted_rand_score(obj.data['cell type'], obj.data['cluster'])),
			sep='\t',
		)
		if n_clust == n_clust_target: break
		if n_clust > n_clust_target: rb = mid
		else: lb = mid
	return y


def print_datetime():
	return datetime.datetime.now().strftime('[%Y/%m/%d %H:%M:%S]\t')


def array2string(x):
	return np.array2string(x, formatter={'all': '{:.2e}'.format})


def parse_suffix(s):
	return '' if s is None or s == '' else '_' + s


def openH5File(filename, mode='a', num_attempts=5, duration=1):
	for i in range(num_attempts):
		try:
			return h5py.File(filename, mode=mode)
		except OSError as e:
			logging.warning(str(e))
			time.sleep(duration)
	return None


def encode4h5(v):
	if isinstance(v, str): return v.encode('utf-8')
	return v


def parseIiter(g, iiter):
	if iiter < 0: iiter += max(map(int, g.keys())) + 1
	return iiter


def a2i(a, order=None, ignores=()):
	if order is None:
		order = np.array(list(set(a) - set(ignores)))
	else:
		order = order[~np.isin(order, list(ignores))]
	d = dict(zip(order, range(len(order))))
	for k in ignores: d[k] = -1
	a = np.fromiter(map(d.get, a), dtype=int)
	return a, d, order


def zipTensors(*tensors):
	return np.concatenate([
		np.array(a).flatten()
		for a in tensors
	])


def unzipTensors(arr, shapes):
	assert np.all(arr.shape == (np.sum(list(map(np.prod, shapes))),))
	tensors = []
	for shape in shapes:
		size = np.prod(shape)
		tensors.append(arr[:size].reshape(*shape).squeeze())
		arr = arr[size:]
	return tensors


# def getRank(m):
# 	rank = np.empty(m.shape, dtype=int)
# 	for r, a in zip(rank, m):
# 		r[np.argsort(a)] = np.arange(len(r))
# 	return rank


def getRank(m, thr=0):
	rank = np.empty(m.shape, dtype=int)
	for r, a in zip(rank, m):
		r[np.argsort(a)] = np.arange(len(r))
		mask = a < thr
		r[mask] = np.mean(r[mask])
	return rank
