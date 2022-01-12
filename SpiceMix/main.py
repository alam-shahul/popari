import os, sys, time, itertools, resource, gc, argparse, re, logging
from tqdm.auto import tqdm, trange
from pathlib import Path

import numpy as np, pandas as pd
import torch

from model import SpiceMixPlus
from util import clustering_louvain_nclust, evaluate_embedding
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA, NMF, TruncatedSVD
from sklearn.metrics import silhouette_score, adjusted_rand_score

from umap import UMAP


def parse_arguments():
	parser = argparse.ArgumentParser()

	# dataset
	parser.add_argument(
		'--path2dataset', type=Path,
		help='name of the dataset, ../data/<dataset> should be a folder containing a subfolder named \'files\''
	)
	parser.add_argument(
		'-K', type=int, default=20,
		help='Number of metagenes'
	)
	parser.add_argument(
		'--neighbor_suffix', type=str, default='',
		help='Suffix of the name of the file that contains interacting cell pairs'
	)
	parser.add_argument(
		'--expression_suffix', type=str, default='',
		help='Suffix of the name of the file that contains expressions'
	)
	parser.add_argument(
		'--repli_list', type=lambda x: list(map(str, eval(x))),
		help='list of names of the experiments, a Python expression, e.g., "[0,1,2]", "range(5)"'
	)
	parser.add_argument(
		'--use_spatial', type=eval,
		help='list of true/false indicating whether to use the spatial information in each experiment, '
			 'a Python expression, e.g., "[True,True]", "[False,False,False]", "[True]*5"'
	)

	parser.add_argument('--random_seed', type=int, default=None)
	# parser.add_argument('--random_seed4kmeans', type=int, default=0)

	# training & hyperparameters
	parser.add_argument('--lambda_Sigma_x_inv', type=float, default=1e-4, help='Regularization on Σx^{-1}')
	parser.add_argument('--max_iter', type=int, default=500, help='Maximum number of outer optimization iteration')
	parser.add_argument('--init_NMF_iter', type=int, default=10, help='2 * number of NMF iterations in initialization')
	parser.add_argument(
		'--betas', default=np.ones(1), type=np.array,
		help='Positive weights of the experiments; the sum will be normalized to 1; can be scalar (equal weight) or array-like'
	)
	parser.add_argument('--lambda_x', type=float, default=1., help='Prior of X')

	def parse_cuda(x):
		if x == '-1' or x == 'cpu': return 'cpu'
		if re.match('\d+$', x): return f'cuda:{x}'
		if re.match('cuda:\d+$', x): return x
	parser.add_argument(
		'--device', type=parse_cuda, default='cpu',
		help="Which GPU to use. The value should be either string of form 'cuda:<GPU id>' "
			 "or an integer denoting the GPU id. -1 or 'cpu' for cpu only",
	)
	parser.add_argument('--num_threads', type=int, default=1, help='Number of CPU threads for PyTorch')
	parser.add_argument('--num_processes', type=int, default=1, help='Number of processes')

	parser.add_argument(
		'--result_filename', default=None, help='The name of the h5 file to store results'
	)

	return parser.parse_args()


if __name__ == '__main__':
	np.set_printoptions(linewidth=100000)

	args = parse_arguments()

	# logging.basicConfig(level=logging.INFO)
	logging.basicConfig(level=logging.WARNING)

	logging.info(f'pid = {os.getpid()}')

	# for the on-the-fly short-eval
	# df_meta = []
	# for r in args.repli_list:
	# 	try:
	# 		df = pd.read_csv(args.path2dataset / 'files' / f'meta_{r}.csv')
	# 	except:
	# 		df = pd.read_csv(args.path2dataset / 'files' / f'celltypes_{r}.txt', header=None)
	# 		df.columns = ['cell type']
	# 	df['repli'] = r
	# 	df_meta.append(df)
	# df_meta = pd.concat(df_meta, axis=0).reset_index(drop=True)
	# df_meta['cell type'] = pd.Categorical(df_meta['cell type'], categories=np.unique(df_meta['cell type']))

	if args.random_seed is not None:
		np.random.seed(args.random_seed)
		logging.info(f'random seed = {args.random_seed}')

	torch.set_num_threads(args.num_threads)

	betas = np.broadcast_to(args.betas, [len(args.repli_list)]).copy().astype(float)
	assert (betas > 0).all()
	betas /= betas.sum()

	context = dict(device=args.device, dtype=torch.float32)

	obj = SpiceMixPlus(
		K=args.K, lambda_Sigma_x_inv=args.lambda_Sigma_x_inv,
		repli_list=args.repli_list,
		context=context,
		# context_Y=dict(dtype=torch.float32, device='cpu'),
		context_Y=context,
	)
	obj.load_dataset(args.path2dataset)
	obj.initialize(
		method='kmeans',
		# method='svd',
		random_state=args.random_seed,
	)

	for iiter in range(args.init_NMF_iter):
		obj.estimate_weights(iiter=iiter, use_spatial=[False] * obj.num_repli)
		obj.estimate_parameters(iiter=iiter, use_spatial=[False] * obj.num_repli)
	# short_eval(obj, df_meta)
	evaluate_embedding(obj, embedding='X', do_plot=False, do_sil=False)
	obj.initialize_Sigma_x_inv()
	for iiter in range(1, args.max_iter+1):
		obj.estimate_parameters(iiter=iiter, use_spatial=args.use_spatial)
		obj.estimate_weights(iiter=iiter, use_spatial=args.use_spatial)
		if iiter % 10 == 0 or iiter == args.max_iter:
			# short_eval(obj, df_meta)
			evaluate_embedding(obj, embedding='X', do_plot=False, do_sil=False)
