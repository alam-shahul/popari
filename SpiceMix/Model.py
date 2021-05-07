import sys, time, itertools, psutil, resource, logging, h5py, os
from pathlib import Path
import multiprocessing
from multiprocessing import Pool
from util import print_datetime, parseSuffix, openH5File, encode4h5

import numpy as np
import gurobipy as grb
import torch

from load_data import load_expression, load_edges
from initialization import initialize_M_by_kmeans, initialize_sigma_x_inverse, partial_nmf
from estimateWeights import estimateWeightsICM, estimateWeightsWithoutNeighbor
from estimateParameters import estimateParametersX, estimateParametersY

class Model:
    def __init__(self, path2dataset, repli_list, use_spatial, neighbor_suffix, expression_suffix, K,
                 lambda_SigmaXInv, betas, prior_x_modes, result_filename=None, PyTorch_device='cpu', num_processes=1):

        self.PyTorch_device = PyTorch_device
        self.num_processes = num_processes

        self.path2dataset = Path(path2dataset)
        self.repli_list = repli_list
        self.use_spatial = use_spatial
        self.num_repli = len(self.repli_list)
        assert len(self.repli_list) == len(self.use_spatial)
        self.load_dataset(neighbor_suffix=neighbor_suffix, expression_suffix=expression_suffix)

        self.K = K
        self.YTs = [G / self.max_genes * self.K * YT / YT.sum(1).mean() for YT, G in zip(self.YTs, self.Gs)]
        self.lambda_SigmaXInv = lambda_SigmaXInv
        self.betas = betas
        self.prior_x_modes = prior_x_modes
        self.M_constraint = 'sum2one'
        self.X_constraint = 'none'
        self.dropout_mode = 'raw'
        self.sigma_yx_inverse_mode = 'average'
        self.pairwise_potential_mode = 'normalized'

        if result_filename:
            os.makedirs(self.path2dataset / 'results', exist_ok=True)
            self.result_filename = self.path2dataset / 'results' / result_filename
            logging.info(f'{print_datetime()}result file = {self.result_filename}')

        self.saveHyperparameters()

    # def __del__(self):
    #   pass
        # if self.result_h5 is not None:
        #   self.result_h5.close()

    def load_dataset(self, neighbor_suffix=None, expression_suffix=None):
        """Load spatial transcriptomics data from relevant filepaths.

        This function is called during the initialization of any Model object, and it
        sets important attributes that will be used during the SpiceMix optimization.

        Args:
            neighbor_suffix: pattern to match at end of neighborhood filename.
            expression_suffix: pattern to match at end of expression data filename.
        """

        neighbor_suffix = parseSuffix(neighbor_suffix)
        expression_suffix = parseSuffix(expression_suffix)

        self.YTs = []
        for replicate in self.repli_list:
            # TODO: is it necessary to allow multiple extensions, or can we require that the expression data are
            # in .txt files?
            for extension in ['txt', 'pkl', 'pickle']:
                filepath = self.path2dataset / 'files' / f'expression_{replicate}{expression_suffix}.{extension}'
                if not filepath.exists():
                    continue

                gene_expression = load_expression(filepath)
                self.YTs.append(gene_expression)

        self.Ns, self.Gs = zip(*map(np.shape, self.YTs))
        self.max_genes = max(self.Gs)

        # TODO: change to use dictionary with empty mappings instead of list of empty lists
        self.Es = [
            load_edges(self.path2dataset / 'files' / f'neighborhood_{replicate}{neighbor_suffix}.txt', num_nodes)
            if u else [[] for _ in range(num_nodes)]
            for replicate, num_nodes, u in zip(self.repli_list, self.Ns, self.use_spatial)
        ]

        self.Es_empty = [sum(map(len, E)) == 0 for E in self.Es]

    def initialize(self, random_seed4kmeans, initial_nmf_iterations=5, sigma_x_inverse_mode='Constant'):
        logging.info(f'{print_datetime()}Initialization begins')
        
        # initialize M
        self.M = initialize_M_by_kmeans(self.YTs, self.K, random_seed4kmeans=random_seed4kmeans)
        if self.M_constraint == 'sum2one':
            self.M = np.maximum(self.M, 0)
            self.M /= self.M.sum(0, keepdims=True)
        else:
            raise NotImplementedError(f'Constraint on M {self.M_constraint} is not implemented')
        logging.info(f'{print_datetime()}Initialized M with shape {self.M.shape}')
    
        # initialize XT and perhaps update M
        # sigma_yx is estimated from XT and M
        self.M, self.XTs, self.sigma_yx_inverses, self.prior_xs = partial_nmf(self, prior_x_modes=self.prior_x_modes, initial_nmf_iterations=initial_nmf_iterations)
    
        if all(self.Es_empty): 
            sigma_x_inverse_mode = 'Constant'
    
        self.sigma_x_inverse = initialize_sigma_x_inverse(self.K, self.XTs, self.Es, self.betas, sigma_x_inverse_mode=sigma_x_inverse_mode)
        # logging.info(f'{print_datetime()}sigma_x_inverse_mode = {sigma_x_inverse_mode}')
        # if sigma_x_inverse_mode == 'Constant':
        #     self.sigma_x_inverse = np.zeros([self.K, self.K])
        # elif sigma_x_inverse_mode.startswith('Identity'):
        #     factor = float(sigma_x_inverse_mode.split()[1])
        #     self.sigma_x_inverse = np.eye(self.K) * factor
        # elif sigma_x_inverse_mode.startswith('EmpiricalFromX'):
        #     factor = float(sigma_x_inverse_mode.split()[1])
        #     sigma_x = np.zeros([self.K, self.K])
        #     for XT, adjacency_list, beta in zip(self.XTs, self.Es, self.betas):
        #         t = np.zeros_like(sigma_x)
        #         for XTi, neighbors in zip(XT, adjacency_list):
        #             t += np.outer(XTi, XT[neighbors].sum(0))
        #         self.sigma_x += t * beta
        #     sigma_x /= np.dot(self.betas, [sum(map(len, E)) for E in self.Es])
        #     self.sigma_x_inverse = np.linalg.inv(sigma_x)
        #     
        #     del sigma_x
        #     self.sigma_x_inverse *= factor
        # else:
        #     raise NotImplementedError(f'Initialization for sigma_x_inverse {sigma_x_inverse_mode} is not implemented')
        # 
        # self.delta_x = np.zeros(self.K)

    # def initialize(self, *args, **kwargs):
    #     ret = initialize(self, *args, **kwargs)
        self.saveWeights(iiter=0)
        self.saveParameters(iiter=0)
    #     return ret

    def estimateWeights(self, iiter):
        logging.info(f'{print_datetime()}Updating latent states')

        assert self.X_constraint == 'none'
        assert self.pairwise_potential_mode == 'normalized'

        rXTs = []
        with Pool(min(self.num_processes, self.num_repli)) as pool:
            for replicate in range(self.num_repli):
                print("Encountering Es_empty")
                if self.Es_empty[replicate]:
                    rXTs.append(pool.apply_async(estimateWeightsWithoutNeighbor, args=(
                        self.YTs[replicate],
                        self.M[:self.Gs[replicate]], self.XTs[replicate], self.prior_xs[replicate], self.sigma_yx_inverses[replicate],
                        self.X_constraint, self.dropout_mode, i,
                    )))
                else:
                    rXTs.append(pool.apply_async(estimateWeightsICM, args=(
                        self.YTs[replicate], self.Es[replicate],
                        self.M[:self.Gs[replicate]], self.XTs[replicate], self.prior_xs[replicate], self.sigma_yx_inverses[replicate], self.sigma_x_inverse,
                        self.X_constraint, self.dropout_mode, self.pairwise_potential_mode, replicate,
                    )))
            self.XTs = [new_XT.get(1e9) if isinstance(new_XT, multiprocessing.pool.ApplyResult) else new_XT for new_XT in rXTs]
        pool.join()

        self.saveWeights(iiter=iiter)

    def estimateParameters(self, iiter):
        logging.info(f'{print_datetime()}Updating model parameters')

        self.Q = 0
        if self.pairwise_potential_mode == 'normalized' and all(
                prior_x[0] in ['Exponential', 'Exponential shared', 'Exponential shared fixed']
                for prior_x in self.prior_xs):
            # pool = Pool(1)
            # Q_Y = pool.apply_async(estimateParametersY, args=([self])).get(1e9)
            # pool.close()
            # pool.join()
            Q_Y = estimateParametersY(self)
            self.Q += Q_Y

            Q_X = estimateParametersX(self, iiter)
            self.Q += Q_X
        else:
            raise NotImplementedError

        self.saveParameters(iiter=iiter)
        self.saveProgress(iiter=iiter)

        return self.Q

    def skipSaving(self, iiter):
        return iiter % 10 != 0

    def saveHyperparameters(self):
        if self.result_filename is None: return

        with h5py.File(self.result_filename, 'w') as f:
            f['hyperparameters/repli_list'] = [_.encode('utf-8') for _ in self.repli_list]
            for k in ['prior_x_modes']:
                for repli, v in zip(self.repli_list, getattr(self, k)):
                    f[f'hyperparameters/{k}/{repli}'] = encode4h5(v)
            for k in ['lambda_SigmaXInv', 'betas', 'K']:
                f[f'hyperparameters/{k}'] = encode4h5(getattr(self, k))

    def saveWeights(self, iiter):
        if self.result_filename is None:
            return
        if self.skipSaving(iiter):
            return

        f = openH5File(self.result_filename)
        if f is None: return

        for repli, XT in zip(self.repli_list, self.XTs):
            f[f'latent_states/XT/{repli}/{iiter}'] = XT

        f.close()

    def saveParameters(self, iiter):
        if self.result_filename is None: return
        if self.skipSaving(iiter): return 
        f = openH5File(self.result_filename)
        if f is None: return

        for k in ['M', 'sigma_x_inverse']:
            f[f'parameters/{k}/{iiter}'] = getattr(self, k)

        for k in ['sigma_yx_inverses']:
            for repli, v in zip(self.repli_list, getattr(self, k)):
                f[f'parameters/{k}/{repli}/{iiter}'] = v

        for k in ['prior_xs']:
            for repli, v in zip(self.repli_list, getattr(self, k)):
                f[f'parameters/{k}/{repli}/{iiter}'] = np.array(v[1:])

        f.close()


    def saveProgress(self, iiter):
        if self.result_filename is None: return

        f = openH5File(self.result_filename)
        if f is None:
            return

        f[f'progress/Q/{iiter}'] = self.Q

        f.close()
