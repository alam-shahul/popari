import sys, time, itertools, psutil, resource, logging, h5py, os
from pathlib import Path
import multiprocessing
from multiprocessing import Pool
from util import print_datetime, parseSuffix, openH5File, encode4h5, save_dict_to_hdf5

import numpy as np
import gurobipy as grb
import torch

from load_data import load_expression, load_edges
from initialization import initialize_M_by_kmeans, initialize_sigma_x_inverse, partial_nmf
from estimate_weights import estimate_weights_icm, estimate_weights_no_neighbors
from estimate_parameters import estimate_parameters_x, estimate_parameters_y

class SpiceMix:
    """SpiceMix optimization model.

    Provides state and functions to fit spatial transcriptomics data using the NMF-HMRF model. Can support multiple
    fields-of-view (FOVs).

    Attributes:
        device: device to use for PyTorch operations
        num_processes: number of parallel processes to use for optimizing weights (should be <= #FOVs)
        replicate_names: names of replicates/FOVs in input dataset

        TODO: finish docstring
    """

    def __init__(self, path2dataset, replicate_names, use_spatial, neighbor_suffix, expression_suffix, K,
                 lambda_sigma_x_inverse, betas, prior_x_modes, result_filename, resume_training=False, device='cpu', num_processes=1):

        self.device = device
        self.num_processes = num_processes

        self.path2dataset = Path(path2dataset)
        self.replicate_names = replicate_names
        self.use_spatial = use_spatial
        self.num_repli = len(self.replicate_names)
        assert len(self.replicate_names) == len(self.use_spatial)
        self.K = K
        self.load_dataset(neighbor_suffix=neighbor_suffix, expression_suffix=expression_suffix)

        self.completed_iterations = 0
        self.lambda_sigma_x_inverse = lambda_sigma_x_inverse
        self.betas = betas
        self.prior_x_modes = prior_x_modes
        self.M_constraint = 'sum2one'
        self.X_constraint = 'none'
        self.dropout_mode = 'raw'
        self.sigma_yx_inverse_mode = 'average'
        self.pairwise_potential_mode = 'normalized'

        self.result_filename = Path(result_filename)
        logging.info(f'{print_datetime()}result file = {self.result_filename}')

        self.save_hyperparameters()
        self.save_dataset()

    # def __del__(self):
    #   pass
        # if self.result_h5 is not None:
        #   self.result_h5.close()

    def load_dataset(self, neighbor_suffix=None, expression_suffix=None):
        """Load spatial transcriptomics data from relevant filepaths.

        This function is called during the initialization of any SpiceMix object, and it
        sets important attributes that will be used during the SpiceMix optimization.

        Args:
            neighbor_suffix: pattern to match at end of neighborhood filename.
            expression_suffix: pattern to match at end of expression data filename.
        """

        neighbor_suffix = parseSuffix(neighbor_suffix)
        expression_suffix = parseSuffix(expression_suffix)

        self.unscaled_YTs = []
        for replicate in self.replicate_names:
            # TODO: is it necessary to allow multiple extensions, or can we require that the expression data are
            # in .txt files?
            for extension in ['txt', 'pkl', 'pickle']:
                filepath = self.path2dataset / 'files' / f'expression_{replicate}{expression_suffix}.{extension}'
                if not filepath.exists():
                    continue

                gene_expression = load_expression(filepath)
                self.unscaled_YTs.append(gene_expression)
        
        self.Ns, self.Gs = zip(*map(np.shape, self.unscaled_YTs))
        self.max_genes = max(self.Gs)
        
        self.YTs = [G / self.max_genes * self.K * unscaled_YT / unscaled_YT.sum(axis=1).mean() for unscaled_YT, G in zip(self.unscaled_YTs, self.Gs)]


        # TODO: change to use dictionary with empty mappings instead of list of empty lists
        self.Es = {}
        for replicate, num_nodes, use_spatial in zip(range(self.num_repli), self.Ns, self.use_spatial):
            if use_spatial:
                E = load_edges(self.path2dataset / 'files' / f'neighborhood_{replicate}{neighbor_suffix}.txt', num_nodes)
            else:
                E = {node: [] for node in range(num_nodes)}

            self.Es[replicate] = E

        self.total_edge_counts = [sum(map(len, E.values())) for E in self.Es.values()]
        self.gene_sets = [np.loadtxt(self.path2dataset / 'files' / f'genes_{replicate}{expression_suffix}.txt', dtype=str) for replicate in self.replicate_names]

    def initialize_model(self, random_seed4kmeans, initial_nmf_iterations=5, sigma_x_inverse_mode='Constant'):
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
        self.M, self.XTs, self.sigma_yx_inverses, self.prior_x_parameter_sets = partial_nmf(self, prior_x_modes=self.prior_x_modes, initial_nmf_iterations=initial_nmf_iterations)
    
        if sum(self.total_edge_counts) == 0: 
            sigma_x_inverse_mode = 'Constant'
    
        self.sigma_x_inverse = initialize_sigma_x_inverse(self.K, self.XTs, self.Es, self.betas, sigma_x_inverse_mode=sigma_x_inverse_mode)
        # TODO: this block was commented out originally; can we just remove it?
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
        self.save_weights(iiter=0)
        self.save_parameters(iiter=0)
    #     return ret

    def estimate_weights(self, iiter):
        logging.info(f'{print_datetime()}Updating latent states')

        updated_XTs = []
        with Pool(min(self.num_processes, self.num_repli)) as pool:
            for replicate in range(self.num_repli):
                if self.total_edge_counts[replicate] == 0:
                    updated_XTs.append(pool.apply_async(estimate_weights_no_neighbors, args=(
                        self.YTs[replicate],
                        self.M[:self.Gs[replicate]], self.XTs[replicate], self.prior_x_parameter_sets[replicate], self.sigma_yx_inverses[replicate],
                        self.X_constraint, self.dropout_mode, replicate,
                    )))
                else:
                    updated_XTs.append(pool.apply_async(estimate_weights_icm, args=(
                        self.YTs[replicate], self.Es[replicate],
                        self.M[:self.Gs[replicate]], self.XTs[replicate], self.prior_x_parameter_sets[replicate], self.sigma_yx_inverses[replicate], self.sigma_x_inverse,
                        self.X_constraint, self.dropout_mode, self.pairwise_potential_mode, replicate,
                    )))

            # TODO: is this line necessary? Seems like the results will always be of type ApplyResult
            self.XTs = [updated_XT.get(1e9) if isinstance(updated_XT, multiprocessing.pool.ApplyResult) else updated_XT for updated_XT in updated_XTs]
        pool.join()

        self.save_weights(iiter=iiter)

    def estimate_parameters(self, iiter):
        logging.info(f'{print_datetime()}Updating model parameters')

        self.Q = 0
        # pool = Pool(1)
        # Q_Y = pool.apply_async(estimateParametersY, args=([self])).get(1e9)
        # pool.close()
        # pool.join()
        Q_Y = estimate_parameters_y(self)
        Q_X = estimate_parameters_x(self)
        
        self.Q += (Q_X + Q_Y)

        self.save_parameters(iiter=iiter)

        return self.Q

    def fit(self, max_iterations):
        """Fit SpiceMix model using NMF-HMRF updates.

        Alternately updates weights (XTs) and parameters (M, sigma_x_inverse, sigma_yx_inverse, prior_x_parameter_sets).

        Args:
            max_iterations: max number of complete iterations of NMF-HMRF updates.
        """

        last_Q = np.nan
        for iteration in range(self.completed_iterations + 1, max_iterations+1):
            logging.info(f'{print_datetime()}Iteration {iteration} begins')

            self.estimate_weights(iiter=iteration)
            self.estimate_parameters(iiter=iteration)
            self.saveProgress(iiter=iteration)

            logging.info(f'{print_datetime()}Q = {self.Q:.4f}\tdiff Q = {self.Q-last_Q:.4e}')
            last_Q = self.Q
            self.completed_iterations += 1

    def skipSaving(self, iiter):
        return iiter % 10 != 0

    def save_dataset(self):
        state_update = {
            "dataset": {
                "YTs": self.YTs,
                "unscaled_YTs": self.unscaled_YTs,
                "Es": self.Es,
                "gene_sets": [np.char.encode(gene_set, encoding="utf-8") for gene_set in self.gene_sets]
            }
        }

        save_dict_to_hdf5(self.result_filename, state_update)

    def save_hyperparameters(self):
        # if self.result_filename is None: return
        #
        state_update = {
            "hyperparameters": {
                "replicate_names": [replicate_name.encode('utf-8') for replicate_name in self.replicate_names],
                "prior_x_modes": {
                    replicate_name: prior_x_mode.encode('utf-8') for replicate_name, prior_x_mode in zip(self.replicate_names, self.prior_x_modes)
                },
                "use_spatial": {
                    replicate_name: use_spatial for replicate_name, use_spatial in zip(self.replicate_names, self.use_spatial)
                },
                "lambda_sigma_x_inverse": self.lambda_sigma_x_inverse,
                "betas": self.betas,
                "K": self.K,
                "completed_iterations": self.completed_iterations
            }
        }

        save_dict_to_hdf5(self.result_filename, state_update)
            # f['hyperparameters/replicate_names'] = [replicate_name.encode('utf-8') for replicate_name in self.replicate_names]
            # for repli, v in zip(self.replicate_names, self.prior_x_modes):
            #     f[f'hyperparameters/{k}/{repli}'] = encode4h5(v)
            # for k in ['lambda_sigma_x_inverse', 'betas', 'K']:
            #     f[f'hyperparameters/{k}'] = encode4h5(getattr(self, k))

    def save_weights(self, iiter):
        # if self.result_filename is None:
        #     return
        if self.skipSaving(iiter):
            return

        # f = openH5File(self.result_filename)
        # if f is None: return

        # for repli, XT in zip(self.replicate_names, self.XTs):
        #     f[f'latent_states/XT/{repli}/{iiter}'] = XT

        # f.close()
        state_update = {
            "weights": {
                replicate_name: {iiter: XT} for replicate_name, XT in zip(self.replicate_names, self.XTs)
            }
        }

        if self.skipSaving(iiter):
            return 
        
        save_dict_to_hdf5(self.result_filename, state_update)

    def save_parameters(self, iiter):
        # if self.result_filename is None:
        #     return
        state_update = {
            "parameters": {
                "sigma_x_inverse": {
                    iiter: self.sigma_x_inverse
                },
                "M": {
                    iiter: self.M
                },
                "sigma_yx_inverses": {
                    replicate_name: {iiter: sigma_yx_inverse} for replicate_name, sigma_yx_inverse in zip(self.replicate_names, self.sigma_yx_inverses)
                },
                "prior_x_parameter": {
                    replicate_name: {iiter: prior_x_parameters} for replicate_name, (_, prior_x_parameters) in zip(self.replicate_names, self.prior_x_parameter_sets)
                }
            }
        }

        if self.skipSaving(iiter):
            return 
        
        save_dict_to_hdf5(self.result_filename, state_update)

        # f = openH5File(self.result_filename)

        # for k in ['M', 'sigma_x_inverse']:
        #     f[f'parameters/{k}/{iiter}'] = getattr(self, k)

        # for k in ['sigma_yx_inverses']:
        #     for repli, v in zip(self.replicate_names, getattr(self, k)):
        #         f[f'parameters/{k}/{repli}/{iiter}'] = v

        # for k in ['prior_x_parameter_sets']:
        #     for repli, v in zip(self.replicate_names, getattr(self, k)):
        #         f[f'parameters/{k}/{repli}/{iiter}'] = np.array(v[1:])

        # f.close()


    def saveProgress(self, iiter):
        # if self.result_filename is None:
        #     return

        f = openH5File(self.result_filename)
        if f is None:
            return

        f[f'progress/Q/{iiter}'] = self.Q

        f.close()
