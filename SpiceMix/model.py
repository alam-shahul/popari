import sys, time, itertools, resource, logging, h5py, os, pickle
from pathlib import Path
import multiprocessing
from multiprocessing import Pool
from util import print_datetime, openH5File, encode4h5, parse_suffix

import numpy as np, pandas as pd
import torch
import anndata as ad

from load_data import load_expression, load_edges, load_genelist, load_anndata, save_anndata
from initialization import initialize_kmeans, initialize_Sigma_x_inv, initialize_svd
from estimate_weights import estimate_weight_wonbr, estimate_weight_wnbr, \
    estimate_weight_wnbr_phenotype, estimate_weight_wnbr_phenotype_v2
from estimate_parameters import estimate_M, estimate_Sigma_x_inv, estimate_phenotype_predictor

class SpiceMixPlus:
    """SpiceMixPlus optimization model.

    Models spatial biological data using the NMF-HMRF formulation. Supports multiple
    fields-of-view (FOVs) and multimodal data.

    Attributes:
        device: device to use for PyTorch operations
        num_processes: number of parallel processes to use for optimizing weights (should be <= #FOVs)
        replicate_names: names of replicates/FOVs in input dataset
        TODO: finish docstring
    """
    def __init__(
            self,
            K, lambda_Sigma_x_inv, repli_list, betas=None, prior_x_modes=None,
            path2result=None, context=None, context_Y=None, random_state=0
    ):

        if not context:
            context = dict(device='cpu', dtype=torch.float32)
        if not context_Y:
            context_Y = context

        self.context = context
        self.context_Y = context_Y
        self.repli_list = repli_list
        self.num_repli = len(self.repli_list)

        self.random_state = random_state
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.K = K
        self.lambda_Sigma_x_inv = lambda_Sigma_x_inv
        if betas is None:
            self.betas = np.full(self.num_repli, 1/self.num_repli)
        else:
            self.betas = np.array(betas, copy=False) / sum(betas)

        if prior_x_modes is None:
            prior_x_modes = ['exponential shared fixed'] * self.num_repli

        self.prior_x_modes = prior_x_modes
        self.M_constraint = 'simplex'
        # self.M_constraint = 'unit sphere'
        self.X_constraint = 'none'
        self.dropout_mode = 'raw'
        self.sigma_yx_inv_mode = 'average'
        self.pairwise_potential_mode = 'normalized'

        if path2result is not None:
            self.path2result = path2result
            self.result_filename = self.path2result
            logging.info(f'result file = {self.result_filename}')
        else:
            self.result_filename = None
        # self.save_hyperparameters()

        self.Ys = self.meta = self.Es = self.Es_isempty = self.genes = self.Ns = self.Gs = self.GG = None
        self.Sigma_x_inv = self.Xs = self.sigma_yxs = None
        
        self.M = None
        self.Ms = self.M_bar = None
        self.lambda_M = 0
        self.Zs = self.Ss = self.Z_optimizers = None
        self.optimizer_Sigma_x_inv = None
        self.prior_xs = None
        self.phenotypes = self.phenotype_predictors = None
        self.meta_repli = None

    def load_dataset(self, path2dataset, data_format="raw", iteration=None, neighbor_suffix=None, expression_suffix=None):
        """Load dataset into SpiceMix object.

        TODO: change this to accept anndata objects.
        """

        path2dataset = Path(path2dataset)
        neighbor_suffix = parse_suffix(neighbor_suffix)
        expression_suffix = parse_suffix(expression_suffix)
        
        if data_format == "anndata":
            if not iteration:
                raise ValueError("Must pass an integer value for iteration if using data_format=\"anndata\"")

            self.datasets = load_anndata(path2dataset / f"trained_{iteration}_iterations.h5", self.repli_list, self.context)
            self.Ys = []
            for dataset in self.datasets:
                Y = torch.tensor(dataset.X, **self.context_Y)
                self.Ys.append(Y)

        self.raw_Ys = []
        for replicate in self.repli_list:
            for s in ['pkl', 'txt', 'tsv', 'pickle']:
                path2file = path2dataset / 'files' / f'expression_{replicate}{expression_suffix}.{s}'
                if not path2file.exists():
                    continue

                self.raw_Ys.append(load_expression(path2file))
                break
        assert len(self.raw_Ys) == len(self.repli_list)
        self.Ns, self.Gs = zip(*map(np.shape, self.raw_Ys))
        self.GG = max(self.Gs)
        self.genes =  [
            load_genelist(path2dataset / 'files' / f'genes_{r}{expression_suffix}.txt')
            for r in self.repli_list
        ]

        self.Es = [
            load_edges(path2dataset / 'files' / f'neighborhood_{i}{neighbor_suffix}.txt', N)
            for i, N in zip(self.repli_list, self.Ns)
        ]
        self.Es_isempty = [sum(map(len, E)) == 0 for E in self.Es]

        if data_format == "raw":
            self.datasets = []
            self.Ys = []
            for index, (raw_gene_expression, gene_list, n_obs, replicate, adjacency_list) in enumerate(zip(self.raw_Ys, self.genes, self.Ns, self.repli_list, self.Es)):
                gene_expression = len(gene_list) / self.GG * self.K * raw_gene_expression / raw_gene_expression.sum(axis=1).mean()
                gene_expression = torch.tensor(gene_expression, **self.context_Y)
                if self.context_Y.get("device", "cpu") == "cpu":
                    gene_expression = gene_expression.pin_memory()

                self.Ys.append(gene_expression)
                raw_dataframe = pd.DataFrame(raw_gene_expression, columns=gene_list, index=np.arange(len(gene_expression), dtype=int).astype(str))
                dataset = ad.AnnData(raw_dataframe)
                dataset.raw = dataset
                dataset.X = gene_expression.detach().cpu().numpy()

                def get_adjacency_matrix(adjacency_list):
                    """ According to the context, create an adjacency matrix from an adjacency list

                    """

                    edges = [(i, j) for i, e in enumerate(adjacency_list) for j in e]
                    adjacency_matrix = torch.sparse_coo_tensor(np.array(edges).T, np.ones(len(edges)), size=(len(adjacency_list), len(adjacency_list)), **self.context)
                    
                    return adjacency_matrix

                adjacency_list = load_edges(path2dataset / 'files' / f'neighborhood_{replicate}{neighbor_suffix}.txt', n_obs)
                coordinates = np.loadtxt(path2dataset / 'files' / f'coordinates_{replicate}.txt')
                adjacency_matrix = get_adjacency_matrix(adjacency_list)
                
                dataset.obsp["adjacency_matrix"] = adjacency_matrix
                dataset.obs["adjacency_list"] = adjacency_list
                dataset.obsm["spatial"] = coordinates
                dataset.obs["replicate"] = np.full(n_obs, replicate, dtype=str)
                dataset.uns["Sigma_x_inv"] = {f"{replicate}": None}
                
                self.datasets.append(dataset)

        # self.Ys = [G / self.GG * self.K * Y / Y.sum(1).mean() for Y, G in zip(self.Ys, self.Gs)]
        # self.Ys = [
        #     torch.tensor(Y, **self.context_Y).pin_memory()
        #     if self.context_Y.get('device', 'cpu') == 'cpu' else
        #     torch.tensor(Y, **self.context_Y)
        #     for Y in self.Ys
        # ]
        # TODO: save Ys in cpu and calculate matrix multiplications (e.g., MT Y) using mini-batch.
        df_all = []
        for r, dataset in zip(self.repli_list, self.datasets):
            path2file = path2dataset / 'files' / f'meta_{r}.pkl'
            if os.path.exists(path2file):
                with open(path2file, 'rb') as f:
                    data = pickle.load(f)
                    df_all.append(data)
            path2file = path2dataset / 'files' / f'meta_{r}.csv'
            if os.path.exists(path2file):
                df_all.append(pd.read_csv(path2file))
                continue
            path2file = path2dataset / 'files' / f'celltypes_{r}.txt'
            if os.path.exists(path2file):
                df = pd.read_csv(path2file, header=None)
                df.columns = ['cell type']
                dataset.obs["cell_type"] = df.values
                # df['repli'] = r
                df_all.append(df)
                continue
            raise FileNotFoundError(r)
        assert len(df_all) == len(self.repli_list)
        for r, df in zip(self.repli_list, df_all):
            df['repli'] = r

        df_all = pd.concat(df_all)
        self.merged_dataset = ad.concat(self.datasets, label="batch", keys=self.repli_list, merge="unique", uns_merge="unique", pairwise=True)
        self.meta = df_all

        self.phenotypes = [{} for i in range(self.num_repli)]
        self.phenotype_predictors = {}

        if data_format != "anndata":
            pass
            # save_anndata(path2dataset / "all_data.h5", self.datasets, self.repli_list)

    def register_phenotype_predictors(self, phenotype2predictor):
        """Register a phenotype and its predictor to the SpiceMixPlus object.
        
        Args:
            phenotype2predictor: a dictionary. Each key is a column name in self.meta. The corresponding value is a
            3-element tuple that contains:
                1) a predictor, an instance of torch.nn.Module
                2) an optimizer
                3) a loss function
        """
        for phenotype_name, (predictor, optimizer, loss_function) in phenotype2predictor.items():
            self.register_phenotype_predictor(phenotype_name, predictor, optimizer, loss_function)

        # self.phenotype_predictors = phenotype2predictor
        # for phenotype_name, (predictor, optimizer, _) in phenotype2predictor.items():
        #     for param in predictor.parameters():
        #         param.requires_grad_(False)
        #     for dataset in self.datasets:
        #         dataset.uns[f"{phenotype_name}_predictor"] = predictor
        #         dataset.uns[f"{phenotype_name}_optimizer"] = optimizer

        # self.phenotypes = [None] * len(self.repli_list)
        # for (repli, df), dataset in zip(self.meta.groupby('repli'), self.datasets):
        #     phenotypes = {}
        #     for key in phenotype2predictor:
        #         phenotype = torch.tensor(df[key].values, device=self.context.get('device', 'cpu'))
        #         phenotypes[key] = phenotype
        #     self.phenotypes[self.repli_list.index(repli)] = phenotypes
    
    def register_phenotype_predictor(self, phenotype_name, predictor, optimizer, loss_function):
        """Register a phenotype and its predictor to the SpiceMixPlus object.
        
        Args:
            phenotype2predictor: a dictionary. Each key is a column name in self.meta. The corresponding value is a
            3-element tuple that contains:
                1) a predictor, an instance of torch.nn.Module
                2) an optimizer
                3) a loss function
        """
        self.phenotype_predictors[phenotype_name] = (predictor, optimizer, loss_function)
        for param in predictor.parameters():
            param.requires_grad_(False)
        
        for (repli, df) in self.merged_dataset.obs.groupby('batch'):
            phenotypes = {}
            phenotype = torch.tensor(df[phenotype_name].values, device=self.context.get('device', 'cpu'))
            self.phenotypes[self.repli_list.index(repli)][phenotype_name] = phenotype

    def register_meta_repli(self, df_meta_repli):
        self.meta_repli = df_meta_repli

    # def resume(self, iiter):

    def get_M(self, repli):
        key = 'metagene group'
        if self.meta_repli is None or key not in self.meta_repli.columns:
            key = 'shared'
        else:
            key = self.meta_repli.loc[repli, key]
        if self.Ms is None:
            return key
        else:
            return self.Ms[key]

    def initialize(self, method='kmeans'):
        # if method == 'kmeans':
        #   self.M, self.Xs = initialize_kmeans(
        #       self.K, self.Ys,
        #       kwargs_kmeans=dict(random_state=random_state),
        #       context=self.context,
        #   )
        # elif method == 'svd':
        #   self.M, self.Xs = initialize_svd(self.K, self.Ys, context=self.context)
        # else:
        #   raise NotImplementedError

        if self.phenotype_predictors:
            key = next(iter(self.phenotype_predictors.keys()))
            set_of_labels = np.unique(sum([phenotype[key].cpu().numpy().tolist() for phenotype in self.phenotypes if phenotype[key] is not None], []))
           
            # Seeding some metagenes to explain the phenotype(s)
            L = len(set_of_labels)
            M = torch.zeros([self.GG, L], **self.context)
            num_by_class = torch.zeros([L], **self.context)
            Xs = [torch.zeros([N, L], **self.context) for N in self.Ns]
            is_valid = lambda Y, phenotype: Y.shape[1] == M.shape[0] and phenotype[key] is not None
            for phenotype, dataset, X, Y in zip(self.phenotypes, self.datasets, Xs, self.Ys):
                if not is_valid(dataset.X, phenotype):
                    continue

                for i, c in enumerate(set_of_labels):
                    mask = phenotype[key] == c
                    M[:, i] += Y[mask].sum(axis=0)
                    num_by_class[i] += mask.sum()
                    X[mask, i] = 1

            M /= num_by_class

            Ys = []
            for phenotype, dataset, Y in zip(self.phenotypes, self.datasets, self.Ys):
                if not is_valid(dataset.X, phenotype):
                    continue

                Y = Y.clone()
                for i, c in enumerate(set_of_labels):
                    mask = phenotype[key] == c
                    Y[mask] -= M[:, i]
                Ys.append(Y)

            M_res, Xs_res = initialize_svd(self.K - L, Ys, context=self.context)
            self.M = torch.cat([M, M_res], dim=1)
            self.Xs = []
            Xs_res_iter = iter(Xs_res)
            for phenotype, dataset, Y, X in zip(self.phenotypes, self.datasets, self.Ys, Xs):
                if is_valid(Y, phenotype):
                    # If phenotype information is available, use seeded hidden states
                    self.Xs.append(torch.cat([X, next(Xs_res_iter)], dim=1))
                else:
                    # Otherwise, use least squares to fit X to initialized M
                    self.Xs.append(torch.linalg.lstsq(self.M, Y.T)[0].clip_(min=0).T.contiguous())

        elif method == 'kmeans':
            self.M, self.Xs = initialize_kmeans(
                self.K, self.Ys,
                kwargs_kmeans=dict(random_state=self.random_state),
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
            scale_factor = torch.linalg.norm(self.M, axis=0, ord=1, keepdim=True)
        elif self.M_constraint == 'unit_sphere':
            scale_factor = torch.linalg.norm(self.M, axis=0, ord=2, keepdim=True)
        else:
            raise NotImplementedError
        
        self.M.div_(scale_factor)
        for X in self.Xs:
            X.mul_(scale_factor)
        del scale_factor
        keys = set(self.get_M(r) for r in self.repli_list)
        self.M_bar = self.M
        self.Ms = {key: self.M_bar.clone() for key in keys}

        if all(prior_x_mode == 'exponential shared fixed' for prior_x_mode in self.prior_x_modes):
            self.prior_xs = [(torch.ones(self.K, **self.context),) for _ in range(self.num_repli)]
        else:
            raise NotImplementedError

        self.estimate_sigma_yx()
        self.estimate_phenotype_predictor()
        #
        # self.Zs = []
        # self.Ss = []
        # for X in self.Xs:
        #   S = torch.linalg.norm(X, ord=1, dim=1, keepdim=True)
        #   self.Ss.append(S)
        #   self.Zs.append(X / S)
        # self.Z_optimizers = [
        #   torch.optim.Adam(
        #       [Z],
        #       lr=1e-3,
        #       betas=(.3, .9),
        #   )
        #   for Z in self.Zs
        # ]

        for dataset, X, replicate in zip(self.datasets, self.Xs, self.repli_list):
            dataset.uns["M"] = {f"{replicate}": self.M}
            dataset.uns["M_bar"] = {f"{replicate}": self.M_bar}
            dataset.obsm["X"] = X

        # self.save_weights(iiter=0)
        # self.save_parameters(iiter=0)

    def initialize_Sigma_x_inv(self):
        self.Sigma_x_inv = initialize_Sigma_x_inv(self.K, self.Xs, self.betas, self.context, self.datasets)

        # Zero-centering Sigma_x_inv
        self.Sigma_x_inv -= self.Sigma_x_inv.mean()
        # self.Sigma_x_inv.requires_grad_(True)
        
        for replicate, dataset in zip(self.repli_list, self.datasets):
            dataset.uns["Sigma_x_inv"][f"{replicate}"] = self.Sigma_x_inv
       
        # This optimizer retains its state throughout the optimization
        self.optimizer_Sigma_x_inv = torch.optim.Adam(
            [self.Sigma_x_inv],
            lr=1e-1,
            betas=(.5, .9),
        )

    def estimate_sigma_yx(self):
        squared_loss = np.array([
            torch.linalg.norm(torch.addmm(Y, X, self.M.T, alpha=-1), ord='fro').item() ** 2
            for Y, X in zip(self.Ys, self.Xs)
        ])
        sizes = np.array([np.prod(dataset.X.shape) for dataset in self.datasets])
        if self.sigma_yx_inv_mode == 'separate':
            self.sigma_yxs = np.sqrt(squared_loss / sizes)
        elif self.sigma_yx_inv_mode == 'average':
            sigma_yx = np.sqrt(np.dot(self.betas, squared_loss) / np.dot(self.betas, sizes))
            self.sigma_yxs = np.full(self.num_repli, float(sigma_yx))
        else:
            raise NotImplementedError

        for dataset, sigma_yx in zip(self.datasets, self.sigma_yxs):
            dataset.uns["sigma_yx"] = sigma_yx

    def estimate_weights(self, iiter, use_spatial):
        logging.info(f'{print_datetime()}Updating latent states')
        assert len(use_spatial) == self.num_repli

        assert self.X_constraint == 'none'
        assert self.pairwise_potential_mode == 'normalized'

        loss_list = []
        for i, (X, sigma_yx, E, phenotype, prior_x_mode, prior_x, replicate, dataset, Y) in enumerate(zip(
                self.Xs, self.sigma_yxs, self.Es, self.phenotypes, self.prior_x_modes, self.prior_xs, self.repli_list, self.datasets, self.Ys)):
            valid_keys = [k for k, v in phenotype.items() if v is not None]
            if len(valid_keys) > 0:
                loss = estimate_weight_wnbr_phenotype(
                    Y, self.M, X, sigma_yx, replicate, prior_x_mode, prior_x,
                    {k: phenotype[k] for k in valid_keys}, {k: self.phenotype_predictors[k] for k in valid_keys},
                    dataset, context=self.context,
                )

                # S = self.Ss[i]
                # Z = self.Zs[i]
                # S[:] = torch.linalg.norm(X, ord=1, dim=1, keepdim=True)
                # Z[:] = X / S
                #
                # loss = estimate_weight_wnbr_phenotype_v2(
                #   Y, self.M, Z, S, sigma_yx, self.Sigma_x_inv, E, prior_x_mode, prior_x,
                #   self.Z_optimizers[i],
                #   {k: phenotype[k] for k in valid_keys}, {k: self.phenotype_predictors[k] for k in valid_keys},
                #   context=self.context,
                # )
                #
                # X[:] = Z * S
            elif self.Es_isempty[i] or not use_spatial[i]:
                loss = estimate_weight_wonbr(
                    Y, self.M, X, sigma_yx, prior_x_mode, prior_x, context=self.context)
            else:
                loss, self.Xs[i] = estimate_weight_wnbr(
                    Y, self.M, X, sigma_yx, replicate, prior_x_mode, prior_x, dataset, context=self.context)

                # S = self.Ss[i]
                # Z = self.Zs[i]
                # S[:] = torch.linalg.norm(X, ord=1, dim=1, keepdim=True)
                # Z[:] = X / S
                #
                # loss = estimate_weight_wnbr_phenotype_v2(
                #   Y, self.M, Z, S, sigma_yx, self.Sigma_x_inv, E, prior_x_mode, prior_x,
                #   self.Z_optimizers[i],
                #   {k: phenotype[k] for k in valid_keys}, {k: self.phenotype_predictors[k] for k in valid_keys},
                #   context=self.context,
                # )
                #
                # X[:] = Z * S
            loss_list.append(loss)

        # self.save_weights(iiter=iiter)

        return loss_list

    def estimate_phenotype_predictor(self):
        for phenotype_name, predictor_tuple in self.phenotype_predictors.items():
            input_target_list = []
            for X, phenotype, dataset in zip(self.Xs, self.phenotypes, self.datasets):
                if phenotype[phenotype_name] is None:
                    continue

                # Normalizing X per cell
                X = X / torch.linalg.norm(X, ord=1, dim=1, keepdim=True)
                input_target_list.append((X, phenotype[phenotype_name]))
            estimate_phenotype_predictor(*zip(*input_target_list), phenotype_name, *predictor_tuple)

    def estimate_M(self):
        for (group, M), dataset, replicate in zip(self.Ms.items(), self.datasets, self.repli_list):
            is_valid = [self.get_M(repli) is M for repli in self.repli_list]
            Xs, sigma_yxs, betas, datasets = zip(*itertools.compress(zip(
                self.Xs, self.sigma_yxs, self.betas, self.datasets), is_valid))
            betas = np.array(betas)
            betas /= sum(betas) # not sure if we should do this
            self.Ms[group] = estimate_M(
                # self.Ys, self.Xs, self.M, self.sigma_yxs, self.betas,
                Xs, M, np.array(sigma_yxs), betas, datasets,
                M_bar=self.M_bar, lambda_M=self.lambda_M,
                M_constraint=self.M_constraint, context=self.context,
            )
            dataset.uns["M"][f"{replicate}"] = self.Ms[group]

        # Set M_bar to average of self.Ms
        self.M_bar.zero_()
        for group, M in self.Ms.items():
            self.M_bar.add_(M)
        self.M_bar.div_(len(self.Ms))

    def estimate_parameters(self, iiter, use_spatial, update_Sigma_x_inv=True):
        logging.info(f'{print_datetime()}Updating model parameters')

        if update_Sigma_x_inv and np.any(use_spatial):
            updated_Sigma_x_inv = estimate_Sigma_x_inv(
                self.Xs, self.Sigma_x_inv, self.Es, use_spatial, self.lambda_Sigma_x_inv,
                self.betas, self.optimizer_Sigma_x_inv, self.context, self.datasets)
            with torch.no_grad():
                # Note: in-place update is necessary here in order for optimizer to track same object
                self.Sigma_x_inv[:] = updated_Sigma_x_inv
                for replicate, dataset in zip(self.repli_list, self.datasets):
                    dataset.uns["Sigma_x_inv"][f"{replicate}"][:] = updated_Sigma_x_inv

            # self.optimizer_Sigma_x_inv.param_groups[0]["params"] = self.Sigma_x_inv

        self.estimate_M()
        self.estimate_sigma_yx()
        self.estimate_phenotype_predictor()

        # self.save_parameters(iiter=iiter)

    def save_results(self, path2dataset, iteration, filename=None):
        if not filename:
            filename = f"trained_iteration_{iteration}.h5"

        save_anndata(path2dataset, filename, self.datasets, self.repli_list)

    # def skip_saving(self, iiter):
    #     return iiter % 10 != 0

    # def save_hyperparameters(self):
    #     if self.result_filename is None: return

    #     with h5py.File(self.result_filename, 'w') as f:
    #         f['hyperparameters/repli_list'] = [_.encode('utf-8') for _ in self.repli_list]
    #         for k in ['prior_x_modes']:
    #             for repli, v in zip(self.repli_list, getattr(self, k)):
    #                 f[f'hyperparameters/{k}/{repli}'] = encode4h5(v)
    #         for k in ['lambda_Sigma_x_inv', 'betas', 'K']:
    #             f[f'hyperparameters/{k}'] = encode4h5(getattr(self, k))

    # def save_weights(self, iiter):
    #     if not self.result_filename:
    #         return
    #     if self.skip_saving(iiter):
    #         return

    #     f = openH5File(self.result_filename)
    #     if f is None:
    #         return

    #     for repli, XT in zip(self.repli_list, self.Xs):
    #         f[f'latent_states/XT/{repli}/{iiter}'] = XT.cpu().numpy()

    #     f.close()

    # def save_parameters(self, iiter):
    #     if self.result_filename is None: return
    #     if self.skip_saving(iiter): return

    #     f = openH5File(self.result_filename)
    #     if f is None: return

    #     M = self.M
    #     if M is not None:
    #         print(M)
    #         M = M.cpu().numpy()
    #         f[f'parameters/M/{iiter}'] = M

    #     M_bar = self.M_bar
    #     if M_bar is not None:
    #         print(M_bar)
    #         M_bar = M_bar.cpu().numpy()
    #         f[f'parameters/M_bar/{iiter}'] = M_bar

    #     Sigma_x_inv = self.Sigma_x_inv
    #     if Sigma_x_inv is not None:
    #         Sigma_x_inv = Sigma_x_inv.cpu().numpy()
    #         f[f'parameters/Sigma_x_inv/{iiter}'] = Sigma_x_inv

    #     
    #     for repli, M in zip(self.repli_list, self.Ms):
    #         if M is not None and not isinstance(M, str):
    #             print(M)
    #             M = M.cpu().numpy()
    #             f[f'latent_states/Ms/{repli}/{iiter}'] = M

    #     for k in ['sigma_yxs']:
    #         for repli, v in zip(self.repli_list, getattr(self, k)):
    #             f[f'parameters/{k}/{repli}/{iiter}'] = v

    #     for k in ['prior_xs']:
    #         for repli, v in zip(self.repli_list, getattr(self, k)):
    #             f[f'parameters/{k}/{repli}/{iiter}'] = np.array(v[1:])

    #     f.close()

    # def save_progress(self, iiter):
    #     return
    #     if self.result_filename is None: return

    #     f = openH5File(self.result_filename)
    #     if f is None:
    #         return

    #     f[f'progress/Q/{iiter}'] = self.Q

    #     f.close()
