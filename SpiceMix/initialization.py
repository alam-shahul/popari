import sys, logging, time, resource, gc, os
from multiprocessing import Pool
from util import print_datetime

import numpy as np
from sklearn.cluster import KMeans
import gurobipy as grb

def NMF_stepX(YT, M, XT, prior_x, X_constraint, dropout_mode):
    N, num_genes = YT.shape
    K = M.shape[1]

    mX = grb.Model('init_X')
    mX.setParam('OutputFlag', False)
    mX.setParam('Threads', 1)
    vx = mX.addVars(K, lb=0.)
    if X_constraint == 'sum2one':
        mX.addConstr(vx.sum('*') == 1)
        raise NotImplementedError
    elif X_constraint == 'none':
        pass
    else:
        raise NotImplementedError(f'Constraint on X {X_constraint} is not implemented')

    obj_share = 0
    if dropout_mode == 'raw':
        MTM = M[:num_genes].T @ M[:num_genes] + 1e-5*np.eye(K)
        obj_share += grb.quicksum([MTM[i, j] * vx[i] * vx[j] for i in range(K) for j in range(K)])  # quadratic term of X and M
        del MTM
        YTM = YT @ M[:num_genes] * -2
    else:
        raise NotImplementedError(f'Dropout mode {dropout_mode} is not implemented')

    for x, y, yTM in zip(XT, YT, YTM):
        obj = obj_share
        if dropout_mode != 'raw':
            raise NotImplementedError(f'Dropout mode {dropout_mode} is not implemented')
        obj = obj + grb.quicksum([yTM[k] * vx[k] for k in range(K)]) + np.dot(y, y)
        mX.setObjective(obj, grb.GRB.MINIMIZE)
        mX.optimize()
        x[:] = np.fromiter((vx[i].x for i in range(K)), dtype=float)

    return XT

def partial_nmf(self, prior_x_modes, num_NMF_iterations, num_processes=1):
    """Determine initial values for XTs using partial NMF of gene expression array.

    """

    self.XTs = [np.zeros([N, self.K], dtype=float) for N in self.Ns]

    print("Setting sigma_yx_inverses")
    self.sigma_yx_inverses = [1 / gene_expression.std(axis=0).mean() for gene_expression in self.YTs]
    self.prior_xs = []
    for prior_x_mode, gene_expression in zip(prior_x_modes, self.YTs):
        _, num_genes = gene_expression.shape
        total_gene_expression = gene_expression.sum(axis=1)
        if prior_x_mode == 'Truncated Gaussian' or prior_x_mode == 'Gaussian':
            mu_x = np.full(self.K, total_gene_expression.mean() / self.K)
            sigma_x_inverse = np.full(self.K, np.sqrt(self.K) / total_gene_expression.std())
            self.prior_xs.append((prior_x_mode, mu_x, sigma_x_inverse))
        elif prior_x_mode in ['Exponential', 'Exponential shared', 'Exponential shared fixed']:
            lambda_x = np.full(self.K, num_genes / self.max_genes * self.K / total_gene_expression.mean())
            self.prior_xs.append((prior_x_mode, lambda_x))
        else:
            raise NotImplementedError(f'Prior on X {prior_x_mode} is not implemented')

    metagene_model = grb.Model('init_M')
    metagene_model.setParam('OutputFlag', False)
    metagene_model.setParam('Threads', 1)
    if self.M_constraint == 'sum2one':
        metagene_parameters = metagene_model.addVars(self.max_genes, self.K, lb=0.)
        metagene_model.addConstrs((metagene_parameters.sum('*', i) == 1 for i in range(self.K)))
    elif self.M_constraint == 'nonnegative':
        metagene_parameters = metagene_model.addVars(self.K, lb=0.)
    else:
        raise NotImplementedError(f'Constraint on M {self.M_constraint} is not implemented')

    iteration = 0
    last_M = np.copy(self.M)
    last_rmse = np.nan

    for iteration in range(num_NMF_iterations):
    # while iteration < num_NMF_iterations:
        # update XT
        with Pool(min(num_processes, len(self.YTs))) as pool:
            self.XTs = pool.starmap(NMF_stepX, zip(
                self.YTs, [self.M]*self.num_repli, self.XTs, prior_x_modes,
                [self.X_constraint]*self.num_repli, [self.dropout_mode]*self.num_repli,
            ))
        pool.close()
        pool.join()
        del pool

        # iteration += 1

        Ns = self.Ns
        normalized_XTs = [XT / (XT.sum(1, keepdims=True)+1e-30) for XT in self.XTs]
        logging.info(print_datetime() + 'At iter {}: X: #0 = {},\t#all0 = {},\t#<1e-10 = {},\t#<1e-5 = {},\t#<1e-2 = {},\t#>1e-1 = {}'.format(
            iteration,
            ', '.join(map(lambda x: '%.2f' % x, [(normalized_XT == 0).sum()/N for N, normalized_XT in zip(Ns, normalized_XTs)])),
            ', '.join(map(lambda x: '%d' % x, [(normalized_XT == 0).all(axis=1).sum() for N, normalized_XT in zip(Ns, normalized_XTs)])),
            ', '.join(map(lambda x: '%.1f' % x, [(normalized_XT<1e-10).sum()/N for N, normalized_XT in zip(Ns, normalized_XTs)])),
            ', '.join(map(lambda x: '%.1f' % x, [(normalized_XT<1e-5 ).sum()/N for N, normalized_XT in zip(Ns, normalized_XTs)])),
            ', '.join(map(lambda x: '%.1f' % x, [(normalized_XT<1e-2 ).sum()/N for N, normalized_XT in zip(Ns, normalized_XTs)])),
            ', '.join(map(lambda x: '%.1f' % x, [(normalized_XT>1e-1 ).sum()/N for N, normalized_XT in zip(Ns, normalized_XTs)])),
            ))
        del normalized_XTs

        # update prior_x
        prior_xs_old = self.prior_xs
        self.prior_xs = []
        for prior_x, XT in zip(prior_xs_old, self.XTs):
            if prior_x[0] == 'Truncated Gaussian' or prior_x[0] == 'Gaussian':
                mu_x = XT.mean(0)
                sigma_x_inv = 1. / XT.std(0)
                # sigma_x_inv /= 2          # otherwise, σ^{-1} is overestimated ???
                # sigma_x_inv = np.minimum(sigma_x_inv, 1e2)
                prior_x = (prior_x[0], mu_x, sigma_x_inv)
            elif prior_x[0] in ['Exponential', 'Exponential shared']:
                lambda_x = 1. / XT.mean(0)
                prior_x = (prior_x[0], lambda_x)
            elif prior_x[0] == 'Exponential shared fixed':
                pass
            else:
                raise NotImplementedError(f'Prior on X {prior_x[0]} is not implemented')
            self.prior_xs.append(prior_x)
        
        # TODO: why is this here? It seems like "Exponential shared" is already handled above
        if any(prior_x[0] == 'Exponential shared' for prior_x in self.prior_xs):
            raise NotImplementedError(f'Prior on X Exponential shared is not implemented')

        # update sigma_yx_inv
        ds = [gene_expression - weights @ self.M[:num_genes].T for gene_expression, weights, num_genes in zip(self.YTs, self.XTs, self.Gs)]
        if self.dropout_mode == 'raw':
            ds = [d.ravel() for d in ds]
            sizes = np.fromiter(map(np.size, self.YTs), dtype=float)
        else:
            raise NotImplementedError(f'Dropout mode {self.dropout_mode} is not implemented')
        
        ds = np.fromiter((np.dot(d, d) for d in ds), dtype=float)

        if self.sigma_yx_inv_mode == 'separate':
            sigma_yx_inverses = ds/sizes
            rmse = np.sqrt(np.dot(sigma_yx_inverses, self.betas))
            self.sigma_yx_inverses = 1. / np.sqrt(sigma_yx_inverses + 1e-10)
        elif self.sigma_yx_inv_mode == 'average':
            sigma_yx_inverses = np.dot(self.betas, ds) / np.dot(self.betas, sizes)
            rmse = np.sqrt(sigma_yx_inverses)
            self.sigma_yx_inverses = np.full(self.num_repli, 1 / np.sqrt(sigma_yx_inverses + 1e-10))
        elif self.sigma_yx_inv_mode.startswith('average '):
            idx = np.fromiter(map(int, self.sigma_yx_inv_str.split(' ')[1:]), dtype=int)
            sigma_yx_inverses = np.dot(self.betas[idx], ds[idx]) / np.dot(self.betas[idx], sizes[idx])
            rmse = np.sqrt(sigma_yx_inverses)
            self.sigma_yx_inverses = np.full(self.num_repli, 1 / np.sqrt(sigma_yx_inverses + 1e-10))
        else:
            raise NotImplementedError(f'σ_y|x mode {self.sigma_yx_inv_mode} is not implemented')

        logging.info(f'{print_datetime()}At iter {iteration}: rmse: RMSE = {rmse:.2e}, diff = {last_rmse - rmse:.2e},')

        # if iteration >= num_NMF_iterations:
        #     break

        if self.M_constraint == 'sum2one':
            obj = 0
            for XT, YT, G, beta, sigma_yx_inv in zip(self.XTs, self.YTs, self.Gs, self.betas, self.sigma_yx_inverses):
                if self.dropout_mode == 'raw':
                    # quadratic term
                    XXT = XT.T @ XT * (beta * sigma_yx_inv**2)
                    obj += grb.quicksum(XXT[i, i] * metagene_parameters[k, i] * metagene_parameters[k, i] for k in range(G) for i in range(self.K))
                    XXT *= 2
                    obj += grb.quicksum(XXT[i, j] * metagene_parameters[k, i] * metagene_parameters[k, j] for k in range(G) for i in range(self.K) for j in range(i+1, self.K))
                    # linear term
                    YXT = YT.T @ XT * (-2 * beta * sigma_yx_inv**2)
                    YTY = np.dot(YT.ravel(), YT.ravel()) * beta * sigma_yx_inv**2
                else:
                    raise NotImplementedError(f'Dropout mode {self.dropout_mode} is not implemented')

                obj += grb.quicksum(YXT[i, j] * metagene_parameters[i, j] for i in range(G) for j in range(self.K))
                obj += beta * YTY
            kk = 1e-2
            if kk != 0:
                obj += grb.quicksum([kk/2 * metagene_parameters[k, i] * metagene_parameters[k, i] for k in range(self.max_genes) for i in range(self.K)])
            metagene_model.setObjective(obj)
            metagene_model.optimize()

            self.M = np.array([[metagene_parameters[i, j].x for j in range(self.K)] for i in range(self.max_genes)])
            # TODO: Remove this condition (seems unnecessary)
            #  if self.M_constraint == 'sum2one':
            #     pass
            # # elif M_sum2one == 'L1':
            # #   M /= np.abs(M).sum(0, keepdims=True) + 1e-10
            # # elif M_sum2one == 'L2':
            # #   M /= np.sqrt((M**2).sum(0, keepdims=True)) + 1e-10
            # else:
            #     raise NotImplementedError(f'Constraint on M {self.M_constraint} is not implemented')
        else:
            YXTs = [(YT.T @ XT) * beta for YT, XT, beta in zip(self.YTs, self.XTs, self.betas)]
            obj_2s = []
            for XT, beta in zip(self.XTs, self.betas):
                XXT = XT.T @ XT * beta
                obj_2s.append(grb.quicksum([XXT[i,j]*metagene_parameters[i]*metagene_parameters[j] for i in range(self.K) for j in range(self.K)]))
            for g, Mg in enumerate(self.M):
                obj = []
                for G, YXT, XT, obj_2 in zip(self.Gs, YXTs, self.XTs, obj_2s):
                    if g >= G: continue
                    obj.append(obj_2)
                    obj.append(grb.quicksum([-2*YXT[g, i]*metagene_parameters[i] for i in range(self.K)]))
                metagene_model.setObjective(sum(obj, []), grb.GRB.MINIMIZE)
                metagene_model.optimize()
                Mg[:] = np.array([metagene_parameters[i].x for i in range(self.K)])
            raise NotImplementedError(f'Constraint on M {self.M_constraint} is not implemented')

        dM = self.M - last_M

        # iteration += 1

        logging.info(
            f'{print_datetime()}'
            f'At iter {iteration}: '
            f'Diff M: max = {np.abs(dM).max():.2e}, '
            f'RMS = {np.sqrt(np.mean(np.abs(dM)**2)):.2e}, '
            f'mean = {np.abs(dM).mean():.2e}\t'
        )
        # print(prior_xs)
        sys.stdout.flush()

        last_M = np.copy(self.M)
        last_rmse = rmse

    # return M, XTs, sigma_yx_inverses, prior_xs

def initialize_M_by_kmeans(YTs, K, random_seed4kmeans=0, n_init=10):
    """Use k-means clustering for initial estimate of metagene matrix M.

    Args:
        YTs: A list of gene expression matrices, each with dimensions (num_individuals, num_genes)
        K: Inner-dimensionality of metagene matrix (i.e. number of metagenes desired)
        random_seed4kmeans:

    Returns:
        M_initial, the initial estimate for the metagene matrix, with dimensions (num_genes, K)
    """

    Ns, Gs = zip(*[YT.shape for YT in YTs])
    max_genes = max(Gs)

    concatenated_expression_vectors = np.concatenate([YT for YT in YTs if YT.shape[1] == max_genes], axis=0)

    logging.info(f'{print_datetime()}random seed for K-Means = {random_seed4kmeans}')
    logging.info(f'{print_datetime()}n_init for K-Means = {n_init}')
    
    kmeans = KMeans(
        n_clusters=K,
        random_state=random_seed4kmeans,
        n_jobs=1,
        n_init=n_init,
        tol=1e-8,
    )
    kmeans.fit(concatenated_expression_vectors)

    M_initial = kmeans.cluster_centers_.T

    return M_initial

def initialize_sigma_x_inverse(K, XTs, Es, betas, sigma_x_inverse_mode):
    """Initialize metagene pairwise affinity matrix (sigma_x_inverse).

    Args:
        K: number of metagenes
        XTs: list of initial weightings of metagenes for each sample
        Es: list of adjacency lists for connectivity graph of each sample
        betas: list of beta factors that weight each sample
    Returns:
        Initial estimate of pairwise affinity matrix (sigma_x_inverse).
    """
    
    logging.info(f'{print_datetime()}sigma_x_inverse_mode = {sigma_x_inverse_mode}')
    if sigma_x_inverse_mode == 'Constant':
        sigma_x_inverse = np.zeros([K, K])
    elif sigma_x_inverse_mode.startswith('Identity'):
        factor = float(sigma_x_inverse_mode.split()[1])
        sigma_x_inverse = np.eye(K) * factor
    elif sigma_x_inverse_mode.startswith('EmpiricalFromX'):
        factor = float(sigma_x_inverse_mode.split()[1])
        sigma_x = np.zeros([K, K])
        for XT, adjacency_list, beta in zip(XTs, Es, betas):
            t = np.zeros_like(sigma_x)
            for XTi, neighbors in zip(XT, adjacency_list):
                t += np.outer(XTi, XT[neighbors].sum(0))
            # TODO: seems like sigma_x isn't used anywhere. Can this safely be commented out, then?
            # self.sigma_x += t * beta
        sigma_x /= np.dot(betas, [sum(map(len, E)) for E in Es])
        sigma_x_inverse = np.linalg.inv(sigma_x)
        
        del sigma_x
        sigma_x_inverse *= factor
    else:
        raise NotImplementedError(f'Initialization for sigma_x_inverse {sigma_x_inverse_mode} is not implemented')
   
    return sigma_x_inverse 

def initializeByKMean(self, random_seed4kmeans, num_NMF_iterations=5, sigma_x_inverse_mode='Constant'):
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
    partial_nmf(self, prior_x_modes=self.prior_x_modes, num_NMF_iterations=num_NMF_iterations)

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
