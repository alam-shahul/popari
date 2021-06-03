import sys, logging, time, resource, gc, os
from multiprocessing import Pool
from util import print_datetime

import numpy as np
from sklearn.cluster import KMeans
import gurobipy as grb

def nmf_update(YT, M, XT, X_constraint, dropout_mode):
    """Perform one step of the NMF optimization to update the metagene weights XT.
   
    Uses linear programming formulation to find sparse solution to NMF factorization.

    Args:
        YT: transpose of gene expression matrix for a single replicate
        M: current metagene matrix
        XT: transpose of metagene weight matrix for a single replicate
        X_constraint: constraint on metagene weight parameters
        dropout_mode: TODO

    Returns:
        Updated estimate of XT 
    """
    
    if dropout_mode != 'raw':
        raise NotImplementedError(f'Dropout mode {dropout_mode} is not implemented')

    _, num_genes = YT.shape
    _, num_metagenes  = M.shape

    weight_model = grb.Model('init_X')
    weight_model.setParam('OutputFlag', False)
    weight_model.setParam('Threads', 1)
    weight_variables = weight_model.addVars(num_metagenes, lb=0.)
    if X_constraint == 'sum2one':
        weight_model.addConstr(weight_variables.sum('*') == 1)
        raise NotImplementedError
    elif X_constraint == 'none':
        pass
    else:
        raise NotImplementedError(f'Constraint on X {X_constraint} is not implemented')

    shared_objective = 0
    MTM = M[:num_genes].T @ M[:num_genes] + 1e-5*np.eye(num_metagenes)
    shared_objective += grb.quicksum([MTM[i, j] * weight_variables[i] * weight_variables[j] for i in range(num_metagenes) for j in range(num_metagenes)])  # quadratic term of X and M
    del MTM
    YTM = YT @ M[:num_genes] * -2

    updated_XT = XT
    for cell_index, (y, yTM) in enumerate(zip(YT, YTM)):
        objective = shared_objective
        objective = (objective + grb.quicksum([yTM[i] * weight_variables[i] for i in range(num_metagenes)])) + np.dot(y, y)
        weight_model.setObjective(objective, grb.GRB.MINIMIZE)
        weight_model.optimize()
        updated_XT[cell_index] = np.fromiter((weight_variables[i].x for i in range(num_metagenes)), dtype=float)

    return updated_XT

def partial_nmf(model, prior_x_modes, initial_nmf_iterations, num_processes=1):
    """Determine initial values for XTs using partial NMF of gene expression array.

    Args:
        model: an initialized SpiceMix model object
        prior_x_modes: list of probability distribution types for each replicate
        initial_nmf_iterations: number of iterations to use for NMF-based initialization
        num_processes: number of parallel processes to use for running initialization.

    Returns:
        TODO
    """

    model.XTs = [np.zeros([N, model.K], dtype=float) for N in model.Ns]

    print("Setting sigma_yx_inverses")
    model.sigma_yx_inverses = [1 / gene_expression.std(axis=0).mean() for gene_expression in model.YTs]
    model.prior_x_parameter_sets = []
    for prior_x_mode, gene_expression in zip(prior_x_modes, model.YTs):
        _, num_genes = gene_expression.shape
        total_gene_expression = gene_expression.sum(axis=1)
        if prior_x_mode == 'Truncated Gaussian' or prior_x_mode == 'Gaussian':
            mu_x = np.full(model.K, total_gene_expression.mean() / model.K)
            sigma_x_inverse = np.full(model.K, np.sqrt(model.K) / total_gene_expression.std())
            model.prior_x_parameter_sets.append((prior_x_mode, mu_x, sigma_x_inverse))
        elif prior_x_mode in ('Exponential', 'Exponential shared', 'Exponential shared fixed'):
            lambda_x = np.full(model.K, num_genes / model.max_genes * model.K / total_gene_expression.mean())
            model.prior_x_parameter_sets.append((prior_x_mode, lambda_x))
        else:
            raise NotImplementedError(f'Prior on X {prior_x_mode} is not implemented')

    metagene_model = grb.Model('init_M')
    metagene_model.setParam('OutputFlag', False)
    metagene_model.setParam('Threads', 1)
    if model.M_constraint == 'sum2one':
        metagene_parameters = metagene_model.addVars(model.max_genes, model.K, lb=0.)
        metagene_model.addConstrs((metagene_parameters.sum('*', i) == 1 for i in range(model.K)))
    elif model.M_constraint == 'nonnegative':
        metagene_parameters = metagene_model.addVars(model.K, lb=0.)
    else:
        raise NotImplementedError(f'Constraint on M {model.M_constraint} is not implemented')

    iteration = 0
    last_M = np.copy(model.M)
    last_rmse = np.nan

    for iteration in range(initial_nmf_iterations):
        # update XT
        with Pool(min(num_processes, len(model.YTs))) as pool:
            model.XTs = pool.starmap(nmf_update, zip(
                model.YTs, [model.M]*model.num_repli, model.XTs,
                [model.X_constraint]*model.num_repli, [model.dropout_mode]*model.num_repli,
            ))
        pool.close()
        pool.join()
        del pool

        num_cells_list = model.Ns
        normalized_XTs = [XT / (XT.sum(axis=1, keepdims=True) + 1e-30) for XT in model.XTs]
        logging.info(print_datetime() + 'At iter {}: X: #0 = {}\t#all0 = {}\t#<1e-10 = {}\t#<1e-5 = {}\t#<1e-2 = {}\t#>1e-1 = {}'.format(
            iteration,
            ', '.join(map(lambda x: '%.2f' % x, [(normalized_XT == 0).sum() / num_cells for num_cells, normalized_XT in zip(num_cells_list, normalized_XTs)])),
            ', '.join(map(lambda x: '%d' % x, [(normalized_XT == 0).all(axis=1).sum() for num_cells, normalized_XT in zip(num_cells_list, normalized_XTs)])),
            ', '.join(map(lambda x: '%.1f' % x, [(normalized_XT<1e-10).sum() / num_cells for num_cells, normalized_XT in zip(num_cells_list, normalized_XTs)])),
            ', '.join(map(lambda x: '%.1f' % x, [(normalized_XT<1e-5 ).sum() / num_cells for num_cells, normalized_XT in zip(num_cells_list, normalized_XTs)])),
            ', '.join(map(lambda x: '%.1f' % x, [(normalized_XT<1e-2 ).sum() / num_cells for num_cells, normalized_XT in zip(num_cells_list, normalized_XTs)])),
            ', '.join(map(lambda x: '%.1f' % x, [(normalized_XT>1e-1 ).sum() / num_cells for num_cells, normalized_XT in zip(num_cells_list, normalized_XTs)])),
            )) 
        del normalized_XTs

        # update prior_x
        prior_x_parameter_sets_old = model.prior_x_parameter_sets
        model.prior_x_parameter_sets = []
        for prior_x, XT in zip(prior_x_parameter_sets_old, model.XTs):
            if prior_x[0] == 'Truncated Gaussian' or prior_x[0] == 'Gaussian':
                mu_x = XT.mean(0)
                sigma_x_inv = 1. / XT.std(0)
                # TODO: remove this?
                # sigma_x_inv /= 2          # otherwise, σ^{-1} is overestimated ???
                # sigma_x_inv = np.minimum(sigma_x_inv, 1e2)
                prior_x = (prior_x[0], mu_x, sigma_x_inv)
            elif prior_x[0] in ['Exponential', 'Exponential shared']:
                lambda_x = 1. / XT.mean(axis=0)
                prior_x = (prior_x[0], lambda_x)
            elif prior_x[0] == 'Exponential shared fixed':
                pass
            else:
                raise NotImplementedError(f'Prior on X {prior_x[0]} is not implemented')
            model.prior_x_parameter_sets.append(prior_x)
        
        # TODO: why is this here? It seems like "Exponential shared" is already handled above
        if any(prior_x[0] == 'Exponential shared' for prior_x in model.prior_x_parameter_sets):
            raise NotImplementedError(f'Prior on X Exponential shared is not implemented')

        # update sigma_yx_inv
        squared_differences = [gene_expression - weights @ model.M[:num_genes].T for gene_expression, weights, num_genes in zip(model.YTs, model.XTs, model.Gs)]
        if model.dropout_mode == 'raw':
            squared_unraveled_differences = [squared_difference.ravel() for squared_difference in squared_differences]
            sizes = np.fromiter(map(np.size, model.YTs), dtype=float)
        else:
            raise NotImplementedError(f'Dropout mode {model.dropout_mode} is not implemented')
        
        nmf_objective_values = np.fromiter((np.dot(squared_unraveled_difference, squared_unraveled_difference) for squared_unraveled_difference in squared_unraveled_differences), dtype=float)

        if model.sigma_yx_inverse_mode == 'separate':
            sigma_yx_inverses = nmf_objective_values / sizes
            rmse = np.sqrt(np.dot(sigma_yx_inverses, model.betas))
            model.sigma_yx_inverses = 1. / np.sqrt(sigma_yx_inverses + 1e-10)
        elif model.sigma_yx_inverse_mode == 'average':
            sigma_yx_inverses = np.dot(model.betas, nmf_objective_values) / np.dot(model.betas, sizes)
            rmse = np.sqrt(sigma_yx_inverses)
            model.sigma_yx_inverses = np.full(model.num_repli, 1 / np.sqrt(sigma_yx_inverses + 1e-10))
        elif model.sigma_yx_inverse_mode.startswith('average '):
            idx = np.fromiter(map(int, model.sigma_yx_inv_str.split(' ')[1:]), dtype=int)
            sigma_yx_inverses = np.dot(model.betas[idx], nmf_objective_values[idx]) / np.dot(model.betas[idx], sizes[idx])
            rmse = np.sqrt(sigma_yx_inverses)
            model.sigma_yx_inverses = np.full(model.num_repli, 1 / np.sqrt(sigma_yx_inverses + 1e-10))
        else:
            raise NotImplementedError(f'σ_y|x mode {model.sigma_yx_inverse_mode} is not implemented')

        logging.info(f'{print_datetime()}At iter {iteration}: rmse: RMSE = {rmse:.2e}, diff = {last_rmse - rmse:.2e},')

        if model.M_constraint == 'sum2one':
            objective = 0
            for XT, YT, num_genes, beta, sigma_yx_inverse in zip(model.XTs, model.YTs, model.Gs, model.betas, model.sigma_yx_inverses):
                if model.dropout_mode == 'raw':
                    # quadratic term
                    XXT = XT.T @ XT * (beta * sigma_yx_inverse**2)
                    objective += grb.quicksum(XXT[metagene, metagene] * metagene_parameters[gene, metagene] * metagene_parameters[gene, metagene] for gene in range(num_genes) for metagene in range(model.K))
                    XXT *= 2 # TODO: why 2?
                    objective += grb.quicksum(XXT[metagene, second_metagene] * metagene_parameters[gene, metagene] * metagene_parameters[gene, second_metagene]
                            for gene in range(num_genes) for metagene in range(model.K) for second_metagene in range(metagene+1, model.K))

                    # linear term
                    YXT = YT.T @ XT * (-2 * beta * sigma_yx_inverse**2)
                    YTY = np.dot(YT.ravel(), YT.ravel()) * beta * sigma_yx_inverse**2
                else:
                    raise NotImplementedError(f'Dropout mode {model.dropout_mode} is not implemented')

                objective += grb.quicksum(YXT[i, j] * metagene_parameters[i, j] for i in range(num_genes) for j in range(model.K))
                # TODO: is beta suppose to be here twice?
                objective += beta * YTY

            regularization_multiplier = 1e-2 / 2
            objective += grb.quicksum([regularization_multiplier * metagene_parameters[gene, metagene] * metagene_parameters[gene, metagene] for gene in range(model.max_genes) for metagene in range(model.K)])

            metagene_model.setObjective(objective)
            metagene_model.optimize()

            model.M = np.array([[metagene_parameters[gene, metagene].x for metagene in range(model.K)] for gene in range(model.max_genes)])
            # TODO: Remove this condition (seems unnecessary)
            #  if model.M_constraint == 'sum2one':
            #     pass
            # # elif M_sum2one == 'L1':
            # #   M /= np.abs(M).sum(0, keepdims=True) + 1e-10
            # # elif M_sum2one == 'L2':
            # #   M /= np.sqrt((M**2).sum(0, keepdims=True)) + 1e-10
            # else:
            #     raise NotImplementedError(f'Constraint on M {model.M_constraint} is not implemented')
        # TODO: do we need to keep the below code block if it currently ends in a NotImplementedError?
        else:
            YXTs = [(YT.T @ XT) * beta for YT, XT, beta in zip(model.YTs, model.XTs, model.betas)]
            objective_2s = []
            for XT, beta in zip(model.XTs, model.betas):
                XXT = XT.T @ XT * beta
                objective_2s.append(grb.quicksum([XXT[i,j]*metagene_parameters[i]*metagene_parameters[j] for i in range(model.K) for j in range(model.K)]))
            for gene_index, Mg in enumerate(model.M):
                objective = []
                for num_genes, YXT, XT, objective_2 in zip(model.Gs, YXTs, model.XTs, objective_2s):
                    if gene_index >= num_genes:
                        continue
                    objective.append(objective_2)
                    objective.append(grb.quicksum([-2*YXT[gene_index, i]*metagene_parameters[i] for i in range(model.K)]))
                metagene_model.setObjective(sum(objective, []), grb.GRB.MINIMIZE)
                metagene_model.optimize()
                Mg[:] = np.array([metagene_parameters[i].x for i in range(model.K)])
            raise NotImplementedError(f'Constraint on M {model.M_constraint} is not implemented')

        dM = model.M - last_M
        
        logging.info(
            f'{print_datetime()}'
            f'At iter {iteration}: '
            f'Diff M: max = {np.abs(dM).max():.2e}, '
            f'RMS = {np.sqrt(np.mean(np.abs(dM)**2)):.2e}, '
            f'mean = {np.abs(dM).mean():.2e}\t'
        )
        sys.stdout.flush()

        last_M = np.copy(model.M)
        last_rmse = rmse

    return model.M, model.XTs, model.sigma_yx_inverses, model.prior_x_parameter_sets

def initialize_M_by_kmeans(YTs, K, random_seed4kmeans=0, n_init=10):
    """Use k-means clustering for initial estimate of metagene matrix M.

    Args:
        YTs: A list of gene expression matrices, each with dimensions (num_individuals, num_genes)
        K: Inner-dimensionality of metagene matrix (i.e. number of metagenes desired)
        random_seed4kmeans:

    Returns:
        M_initial, the initial estimate for the metagene matrix, with dimensions (num_genes, K)
    """

    num_cells_list, Gs = zip(*[YT.shape for YT in YTs])
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
        XTs: list of initial weightings of metagenes for each replicate
        Es: list of adjacency lists for connectivity graph of each replicate
        betas: list of beta factors that weight each replicate
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
        for XT, adjacency_list, beta in zip(XTs, Es.values(), betas):
            t = np.zeros_like(sigma_x)
            for XTi, neighbors in zip(XT, adjacency_list.values()):
                t += np.outer(XTi, XT[neighbors].sum(0))
            # TODO: seems like sigma_x isn't used anywhere. Can this safely be commented out, then?
            # self.sigma_x += t * beta
        sigma_x /= np.dot(betas, [sum(map(len, E.values())) for E in Es.values()])
        sigma_x_inverse = np.linalg.inv(sigma_x)
        
        del sigma_x
        sigma_x_inverse *= factor
    else:
        raise NotImplementedError(f'Initialization for sigma_x_inverse {sigma_x_inverse_mode} is not implemented')
   
    return sigma_x_inverse 
