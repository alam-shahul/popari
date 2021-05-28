import sys, logging, time, resource, gc, os
import multiprocessing
from multiprocessing import Pool
from util import print_datetime

import numpy as np
import gurobipy as grb
import torch

def estimate_weights_no_neighbors(YT, M, XT, prior_x_parameter_set, sigma_yx_inverse, X_constraint, dropout_mode, replicate):
    """Estimate weights for a single replicate in the SpiceMix model without considering neighbors.

    This is essentially a benchmarking convenience function, and should return similar results to running vanilla NMF.

    Args:
        YT: transpose of gene expression matrix for sample, with shape (num_cells, num_genes)
        M: current estimate of metagene matrix, with shape (num_genes, num_metagenes)
        XT: transpose of metagene weights for sample, with shape
    Returns:
        New estimate of transposed metagene weight matrix XT.
    """

    if dropout_mode != 'raw':
        raise NotImplemented

    logging.info(f'{print_datetime()}Estimating weights without neighbors in repli {replicate}')
    _, num_metagenes = XT.shape

    updated_XT = np.zeros_like(XT)
    weight_model = grb.Model('X w/o n')
    weight_model.setParam('OutputFlag', False)
    weight_model.Params.Threads = 1
    weight_variables = weight_model.addVars(num_metagenes, lb=0.)
    assert X_constraint == 'none'

    # Adding shared components of the objective
    # quadratic term in log Pr[ Y | X, Theta ]
    shared_objective = 0
    if dropout_mode == 'raw':
        # MTM = M.T @ M * (sigma_yx_inverse**2 / 2.)
        MTM = (M.T @ M + 1e-6 * np.eye(num_metagenes)) * (sigma_yx_inverse ** 2 / 2.)
        shared_objective += grb.quicksum([weight_variables[index] * MTM[index, index] * weight_variables[index] for index in range(num_metagenes)])
        MTM *= 2
        shared_objective += grb.quicksum([weight_variables[index] * MTM[index, j] * weight_variables[j] for index in range(num_metagenes) for j in range(index+1, num_metagenes)])
        
        del MTM
        YTM = YT @ M * (-sigma_yx_inverse ** 2)
    else:
        raise NotImplementedError

    # prior on X
    prior_x_mode, *prior_x_parameters = prior_x_parameter_set
    if prior_x_mode in ('Truncated Gaussian', 'Gaussian'):
        mu_x, sigma_x_inv = prior_x_parameters
        assert (sigma_x_inv > 0).all()
        t = sigma_x_inv ** 2 / 2
        shared_objective += grb.quicksum([t[metagene] * weight_variables[metagene] * weight_variables[metagene] for metagene in range(num_metagenes)])
        t *= - 2 * mu_x
        shared_objective += grb.quicksum([t[metagene] * weight_variables[metagene] for metagene in range(num_metagenes)])
        shared_objective += np.dot(mu_x**2, sigma_x_inv**2) / 2
    elif prior_x_mode in ('Exponential', 'Exponential shared', 'Exponential shared fixed'):
        lambda_x, = prior_x_parameters
        assert (lambda_x >= 0).all()
        shared_objective += grb.quicksum([lambda_x[metagene] * weight_variables[metagene] for metagene in range(num_metagenes)])
    else:
        raise NotImplementedError

    for cell_index, (y, yTM) in enumerate(zip(YT, YTM)):
        objective = shared_objective + grb.quicksum(yTM[metagene] * weight_variables[metagene] for metagene in range(num_metagenes)) + np.dot(y, y) * sigma_yx_inverse / 2.
        weight_model.setObjective(objective, grb.GRB.MINIMIZE)
        weight_model.optimize()
        updated_XT[cell_index] = [weight_variables[metagene].x for metagene in range(num_metagenes)]

    return updated_XT

def estimate_weights_icm(YT, E, M, XT, prior_x_parameter_set, sigma_yx_inverse, sigma_x_inverse, X_constraint, dropout_mode, pairwise_potential_mode, replicate):
    r"""Estimate weights for a single replicate in the SpiceMix model using the Iterated Conditional Model (ICM).

    Notes:
        .. math::
           \hat{X}_{\text{MAP}} &= \mathop{\text{\argmax}}_{X \in \mathbb{R}_+^{K \times N}} \left{ \sum_{i \in \mathcal{V}}\right} \\

           s_i &= \frac{ - \lambda_x^\top z_i}{(Mz_i)^\top Mz_i} \\
           z_i &= \frac{}{}

        We write XT in terms of size factors S such that XT = S * ZT.

    Args:
        YT: transpose of gene expression matrix for replicate, with shape (num_cells, num_genes)
        E: adjacency list for neighborhood graph in this replicate
        M: current estimate of metagene matrix, with shape (num_genes, num_metagenes)
        XT: transpose of weight matrix, with shape (num_cells, num_metagenes)
        prior_x_parameter_set: set of parameters defining prior distribution on weights, with structure (prior_x_mode, ∗prior_x_parameters)
        sigma_yx_inverse: TODO
        sigma_x_inverse: inverse of metagene affinity matrix
        X_constraint: constraint on elements of weight matrix
        dropout_mode: TODO:
        pairwise_potential_mode: TODO

    Returns:
        New estimate of transposed metagene weight matrix XT.
    """

    prior_x_mode, *prior_x_parameters = prior_x_parameter_set
    num_cells, _ = YT.shape
    _, num_metagenes = M.shape
    MTM = None
    YTM = None
    
    # Precomputing some important matrix products
    if dropout_mode == 'raw':
        MTM = M.T @ M * sigma_yx_inverse**2 / 2
        YTM = YT @ M * sigma_yx_inverse**2 / 2
    else:
        raise NotImplementedError

    def calculate_objective(S, ZT):
        """Calculate current value of ICM objective.

        Args:
            YT: transpose of gene expression matrix for a particular sample
            S: a vector of total metagene expressions for each cell
            ZT: current estimate of weights for the sample, divided by the total for each cell

        Returns:
            value of ICM objective
        """
        
        objective = 0
        difference = YT - ( S * ZT ) @ M.T
        if dropout_mode == 'raw':
            difference = difference.ravel()
        else:
            raise NotImplementedError

        objective += np.dot(difference, difference) * sigma_yx_inverse**2 / 2
        if pairwise_potential_mode == 'normalized':
            for neighbors, z_i in zip(E, ZT):
                objective += z_i @ sigma_x_inverse @ ZT[neighbors].sum(axis=0) / 2
        else:
            raise NotImplementedError

        if prior_x_mode in ('Exponential', 'Exponential shared', 'Exponential shared fixed'):
            lambda_x, = prior_x_parameters
            objective += lambda_x @ (S * ZT).sum(axis=0)
            del lambda_x
        else:
            raise NotImplementedError

        objective /= YT.size

        return objective

    def update_s_i(z_i, yTM):
        """Calculate closed form update for s_i.

        Assuming fixed value of z_i, update for s_i takes the following form:
        TODO

        Args:
            z_i: current estimate of normalized metagene expression
            neighbors: list of neighbors of current cell
            yTM: row of YTM corresponding to current cell
            MTM: row of MTM corresponding to current cell

        Returns:
            Updated estimate of s_i

        """

        denominator =  z_i @ MTM @ z_i
        numerator = yTM @ z_i
        if prior_x_mode in ('Exponential', 'Exponential shared', 'Exponential shared fixed'):
            lambda_x, = prior_x_parameters
            # TODO: do we need the 1/2 here?
            numerator -= lambda_x @ z_i / 2
            del lambda_x
        else:
            raise NotImplementedError
        
        numerator = np.maximum(numerator, 0)
        s_i_new = numerator / denominator

        return s_i_new

    def update_z_i(s_i, y_i, yTM, eta):
        """Calculate update for z_i using Gurobi simplex algorithm.

        Assuming fixed value of s_i, update for z_i is a linear program of the following form:
        TODO

        Args:
            s_i: current estimate of size factor
            yTM: row of YTM corresponding to current cell
            eta: aggregate contribution of neighbor z_j's, weighted by affinity matrix (sigma_x_inverse)

        Returns:
            Updated estimate of z_i

        """

        objective = 0

        # Element-wise matrix multiplication (Mz_is_i)^\top(Mz_is_i)
        factor = s_i**2 * MTM
        objective += grb.quicksum([weight_variables[index] * factor[index, index] * weight_variables[index] for index in range(num_metagenes)])
        factor *= 2
        objective += grb.quicksum([weight_variables[index] * factor[index, j] * weight_variables[j] for index in range(num_metagenes) for j in range(index+1, num_metagenes)])
       
       
        # Adding terms for -2 y_i M z_i s_i
        factor = -2 * s_i * yTM
        # TODO: fix formula below
        # objective += grb.quicksum([weight_variables[index] * factor[index] for index in range(num_metagenes)])
        # objective += y_i @ y_i
        # objective *= sigma_yx_inverse**2 / 2
        factor += eta
        # factor = eta

        if prior_x_mode in ('Exponential'):
            lambda_x, = prior_x_parameters
            factor += lambda_x * s_i
            del lambda_x
        elif prior_x_mode in ('Exponential shared', 'Exponential shared fixed'):
            pass
        else:
            raise NotImplementedError

        objective += grb.quicksum([weight_variables[index] * factor[index] for index in range(num_metagenes)])
        # TODO: is this line necessary? Doesn't seem like z_i affects this term of the objective
        objective += y_i @ y_i * sigma_yx_inverse**2 / 2
        weight_model.setObjective(objective, grb.GRB.MINIMIZE)
        weight_model.optimize()

        z_i_new = np.array([weight_variables[index].x for index in range(num_metagenes)])

        return z_i_new

    global_iterations = 100
    local_iterations = 100

    weight_model = grb.Model('ICM')
    weight_model.Params.OutputFlag = False
    weight_model.Params.Threads = 1
    weight_model.Params.BarConvTol = 1e-6
    weight_variables = weight_model.addVars(num_metagenes, lb=0.)
    weight_model.addConstr(weight_variables.sum() == 1)

    S = XT.sum(axis=1, keepdims=True)
    ZT = XT / (S +  1e-30)

    last_objective = calculate_objective(S, ZT)
    best_objective, best_iteration = last_objective, -1

    for global_iteration in range(global_iterations):
        last_ZT = np.copy(ZT)
        last_S = np.copy(S)

        locally_converged = False
        if pairwise_potential_mode == 'normalized':
            for index, (neighbors, y_i, yTM, z_i, s_i) in enumerate(zip(E, YT, YTM, ZT, S)):
                eta = ZT[neighbors].sum(axis=0) @ sigma_x_inverse
                for local_iteration in range(local_iterations):
                    s_i_new = update_s_i(z_i, yTM) 
                    s_i_new = np.maximum(s_i_new, 1e-15)
                    delta_s_i = s_i_new - s_i
                    s_i = s_i_new

                    z_i_new = update_z_i(s_i, y_i, yTM, eta)
                    delta_z_i = z_i_new - z_i
                    z_i = z_i_new
                    
                    locally_converged |= (np.abs(delta_s_i) / (s_i + 1e-15) < 1e-3 and np.abs(delta_z_i).max() < 1e-3)
                    if locally_converged:
                        break

                if not locally_converged:
                    logging.warning(f'Cell {i} in the {replicate}-th replicate did not converge in {local_iterations} iterations;\ts = {s:.2e}, delta_s_i = {delta_s_i:.2e}, max delta_z_i = {np.abs(delta_z_i).max():.2e}')

                ZT[index] = z_i
                S[index] = s_i
        else:
            raise NotImplementedError

        globally_converged = False

        dZT = ZT - last_ZT
        dS = S - last_S
        current_objective = calculate_objective(S, ZT)

        globally_converged |= (np.abs(dZT).max() < 1e-2 and np.abs(dS / (S + 1e-15)).max() < 1e-2 and current_objective > last_objective - 1e-4) 

        # TODO: do we need to keep this?
        force_show_flag = False
        # force_show_flag |= np.abs(dZT).max() > 1-1e-5

        if global_iteration % 5 == 0 or globally_converged or force_show_flag:
            print(f'>{replicate} current_objective at iteration {global_iteration} = {current_objective:.2e},\tdiff = {np.abs(dZT).max():.2e}\t{np.abs(dS).max():.2e}\t{current_objective - last_objective:.2e}')

            print(
                f'ZT summary statistics: '
                f'# <0 = {(ZT < 0).sum().astype(np.float) / num_cells:.1f}, '
                f'# =0 = {(ZT == 0).sum().astype(np.float) / num_cells:.1f}, '
                f'# <1e-10 = {(ZT < 1e-10).sum().astype(np.float) / num_cells:.1f}, '
                f'# <1e-5 = {(ZT < 1e-5).sum().astype(np.float) / num_cells:.1f}, '
                f'# <1e-2 = {(ZT < 1e-2).sum().astype(np.float) / num_cells:.1f}, '
                f'# >1e-1 = {(ZT > 1e-1).sum().astype(np.float) / num_cells:.1f}'
            )

            print(
                f'S summary statistics: '
                f'# 0 = {(S == 0).sum()}, '
                f'min = {S.min():.1e}, '
                f'max = {S.max():.1e}'
            )

            sys.stdout.flush()

        # TODO: do we need this assertion still?
        assert not current_objective > last_objective + 1e-6
        last_objective = current_objective
        if current_objective < best_objective:
            best_objective, best_iteration = current_objective, global_iteration

        if globally_converged:
            break

    del weight_model

    # Enforce positivity constraint on S
    XT = np.maximum(S, 1e-15) * ZT
    
    return XT
