import sys, logging, time, resource, gc, os
import multiprocessing
from multiprocessing import Pool
from util import print_datetime

import numpy as np
import gurobipy as grb
import torch

def estimate_weights_no_neighbors(YT, M, XT, prior_x_parameter_set, sigma_yx_inverse, X_constraint, dropout_mode, replicate):
    """Estimate weights for a single replicate in the total_metagene_expressionspiceMix model without considering neighbors.

    This is essentially a benchmarking convenience function, and should return similar results to running vanilla NMF

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
        shared_objective += grb.quicksum([weight_variables[index] * MTM[index, i] * weight_variables[index] for i in range(num_metagenes)])
        MTM *= 2
        shared_objective += grb.quicksum([weight_variables[index] * MTM[index, j] * weight_variables[j] for i in range(num_metagenes) for j in range(i+1, num_metagenes)])
        
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
        objective = shared_objective + grb.quicksum(yTM[index]*weight_variables[metagene] for metagene in range(num_metagenes)) + np.dot(y, y) * sigma_yx_inverse / 2.
        weight_model.setObjective(objective, grb.GRB.MINIMIZE)
        weight_model.optimize()
        updated_XT[cell_index] = [weight_variables[metagene].x for metagene in range(num_metagenes)]

    del weight_model

    return updated_XT


def estimate_weights_icm(YT, E, M, XT, prior_x_parameter_set, sigma_yx_inverse, sigma_x_inverse, X_constraint, dropout_mode, pairwise_potential_mode, replicate):
    """Estimate weights for a single replicate in the SpiceMix model using the Iterated Conditional Model (ICM).

    Args:
        YT: transpose of gene expression matrix for replicate, with shape (num_cells, num_genes)
        E: adjacency list for neighborhood graph in this replicate
        M: current estimate of metagene matrix, with shape (num_genes, num_metagenes)
        XT: transpose of weight matrix, with shape (num_cells, num_metagenes)
        prior_x_parameter_set: set of parameters defining prior distribution on weights, with structure (prior_x_mode, *prior_x_parameters)
        sigma_yx_inverse: TODO
        sigma_x_inverse: inverse of metagene affinity matrix
        X_constraint: constraint on elements of weight matrix
        dropout_mode: TODO:
        pairwise_potential_mode: TODO

    Returns:
        New estimate of transposed metagene weight matrix XT.
    """

    def calculate_objective(total_metagene_expressions, normalized_XT):
        """Calculate current value of ICM objective.

        Args:
            YT: transpose of gene expression matrix for a particular sample
            total_metagene_expressions: a vector of (conserved) total metagene expressions for each cell
            normalized_XT: current estimate of weights for the sample, divided by the total for each cell

        Returns:
            value of ICM objective
        """
        
        objective = 0
        difference = YT - ( total_metagene_expressions * normalized_XT ) @ M.T
        if dropout_mode == 'raw':
            difference = difference.ravel()
        else:
            raise NotImplementedError

        objective += np.dot(difference, difference) * sigma_yx_inverse**2 / 2
        if pairwise_potential_mode == 'normalized':
            for neighbors, normalized_cell_metagene_expression in zip(E, normalized_XT):
                objective += normalized_cell_metagene_expression @ sigma_x_inverse @ normalized_XT[neighbors].sum(axis=0) / 2
        else:
            raise NotImplementedError

        prior_x_mode, *prior_x_parameters = prior_x_parameter_set
        if prior_x_mode in ('Exponential', 'Exponential shared', 'Exponential shared fixed'):
            lambda_x, = prior_x_parameters
            objective += lambda_x @ (total_metagene_expressions * normalized_XT).sum(axis=0)
            del lambda_x
        else:
            raise NotImplementedError

        objective /= YT.size

        return objective

    num_cells, _ = YT.shape
    _, num_metagenes = M.shape
    MTM = None
    YTM = None

    if dropout_mode == 'raw':
        MTM = M.T @ M * sigma_yx_inverse**2 / 2
        YTM = YT @ M * sigma_yx_inverse**2 / 2
    else:
        raise NotImplementedError

    max_iter = 100
    max_iter_individual = 100

    weight_model = grb.Model('ICM')
    weight_model.setParam('OutputFlag', False)
    weight_model.Params.Threads = 1
    weight_model.Params.BarConvTol = 1e-6
    weight_variables = weight_model.addVars(num_metagenes, lb=0.)
    weight_model.addConstr(weight_variables.sum() == 1)

    total_metagene_expressions = XT.sum(axis=1, keepdims=True)
    normalized_XT = XT / (total_metagene_expressions +  1e-30)

    last_objective = calculate_objective(total_metagene_expressions, normalized_XT)
    best_objective, best_iteration = last_objective, -1

    for iiter in range(max_iter):
        last_normalized_XT = np.copy(normalized_XT)
        last_total_metagene_expressions = np.copy(total_metagene_expressions)

        if pairwise_potential_mode == 'normalized':
            for index, (neighbors, y, yTM, normalized_cell_metagene_expression, total_cell_metagene_expression) in enumerate(zip(E, YT, YTM, normalized_XT, total_metagene_expressions)):
                eta = normalized_XT[neighbors].sum(axis=0) @ sigma_x_inverse
                for iiiter in range(max_iter_individual):
                    stop_flag = True

                    a = normalized_cell_metagene_expression @ MTM @ normalized_cell_metagene_expression
                    b = yTM @ normalized_cell_metagene_expression
                    if prior_x_parameter_set[0] in ['Exponential', 'Exponential shared', 'Exponential shared fixed']:
                        lambda_x, = prior_x_parameter_set[1:]
                        b -= lambda_x @ normalized_cell_metagene_expression / 2
                        del lambda_x
                    else:
                        raise NotImplementedError
                    b = np.maximum(b, 0)
                    s_new = b / a
                    s_new = np.maximum(s_new, 1e-15)
                    ds = s_new - total_cell_metagene_expression
                    stop_flag &= np.abs(ds) / (total_cell_metagene_expression + 1e-15) < 1e-3
                    total_cell_metagene_expression = s_new

                    objective = 0
                    t = total_cell_metagene_expression**2 * MTM
                    objective += grb.quicksum([weight_variables[index] * t[index, index] * weight_variables[index] for index in range(num_metagenes)])
                    t *= 2
                    objective += grb.quicksum([weight_variables[index] * t[index, j] * weight_variables[j] for index in range(num_metagenes) for j in range(index+1, num_metagenes)])
                    t = -2 * total_cell_metagene_expression * yTM
                    t += eta
                    if prior_x_parameter_set[0] in ['Exponential']:
                        lambda_x, = prior_x_parameter_set[1:]
                        t += lambda_x * total_cell_metagene_expression
                        del lambda_x
                    elif prior_x_parameter_set[0] in ['Exponential shared', 'Exponential shared fixed']:
                        pass
                    else:
                        raise NotImplementedError
                    objective += grb.quicksum([weight_variables[index] * t[index] for index in range(num_metagenes)])
                    objective += y @ y * sigma_yx_inverse**2 / 2
                    weight_model.setObjective(objective, grb.GRB.MINIMIZE)
                    weight_model.optimize()
                    normalized_cell_metagene_expression_new = np.array([weight_variables[index].x for index in range(num_metagenes)])
                    dnormalized_cell_metagene_expression = normalized_cell_metagene_expression_new - normalized_cell_metagene_expression
                    stop_flag &= np.abs(dnormalized_cell_metagene_expression).max() < 1e-3
                    normalized_cell_metagene_expression = normalized_cell_metagene_expression_new

                    if not stop_flag and iiiter == max_iter_individual-1:
                        logging.warning(f'Cell {i} in the {replicate}-th replicate did not converge in {max_iter_individual} iterations;\ts = {s:.2e}, ds = {ds:.2e}, max dz = {np.abs(dz).max():.2e}')

                    if stop_flag:
                        break

                normalized_XT[index] = normalized_cell_metagene_expression
                total_metagene_expressions[index] = total_cell_metagene_expression
        else:
            raise NotImplementedError

        stop_flag = True

        dnormalized_XT = normalized_XT - last_normalized_XT
        dtotal_metagene_expressions = total_metagene_expressions - last_total_metagene_expressions
        stop_flag &= np.abs(dnormalized_XT).max() < 1e-2
        stop_flag &= np.abs(dtotal_metagene_expressions / (total_metagene_expressions + 1e-15)).max() < 1e-2

        current_objective = calculate_objective(total_metagene_expressions, normalized_XT)

        stop_flag &= current_objective > last_objective - 1e-4

        force_show_flag = False
        # force_show_flag |= np.abs(dnormalized_XT).max() > 1-1e-5

        if iiter % 5 == 0 or stop_flag or force_show_flag:
            print(f'>{replicate} current_objective at iter {iiter} = {current_objective:.2e},\tdiff = {np.abs(dnormalized_XT).max():.2e}\t{np.abs(dtotal_metagene_expressions).max():.2e}\t{current_objective - last_objective:.2e}')

            print(
                f'stat of XT: '
                f'#<0 = {(normalized_XT < 0).sum().astype(np.float) / num_cells:.1f}, '
                f'#=0 = {(normalized_XT == 0).sum().astype(np.float) / num_cells:.1f}, '
                f'#<1e-10 = {(normalized_XT < 1e-10).sum().astype(np.float) / num_cells:.1f}, '
                f'#<1e-5 = {(normalized_XT < 1e-5).sum().astype(np.float) / num_cells:.1f}, '
                f'#<1e-2 = {(normalized_XT < 1e-2).sum().astype(np.float) / num_cells:.1f}, '
                f'#>1e-1 = {(normalized_XT > 1e-1).sum().astype(np.float) / num_cells:.1f}'
            )

            print(
                f'stat of s: '
                f'#0 = {(total_metagene_expressions == 0).sum()}, '
                f'min = {total_metagene_expressions.min():.1e}, '
                f'max = {total_metagene_expressions.max():.1e}'
            )

            sys.stdout.flush()

        # print(current_objective, last_objective)
        assert not current_objective > last_objective + 1e-6
        last_objective = current_objective
        if not current_objective >= best_objective:
            best_objective, best_iteration = current_objective, iiter

        if stop_flag:
            break

    del weight_model

    XT = np.maximum(total_metagene_expressions, 1e-15) * normalized_XT
    return XT
