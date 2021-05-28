import sys
import time
import itertools
import resource
import logging
from multiprocessing import Pool, Process

from util import psutil_process, print_datetime, array2string

import torch
import numpy as np
import gurobipy as grb
from scipy.special import loggamma

def estimate_parameters_y(self, max_iterations=10):
    """Estimate model parameters that depend on Y, assuming a fixed value X = X_t.


    """
    
    def calculate_unscaled_sigma_yx_inverse(M, MTM, YT, YXT, XXT):
        """Calculated closed-form expression for sigma_yx_inverse.

        TODO: fill out docstring
        TODO: MTM is a placeholder for now; fix later
        """

        MTM = M.T @ M

        sigma_yx_inverse = np.dot(YT.ravel(), YT.ravel()) - 2*np.dot(YXT.ravel(), M.ravel()) + np.dot(XXT.ravel(), MTM.ravel())

        return sigma_yx_inverse

    logging.info(f'{print_datetime()}Estimating M and sigma_yx_inverse')

    YXTs = []
    XXTs = []
    sizes = np.fromiter(map(np.size, self.YTs), dtype=float)
    for YT, XT in zip(self.YTs, self.XTs):
        if self.dropout_mode == 'raw':
            YXTs.append(YT.T @ XT)
            XXTs.append(XT.T @ XT)
        else:
            raise NotImplementedError

    metagene_model = grb.Model('M')
    metagene_model.setParam('OutputFlag', False)
    metagene_model.Params.Threads = 1
    if self.M_constraint == 'sum2one':
        metagene_variables = metagene_model.addVars(self.max_genes, self.K, lb=0.)
        metagene_model.addConstrs((metagene_variables.sum('*', i) == 1 for i in range(self.K)))
    else:
        raise NotImplementedError

    for iteration in range(max_iterations):
        # Estimating M
        objective = 0
        for beta, YT, sigma_yx_inverse, YXT, XXT, G, XT in zip(self.betas, self.YTs, self.sigma_yx_inverses, YXTs, XXTs, self.Gs, self.XTs):
            # constant terms
            if self.dropout_mode == 'raw':
                flattened_YT = YT.ravel()
            else:
                raise NotImplementedError

            # TODO: do we need this? This component of objective does not depend on M
            objective += beta * sigma_yx_inverse**2 * np.dot(flattened_YT, flattened_YT)
            
            # linear terms - Adding terms for -2 y_i (M x_i)^\top
            factor = -2 * beta * sigma_yx_inverse**2 * YXT
            objective += grb.quicksum([factor[i, j] * metagene_variables[i, j] for i in range(G) for j in range(self.K)])
            
            # quadratic terms - Element-wise matrix multiplication (Mx_i)^\top(Mx_i)
            if self.dropout_mode == 'raw':
                factor = beta * sigma_yx_inverse**2 * XXT
                factor[np.diag_indices(self.K)] += 1e-5
                objective += grb.quicksum([factor[metagene, metagene] * metagene_variables[k, metagene] * metagene_variables[k, metagene] for k in range(G) for metagene in range(self.K)])
                factor *= 2
                objective += grb.quicksum([factor[metagene, j] * metagene_variables[k, metagene] * metagene_variables[k, j] for k in range(G) for metagene in range(self.K) for j in range(metagene+1, self.K)])
            else:
                raise NotImplementedError
        
        # TODO: what is this for? Is it regularization on the size of M? Do we need to keep it?
        # kk = 0
        # if kk != 0:
        #     objective += grb.quicksum([kk/2 * metagene_variables[k, i] * metagene_variables[k, i] for k in range(self.max_genes) for i in range(self.K)])

        metagene_model.setObjective(objective, grb.GRB.MINIMIZE)
        metagene_model.optimize()
        M = np.array([[metagene_variables[i, j].x for j in range(self.K)] for i in range(self.max_genes)])
        
        if self.M_constraint in ('sum2one', 'none'):
            pass
        elif self.M_constraint == 'L1':
            M /= np.abs(M).sum(0, keepdims=True)
        elif self.M_constraint == 'L2':
            M /= np.sqrt((M ** 2).sum(0, keepdims=True))
        else:
            raise NotImplementedError

        self.M = M

        # Estimating sigma_yx_inverses
        last_sigma_yx_inverses = np.copy(self.sigma_yx_inverses)
        unscaled_sigma_yx_inverses = np.array([calculate_unscaled_sigma_yx_inverse(self.M[:G], None, YT, YXT, XXT) for YT, YXT, XXT, G in zip(self.YTs, YXTs, XXTs, self.Gs)])

        if self.sigma_yx_inverse_mode == 'separate':
            sigma_yx_inverses = unscaled_sigma_yx_inverses / sizes
            self.sigma_yx_inverses = 1. / np.sqrt(sigma_yx_inverses)
        elif self.sigma_yx_inverse_mode == 'average':
            sigma_yx_inverses = np.dot(self.betas, unscaled_sigma_yx_inverses) / np.dot(self.betas, sizes)
            self.sigma_yx_inverses = np.full(self.num_repli, 1 / (np.sqrt(sigma_yx_inverses) + 1e-20))
        elif self.sigma_yx_inverse_mode.startswith('average '):
            index = np.array(list(map(int, self.sigma_yx_inverse_mode.split(' ')[1:])))
            sigma_yx_inverses = np.dot(self.betas[index], unscaled_sigma_yx_inverses[index]) / np.dot(self.betas[index], sizes[index])
            self.sigma_yx_inverses = np.full(self.num_repli, 1 / (np.sqrt(sigma_yx_inverses) + 1e-20))
        else:
            raise NotImplementedError

        delta = self.sigma_yx_inverses - last_sigma_yx_inverses
        logging.info(f"{print_datetime()}At iteration {iteration}, σ_yxInv: {array2string(delta)} -> {array2string(self.sigma_yx_inverses)}")

        if (np.abs(delta) / self.sigma_yx_inverses).max() < 1e-5 or self.num_repli <= 1 or self.sigma_yx_inverse_mode.startswith('average'):
            break

    # emission
    Q_Y = -np.dot(self.betas, sizes) / 2

    # TODO: how does this Q score calculation part work?
    # log of partition function - Pr [ Y | X, Theta ]
    Q_Y -= np.dot(self.betas, sizes) * np.log(2*np.pi) / 2
    Q_Y += (sizes * self.betas * np.log(self.sigma_yx_inverses)).sum()

    return Q_Y

def estimate_parameters_x(self, torch_dtype=torch.double, requires_grad=False, max_torch_iterations=1000, iterations_per_epoch=100): 
    """Estimate model parameters that depend on X, 

    Can operate in two modes: either using PyTorch's backward method, or using custom gradients.

    .. math::
                

    """
    logging.info(f'{print_datetime()}Estimating sigma_x_inverse and prior_x_parameter_sets')

    average_metagene_expressions = []
    # average_metagene_expression_es = []
    sigma_x_inverse_gradient = torch.zeros([self.K, self.K], dtype=torch_dtype, device=self.device)
    z_j_sums = []
    for YT, adjacency_list, XT, beta in zip(self.YTs, self.Es, self.XTs, self.betas):
        XT = torch.tensor(XT, dtype=torch_dtype, device=self.device)
        N, G = YT.shape
        average_metagene_expressions.append(XT.sum(axis=0))
        
        # average_metagene_expression_e = torch.tensor([len(neighbor_list) for neighbor_list in adjacency_list], dtype=torch_dtype, device=self.device) @ XT
        # average_metagene_expression_es.append(average_metagene_expression_e)

        # Normalizing XT
        ZT = XT / XT.sum(axis=1, keepdim=True).add(1e-30)
       
        # Each row of z_j_sum is the sum of the z_j of its neighbors
        z_j_sum = torch.empty([N, self.K], dtype=torch_dtype, device=self.device)
        for index, neighbor_list in enumerate(adjacency_list):
            z_j_sum[index] = ZT[neighbor_list].sum(axis=0)
        z_j_sums.append(z_j_sum)

        sigma_x_inverse_gradient = sigma_x_inverse_gradient.addmm(alpha=beta, mat1=ZT.t(), mat2=z_j_sum)

    Q_X = 0
    if all(prior_x_mode == 'Gaussian' for prior_x_mode, *_ in self.prior_x_parameter_sets) and self.pairwise_potential_mode == 'linear':
        raise NotImplementedError
    elif self.pairwise_potential_mode in ('linear', 'linear w/ shift'):
        raise NotImplementedError
    elif all(prior_x_mode in ('Exponential shared', 'Exponential shared fixed') for prior_x_mode, *_ in self.prior_x_parameter_sets) and self.pairwise_potential_mode == 'normalized':
        # Calculating Q_X and potentially updating priors on X values
        new_prior_x_parameter_sets = []
        loggamma_K = loggamma(self.K)
        n_cache = 2**14
        precomputed_log_gamma = torch.tensor(loggamma(np.arange(1, n_cache)), dtype=torch_dtype, device=self.device)
        for num_genes, (prior_x_mode, *prior_x_parameters), average_metagene_expression in zip(self.Ns, self.prior_x_parameter_sets, average_metagene_expressions):
            if prior_x_mode == 'Exponential shared':
                lambda_x = (average_metagene_expression.mean() / num_genes).pow(-1).cpu().data.numpy()
                Q_X -= lambda_x * average_metagene_expression.sum().cpu().data.numpy()
                Q_X += num_genes * self.K * np.log(lambda_x) - num_genes * loggamma_K
                prior_x_parameter_set = (prior_x_mode, np.full(self.K, lambda_x))
                new_prior_x_parameter_sets.append(prior_x_parameter_set)
            elif prior_x_mode == 'Exponential shared fixed':
                lambda_x, = prior_x_parameters
                Q_X -= lambda_x.mean() * average_metagene_expression.sum().cpu().data.numpy()
                new_prior_x_parameter_sets.append((prior_x_mode, *prior_x_parameters))
            else:
                raise NotImplementedError

        self.prior_x_parameter_sets = new_prior_x_parameter_sets

        # Skip sigma_x_inverse estimation if not using spatial information
        if sum(self.total_edge_counts) == 0:
            return Q_X
        
        # optimizers = []
        # schedulers = []
        sigma_x_inverse = torch.tensor(self.sigma_x_inverse, dtype=torch_dtype, device=self.device, requires_grad=requires_grad)

        optimizer = torch.optim.Adam([sigma_x_inverse], lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, iterations_per_epoch, gamma=0.98)
        # optimizers.append(optimizer)
        # schedulers.append(scheduler)
        # del optimizer, scheduler

        variable_list = [sigma_x_inverse]
        tprior_x_parameter_sets = []
        for prior_x_mode, *prior_x_parameters in self.prior_x_parameter_sets:
            if prior_x_mode in ('Exponential shared', 'Exponential shared fixed'):
                lambda_x, = prior_x_parameters
                lambda_x = torch.tensor(lambda_x, dtype=torch_dtype, device=self.device, requires_grad=requires_grad)
                tprior_x_parameter_sets.append((prior_x_mode, lambda_x,))
                variable_list.append(lambda_x)
                del lambda_x
            else:
                raise NotImplementedError
        
        for variable in variable_list:
            variable.grad = torch.zeros_like(variable)

        # tdiagXXTs = [torch.tensor(np.diag(XT).copy(), dtype=torch_dtype, device=self.device) for XT in XXTs]
        # tNus = [z_j_sum.sum(axis=0) for z_j_sum in z_j_sums]
        # tNu2s = [z_j_sum.t() @ z_j_sum for z_j_sum in z_j_sums]
        # average_metagene_expression_e_all = torch.zeros_like(average_metagene_expression_es[0])
        # for beta, average_metagene_expression_e in zip(self.betas, average_metagene_expression_es):
        #     average_metagene_expression_e_all.add_(alpha=beta, other=average_metagene_expression_e)
        
        # total_edge_counts = [sum(map(len, E)) for E in self.Es]
        adjacency_counts = [torch.tensor(list(map(len, E)), dtype=torch_dtype, device=self.device) for E in self.Es]
        # tZTs = [torch.tensor(XT, dtype=torch_dtype, device=self.device) for XT in self.XTs]
        # for tZT in tZTs:
        #     tZT.div_(tZT.sum(axis=1, keepdim=True))

        # sigma_x_inverse_ub = 1.
        # sigma_x_inverse_lb = -1.
        sigma_x_inverse_lb = None
        sigma_x_inverse_ub = None
        sigma_x_inverse_constraint = None               # complete matrix
        # sigma_x_inverse_constraint = 'diagonal'           # diagonal matrix
        # sigma_x_inverse_constraint = 'diagonal same'  # diagonal matrix, diagonal values are all the same

        # row_idx, col_idx = np.triu_indices(self.K, 0)

        # tZes = [None] * self.num_repli
        
        num_samples = 64

        objective, last_objective = None, torch.empty([], dtype=torch_dtype, device=self.device).fill_(np.nan)
        best_objective, best_iteration = torch.empty([], dtype=torch_dtype, device=self.device).fill_(np.nan), -1
        sigma_x_inverse_best = None
        has_converged = False
        for torch_iteration in range(max_torch_iterations + 1):
            beginning_of_epoch = (torch_iteration % iterations_per_epoch  == 0)
            if requires_grad:
                # for optimizer in optimizers:
                #     optimizer.zero_grad()
                optimizer.zero_grad()
            else:
                for variable in variable_list:
                    variable.grad.zero_()

            if sigma_x_inverse_lb:
                sigma_x_inverse.clamp_(min=sigma_x_inverse_lb)
            if sigma_x_inverse_ub:
                sigma_x_inverse.clamp_(max=sigma_x_inverse_ub)
            if sigma_x_inverse_constraint in ('diagonal', 'diagonal same'):
                sigma_x_inverse = sigma_x_inverse.triu().tril()
            if sigma_x_inverse_constraint == 'diagonal same':
                sigma_x_inverse[(range(self.K), range(self.K))] = sigma_x_inverse[(range(self.K), range(self.K))].mean()

            objective = torch.zeros([], dtype=torch_dtype, device=self.device)
            objective_grad = torch.zeros([], dtype=torch_dtype, device=self.device, requires_grad=True)

            # pairwise potential
            sigma_x_inverse.grad += sigma_x_inverse_gradient
            if requires_grad:
                objective_grad = objective_grad + sigma_x_inverse_gradient.view(-1) @ sigma_x_inverse.view(-1)
            else:
                objective += sigma_x_inverse_gradient.view(-1) @ sigma_x_inverse.view(-1)

            for N, total_edge_count, adjacency_count, E, beta, z_j_sum, tprior_x in zip(self.Ns, self.total_edge_counts, adjacency_counts, self.Es, self.betas, z_j_sums, tprior_x_parameter_sets):
                
                if total_edge_count == 0:
                    continue

                if tprior_x[0] in ('Exponential shared', 'Exponential shared fixed'):
                    if beginning_of_epoch:
                        index = slice(None)
                    else:
                        index = np.random.choice(N, min(num_samples, N), replace=False)

                    z_j_sum = z_j_sum[index].contiguous()
                    edge_proportion = total_edge_count / adjacency_count[index].sum()
                    
                    # Z^z(\theta)
                    beta_i = z_j_sum @ sigma_x_inverse
                    beta_i.grad = torch.zeros_like(beta_i)
                    # torch.manual_seed(iteration)
                    # TODO: delete if unncessary logic
                    # if iteration > 1 or torch_iteration > 100:
                    #     # log_Z = integrateOfExponentialOverSimplexSampling(beta_i, requires_grad=requires_grad, seed=iteration*max_torch_iterations+torch_iteration)
                    #     log_Z = integrateOfExponentialOverSimplexInduction2(beta_i, grad=c, requires_grad=requires_grad, )
                    # else:
                    #     # log_Z = integrateOfExponentialOverSimplexSampling(beta_i, requires_grad=requires_grad, seed=iteration*max_torch_iterations+torch_iteration)
                    #     log_Z = integrateOfExponentialOverSimplexInduction2(beta_i, grad=c, requires_grad=requires_grad)
                    
                    # TODO: why are we precomputing log_gamma? It seems to take no time at all.
                    log_Z = integrate_over_simplex(beta_i, grad=edge_proportion, requires_grad=requires_grad, device=self.device, precomputed_log_gamma=precomputed_log_gamma)
                    
                    if requires_grad:
                        objective_grad = objective_grad.add(beta * edge_proportion, log_Z.sum())
                    else:
                        objective = objective.add(alpha=beta * edge_proportion, other=log_Z.sum())
                        sigma_x_inverse.grad = sigma_x_inverse.grad.addmm(alpha=beta, mat1=z_j_sum.t(), mat2=beta_i.grad)
                else:
                    raise NotImplementedError

            if requires_grad:
                objective_grad.backward()
                objective = objective + objective_grad

            # prior on Σ_x^inv
            # TODO: can we delete this?
            # num_burnin_iteration = 200
            # if iteration <= num_burnin_iteration:
            #   kk = 1e-1 * np.dot(betas, list(map(len, Es))) * 1e-1**((num_burnin_iteration-iteration+1)/num_burnin_iteration)
            # else:
            #   kk = 1e-1 * np.dot(betas, list(map(len, Es)))

            weighted_total_edge_count = np.dot(self.betas, self.total_edge_counts)
            regularization_factor = self.lambda_sigma_x_inverse * weighted_total_edge_count
            sigma_x_inverse.grad += regularization_factor * sigma_x_inverse
            objective += regularization_factor / 2 * sigma_x_inverse.pow(2).sum()

            # normalize gradient by the weighted sizes of data sets
            sigma_x_inverse.grad.div_(weighted_total_edge_count)
            objective.div_(np.dot(self.betas, self.Ns))

            sigma_x_inverse.grad.add_(sigma_x_inverse.grad.clone().t()).div_(2)

            if sigma_x_inverse_lb:
                sigma_x_inverse.grad[(sigma_x_inverse <= sigma_x_inverse_lb) * (sigma_x_inverse.grad > 0)] = 0
            if sigma_x_inverse_ub:
                sigma_x_inverse.grad[(sigma_x_inverse >= sigma_x_inverse_ub) * (sigma_x_inverse.grad < 0)] = 0
            if sigma_x_inverse_constraint in ['diagonal', 'diagonal same']:
                sigma_x_inverse.grad.triu_().tril_()
            if sigma_x_inverse_constraint in ['diagonal same']:
                sigma_x_inverse.grad[(range(self.K), range(self.K))] = sigma_x_inverse.grad[(range(self.K), range(self.K))].mean()

            pseudovalue = 1e-1
            threshold = 1e-2
            normalized_gradient_magnitude = (sigma_x_inverse.grad.abs() / (sigma_x_inverse.abs() + pseudovalue)).abs().max().item()
            has_converged = (normalized_gradient_magnitude < threshold)
            
            for tprior_x in tprior_x_parameter_sets:
                if tprior_x[0] in ('Exponential shared',):
                    lambda_x, = tprior_x[1:]
                    has_converged &= lambda_x.grad.abs().max().item() < 1e-4
                    del lambda_x
                elif tprior_x[0] in ('Exponential shared fixed',):
                    pass
                else:
                    raise NotImplementedError
            
            has_converged &= not (objective <= last_objective - 1e-3 * iterations_per_epoch)
            has_converged |= not torch_iteration < best_iteration + 2*iterations_per_epoch
            
            display_warning = ((objective > last_objective + 1e-10) and beginning_of_epoch)
            # display_warning = True

            if display_warning:
                print('Warning: objective value has increased.')

            if beginning_of_epoch or torch_iteration == max_torch_iterations or display_warning:
                print(
                    f'At iteration {torch_iteration},\t'
                    f'objective = {(objective - last_objective).item():.2e} -> {objective.item():.2e}\t'
                    f'Σ_x^inv: {sigma_x_inverse.max().item():.1e} - {sigma_x_inverse.min().item():.1e} = {sigma_x_inverse.max() - sigma_x_inverse.min():.1e} '
                    f'grad = {sigma_x_inverse.grad.min().item():.2e} {sigma_x_inverse.grad.max().item():.2e}\t'
                    f'var/grad = {normalized_gradient_magnitude:.2e}'
                    # f'δ_x: {tdelta_x.max().item():.1e} - {tdelta_x.min().item():.1e} = {tdelta_x.max() - tdelta_x.min():.1e} '
                    # f'grad = {tdelta_x.grad.min().item():.2e} {tdelta_x.grad.max().item():.2e}'
                )
                sys.stdout.flush()

            # for optimizer in optimizers:
            #     optimizer.step()
            # for scheduler in schedulers:
            #     scheduler.step()
            
            optimizer.step()
            scheduler.step()
            
            # TODO: Do we need to enforce this?
            if beginning_of_epoch:
                last_objective = objective
                if not best_objective <= objective:
                    best_objective, best_iteration = objective, torch_iteration
                    sigma_x_inverse_best = sigma_x_inverse.clone().detach()

                if has_converged:
                    break


        sigma_x_inverse, objective = sigma_x_inverse_best, best_objective
        self.sigma_x_inverse = sigma_x_inverse.cpu().data.numpy()

        objective *= np.dot(self.betas, self.Ns)
        Q_X -= objective.item()

    elif all(prior_x[0] == 'Exponential' for prior_x in self.prior_x_parameter_sets) and self.pairwise_potential_mode == 'normalized':
        raise NotImplementedError
    else:
        raise NotImplementedError

    return Q_X

def integrate_over_simplex(beta_i, grad=None, requires_grad=False, device='cpu', precomputed_log_gamma=None):
    """Approximate the integral of the partition function over the simplex using Taylor approximation.

    Todo:
        Figure out how this works.

    Args:
        beta_i:
        grad:
        required_grad:
        device:
    
    Returns:
        An array of approximate values for the log of the Z component of the partition function (log Z_i^z(\Theta))
    """
    
    num_cells, num_metagenes = beta_i.shape
    
    if grad is None:
        grad = torch.tensor([1.], dtype=torch.double, device=device)

    if precomputed_log_gamma is None:
        precomputed_log_gamma = torch.tensor(loggamma(np.arange(1, n_cache)), dtype=torch.double, device=device)

    beta_i_offset = beta_i.max(axis=-1, keepdim=True)[0] + 1e-5
    sigma_x_inverse_range = (beta_i.max() - beta_i.min()).item()
    num_taylor_terms = int(max(sigma_x_inverse_range+10, sigma_x_inverse_range*1.1))

    log_gamma = precomputed_log_gamma[num_metagenes-1: num_metagenes+num_taylor_terms-1]

    if requires_grad:
        beta_i = beta_i - beta_i_offset
        beta_i = beta_i.neg()
        # beta_i = beta_i.sort()[0]

        f = torch.zeros([num_cells, num_metagenes], dtype=torch.double, device=device)
        integral = torch.zeros([num_taylor_terms, num_cells], dtype=torch.double, device=device)

        for degree in range(1, num_taylor_terms):
            f = f + beta_i.log()
            offset = f.max(-1, keepdim=True)[0]
            f = f.sub(offset).exp().cumsum(dim=-1).log().add(offset)
            integral[degree].copy_(f[:, -1])

        integral = integral.sub(log_gamma[:, None])
        offset = integral.max(0, keepdim=True)[0]
        integral = integral.sub(offset).exp().sum(0).log().add(offset.squeeze(0)).sub(beta_i_offset.squeeze(-1))
    else:
        beta_i_grad = beta_i.grad
        beta_i = -(beta_i - beta_i_offset)
        beta_i.grad = beta_i_grad

        integral = torch.empty(num_cells, dtype=torch.double, device=device)

        # Operate on chunks of 32 cells at a time
        chunk_size = 32
        for beta_i_chunk, beta_i_grad_chunk, integral_chunk in zip(beta_i.split(chunk_size, 0), beta_i.grad.split(chunk_size, 0), integral.split(chunk_size, 0)):
            actual_chunk_size = len(beta_i_chunk)
            log_beta_i_chunk = beta_i_chunk.log()
            taylor_terms = torch.zeros([num_taylor_terms, actual_chunk_size], dtype=torch.double, device=device)
            gradient = torch.full([num_taylor_terms, actual_chunk_size, num_metagenes], -np.inf, dtype=torch.double, device=device)
            taylor_term = torch.zeros([actual_chunk_size, num_metagenes], dtype=torch.double, device=device)
            taylor_term_gradient = torch.full([actual_chunk_size, num_metagenes, num_metagenes], -np.inf, dtype=torch.double, device=device)
            for degree in range(1, num_taylor_terms):
                taylor_term_gradient += log_beta_i_chunk[:, None, :]
                
                # Stepping with stride num_metagenes+1 retrieves the diagonal from the gradient tensor
                gradient_diagonal = taylor_term_gradient.view(actual_chunk_size, num_metagenes**2)[:, ::num_metagenes+1]
                
                offset = torch.max(gradient_diagonal, taylor_term)
                offset[offset == -np.inf] = 0
                assert (offset != -np.inf).all()
                gradient_diagonal.copy_(((gradient_diagonal - offset).exp() + (taylor_term - offset).exp()).log() + offset)

                offset = taylor_term_gradient.max(-1, keepdim=True)[0]
                assert (offset != -np.inf).all()
                taylor_term_gradient = (taylor_term_gradient - offset).exp().cumsum(dim=-1).log() + offset
                gradient[degree] = taylor_term_gradient[:, :, -1]

                taylor_term += log_beta_i_chunk
                offset = taylor_term.max(dim=-1, keepdim=True)[0]
                taylor_term = (taylor_term - offset).exp().cumsum(dim=-1).log() + offset
                taylor_terms[degree] = taylor_term[:, -1]

            taylor_terms -= log_gamma[:, None]
            offset = taylor_terms.max(0, keepdim=True)[0]
            taylor_terms = (taylor_terms - offset).exp()
            integral_chunk.copy_(taylor_terms.sum(dim=0).log() + offset.squeeze(dim=0))

            gradient -= log_gamma[:, None, None]
            beta_i_grad_chunk -= grad * (gradient - integral_chunk[None, :, None]).exp().sum(dim=0)

        integral -= beta_i_offset.squeeze(-1)
        beta_i.neg_().add_(beta_i_offset)

    return integral
