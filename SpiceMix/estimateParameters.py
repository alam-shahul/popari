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

from sampleForIntegral import integrate_over_simplex

def estimateParametersY(self, max_iteration=10):
    """Estimate model parameters that depend on Y, assuming a fixed value X = X_t.


    """
    
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

    for iteration in range(max_iteration):
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
        def calculate_unscaled_sigma_yx_inverse(M, MTM, YT, YXT, XXT):
            """Calculated closed-form expression for sigma_yx_inverse.

            TODO: fill out docstring
            TODO: MTM is a placeholder for now; fix later
            """

            MTM = M.T @ M

            sigma_yx_inverse = np.dot(YT.ravel(), YT.ravel()) - 2*np.dot(YXT.ravel(), M.ravel()) + np.dot(XXT.ravel(), MTM.ravel())

            return sigma_yx_inverse
        
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

def estimateParametersX(self, iteration, torch_dtype=torch.double):
    """Estimate model parameters that depend on X, 

    Can operate in two modes: either using PyTorch's backward method, or using custom gradients.

    .. math::
                

    """

    logging.info(f'{print_datetime()}Estimating sigma_x_inverse and prior_x_parameter_sets')

    device = self.PyTorch_device

    XXTs = []
    alpha_tensors = []
    alpha_tensor_es = []
    tC = torch.zeros([self.K, self.K], dtype=torch_dtype, device=device)
    nu_tensors = []
    for YT, adjacency_list, XT, beta in zip(self.YTs, self.Es, self.XTs, self.betas):
        if self.dropout_mode == 'raw':
            XXTs.append(XT.T @ XT)
        else:
            raise NotImplementedError

        XT_tensor = torch.tensor(XT, dtype=torch_dtype, device=device)
        N, G = YT.shape
        alpha_tensors.append(XT_tensor.sum(axis=0))
        
        alpha_tensor_e = torch.tensor([len(neighbor_list) for neighbor_list in adjacency_list], dtype=torch_dtype, device=device) @ XT_tensor
        alpha_tensor_es.append(alpha_tensor_e)

        # Normalizing XT_tensor
        ZT_tensor = torch.div(XT_tensor, XT_tensor.sum(axis=1, keepdim=True).add_(1e-30))
       
        # Each row of nu_tensor is the sum of the x_i of its neighbors
        nu_tensor = torch.empty([N, self.K], dtype=torch_dtype, device=device)
        for index, neighbor_list in enumerate(adjacency_list):
            nu_tensor[index] = ZT_tensor[neighbor_list].sum(axis=0)
        
        nu_tensors.append(nu_tensor)

        tC = torch.add(tC, alpha=beta, other=ZT_tensor.t() @ nu_tensor)

    Q_X = 0
    if all(prior_x_mode == 'Gaussian' for prior_x_mode, *_ in self.prior_x_parameter_sets) and self.pairwise_potential_mode == 'linear':
        raise NotImplementedError
    elif self.pairwise_potential_mode in ('linear', 'linear w/ shift'):
        raise NotImplementedError
    elif all(prior_x_mode in ('Exponential shared', 'Exponential shared fixed') for prior_x_mode, *_ in self.prior_x_parameter_sets) and self.pairwise_potential_mode == 'normalized':
        # Calculating Q_X and potentially updatign priors on X values
        new_prior_x_parameter_sets = []
        for num_genes, (prior_x_mode, *prior_x_parameters), alpha_tensor in zip(self.Ns, self.prior_x_parameter_sets, alpha_tensors):
            if prior_x_mode == 'Exponential shared':
                lambda_x, = prior_x_parameters
                lambda_x = (alpha_tensor.mean() / num_genes).pow(-1).cpu().data.numpy()
                Q_X -= lambda_x * alpha_tensor.sum().cpu().data.numpy()
                Q_X += num_genes * self.K * np.log(lambda_x) - num_genes * loggamma(self.K)
                prior_x_parameter_set = (prior_x_mode, np.full(self.K, lambda_x))
                new_prior_x_parameter_sets.append(prior_x_parameter_set)
            elif prior_x_mode == 'Exponential shared fixed':
                lambda_x, = prior_x_parameters
                Q_X -= lambda_x.mean() * alpha_tensor.sum().cpu().data.numpy()
                new_prior_x_parameter_sets.append((prior_x_mode, *prior_x_parameters))
            else:
                raise NotImplementedError

        self.prior_x_parameter_sets = new_prior_x_parameter_sets

        if not sum(self.total_edge_counts) == 0:
            valid_diteration = 100
            max_torch_iterations = 1000
            
            # TODO: what is this for? I don't see it anywhere else in the code.
            batch_sizes = [512, ] * self.num_repli
            # requires_grad = True
            requires_grad = False

            variable_list = []
            optimizers = []
            schedulers = []
            sigma_x_inverse_tensor = torch.tensor(self.sigma_x_inverse, dtype=torch_dtype, device=device, requires_grad=requires_grad)
            variable_list += [sigma_x_inverse_tensor,]

            optimizer = torch.optim.Adam([sigma_x_inverse_tensor], lr=1e-2)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, valid_diteration, gamma=0.98)
            # optimizers.append(optimizer)
            # schedulers.append(scheduler)
            # del optimizer, scheduler

            tprior_x_parameter_sets = []
            for prior_x_mode, *prior_x_parameters in self.prior_x_parameter_sets:
                if prior_x_mode in ('Exponential shared', 'Exponential shared fixed'):
                    lambda_x, = prior_x_parameters
                    lambda_x_tensor = torch.tensor(lambda_x, dtype=torch_dtype, device=device, requires_grad=requires_grad)
                    tprior_x_parameter_sets.append((prior_x_mode, lambda_x_tensor,))
                    variable_list.append(lambda_x_tensor)
                    del lambda_x
                else:
                    raise NotImplementedError
            
            for variable in variable_list:
                variable.grad = torch.zeros_like(variable)

            # tdiagXXTs = [torch.tensor(np.diag(XT).copy(), dtype=torch_dtype, device=device) for XT in XXTs]
            # tNus = [nu_tensor.sum(axis=0) for nu_tensor in nu_tensors]
            # tNu2s = [nu_tensor.t() @ nu_tensor for nu_tensor in nu_tensors]
            alpha_tensor_e_all = torch.zeros_like(alpha_tensor_es[0])
            for beta, alpha_tensor_e in zip(self.betas, alpha_tensor_es):
                alpha_tensor_e_all.add_(alpha=beta, other=alpha_tensor_e)
            
            # total_edge_counts = [sum(map(len, E)) for E in self.Es]
            adjacency_counts = [torch.tensor(list(map(len, E)), dtype=torch_dtype, device=device) for E in self.Es]
            # tZTs = [torch.tensor(XT, dtype=torch_dtype, device=device) for XT in self.XTs]
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

            assumption_str = 'mean-field'
            # assumption_str = None
            # assumption_str = 'independent'
            random_flag = assumption_str in [
                'independent',
                'mean-field',
            ]

            regenerate_diteration = int(1e10)
            # tZes = [None] * self.num_repli
            
            num_samples = 64
            if assumption_str == None:
                raise NotImplementedError
            elif assumption_str == 'mean-field':
                pass
            elif assumption_str == 'independent':
                raise NotImplementedError
            else:
                raise NotImplementedError

            if assumption_str in (None, 'independent'):
                tC.div_(2)

            # TODO: do we need this?
            loggamma_K = loggamma(self.K)

            func, last_func = None, torch.empty([], dtype=torch_dtype, device=device).fill_(np.nan)
            best_func, best_iteration = torch.empty([], dtype=torch_dtype, device=device).fill_(np.nan), -1
            sigma_x_inverse_tensor_best = None
            for torch_iteration in range(max_torch_iterations + 1):
                if not requires_grad:
                    for variable in variable_list:
                        variable.grad.zero_()
                else:
                    # for optimizer in optimizers:
                    #     optimizer.zero_grad()
                    optimizer.zero_grad()

                if sigma_x_inverse_lb:
                    sigma_x_inverse_tensor.clamp_(min=sigma_x_inverse_lb)
                if sigma_x_inverse_ub:
                    sigma_x_inverse_tensor.clamp_(max=sigma_x_inverse_ub)
                if sigma_x_inverse_constraint in ('diagonal', 'diagonal same'):
                    sigma_x_inverse_tensor = sigma_x_inverse_tensor.triu().tril()
                if sigma_x_inverse_constraint == 'diagonal same':
                    sigma_x_inverse_tensor[(range(self.K), range(self.K))] = sigma_x_inverse_tensor[(range(self.K), range(self.K))].mean()

                func = torch.zeros([], dtype=torch_dtype, device=device)
                # if requires_grad:
                func_grad = torch.zeros([], dtype=torch_dtype, device=device, requires_grad=True)

                # pairwise potential
                sigma_x_inverse_tensor.grad += tC
                if requires_grad:
                    func_grad = func_grad + tC.view(-1) @ sigma_x_inverse_tensor.view(-1)
                else:
                    func.add_(tC.view(-1) @ sigma_x_inverse_tensor.view(-1))

                for N, total_edge_count, adjacency_count, E, beta, alpha_tensor, nu_tensor, tprior_x in zip(self.Ns, self.total_edge_counts, adjacency_counts, self.Es, self.betas,
                        alpha_tensors, nu_tensors, tprior_x_parameter_sets):
                    
                    if total_edge_count == 0:
                        continue

                    if assumption_str == 'mean-field':
                        if tprior_x[0] in ('Exponential shared', 'Exponential shared fixed'):
                            if torch_iteration % valid_diteration == 0:
                                index = slice(None)
                            else:
                                index = np.random.choice(N, min(num_samples, N), replace=False)

                            nu_tensor = nu_tensor[index].contiguous()
                            factor = total_edge_count / adjacency_count[index].sum()
                            
                            # Z^z(\theta)
                            beta_i_tensor = nu_tensor @ sigma_x_inverse_tensor
                            beta_i_tensor.grad = torch.zeros_like(beta_i_tensor)
                            # torch.manual_seed(iteration)
                            # TODO: delete if unncessary logic
                            # if iteration > 1 or torch_iteration > 100:
                            #     # log_Z_tensor = integrateOfExponentialOverSimplexSampling(beta_i_tensor, requires_grad=requires_grad, seed=iteration*max_torch_iterations+torch_iteration)
                            #     log_Z_tensor = integrateOfExponentialOverSimplexInduction2(beta_i_tensor, grad=c, requires_grad=requires_grad, )
                            # else:
                            #     # log_Z_tensor = integrateOfExponentialOverSimplexSampling(beta_i_tensor, requires_grad=requires_grad, seed=iteration*max_torch_iterations+torch_iteration)
                            #     log_Z_tensor = integrateOfExponentialOverSimplexInduction2(beta_i_tensor, grad=c, requires_grad=requires_grad)
                            log_Z_tensor = integrate_over_simplex(beta_i_tensor, grad=factor, requires_grad=requires_grad)
                            
                            if requires_grad:
                                func_grad = func_grad.add(beta*factor, log_Z_tensor.sum())
                            else:
                                func = func.add(alpha=beta*factor, other=log_Z_tensor.sum())
                                sigma_x_inverse_tensor.grad = sigma_x_inverse_tensor.grad.addmm(alpha=beta, mat1=nu_tensor.t(), mat2=beta_i_tensor.grad)
                        else:
                            raise NotImplementedError
                    elif assumption_str == None:
                        raise NotImplementedError
                    elif assumption_str == 'independent':
                        raise NotImplementedError
                    else:
                        raise NotImplementedError

                if requires_grad:
                    func_grad.backward()
                    func = func + func_grad

                # prior on Σ_x^inv
                # TODO: can we delete this?
                # num_burnin_iteration = 200
                # if iteration <= num_burnin_iteration:
                #   kk = 1e-1 * np.dot(betas, list(map(len, Es))) * 1e-1**((num_burnin_iteration-iteration+1)/num_burnin_iteration)
                # else:
                #   kk = 1e-1 * np.dot(betas, list(map(len, Es)))


                weighted_total_edge_count = np.dot(self.betas, self.total_edge_counts)
                kk = self.lambda_sigma_x_inverse * weighted_total_edge_count
                sigma_x_inverse_tensor.grad += kk * sigma_x_inverse_tensor
                func += kk / 2 * sigma_x_inverse_tensor.pow(2).sum()

                # normalize gradient by the weighted sizes of data sets
                sigma_x_inverse_tensor.grad.div_(weighted_total_edge_count)
                func.div_(np.dot(self.betas, self.Ns))

                sigma_x_inverse_tensor.grad.add_(sigma_x_inverse_tensor.grad.clone().t()).div_(2)

                if sigma_x_inverse_lb:
                    sigma_x_inverse_tensor.grad[(sigma_x_inverse_tensor <= sigma_x_inverse_lb) * (sigma_x_inverse_tensor.grad > 0)] = 0
                if sigma_x_inverse_ub:
                    sigma_x_inverse_tensor.grad[(sigma_x_inverse_tensor >= sigma_x_inverse_ub) * (sigma_x_inverse_tensor.grad < 0)] = 0
                if sigma_x_inverse_constraint in ['diagonal', 'diagonal same']:
                    sigma_x_inverse_tensor.grad.triu_().tril_()
                if sigma_x_inverse_constraint in ['diagonal same']:
                    sigma_x_inverse_tensor.grad[(range(self.K), range(self.K))] = sigma_x_inverse_tensor.grad[(range(self.K), range(self.K))].mean()

                best_flag = False
                if not random_flag or torch_iteration % valid_diteration == 0:
                    best_flag = not best_func <= func
                    if best_flag:
                        best_func, best_iteration = func, torch_iteration
                        sigma_x_inverse_tensor_best = sigma_x_inverse_tensor.clone().detach()

                stop_flag = True
                # stop_flag = False
                stop_sigma_x_inverse_tensor_grad_pseudo = 1e-1
                stop_flag &= (sigma_x_inverse_tensor.grad.abs() / (sigma_x_inverse_tensor.abs() + stop_sigma_x_inverse_tensor_grad_pseudo)).abs().max().item() < 1e-2
                for tprior_x in tprior_x_parameter_sets:
                    if tprior_x[0] in ['Exponential shared', ]:
                        lambda_x_tensor, = tprior_x[1:]
                        stop_flag &= lambda_x_tensor.grad.abs().max().item() < 1e-4
                        del lambda_x_tensor
                    elif tprior_x[0] in ['Exponential shared fixed', ]:
                        pass
                    else:
                        raise NotImplementedError
                
                if random_flag:
                    stop_flag &= not bool(func <= last_func - 1e-3 * valid_diteration)
                else:
                    stop_flag &= not bool(func <= last_func - 1e-3)

                stop_flag |= random_flag and not torch_iteration < best_iteration + 2*valid_diteration
                # stop_flag |= best_func == func and torch_iteration > best_iteration + 20
                if random_flag and torch_iteration % valid_diteration != 0:
                    stop_flag = False

                warning_flag = bool(func > last_func + 1e-10)
                warning_flag &= not random_flag or torch_iteration % valid_diteration == 0
                # warning_flag = True

                if torch_iteration % valid_diteration == 0 or torch_iteration == max_torch_iterations or warning_flag or (regenerate_diteration != 1 and (torch_iteration % regenerate_diteration == 0 or (torch_iteration+1) % regenerate_diteration == 0)):
                    print(
                        f'At iteration {torch_iteration},\t'
                        f'func = {(func - last_func).item():.2e} -> {func.item():.2e}\t'
                        f'Σ_x^inv: {sigma_x_inverse_tensor.max().item():.1e} - {sigma_x_inverse_tensor.min().item():.1e} = {sigma_x_inverse_tensor.max() - sigma_x_inverse_tensor.min():.1e} '
                        f'grad = {sigma_x_inverse_tensor.grad.min().item():.2e} {sigma_x_inverse_tensor.grad.max().item():.2e}\t'
                        f'var/grad = {(sigma_x_inverse_tensor.grad.abs()/(sigma_x_inverse_tensor.abs() + stop_sigma_x_inverse_tensor_grad_pseudo)).abs().max().item():.2e}'
                        # f'δ_x: {tdelta_x.max().item():.1e} - {tdelta_x.min().item():.1e} = {tdelta_x.max() - tdelta_x.min():.1e} '
                        # f'grad = {tdelta_x.grad.min().item():.2e} {tdelta_x.grad.max().item():.2e}'
                        , end=''
                    )
                    
                    if warning_flag:
                        print('\tWarning', end='')
                    if best_flag:
                        print('\tbest', end='')
                    print()
                    sys.stdout.flush()

                # stop_flag = True

                if stop_flag:
                    break

                # for optimizer in optimizers:
                #     optimizer.step()
                # for scheduler in schedulers:
                #     scheduler.step()
                
                optimizer.step()
                scheduler.step()

                if not random_flag or torch_iteration % valid_diteration == 0:
                    last_func = func

            sigma_x_inverse_tensor = sigma_x_inverse_tensor_best
            func = best_func
            self.sigma_x_inverse = sigma_x_inverse_tensor.cpu().data.numpy()

            func *= np.dot(self.betas, self.Ns)
            Q_X -= func.item()

    elif all(prior_x[0] == 'Exponential' for prior_x in self.prior_x_parameter_sets) and self.pairwise_potential_mode == 'normalized':
        raise NotImplementedError
    else:
        raise NotImplementedError

    return Q_X
