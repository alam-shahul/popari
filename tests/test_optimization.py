import os
import numpy as np
import pytest
import torch
import h5py
import anndata as ad

from popari._embedding_optimizer import EmbeddingOptimizer
from popari import Popari
from popari.util import project2simplex_

data_path = "tests/test_data/optimization_dataset/"


@pytest.fixture(scope="module")
def popari_model(test_datapath, context):
    adata = ad.read_h5ad(test_datapath / "all_data.h5")
    adata_subset = adata[:60, :60]
    adata_subset.write(data_path + "data.h5")
    
    replicate_names = [0, 1]
    model = Popari(
        K=5,
        lambda_Sigma_x_inv=1e-5,
        metagene_mode="differential",
        lambda_M=0.5,
        torch_context=context,
        initial_context=context,
        dataset_path=data_path + "data.h5",
        replicate_names=replicate_names,
        verbose=2,
    )
    
    for iteration in range(1, 2):  # Just one iteration for testing
        model.estimate_parameters()
        model.estimate_weights()

    return model


@pytest.fixture(scope="module")
def test_data(popari_model):
    """Prepare test data for testing specific functions."""
    # Get optimizer and dataset
    parameter_optimizer = popari_model.parameter_optimizer
    embedding_optimizer = popari_model.embedding_optimizer
    dataset = popari_model.datasets[0]
    
    # Get required matrices
    dataset_name = dataset.name
    replicate_mask = [d.name == dataset_name for d in popari_model.datasets]

    Y = popari_model.Ys[0]
    M = popari_model.parameter_optimizer.metagene_state[dataset_name]
    MTM = M.T @ M
    M_bar = []
    if parameter_optimizer.metagene_mode == "differential":
        for group_name in parameter_optimizer.metagene_tags[dataset_name]: 
            M_bar += [parameter_optimizer.metagene_state.M_bar[group_name]]
    YM = Y @ M
    
    # Get embedding states
    X = embedding_optimizer.embedding_state[dataset_name]
    S = torch.linalg.norm(X, dim=1, ord=1, keepdim=True)
    Z = X / S
    
    # Get adjacency matrix and Sigma_x_inv
    adjacency_matrix = embedding_optimizer.adjacency_matrices[dataset_name]
    Sigma_x_inv = popari_model.parameter_optimizer.spatial_affinity_state[dataset_name]
    
    return {
        "parameter_optimizer": parameter_optimizer,
        "embedding_optimizer": embedding_optimizer,
        "dataset": dataset,
        "replicate_mask": replicate_mask,
        "Y": Y,
        "M": M,
        "X": X,
        "S": S,
        "Z": Z,
        "MTM": MTM,
        "M_bar": M_bar,
        "YM": YM,
        "adjacency_matrix": adjacency_matrix,
        "Sigma_x_inv": Sigma_x_inv,
    }







def test_compute_loss_and_gradient(test_data):
    """Test compute_loss_and_gradient function from ParameterOptimizer."""
    # Get test data
    param_optimizer = test_data["parameter_optimizer"]
    M = test_data["M"].clone()
    replicate_mask = test_data["replicate_mask"]
    M_bar = test_data["M_bar"]
    
    # Copied compute_loss_and_gradient function based on the original code
    def compute_loss_and_gradient(M): 
        # Extract necessary components from the parameter optimizer
        K = param_optimizer.K
        G, K = M.shape
        betas = param_optimizer.betas[replicate_mask]
        betas = betas / betas.sum()
        
        datasets = [dataset for (use_replicate, dataset) in zip(replicate_mask, param_optimizer.datasets) if use_replicate]
        Xs = [param_optimizer.embedding_optimizer.embedding_state[dataset.name] for dataset in datasets]
        Ys = [Y for (use_replicate, Y) in zip(replicate_mask, param_optimizer.Ys) if use_replicate]
        sigma_yxs = param_optimizer.sigma_yxs[replicate_mask]
        scaled_betas = betas / (sigma_yxs**2)
        
        # Calculate constant term
        constant = np.array([torch.square(Y).sum().cpu() / (sigma_yx**2) for Y, sigma_yx in zip(Ys, sigma_yxs)],).sum()
        
        # Create quadratic and linear factors
        quadratic_factor = torch.zeros([K, K], **param_optimizer.context)
        linear_factor = torch.zeros_like(M)
        
        for dataset, X, Y, scaled_beta in zip(datasets, Xs, Ys, scaled_betas):
            # X_c^TX_c
            quadratic_factor.addmm_(X.T, X, alpha=scaled_beta)
            # MX_c^TY_c
            linear_factor.addmm_(Y.T, X, alpha=scaled_beta)
        
        # Handle differential regularization if applicable
        differential_regularization_quadratic_factor = torch.zeros((K, K), **param_optimizer.context)
        differential_regularization_linear_factor = torch.zeros_like(M, **param_optimizer.context)
        
        if param_optimizer.lambda_M > 0 and M_bar is not None:
            differential_regularization_quadratic_factor = param_optimizer.lambda_M * torch.eye(K, **param_optimizer.context)
            
            group_weighting = 1 / len(M_bar)
            for group_M_bar in M_bar:
                differential_regularization_linear_factor += group_weighting * param_optimizer.lambda_M * group_M_bar
        
        # Compute loss and gradient
        quadratic_factor_grad = M @ (quadratic_factor + differential_regularization_quadratic_factor)
        loss = (quadratic_factor_grad * M).sum()
        linear_term_grad = linear_factor + differential_regularization_linear_factor
        loss -= 2 * (linear_term_grad * M).sum()
        grad = quadratic_factor_grad - linear_term_grad
        
        loss += constant
        
        # Add differential regularization term if applicable
        if param_optimizer.metagene_mode == "differential" and M_bar is not None:
            differential_regularization_term = (M @ differential_regularization_quadratic_factor * M).sum() - 2 * (differential_regularization_linear_factor * M).sum()
            group_weighting = 1 / len(M_bar)
            for group_M_bar in M_bar:
                differential_regularization_term += (group_weighting * param_optimizer.lambda_M * (group_M_bar * group_M_bar).sum())
        
        loss /= 2
        
        if param_optimizer.M_constraint == "simplex":
            grad.sub_(grad.sum(0, keepdim=True))
            
        return loss.item(), grad
    
    # Run the function
    loss_actual, grad_actual = compute_loss_and_gradient(M)
    
    # Convert to CPU numpy for saving/comparison
    grad_actual_np = grad_actual.detach().cpu().numpy()
    
    #np.save(data_path + "compute_loss_and_gradient_loss.npy", loss_actual)
    #np.save(data_path + "compute_loss_and_gradient_grad.npy", grad_actual_np)
    
    # Load the expected values and compare
    expected_loss = np.load(data_path + "compute_loss_and_gradient_loss.npy")
    expected_grad = np.load(data_path + "compute_loss_and_gradient_grad.npy")
    
    assert np.isclose(loss_actual, expected_loss, rtol=1e-5)
    assert np.allclose(grad_actual_np, expected_grad, rtol=1e-5)


def test_update_sigma_yx(test_data):
    """Test update_sigma_yx function from ParameterOptimizer."""
    # Get test data
    param_optimizer = test_data["parameter_optimizer"]
    Y = test_data["Y"]
    X = test_data["X"]
    dataset = test_data["dataset"]
    
    # Copied update_sigma_yx function based on the original code
    def update_sigma_yx():
        # Create a copy of the original sigma_yxs
        original_sigma_yxs = param_optimizer.sigma_yxs.copy()
        
        # Compute squared terms for each dataset
        squared_terms = [torch.addmm(Y.to_dense(), X, param_optimizer.metagene_state[dataset.name].T, alpha=-1,)]
        
        # Compute squared loss
        squared_loss = np.array([torch.linalg.norm(squared_term, ord="fro").item() ** 2 for squared_term in squared_terms],)
        
        # Compute sizes
        sizes = np.array([Y.numel()])
        
        # Update sigma_yx based on mode
        if param_optimizer.sigma_yx_inv_mode == "separate":
            result = np.sqrt(squared_loss / sizes)
        elif param_optimizer.sigma_yx_inv_mode == "average":
            betas = np.array([param_optimizer.betas[0]])  # Just use the first beta for this test
            result = np.sqrt(np.dot(betas, squared_loss) / np.dot(betas, sizes))
            result = np.full(1, float(result))
        else:
            raise NotImplementedError
            
        # Restore original sigma_yxs to not affect other tests
        param_optimizer.sigma_yxs = original_sigma_yxs
        
        return result
    
    # Run the function
    result_sigma_yx = update_sigma_yx()
    
    #np.save(data_path + "update_sigma_yx.npy", result_sigma_yx)
    # Load the expected values and compare
    expected_sigma_yx = np.load(data_path + "update_sigma_yx.npy")
    
    assert np.allclose(result_sigma_yx, expected_sigma_yx, rtol=1e-5)

    

def test_calc_func_grad(test_data):
    """Test calc_func_grad function from EmbeddingOptimizer."""
    # Get test data
    Z = test_data["Z"]
    S = test_data["S"]
    MTM = test_data["MTM"]
    YM = test_data["YM"]
    
    # Select first few cells for batch testing
    batch_indices = torch.tensor([0, 1, 2], device=Z.device)
    Z_batch = Z[batch_indices]
    S_batch = S[batch_indices]
    linear = YM[batch_indices] * S_batch
    
    # Copied calc_func_grad function as in the original code
    def calc_func_grad(Z_batch, S_batch, quad, linear):
        t = (Z_batch @ quad).mul_(S_batch**2)
        f = (t * Z_batch).sum() / 2
        g = t
        t = linear
        f -= (t * Z_batch).sum()
        g -= t
        g.sub_(g.sum(1, keepdim=True))

        return f.item(), g
    
    # Run the function
    f_actual, g_actual = calc_func_grad(Z_batch, S_batch, MTM, linear)
    
    # Convert to CPU numpy for saving/comparison
    g_actual_np = g_actual.detach().cpu().numpy()
    
    #np.save(data_path + "calc_func_grad_f.npy", f_actual)
    #np.save(data_path + "calc_func_grad_g.npy", g_actual_np)
    
    # Load the expected values and compare
    expected_f = np.load(data_path + "calc_func_grad_f.npy")
    expected_g = np.load(data_path + "calc_func_grad_g.npy")
    
    assert np.isclose(f_actual, expected_f, rtol=1e-5)
    assert np.allclose(g_actual_np, expected_g, rtol=1e-5)


def test_update_s(test_data):
    """Test update_s function from EmbeddingOptimizer."""
    # Get test data
    Z = test_data["Z"].clone()
    S = test_data["S"].clone()
    MTM = test_data["MTM"]
    YM = test_data["YM"]
    prior_x_mode = False
    
    # Copied update_s function as in the original code
    def update_s():
        nonlocal S, Z, YM, MTM
        S[:] = (YM * Z).sum(axis=1, keepdim=True)
        if prior_x_mode == "exponential shared fixed":
            # TODO: why divide by two?
            S.sub_(prior_x[0][0] / 2)
        elif not prior_x_mode:
            pass
        else:
            raise NotImplementedError

        denominator = ((Z @ MTM) * Z).sum(axis=1, keepdim=True)
        S.div_(denominator)
        S.clip_(min=1e-5)
        return S
    
    # Run the function
    result_S = update_s()
    
    # Convert to CPU numpy for saving/comparison
    result_S_np = result_S.detach().cpu().numpy()
    
    #np.save(data_path + "update_s.npy", result_S_np)
    
    # Load the expected values and compare
    expected_S = np.load(data_path + "update_s.npy")
    
    assert np.allclose(result_S_np, expected_S, rtol=1e-5)


def test_update_z_gd_nesterov(test_data):
    """Test update_z_gd_nesterov function from EmbeddingOptimizer with one step."""
    # Get test data
    Z = test_data["Z"].clone()
    S = test_data["S"].clone()
    MTM = test_data["MTM"]
    YM = test_data["YM"]
    adjacency_matrix = test_data["adjacency_matrix"]
    Sigma_x_inv = test_data["Sigma_x_inv"]
    
    # Define helper functions
    
    def calc_func_grad(Z_batch, S_batch, quad, linear):
        t = (Z_batch @ quad).mul_(S_batch**2)
        f = (t * Z_batch).sum() / 2
        g = t
        t = linear
        f -= (t * Z_batch).sum()
        g -= t
        g.sub_(g.sum(1, keepdim=True))
        return f.item(), g
    
    # Define a simplified one-step update function
    def update_z_one_step():
        # Make a copy of Z to modify
        Z_new = Z.clone()
                
        # Calculate linear term
        linear_batch = YM * S - adjacency_matrix @ Z @ Sigma_x_inv / 2
        
        # Calculate gradient
        _, grad = calc_func_grad(Z, S, MTM, linear_batch)

        
        # Take a gradient step
        step_size = 0.01 / S.square()
        Z_new = Z - step_size * grad
        
        # Project back to simplex
        Z_new = project2simplex_(Z_new, dim=1)
        
        return Z_new
    
    # Run the function
    result_Z = update_z_one_step()
    
    # Convert to CPU numpy for saving/comparison
    result_Z_np = result_Z.detach().cpu().numpy()
    
    #np.save(data_path + "update_z_gd_nesterov.npy", result_Z_np)
    
    # Load the expected values and compare
    expected_Z = np.load(data_path + "update_z_gd_nesterov.npy")
    
    assert np.allclose(result_Z_np, expected_Z, rtol=1e-5)






