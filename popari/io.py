from typing import Union, Sequence, Optional
from pathlib import Path

import numpy as np
import anndata as ad
import torch
from scipy.sparse import csr_matrix, issparse

from popari.components import PopariDataset
from popari.util import convert_numpy_to_pytorch_sparse_coo

def load_anndata(filepath: Union[str, Path], replicate_names: Sequence[str] = None):
    """Load AnnData object from h5ad file and reformat for Popari.

    """                                     

    # TODO: make it so that replicate_names can rename the datasets

    merged_dataset = ad.read_h5ad(filepath)

    indices = merged_dataset.obs.groupby("batch").indices.values()
    datasets = [merged_dataset[index] for index in indices]
   
    if replicate_names == None:
        replicate_names = [dataset.obs["batch"].unique()[0] for dataset in datasets]
    
    datasets = [PopariDataset(dataset, replicate_name) for dataset, replicate_name in zip(datasets, replicate_names)]

    if len(replicate_names) != len(datasets):
        raise ValueError(f"List of replicates '{replicate_names}' does not match number of datasets ({len(datasets)}) stored in AnnData object.")
        
    for replicate, dataset in zip(replicate_names, datasets):
        replicate_string = f"{replicate}"
        if "Sigma_x_inv" in dataset.uns:
            # Keep only Sigma_x_inv corresponding to a particular replicate
            replicate_Sigma_x_inv = dataset.uns["Sigma_x_inv"][replicate_string]
            if np.isscalar(replicate_Sigma_x_inv) and replicate_Sigma_x_inv == -1:
                replicate_Sigma_x_inv = None

            dataset.uns["Sigma_x_inv"] = {
                replicate_string: replicate_Sigma_x_inv
            }
        
        if "Sigma_x_inv_bar" in dataset.uns:
            # Keep only Sigma_x_inv_bar corresponding to a particular replicate
            replicate_Sigma_x_inv_bar = dataset.uns["Sigma_x_inv_bar"][replicate_string]
            if np.isscalar(replicate_Sigma_x_inv_bar) and replicate_Sigma_x_inv_bar == -1:
                replicate_Sigma_x_inv_bar = None

            dataset.uns["Sigma_x_inv_bar"] = {
                replicate_string: replicate_Sigma_x_inv_bar
            }

        # Hacks to load adjacency matrices efficiently
        if "adjacency_matrix" in dataset.uns:
            dataset.obsp["adjacency_matrix"] = csr_matrix(dataset.uns["adjacency_matrix"][replicate_string])

        adjacency_matrix = dataset.obsp["adjacency_matrix"].tocoo()

        num_cells, _ = adjacency_matrix.shape
        adjacency_list = [[] for _ in range(num_cells)]
        for x, y in zip(*adjacency_matrix.nonzero()):
            adjacency_list[x].append(y)

        dataset.obs["adjacency_list"] = adjacency_list

        if "M" in dataset.uns:
            if replicate_string in dataset.uns["M"]:
                replicate_M = make_hdf5_compatible(dataset.uns["M"][replicate_string])
                dataset.uns["M"] = {replicate_string: replicate_M}
    
        if "M_bar" in dataset.uns:
            if replicate_string in dataset.uns["M_bar"]:
                replicate_M_bar = make_hdf5_compatible(dataset.uns["M_bar"][replicate_string])
                dataset.uns["M_bar"] = {replicate_string: replicate_M_bar}
        
        if "popari_hyperparameters" in dataset.uns:
            if "prior_x" in dataset.uns["popari_hyperparameters"]:
                prior_x = make_hdf5_compatible(dataset.uns["popari_hyperparameters"]["prior_x"])
                dataset.uns["popari_hyperparameters"]["prior_x"] = prior_x
 
        if "X" in dataset.obsm:
            replicate_X = make_hdf5_compatible(dataset.obsm["X"])
            dataset.obsm["X"] = replicate_X

        if issparse(dataset.X):
            dataset.X = dataset.X.toarray()

    return datasets, replicate_names

def save_anndata(filepath: Union[str, Path], datasets: Sequence[PopariDataset], replicate_names: Sequence[str], ignore_raw_data: bool = False):
    """Save Popari state as AnnData object.

    """
    dataset_copies = []
    for replicate, dataset in zip(replicate_names, datasets):
        replicate_string = f"{replicate}"
        if ignore_raw_data:
            X = csr_matrix(dataset.X.shape)
        else:
            X = dataset.X
        dataset_copy = ad.AnnData(
                X=X,
                obs=dataset.obs,
                var=dataset.var,
                uns=dataset.uns,
                obsm=dataset.obsm,
        )
        # Hacks to store adjacency matrices efficiently
        if "adjacency_matrix" in dataset.obsp:
            adjacency_matrix = make_hdf5_compatible(dataset.obsp["adjacency_matrix"])
            dataset_copy.uns["adjacency_matrix"] = {replicate_string: adjacency_matrix }
        
        if "Sigma_x_inv" in dataset_copy.uns:
            replicate_Sigma_x_inv = dataset_copy.uns["Sigma_x_inv"][replicate_string]
            if replicate_string in dataset_copy.uns["Sigma_x_inv"]:
                # Using a sentinel value - hopefully this can be fixed in the future!
                if replicate_Sigma_x_inv is None: 
                    replicate_Sigma_x_inv = -1
                else:
                    replicate_Sigma_x_inv = make_hdf5_compatible(replicate_Sigma_x_inv)
            dataset_copy.uns["Sigma_x_inv"] = {replicate_string: replicate_Sigma_x_inv}
        
        if "Sigma_x_inv_bar" in dataset_copy.uns:
            replicate_Sigma_x_inv_bar = dataset_copy.uns["Sigma_x_inv_bar"][replicate_string]
            if replicate_string in dataset_copy.uns["Sigma_x_inv_bar"]:
                # Using a sentinel value - hopefully this can be fixed in the future!
                if replicate_Sigma_x_inv_bar is None: 
                    replicate_Sigma_x_inv_bar = -1
                else:
                    replicate_Sigma_x_inv_bar = make_hdf5_compatible(replicate_Sigma_x_inv_bar)
            dataset_copy.uns["Sigma_x_inv_bar"] = {replicate_string: replicate_Sigma_x_inv_bar}
        
        if "M" in dataset_copy.uns:
            if replicate_string in dataset_copy.uns["M"]:
                replicate_M = make_hdf5_compatible(dataset_copy.uns["M"][replicate_string])
                dataset_copy.uns["M"] = {replicate_string: replicate_M}
        
        if "M_bar" in dataset_copy.uns:
            if replicate_string in dataset_copy.uns["M_bar"]:
                replicate_M_bar = make_hdf5_compatible(dataset_copy.uns["M_bar"][replicate_string])
                dataset_copy.uns["M_bar"] = {replicate_string: replicate_M_bar}
        
        if "popari_hyperparameters" in dataset_copy.uns:
            if "prior_x" in dataset_copy.uns["popari_hyperparameters"]:
                prior_x = make_hdf5_compatible(dataset_copy.uns["popari_hyperparameters"]["prior_x"])
                dataset_copy.uns["popari_hyperparameters"]["prior_x"] = prior_x
            dataset_copy.uns["popari_hyperparameters"]["name"] = dataset.name
        else:
            dataset_copy.uns["popari_hyperparameters"] = {"name": dataset.name}
        
        if "X" in dataset_copy.obsm:
            replicate_X = make_hdf5_compatible(dataset_copy.obsm["X"])
            dataset_copy.obsm["X"] = replicate_X
        dataset_copies.append(dataset_copy)
        
        if "adjacency_list" in dataset_copy.obs:
            del dataset_copy.obs["adjacency_list"]

    dataset_names = [dataset.name for dataset in datasets]
    merged_dataset = ad.concat(dataset_copies, label="batch", keys=dataset_names, merge="unique", uns_merge="unique")
    merged_dataset.write(filepath)

    return merged_dataset

def make_hdf5_compatible(array: Union[torch.Tensor, np.ndarray]):
    """Convert tensors to Numpy array for compatibility with HDF5.

    Args:
        array: array to convert to Numpy.
    """
    if type(array) == torch.Tensor:
        cpu_tensor = array.detach().cpu()
        if cpu_tensor.is_sparse:
            cpu_tensor = cpu_tensor.to_dense()
        return cpu_tensor.numpy()
    return array
