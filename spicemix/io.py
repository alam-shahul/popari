from typing import Union, Sequence, Optional
from pathlib import Path

import numpy as np
import anndata as ad
import torch

from spicemix.components import SpiceMixDataset

def load_anndata(filepath: Union[str, Path], replicate_names: Sequence[str] = None, context: str = "numpy"):
    """Load AnnData object from h5ad file and reformat for SpiceMixPlus.

    """                                     

    # TODO: make it so that replicate_names can rename the datasets

    merged_dataset = ad.read_h5ad(filepath)

    indices = merged_dataset.obs.groupby("batch").indices.values()
    datasets = [merged_dataset[index] for index in indices]
   
    if replicate_names == None:
        replicate_names = [dataset.obs["batch"].unique()[0] for dataset in datasets]
    
    datasets = [SpiceMixDataset(dataset, replicate_name) for dataset, replicate_name in zip(datasets, replicate_names)]

    if len(replicate_names) != len(datasets):
        raise ValueError(f"List of replicates '{replicate_names}' does not match number of datasets ({len(datasets)}) stored in AnnData object.")
        
    for replicate, dataset in zip(replicate_names, datasets):
        if "Sigma_x_inv" in dataset.uns:
            # Keep only Sigma_x_inv corresponding to a particular replicate
            replicate_Sigma_x_inv = dataset.uns["Sigma_x_inv"][f"{replicate}"]
            if np.isscalar(replicate_Sigma_x_inv) and replicate_Sigma_x_inv == -1:
                replicate_Sigma_x_inv = None
            elif context != "numpy":
                replicate_Sigma_x_inv = torch.tensor(replicate_Sigma_x_inv, **context)

            dataset.uns["Sigma_x_inv"] = {
                f"{replicate}": replicate_Sigma_x_inv
            }
        
        if "Sigma_x_inv_bar" in dataset.uns:
            # Keep only Sigma_x_inv_bar corresponding to a particular replicate
            replicate_Sigma_x_inv_bar = dataset.uns["Sigma_x_inv_bar"][f"{replicate}"]
            if np.isscalar(replicate_Sigma_x_inv_bar) and replicate_Sigma_x_inv_bar == -1:
                replicate_Sigma_x_inv_bar = None
            elif context != "numpy":
                replicate_Sigma_x_inv_bar = torch.tensor(replicate_Sigma_x_inv_bar, **context)

            dataset.uns["Sigma_x_inv_bar"] = {
                f"{replicate}": replicate_Sigma_x_inv_bar
            }

        adjacency_matrix = dataset.obsp["adjacency_matrix"].tocoo()

        num_cells, _ = adjacency_matrix.shape
        adjacency_list = [[] for _ in range(num_cells)]
        for x, y in zip(*adjacency_matrix.nonzero()):
            adjacency_list[x].append(y)

        dataset.obs["adjacency_list"] = adjacency_list
        
        if "M" in dataset.uns:
            if context == "numpy":
                if f"{replicate}" in dataset.uns["M"]:
                    replicate_M = make_hdf5_compatible(dataset.uns["M"][f"{replicate}"])
                    dataset.uns["M"] = {f"{replicate}": replicate_M}
    
        if "M_bar" in dataset.uns:
            if context == "numpy":
                if f"{replicate}" in dataset.uns["M_bar"]:
                    replicate_M_bar = make_hdf5_compatible(dataset.uns["M_bar"][f"{replicate}"])
                    dataset.uns["M_bar"] = {f"{replicate}": replicate_M_bar}
        
        if "spicemixplus_hyperparameters" in dataset.uns:
            if context == "numpy":
                if "prior_x" in dataset.uns["spicemixplus_hyperparameters"]:
                    prior_x = make_hdf5_compatible(dataset.uns["spicemixplus_hyperparameters"]["prior_x"])
                    dataset.uns["spicemixplus_hyperparameters"]["prior_x"] = prior_x
 
        if "X" in dataset.obsm:
            if context == "numpy":
                replicate_X = make_hdf5_compatible(dataset.obsm["X"])
                dataset.obsm["X"] = replicate_X

        if context != "numpy":
            dataset.obsp["adjacency_matrix"] = convert_numpy_to_pytorch_sparse_coo(adjacency_matrix, context)

    return datasets, replicate_names

def save_anndata(filepath: Union[str, Path], datasets: Sequence[SpiceMixDataset], replicate_names: Sequence[str]):
    """Save SpiceMixPlus state as AnnData object.

    """
    dataset_copies = []
    for replicate, dataset in zip(replicate_names, datasets):
        dataset_copy = ad.AnnData(
                X=dataset.X,
                obs=dataset.obs,
                var=dataset.var,
                uns=dataset.uns,
                obsm=dataset.obsm,
                obsp=dataset.obsp,
        )

        dataset_copy.obsp["adjacency_matrix"] = make_hdf5_compatible(dataset.obsp["adjacency_matrix"])
       
        if "Sigma_x_inv" in dataset_copy.uns:
            replicate_Sigma_x_inv = dataset_copy.uns["Sigma_x_inv"][f"{replicate}"]
            if f"{replicate}" in dataset_copy.uns["Sigma_x_inv"]:
                # Using a sentinel value - hopefully this can be fixed in the future!
                if replicate_Sigma_x_inv is None: 
                    replicate_Sigma_x_inv = -1
                else:
                    replicate_Sigma_x_inv = make_hdf5_compatible(replicate_Sigma_x_inv)
        
            dataset_copy.uns["Sigma_x_inv"] = {f"{replicate}": replicate_Sigma_x_inv}
        
        if "Sigma_x_inv_bar" in dataset_copy.uns:
            replicate_Sigma_x_inv_bar = dataset_copy.uns["Sigma_x_inv_bar"][f"{replicate}"]
            if f"{replicate}" in dataset_copy.uns["Sigma_x_inv_bar"]:
                # Using a sentinel value - hopefully this can be fixed in the future!
                if replicate_Sigma_x_inv_bar is None: 
                    replicate_Sigma_x_inv_bar = -1
                else:
                    replicate_Sigma_x_inv_bar = make_hdf5_compatible(replicate_Sigma_x_inv_bar)
        
            dataset_copy.uns["Sigma_x_inv_bar"] = {f"{replicate}": replicate_Sigma_x_inv_bar}

        if "M" in dataset_copy.uns:
            if f"{replicate}" in dataset_copy.uns["M"]:
                replicate_M = make_hdf5_compatible(dataset_copy.uns["M"][f"{replicate}"])
                dataset_copy.uns["M"] = {f"{replicate}": replicate_M}
    
        if "M_bar" in dataset_copy.uns:
            if f"{replicate}" in dataset_copy.uns["M_bar"]:
                replicate_M_bar = make_hdf5_compatible(dataset_copy.uns["M_bar"][f"{replicate}"])
                dataset_copy.uns["M_bar"] = {f"{replicate}": replicate_M_bar}
        
        if "spicemixplus_hyperparameters" in dataset_copy.uns:
            if "prior_x" in dataset_copy.uns["spicemixplus_hyperparameters"]:
                prior_x = make_hdf5_compatible(dataset_copy.uns["spicemixplus_hyperparameters"]["prior_x"])
                dataset_copy.uns["spicemixplus_hyperparameters"]["prior_x"] = prior_x
            
            dataset_copy.uns["spicemixplus_hyperparameters"]["name"] = dataset.name
        else:
            dataset_copy.uns["spicemixplus_hyperparameters"] = {"name": dataset.name}

                
        if "X" in dataset_copy.obsm:
            replicate_X = make_hdf5_compatible(dataset_copy.obsm["X"])
            dataset_copy.obsm["X"] = replicate_X
 
        dataset_copies.append(dataset_copy)

        if "adjacency_list" in dataset_copy.obs:
            del dataset_copy.obs["adjacency_list"]


    dataset_names = [dataset.name for dataset in datasets]
    merged_dataset = ad.concat(dataset_copies, label="batch", keys=dataset_names, merge="unique", uns_merge="unique", pairwise=True)
    merged_dataset.write(filepath)
   
    reloaded_dataset = ad.read_h5ad(filepath)

    return merged_dataset

def make_hdf5_compatible(array: Union[torch.Tensor, np.ndarray]):
    """Convert tensors to Numpy array for compatibility with HDF5.

    Args:
        array: array to convert to Numpy.
    """

    if type(array) == torch.Tensor:
        cpu_tensor = array.detach().cpu()
        if cpu_tensor.is_sparse:
            return cpu_tensor.to_dense()

        return cpu_tensor.numpy()

    return array

def convert_numpy_to_pytorch_sparse_coo(numpy_coo, context):
    indices = numpy_coo.nonzero()
    values = numpy_coo.data[numpy_coo.data.nonzero()]

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    size = numpy_coo.shape

    torch_coo = torch.sparse_coo_tensor(i, v, size=size, **context)

    return torch_coo
