import os, pickle, logging
from matplotlib import pyplot as plt

import numpy as np
import anndata as ad
import torch

from util import print_datetime, parse_suffix


def load_expression(filename):
    if filename.suffix == '.pkl':
        with open(filename, 'rb') as f:
            expr = pickle.load(f)
    elif filename.suffix in ['.txt', '.tsv']:
        expr = np.loadtxt(filename, dtype=np.float)
    else:
        raise ValueError(f'Invalid file format {filename}')
    N, G = expr.shape
    logging.info(f'{print_datetime()}Loaded {N} cells and {G} genes from {filename}')
    return expr


def load_edges(filename, num_cells):
    edges = np.loadtxt(filename, dtype=np.int)
    assert edges.shape[1] == 2, f'Detected an edge that does not contain two nodes'
    assert np.all(0 <= edges) and np.all(edges < num_cells), f'Node ID exceeded range [0, N)'
    edges = np.sort(edges, axis=1)
    assert np.all(edges[:, 0] < edges[:, 1]), f'Detected {(edges[:, 0] == edges[:, 1]).sum()} self-loop(s)'
    if len(np.unique(edges, axis=0)) != len(edges):
        logging.warning(
            f'Detected {len(edges)-len(np.unique(edges, axis=0))} duplicate edge(s) from {len(edges)} loaded edges.'
            f'Duplicate edges are discarded.'
        )
        edges = np.unique(edges, axis=0)
    logging.info(f'{print_datetime()}Loaded {len(edges)} edges from {filename}')
    E = [[] for _ in range(num_cells)]
    for (u, v) in edges:
        E[u].append(v)
        E[v].append(u)
    return E


def load_genelist(filename):
    genes = np.loadtxt(filename, dtype=str)
    logging.info(f'{print_datetime()}Loaded {len(genes)} genes from {filename}')
    return genes

def load_anndata(filepath, replicate_names, context):
    """Load AnnData object from h5ad file and reformat for SpiceMixPlus.

    When reloading the object, we need to reformat it for compatibility with SpiceMixPlus:

        1. Load adjacency matrix as torch.sparse.coo_tensor
        2. Generate adjacency list from adjacency matrix

    """

    merged_dataset = ad.read_h5ad(filepath)

    indices = merged_dataset.obs.groupby("batch").indices.values()
    datasets = [merged_dataset[index] for index in indices]
   
    if len(replicate_names) != len(datasets):
        raise ValueError("List of replicates does not match number of datasets stored in AnnData object.")
        
    for replicate, dataset in zip(replicate_names, datasets):
        # Keep only Sigma_x_inv corresponding to a particular replicate
        replicate_Sigma_x_inv = dataset.uns["Sigma_x_inv"][f"{replicate}"]
        if replicate_Sigma_x_inv is -1:
            replicate_Sigma_x_inv = None
        else:
            replicate_Sigma_x_inv = torch.tensor(replicate_Sigma_x_inv, **context)

        dataset.uns["Sigma_x_inv"] = {
            f"{replicate}": replicate_Sigma_x_inv
        }

        adjacency_matrix = dataset.obsp["adjacency_matrix"].tocoo()

        num_cells, _ = adjacency_matrix.shape
        adjacency_list = [[] for _ in range(num_cells)]
        for x, y in zip(*adjacency_matrix.nonzero()):
            adjacency_list[x].append(y)

        dataset.obs["adjacency_list"] = adjacency_list

        indices = adjacency_matrix.nonzero()
        values = adjacency_matrix.data[adjacency_matrix.data.nonzero()]

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        size = adjacency_matrix.shape

        adjacency_matrix = torch.sparse_coo_tensor(i, v, size=size, **context)

        dataset.obsp["adjacency_matrix"] = adjacency_matrix

    return datasets

def save_anndata(filepath, datasets, replicate_names):
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

        dataset_copy.obsp["adjacency_matrix"] = dataset.obsp["adjacency_matrix"].detach().cpu().to_dense()
        
        replicate_Sigma_x_inv = dataset_copy.uns["Sigma_x_inv"][f"{replicate}"]
        if f"{replicate}" in dataset_copy.uns["Sigma_x_inv"]:
            # Using a sentinel value - hopefully this can be fixed in the future!
            if replicate_Sigma_x_inv == None: 
                replicate_Sigma_x_inv = -1
            else:
                replicate_Sigma_x_inv = replicate_Sigma_x_inv.detach().cpu().numpy()
        
            dataset_copy.uns["Sigma_x_inv"] = {f"{replicate}": replicate_Sigma_x_inv}

        if f"{replicate}" in dataset_copy.uns["M"]:
            replicate_M = dataset_copy.uns["M"][f"{replicate}"].detach().cpu().numpy()
            dataset_copy.uns["M"] = {f"{replicate}": replicate_M}
    
        if f"{replicate}" in dataset_copy.uns["M_bar"]:
            replicate_M_bar = dataset_copy.uns["M_bar"][f"{replicate}"].detach().cpu().numpy()
            dataset_copy.uns["M_bar"] = {f"{replicate}": replicate_M_bar}
        
        replicate_X = dataset_copy.obsm["X"].detach().cpu().numpy()
        dataset_copy.obsm["X"] = replicate_X
 
        dataset_copies.append(dataset_copy)

        del dataset_copy.obs["adjacency_list"]


    merged_dataset = ad.concat(dataset_copies, label="batch", merge="unique", uns_merge="unique", pairwise=True)
    merged_dataset.write(filepath)
   
    reloaded_dataset = ad.read_h5ad(filepath)

    return merged_dataset
