from pathlib import Path
from typing import Optional, Sequence, Union

import anndata as ad
import awkward as ak
import numpy as np
import torch
from scipy.sparse import csr_array, issparse

from popari.components import PopariDataset
from popari.util import concatenate, unconcatenate


def load_anndata(filepath: Union[str, Path]):
    """Load AnnData object from h5ad file and reformat for Popari."""

    merged_dataset = ad.read_h5ad(filepath)

    datasets, replicate_names = unmerge_anndata(merged_dataset)

    return datasets, replicate_names


def unmerge_anndata(merged_dataset: ad.AnnData):
    """Unmerge composite AnnData object into constituent datasets."""

    merged_dataset.X = csr_array(merged_dataset.X)
    datasets = unconcatenate(merged_dataset)

    for dataset in datasets:
        replicate_string = f"{dataset.name}"
        if "Sigma_x_inv" in dataset.uns:
            # Keep only Sigma_x_inv corresponding to a particular replicate
            replicate_Sigma_x_inv = dataset.uns["Sigma_x_inv"][replicate_string]
            if np.isscalar(replicate_Sigma_x_inv) and replicate_Sigma_x_inv == -1:
                replicate_Sigma_x_inv = None

            dataset.uns["Sigma_x_inv"] = {
                replicate_string: replicate_Sigma_x_inv,
            }

        if "Sigma_x_inv_bar" in dataset.uns:
            # Keep only Sigma_x_inv_bar corresponding to a particular replicate
            replicate_Sigma_x_inv_bar = dataset.uns["Sigma_x_inv_bar"][replicate_string]
            if np.isscalar(replicate_Sigma_x_inv_bar) and replicate_Sigma_x_inv_bar == -1:
                replicate_Sigma_x_inv_bar = None

            dataset.uns["Sigma_x_inv_bar"] = {
                replicate_string: replicate_Sigma_x_inv_bar,
            }

        # Hacks to load adjacency matrices efficiently
        if "adjacency_matrix" in dataset.uns:
            dataset.obsp["adjacency_matrix"] = csr_array(
                dataset.uns["adjacency_matrix"][replicate_string],
            )

        adjacency_matrix = dataset.obsp["adjacency_matrix"].tocoo()

        num_cells, _ = adjacency_matrix.shape
        adjacency_list = [[] for _ in range(num_cells)]
        for x, y in zip(*adjacency_matrix.nonzero()):
            adjacency_list[x].append(y)

        dataset.obsm["adjacency_list"] = ak.Array(adjacency_list)

        # if "M" in dataset.uns:
        #     if replicate_string in dataset.uns["M"]:
        #         replicate_M = make_hdf5_compatible(dataset.uns["M"][replicate_string])
        #         dataset.uns["M"] = {replicate_string: replicate_M}

        # if "M_bar" in dataset.uns:
        #     if replicate_string in dataset.uns["M_bar"]:
        #         replicate_M_bar = make_hdf5_compatible(
        #             dataset.uns["M_bar"][replicate_string],
        #         )
        #         dataset.uns["M_bar"] = {replicate_string: replicate_M_bar}

        if "popari_hyperparameters" in dataset.uns:
            # if "prior_x" in dataset.uns["popari_hyperparameters"]:
            #     prior_x = make_hdf5_compatible(
            #         dataset.uns["popari_hyperparameters"]["prior_x"],
            #     )
            #     dataset.uns["popari_hyperparameters"]["prior_x"] = prior_x

            if "spatial_affinity_groups" in dataset.uns["popari_hyperparameters"]:
                name_parts = dataset.name.split("_level_")
                if len(name_parts) > 1:
                    *_, level = name_parts
                    level = int(level)
                else:
                    level = 0

                groups = dataset.uns["popari_hyperparameters"]["spatial_affinity_groups"]
                filtered_groups = {}
                for group, group_replicates in groups.items():
                    group_name_parts = group.split("_level_")
                    if len(group_name_parts) > 1:
                        *_, group_level = group_name_parts
                        group_level = int(group_level)
                    else:
                        group_level = 0

                    if group_level == level:
                        filtered_groups[group] = group_replicates

                dataset.uns["popari_hyperparameters"]["spatial_affinity_groups"] = filtered_groups

            if "metagene_groups" in dataset.uns["popari_hyperparameters"]:
                name_parts = dataset.name.split("_level_")
                if len(name_parts) > 1:
                    *_, level = name_parts
                    level = int(level)
                else:
                    level = 0

                groups = dataset.uns["popari_hyperparameters"]["metagene_groups"]
                filtered_groups = {}
                for group, group_replicates in groups.items():
                    group_name_parts = group.split("_level_")
                    if len(group_name_parts) > 1:
                        *_, group_level = group_name_parts
                        group_level = int(group_level)
                    else:
                        group_level = 0

                    if group_level == level:
                        filtered_groups[group] = group_replicates

                dataset.uns["popari_hyperparameters"]["metagene_groups"] = filtered_groups

        # if "X" in dataset.obsm:
        #     replicate_X = make_hdf5_compatible(dataset.obsm["X"])
        #     dataset.obsm["X"] = replicate_X

    replicate_names = [dataset.name for dataset in datasets]
    return datasets, replicate_names


def merge_anndata(datasets: Sequence[PopariDataset], ignore_raw_data: bool = False):
    """Merge multiple PopariDatasets into a single AnnData object (for
    storage)."""

    dataset_copies = []
    for dataset in datasets:
        replicate = dataset.name
        replicate_string = f"{replicate}"
        if ignore_raw_data:
            dataset.X = csr_array(dataset.X.shape)
        else:
            dataset.X = csr_array(dataset.X)

        dataset_copy = PopariDataset(dataset, dataset.name)

        # Hacks to store adjacency matrices efficiently
        if "adjacency_matrix" in dataset.obsp:
            adjacency_matrix = make_hdf5_compatible(dataset.obsp["adjacency_matrix"])
            dataset_copy.uns["adjacency_matrix"] = {replicate_string: adjacency_matrix}
        elif "adjacency_matrix" in dataset.uns:
            dataset_copy.uns["adjacency_matrix"] = {
                replicate_string: make_hdf5_compatible(adjacency_matrix)
                for replicate_string, adjacency_matrix in dataset.uns["adjacency_matrix"]
            }

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
                    replicate_Sigma_x_inv_bar = make_hdf5_compatible(
                        replicate_Sigma_x_inv_bar,
                    )
            dataset_copy.uns["Sigma_x_inv_bar"] = {
                replicate_string: replicate_Sigma_x_inv_bar,
            }

        # if "M" in dataset_copy.uns:
        # if replicate_string in dataset_copy.uns["M"]:
        #     replicate_M = make_hdf5_compatible(
        #         dataset_copy.uns["M"][replicate_string],
        #     )
        #     dataset_copy.uns["M"] = {replicate_string: replicate_M}

        # if "M_bar" in dataset_copy.uns:
        # if replicate_string in dataset_copy.uns["M_bar"]:
        #     replicate_M_bar = make_hdf5_compatible(
        #         dataset_copy.uns["M_bar"][replicate_string],
        #     )
        #     dataset_copy.uns["M_bar"] = {replicate_string: replicate_M_bar}

        # if "popari_hyperparameters" in dataset_copy.uns:
        # if "prior_x" in dataset_copy.uns["popari_hyperparameters"]:
        #     prior_x = make_hdf5_compatible(
        #         dataset_copy.uns["popari_hyperparameters"]["prior_x"],
        #     )
        #     dataset_copy.uns["popari_hyperparameters"]["prior_x"] = prior_x

        if "dataset_name" not in dataset_copy.uns:
            dataset_copy.uns["dataset_name"] = dataset.name

        # if "X" in dataset_copy.obsm:
        #     replicate_X = make_hdf5_compatible(dataset_copy.obsm["X"])
        #     dataset_copy.obsm["X"] = replicate_X
        dataset_copies.append(dataset_copy)

        if "adjacency_list" in dataset_copy.obs:
            del dataset_copy.obsm["adjacency_list"]

    merged_dataset = concatenate(dataset_copies, join="outer")

    return merged_dataset


def save_anndata(
    filepath: Union[str, Path],
    datasets: Sequence[PopariDataset],
    ignore_raw_data: bool = False,
):
    """Save Popari state as AnnData object."""

    merged_dataset = merge_anndata(datasets, ignore_raw_data=ignore_raw_data)
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
