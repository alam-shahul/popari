from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pymetis
from anndata import AnnData
from numpy.typing import NDArray
from pymetis import Options, part_graph
from scipy.sparse import csr_array
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

from popari._popari_dataset import PopariDataset


class Downsampler(ABC):
    """Abstraction for downsampling spots to bins."""

    @abstractmethod
    def generate_bin_assignments(
        self,
        dataset: PopariDataset,
        bin_assignments_key: str,
        coordinates_key: str,
        downsample_rate: float,
        **bin_assignment_kwargs,
    ):
        """Generate bin assignments.

        Implementation must try to achieve the `downsample_rate` when binning, i.e. the number of
        bins should approximately equal `downsample_rate` fraction of the number of original spots.

        Returns:

        """

    @abstractmethod
    def update_binning_kwargs(previous_dataset: PopariDataset, kwargs: dict):
        """Update kwargs for the next round of bin assignment (for hierarchical
        binning).

        Since binning may need to be consistent between datasets, this allows the next round of
        binning to be informed by the results from the previous_dataset.

        Args:
            previous_dataset: the binning result from the previous round
            kwargs: keyword arguments used for the previous round of binning

        Returns:
            Updated kwargs for the next round of bin assignment.

        """

    def one_hot_encode(self, data: NDArray):
        one_hot_encoder = OneHotEncoder()
        one_hot_encoding = one_hot_encoder.fit_transform(np.array(data).reshape(-1, 1))
        one_hot_encoding = csr_array(one_hot_encoding).astype(int)

        return one_hot_encoding

    def bin_expression(self, dataset: PopariDataset, bin_assignments_key: str):

        bin_assignments = dataset.obsm[bin_assignments_key].T

        num_spots, num_genes = dataset.X.shape
        num_bins, _ = bin_assignments.shape

        binned_expression = bin_assignments @ dataset.X

        return binned_expression

    def bin_coordinates(self, dataset: PopariDataset, bin_assignments_key: str, coordinates_key: str):
        coordinates = dataset.obsm[coordinates_key]
        bin_assignments = dataset.obsm[bin_assignments_key].T

        num_assignments = np.expand_dims(bin_assignments.sum(axis=1), axis=1)
        summed_coordinates = bin_assignments @ coordinates

        binned_coordinates = summed_coordinates / num_assignments

        return binned_coordinates

    def downsample(
        self,
        dataset: PopariDataset,
        coordinates_key: str = "spatial",
        bin_assignments_key: str = "bin_assignments",
        downsample_rate: float = 0.2,
        **bin_assignment_kwargs,
    ):
        effective_kwargs = self.generate_bin_assignments(
            dataset,
            bin_assignments_key=bin_assignments_key,
            coordinates_key=coordinates_key,
            downsample_rate=downsample_rate,
            **bin_assignment_kwargs,
        )
        binned_expression = self.bin_expression(dataset, bin_assignments_key=bin_assignments_key)

        binned_dataset = AnnData(X=binned_expression)
        binned_dataset.obsm[bin_assignments_key] = dataset.obsm[bin_assignments_key].T
        binned_dataset.obsm[coordinates_key] = self.bin_coordinates(
            dataset,
            bin_assignments_key=bin_assignments_key,
            coordinates_key=coordinates_key,
        )
        binned_dataset.var_names = dataset.var_names

        for key, kwarg in effective_kwargs.items():
            binned_dataset.uns[key] = kwarg

        return binned_dataset, effective_kwargs


class GridDownsampler(Downsampler):
    """Overlay grids, and use these to downsample/bin a spatial dataset."""

    def update_binning_kwargs(previous_dataset: PopariDataset, kwargs: dict):
        for key, kwarg in kwargs.items():
            kwargs[key] = previous_dataset.uns.get(key, kwarg)

        return kwargs

    def generate_bin_assignments(
        self,
        dataset: PopariDataset,
        coordinates_key: str,
        bin_assignments_key: str,
        downsample_rate: float,
        chunks: int = 2,
        chunk_size: Optional[int] = None,
        chunk_1d_density: Optional[int] = None,
    ):
        coordinates = dataset.obsm[coordinates_key]
        binned_coordinates, chunk_size, chunk_1d_density = chunked_downsample_on_grid(
            coordinates,
            downsample_rate=downsample_rate,
            chunks=chunks,
            chunk_size=chunk_size,
            downsampled_1d_density=chunk_1d_density,
        )

        filtered_bin_coordinates = filter_gridpoints(coordinates, binned_coordinates, num_jobs=2)

        neigh = NearestNeighbors(n_neighbors=1, n_jobs=2)
        neigh.fit(filtered_bin_coordinates)
        indices = np.squeeze(neigh.kneighbors(coordinates, return_distance=False))

        bin_assignments = self.one_hot_encode(indices).T
        dataset.obsm[bin_assignments_key] = bin_assignments.T

        return {
            "chunks": None,
            "chunk_size": chunk_size,
            "chunk_1d_density": chunk_1d_density,
        }


class PartitionDownsampler(Downsampler):
    """Use graph partitioning on the Delaunay triangulation to bin a dataset."""

    def update_binning_kwargs(previous_dataset: PopariDataset, kwargs: dict):
        return kwargs

    def generate_bin_assignments(
        self,
        dataset: PopariDataset,
        adjacency_list_key: str,
        coordinates_key: str,
        bin_assignments_key: str,
        downsample_rate: float = 0.2,
    ):

        num_bins = round(len(dataset) * downsample_rate)
        adjacency_list = dataset.obsm[adjacency_list_key]

        options = Options(seed=0)  # TODO: this doesn't seem to work...
        _, indices = part_graph(num_bins, adjacency_list, options=options)

        # index_reducer = {old_index: new_index for new_index, old_index in enumerate(set(indices))}
        # reduced_indices = [index_reducer[index] for index in indices]

        bin_assignments = self.one_hot_encode(indices).T

        dataset.obsm[bin_assignments_key] = bin_assignments.T

        return {
            "adjacency_list_key": adjacency_list_key,
        }


def chunked_coordinates(coordinates: NDArray, chunks: int = None, step_size: float = None):
    """Split a list of 2D coordinates into local chunks.

    Args:
        chunks: number of equal-sized chunks to split horizontal axis. Vertical chunks are constructed
            with the same chunk size determined by splitting the horizontal axis.

    Yields:
        The next chunk of coordinates, in row-major order.

    """

    num_points, _ = coordinates.shape

    horizontal_base, vertical_base = np.min(coordinates, axis=0)
    horizontal_range, vertical_range = np.ptp(coordinates, axis=0)

    if step_size is None and chunks is None:
        raise ValueError("One of `chunks` or `step_size` must be specified.")

    if step_size is None:
        horizontal_borders, step_size = np.linspace(
            horizontal_base,
            horizontal_base + horizontal_range,
            chunks + 1,
            retstep=True,
        )
    elif chunks is None:
        horizontal_borders = np.arange(horizontal_base, horizontal_base + horizontal_range, step_size)

        # Adding endpoint
        horizontal_borders = np.append(horizontal_borders, horizontal_borders[-1] + step_size)

    vertical_borders = np.arange(vertical_base, vertical_base + vertical_range, step_size)

    # Adding endpoint
    vertical_borders = np.append(vertical_borders, vertical_borders[-1] + step_size)

    for i in range(len(horizontal_borders) - 1):
        horizontal_low, horizontal_high = horizontal_borders[i : i + 2]
        for j in range(len(vertical_borders) - 1):
            vertical_low, vertical_high = vertical_borders[j : j + 2]
            horizontal_mask = (coordinates[:, 0] > horizontal_low) & (coordinates[:, 0] <= horizontal_high)
            vertical_mask = (coordinates[:, 1] > vertical_low) & (coordinates[:, 1] <= vertical_high)
            chunk_coordinates = coordinates[horizontal_mask & vertical_mask]

            chunk_data = {
                "horizontal_low": horizontal_low,
                "horizontal_high": horizontal_high,
                "vertical_low": vertical_low,
                "vertical_high": vertical_high,
                "step_size": step_size,
                "chunk_coordinates": chunk_coordinates,
            }
            yield chunk_data


def finetune_chunk_number(coordinates: NDArray, chunks: int, downsample_rate: float, max_nudge: Optional[int] = None):
    """Heuristically search for a chunk number that cleanly splits up points.

    Using a linear search, searches for a better value of the ``chunks`` value such that
    the average points-per-chunk is closer to the number of points that will be in the chunk.

    Args:
        coordinates: original spot coordinates
        chunks: number of equal-sized chunks to split horizontal axis
        downsample_rate: approximate desired ratio of meta-spots to spots after downsampling

    Returns:
        finetuned number of chunks

    """
    if max_nudge is None:
        max_nudge = chunks // 2

    num_points, num_dimensions = coordinates.shape
    target_points = num_points * downsample_rate

    horizontal_base, vertical_base = np.min(coordinates, axis=0)
    horizontal_range, vertical_range = np.ptp(coordinates, axis=0)

    direction = 0
    for chunk_nudge in range(max_nudge):
        valid_chunks = []
        for chunk_data in chunked_coordinates(coordinates, chunks=chunks + direction * chunk_nudge):
            if len(chunk_data["chunk_coordinates"]) > 0:
                valid_chunks.append(chunk_data)

        points_per_chunk = num_points * downsample_rate / len(valid_chunks)
        downsampled_1d_density = int(np.round(np.sqrt(points_per_chunk)))
        new_direction = 1 - 2 * ((downsampled_1d_density**2) > points_per_chunk)
        if direction == 0:
            direction = new_direction
        else:
            if direction != new_direction:
                break

    return chunks + direction * chunk_nudge


def chunked_downsample_on_grid(
    coordinates: NDArray,
    downsample_rate: float,
    chunks: Optional[int] = None,
    chunk_size: Optional[float] = None,
    downsampled_1d_density: Optional[int] = None,
):
    """Downsample spot coordinates to a square grid of meta-spots using chunks.

    By chunking the coordinates, we can:

    1. Remove unused chunks.
    2. Estimate the density of spots at chunk-sized resolution.

    We use this information when downsampling in order to

    Args:
        coordinates: original spot coordinates
        chunks: number of equal-sized chunks to split horizontal axis
        downsample_rate: approximate desired ratio of meta-spots to spots after downsampling

    Returns:
        coordinates of downsampled meta-spots

    """

    num_points, num_dimensions = coordinates.shape

    horizontal_base, vertical_base = np.min(coordinates, axis=0)
    horizontal_range, vertical_range = np.ptp(coordinates, axis=0)

    if chunks is not None:
        chunks = finetune_chunk_number(coordinates, chunks, downsample_rate)

    valid_chunks = []
    for chunk_data in chunked_coordinates(coordinates, chunks=chunks, step_size=chunk_size):
        if len(chunk_data["chunk_coordinates"]) > 0:
            valid_chunks.append(chunk_data)

    points_per_chunk = num_points * downsample_rate / len(valid_chunks)

    if downsampled_1d_density is None:
        downsampled_1d_density = int(np.round(np.sqrt(points_per_chunk)))

    if points_per_chunk < 2:
        raise ValueError("Chunk density is < 1")

    all_new_coordinates = []
    for index, chunk_data in enumerate(valid_chunks):
        horizontal_low = chunk_data["horizontal_low"]
        horizontal_high = chunk_data["horizontal_high"]
        vertical_low = chunk_data["vertical_low"]
        vertical_high = chunk_data["vertical_high"]
        step_size = chunk_data["step_size"]

        x = np.linspace(horizontal_low, horizontal_high, downsampled_1d_density, endpoint=False)
        if np.allclose(horizontal_high, horizontal_base + horizontal_range):
            x_gap = x[-1] - x[-2]
            x = np.append(x, x.max() + x_gap)

        y = np.linspace(vertical_low, vertical_high, downsampled_1d_density, endpoint=False)
        if np.allclose(vertical_high, vertical_base + vertical_range):
            y_gap = y[-1] - y[-2]
            y = np.append(y, y.max() + y_gap)

        xv, yv = np.meshgrid(x, y)

        new_coordinates = np.array(list(zip(xv.flat, yv.flat)))
        all_new_coordinates.append(new_coordinates)

    new_coordinates = np.vstack(all_new_coordinates)
    new_coordinates = np.unique(new_coordinates, axis=0)

    return new_coordinates, step_size, downsampled_1d_density


def filter_gridpoints(spot_coordinates: NDArray, grid_coordinates: NDArray, num_jobs: int):
    """Use nearest neighbors approach to filter out relevant grid coordinates.

    Keeps only the grid coordinates that are mapped to at least a single original spot.

    Args:
        spot_coordinates: coordinates of original spots
        grid_coordinates: coordinates of downsampled grid

    Returns:
        metaspots that meet the filtering criterion

    """

    spot_to_metaspot_mapper = NearestNeighbors(n_neighbors=1, n_jobs=num_jobs)
    spot_to_metaspot_mapper.fit(grid_coordinates)

    indices = spot_to_metaspot_mapper.kneighbors(spot_coordinates, return_distance=False)

    used_bins = set(indices.flat)

    filtered_bin_coordinates = grid_coordinates[list(used_bins)]

    return filtered_bin_coordinates
