import numpy as np
import torch

from popari._graph import graph_neighbors


class IndependentSet:
    """Yield batches whose nodes are pairwise nonadjacent."""

    def __init__(self, adjacency, device, batch_size=50, indices=None):
        self.N = adjacency.shape[0]
        self.adjacency = adjacency
        self.batch_size = batch_size
        self.indices = np.arange(self.N) if indices is None else np.asarray(indices, dtype=np.int64)
        self.indices_remaining = None
        self.device = device

    def __iter__(self):
        self.indices_remaining = set(self.indices)
        return self

    def __next__(self):
        if len(self.indices_remaining) == 0:
            raise StopIteration

        valid_indices = sample_graph_iid(self.adjacency, self.indices_remaining, self.batch_size)
        self.indices_remaining -= set(valid_indices)
        return torch.tensor(valid_indices, device=self.device, dtype=torch.long)


def sample_graph_iid(adjacency, indices_remaining, sample_size):
    valid_indices = []
    excluded_indices = set()
    effective_batch_size = min(sample_size, len(indices_remaining))
    candidate_indices = np.random.choice(
        list(indices_remaining),
        size=effective_batch_size,
        replace=False,
    )
    for index in candidate_indices:
        if index not in excluded_indices:
            valid_indices.append(index)
            excluded_indices.update(graph_neighbors(adjacency, index))

    return valid_indices
