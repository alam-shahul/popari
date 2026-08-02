import awkward as ak
import numpy as np
import pandas as pd
import torch
from kneed import KneeLocator
from scipy.sparse import csr_array, sparray


class NesterovGD:
    """Optimizer that implements Nesterov's Accelerated Gradient Descent.

    See below for implementation details:
    # https://blogs.princeton.edu/imabandit/2013/04/01/acceleratedgradientdescent/

    Attributes:
        parameters: object to optimize with Nesterov
        step_size: size of gradient update

    """

    def __init__(self, parameters: torch.Tensor, step_size: float):
        """Initialize Nesterov optimization.

        Args:
            parameters: object to optimize with Nesterov
            step_size: size of gradient update

        """
        self.parameters = parameters
        self.step_size = step_size
        # self.y = x.clone()
        self.y = torch.zeros_like(parameters)
        # self.lam = 0
        self.k = 0

    def set_parameters(self, parameters):
        """Reset parameters.

        Args:
            parameters: object to optimize with Nesterov

        """
        self.parameters = parameters

    def step(self, grad: torch.Tensor):
        """Update parameters according to state and step size.

        Args:
            grad: naive gradient for updating parameters

        """

        # method 1
        # lam_new = (1 + np.sqrt(1 + 4 * self.lam ** 2)) / 2
        # gamma = (1 - self.lam) / lam_new
        # method 2
        self.k += 1
        gamma = -(self.k - 1) / (self.k + 2)
        # method 3 - GD
        # gamma = 0
        # y_new = self.x.sub(grad, alpha=self.step_size)
        y_new = self.parameters - grad * self.step_size  # use addcmul
        self.y = (self.y * gamma).add(y_new, alpha=(1 - gamma))
        self.parameters = self.y
        # self.lam = lam_new
        self.y = y_new

        return self.parameters


@torch.no_grad()
def project_M(M, M_constraint):
    result = M.clone()
    if M_constraint == "simplex":
        result = project2simplex(result, dim=0, zero_threshold=1e-5)
    elif M_constraint == "unit sphere":
        result = M.div(torch.linalg.norm(result, ord=2, dim=0, keepdim=True))
    elif M_constraint == "nonneg unit sphere":
        result = M.clip(1e-10).div(torch.linalg.norm(result, ord=2, dim=0, keepdim=True))
    else:
        raise NotImplementedError
    return result


def project_M_(M, M_constraint):
    result = M.clone()
    if M_constraint == "simplex":
        result = project2simplex_(result, dim=0, zero_threshold=1e-5)
    elif M_constraint == "unit sphere":
        result = M.div(torch.linalg.norm(result, ord=2, dim=0, keepdim=True))
    elif M_constraint == "nonneg unit sphere":
        result = M.clip(1e-10).div(torch.linalg.norm(result, ord=2, dim=0, keepdim=True))
    else:
        raise NotImplementedError
    return result


def project2simplex(y, dim: int = 0, zero_threshold: float = 1e-10) -> torch.Tensor:
    """Projects a matrix such that the columns (or rows) lie on the unit
    simplex.

    See https://math.stackexchange.com/questions/2402504/orthogonal-projection-onto-the-unit-simplex
    for a reference.

    The goal is to find a scalar mu such that || (y-mu)_+ ||_1 = 1

    Currently uses Newton's method to optimize || y - mu ||^2

    TODO: try implementing it this way instead: https://arxiv.org/pdf/1101.6081.pdf

    Args:
        y: list of vectors to be projected to unit simplex
        dim: dimension along which to project
        zero_threshold: threshold to treat as zero for numerical stability purposes

    """

    num_components = y.shape[dim]

    mu = (y.sum(dim=dim, keepdim=True) - 1) / num_components
    previous_derivative = derivative = None
    for _ in range(num_components):
        difference = y - mu
        derivative = -(difference > zero_threshold).sum(dim=dim, keepdim=True).to(y.dtype)
        assert -derivative.min() > 0, difference.clip(min=0).sum(dim).min()
        if previous_derivative is not None and (derivative == previous_derivative).all():
            break
        objective_value = torch.clip(difference, min=zero_threshold).sum(dim=dim, keepdim=True) - 1
        newton_update = objective_value / derivative
        mu -= newton_update

        previous_derivative = derivative
    assert (derivative == previous_derivative).all()

    assert not torch.isnan(y).any(), y

    y = (y - mu).clip(min=zero_threshold)
    assert not torch.isnan(y).any(), (mu, derivative)

    assert y.sum(dim=dim).sub_(1).abs_().max() < 1e-3, y.sum(dim=dim).sub_(1).abs_().max()

    return y


def project2simplex_(y, dim: int = 0, zero_threshold: float = 1e-10) -> torch.Tensor:
    """(In-place) Projects a matrix such that the columns (or rows) lie on the
    unit simplex.

    See https://math.stackexchange.com/questions/2402504/orthogonal-projection-onto-the-unit-simplex
    for a reference.

    The goal is to find a scalar mu such that || (y-mu)_+ ||_1 = 1

    Currently uses Newton's method to optimize || y - mu ||^2

    TODO: try implementing it this way instead: https://arxiv.org/pdf/1101.6081.pdf

    Args:
        y: list of vectors to be projected to unit simplex
        dim: dimension along which to project
        zero_threshold: threshold to treat as zero for numerical stability purposes

    """
    y_copy = y.clone()
    num_components = y.shape[dim]

    y_copy.sub_(y_copy.sum(dim=dim, keepdim=True).sub_(1), alpha=1 / num_components)
    mu = y_copy.max(dim=dim, keepdim=True)[0].div_(2)
    derivative_prev, derivative = None, None
    for _ in range(num_components):
        difference = y_copy.sub(mu)
        objective_value = difference.clip_(min=zero_threshold).sum(dim, keepdim=True).sub_(1)
        derivative = difference.gt_(zero_threshold).sum(dim, keepdim=True)

        if derivative_prev is not None and (derivative == derivative_prev).all():
            break

        mu.addcdiv_(objective_value, derivative)
        derivative_prev = derivative

    y_copy.sub_(mu).clip_(min=zero_threshold)
    assert y_copy.sum(dim=dim).sub_(1).abs_().max() < 1e-4, y_copy.sum(dim=dim).sub_(1).abs_().max()
    return y_copy


class IndependentSet:
    """Iterator class that yields a list of batch_size independent nodes from a
    spatial graph.

    For each iteration, no pair of yielded nodes can be neighbors of each other according to the
    adjacency matrix.

    Attributes:
        N: number of nodes in graph
        adjacency_list: graph neighbor information stored in adjacency list format
        batch_size: number of nodes to draw independently every iteration

    """

    def __init__(self, adjacency_list, device, batch_size=50):
        self.N = len(adjacency_list)
        self.adjacency_list = adjacency_list
        self.batch_size = batch_size
        self.indices_remaining = None
        self.device = device

    def __iter__(self):
        """Return iterator over nodes in graph.

        Resets indices_remaining before returning the iterator.

        """
        self.indices_remaining = set(range(self.N))
        return self

    def __next__(self):
        """Returns the indices of batch_size nodes such that none of them are
        neighbors of each other.

        Makes sure selected nodes are not adjacent to each other, i.e., finds an
        independent set of `valid indices` in a greedy manner

        """
        if len(self.indices_remaining) == 0:
            raise StopIteration

        valid_indices = sample_graph_iid(self.adjacency_list, self.indices_remaining, self.batch_size)
        self.indices_remaining -= set(valid_indices)

        return torch.tensor(valid_indices, device=self.device, dtype=torch.long)


def sample_graph_iid(adjacency_list, indices_remaining, sample_size):
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
            excluded_indices |= set(adjacency_list[index])

    return valid_indices


def convert_numpy_to_pytorch_sparse_coo(numpy_coo, context):
    indices = np.array(numpy_coo.nonzero())
    values = numpy_coo.data[numpy_coo.data.nonzero()]

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    size = numpy_coo.shape

    torch_coo = torch.sparse_coo_tensor(i, v, size=size, **context)

    return torch_coo


def compute_neighborhood_enrichment(features: np.ndarray, adjacency_matrix: csr_array):
    r"""Compute the normalized enrichment of features in direct neighbors on a
    graph.

    Args:
        features: attributes on the nodes of the graph on which to compute enrichment.
        adjacency_matrix: sparse graph representation.

    """

    adjacency_matrix = (adjacency_matrix + adjacency_matrix.T).astype(bool).astype(adjacency_matrix.dtype)
    edges_per_node = np.squeeze(np.asarray(adjacency_matrix.sum(axis=0)))
    connected_mask = edges_per_node > 0

    features = features[connected_mask]
    adjacency_matrix = adjacency_matrix[connected_mask][:, connected_mask]

    total_counts = features.sum(axis=0)[:, np.newaxis]
    assert np.all(total_counts > 0)

    normalized_enrichment = ((1 / total_counts) * features.T) @ (
        1 / edges_per_node[connected_mask][:, np.newaxis] * adjacency_matrix.toarray() @ features
    )

    return np.asarray(normalized_enrichment)


def normalize_expression_by_threshold(dataset, thresholded_key: str = "elbowed_X", threshold: float = 99.0):
    """Replacement for Z-score threshold."""

    thresholded_expression = dataset.obsm[thresholded_key]
    expression_threshold = np.percentile(thresholded_expression, threshold, axis=0)
    mask = thresholded_expression > expression_threshold

    total_entities = mask.sum(axis=0)
    total_expression = (expression_threshold * mask).sum(axis=0)

    normalized_thresholded_expression = thresholded_expression / total_expression

    dataset.obsm["normalized_thresholded_expression"] = normalized_thresholded_expression

    return normalized_thresholded_expression


def smooth_metagene_expression(
    dataset,
    processed_key: str = "normalized_thresholded_expression",
    adjacency_list_key: str = "adjacency_list",
    adjacency_key: str = "adjacency_matrix",
):
    """"""
    processed_expression = dataset.obsm[processed_key]
    if adjacency_key in dataset.obsp:
        adjacency = csr_array(dataset.obsp[adjacency_key]).astype(bool).astype(float)
        degree = np.asarray(adjacency.sum(axis=1)).reshape(-1, 1)
        smoothed_expression = (processed_expression + adjacency @ processed_expression) / (degree + 1)
    else:
        adjacency_list = dataset.obsm[adjacency_list_key]
        smoothed_expression = np.zeros_like(processed_expression)
        for entity in np.arange(len(dataset)):
            adjacencies = adjacency_list[entity]
            neighbor_expressions = processed_expression[adjacencies]
            average_expression = (processed_expression[entity] + neighbor_expressions.sum(axis=0)) / (
                len(neighbor_expressions) + 1
            )
            smoothed_expression[entity] = average_expression

    dataset.obsm["smoothed_expression"] = smoothed_expression

    return smoothed_expression


def spatially_smooth_feature(labels, adjacency_list, max_smoothing_rounds=1, smoothing_threshold=0.5):
    """"""
    labels = np.asarray(labels)
    num_entities = len(labels)

    smoothed_labels = labels.copy()
    for _ in range(max_smoothing_rounds):
        new_labels = smoothed_labels.copy()
        for entity in np.arange(num_entities):
            current_cluster = smoothed_labels[entity]

            adjacencies = adjacency_list[entity]
            neighbor_labels = smoothed_labels[adjacencies]
            num_neighbors = len(neighbor_labels)
            if num_neighbors == 0:
                new_labels[entity] = current_cluster
                continue

            values, counts = np.unique(neighbor_labels, return_counts=True)

            max_index = np.argmax(counts)
            max_cluster = values[max_index]

            ratio = (counts[max_index] + (max_cluster == current_cluster)) / (num_neighbors + 1)
            if ratio >= smoothing_threshold:
                new_labels[entity] = max_cluster
            else:
                new_labels[entity] = current_cluster

        if np.all(smoothed_labels == new_labels):
            break

        smoothed_labels = new_labels

    return new_labels


def smooth_labels(
    dataset,
    label_key: str = "leiden",
    output_key: str = "smoothed_leiden",
    smoothing_threshold: float = 0.5,
    max_smoothing_rounds: int = 1,
    adjacency_list_key: str = "adjacency_list",
    adjacency_key: str = "adjacency_matrix",
):
    """"""
    if adjacency_list_key in dataset.obsm:
        adjacency_list = dataset.obsm[adjacency_list_key]
    else:
        adjacency_list = convert_adjacency_matrix_to_awkward_array(
            dataset.obsp[adjacency_key],
        )

    labels = dataset.obs[label_key]
    dataset.obs[output_key] = pd.Categorical(
        spatially_smooth_feature(
            labels,
            adjacency_list,
            max_smoothing_rounds,
            smoothing_threshold,
        ),
    )

    return dataset.obs[output_key]


def get_metagene_signature(
    metagene,
    gene_names,
    sensitivity: float = 1.0,
    type: str = "upregulated",
    show_plot: bool = False,
):
    """Use knee-detection algorithm to get top genes for metagene."""

    num_genes = len(metagene)

    sort_indices = np.argsort(metagene)
    curve = "convex" if type == "upregulated" else "concave"

    kneedle = KneeLocator(range(num_genes), metagene[sort_indices], S=sensitivity, curve=curve, direction="increasing")

    signature_range = slice(kneedle.knee, None) if type == "upregulated" else slice(None, kneedle.knee)
    signature_genes = gene_names[sort_indices[signature_range]]

    if show_plot:
        kneedle.plot_knee()

    return list(signature_genes)


def get_matching_order(scores):
    """Order the metagenes by correspondence to rows in AUROC score matrix.

    Return:
        a list specific the indices that place the metagenes in the best order

    """

    order = np.argsort(np.argmax(scores, axis=0) - np.max(scores, axis=0) / (np.max(scores) + 1))

    return order


def convert_adjacency_matrix_to_awkward_array(adjacency_matrix: sparray):
    """Convert COO adjacency matrix to ragged Awkward Array."""

    num_cells, _ = adjacency_matrix.shape
    adjacency_list = [[] for _ in range(num_cells)]
    for x, y in zip(*adjacency_matrix.nonzero()):
        adjacency_list[x].append(y)

    adjacency_list = ak.Array(adjacency_list)

    return adjacency_list
