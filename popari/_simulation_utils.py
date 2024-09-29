import numpy as np
from multiprocess import Pool
from ortools.graph.python import min_cost_flow
from scipy.spatial.distance import pdist, squareform

SCALING_FACTOR = int(1e4)


def spatial_wasserstein(
    spatial_coordinates: np.ndarray,
    embeddings_truth: np.ndarray,
    embeddings_pred: np.ndarray,
    weight_scaling_factor=SCALING_FACTOR,
    demand_scaling_factor=SCALING_FACTOR,
):
    """Use max flow/min cut formulation to determine spatial Wasserstein
    distance."""

    assert len(spatial_coordinates) == len(embeddings_truth) == len(embeddings_pred)
    assert spatial_coordinates.ndim == 2
    assert embeddings_truth.ndim == embeddings_truth.ndim == 1
    assert embeddings_truth.min() >= 0
    assert embeddings_pred.min() >= 0

    if embeddings_truth.sum() == 0:
        return np.inf

    # the pairwise distances, a.k.a., weights are linearly scaled so that the minimum weight is equal to weight_scaling_factor
    pairwise_dist = squareform(pdist(spatial_coordinates))
    np.fill_diagonal(pairwise_dist, np.inf)
    pairwise_dist_min = pairwise_dist.min()
    pairwise_dist = (pairwise_dist / pairwise_dist.min() * weight_scaling_factor).astype(int)
    weight_scaling_factor_full = weight_scaling_factor / pairwise_dist_min

    # # the demands are linearly scaled so that the maximum weight is equal to demand_scaling_factor
    embeddings_truth = embeddings_truth / embeddings_truth.sum()
    embeddings_pred = embeddings_pred / embeddings_pred.sum()
    demands = embeddings_pred - embeddings_truth
    demands_max = np.abs(demands).max()
    if demands_max == 0:
        return 0
    demands = (demands / demands_max * demand_scaling_factor).astype(int)
    demand_scaling_factor_full = demand_scaling_factor / demands_max

    idx = np.argmax(np.abs(demands))
    demands[idx] -= demands.sum()

    smcf = min_cost_flow.SimpleMinCostFlow()

    edges = []

    start_nodes = []
    end_nodes = []
    capacities = []
    unit_costs = []
    for cell_idx_i, pairwise_dist_row in enumerate(pairwise_dist):
        for cell_idx_j, dist in enumerate(pairwise_dist_row):
            if cell_idx_i != cell_idx_j and demands[cell_idx_i] < 0 and demands[cell_idx_j] > 0:
                edge = (cell_idx_i, cell_idx_j, {"capacity": demand_scaling_factor, "weight": dist})
                edges.append(edge)
                start_nodes.append(cell_idx_i)
                end_nodes.append(cell_idx_j)
                capacities.append(demand_scaling_factor)
                unit_costs.append(dist)

    _ = smcf.add_arcs_with_capacity_and_unit_cost(
        start_nodes,
        end_nodes,
        capacities,
        unit_costs,
    )

    smcf.set_nodes_supplies(np.arange(0, len(demands)), -demands)

    status = smcf.solve()

    if status != smcf.OPTIMAL:
        print("There was an issue with the min cost flow input.")
        print(f"Status: {status}")
        exit(1)

    return smcf.optimal_cost() / demand_scaling_factor_full / weight_scaling_factor_full


def all_pairs_spatial_wasserstein(
    dataset,
    spatial_key: str = "spatial",
    embeddings_truth_key: str = "ground_truth_X",
    embeddings_pred_key: str = "X",
    weight_scaling_factor=SCALING_FACTOR,
    demand_scaling_factor=SCALING_FACTOR,
):
    """Compute spatial Wasserstein metric between all metagene pairs embeddings.

    Uses multiprocess.Pool to compute the distance matrix in embarassingly
    parallel fashion.

    """

    spatial_coordinates = dataset.obsm[spatial_key]
    embeddings_truth = dataset.obsm[embeddings_truth_key]
    embeddings_pred = dataset.obsm[embeddings_pred_key]

    def metric(pair):
        return spatial_wasserstein(spatial_coordinates, *pair)

    num_truth = embeddings_truth.shape[1]
    num_pred = embeddings_pred.shape[1]

    pairs = [[(truth, pred) for pred in embeddings_pred.T] for truth in embeddings_truth.T]
    pairs = np.array(pairs).reshape(-1, 2, len(spatial_coordinates))

    with Pool(processes=16) as pool:
        results = pool.map(metric, pairs)

    distances = np.array(list(results)).reshape((num_truth, num_pred))

    return distances
