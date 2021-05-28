import os, pickle, logging
from matplotlib import pyplot as plt

import numpy as np

from util import print_datetime, parseSuffix

def load_expression(filename):
    """Load gene expression data for spatial transcriptomics data.

    Args:
        filename: path to file (.txt or .pkl) that contains gene expression data.

    Returns:
        (num_datapoints, num_genes) matrix of gene expression.
    """

    if filename.suffix == '.pkl':
        with open(filename, 'rb') as f:
            gene_expression = pickle.load(f)
    elif filename.suffix == '.txt':
        gene_expression = np.loadtxt(filename, dtype=np.float)
    else:
        raise ValueError(f'Invalid file format for {filename}')

    num_datapoints, num_genes = gene_expression.shape
    logging.info(f'{print_datetime()}Loaded {num_datapoints} cells and {num_genes} genes from {filename}')
    
    return gene_expression

def load_edges(filename, num_nodes):
    """Load HMRF edges for connectivity graph derived from spatial transcriptomics coordinates.

    Args:
        filename: path to .txt file that contains edges as tuples of node IDs.
        num_nodes: total number of nodes in connectivity graph.

    Returns:
        A dictionary mapping each node ID to a list of node IDs that are its neighbors.
    """

    edges = np.loadtxt(filename, dtype=np.int)
    if edges.shape[1] != 2:
        raise ValueError(f'Detected an edge that does not contain two nodes')
    if np.any(0 > edges) or np.any(edges >= num_nodes):
        raise ValueError(f'Node ID exceeded range [0, N)')

    edges = np.sort(edges, axis=1)
    if np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError(f'Detected {(edges[:, 0] == edges[:, 1]).sum()} self-loop(s)')

    unique_edges = np.unique(edges, axis=0)
    if len(unique_edges) != len(edges):
        logging.warning(f'Detected {len(edges)-len(np.unique(edges, axis=0))} duplicate edge(s) from {len(edges)} loaded edges. Duplicate edges are discarded.')
        edges = unique_edges
    logging.info(f'{print_datetime()}Loaded {len(edges)} edges from {filename}')
    
    adjacency_list = [[] for _ in range(num_nodes)]
    for (source, sink) in edges:
        adjacency_list[source].append(sink)
        adjacency_list[sink].append(source)
    
    return adjacency_list

def loadGeneList(filename):
    genes = np.loadtxt(filename, dtype=str)
    logging.info(f'{print_datetime()}Loaded {len(genes)} genes from {filename}')
    return genes

def load_dataset(self, neighbor_suffix=None, expression_suffix=None):
    neighbor_suffix = parseSuffix(neighbor_suffix)
    expression_suffix = parseSuffix(expression_suffix)

    self.YTs = []
    for i in self.replicate_names:
        for s in ['txt', 'tsv', 'pkl', 'pickle']:
            path2file = self.path2dataset / 'files' / f'expression_{i}{expression_suffix}.{s}'
            if not path2file.exists(): continue
            self.YTs.append(load_expression(path2file))
    self.Ns, self.Gs = zip(*map(np.shape, self.YTs))
    self.GG = max(self.Gs)
    self.Es = [
        load_edges(self.path2dataset / 'files' / f'neighborhood_{i}{neighbor_suffix}.txt', N)
        if u else [[] for _ in range(N)]
        for i, N, u in zip(self.replicate_names, self.Ns, self.use_spatial)
    ]
    self.Es_empty = [sum(map(len, E)) == 0 for E in self.Es]
    try:
        self.genes = [
            loadGeneList(self.path2dataset / 'files' / f'genes_{i}{expression_suffix}.txt')
            for i in self.replicate_names
        ]
    except:
        pass

# def loadImage(dataset, filename):
#   path2file = os.path.join(dataFolder(dataset), filename)
#   if os.path.exists(path2file): return plt.imread(path2file)
#   else: return None
