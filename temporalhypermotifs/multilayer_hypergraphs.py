"""
Multilayer hypergraphs.
"""

import math
import numpy as np
import networkx as nx
from collections import defaultdict
from itertools import combinations
from typing import List, Tuple

from scipy.optimize import minimize
from scipy.sparse import lil_matrix, block_diag
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import seaborn as sns



__all__ = [
    "hypergraph_co_membership_adjacency_multi",
    "build_global_node_list",
    "extract_edge_list_from_layer",
    "co_membership_from_edge_list",
    "extract_all_motif_unions",
    "extract_structural_motifs",
    "build_supra_adjacency_structural",
    "build_supra_adjacency_comembership",
    "off_diagonal_row_sum",
    "compute_offdiag_degree_list",
    "supra_diagonal_degree_matrix",
    "supra_laplacian",
    "count_multilayer_motifs",
]



# --------------------------------------------------
# Hypergraph co-membership adjacency A = BB^T - D
# --------------------------------------------------

def remove_columns_with_sum_one(matrix):
    """Remove columns of a matrix whose entries sum to exactly 1.
    Parameters
    ----------
    matrix: array-like (2D matrix (numpy array))
    Returns
    -------
    numpy.ndarray (Matrix with columns removed)
    """
    M = np.array(matrix)
    
    col_sums = M.sum(axis=0) #compute column sums.
    mask = col_sums != 1 #keep columns whose sum is NOT equal to 1.
    
    return M[:, mask]

def hypergraph_co_membership_adjacency_multi(H):
    """Returns the co-membership adjacency matrix.
    Parameters
    ---------
    H: hypergraph.
    """
    node_list = list(H.nodes)
    B_with_sigletons = H.incidence_matrix().toarray()
    B = remove_columns_with_sum_one(B_with_sigletons)
    BBt = B @ B.T
    degrees = B.sum(axis=1)
    D = np.diag(degrees)
    A = BBt - D

    return A, node_list


#-----------------------------------------------------

def build_global_node_list(H_layers):
    """Build global node list (shared across all layers).
    """
    global_node_set = set()

    for H_time_seq in H_layers:
        for H in H_time_seq:
            for node in H.nodes:
                global_node_set.add(node)

    return sorted(global_node_set)


#-----------------------------------------------------

def extract_edge_list_from_layer(H_time_seq):
    """Extract all hyperedges across time for one layer.
    """
    edge_list = []

    for H in H_time_seq:
        for e in H.edges:
            edge_list.append(set(H.edges[e]))

    return edge_list


#-----------------------------------------------

def co_membership_from_edge_list(edge_list, global_node_list):
    """Build co-membership adjacency.
       (Remove singleton hyperedges; keep all nodes (multiplex consistency)).
    """
    node_index = {node: i for i, node in enumerate(global_node_list)}

    N = len(global_node_list)
    M = len(edge_list)

    if M == 0:
        return np.zeros((N, N))

    B = np.zeros((N, M))

    for j, edge in enumerate(edge_list):
        for node in edge:
            if node in node_index:
                B[node_index[node], j] = 1

    #Remove singleton hyperedges strictly
    col_sums = np.sum(B, axis=0)
    keep = col_sums >= 2   # IMPORTANT: strictly >=2

    if not np.any(keep):
        return np.zeros((N, N))

    B = B[:, keep]

    BBt = B @ B.T
    degrees = np.sum(B, axis=1)
    D = np.diag(degrees)

    A = BBt - D

    return A


#------------------------------------------

def extract_all_motif_unions(H_layers, motifs_layers):
    """Extract motif unions (timestamp-aware).
    H_layers: list of layers (each layer is a list of temporal hypergraphs 
                              ordered in time: [H_t1, H_t2, H_t3, ...])
    motifs_layers: list of motif triples per layer
        Each triple (i, j, k) refers to:
            - hyperedge index i in H_t1
            - hyperedge index j in H_t2
            - hyperedge index k in H_t3
    """
    union_presence = defaultdict(set)

    for layer_idx, (H_time_seq, motif_list) in enumerate(zip(H_layers, motifs_layers)):
        # Ensure we have at least 3 timestamps
        if len(H_time_seq) < 3:
            continue

        H1 = H_time_seq[0]
        H2 = H_time_seq[1]
        H3 = H_time_seq[2]

        edges1 = list(H1.edges)
        edges2 = list(H2.edges)
        edges3 = list(H3.edges)

        for (i, j, k) in motif_list:
            e1 = set(H1.edges[edges1[i]])
            e2 = set(H2.edges[edges2[j]])
            e3 = set(H3.edges[edges3[k]])

            union_nodes = frozenset(e1 | e2 | e3)
            union_presence[union_nodes].add(layer_idx)

    return union_presence


def extract_structural_motifs(H_layers, motifs_layers):
    """Extract structural motifs.
    """
    motif_presence = defaultdict(set)
    for layer_idx, (H_time_seq, motif_list) in enumerate(zip(H_layers, motifs_layers)):

        H1, H2, H3 = H_time_seq[0], H_time_seq[1], H_time_seq[2]
        edges1 = list(H1.edges)
        edges2 = list(H2.edges)
        edges3 = list(H3.edges)

        for (i, j, k) in motif_list:
            e1 = frozenset(H1.edges[edges1[i]])
            e2 = frozenset(H2.edges[edges2[j]])
            e3 = frozenset(H3.edges[edges3[k]])

            # Strict motif condition
            if len(e1) < 2 or len(e2) < 2 or len(e3) < 2:
                continue

            motif_signature = (e1, e2, e3)
            motif_presence[motif_signature].add(layer_idx)

    return motif_presence



#---------------------------------------------

def build_supra_adjacency_structural(H_layers, H_layers_MH, motifs_layers):
    """Build supra-adjacency matrix (exact structural matching).
    """
    L = len(H_layers)

    #Global node alignment:
    global_nodes = set()
    for H_time_seq in H_layers:
        for H in H_time_seq:
            global_nodes.update(H.nodes)

    global_node_list = sorted(global_nodes)
    node_index = {node: i for i, node in enumerate(global_node_list)}

    N = len(global_node_list)
    supra = np.zeros((L * N, L * N))

    #Diagonal blocks (use your original method):
    A_blocks = []
    for i in range(len(H_layers_MH)):
        A_blocks.append(hypergraph_co_membership_adjacency_multi(H_layers_MH[i])[0])

    #Insert diagonal blocks
    for l in range(L):
        supra[l*N:(l+1)*N, l*N:(l+1)*N] = A_blocks[l]

    #Structural motif matching across layers
    motif_presence = extract_structural_motifs(H_layers, motifs_layers)

    for motif_signature, layers in motif_presence.items():
        layers = list(layers)
        if len(layers) < 2:
            continue

        e1, e2, e3 = motif_signature
        involved_nodes = e1 | e2 | e3

        for a in range(len(layers)):
            for b in range(a + 1, len(layers)):
                l = layers[a]
                m = layers[b]

                for node in involved_nodes:
                    if node in node_index:
                        i = node_index[node]
                        supra[l*N + i, m*N + i] = 1
                        supra[m*N + i, l*N + i] = 1

    return supra



#---------------------------------------------------

def build_supra_adjacency_comembership(H_layers, H_layers_MH, motifs_layers):
    """Build supra-adjacency matrix (union-based).
    """
    L = len(H_layers)

    #Global node alignment:
    global_node_list = build_global_node_list(H_layers)
    N = len(global_node_list)

    A_blocks_old = []
    A_blocks = []

    #Diagonal blocks (temporal aggregation per layer):
    for H_time_seq in H_layers:
        edge_list = extract_edge_list_from_layer(H_time_seq)

        A_layer = co_membership_from_edge_list(
            edge_list,
            global_node_list
        )
        A_blocks_old.append(A_layer)
        
    for i in range(len(H_layers_MH)):
        A_blocks.append(hypergraph_co_membership_adjacency_multi(H_layers_MH[i])[0])

    #Initialize supra matrix:
    supra = np.zeros((L * N, L * N))

    #Insert diagonal blocks:
    for l in range(L):
        supra[l*N:(l+1)*N, l*N:(l+1)*N] = A_blocks[l]

    #Interlayer motif persistence coupling:
    union_presence = extract_all_motif_unions(H_layers, motifs_layers)
    node_index = {node: i for i, node in enumerate(global_node_list)}

    for union_nodes, layers in union_presence.items():
        layers = list(layers)
        if len(layers) >= 2:
            for a in range(len(layers)):
                for b in range(a + 1, len(layers)):
                    l = layers[a]
                    m = layers[b]

                    for node in union_nodes:
                        if node in node_index:
                            i = node_index[node]
                            supra[l*N + i, m*N + i] = 1
                            supra[m*N + i, l*N + i] = 1

    return supra



def off_diagonal_row_sum(A_supra, node_index, L, N):
    """Compute the sum of off-diagonal block elements for a given supra node.
    Parameters
    ----------
    A_supra: (LN x LN) numpy array.
    node_index: int (supra node index).
    L: number of layers.
    N: nodes per layer.
    Returns
    -------
    float: sum of interlayer coupling for that node.
    """

    layer = node_index // N
    row = A_supra[node_index]
    total_sum = np.sum(row) #total row sum

    # intralayer block indices
    start = layer * N
    end = (layer + 1) * N
    intralayer_sum = np.sum(row[start:end])

    return total_sum - intralayer_sum



def compute_offdiag_degree_list(A_supra, L, N):
    """Compute interlayer degree (off-diagonal degree) for each node in each layer
    from a supra-adjacency matrix.
    Parameters
    ----------
    A_supra: (L*N, L*N) numpy array (Supra-adjacency matrix).
    L: int (Number of layers).
    N: int (Number of nodes per layer).
    Returns
    -------
    offdiag_degree_list: list of length L.
        Each element is a vector of length N containing the
        interlayer degree of nodes in that layer.
    """

    A_supra = np.asarray(A_supra)
    if A_supra.shape != (L*N, L*N):
        raise ValueError("Supra adjacency matrix has incorrect shape.")

    offdiag_degree_list = []

    for l in range(L):
        start = l * N
        end = (l + 1) * N

        #Full block row corresponding to layer l
        block_row = A_supra[start:end, :]

        #Intralayer block (diagonal block)
        intralayer_block = A_supra[start:end, start:end]

        #Interlayer contributions = block_row minus intralayer part
        interlayer_block = block_row.copy()
        interlayer_block[:, start:end] = 0

        #row sums = interlayer degree
        inter_degree = np.sum(interlayer_block, axis=1)
        offdiag_degree_list.append(inter_degree)

    return offdiag_degree_list


def supra_diagonal_degree_matrix(B_list_input, offdiag_degree_list):
    """
    """
    B_list = []
    for B_inc in B_list_input:
        B = remove_columns_with_sum_one(B_inc)
        B_list.append(B)

    L = len(B_list)
    N = B_list[0].shape[0]

    D_diag = np.zeros(L * N)

    for l in range(L):

        B = np.asarray(B_list[l])

        if B.ndim != 2:
            raise ValueError(f"B[{l}] is not 2D. Shape: {B.shape}")
        if B.shape[0] != N:
            raise ValueError(f"B[{l}] has inconsistent node count.")

        #hyperedge degree = row sum of incidence
        hyper_degree = np.sum(B, axis=1)

        if hyper_degree.ndim != 1:
            hyper_degree = hyper_degree.flatten()

        inter_degree = np.asarray(offdiag_degree_list[l]).flatten()

        if len(inter_degree) != N:
            raise ValueError(f"Interlayer degree length mismatch in layer {l}")

        D_layer = hyper_degree + inter_degree
        start = l * N
        end = (l + 1) * N
        D_diag[start:end] = D_layer
    return np.diag(D_diag)

    
def supra_laplacian(D_supra, A_supra):
    """
    """
    Q_supra = D_supra + A_supra
    return Q_supra
    

def count_multilayer_motifs(H_layers, motifs_layers):
    """Count multilayer motifs using motif lists.
    Parameters
    ----------
    H_layers: list of hnx.Hypergraph.
    motifs_layers: list of motif lists [(i,j,k), ...].
    Returns
    -------
    dict with multilayer motif statistics
    """

    # Map motif node-set -> layers where it appears
    motif_presence = defaultdict(list)

    for layer_idx, (H, motif_list) in enumerate(zip(H_layers, motifs_layers)):
        for motif in motif_list:
            node_union = extract_motif_unions(H, motif)
            motif_presence[node_union].append(layer_idx)

    multilayer_motifs = []
    for node_union, layers in motif_presence.items():

        if len(layers) >= 2:
            multilayer_motifs.append({
                "nodes": node_union,
                "layers": layers,
                "n_layers": len(layers),
                "size": len(node_union)
            })
    return {
        "total_multilayer_motifs": len(multilayer_motifs),
        "motifs": multilayer_motifs}