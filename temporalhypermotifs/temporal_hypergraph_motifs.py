"""
Temporal Hypergraph Motifs.
"""

import math
import numpy as np
import networkx as nx
import hypernetx as hnx
from collections import defaultdict
from itertools import combinations
from typing import List, Tuple


__all__ = [
    "remove_singleton_edges",
    "incidence_to_bitsets",
    "is_subset",
    "find_temporal_overlap_motifs_no_subsets",
    "find_hypermotifs",
    "graph_to_hypergraph_from_cliques",
    "build_motif_hyperedge_hypergraph",
    "incidence_to_bitsets",
    "hnx_to_bitsets",
    "bitset_size",
    "overlap_size",
    "triple_overlap_size",
    "classify_single_motif",
    "classify_motifs",
    "union_fixed_nodes",
    "union_fixed_nodes_rename",
]


#=========================================================
#========================= Utils =========================
#=========================================================

def remove_singleton_edges(H):
    # Keep only edges with size > 1
    filtered_edges = {
        e: set(nodes)
        for e, nodes in H.edges.incidence_dict.items()
        if len(nodes) > 1
    }

    return hnx.Hypergraph(filtered_edges)


def incidence_to_bitsets(H: np.ndarray) -> List[int]:
    """
    Convert an incidence matrix (n_nodes x n_hyperedges)
    to a list of integer bitsets.
    """
    n_nodes, n_edges = H.shape
    bitsets = []

    for j in range(n_edges):
        bits = 0
        for i in range(n_nodes):
            if H[i, j]:
                bits |= (1 << i)
        bitsets.append(bits)

    return bitsets


def is_subset(e1: int, e2: int) -> bool:
    """
    Return True if e1 ⊆ e2 (bitset version).
    """
    return (e1 & ~e2) == 0



#===========================================================
#====================== Finding THMs =======================
#===========================================================

def find_temporal_overlap_motifs_no_subsets(H1: np.ndarray, H2: np.ndarray, H3: np.ndarray) -> List[Tuple[int, int, int]]:
    """Find temporal hypergraph motifs across three consecutive timestamps such that:
      - e1 ∩ e2 ≠ ∅.
      - e2 ∩ e3 ≠ ∅.
      - no hyperedge is a subset of another.
    Parameters:
    ---------
    H1, H2, H3 : incidence matrices (n_nodes x n_hyperedges_t).
    Returns:
    -------
      List of tuples (i, j, k) indexing hyperedges in (H1, H2, H3).
    """

    E1 = incidence_to_bitsets(H1)
    E2 = incidence_to_bitsets(H2)
    E3 = incidence_to_bitsets(H3)
    motifs = []

    # Precompute overlaps for pruning
    overlap12 = [[(e1 & e2) != 0 for e2 in E2] for e1 in E1]
    overlap23 = [[(e2 & e3) != 0 for e3 in E3] for e2 in E2]

    for i, e1 in enumerate(E1):
        for j, e2 in enumerate(E2):

            # must overlap in time
            if not overlap12[i][j]:
                continue

            # forbid subset relations
            if is_subset(e1, e2) or is_subset(e2, e1):
                continue

            for k, e3 in enumerate(E3):

                if not overlap23[j][k]:
                    continue

                # forbid subset relations across all pairs
                if (
                    is_subset(e1, e3) or is_subset(e3, e1) or
                    is_subset(e2, e3) or is_subset(e3, e2)
                ):
                    continue

                motifs.append((i, j, k))

    return motifs


def find_hypermotifs(H_hnx1, H_hnx2, H_hnx3):
    if len(H_hnx1.edges) == 0 or len(H_hnx2.edges) == 0 or len(H_hnx3.edges) == 0:
        motifs = []
    else:
        H1 = H_hnx1.incidence_matrix()
        H2 = H_hnx2.incidence_matrix()
        H3 = H_hnx3.incidence_matrix()
        motifs = find_temporal_overlap_motifs_no_subsets(H1, H2, H3)
    return motifs


#===============================================================================
#====================== Hypergraphs from Maximal Cliques =======================
#===============================================================================

def graph_to_hypergraph_from_cliques(G: nx.Graph, k: int):
    """
    Build a HyperNetX hypergraph from maximal cliques (size >= k),
    preserving all original nodes of G.
    """

    edges_dict = {}
    edge_id = 0

    # Add cliques of size >= k
    for clique in nx.find_cliques(G):
        if len(clique) >= k:
            edges_dict[f"e_{edge_id}"] = set(clique)
            edge_id += 1

    # Add dummy singleton edges for nodes not covered
    nodes_in_cliques = set().union(*edges_dict.values()) if edges_dict else set()
    isolated_nodes = set(G.nodes) - nodes_in_cliques

    for node in isolated_nodes:
        edges_dict[f"isolated_{node}"] = {node}

    H = hnx.Hypergraph(edges_dict)

    return H


#==============================================================
#====================== THM Hypergraphs =======================
#==============================================================


def build_motif_hyperedge_hypergraph(H1, H2, H3, motifs):
    """Build a hypergraph containing all hyperedges that
    participate in at least one motif, preserving ALL
    original nodes from H1, H2, and H3.
    """

    # --- Collect hyperedge indices participating in motifs ---
    idx1, idx2, idx3 = set(), set(), set()

    for i, j, k in motifs:
        idx1.add(i)
        idx2.add(j)
        idx3.add(k)

    edges_dict = {}

    #Case 1: HyperNetX input
    if isinstance(H1, hnx.Hypergraph):

        edges1 = list(H1.edges)
        edges2 = list(H2.edges)
        edges3 = list(H3.edges)

        #Add selected hyperedges
        for i in idx1:
            edges_dict[f"H1_e{i}"] = set(H1.edges[edges1[i]])

        for j in idx2:
            edges_dict[f"H2_e{j}"] = set(H2.edges[edges2[j]])

        for k in idx3:
            edges_dict[f"H3_e{k}"] = set(H3.edges[edges3[k]])

        # Collect full node universe
        all_nodes = set(H1.nodes) | set(H2.nodes) | set(H3.nodes)

    #Case 2: Incidence matrix input
    elif isinstance(H1, np.ndarray):

        def incidence_column_to_nodes(H, col_idx):
            return set(np.where(H[:, col_idx] == 1)[0])

        for i in idx1:
            edges_dict[f"H1_e{i}"] = incidence_column_to_nodes(H1, i)

        for j in idx2:
            edges_dict[f"H2_e{j}"] = incidence_column_to_nodes(H2, j)

        for k in idx3:
            edges_dict[f"H3_e{k}"] = incidence_column_to_nodes(H3, k)

        all_nodes = set(range(H1.shape[0]))

    else:
        raise ValueError("Input must be either hnx.Hypergraph or incidence matrices.")

    # --- Ensure isolated nodes are preserved ---
    nodes_in_edges = set().union(*edges_dict.values()) if edges_dict else set()
    isolated_nodes = all_nodes - nodes_in_edges

    for node in isolated_nodes:
        edges_dict[f"isolated_{node}"] = {node}

    # --- Build final hypergraph ---
    H_out = hnx.Hypergraph(edges_dict)

    return H_out



#==================================================================
#====================== THMs Classification =======================
#==================================================================

def incidence_to_bitsets(H: np.ndarray):
    """Convert incidence matrix to list of bitsets.
    """
    n_nodes, n_edges = H.shape
    bitsets = []

    for j in range(n_edges):
        bits = 0
        for i in range(n_nodes):
            if H[i, j]:
                bits |= (1 << i)
        bitsets.append(bits)

    return bitsets


def hnx_to_bitsets(H: hnx.Hypergraph):
    """Convert HyperNetX hypergraph to list of bitsets.
    Assumes nodes are labeled 0..n-1 or convertible to an integer.
    """

    nodes = list(H.nodes)
    node_index = {node: idx for idx, node in enumerate(nodes)}

    bitsets = []

    for edge in H.edges:
        bits = 0
        for node in H.edges[edge]:
            bits |= (1 << node_index[node])
        bitsets.append(bits)

    return bitsets


def bitset_size(e: int) -> int:
    return e.bit_count()


def overlap_size(e1: int, e2: int) -> int:
    return (e1 & e2).bit_count()


def triple_overlap_size(e1: int, e2: int, e3: int) -> int:
    return (e1 & e2 & e3).bit_count()


def classify_single_motif(e1: int, e2: int, e3: int, tol: int = 0) -> str:
    """Classify a single temporal motif using structural rules.
    tol = tolerance for near-equal sizes.
    """

    s1 = bitset_size(e1)
    s2 = bitset_size(e2)
    s3 = bitset_size(e3)

    # Persistent
    if abs(s1 - s2) <= tol and abs(s2 - s3) <= tol:
        return "P"

    # Expanding
    if s1 < s2 and s2 < s3:
        return "E"

    # Contracting
    if s1 > s2 and s2 > s3:
        return "C"

    # Otherwise: Reconfiguration
    return "R"


def classify_motifs(motifs, H1, H2, H3, tol: int = 0):
    """Classify temporal hypergraph motifs.
    Parameters
    ----------
    motifs: list of (i, j, k) Hyperedge index triples.
    H1, H2, H3: either incidence matrices (numpy arrays) or hnx.Hypergraph objects.
    tol: int (size tolerance for persistent class).
    Returns
    -------
    List of dicts (with classification results).
    """

    # Convert inputs to bitsets
    if isinstance(H1, np.ndarray):
        E1 = incidence_to_bitsets(H1)
        E2 = incidence_to_bitsets(H2)
        E3 = incidence_to_bitsets(H3)

    elif isinstance(H1, hnx.Hypergraph):
        E1 = hnx_to_bitsets(H1)
        E2 = hnx_to_bitsets(H2)
        E3 = hnx_to_bitsets(H3)

    else:
        raise ValueError("Input must be incidence matrices or hnx.Hypergraph objects.")

    classified = []

    for (i, j, k) in motifs:

        e1 = E1[i]
        e2 = E2[j]
        e3 = E3[k]

        label = classify_single_motif(e1, e2, e3, tol)

        classified.append({
            "indices": (i, j, k),
            "class": label,
            "sizes": (
                bitset_size(e1),
                bitset_size(e2),
                bitset_size(e3)
            ),
            "pairwise_overlaps": (
                overlap_size(e1, e2),
                overlap_size(e2, e3),
                overlap_size(e1, e3)
            ),
            "triple_overlap": triple_overlap_size(e1, e2, e3)
        })

    return classified



#===============================================================
#====================== Time Aggregation =======================
#===============================================================

def union_fixed_nodes(H1, H2):
    nodes1 = set(H1.nodes)
    nodes2 = set(H2.nodes)

    if nodes1 != nodes2:
        raise ValueError("Hypergraphs must have identical node sets.")

    # Copy edge dictionaries
    edges_union = {}

    # Add edges from H1
    for e, nodes in H1.edges.incidence_dict.items():
        edges_union[e] = set(nodes)

    # Add edges from H2 (no merging)
    for e, nodes in H2.edges.incidence_dict.items():
        if e in edges_union:
            raise ValueError(f"Edge name conflict: {e}")
        edges_union[e] = set(nodes)

    return hnx.Hypergraph(edges_union)


def union_fixed_nodes_rename(H1, H2):
    nodes1 = set(H1.nodes)
    nodes2 = set(H2.nodes)

    if nodes1 != nodes2:
        raise ValueError("Hypergraphs must have identical node sets.")

    edges_union = {}

    #add H1 edges
    for e, nodes in H1.edges.incidence_dict.items():
        edges_union[f"H1_{e}"] = set(nodes)

    #add H2 edges
    for e, nodes in H2.edges.incidence_dict.items():
        edges_union[f"H2_{e}"] = set(nodes)

    return hnx.Hypergraph(edges_union)



