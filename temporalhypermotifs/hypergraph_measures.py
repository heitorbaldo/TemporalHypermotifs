"""
Hypergraph measures:
1. Motif-Level Measures
2. Single-Layer Hypergraph Measures
3. Multilayer Hypergraph Measures
4. Spectral Measures
"""

import math
import numpy as np
import itertools
import hypernetx as hnx
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from collections import defaultdict

__all__ = [
    "ensure_csr",
    "spectral_density_sequence",
    "extract_all_motif_unions",
    "motif_metrics_from_indices",
    "motif_global_statistics",
    "compute_HSDC",
    "compute_IEI",
    "compute_HON",
    "compute_IAR",
    "compute_HPI",
    "compute_HER",
    "multilayer_motif_persistence_index",
    "multilayer_motif_entropy",
    "multilayer_motif_entropy_normalized",
    "cross_layer_motif_density",
    "multilayer_motif_participation_ratio",
    "multilayer_motif_participation_ratio_normalized",
    "motif_layer_coupling_strength",
    "motif_layer_coupling_strength_normalized",
    "node_multilayer_recruitment",
    "multilayer_motif_overlap_energy",
    "multilayer_motif_overlap_energy_normalized",
    "multilayer_motif_overlap_energy_node_normalized",
    "normalized_spectral_density",
    "spectral_shannon_entropy",
    "compute_SIC",
    "normalized_spectral_curvature_eigen",
    "compute_exact_SDC",
    "compute_SDC",
    "spectral_wasserstein",
]


#=========================== Utils ===========================

def ensure_csr(B):
    if not sps.isspmatrix_csr(B):
        B = B.tocsr()
    return B.astype(float)


def spectral_density_sequence(graphs):
    spec_seq = []
    for i in range(len(graphs)):
        eigenvalues = np.linalg.eigvalsh(graphs[i])
        spec_seq.append(eigenvalues)
    return spec_seq


def extract_all_motif_unions(H_layers, motifs_layers):
    """Extract all motif unions.
    H_layers: list of layers (each layer is a list of temporal hypergraphs
              ordered in time: [H_t1, H_t2, H_t3, ...]).
    motifs_layers: list of motif triples per layer
        Each triple (i, j, k) refers to:
            - hyperedge index i in H_t1
            - hyperedge index j in H_t2
            - hyperedge index k in H_t3
    """
    union_presence = defaultdict(set)

    for layer_idx, (H_time_seq, motif_list) in enumerate(zip(H_layers, motifs_layers)):

        #we must have at least 3 timestamps
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



#==============================================================
#=================== Motif-Level Measures =====================
#==============================================================

def motif_metrics_from_indices(H1, H2, H3, motifs):
    """Compute motif metrics from motif index list.
    Parameters
    ----------
    H1, H2, H3: hnx.Hypergraph (consecutive hypergraphs at times t1, t2, t3).
    motifs: list of tuples [(i, j, k), ...] (hyperedge indices).
    Returns
    -------
    list of dictionaries (metrics per motif).
    """

    edges1 = list(H1.edges)
    edges2 = list(H2.edges)
    edges3 = list(H3.edges)

    results = []
    for i, j, k in motifs:
        e1 = set(H1.edges[edges1[i]])
        e2 = set(H2.edges[edges2[j]])
        e3 = set(H3.edges[edges3[k]])

        U = e1 | e2 | e3

        I12 = len(e1 & e2)
        I23 = len(e2 & e3)
        I13 = len(e1 & e3)

        triple_intersection = len(e1 & e2 & e3)
        volume = len(U)
        persistence = triple_intersection / volume if volume > 0 else 0
        overlap_energy = I12 + I23 + I13

        jaccard = 0
        if len(e1 | e2) > 0:
            jaccard += I12 / len(e1 | e2)
        if len(e2 | e3) > 0:
            jaccard += I23 / len(e2 | e3)

        expansion = len(e3) - len(e1)
        recruitment = (volume - triple_intersection) / volume if volume > 0 else 0
        turnover = len(e1 ^ e2) + len(e2 ^ e3)
        density = overlap_energy / volume if volume > 0 else 0

        stability_ratio = (
            triple_intersection /
            min(len(e1), len(e2), len(e3))
            if min(len(e1), len(e2), len(e3)) > 0 else 0)

        results.append({
            "volume": volume,
            "persistence": persistence,
            "overlap_energy": overlap_energy,
            "jaccard": jaccard,
            "expansion": expansion,
            "recruitment": recruitment,
            "turnover": turnover,
            "density": density,
            "stability_ratio": stability_ratio
        })
    return results


def motif_global_statistics(motif_metric_list):
    """Motif statistics.
    """
    volumes = np.array([m["volume"] for m in motif_metric_list])
    persistence = np.array([m["persistence"] for m in motif_metric_list])
    turnover = np.array([m["turnover"] for m in motif_metric_list])

    stats = {
        "mean_volume": np.mean(volumes) if len(volumes) > 0 else 0,
        "var_volume": np.var(volumes) if len(volumes) > 0 else 0,
        "mean_persistence": np.mean(persistence) if len(persistence) > 0 else 0,
        "mean_turnover": np.mean(turnover) if len(turnover) > 0 else 0,
    }

    if np.sum(volumes) > 0:
        p = volumes / np.sum(volumes)
        stats["entropy_volume"] = -np.sum(p * np.log(p + 1e-12))
    else:
        stats["entropy_volume"] = 0

    return stats



#========================================================================
#=================== Single-Layer Hypergraph Measures ===================
#========================================================================

#Hyperedge Size Dispersion Curvature (HSDC)
def compute_HSDC(B):
    """Column-sum dispersion.
    """
    B = ensure_csr(B)
    sizes = np.array(B.sum(axis=0)).flatten()
    
    if len(sizes) == 0:
        return 0.0
    
    mu = np.mean(sizes)
    var = np.var(sizes)
    
    return var / (mu**2 + 1e-12)


#Incidence Energy Index (IEI)
def compute_IEI(B):
    """Frobenius norm of BB^T.
    """
    B = ensure_csr(B)
    L = B @ B.T
    
    frob_sq = np.sum(L.data**2)
    n = B.shape[0]
    
    return frob_sq / (n**2 + 1e-12)

#Hyperedge Overlap Norm (HON) 
def compute_HON(B):
    """Frobenius norm of B^TB.
    """
    B = ensure_csr(B)
    O = B.T @ B
    
    frob_sq = np.sum(O.data**2)
    m = B.shape[1]
    
    return frob_sq / (m**2 + 1e-12)

#Incidence Anisotropy Ratio (IAR)
def compute_IAR(B, k=5):
    """Singular value concentration.
    """
    B = ensure_csr(B)
    k = min(k, min(B.shape) - 1)
    if k <= 0:
        return 0.0
    
    _, s, _ = spla.svds(B, k=k)
    s = np.sort(s)[::-1]
    
    return s[0] / (np.sum(s) + 1e-12)


#Hypergraph Participation Inequality (HPI)
def compute_HPI(B):
    """Row-sum inequality.
    """
    B = ensure_csr(B)
    degrees = np.array(B.sum(axis=1)).flatten()
    
    if len(degrees) == 0 or np.sum(degrees) == 0:
        return 0.0
    
    diff_sum = np.sum(np.abs(degrees[:, None] - degrees))
    return diff_sum / (2 * len(degrees) * np.sum(degrees) + 1e-12)


#Hyperedge Expansion Ratio (HER)
def compute_HER(B):
    """Overlap-induced boundary growth.
    """
    B = ensure_csr(B)
    O = B.T @ B   #hyperedge overlap matrix
    
    m = B.shape[1]
    if m == 0:
        return 0.0
    
    sizes = np.array(B.sum(axis=0)).flatten()
    total = 0.0
    
    for i in range(m):
        overlapping = O.getrow(i).indices
        boundary_nodes = set()
        
        for j in overlapping:
            if j != i:
                nodes_j = set(B[:, j].nonzero()[0])
                nodes_i = set(B[:, i].nonzero()[0])
                boundary_nodes |= (nodes_j - nodes_i)
        
        total += len(boundary_nodes) / (sizes[i] + 1e-12)
    
    return total / m




#============================================================================
#=================== Multilayer Hypergraph Measures =========================
#============================================================================

#Multilayer Motif Persistence Index (MMPI)
def multilayer_motif_persistence_index(union_presence, L):
    """Multilayer Motif Persistence Index.
    """
    total = len(union_presence)
    if total == 0:
        return 0

    return sum(len(layers)/L for layers in union_presence.values()) / total

#Multilayer Motif Entropy (MME)
def multilayer_motif_entropy(union_presence, L):
    """Multilayer Motif Entropy.
    """
    counts = np.zeros(L)

    for layers in union_presence.values():
        counts[len(layers)-1] += 1

    p = counts / counts.sum() if counts.sum() > 0 else counts
    p = p[p > 0]

    return -np.sum(p * np.log(p))


def multilayer_motif_entropy_normalized(union_presence, L):
    """Normalized multilayer motif entropy.
    """
    counts = np.zeros(L)
    for layers in union_presence.values():
        counts[len(layers)-1] += 1

    total = counts.sum()
    if total == 0 or L <= 1:
        return 0.0

    p = counts / total
    p = p[p > 0]

    H = -np.sum(p * np.log(p))
    return H / np.log(L)


#Cross-Layer Motif Density (CLMD)
def cross_layer_motif_density(union_presence):
    """Cross-Layer Motif Density (CLMD)
    """
    total = len(union_presence)
    if total == 0:
        return 0

    multilayer = sum(1 for layers in union_presence.values()
                     if len(layers) >= 2)

    return multilayer / total


def multilayer_motif_participation_ratio(union_presence):
    """Multilayer motif participation ratio.
    """
    pis = np.array([len(layers) for layers in union_presence.values()])
    if len(pis) == 0:
        return 0
    return (pis.sum()**2) / np.sum(pis**2)


def multilayer_motif_participation_ratio_normalized(union_presence):
    """Normalized multilayer motif participation ratio.
    """
    pis = np.array([len(layers) for layers in union_presence.values()])
    M = len(pis)
    if M == 0:
        return 0.0

    pr = (pis.sum()**2) / np.sum(pis**2)
    return pr / M


#Motif-Layer Coupling Strength
def motif_layer_coupling_strength(union_presence):
    """Motif-layer coupling strength.
    """
    total = 0

    for layers in union_presence.values():
        k = len(layers)
        total += k * (k - 1) / 2

    return total


def motif_layer_coupling_strength_normalized(union_presence, L):
    """Normalized motif-layer coupling strength.
    """
    ks = np.array([len(layers) for layers in union_presence.values()])
    M = len(ks)
    if M == 0 or L < 2:
        return 0.0

    total = np.sum(ks * (ks - 1) / 2)
    max_possible = M * (L * (L - 1) / 2)
    return total / max_possible


#Node-Level Multilayer Recruitment
def node_multilayer_recruitment(union_presence):
    """Node-level multilayer recruitment.
    """
    recruitment = defaultdict(int)

    for nodes, layers in union_presence.items():
        if len(layers) >= 2:
            for node in nodes:
                recruitment[node] += 1

    return dict(recruitment)


#Multilayer Motif Overlap Energy
def multilayer_motif_overlap_energy(union_presence):
    """Multilayer Motif Overlap Energy.
    """
    energy = 0

    for nodes, layers in union_presence.items():
        energy += len(nodes) * len(layers)

    return energy


def multilayer_motif_overlap_energy_normalized(union_presence, N, L):
    """Normalized Multilayer Motif Overlap Energy.
    """
    M = len(union_presence)

    if M == 0:
        return 0.0

    energy = 0

    for nodes, layers in union_presence.items():
        energy += len(nodes) * len(layers)

    return energy / (M * N * L)


def multilayer_motif_overlap_energy_node_normalized(union_presence, N, L):
    """Normalized Multilayer Motif Overlap Energy (node).
    """
    energy = 0
    for nodes, layers in union_presence.items():
        energy += len(nodes) * len(layers)

    return energy / (N * L)




#=============================================================
#=================== Spectral Measures =======================
#=============================================================

def normalized_spectral_density(Q):
    """Compute exact normalized spectral density of signless Laplacian.
    Parameters
    ----------
    Q: array-like or sparse matrix (signless Laplacian matrix).
    n: int (number of nodes).
    Returns
    -------
    eigenvalues: ndarray (sorted eigenvalues).
    weights: ndarray (corresponding normalized weights (1/n each)).
    """
    n=len(Q)
    
    #Convert sparse to dense if necessary
    if sps.issparse(Q):
        Q = Q.toarray()
        
    if Q.shape != (n, n):
        raise ValueError("Matrix dimension does not match number of nodes.")
    
    eigenvalues = np.linalg.eigvalsh(Q)  # symmetric matrix
    eigenvalues = np.sort(np.real(eigenvalues))
        
    weights = np.ones(n) / n
    return eigenvalues*weights



def spectral_shannon_entropy(Q, normalized=False):
    """Compute Spectral Shannon Entropy of the signless Laplacian.
    Parameters
    ----------
    Q: ndarray or sparse matrix (signless Laplacian matrix)
    n: int Number of nodes.
    normalized : bool (if True, return entropy normalized by log(n)).
    Returns
    -------
    H: float (spectral Shannon entropy).
    """
    n = len(Q)
    
    #convert sparse to dense if needed
    if sps.issparse(Q):
        Q = Q.toarray()
    
    #check dimension:
    if Q.shape != (n, n):
        raise ValueError("Matrix dimension does not match n.")
    
    #compute full spectrum (exact):
    eigenvalues = np.linalg.eigvalsh(Q)
    eigenvalues = np.real(eigenvalues)
    
    #remove numerical negatives
    eigenvalues = np.maximum(eigenvalues, 0)
    total = np.sum(eigenvalues)
    
    if total == 0:
        return 0.0
    
    p = eigenvalues / total  #normalize eigenvalues
    
    #remove zero entries to avoid log(0)
    p = p[p > 1e-15]
    H = -np.sum(p * np.log(p))
    if normalized:
        H /= np.log(n)
    return H


def compute_SIC(vals):
    vals = np.maximum(vals, 1e-12)
    p = vals / np.sum(vals)
    return np.sum(p**2)


def normalized_spectral_curvature_eigen(Q, normalize=True):
    """Bin-free spectral curvature measure.
    Parameters
    ----------
    Q: ndarray or sparse matrix (signless Laplacian).
    n: int (Number of nodes)
    normalize: bool (if True, normalize eigenvalues for scale invariance).
    Returns
    -------
    curvature: float
    """
    n = len(Q)
    if sps.issparse(Q):
        Q = Q.toarray()
    
    if Q.shape != (n, n):
        raise ValueError("Matrix dimension does not match n.")
    
    eigenvalues = np.linalg.eigvalsh(Q)
    eigenvalues = np.real(eigenvalues)
    
    #sort
    eigenvalues = np.sort(eigenvalues)
    
    if normalize:
        total = np.sum(eigenvalues)
        if total > 0:
            eigenvalues = eigenvalues / total
    
    if len(eigenvalues) < 3:
        return 0.0
    
    #first differences
    d1 = np.diff(eigenvalues)
    
    #second differences
    d2 = np.diff(d1)
    
    numerator = np.sum(d2**2)
    denominator = np.sum(d1**2)
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator



#Spectral Density Curvature (SDC)
def compute_exact_SDC(Q, bins=50):
    """Approximate second derivative of density histogram.
    """
     # Compute full spectrum (exact)
    n = len(Q)
    eigenvalues = np.linalg.eigvalsh(Q)
    eigenvalues = np.real(eigenvalues)/n
    
    hist, _ = np.histogram(eigenvalues, bins=bins, density=True)
    second_diff = np.diff(hist, n=2)
    return np.sum(second_diff**2)


#Spectral Density Curvature (SDC)
def compute_SDC(freq, rho):
    """Approximate second derivative of density histogram.
    """
    hist, _ = np.histogram(freq, bins=rho, density=True)
    second_diff = np.diff(hist, n=2)
    return np.sum(second_diff**2)


#Spectral Wasserstein Distance
def spectral_wasserstein(eig1, eig2):
    return wasserstein_distance(eig1, eig2)




