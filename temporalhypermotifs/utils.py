"""
Utils.
"""

import sys
import scipy.io
import numpy as np
import networkx as nx
import hypernetx as hnx
import random
from random import choice, seed
from math import comb
from time import time

__all__ = [
    "performance",
    "inputmatrix",
    "is_symmetric_numpy",
    "convert_to_pypph_format",
    "nodes_to_array",
    "dict_to_array",
    "connect_array",
    "is_square",
    "diag_zero",
    "to_binary",
    "remove_double_edges",
    "remove_double_edges_rand",
    "remove_weighted_double_edges",
    "to_natural",
    "th_weights",
    "cliques_gtoet_k",
    "triangles_to_digraph",
    "loopless_graph",
    "lists_to_tuples",
    "to_hnx",
    "incidence_to_hyperedges",
    "hnx_to_incidence_matrix"
]


#============================= Graph Utils =============================

#Returns the execution time of a function.
def performance(fn):
    def wrapper(*args, **kwargs):
        t1 = time()
        results = fn(*args, **kwargs)
        t2 = time()
        print(f'took {t2 - t1}')
        return results
    return wrapper

#Convert MATLAB matrix to NumPy matrix.
def inputmatrix(path, n, name):
    s = 'mat' + "{}".format(n)
    moduleName = __name__
    currModule = sys.modules[moduleName]
    s = scipy.io.loadmat(path + name + 'M' + "{}".format(n) + '.m', appendmat=False)
    return s

#Checks if a matrix is symmetric using NumPy.
def is_symmetric_numpy(matrix_list):
    arr = np.array(matrix_list)

    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return False

    if np.issubdtype(arr.dtype, np.floating):
        return np.allclose(arr, arr.T)
    else:
        return np.array_equal(arr, arr.T)
        

#Convert a NumPy matrix to pph format
def convert_to_pypph_format(M, file_id):
    A = M.transpose()
    f = open("M"+file_id+".txt","w+")
    f.write(str(len(A))+"\n")
    for i in range(len(A)):
        for j in range(len(A)):
            if A[i,j] != 0:
                f.write(str(i) + " " + str(j) + " " + str(round(A[i,j],4))+"\n")
            else:
                pass
    f.close()
    return f

def nodes_to_array(M):
    '''Returns an array of nodes.
    M: numpy matrix.
    '''
    nodes = []
    for i in range(len(M)):
        nodes.append([i])
    return nodes

#Dictionary to array:
def dict_to_array(dict):
    Arr = []
    for i in range(len(dict)):
        Arr.append(dict[i])
    return Arr

#Connect array:
def connect_array(x):
    y = []
    for i in range(len(x)):
        for j in range(len(x[i])):
            y.append(x[i][j])
    return y

#Check if a matrix is square.
def is_square(M):
    if len(M[:,0]) == len(M[0,:]):
        return True
    else:
        return False

#Fill the diagonal with zeros.
def diag_zero(M):
    '''
    M: an adjacency matrix.
    Return: M with a null diagonal.
    '''
    for i in range(len(M)):
        M[i, i] = 0
    return M

#Convert a weighted matrix to a binary matrix.
def to_binary(M):
    '''Returns a binary matrix.
    M: a weighted adjacency matrix.
    Return: a binary adjacency matrix.
    '''
    for i in range(len(M)):
        for j in range(len(M)):
            if(M[i, j] > 0):
                M[i, j] = 1
            else:
                M[i, j] = 0
    return M


def normalize_weights(M):
    '''Returns a normalized weighted matrix.
    M: a weighted adjacency matrix.
    Return: a binary adjacency matrix.
    '''
    return M


#Remove double edges
def remove_double_edges(M):
    for i in range(len(M)):
        for j in range(len(M)):
            if M[i, j] !=0 and M[j,i] != 0:
                M[j,i] = 0
    return M

#Randomly removes double edges.
def remove_double_edges_rand(G):
    A = nx.adjacency_matrix(G)
    M = A.todense()
    for i in range(len(M)):
        for j in range(len(M)):
            if i != j and M[i,j] != 0 and M[j,i] !=0:
                r = random.choice([i,j])
                M[r, abs(r - (i+j))] = 0
    H = nx.from_numpy_array(M, create_using=nx.DiGraph())
    return H

#Remove double edges based on their weights.
def remove_weighted_double_edges(M):
    for i in range(len(M)):
        for j in range(len(M)):
            if M[i, j] !=0 and M[j,i] != 0:
                if M[i,j] >= M[j,i]:
                    M[j,i] = 0
                else:
                    M[i,j] = 0
    return M

#Convert real numbers to natural numbers
def to_natural(M):
    for i in range(len(M)):
        for j in range(len(M)):
            M[i, j] = round(100*M[i, j])
    return M

#Set an additional threshold to the weights
def th_weights(M, th):
    for i in range(len(M)):
        for j in range(len(M)):
            if(M[i, j] < th):
                M[i, j] = 0
            else:
                pass
    return M

def cliques_gtoet_k(C, k):
    cliques = []
    for cliq in C:
        if len(cliq) >= k:
            cliques.append(cliq)
    return cliques

#Build a digraph from temporal directed acyclic triangles.
def triangles_to_digraph(triangles, n):
    """Build a digraph from directed acyclic triangles.
    ----------
    triangles: list of tuples (Each tuple is (v, u, w, t1, t2, t3) (only the nodes (v, u, w) are used)).
    n : int (Number of nodes).
    Return: networkx.DiGraph
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    for tri in triangles:
        v, u, w = tri[:3]
        G.add_edge(v, u)
        G.add_edge(u, w)
        G.add_edge(v, w)

    return G

def loopless_graph(M):
    G = nx.from_numpy_array(M, create_using=nx.Graph())
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


#============================= Hypergraph Utils =============================

def lists_to_tuples(list_of_lists):
    return [tuple(lst) for lst in list_of_lists]


def to_hnx(edges_t):
    edge_dict = {f"e{i}": list(e) for i, e in enumerate(edges_t)}
    return hnx.Hypergraph(edge_dict)


def incidence_to_hyperedges(incidence):
    """Convert incidence matrix (nodes x hyperedges)
    into a list of hyperedges (tuples of nodes).
    """
    hyperedges = []
    for j in range(incidence.shape[1]):
        nodes = tuple(np.where(incidence[:, j] != 0)[0])
        hyperedges.append(nodes)
    return hyperedges


def hnx_to_incidence_matrix(H):
    """Convert a HyperNetX hypergraph to its incidence matrix.
    H: HyperNetX hypergraph
    Returns: numpy.ndarray (n_nodes x n_edges)
        node_index: dict (node -> row index)
        edge_index: dict (edge -> column index)
    """
    nodes = list(H.nodes)
    edges = list(H.edges)

    node_index = {v: i for i, v in enumerate(nodes)}
    edge_index = {e: j for j, e in enumerate(edges)}

    B = np.zeros((len(nodes), len(edges)), dtype=int)

    for e in edges:
        j = edge_index[e]
        for v in H.edges[e]:
            i = node_index[v]
            B[i, j] = 1

    return B