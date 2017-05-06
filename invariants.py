'''
invariants.py
Sha Lai
5/6/2017

This file includes functions that compute some invariants of a graph. Currently
the invariants includes:

completeness
connectivity
characteristic polynomial
chromatic polynomial
chromatic number
maximal cliques

'''

import numpy as np
from time import time
from copy import deepcopy

BASE = 1

###############################################################################
#                             Helper Functions                                #

def print_list(list):
    """Prints a list of elements one per line.
    Parameters
    ----------
    list : list[object]
    Returns
    -------
    None
    """
    for l in list:
        print(l)

def get_adj_mat(edge_list, vertex_amount):
    """Returns the adjacency matrix of a graph.
    Parameters
    ----------
    edge_list : list[[int, int]]
        A list of edges of a graph.
    vertex_amount : int
        The amount of vertices of a graph.
    Returns
    -------
    adj_mat : List[List[int]]
        The adjacency matrix of a graph.
    """
    adj_mat = []
    for i in range(vertex_amount):
        adj_mat.append([0] * vertex_amount)
    for edge in edge_list:
        adj_mat[edge[0] - BASE][edge[1] - BASE] = 1
        adj_mat[edge[1] - BASE][edge[0] - BASE] = 1
    return adj_mat

def explore(current_vertex, adj_mat, discovered):
    """Uses brute force to check if a graph is connected.
    Parameters
    ----------
    current_vertex : int
        Marks the current vertex index minus 1 (so that it's actually the index
        in the list).
    adj_mat : List[List[int]]
        The adjacency matrix.
    discovered : List[int]
        Stores the indices of the vertices (again, base 0) that have been
        reached by this algorithm.
    Returns
    -------
    None
    """
    discovered.append(current_vertex)
    for i in range(len(adj_mat[current_vertex])):
        if adj_mat[current_vertex][i] == 1 and (not i in discovered):
            explore(i, adj_mat, discovered)
            
def subtract(vec1, vec2):
    """Returns the result vector of vec1 - vec2. The program will terminate if
    their lengths do not agree.
    Parameters
    ----------
    vec1, vec2 : List[int]
        Vectors of integers.
    Returns
    -------
    result : List[int]
        The result of vec1 - vec2.
    """
    if len(vec1) != len(vec2):
        print("Vectors' length do not agree.")
        sys.exit()
    else:
        result = []
        for i in range(len(vec1)):
            result.append(vec1[i] - vec2[i])
        return result
        
def get_set(list):
    """Returns a set version of a list.
    Parameters
    ----------
    list : List[object]
    Returns
    -------
    new set
    """
    if len(list) == 0:
        return set()
    else:
        return {e for e in list}
        
def get_endpoints_from_edges(edges, vertex_amount):
    """Given a list of edges, returns a list of endpoint list of base 1.
    Parameters
    ----------
    edges : List[[int, int]]
        The edge list, any integer inside must base 1.
    vertex_amount : int
        The amount of vertices.
    Returns
    -------
    endpoints : List[List[int]]
    """
    endpoints = []
    for i in range(vertex_amount + 1):
        endpoints.append([])
    for e in edges:
        endpoints[e[0]].append(e[1])
        endpoints[e[1]].append(e[0])
    return endpoints
    

           
#                               End of Section                                #
###############################################################################
###############################################################################
#                               Main Functions                                #

def is_connected(adj_mat, vertex_amount):
    """Determines if a graph is connected.
    Parameters
    ----------
    adj_mat : List[List[int]]
        The adjacency matrix of a graph.
    vertex_amount : int
        The amount of vertices in a graph.
    Returns
    -------
    boolean value
        The result of the examination.
    """
    if len(adj_mat) == 0:
        return True
    else:
        discovered = []
        explore(0, adj_mat, discovered)
        return len(discovered) == len(adj_mat)
        
def is_complete(adj_mat):
    """Determines if a graph is complete.
    Parameters
    ----------
    adj_mat : List[List[int]]
        The adjacency matrix of a graph.
    Returns
    -------
    boolean value
        The result of the examination.
    """
    if len(adj_mat) == 0 or len(adj_mat[0]) == 0:
        return True
    else:
        for i in range(len(adj_mat)):
            for j in range(len(adj_mat[0])):
                if i != j and adj_mat[i][j] == 0:
                    return False
        return True
        
def get_chrom_poly(edge_list, n_current, n_total):
    """Computes the coefficients of the chromatic polynomial of a graph in such
    an order: c1, c2, c3, c4, etc. This function uses Zykov's algorithm.
    Parameters
    ----------
    edge_list : List[[int, int]]
        The list of edges.
    n_current : int
        The number of vertices in the current subgraph.
    n_total : int
        The number of vertices in the original graph.
    Returns
    -------
    result : List[int]
        The list of coefficients of the chromatic polynomial.
    """
    if (len(edge_list) == 0):
        result = [0] * n_total
        result[n_current - BASE] = 1
        return result
    else:
        redu1 = deepcopy(edge_list[1:])
        redu2 = deepcopy(edge_list[1:])
        first = edge_list[0]
        for edge in redu2:
            for i in range(len(edge)):
                if edge[i] == first[1]:
                    edge[i] = first[0]
        adj_mat = []
        for i in range(n_current):
            row = [0] * n_current
            for j in range(n_current):
                if [i + BASE, j + BASE] in redu2 or [j + BASE, i + BASE] in redu2:
                    row[j] = 1
            adj_mat.append(row)
        redu2 = []
        for i in range(n_current):
            for j in range(i + 1, n_current):
                if adj_mat[i][j] == 1:
                    redu2.append([i + BASE, j + BASE])
        result = subtract(get_chrom_poly(redu1, n_current, n_total),\
                        get_chrom_poly(redu2, n_current - 1, n_total))
        return result

def get_chrom_num(chrom_poly):
    """Computes the chromatic number of a graph.
    Parameters
    ----------
    chrom_poly : List[int]
        The list of coefficients of the chromatic polynomial.
    Returns
    -------
    chrom_num : int
        The chromatic number of a graph.
    """
    chrom_num = 0
    value = 0
    while value <= 0:
        chrom_num += 1
        for i in range(len(chrom_poly)):
            value += pow(chrom_num, i + BASE) * chrom_poly[i]
    return chrom_num

def BronKerbosch2(R, P, X, result, endpoints):
    """Searches for all maximal cliques in a given graph G using Bron-Kerbosch
    algorithm with pivot.
    Parameters
    ----------
    R, P, X : List[int]
        A list of vertices.
    result : List[List[int]]
        A list of maximal cliques.
    endpoints : List[List[int]]
        A list of neighbors for each vertex.
    Returns
    -------
    (by reference)result : List[List[int]]
        A list of maximal cliques.
    """
    if len(P) == 0 and len(X) == 0:
        result.append(R)
    else:
        union = P | X
        list_temp = list(union)
        u = list_temp[0]
        for v in (P - get_set(endpoints[u])):
            BronKerbosch2(R | {v}, P & get_set(endpoints[v]),\
                            X & get_set(endpoints[v]), result, endpoints)
            P = P - {v}
            X = X | {v}

def get_invariants(edge_list, vertex_amount):
    """Evaluates the invariants of a graph.
    Parameters
    ----------
    edge_list : List[[int, int]]
        The list of edges.
    vertex_amount : int
        The amount of vertices in a graph.
    Returns
    -------
    None
    """
    endpoints = get_endpoints_from_edges(edge_list, vertex_amount)
    start = time()
    adj_mat = get_adj_mat(edge_list, vertex_amount)
    completeness = is_complete(adj_mat)
    connectivity = is_connected(adj_mat, vertex_amount)
    end = time()
    print("Connected? " + str(connectivity))
    #print("Time used to determine connectivity = " + str(end - start) + " sec\n")
    
    start = time()
    char_poly = np.poly(adj_mat)
    end = time()
    print("Characteristic Polynomial Coefficients: " + str(char_poly))
    #print("Time used to determine characteristic polynomial = " + str(end - start) + " sec\n")
    
    
    start = time()
    chrom_poly = get_chrom_poly(edge_list, vertex_amount, vertex_amount)
    end = time()
    print("Chromatic Polynomial Coefficients: " + str(chrom_poly))
    #print("Time used to determine chromatic polynomial = " + str(end - start) + " sec\n")
    
    start = time()
    chrom_num = get_chrom_num(chrom_poly)
    end = time()
    print("Chromatic Number: " + str(chrom_num))
    #print("Time used to determine chromatic number = " + str(end - start) + " sec\n")
    
    
    start = time()
    maximal_cliques = []
    BronKerbosch2(set(), {(i + BASE) for i in range(vertex_amount)}, set(), maximal_cliques, endpoints)
    end = time()
    print("Maximal Cliques: " + str(maximal_cliques))
    #print("Time used to determine cliques = " + str(end - start) + " sec\n")
    
    '''
    print("Connected? " + str(connectivity))
    print("Characteristic Polynomial Coefficients: " + str(char_poly))
    print("Chromatic Polynomial Coefficients: " + str(chrom_poly))
    print("Chromatic Number: " + str(chrom_num))
    print("Maximal Cliques: " + str(maximal_cliques))
    '''

#                               End of Section                                #
###############################################################################
###############################################################################
#                               Executing Codes                               #
            
'''
edge_list = [[1,2],[1,6],[2,3],[2,9],[3,4],[3,12],[4,5],[4,15],[5,1],[5,18],[6,7],[6,20],[7,8],[7,21],[8,9],[8,24],[9,10],[10,11],[10,25],[11,12],[11,28],[12,13],[13,14],[13,29],[14,15],[14,32],[15,16],[16,17],[16,33],[17,18],[17,36],[18,19],[19,20],[19,37],[20,40],[21,22],[21,40],[22,23],[22,41],[23,24],[23,43],[24,25],[25,26],[26,27],[26,44],[27,28],[27,46],[28,29],[29,30],[30,31],[30,47],[31,32],[31,49],[32,33],[33,34],[34,35],[34,50],[35,36],[35,52],[36,37],[37,38],[38,39],[38,53],[39,40],[39,55],[41,42],[41,55],[42,43],[42,56],[43,44],[44,45],[45,46],[45,57],[46,47],[47,48],[48,49],[48,58],[49,50],[50,51],[51,52],[51,59],[52,53],[53,54],[54,55],[54,60],[56,57],[56,60],[57,58],[58,59],[59,60]]
vertex_amount = 60
#edge_list = [[1, 2], [2, 3], [3, 1]]
#vertex_amount = 3
endpoints = get_endpoints_from_edges(edge_list, vertex_amount)
deg_seq = []
for l in endpoints:
    deg_seq.append(len(l))
print(deg_seq)
get_invariants(edge_list, vertex_amount, endpoints)
'''

#                               End of Section                                #
###############################################################################