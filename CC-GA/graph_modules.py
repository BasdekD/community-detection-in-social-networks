# In this file I include necessary data and helper functions for graph representation

import numpy as np


# Method to create the adjacency matrix (in case not provided)
def crt_adj_mat(node_num, edge_list):
    '''
        edje_list: Should be a list of tuples
    '''
    # Initially create a zero square matrix
    adjmat = np.zeros(shape=(node_num,node_num))
    # Then populate it
    for x,y in edge_list:
        adjmat[int(x),int(y)] = 1
        adjmat[int(y),int(x)] = 1
    return(adjmat)


#### Calculate nodes' Clustering Coefficient matrix based on the adjacency matrix ####
# First create helper matrices containing nodes' degrees and L values, where Li is
# the total number of links among neighbours of the node vi, excluding the node itself

# Method to create nodes' degree matrix based on the adjacency matrix
# For each node I calculate the sum of corresponding row (or column) on adjacency matrix
def calc_dg_mat(adjmatrix):
    # Number of nodes
    nd_num = adjmatrix.shape[0]
    # Initially create a zero matrix
    dgmat = np.zeros(shape=(nd_num))
    # Then populate it accordingly
    for i in range(nd_num):
        dgmat[i] = adjmatrix[i].sum()
    return(dgmat)


# Method to calculate the L values of nodes based on the adjacency matrix
def calc_l_mat(adjmatrix):
    # Number of nodes
    nd_num = adjmatrix.shape[0]
    lmat = np.zeros(shape=(nd_num))
    # For each node
    for i in range(nd_num):
        # Find node's neighbors (indices)
        neighs = np.where(adjmatrix[i] == 1)[0]
        # For each possible couple of neighbors add 1 if there is an edge among them
        total = 0
        for nbr1 in neighs:
            for nbr2 in neighs:
                total += adjmatrix[nbr1,nbr2]
         # Each edge is taken in account twice, so keep half of total calculated
        lmat[i] = total/2
    return(lmat)


# Method to calculate clustering coefficient matrix based on the adjacency matrix
def calc_cc_mat(adjmatrix):
    # First gather all needed values
    # Number of nodes
    nd_num = adjmatrix.shape[0]
    # Matrix with nodes' degrees
    dgmat = calc_dg_mat(adjmatrix)
    # Matrix with nodes' L values
    lmat = calc_l_mat(adjmatrix)
    
    # Create an empty clustering coefficient matrix
    ccmat = np.zeros(shape=(nd_num))
    # For each node
    for i in range(nd_num):
        # denominator should never be zero
        if(dgmat[i] < 2):
            ccmat[i] =0
        else:
            ccmat[i] = 2*lmat[i]/(dgmat[i]*(dgmat[i]-1))

    return(ccmat)



