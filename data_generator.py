import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
import seaborn as sns

#################################################################################
# A simple function for generating random W,H matrices to be used 
# together with a diffusion kernel as input for our algorithm 
#
# INPUT:
# n - number of rows in the W matrix
# m - number of columns in H matrix
# rank - integer representing desired rank of the data, 
#           determines the shape of the output W,H
#
# state - integer for random seed used to help reproduce results
#
# OUTPUT: 
# W - a n x rank matrix of sin waves with differing frequency and 
#   amplitude with values strictly between 0 and 1
#
# H - a rank x m sparse matrix with entries between 0 and 1
#
# Note: the matrix product of W and H produces an n x m matrix
#
#################################################################################


def gen_decomposition(n, m, rank, state = None):
    
    # Generate random H with scipy's sparse random
    test_h = sp.random(rank, m, density = 0.07, random_state = state).A
    
    # generate new H if we have any zero rows
    while np.where(np.sum(test_h, axis = 1) == 0)[0].size != 0:
        test_h = sp.random(rank, m, density = 0.07, random_state = state).A
     
    # Normalize row sums of H so that they all sum to 1
    for g in range(len(test_h)):
        scal = test_h[g,:].sum()
        test_h[g,:] /= scal

    H = pd.DataFrame(test_h)
    
    
    # generate sin waves of data for W
    time = np.linspace(1,n,n)
    np.random.seed(state)
    freqs = np.random.normal(0,0.2,rank)
    waves = np.outer(time, freqs)
    for col in range(waves.shape[1]):
        waves[:,col] = (col + 1) * np.sin(waves[:,col])
    
    # Normalize W to lie between 0 and 1
    W = pd.DataFrame(waves)
    W /= W.max().max() * 2
    W += 0.5
    
    return W,H

########################################################################
# Another quick function to generate laplacian matrices
# Simply generates a random geometric graph based on given
# inputs:
#
# size - number of nodes in the graph
# H - input H matrix generated from the function above
# p_edge - probability that an edge exist between any two nodes
# state - random seed for graph generation
#
# And outputs the corresponding laplacian matrix of the graph
########################################################################

def gen_laplacian(size, radius = 1, state = None):
    graph = nx.generators.geometric.random_geometric_graph(n = size, radius = radius, seed = state)
    
    #palette = sns.color_palette(n_colors = H.shape[0])
    # assign initial colors/weights
    ''' 
    for i,j in graph.edges:
        graph[i][j]['color'] = (0,0,0)
        graph[i][j]['width'] = 0.5
    '''
        
    laplacian = nx.linalg.laplacianmatrix.laplacian_matrix(graph)
    return graph, laplacian.toarray()



################################################################################
# Very simple function for adding random noise to synthetic data
# By iterating over every element and adding a sample from the 
# normal distribution with mean: 0 and standard deviation: std_dev
#
# input:
#   data - numpy array representing synthetic data matrix
#   std_dev - standard of deviation used for random sampling
#
# output:
#   matr - numpy array of data with random noise added 
#
################################################################################

def add_noise(data, std_dev = None):
    matr = np.matrix.copy(data)
    if std_dev is None:
        std_dev = matr.std() / 100

    for rower in range(matr.shape[0]):
        for coler in range(matr.shape[1]):
            noisy = np.random.normal(scale = std_dev)
            if matr[rower, coler] + noisy < 0:
                matr[rower, coler] = 0
            else:
                matr[rower, coler] += noisy
    
    return matr


##################################################################################################
# Creates a matrix of 0s -- unknown values -- and 1's -- known values ---
# in order to "mask" the data and split it into train/test sets
#
# input:
#   data - numpy array data matrix
#   percent_hide - the percent of matrix entries which will be randomly deemed as unknown -- 0
#
# output:
#   mask - a numpy array of 0's and 1's where each 0 entry denotes an entry that will
#            be hidden for testing and each 1 denotes an entry that will be used for training.
#
################################################################################################

def train_mask(data, percent_hide):
    # "hide" a given percentage of the data
    num_entries = data.shape[0]*data.shape[1]
    mask = np.zeros(num_entries)
    mask[:int(num_entries * (1 - percent_hide))] = 1
    np.random.shuffle(mask)
    mask = mask.reshape(data.shape)
    
    return mask