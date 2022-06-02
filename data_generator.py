import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
import seaborn as sns
from sklearn.isotonic import IsotonicRegression
import copy

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


##################################################################################################
# Isotonic Regression - fixes the problem of decreasing data points
# for data like cumulative COVID-19 cases, every new day should have cases >= the previous day.
# However, the raw data we have collected does not always reflect this (errors in case counts).
# To fix this we fit a strictly increasing line to the data. (Note that if a feature is already 
# strictly increasing then nothing will change)
#
# input:
#   data - numpy or pandas data matrix
#   axis - 0 or 1 (0 to operate row-wise and 1 to operate column-wise)
#
# output:
#   iso_data - new data matrix for which all entries along given axis are strictly increasing
#
###################################################################################################

def iso_regression(data, axis = 1):
    # check inputs:
    
    # store column and row names if working with a pandas dataframe (will return them later)
    panda = False
    if isinstance(data, pd.DataFrame):
        panda = True
        ind = data.index
        cols = data.columns
        data = data.to_numpy()

    # if we need to work with rows, just transpose for now and transpose back at the end
    if axis == 0:
        data = data.T
    elif axis != 1:
        raise ValueError("axis variable must be either 0 or 1")
    
    # empty matrix to fill with corrected columns
    iso_data = np.zeros(data.shape)
    
    # perform isotonic regression on all columns
    for column in range(data.shape[1]):
        iso = IsotonicRegression(out_of_bounds = "clip")
        iso.fit(range(data.shape[0]), data[:,column])
        iso_data[:,column] = iso.predict(range(data.shape[0]))
    
    
    # return shape and type of original input
    if axis == 0:
        iso_data = iso_data.T
    
    if panda:
        iso_data = pd.DataFrame(iso_data, index = ind, columns = cols)
        
    
    return iso_data


#################################################################################
# A quick and easy function that helps with presentation of output matrices
# from the diffusion NMF algorithm

# given that D-NMF outputs X and V matrices this function rescales them as follows:
# X is divided by its maximum element so that everything falls on a scale of 0-1
# V is multiplied by X's maximum element in order to absorb the term that was factored out of X

# input:
#   - X - (m x k) matrix from D-NMF output
#   - V - (k x n) matrix from D-NMF output

# output:
#   - rescaled X
#   - rescaled V

################################################################################
def rescale(X,V):
    X = copy.deepcopy(X)
    V = copy.deepcopy(V)

    maxer = np.amax(X)
    X /= maxer
    V *= maxer
    
    return X,V