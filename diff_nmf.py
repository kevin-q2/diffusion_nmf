import math
import numpy as np 
import pandas as pd
#from sklearn.base import BaseEstimator, TransformerMixin
#from scipy.sparse import random

#######################################################################################################
# DIFFUSION NMF - Kevin Quinn 12/20/2021
#
# The following is an implementation of a Diffusion NMF algorithm
# i.e. a modified NMF algorithm which is formulated as follows:
#
# D = X V K
#
# Where D is the original data matrix 
# K is a diffusion Kernel which describes the diffusion process in the data
#   Note: for our purposes we have chosen to  use a kernel of the form (I + beta * Laplacian)^-1
# And X,V are output factorizations similar to the results obtained from a standard NMF factorization
#
# Using this code requires using the diffusionNMF class which is described below:
#
#######################################################################################################



# MultUpdate():
# First, a function to compute the multiplicative update step within our algorithm
# i.e. at each iteration this update is performed and is formulated within our paper
# INPUT:
#   D - Data Matrix
#   X - current version of X factorization
#   V - current version of V factorization
#   K - Diffusion Kernel
#   mask - matrix of equal shape to D where each entry is 0/1 corresponding to hidden/unhidden
#
# OUTPUT:
#   X - Updated X factorization
#   V - updated V factorization
#

def MultUpdate(D, X, V, K, mask):

    # Adjust for Masked Values:
    masked_d = mask * D
    masked_xvk = mask * np.dot(X, np.dot(V, K))
    
    # Update X
    num_x = np.dot(masked_d, np.dot(K.T, V.T))
    denom_x = np.dot(masked_xvk, np.dot(K.T, V.T)) + 1e-10 # add small noise to ensure the denominator is never 0
    grad_x = np.divide(num_x, denom_x)
    X = np.multiply(X, grad_x)
    
    
    # Update V
    masked_xvk = mask * np.dot(X, np.dot(V, K))
    num_v = np.dot(X.T, np.dot(masked_d, K.T))
    denom_v = np.dot(X.T, np.dot(masked_xvk, K.T)) + 1e-10
    grad_v = np.divide(num_v, denom_v)
    V = np.multiply(V, grad_v)
    
    return X, V


# solver():
# is a the function which performs the iterative steps in our algorithm
# it calls MultUpdate until convergence is acheived
#
# INPUT:
#   D - Data Matrix
#   X - current version of X factorization
#   V - current version of V factorization
#   K - Diffusion Kernel
#   mask - matrix of equal shape to D where each entry is 0/1 corresponding to hidden/unhidden
#   max_iter - maximum iterations to go through before declaring we've finished
#   tol - tolerance level at which we can claim convergence
 
def solver(D, X, V, K, mask, max_iter, tol):
    # initial objective (cost function)
    O = np.linalg.norm(mask * (D - np.dot(X, np.dot(V, K))))

    iteri = 0

    while iteri < max_iter:
        old_dist = O

        X, V = MultUpdate(D, X, V, K, mask)

        # calculate change in cost and return if its small enough
        O = np.linalg.norm(mask * (D- np.dot(X, np.dot(V, K))))
        change = abs(old_dist - O)

        if change < tol:
            break

        iteri += 1
        
    
    return X, V, iteri


######################################################################################
#
# diffusionNMF:
# 
# INPUT:
#   n_components (int) 
#       - chosen rank for which to compute the factorization
#   kernel (2d numpy array or similar)
#       - Diffusion Kernel to use in factorization
#   mask (2d numpy array or similar)
#       - training/testing 0-1 data matrix to represent hidden/unhidden values
#   n_iter (int) 
#       - absolute maximum number of iterations to perform
#   tol (float) 
#       - tolerance level at which to declare convergence
#   progress (bool) 
#       - prints progress of the algorithm (NOT IMPLEMENTED YET)
#
# TO USE:
#   1. pass in values to create a diffusion NMF object 
#   2. call fit_transform(Data) -- further described below
#
#######################################################################################



#class diffusionNMF(TransformerMixin, BaseEstimator):
class diff_nmf():
    def __init__(self, n_components = None, kernel = None, mask = None, n_iter = 500, tol = 1e-10, progress = False):
        self.n_components = n_components
        self.kernel = kernel
        self.mask = mask
        self.n_iter = n_iter
        self.tol = tol
        self.progress = progress
        
        
    # Check initial parameters of the data to make sure they're adequate    
    def _check_params(self, X):
        # method to check all initial input parameters
        
        # Check input data
        if X.min() < 0:
            raise ValueError("all elements of input data must be positive")
        
        # Check Kernel input
        if self.kernel is None:
            raise ValueError("Need to provide diffusion kernel")
        else:
            try:
                self.kernel = np.array(self.kernel)
            except:
                raise ValueError('Input array must be 2d numpy array or similar')
            
            if self.kernel.shape[0] != self.kernel.shape[1]:
                raise ValueError('Diffusion Kernel must be a square matrix')
            elif self.kernel.shape[0] != X.shape[1]:
                print(self.kernel.shape)
                print(X.shape)
                raise ValueError("Size of diffusion kernel must match the size of the data's features")
            
        
        # check mask input:
        if self.mask is None:
            self.mask = np.ones(X.shape)
        else:
            if self.mask.shape != X.shape:
                raise ValueError("Input mask must match the size of the data")
        
        
        # Check rank parameter 
        if not isinstance(self.n_components, int) or self.n_components <= 0:
            raise ValueError("Rank must be a positive integer")
        
        # check iterations 
        if not isinstance(self.n_iter, int) or self.n_iter <= 0:
            raise ValueError("Number of iterations must be a positive integer")
        
        
        # check tolerance level
        if not isinstance(self.tol, float) or self.tol <= 0:
            raise ValueError("Tolerance level must be positive floating point value")
            
            
        # Check progress parameter
        if not isinstance(self.progress, bool):
            raise ValueError("Progress parameter must be boolean value")
        
        

    # IF initial X,V are given, check to make sure theyre adequate     
    # ElsE if theyre not initialized, randomly initialize them 
    def check_w_h(self, D, X, V):
        if X is None:
            X = np.random.rand(len(D), self.n_components)
        else:
            if X.shape != (len(D), self.n_components):
                raise ValueError("X input should be of size " + str((len(D), self.n_components)))
            
        if V is None:
            V = np.random.rand(self.n_components, D.shape[1])
        else:
            if V.shape != (self.n_components, D.shape[1]):
                raise ValueError("V input should be of size " + str((self.n_components, D.shape[1])))
            
        return X,V
        
        
    # Starts the multiplicative update process for given data using given attributes
    # INPUT:
    #   D - data matrix -- 2d numpy array or similar
    #   X (OPTIONAL) - initial factorization matrix -- 2d numpy array or similar
    #   V (OPTIONAL) - initial factorization matrix -- 2d numpy array or similar
    #
    # OUTPUT:
    #   X - computed factorization matrix
    #   V - computed factorization matrix
    #
      
    def fit_transform(self, D, X = None, V = None):
        
        try:
            D = np.array(D)
        except:
            raise ValueError("Input data must be array-like")
        
        X,V = self.check_w_h(D, X, V)
        
        self._check_params(D)
        
        
        X,V, n_iters = solver(D, X, V, self.kernel, self.mask, self.n_iter, self.tol)
        
        if n_iters == self.n_iter:
            print("Max iterations reached, increase to converge on given tolerance")
        
        return X,V