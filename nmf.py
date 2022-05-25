import numpy as np
import pandas as pd


#####################################################################################################################
# A simple Non Negative Matrix Factorization implementation
# Author: Kevin Quinn 10/21/21

# Update rules based on the results established in https://arxiv.org/pdf/1612.06037.pdf

# Specifically, this implementation takes into account that not all values are known,
# and masks the known/unknown values to account for this (More details in the paper)

# INPUT:

#     X - Initial n x m data matrix (with 0s or some other float value to represent missing values)

#     ncomponents - The desired rank k to be used in Factorization

#     M - data mask (1's for known values and 0's for unknown)

#     iterations - desired number of iterations to complete before convergence

#     tol - value at which we decide the change in cost function is small enough to declare convergence 

# OUTPUT:
#     W - n x k factor matrix
#     H - k x m factor matrix

####################################################################################################################

def MultUpdate(X, W, H, mask):
    # Update based on standard multiplicative update rules + adjusting for masked values

    # Mask unknown values if any
    masked_X = np.multiply(mask, X)
    masked_WH = np.multiply(mask, np.dot(W, H))

    # Update W
    num_w = np.dot(masked_X, H.T)
    denom_w = np.dot(masked_WH, H.T) + 1e-9 # add 1e-9 to make sure no 0s in the denominator
    lrw = np.divide(num_w, denom_w)
    W = np.multiply(W, lrw)

    # re-apply mask after updating W
    masked_WH = np.multiply(mask, np.dot(W, H))

    # Update H
    num_h = np.dot(W.T, masked_X)
    denom_h = np.dot(W.T, masked_WH) + 1e-9 
    lrh = np.divide(num_h, denom_h)
    H = np.multiply(H, lrh)
    
    return W, H


def solver(X, W, H, mask, max_iter, tol):
    # initial objective (cost function)
    O = np.linalg.norm((mask ** (1/2)) * (X - np.dot(W, H)))

    iteri = 0

    while iteri < max_iter:
        old_dist = O

        W, H = MultUpdate(X, W, H, mask)

        # calculate change in cost and return if its small enough
        O = np.linalg.norm((mask ** (1/2)) * (X - np.dot(W, H)))
        change = abs(old_dist - O)

        if change < tol:
            break

        iteri += 1
        
    
    return W, H, iteri



class nmf:
    def __init__(self, n_components = None, mask = None, n_iter = 200, tol = 1e-5):
        self.n_components = n_components
        self.mask = mask
        self.n_iter = n_iter
        self.tol = tol


    def _check_params(self, X):
        # method to check all initial input parameters
        
        # Check input data
        if X.min() < 0:
            raise ValueError("all elements of input data must be positive")
            
        
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
        
        

        
    def check_w_h(self, X, W, H):
        if W is None:
            W = np.random.rand(len(X), self.n_components)
        else:
            if W.shape != (len(X), self.n_components):
                raise ValueError("W input should be of size " + str((len(X), self.n_components)))
            
        if H is None:
            H = np.random.rand(self.n_components, X.shape[1])
        else:
            if H.shape != (self.n_components, X.shape[1]):
                raise ValueError("H input should be of size " + str((self.n_components, X.shape[1])))
            
        return W,H
        
        
        
    def fit_transform(self, X, W = None, H = None):
        # initialize X and V
        
        #X = self._validate_data(X, dtype=[np.float64, np.float32])
        try:
            X = np.array(X)
        except:
            raise ValueError("Input data must be array-like")
        
        W,H = self.check_w_h(X, W, H)
        
        self._check_params(X)
        
        
        W,H, n_iters = solver(X, W, H, self.mask, self.n_iter, self.tol)
        
        if n_iters == self.n_iter:
            print("Max iterations reached, increase to converge on given tolerance")
        
        return W,H

