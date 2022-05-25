import numpy as np 
import pandas as pd
from joblib import Parallel, delayed
import diff_nmf
import nmf
from diff_nmf import diff_nmf
from nmf import nmf


#########################################################################################################
#
# Class to help with the search for correct paramters to use as input for 
# Diffusion NMF. In particular this will perform a grid search over a given
# input space of parameters and report the error corresponding to each point 
# on the grid
#
# INPUT:
#   X - data matrix (numpy array)
#   laplacian - graph laplacian used to make the diffusion kernel (numpy array)
#   algorithm - "diffusion" for diffusion NMF or "nmf" for standard NMF
#   max_iter - max iterations for which to run the factorization algorithm
#   tolerance - tolerance level at which to run the factorization algorithm
#   percent_hide - percent of entries to mask (for train/test purposes)
#   noise - standard deviation of random noise to add to the data
#   validate - number of trials to run for each point on the parameter grid
#   saver - filename to save results to
#
# METHODS:
#   kernelize(beta) 
#       - given a beta value, use the laplacian attribute to compute the diffusion kernel
#   train_mask()
#       - create a random 0/1 matrix (0 = hidden, 1 = non hidden) using percent_hide attribute
#   add_noise(data, std_dev)
#       - given a set of data and a std_dev value, add samples from a normal(0, std_dev) distribution to
#            each element of the data
#   relative_error(W,H,K,mask)
#       - given the data, its factorization, the kernel, and the mask, compute the relative error on the hidden
#        (test) entries
#   param_solver(rank, beta)
#       - for a given rank and beta, dispatch a run of diffusion NMF or NMF from which we can then compute error
#   post_process(results)
#       - for internal use only, just takes the outputs and puts them into a pandas dataframe
#   
#   **grid_search(rank_list, beta_list)
#       - Where the magic happens! to begin a grid search pass in a list of ranks and beta values to sample from
#         this method dispatches trials in parallel and will return results in a nice pandas dataframe!
#
##########################################################################################################

class gridSearcher:
    def __init__(self, X, laplacian = None, algorithm = "diffusion", max_iter = 100000, tolerance = 1e-9, percent_hide = 0.2, noise = None, validate = 5, saver = None):
        self.X = X
        self.laplacian = laplacian
        self.algorithm = algorithm 
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.percent_hide = percent_hide
        self.noise = noise
        self.validate = validate
        self.saver = saver

    def kernelize(self, beta):
        I = np.identity(self.laplacian.shape[0])
        return np.linalg.inv(I + beta * self.laplacian)
    
    def train_mask(self):
        # "hide" a given percentage of the data
        num_entries = self.X.shape[0]*self.X.shape[1]
        mask = np.zeros(num_entries)
        mask[:int(num_entries * (1 - self.percent_hide))] = 1
        np.random.shuffle(mask)
        mask = mask.reshape(self.X.shape)

        return mask
    
    def add_noise(self, data, std_dev = None):
        if std_dev is None:
            return data

        else:
            matr = np.matrix.copy(data)
            for rower in range(matr.shape[0]):
                for coler in range(matr.shape[1]):
                    noisy = np.random.normal(scale = std_dev)
                    if matr[rower, coler] + noisy < 0:
                        matr[rower, coler] = 0
                    else:
                        matr[rower, coler] += noisy
            
            return matr
    
    def relative_error(self, W, H, K, mask):
        if self.algorithm == "nmf":
            error = np.linalg.norm((1 - mask) * (self.X - np.dot(W,H)))
            baseline = np.linalg.norm((1 - mask) * self.X)
        else:
            error = np.linalg.norm((1 - mask) * (self.X - np.dot(W, np.dot(H, K))))
            baseline = np.linalg.norm((1 - mask) * self.X)
            
        return error/baseline
    
    def param_solver(self, rank, beta):
        M = self.train_mask()
        K = None
        noisy = self.add_noise(self.X, self.noise)
        
        if self.algorithm == "nmf":
            nSolver = nmf(n_components = rank, mask = M, n_iter = self.max_iter, tol = self.tolerance)
            W,H = nSolver.fit_transform(noisy)
        else:
            K = self.kernelize(beta)
            dSolver = diff_nmf(n_components = rank, kernel = K, mask = M, n_iter = self.max_iter, tol = self.tolerance)
            W,H = dSolver.fit_transform(noisy)
        
        rel_error = self.relative_error(W,H,K,M)
        return rank, beta, rel_error
    
    
    def post_process(self, results):
        res_frame = pd.DataFrame(results)
        res_frame.columns = ["rank", "beta", "relative error"]
        m = res_frame.groupby(["rank","beta"]).mean()
        s = res_frame.groupby(["rank","beta"]).std()
        s.columns = ["std error"]
        
        res_frame = pd.concat([m,s], axis = 1)
        
        if not self.saver is None:
            res_frame.to_csv(self.saver)
            
        return res_frame
    
    def grid_search(self, rank_list, beta_list):
        trials = []
        
        for r in rank_list:
            for b in beta_list:
                for v in range(self.validate):
                    trials.append((r,b))
                
                
        res = Parallel(n_jobs = -1, verbose = 1)(delayed(self.param_solver)(r,b) for (r,b) in trials)
        
        return self.post_process(res)
    

    