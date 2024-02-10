# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 15:26:07 2024

@author: kevin
"""
import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
import itertools
from scipy.optimize import minimize

import matplotlib 
# %% Note
# generate full discrete time matrix
# all elements contain f*exp(w) and r
# zero-out pairs more than one bit flip

# if works!! (hopefully)
# sim 5 spiking neurons
# get 5 neurons retinal data
def general_M(param):
    """
    a tilted matrix for ctmc neural circuit generalized to arbitrary size
    the parameter contains fields of parameters, f,r,w, which informs the size
    the output is the state x state titled matrix
    """
    fs, rs, w_pair, w_trip = param  # find out a way to load these sublists from a param tuple
    n = len(fs)  # number of neurons
    spins = [0,1]  # binary patterns
    combinations = list(itertools.product(spins, repeat=n))  # possible configurations
    
    ### idea: M = mask*R*F, with mask for ctmc, R for refractory, and F for firing
    mask = np.ones((n,n))  # initialize the tilted matrix
    R = mask*1
    F = mask*1
    mask_r = mask*1
    # make the mask
    for ii in range(n):
        for jj in range(n):
            # Only allow one flip logic in ctmc
            if sum(x != y for x, y in zip(combinations[ii], combinations[jj])) != 1:
                mask[ii,jj] = 0
    
    # make R matrix
    for ii in range(n):
        for jj in range(n):
            # Find the indices where the differences occur
            indices = [index for index, (x, y) in enumerate(zip(combinations[ii], combinations[jj])) if x != y]
            
            # Check if there is exactly one difference, it goes from 1 to 0, and store the indices
            if len(indices) == 1 and combinations[ii][indices[0]] == 1 and combinations[jj][indices[0]] == 0:
                R[ii,jj] = rs[indices[0]]  # use the corresponding refractoriness
                mask_r[ii,jj] = 0
    
    # now make F matrix!
    for ii in range(n):
        for jj in range(n):
            if mask[ii,jj]==1 and mask_r[ii,jj]==1: #only check those that generates one spike
                diff = np.array(combinations[jj]) - np.array(combinations[ii])  # should only have one element that is one!
                pos_fire = np.where(diff==1)[0][0]
                wij = find_wij(combinations[ii],combinations[jj]) # calling a function for w
                F[ii,jj] = fs[pos_fire]*np.exp(wij)

    M = mask*R*F      
    return M

def find_wij(spin_i, spin_j, Ws):
    """
    input two spin configurations and the coupling dictionary Ws, pick the correponsding w
    """
    wij=0 #.... implement this!
    return wij

# %% alternative/better solution
# directly work with f*exp(w)
# solve f, then w is known!
# Cp: start with edge-flux and pi00, pi11

# %%
# given N neurons
# 2^N x 2^N
# number of parameters is the edge count for a hypercube
# N*2^N
