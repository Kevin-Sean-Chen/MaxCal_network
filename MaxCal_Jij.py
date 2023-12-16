# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 23:36:12 2023

@author: kevin
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
import itertools
from scipy.optimize import minimize

import matplotlib 
matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('ytick', labelsize=30)

# %% Discrete-time binary neural network model
def spiking(p_spk, spk_past):
    """
    I: firing probability vector and past spike
    O: binary spiking vector
    """
    N = len(p_spk)
    rand = np.random.rand(N)
    spk = np.zeros(N)
    spk[p_spk>rand] = 1  # probablistic firing
#    spk[spk_past==1] = 0  # refractoriness
    return spk

def nonlinearity(x):
    """
    I: current vector
    O: firing probability vector
    """
    base_rate = 0.0
    f = (1-base_rate)/(1+np.exp(-x)) + base_rate  # sigmoid nonlinearity
    return f

def current(Jij, spk):
    baseline = 0
    I = Jij @ spk + baseline  # coupling and baseline
    return I

# %% initializations
T = 50000
N = 4
gamma = 1  # scaling coupling strength
Jij = np.random.randn(N,N)*gamma/N**0.5  # coupling matrix
sigma = np.zeros((N,T))

# %% Neural dynamics
for tt in range(0,T-1):
    p_spk = nonlinearity( current(Jij, sigma[:,tt]) )
    sigma[:,tt+1] = spiking(p_spk, sigma[:,tt])

plt.figure()
plt.imshow(sigma,aspect='auto',cmap='Greys',  interpolation='none')

# %% Inference
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %% functions
eps = 10**-10*0
def inv_f(x):
    """
    Stable inverse function of sigmoid nonlinearity
    """
    invf = -np.log((1 / (x + eps)) - 1)  # stable inverse of sigmoid
    return invf

def J_inf(sigma):
    """
    Write into a scalable function once confirmed...
    """
    return Jij

# %% counting through time with the wild approxiamtion (ignoring other 'r' states)
Jinf = Jij*0  #inferred Jij

for ii in range(N):
    for jj in range(N):
        temp = sigma[[ii,jj],:]  # selectec i,j pair
        if ii is not jj:
            ### counting inference
            pos10 = np.where((temp.T == (1,0)).all(axis=1))[0]  # find pair matching
            n10 = len(pos10)
            pos01 = np.where((temp.T == (0,1)).all(axis=1))[0]
            pos10_01 = np.intersect1d(pos10-1, pos01)  # find intersept one step behind
            n10_01 = len(pos10_01)
            p_1001 = n10_01 / n10  # normalize counts
            
            pos00 = np.where((temp.T == (0,0)).all(axis=1))[0]
            n00 = len(pos00)
            pos00_01 = np.intersect1d(pos00-1, pos01)
            n00_01 = len(pos00_01)
            p_0001 = n00_01 / n00
            
            if p_1001>0 and p_0001>0:
                Jinf[ii,jj] = inv_f(p_1001) - inv_f(p_0001)  # equation 11, ignoring r pattern
            else:
                Jinf[ii,jj] = np.nan
plt.figure()
mask = np.ones((N,N), dtype=bool)
np.fill_diagonal(mask, 0)
plt.plot(Jij[mask], Jinf[mask], 'ko')
plt.xlabel('true Jij', fontsize=30)
plt.ylabel('inferred Jij', fontsize=30)

# %% calculation in eqn. 11, picking proper 'r' pattern (works!!!)
n_comb = N-2  # the rest other than a pair
spins = [0,1]  # binary patterns
combinations = list(itertools.product(spins, repeat=n_comb))  # spin combinations
neuron_index = np.arange(0,N)  # vector for neuron index
Jinf = Jij*0  # inferred network
r_past = combinations[np.random.randint(N-2)]  # picking the conditional pattern

for ii in range(N):
    for jj in range(N):
        if ii is not jj:
            temp = sigma[[ii,jj],:]  # selectec i,j pair
            ind_r = [neuron_index[i] for i in range(N) if i!=ii and i!=jj]
            temp_r = sigma[ind_r,:]
            pos_rpast =  np.where((temp_r.T == r_past).all(axis=1))[0]  # fixed conditional r pattern
            
            ### 10 to x1
            pos10 = np.where((temp.T == (1,0)).all(axis=1))[0]
            pos10r = np.intersect1d(pos10, pos_rpast) #cond_r)
            n10r = len(pos10r)
            pos01 = np.where((temp.T == (0,1)).all(axis=1))[0]
            pos11 = np.where((temp.T == (1,1)).all(axis=1))[0]
            pos_1 = np.concatenate((pos01,pos11))
            pos10_1 = np.intersect1d(pos10r-1, pos_1)
            n10_1r = len(pos10_1)
            if n10r is not 0:
                p_10r_1 = n10_1r / n10r
                
            ### 00 to x1
            pos00 = np.where((temp.T == (0,0)).all(axis=1))[0]
            pos00r = np.intersect1d(pos00, pos_rpast) #cond_r)
            n00r = len(pos00r)
            pos00_1 = np.intersect1d(pos00r-1, pos_1)
            n00_1r = len(pos00_1)
            if n00r is not 0:
                p_00r_1 = n00_1r / n00r
                
            ### calculation
            if p_10r_1>0 and p_00r_1>0:
                Jinf[ii,jj] = inv_f(p_10r_1) - inv_f(p_00r_1)
            else:
                Jinf[ii,jj] = np.nan
plt.figure()
mask = np.ones((N,N), dtype=bool)
np.fill_diagonal(mask, 0)
plt.plot(Jij[mask], Jinf[mask], 'ko')
plt.xlabel('true Jij', fontsize=30)
plt.ylabel('inferred Jij', fontsize=30)
            

# %% test with coarse-graining (burst measurements)
# inference with fully observed network
# inference with coarse-grained measurements
# what are the features that can and cannot be inferred
# solve with small model, then apply to larger simulations

# %% measure g(x,y) as bursts
Bt = np.sum(sigma,0)  # burst, sum of spikes through time
dB = np.diff(Bt)
# compute some state transitions, and check how it constrains Jij
# compute emperical transitions firsy
N_state = len(np.unique(Bt))
Bij = np.zeros((N_state, N_state))
pi_B = np.zeros(N_state)
for tt in range(T-1):
    current_bst, next_bst = int(Bt[tt]), int(Bt[tt+1])
    Bij[current_bst, next_bst] += 1
    pi_B[current_bst] += 1

Bij = Bij/pi_B[:,None]

# %% MaxCal recipe
def gxy_state_test(x, y, obs=0):
    gxy = 0
    # write down some bust constraints here
    return gxy

###
    # start with 2 neurons
    # 6 observables with bursts: 2pi 3 eta 1 J
    # try MaxCal to infer 2r, 2W, 2f
    # put down 1/3 prior and test if symmetry constains inference
    # ... numerically show it preserves for N=5-10 neurons; theoretical
###

# %% more careful calculaion of eqn. 11, old/incorrect version
#n_comb = N-2  # the rest other than a pair
#spins = [0,1]  # binary patterns
#combinations = list(itertools.product(spins, repeat=n_comb))  # spin combinations
#Jinf = Jij*0
#neuron_index = np.arange(0,N)
#dumy = 0
#r_past = combinations[np.random.randint(N-2)]  # fix one past condition, can late use the frequent one
#
#for ii in range(N):
#    for jj in range(N-1):
#        temp = sigma[[ii,jj],:]  # selectec i,j pair
#        ind_r = [neuron_index[i] for i in range(N) if i!=ii and i!=jj]
#        temp_r = sigma[ind_r,:]
#        if ii is not jj:
#            ### condition on patterns
#            p_1001 = 0  # initialize for each pair
#            p_0001 = 0
#            pos_rpast =  np.where((temp_r.T == r_past).all(axis=1))[0]  # fixed conditional r pattern
#            
#            for rr in range(len(combinations)):
#                r_ = combinations[rr]
#                cond_r = np.where((temp_r.T == r_).all(axis=1))[0]  # all positions conditioned on pattern r
#                
#                pos10 = np.where((temp.T == (1,0)).all(axis=1))[0]
#                pos10r = np.intersect1d(pos10, pos_rpast) #cond_r)
#                n10r = len(pos10r)
#                pos01 = np.where((temp.T == (0,1)).all(axis=1))[0]
#                pos10_01 = np.intersect1d(pos10r-1, pos01)
#                pos10_01r = np.intersect1d(pos10_01, cond_r)
#                n10_01r = len(pos10_01r)
#                if n10r is not 0:
#                    p_1001r = n10_01r / n10r
#                    p_1001 += p_1001r
#                    
#                pos00 = np.where((temp.T == (0,0)).all(axis=1))[0]
#                pos00r = np.intersect1d(pos00, pos_rpast) #cond_r)
#                n00r = len(pos00r)
#                pos00_01 = np.intersect1d(pos00r-1, pos01)
#                pos00_01r = np.intersect1d(pos00_01, cond_r)
#                n00_01r = len(pos00_01r)
#                if n00r is not 0:
#                    p_0001r = n00_01r / n00r
#                    p_0001 += p_0001r
#                elif n00r is 0:
#                    print('devide by zero')
#                
##                if ii==1 and jj==2:  # checking counts
##                    dumy += n00_01
#            
#            if p_1001>0 and p_0001>0:
#                Jinf[ii,jj] = inv_f(p_1001) - inv_f(p_0001)
#            else:
#                Jinf[ii,jj] = np.nan
#plt.figure()
#mask = np.ones((N,N), dtype=bool)
#np.fill_diagonal(mask, 0)
#plt.plot(Jij[mask], Jinf[mask], 'ko')
#plt.xlabel('true Jij', fontsize=30)
#plt.ylabel('inferred Jij', fontsize=30)

# %% test code for Markov process
###############################################################################
# %%
import numpy as np

# Define a transition matrix for a 3-state Markov Chain
P_true = np.array([[0.7, 0.2, 0.1],
                   [0.3, 0.5, 0.2],
                   [0.1, 0.4, 0.5]])

# Generate some synthetic data from the true Markov Chain
np.random.seed(123)
T = 2000  # number of time steps
x_true = np.zeros(T, dtype=int)
x_true[0] = np.random.choice(3)
for t in range(1, T):
    x_true[t] = np.random.choice(3, p=P_true[x_true[t-1]])

# Initialize the estimated transition matrix
P_est = np.random.rand(3, 3)
P_est /= np.sum(P_est, axis=1, keepdims=True)  # normalize rows

# Define the objective function (negative log-likelihood)
def neg_log_likelihood(P, x):
    log_likelihood = 0
    for t in range(1, len(x)):
        log_likelihood += np.log(P[x[t-1], x[t]])
    return -log_likelihood

# Perform gradient descent to minimize the negative log-likelihood
lr = 0.1  # learning rate
for i in range(1000):  # number of iterations
    # Compute the gradient of the negative log-likelihood
    grad = np.zeros((3, 3))
    for t in range(1, T):
        grad[x_true[t-1], x_true[t]] -= 1 / P_est[x_true[t-1], x_true[t]]
    # Update the estimated transition matrix
    P_est -= lr * grad
    P_est /= np.sum(P_est, axis=1, keepdims=True)  # normalize rows

# Print the true and estimated transition matrices
print("True transition matrix:")
print(P_true)
print("Estimated transition matrix:")
print(P_est)
