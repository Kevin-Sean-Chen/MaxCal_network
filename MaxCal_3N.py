# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 01:50:17 2024

@author: kevin
"""


import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
import itertools
from scipy.optimize import minimize
from scipy.linalg import expm
from scipy.stats import expon

import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

###
# Continuous time formulism with 3-neurons
# for Max-Cal in network structure
###

# %% 3-neuron with continuous time structure
f1,f2,f3 = 1,2,3  # firing
r1,r2,r3 = 3,5,7  # refractory
w12,w13,w21,w23,w31,w32 = np.random.rand(6)*3  # first order coupling, can be negative!
w123,w132,w231 = np.random.rand(3)*2  # second-oder coupling
N = 3  # number of neurons
spins = [0,1]  # binary patterns
combinations = list(itertools.product(spins, repeat=N))  # possible configurations
nc = 2**3  # number of total states of 3-neuron
param_true = (f1,f2,f3,\
              r1,r2,r3,\
              w12,w13,w21,w23,w31,w32,\
              w123,w132,w231)  # store ground-truth parameters

def param2frw(param):
    f1,f2,f3,  r1,r2,r3,  w12,w13,w21,w23,w31,w32,  w123,w132,w231 = param
    return f1,f2,f3,  r1,r2,r3,  w12,w13,w21,w23,w31,w32,  w123,w132,w231
    
# %% function for the network dynamics and observations
def ctmc_M(param):
    """
    With continuous-time asymmetric 3-neuron circuit, given parameters 15 frw parameterd
    return transition matrix and steady-state distirubtion
    """
    f1,f2,f3,  r1,r2,r3,  w12,w13,w21,w23,w31,w32,  w123,w132,w231 = param
    M = np.array([[0,    f3,   f2,   0,              f1,   0,               0,               0],
                  [r3,   0,    0,    f2*np.exp(w32), 0,    f1*np.exp(w31),  0,               0],
                  [r2,   0,    0,    f3*np.exp(w23), 0,    0,               f1*np.exp(w21),  0],
                  [0,    r2,   r3,   0,              0,    0,               0,               f1*np.exp(w231)],
                  [r1,   0,    0,    0,              0,    f3*np.exp(w13),  f2*np.exp(w12),  0],
                  [0,    r1,   0,    0,              r3,   0,               0,               f2*np.exp(w132)],
                  [0,    0,    r1,   0,              r2,   0,               0,               f3*np.exp(w123)],
                  [0,    0,    0,    r1,             0,    r2,              r3,              0]]) 
                ## 000,001,010,011,100,101,110,111   # f1*np.exp(w21)
    np.fill_diagonal(M, -np.sum(M,1))  # fill diagonal for continuous time Markov transition Q (is this correct?!)
    uu,vv = np.linalg.eig(M.T)
    zeros_eig_id = np.argmin(np.abs(uu-1))
    pi_ss = vv[:,zeros_eig_id] / np.sum(vv[:,zeros_eig_id])
    # M = pi_ss[:,None]*M  # steady-state transition probability?
    return M, np.real(pi_ss)

def P_frw_ctmc(param):
    """
    get joint probability given parameters (need to change for ctmc??)
    """
    k, pi = ctmc_M(param)  # calling asymmetric network
    Pxy = np.zeros((nc,nc))
    for ii in range(nc):
        Pxy[ii,:] = k[ii,:]*pi[ii]  # compute joint from transition k and steady-state pi (this is wrong using Q!?)
    return Pxy
    
def comp_etas(P):
    return P + P.T

def comp_Js(P):
    return P - P.T

def bursting(P, param):
    """
    Bursting observations, modified for 3-neuron continuous time
    """
    etas = comp_etas(P)
    # Js = comp_Js(P)
    _,pi = ctmc_M(param)
    piB0 = pi[0]*1
    piB3 = pi[-1]*1
    etaB01 = etas[0,1] + etas[0,2] + etas[0,4]  # zeros to one spike
    etaB12 = etas[1,3] + etas[1,5] + etas[2,3] + etas[2,6] + etas[4,5] + etas[4,6]
    etaB23 = etas[3,7] + etas[4,7] + etas[6,7]
    # eta B23, all have three terms... triangle for 1 or 2 spikes, point for 0 and 3 spikes
    return piB0, piB3, etaB01, etaB12, etaB23

def marginalization(P, param):
    """
    Marginal observations, for three-neuron in continuous time
    """
    etas = comp_etas(P)
    _,pi = ctmc_M(param)
    piX0 = pi[0] + pi[1] + pi[2] + pi[3]
    piY0 = pi[0] + pi[1] + pi[4] + pi[5]
    piZ0 = pi[0] + pi[2] + pi[4] + pi[6]
    etaX = etas[1,5] + etas[3,7] + etas[0,4] + etas[2,6]  # is this correct?
    etaY = etas[0,2] + etas[4,6] + etas[5,7] + etas[1,3]
    etaZ = etas[0,1] + etas[2,3] + etas[4,5] + etas[6,7]  
    # three planes for piXYZ, etaXYZ sum edge within plane
    return piX0, etaX, piY0, etaY, piZ0, etaZ

def edge_flux_inf(param):
    """
    compute edge flux with infinite data using pi_i k_ij
    """
    kij, pi = ctmc_M(param)
    flux_ij = np.zeros((nc,nc))
    for ii in range(nc):
        for jj in range(nc):
            if ii is not jj:
                flux_ij[ii,jj] = pi[ii]*kij[ii,jj]
    return flux_ij

def edge_flux_data(param, total_time, time_step):
    """
    edge flux measurements from data
    """
    Q,_ = ctmc_M(param)
    states,_ = sim_Q(Q, total_time, time_step)  ### fix this, don't rerun for optimization!
    flux_ij = np.zeros((nc,nc))  # counting matrix, ignoring the diagonal
    for ii in range(nc):
        for jj in range(nc):
            if ii is not jj:
                posi = np.where(states==ii)[0]
                posj = np.where(states==jj)[0]
                flux_ij[ii,jj] = len(np.intersect1d(posi-1, posj))
    flux_ij = flux_ij/total_time
    return flux_ij  ### need to confirm this calculation...

def sim_Q(Q, total_time, time_step):
    """
    Simulate Markov chain given rate matrix Q, time length and steps
    reading this: https://www.columbia.edu/~ww2040/6711F13/CTMCnotes120413.pdf
    """
    initial_state = np.random.randint(nc) # uniform to begin with
    states = [initial_state]
    times = [0.0]

    current_state = initial_state
    current_time = 0.0

    while current_time < total_time:
        rate = -Q[current_state, current_state]
        next_time = current_time + expon.rvs(scale=1/rate)
        if next_time > total_time:
            break

        transition_probabilities = expm(Q * (next_time - current_time))[current_state, :]
        transition_probabilities /= transition_probabilities.sum()  # Normalize probabilities
        next_state = np.random.choice(len(Q), p=transition_probabilities)

        states.append(next_state)
        times.append(next_time)

        current_state = next_state
        current_time = next_time

    return np.array(states), np.array(times)

# %% Max-Cal functions
def MaxCal_D(Pij, kij0, param):
    """
    KL devergence term, with transition Pij and prior rate kij0 as input
    This term can be unstable in log!
    """
    pi = get_stationary(Pij)
    # kij = Pij / pi[:,None]
    eps = 1e-10
    kl = 0
    n = len(pi)
    for ii in range(n):
        for jj in range(n):
            if ii is not jj:
                kl += Pij[ii,jj]*(np.log(Pij[ii,jj]+eps)-np.log(pi[ii]*kij0[ii,jj]+eps)) \
                      + pi[ii]*kij0[ii,jj] - Pij[ii,jj]
    return kl

def get_stationary(M):
    """
    get stationary state distribution given a transition matrix M
    """
    uu,vv = np.linalg.eig(M.T)
    zeros_eig_id = np.argmin(np.abs(uu-1))
    pix = vv[:,zeros_eig_id] / np.sum(vv[:,zeros_eig_id])
    return pix


def obs_given_frw(param, Cp_condition='all'):
    """
    concatenating different observables as constraints
    """
    Pxy = P_frw_ctmc(param)
    burst_obs = bursting(Pxy, param)
    marg_obs = marginalization(Pxy, param)
    edge_obs = edge_flux_inf(param)
    edge_obs = edge_obs.reshape(-1)
    if Cp_condition=='burst':
        obs = burst_obs*1
    elif Cp_condition=='marginal':
        obs = marg_obs*1
    elif Cp_condition=='all':
        obs = np.concatenate((burst_obs, marg_obs))
    elif Cp_condition=='test_edge':
        obs = np.concatenate((burst_obs, marg_obs, edge_obs))
        # obs = edge_obs*1
    return obs

def C_P(P, observations, param, Cp_condition='all'):
    """
    The C(P,f,r,w) function for constraints
    """
#    f,r,w = param
#    piB0,piB2,etaB01,etaB12,etaB02,JB,piX0,etaX,piY0,etaY = observations
    etas = comp_etas(P)
    Js = comp_Js(P)
    _,pi = ctmc_M(param)
    flux_ij = edge_flux_inf(param)
    c_edge = flux_ij.reshape(-1)
    
    Pxy = P_frw_ctmc(param)
    burst_obs = bursting(Pxy, param)
    marg_obs = marginalization(Pxy, param)
    burst_obs = np.array(burst_obs)
    marg_obs = np.array(marg_obs)
    
    if Cp_condition=='burst':
        c_P_frw = burst_obs*1 # burst only
    elif Cp_condition=='marginal':
        c_P_frw = marg_obs*1 # marginal only
    elif Cp_condition=='all':
        c_P_frw = np.concatenate((burst_obs, marg_obs)) # burst_marginal constraints
    elif Cp_condition=='test_edge':
        c_P_frw = np.concatenate((burst_obs, marg_obs))
        c_P_frw = np.concatenate((c_P_frw, c_edge))
        # c_P_frw = np.concatenate((marg_obs, c_edge))
        # c_P_frw = c_edge*1
    return c_P_frw - observations

def objective_param(param, kij0):
    """
    objective in the parameter space, using frw and adding extra constraints
    """
    Pyx,_ = ctmc_M(param)
    D = MaxCal_D(Pyx, kij0, param)
    return D

def eq_constraint(param, observations, Cp_condition):
    Pxy = P_frw_ctmc(param)
    cp = C_P(Pxy, observations, param, Cp_condition)
    return 0.5*np.sum(cp**2) ### not sure if this hack is legit, but use MSE for now

# %% ##########################################################################
# Notes:
    ### check on CTMC matrix
    ### should I turn back to the constraint-opimization code, rather than using g(x,y)?\
    ### if so we don't need the tilted matrix q right, is there a preference?

# %% constrained optimization
# Constraints
Cp_condition = 'test_edge'  #'burst', 'marginal', 'all'  ### choose one of the here ###
observations = obs_given_frw(param_true, Cp_condition)
constraints = ({'type': 'eq', 'fun': eq_constraint, 'args': (observations, Cp_condition)})
bounds = [(0, 100)]*len(param_true)

# Perform optimization using SLSQP method
P0 = np.ones((nc,nc))  # uniform prior
#P0 = np.random.rand(nc,nc)
# P0 = P0 / P0.sum(axis=1, keepdims=True)
np.fill_diagonal(P0, np.zeros(nc))
np.fill_diagonal(P0, np.sum(P0,1))
param0 = np.random.rand(len(param_true))*1.
result = minimize(objective_param, param0, args=(P0), method='SLSQP', constraints=constraints, bounds=bounds)
frw_inf = result.x

print('true frw:', param_true)
print('inferred frw:', frw_inf)

# %%
###############################################################################
# %% compare constraints!
conditions = ['test_edge','all','marginal']
frw_inference = np.zeros((3,len(param_true)))
for ii in range(3):
    Cp_condition = conditions[ii]
    observations = obs_given_frw(param_true, Cp_condition)
    constraints = ({'type': 'eq', 'fun': eq_constraint, 'args': (observations, Cp_condition)})
    
    # Perform optimization using SLSQP method
    P0 = np.ones((nc,nc))  # uniform prior
    np.fill_diagonal(P0, np.zeros(nc))
    np.fill_diagonal(P0, np.sum(P0,1))
    param0 = np.random.rand(len(param_true))*1
    
    result = minimize(objective_param, param0, args=(P0), method='SLSQP', constraints=constraints, bounds=bounds) 
    
    frw_inf = result.x
    frw_inference[ii,:] = frw_inf

# %%
bars = np.concatenate((np.array(param_true)[None,:], frw_inference))
num_rows, num_cols = bars.shape
bar_positions = np.arange(num_cols)
labels = ['true','edge+all', 'edge+burst', 'burst+marg']

plt.figure(figsize=(10, 6))
for i in range(num_rows):
    plt.bar(bar_positions + i * 0.2, bars[i, :], width=0.2, label=labels[i])
    
plt.legend(fontsize=15)
plt.xlabel('parameters', fontsize=20)

# %% Notes for next step
###############################################################################
# three neurons
# LIF neural model for demo
# apply to retina!
