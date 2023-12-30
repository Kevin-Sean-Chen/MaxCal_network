# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 03:31:18 2023

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
# Continuous time formulism for Max-Cal in networks
###

# %% 2-neuron with continuous time structure
f1,r1,w12 = 0.5,0.9,0.3  # can be larger than one in ctmc!
f2,r2,w21 = 0.2,0.7,0.5
N = 2  # number of neurons
spins = [0,1]  # binary patterns
combinations = list(itertools.product(spins, repeat=N))  # possible configurations
nc = 2**2  # number of total states of 2-neuron
param_true = (f1,r1,w12,\
              f2,r2,w21)  # store ground-truth parameters
    
# %% function for the network dynamics and observations
def ctmc_M(param):
    """
    With continuous-time asymmetric 2-neuron circuit, given parameters f12,r12,w12,21
    return transition matrix and steady-state distirubtion
    """
    f1,r1,w12,  f2,r2,w21 = param
    M = np.array([[0,    f2,   f1,    0],
                  [r2,   0,    0,     w21],
                  [r1,   0,    0,     w12],
                  [0,    r1,   r1,    0]]) ## 00,01,10,11
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
    Bursting observations
    """
    etas = comp_etas(P)
    Js = comp_Js(P)
    _,pi = ctmc_M(param)
    piB0 = pi[0]*1
    piB2 = pi[3]*1
    etaB01 = etas[0,1] + etas[0,2]
    etaB12 = etas[1,3] + etas[2,3]
    etaB02 = etas[0,3]*1   # I assume this does not exist in continuous time?
    JB = Js[0,3]*1
    return piB0, piB2, etaB01, etaB12, etaB02, JB

def marginalization(P, param):
    """
    Marginal observations
    """
    etas = comp_etas(P)
    _,pi = ctmc_M(param)
    piX0 = pi[0] + pi[1]
    etaX = etas[1,3] + etas[1,2] + etas[0,3] + etas[0,2]  # this should be adjusted too...
    piY0 = pi[0] + pi[2]
    etaY = etas[2,3] + etas[0,1] + etas[1,2] + etas[0,3]
    return piX0, etaX, piY0, etaY

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
    states,_ = sim_Q(Q, total_time, time_step)
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

def tilted_q(beta, kij0, constraint_list, combinations, obs_list):
    """
    tilted q matrix for continuous time
    input multiplier beta, prior transition kij0, list of constraints, combination of states, and observation list
    output the tilted matrix q, its eigen-vector and value
    """
    n = kij0.shape[0]
    num_const = len(beta)
    Gxy = np.zeros((n, n, num_const))
    q = np.zeros((n,n))
    for ii in range(n):
        for jj in range(n):
            for cc in range(num_const):
                Gxy[ii,jj,cc] = constraint_list[cc](combinations[ii] , combinations[jj], obs_list[cc])
            if ii is not jj:
                q[ii,jj] = kij0[ii,jj]*np.exp(Gxy[ii,jj,:]@beta)
            else:
                q[ii,jj] = Gxy[ii,jj,:]@beta - np.sum(kij0[ii,:]) + kij0[ii,jj]
    
    ### compute eigen-value and vector
    uu, vr = np.linalg.eig(q)  # right vector
    u2, vl = np.linalg.eig(q.T)  # left vectors
    lamb,rp,lp = np.max(np.real(uu)), np.argmax(np.real(uu)), np.argmax(np.real(u2))  # max real eigen value
    vrx, vlx = np.real(vr[:,rp]), np.real(vl[:,lp])
    vec_dot_norm = np.sqrt(np.abs(np.dot(vrx, vlx)))  # normalization factor for unit dot product
    vrx, vlx = np.abs(vrx)/vec_dot_norm, np.abs(vlx)/vec_dot_norm  # normalize eigen vectors
    return q, np.real(vrx), vlx, np.real(lamb)

def posterior_k(beta, kij0, constraint_list, combinations, obs_list, vrx, lamb):
    """
    Parameters
    ----------
    beta : vector
        multipliers we aim to optimize for.
    kij0 : matrix (N x N)
        prior state transition rate.
    vrx : vector (N)
        right eigen-vector of the tilted matrix.
    lamb : scalar
        eigen-value of the tilted matrix.
    g : tensor (N,N,obs)
        pair-wise observables off-diagonal and unit-wise along the diagonal.

    Returns
    -------
    k : matrix (N x N)
        the posterioir transition matrix.
    """
    n = len(vrx)
    k = np.zeros((n,n))
    num_const = len(beta)
    Gxy = np.zeros((n, n, num_const))
    for ii in range(n):
        for jj in range(n):
            for cc in range(num_const):
                ### gathering the list of patterns and observables, and feed to the function list
                Gxy[ii,jj,cc] = constraint_list[cc](combinations[ii] , combinations[jj], obs_list[cc])
                ### fill in the matrix
            if ii is not jj:
                k[ii,jj] = kij0[ii,jj]*np.exp(Gxy[ii,jj,:]@beta)*vrx[ii]/vrx[jj]
            else:
                k[ii,jj] = Gxy[ii,jj,:]@beta - np.sum(kij0[ii,:]) + kij0[ii,jj] - lamb
    return k

def gij_burst(x,y,a=0):
    """
    construct observable g(i,j) with pairs
    """
    f_ij, g_ij_kl = 0,0
    if (x==(0,0)):
        f_ij = 1
    if (x==(0,0)) and (y==(1,0)):
        g_ij_kl = -1
    gxy = f_ij + g_ij_kl - a
    return gxy

def gij_marginal(test):
    g = 0
    return g

def gij_edge(test):
    g = 0
    return g

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
    c1 = pi[0]
    c2 = pi[3]
    c3 = etas[0,1] + etas[0,2]
    c4 = etas[1,3] + etas[2,3]
    c5 = etas[0,3]
    c6 = Js[0,3]
    c7 = pi[0] + pi[1]
    c8 = etas[1,3] + etas[1,2] + etas[0,3] + etas[0,2]
    c9 = pi[0] + pi[2]
    c10 = etas[2,3] + etas[0,1] + etas[1,2] + etas[0,3]
    if Cp_condition=='burst':
        c_P_frw = np.array([c1,c2,c3,c4,c5,c6]) # burst only
    elif Cp_condition=='marginal':
        c_P_frw = np.array([c7,c8,c9,c10]) # marginal only
    elif Cp_condition=='all':
        c_P_frw = np.array([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10])  # all constraints
    elif Cp_condition=='test_edge':
        c_P_frw = np.array([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10])
        c_P_frw = np.concatenate((c_P_frw, c_edge))
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

def objective_GartnerEllis(beta, g_bar, kij0):
    """
    a function of beta and g_bar, here we optimize for beta*, g_bar needs to be specify
    """
    _,_,_, lamb = tilted_q(beta, kij0, g_bar)
    obj = np.dot(beta, g_bar) - np.log(lamb)
    return -obj # do scipy.minimization on thi

# %% ##########################################################################
# Notes:
    ### check on CTMC matrix
    ### should I turn back to the constraint-opimization code, rather than using g(x,y)?\
    ### if so we don't need the tilted matrix q right, is there a preference?
# %% test optimization (place holder for now)
pi_marg = 0
edg_prob = 0
constraint_list = [gij_burst, gij_marginal, gij_edge]
obs_list = [0, pi_marg, edg_prob]

# %% constrained optimization
# Constraints
Cp_condition = 'test_edge'  #'burst', 'marginal', 'all'  ### choose one of the here ###
observations = obs_given_frw(param_true, Cp_condition)
constraints = ({'type': 'eq', 'fun': eq_constraint, 'args': (observations, Cp_condition)})
bounds = [(0, 100)]*6

# Perform optimization using SLSQP method
P0 = np.ones((nc,nc))  # uniform prior
#P0 = np.random.rand(nc,nc)
# P0 = P0 / P0.sum(axis=1, keepdims=True)
np.fill_diagonal(P0, np.zeros(nc))
np.fill_diagonal(P0, np.sum(P0,1))
param0 = np.random.rand(6)*1.
result = minimize(objective_param, param0, args=(P0), method='SLSQP', constraints=constraints, bounds=bounds)
frw_inf = result.x

print('true frw:', param_true)
print('inferred frw:', frw_inf)

