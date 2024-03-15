# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 21:56:33 2024

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
# Continuous time formulism that generalizes to N neurons
# for Max-Cal in network structure
# not directly working in the f,r,w parameter space but using the transition matrix
###

# %% Note
# generate full discrete time matrix
# all elements contain f*exp(w) and r
# zero-out pairs more than one bit flip

# if works!! (hopefully)
# sim 5 spiking neurons <- 3-4 might be enough! figure out input and strength
# get 5 neurons retinal data
# ... show effective coupling and effective field

# %% alternative/better solution
# directly work with f*exp(w)
# solve f, then w is known!
# Cp: start with edge-flux and pi00, pi11

# %%
# given N neurons
# 2^N x 2^N
# number of parameters is the edge count for a hypercube
# N*2^N edges
# N*2^N/2+N parameters!

# %% N-neuron with continuous time structure
N = 4  # number of neurons
num_params = int((N*2**N)/2+N)  # number of parameters in this model
spins = [0,1]  # binary patterns
combinations = list(itertools.product(spins, repeat=N))  # possible configurations
nc = 2**N  # number of total states of 3-neuron
# fws = np.random.rand(N*2**N-N)  # effective parameter-- firing f and coupling w
# rs = np.random.rand(N)  # refractoriness
# param_true = (rs, fws)  # store true parameters in a tuple for passing functions
param_true = np.random.rand(num_params)*10  # use rand for now...
# param_true = np.round(np.random.rand(num_params)*1, 3)

# %% new generalized tilted function given parameters for size-N network
def general_M(param, N=N):
    """
    a tilted matrix for ctmc neural circuit generalized to arbitrary size
    the parameter contains fields of parameters: r and fw (f*exp(w) is effectively combined) 
    which informs the size and couplings for transition matrix
    the output is the state x state titled matrix and its steady-state
    """
    # rs, fws = param  # find out a way to load these sublists from a param tuple
    rs = param[:N]
    fws = param[N:]
    N = len(rs)  # number of neurons
    nc = 2**N  # number of states
    spins = [0,1]  # binary patterns
    combinations = list(itertools.product(spins, repeat=N))  # possible configurations
    
    ### idea: M = mask*R*F, with mask for ctmc, R for refractory, and F for firing
    mask = np.ones((nc,nc))  # initialize the tilted matrix
    R = mask*1
    F = mask*1
    mask_r = mask*1
    # make the mask
    for ii in range(nc):
        for jj in range(nc):
            # Only allow one flip logic in ctmc
            if sum(x != y for x, y in zip(combinations[ii], combinations[jj])) != 1:
                mask[ii,jj] = 0
    
    # make R matrix
    for ii in range(nc):
        for jj in range(nc):
            # Find the indices where the differences occur
            indices = [index for index, (x, y) in enumerate(zip(combinations[ii], combinations[jj])) if x != y]
            
            # Check if there is exactly one difference, it goes from 1 to 0, and store the indices
            if len(indices) == 1 and combinations[ii][indices[0]] == 1 and combinations[jj][indices[0]] == 0:
                R[ii,jj] = rs[indices[0]]  # use the corresponding refractoriness
                mask_r[ii,jj] = 0
    
    # now make F matrix!
    kk = 0
    for ii in range(nc):
        for jj in range(nc):
            if mask[ii,jj]==1 and mask_r[ii,jj]==1: #only check those that generates one spike
                F[ii,jj] = fws[kk]
                kk = kk+1  # marching forward to fill in f*exp(wij) parts... need to later invert this!!
    # print(kk)
    M = mask*R*F  

    ### compute steady-state
    np.fill_diagonal(M, -np.sum(M,1))  # fill diagonal for continuous time Markov transition Q (is this correct?!)
    uu,vv = np.linalg.eig(M.T)
    zeros_eig_id = np.argmin(np.abs(uu-1))
    pi_ss = vv[:,zeros_eig_id] / np.sum(vv[:,zeros_eig_id])
    
    return M, np.real(pi_ss)
    
# %% function for the network dynamics and observations
def P_frw_ctmc(param):
    """
    get joint probability given parameters (need to change for ctmc??)
    """
    k, pi = general_M(param)  # calling asymmetric network
    nc = len(pi)
    Pxy = np.zeros((nc,nc))
    for ii in range(nc):
        Pxy[ii,:] = k[ii,:]*pi[ii]  # compute joint from transition k and steady-state pi (this is wrong using Q!?)
    return Pxy
    
def comp_etas(P):
    return P + P.T

def comp_Js(P):
    return P - P.T

##### dropping these two for now...
### def bursting(P, param):

### def marginalization(P, param):
#####

def edge_flux_inf(param):
    """
    compute edge flux with infinite data using pi_i k_ij
    """
    kij, pi = general_M(param)
    nc = len(pi)
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
    Q,_ = general_M(param)
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
def MaxCal_D(kij, kij0, param):
    """
    KL devergence term, with transition Pij and prior rate kij0 as input
    This term can be unstable in log!
    """
    pi = get_stationary(kij)
    # kij = Pij / pi[:,None]
    eps = 1e-11
    kl = 0
    n = len(pi)
    for ii in range(n):
        for jj in range(n):
            Pij = pi[ii]*kij[ii,jj]
            if ii is not jj:
                kl += Pij*(np.log(Pij+eps)-np.log(pi[ii]*kij0[ii,jj]+eps)) \
                      + pi[ii]*kij0[ii,jj] - Pij
    return kl

def get_stationary(M):
    """
    get stationary state distribution given a transition matrix M
    """
    uu,vv = np.linalg.eig(M.T)
    zeros_eig_id = np.argmin(np.abs(uu-1))
    pix = vv[:,zeros_eig_id] / np.sum(vv[:,zeros_eig_id])
    return np.real(pix)

def obs_given_frw(param, Cp_condition='Peter'):
    """
    concatenating different observables as constraints
    """
    # Pxy = P_frw_ctmc(param)
    # burst_obs = bursting(Pxy, param)
    # marg_obs = marginalization(Pxy, param)
    edge_obs = edge_flux_inf(param)
    edge_obs = edge_obs.reshape(-1)
    _,pi = general_M(param)
    if Cp_condition=='Peter':
        obs = np.append(edge_obs, pi[0])  # test with edge+pi000
        # obs = np.append(obs, pi[-1])
        # obs = np.concatenate((edge_obs, pi))
    return obs

def C_P(P, observations, param, Cp_condition='Peter'):
    """
    The C(P,f,r,w) function for constraints
    """
    etas = comp_etas(P)
    Js = comp_Js(P)
    _,pi = general_M(param)
    flux_ij = edge_flux_inf(param)
    c_edge = flux_ij.reshape(-1)
    
    # Pxy = P_frw_ctmc(param)
    # burst_obs = bursting(Pxy, param)
    # marg_obs = marginalization(Pxy, param)
    # burst_obs = np.array(burst_obs)
    # marg_obs = np.array(marg_obs)
    
    if Cp_condition=='Peter':
        c_P_frw = np.append(c_edge, pi[0])
        # c_P_frw = np.append(c_P_frw, pi[-1])
        # c_P_frw = np.concatenate((c_edge, pi))
    return c_P_frw - observations

def objective_param(param, kij0):
    """
    objective in the parameter space, using frw and adding extra constraints
    """
    kij,_ = general_M(param)
    D = MaxCal_D(kij, kij0, param)
    return D

def eq_constraint(param, observations, Cp_condition):
    Pxy = P_frw_ctmc(param)
    cp = C_P(Pxy, observations, param, Cp_condition)
    return 0.5*np.sum(cp**2) ### not sure if this hack is legit, but use MSE for now

# %% constrained optimization
# Constraints
Cp_condition = 'Peter'  #'burst', 'marginal', 'all'  ### choose one of the here ###
observations = obs_given_frw(param_true, Cp_condition)
constraints = ({'type': 'eq', 'fun': eq_constraint, 'args': (observations, Cp_condition)})
bounds = [(.0, 100)]*len(param_true)

# Perform optimization using SLSQP method
P0 = np.ones((nc,nc))  # uniform prior
np.fill_diagonal(P0, np.zeros(nc))
np.fill_diagonal(P0, -np.sum(P0,1))
# param0 = (np.random.rand(N),  np.random.rand(N*2*N-N))
param0 = np.random.rand(num_params)*1 + param_true*0
# param0 = frw_inf*1
# result = minimize(objective_param, param0, args=(P0), method='SLSQP', constraints=constraints, bounds=bounds)

tolerance = 1e-10
result = minimize(objective_param, param0, args=(P0), method='SLSQP', constraints=constraints, bounds=bounds, tol=tolerance)

frw_inf = result.x

# print('true frw:', param_true)
# print('inferred frw:', frw_inf)

# %%
plt.figure()
plt.plot(param_true, frw_inf, 'o')
plt.xlabel('true param', fontsize=20)
plt.ylabel('inferred param', fontsize=20)
plt.title('4 neuron')

# %% debugging
M_,pi_ss_ = general_M(frw_inf)
M,pi_ss = general_M(param_true)

plt.figure(figsize=(10, 8))
plt.subplot(121)
plt.imshow(np.abs(M-M_)/M)
plt.colorbar(orientation='horizontal', pad=0.1)
plt.title('|M_true-M_inferred|/M',fontsize=20)
plt.subplot(122)
plt.plot(pi_ss,label='true')
plt.plot(pi_ss_,'--', label='inferred')
plt.legend(fontsize=20)
plt.title('P_steady',fontsize=20)

# %% plot flux
flux_true = edge_flux_inf(param_true)
flux_infer = edge_flux_inf(frw_inf)
plt.figure()
plt.plot(flux_true.reshape(-1), flux_infer.reshape(-1), '.')
plt.title('edge flux',fontsize=20)

# %%
print(param_true[:N])
print(frw_inf[:N])

# %%
###############################################################################
# # %% compare constraints!
# conditions = ['test_edge','Peter','all']
# frw_inference = np.zeros((3,len(param_true)))
# for ii in range(3):
#     Cp_condition = conditions[ii]
#     observations = obs_given_frw(param_true, Cp_condition)
#     constraints = ({'type': 'eq', 'fun': eq_constraint, 'args': (observations, Cp_condition)})
    
#     # Perform optimization using SLSQP method
#     P0 = np.ones((nc,nc))  # uniform prior
#     np.fill_diagonal(P0, np.zeros(nc))
#     np.fill_diagonal(P0, np.sum(P0,1))
#     param0 = np.random.rand(len(param_true))*1
    
#     result = minimize(objective_param, param0, args=(P0), method='SLSQP', constraints=constraints, bounds=bounds) 
    
#     frw_inf = result.x
#     frw_inference[ii,:] = frw_inf

# # %%
# bars = np.concatenate((np.array(param_true)[None,:], frw_inference))
# num_rows, num_cols = bars.shape
# bar_positions = np.arange(num_cols)
# labels = ['true','edge+all', 'edge+pi000', 'edge']

# plt.figure(figsize=(10, 6))
# for i in range(num_rows):
#     plt.bar(bar_positions + i * 0.2, (bars[i, :]), width=0.2, label=labels[i])
    
# plt.legend(fontsize=15)
# plt.xlabel('parameters', fontsize=20)
