# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 23:57:30 2024

@author: kevin
"""

import numpy as np
from matplotlib import pyplot as plt
import itertools
from scipy.optimize import minimize
from scipy.stats import pearsonr

import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

# %% algotrithm to infer the DOF of transition matrix via MaxCal
# generate spike trains
# the setup is the general N*2**N dof matrix
# rank the order of state sylables
# scanning through and appending them to make DOF-KL curve for finite data
# compare KL, correlation, and EP for analytic and finite-data inferences

# %% network setting
N = 3  # number of neurons
num_params = int((N*2**N))  # number of parameters in model without refractory assumption
spins = [0,1]  # binary patterns
combinations = list(itertools.product(spins, repeat=N))  # possible configurations
nc = 2**N  # number of total states of 3-neuron
param_true = np.random.rand(num_params)*1  # use rand for now...

def param2M(param, N=N):
    """
    given array of parameters with length N*2**N, network size N, return transition matrix
    the matrix is a general CTMC form for 
    """
    nc = 2**N  # number of states
    spins = [0,1]  # binary patterns
    combinations = list(itertools.product(spins, repeat=N))  # possible configurations
    
    ### idea: M = mask*FR, with mask for ctmc, FR is the rest of the transitions
    mask = np.ones((nc,nc))  # initialize the tilted matrix
    FR = mask*1
    # make the mask
    for ii in range(nc):
        for jj in range(nc):
            # Only allow one flip logic in ctmc
            if sum(x != y for x, y in zip(combinations[ii], combinations[jj])) != 1:
                mask[ii,jj] = 0
    
    # now make F matrix!
    kk = 0
    for ii in range(nc):
        for jj in range(nc):
            if mask[ii,jj]==1: #only check those that generates one spike
                FR[ii,jj] = param[kk]
                kk = kk+1  # marching forward to fill in f*exp(wij) parts... need to later invert this!!
    # print(kk)
    M = mask*FR  

    ### compute steady-state
    np.fill_diagonal(M, -np.sum(M,1))  # fill diagonal for continuous time Markov transition Q (is this correct?!)
    uu,vv = np.linalg.eig(M.T)
    zeros_eig_id = np.argmin(np.abs(uu-1))
    pi_ss = vv[:,zeros_eig_id] / np.sum(vv[:,zeros_eig_id])
    
    return M, np.real(pi_ss)

# %% network simulation
def sim_Q(Q, total_time, time_step):
    """
    Simulate Markov chain given rate matrix Q, time length and steps
    reading this: https://www.columbia.edu/~ww2040/6711F13/CTMCnotes120413.pdf
    """
    nc = Q.shape[0]
    initial_state = np.random.randint(nc) # uniform to begin with
    states = [initial_state]
    times = [0.0]

    current_state = initial_state
    current_time = 0.0

    while current_time < total_time:
        rate = -Q[current_state, current_state]
        next_time = current_time + np.random.exponential(scale=1/rate)
        if next_time > total_time:
            break

        transition_probabilities = Q[current_state,:]*1#expm(Q * (next_time - current_time))[current_state, :]
        transition_probabilities[current_state] = 0  # remove diagonal
        transition_probabilities /= transition_probabilities.sum()  # Normalize probabilities
        next_state = np.random.choice(len(Q), p=transition_probabilities)
        # print(transition_probabilities)
        
        #####
        # # Generate exponentially distributed time until the next event 
        # rate = abs(Q[current_state, current_state]) 
        # time_to_next_event = np.random.exponential(scale=1/rate) # Update the time and state 
        # time_points.append(time_points[-1] + time_to_next_event) # Determine the next state based on transition probabilities 
        # transition_probs = Q[current_state, :] / rate transition_probs[transition_probs < 0] = 0  # Ensure non-negative probabilities 
        # transition_probs /= np.sum(transition_probs)  # Normalize probabilities to sum to 1 
        # next_state = np.random.choice(len(Q), p=transition_probs) 
        # state_sequence.append(next_state)
        #####
        
        states.append(next_state)
        times.append(next_time)

        current_state = next_state
        current_time = next_time

    return np.array(states), np.array(times)

total_time = 500
time_step = 1  # check with Peter if this is ok... THIS is OK
M,pi_ss = param2M(param_true)
states, times = sim_Q(M, total_time, time_step)

# %%
################################
# times = spk_times*1 # = np.where(np.abs(trans_temp)>0)[0]  # spike timing
# states = spk_states*1 # = states_spk[spk_times] 
################################
# %% counting and ranking analysis
def compute_tauC(states, times, nc=nc):
    """
    given the emperically measured states, measure occupency tau and the transitions C
    """
    tau = np.zeros(nc)
    C = np.zeros((nc,nc))
    # compute occupency time
    for i in range(len(states)-1):
        this_state = states[i]
        tau[this_state] += times[i+1]-times[i]  ### total time occupancy
        
    # compute transitions
    for t in range(len(states)-1):
        ii,jj = states[t], states[t+1]
        if ii != jj:
            C[ii,jj] += 1  ### counting the transtion
    return tau, C    

#### for simulation
tau_finite, C_finite = compute_tauC(states, times)  # emperical measurements

# %% computing statistics for infinite data given parameter
def P_frw_ctmc(param):
    """
    get joint probability given parameters (need to change for ctmc??)
    """
    k, pi = param2M(param)  # calling asymmetric network
    nc = len(pi)
    Pxy = np.zeros((nc,nc))
    for ii in range(nc):
        Pxy[ii,:] = k[ii,:]*pi[ii]  # compute joint from transition k and steady-state pi (this is wrong using Q!?)
    return Pxy

def edge_flux_inf(param):
    """
    compute edge flux with infinite data using pi_i k_ij
    """
    kij, pi = param2M(param)
    nc = len(pi)
    flux_ij = np.zeros((nc,nc))
    for ii in range(nc):
        for jj in range(nc):
            if ii is not jj:
                flux_ij[ii,jj] = pi[ii]*kij[ii,jj]
    return flux_ij

def EP(kij):
    """
    given transition matrix, compute entropy production
    """
    pi = get_stationary(kij)
    # kij = Pij / pi[:,None]
    eps = 1e-11
    ep = 0
    n = len(pi)
    for ii in range(n):
        for jj in range(n):
            Pij = pi[ii]*kij[ii,jj]
            Pji = pi[jj]*kij[jj,ii]
            if ii is not jj:
                ep += Pij*(np.log(Pij+eps)-np.log(Pji+eps))
    return ep 

def corr_param(param_true, param_infer):
    """
    Peerson's  correlation berween true and inferred parameters
    """
    correlation_coefficient, _ = pearsonr(param_true, param_infer)
    return correlation_coefficient

# %% for infinite data
###############################################################################
#### for ground-truth (infinite data!)
tau_infinite = pi_ss*total_time
flux_ij_true = edge_flux_inf(param_true)
C_infinite = flux_ij_true* total_time

# %% Maxcal functions (should write better code and import once confirmed...)
def MaxCal_D(kij, kij0, param):
    """
    KL devergence term, with transition Pij and prior rate kij0 as input
    This term can be unstable in log!
    """
    pi = get_stationary(kij)
    # kij = Pij / pi[:,None]
    eps = 1e-20
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


def C_P(P, observations, param, Cp_condition):
    """
    The C(P,f,r,w) function for constraints
    """
    _,pi = param2M(param)
    flux_ij = edge_flux_inf(param)
    # trick to make all observable a vector and shield with Cp conditions!
    obs_all = np.concatenate((pi, flux_ij.reshape(-1)))
    # obs_all = np.concatenate((flux_ij.reshape(-1), pi))
    cp_dof = obs_all * Cp_condition
    return cp_dof - (observations * Cp_condition)

def objective_param(param, kij0):
    """
    objective in the parameter space, using frw and adding extra constraints
    """
    kij,_ = param2M(param)
    D = MaxCal_D(kij, kij0, param)
    return D

def eq_constraint(param, observations, Cp_condition):
    Pxy = P_frw_ctmc(param)
    cp = C_P(Pxy, observations, param, Cp_condition)
    return 0.5*np.sum(cp**2) ### not sure if this hack is legit, but use MSE for now

# %% building constraints
dofs = num_params*1
dofs_all = nc**2 + nc
target_dof = dofs + nc

### recordings
kls_inf = np.zeros(target_dof) # for infinite data
kls_fin = np.zeros(target_dof)  # for finite data
r2_inf = np.zeros(target_dof)
r2_fin = np.zeros(target_dof)
ep_inf = np.zeros(target_dof)
ep_fin = np.zeros(target_dof)

### sort according to inifinite data
Cp_condition = np.zeros(dofs_all)
rank_tau = np.argsort(tau_infinite)[::-1]  # ranking occupency
rank_C = np.argsort(C_infinite.reshape(-1))[::-1]  # ranking transition
tau_i, C_i = tau_infinite/total_time, C_infinite/total_time # correct normalization
observations_inf = np.concatenate((tau_i, C_i.reshape(-1)))  # observation from simulation!
tau_f, C_f = tau_finite/total_time, C_finite/total_time # correct normalization
observations_fin = np.concatenate((tau_f, C_f.reshape(-1))) 

P0 = np.ones((nc,nc))  # uniform prior
np.fill_diagonal(P0, np.zeros(nc))
np.fill_diagonal(P0, -np.sum(P0,1))
ii = 0
### scan through all dof
while ii < target_dof:
    ### work on tau first
    if ii<nc-1:
        Cp_condition[rank_tau[ii]] = 1
    else:
        Cp_condition[rank_C[ii-(nc-1)]+(nc)] = 1
          
    ### run max-cal!
    constraints_inf = ({'type': 'eq', 'fun': eq_constraint, 'args': (observations_inf, Cp_condition)})
    constraints_fin = ({'type': 'eq', 'fun': eq_constraint, 'args': (observations_fin, Cp_condition)})
    bounds = [(.0, 100)]*len(param_true)

    # Perform optimization using SLSQP method
    param0 = np.ones(num_params)*.1 + np.random.rand(num_params)*0 + param_true*0
    result_inf = minimize(objective_param, param0, args=(P0), method='SLSQP', constraints=constraints_inf, bounds=bounds)
    result_fin = minimize(objective_param, param0, args=(P0), method='SLSQP', constraints=constraints_fin, bounds=bounds)
    
    # computed and record the corresponding KL
    param_inf = result_inf.x
    param_fin = result_fin.x
    kij_inf,_ = param2M(param_inf)
    kij_fin,_ = param2M(param_fin)
    kls_inf[ii] = MaxCal_D(kij_inf, P0, param_inf)
    kls_fin[ii] = MaxCal_D(kij_fin, P0, param_fin)
    ep_inf[ii] = EP(kij_inf)
    ep_fin[ii] = EP(kij_fin)
    r2_inf[ii] = corr_param(param_true, param_inf)
    r2_fin[ii] = corr_param(param_true, param_fin)
    
    print(ii)    
    ii = ii+1

# %% compare 
M_inf, pi_inf = param2M(param_inf)
flux_ij = edge_flux_inf(param_inf)
plt.figure()
plt.subplot(211)
plt.plot(C_infinite.reshape(-1), flux_ij.reshape(-1),'.')

plt.subplot(212)
plt.plot(tau_infinite, pi_inf, '.')

# print(np.max(C_-M_inf))

# %%
# plt.figure()
# plt.plot(np.log(kls[:]),'-o')
# plt.xlabel('ranked dof', fontsize=20)
# plt.ylabel('KL', fontsize=20)

# %% compare finit and inifinite data KL
plt.figure()
plt.plot(kls_inf,'-o', label='analytic')
plt.plot(kls_fin,'-o', label='finite-data')
plt.legend(fontsize=20); plt.ylabel('KL', fontsize=20)

plt.figure()
plt.plot(r2_inf,'-o', label='analytic')
plt.plot(r2_fin,'-o', label='finite-data')
plt.legend(fontsize=20); plt.ylabel('corr', fontsize=20)

plt.figure()
plt.plot(ep_inf,'-o', label='analytic')
plt.plot(ep_fin,'-o', label='finite-data')
plt.legend(fontsize=20); plt.ylabel('EP', fontsize=20)

# %%
# three traces scaling with DOF (infinite vs finite data)
# - KL (maybe keep analytic sorting order)
# - peerson correlation (of the elements)
# - EP (may fluctuate later with C) (pij comes within Maxcal!, pair not transition, k!!!)

# %%
# simulate spikes now!!!
# check motifs and rare events
# in parallel download retina data
