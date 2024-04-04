# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 16:33:14 2024

@author: kevin
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 17:50:11 2024

@author: kevin
"""

import numpy as np
from matplotlib import pyplot as plt
import itertools
from scipy.optimize import minimize
from scipy.stats import pearsonr
import matplotlib 

# %% basics
N = 3
nc = 2**N
spins = [0,1]  # binary patterns
combinations = list(itertools.product(spins, repeat=N))  # possible configurations
lt = 30000

# %%
def spk2statetime(firing, window, lt=lt, N=N, combinations=combinations):
    """
    given the firing (time and neuron that fired) data, we choose time window to slide through,
    then convert to network states and the timing of transition
    """
    states_spk = np.zeros(lt-window)
    for tt in range(lt-window):
        this_window = firing[tt:tt+window]
        word = np.zeros(N)  # template for the binary word
        for ti in range(window): # in time
            if len(this_window[ti][1])>0:
                this_neuron = this_window[ti][1][0]  # the neuron that fired first
                word[int(this_neuron)] = 1
        state_id = combinations.index(tuple(word))
        states_spk[tt] = state_id
    
    # now compute all the transitions
    trans_temp = np.diff(states_spk)  # find transitions
    spk_times = np.where(np.abs(trans_temp)>0)[0]  # spike timing
    spk_states = states_spk[spk_times].astype(int)   # spiking states
    return spk_states, spk_times

def param2M(param, N=N, combinations=combinations):
    """
    given array of parameters with length N*2**N, network size N, return transition matrix
    the matrix is a general CTMC form for 
    """
    nc = 2**N  # number of states
    
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

def compute_tauC(states, times, nc=nc, combinations=combinations, lt=None):
    """
    given the emperically measured states, measure occupency tau and the transitions C
    """
    tau = np.zeros(nc)
    C = np.zeros((nc,nc))
    # compute occupency time
    for i in range(len(states)-1):
        this_state = states[i]
        if i==0:
            tau[this_state] += times[i]  # correct for starting
        elif lt is not None and i==len(states)-1:
            tau[this_state] += lt - times[i+1]
        else:
            tau[this_state] += times[i+1]-times[i]  ### total time occupancy
        
    # compute transitions
    for t in range(len(states)-1):
        ii,jj = states[t], states[t+1]
        if ii != jj:
            if sum(x != y for x, y in zip(combinations[ii], combinations[jj])) == 1:  # ignore those not CTMC for now!
                C[ii,jj] += 1  ### counting the transtion
    return tau, C    

def compute_min_isi(firing, N=N):
    isi = np.zeros(N)
    for nn in range(N):
        # spk_i = np.array([firing[ii][0] for ii in range(lt) if len(firing[ii][0])>0 and firing[ii][1]==nn]).squeeze()
        spk_i = []
        for tt in range(lt):
            if len(firing[tt][0])>0 and len(firing[tt][0])<2 and firing[tt][1]==nn:
                spk_i.append(firing[tt][0])
            elif len(firing[tt][0])>1:  # if we happen to have synchronous spike breaking CTMC!
                spk_i.append(np.array([firing[tt][0][0]]))  # random for the first one!!??
        spk_i = np.array(spk_i).squeeze()
        isi[nn] = np.min(np.diff(spk_i))
    min_isi = np.min(isi)
    return min_isi

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

def edge_flux_inf(param, N=N, combinations=combinations):
    """
    compute edge flux with infinite data using pi_i k_ij
    """
    kij, pi = param2M(param, N, combinations)
    nc = len(pi)
    flux_ij = np.zeros((nc,nc))
    for ii in range(nc):
        for jj in range(nc):
            if ii is not jj:
                flux_ij[ii,jj] = pi[ii]*kij[ii,jj]
    return flux_ij

# %% measurements
def EP(kij):
    """
    given transition matrix, compute entropy production
    """
    pi = get_stationary(kij)
    # kij = Pij / pi[:,None]
    eps = 1e-20
    ep = 0
    n = len(pi)
    for ii in range(n):
        for jj in range(n):
            Pij = pi[ii]*kij[ii,jj]
            Pji = pi[jj]*kij[jj,ii]
            if ii is not jj:
                ep += Pij*(np.log(Pij+eps)-np.log(Pji+eps))
    return ep 

def corr_param(param_true, param_infer, mode='binary'):
    """
    Peerson's  correlation berween true and inferred parameters
    """
    if mode=='binary':
        true_temp = param_true*0 - 1
        infer_temp = param_infer*0 - 1
        true_temp[param_true>0] = 1
        infer_temp[param_infer>0] = 1
        corr = np.dot(true_temp, infer_temp)/np.linalg.norm(true_temp)/np.linalg.norm(infer_temp)
        return corr
    else:
        true_temp = param_true*1
        infer_temp = param_infer*1        
        correlation_coefficient, _ = pearsonr(true_temp, infer_temp)
        return correlation_coefficient

def sign_corr(param_true, param_infer):
    """
    given parameters for CTMC, turn it into matric, then into the effective coupling weights
    """
    w_true = M2weights(param_true)
    w_infer = M2weights(param_infer)
    corr = corr_param(w_true, w_infer, 'binary')
    return corr
    
def M2weights(param):
    M_inf, _ = param2M(param)
    f1,f2,f3 = M_inf[0,4], M_inf[0,2], M_inf[0,1]
    w12,w13,w21 = np.log(M_inf[4,6]/f2), np.log(M_inf[4,5]/f3), np.log(M_inf[2,6]/f1)
    w23,w32,w31 = np.log(M_inf[2,3]/f3), np.log(M_inf[1,3]/f2), np.log(M_inf[1,5]/f1)
    weights = np.array([w12,w13,w21,w23,w32,w31])
    return weights

# %% Maxcal functions (should write better code and import once confirmed...)
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


# %%
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

# %% function calling
# if __name__ == "__main__":
    