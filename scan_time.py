# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 01:24:08 2025

@author: kevin
"""

from maxcal_functions import spk2statetime, compute_tauC, param2M, eq_constraint, objective_param, compute_min_isi,\
                                cos_ang, corr_param

import scipy.io
import numpy as np
from matplotlib import pyplot as plt
import itertools
from scipy.optimize import minimize
from scipy.stats import pearsonr
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

# np.random.seed(42)

# %%
def LIF_firing(synaptic_weights, noise_amp, timesteps=100000):
    """
    given synaptic weights and noise amplitude, turn 3-neuron spiking time series
    """
    dt = 0.1  # time step in milliseconds

    # Neuron parameters
    tau = 10.0  # membrane time constant
    v_rest = -65.0  # resting membrane potential
    v_threshold = -50.0  # spike threshold
    v_reset = -65.0  # reset potential after a spike
            
    S = synaptic_weights*1
    np.fill_diagonal(S, np.zeros(3))
    # noise_amp = 2

    # Synaptic filtering parameters
    tau_synaptic = np.array([5, 5, 5])  #5.0  # synaptic time constant

    # Initialize neuron membrane potentials and synaptic inputs
    v_neurons = np.zeros((3, timesteps))
    synaptic_inputs = np.zeros((3, timesteps))
    spikes = np.zeros((3, timesteps))   ### full array to record spikes
    spike_times = []
    firing = []
    firing.append((np.array([]), np.array([]))) # init

    # Simulation loop
    for t in range(1, timesteps):

        # Update neuron membrane potentials using leaky integrate-and-fire model
        v_neurons[:, t] = v_neurons[:, t - 1] + dt/tau*(v_rest - v_neurons[:, t - 1]) + np.random.randn(3)*noise_amp
        
        # Check for spikes
        spike_indices = np.where(v_neurons[:, t] > v_threshold)[0]
        spikes[spike_indices, t] = 1
        
        # Apply synaptic connections with synaptic filtering
        synaptic_inputs[:, t] = synaptic_inputs[:, t-1] + dt*( \
                                -synaptic_inputs[:, t-1]/tau_synaptic + np.sum(synaptic_weights[:, spike_indices], axis=1))
        
        # Update membrane potentials with synaptic inputs
        v_neurons[:, t] += synaptic_inputs[:, t]*dt
        
        # record firing
        firing.append([t+0*spike_indices, spike_indices])
        
        # reset and record spikes
        v_neurons[spike_indices, t] = v_reset  # Reset membrane potential for neurons that spiked
        if len(spike_indices) > 0:
            spike_times.append(t * dt)
    
    return firing

def firing2isi(firing, individual=False):
    isis = [[],[],[]]
    pop_isi = []
    for tt in range(len(firing)):
        temp = firing[tt]
        if len(firing[tt][0])>0:    
            spkt = temp[0][0]  # absolute time
            spki = temp[1][0]  # cell ID
            isis[spki].append(spkt)
    for nn in range(3):
        isii = np.array((isis[nn]))
        pop_isi.append(np.diff(isii))
    
    if individual:
        return pop_isi
    else:
        return np.concatenate((pop_isi))
    
# %% baiscs
N = 3
nc = 2**3
num_params = int((N*2**N)) 
spins = [0,1]  # binary patterns
combinations = list(itertools.product(spins, repeat=N))  # possible configurations
lt = 100000
reps = 3

# %% scanning loop
### scan time!
t_s = np.array([5000,10000,100000,500000])
# t_s = np.array([50,100,150,200]) ### testing window size

w_s = np.array([1,2,4,8,16])*1  ### for network strength
n_s = np.array([0,2,4,8,16])*1  ### for noist stength
d_s = np.array([1,30,60,90, 120])*1  ### for synaptic delay  (pick one neuron for delay)
h_s = np.array([1,2,4,8,16])  ### for heterogeneious-ness (pick a pair to rescale)

fault_w = np.array([20,20,10]) #20*1  ### defualt variables if not tuned
fault_n = np.array([2,2,2])
fault_d = 0
fault_h = 0

R2s = np.zeros((3, len(n_s), reps))  ### measure 3 motifs as a function of control
signs = R2s*0
coss = R2s*0
entropy = np.zeros((2,3,len(n_s)))
full_isi = []

### EI balanced
Wij_ei = np.array([[0, 1, -2],  # Neuron 0 connections
                [1, 0, -2],  # Neuron 1 connections
                [1, 1, 0]])*2
### common-input
Wij_common = np.array([[0, 0, 0],  # Neuron 0 connections
                [1, 0, 0],  # Neuron 1 connections
                [1, 0, 0]])*5
## convernged with different strength
Wij_converge = np.array([[0, 0, 0],  # Neuron 0 connections
                [0, 0, 0],  # Neuron 1 connections
                [1, 1, 0]])*1.

Ws = (Wij_ei, Wij_common, Wij_converge)  # list of motif types

P0 = np.ones((nc,nc))  # uniform prior
np.fill_diagonal(P0, np.zeros(nc))
np.fill_diagonal(P0, -np.sum(P0,1))
dofs_all = nc**2 + nc
Cp_condition = np.ones(dofs_all)


for rr in range(reps):
    print('repeat: ', rr)
    for ww in range(0,3): #(0,3):
        print('circuit: ', ww)
        Wij = Ws[ww]  ### asign motif
        isi_per_motif = []
        for tt in range(len(t_s)):
            lt = t_s[tt]
            S = Wij*fault_w[ww] #
            firing = LIF_firing(S, fault_n[ww], timesteps=lt)  ### tune noise, delay, or ratio
            minisi = compute_min_isi(firing,lt=lt)
            adapt_window = 150 #int(minisi*10)  #t_s[tt]
            spk_states, spk_times = spk2statetime(firing, adapt_window, lt=lt)  # embedding states
            tau,C = compute_tauC(spk_states, spk_times)  # emperical measurements
            ### not scanning for now...
            rank_tau = np.argsort(tau)[::-1]  # ranking occupency
            rank_C = np.argsort(C.reshape(-1))[::-1]  # ranking transition
            tau_, C_ = (tau + 1)/lt, (C + 1)/lt # correct normalization  #######################################
            observations = np.concatenate((tau_, C_.reshape(-1))) 
            
            # # computed and record the corresponding KL
            M_inf = (C_/tau_[:,None]) #+ 1/lt  # to prevent exploding
            # M_inf, pi_inf = param2M(param_temp)
            f1,f2,f3 = M_inf[0,4], M_inf[0,2], M_inf[0,1]
            w12,w13,w21 = np.log(M_inf[4,6]/f2), np.log(M_inf[4,5]/f3), np.log(M_inf[2,6]/f1)
            w23,w32,w31 = np.log(M_inf[2,3]/f3), np.log(M_inf[1,3]/f2), np.log(M_inf[1,5]/f1)
            inf_w = np.array([w12,w13,w21,w23,w32,w31])
            true_s = np.array([S[1,0],S[2,0],S[0,1],S[2,1],S[1,2],S[0,2]])
    
            correlation_coefficient, _ = pearsonr(inf_w, true_s)
            R2s[ww, tt, rr] = correlation_coefficient
            signs[ww, tt, rr] = corr_param(true_s, inf_w, 'binary')
            coss[ww, tt, rr] = cos_ang(inf_w, true_s)
            print(cos_ang(inf_w, true_s))
            
        
# %% plotting
plt.figure()

for ii in range(0,3):
    plt.subplot(131); plt.errorbar(t_s, np.mean(R2s[ii,:-1,:],1), np.std(R2s[ii,:-1,:],1)); plt.title('R2'); plt.xscale('log');
    plt.subplot(132); plt.errorbar(t_s, np.mean(signs[ii,:-1,:],1), np.std(signs[ii,:-1,:],1)); plt.title('sign'); plt.xscale('log');
    plt.xlabel('time steps',fontsize=20)
    plt.subplot(133); plt.errorbar(t_s, np.mean(coss[ii,:-1,:],1), np.std(coss[ii,:-1,:],1)); plt.title('cos'); plt.xscale('log');

# %%
# scan_x = h_s*1
scan_x = np.array([1, 2, 4, 8, 16])
plt.figure()
plt.subplot(131); plt.semilogx(scan_x, R2s.T,'-o', label=['EI', 'comm', 'conv']); plt.title('R2'); plt.legend()
plt.subplot(132); plt.semilogx(scan_x, signs.T,'-o', label=['EI', 'comm', 'conv']); plt.title('signed corr');plt.xlabel('repeats...')
plt.subplot(133); plt.semilogx(scan_x, coss.T,'-o', label=['EI', 'comm', 'conv']); plt.title('cosine')

#### TO-DO #####
# tune untill the initial point is close, by tuning weights and maybe noise