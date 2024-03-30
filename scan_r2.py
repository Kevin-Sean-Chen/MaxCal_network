# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 16:04:55 2024

@author: kevin
"""

from maxcal_functions import spk2statetime, compute_tauC, param2M, eq_constraint, objective_param, compute_min_isi

import scipy.io
import numpy as np
from matplotlib import pyplot as plt
import itertools
from scipy.optimize import minimize
from scipy.stats import pearsonr
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

# %%
def LIF_firing(synaptic_weights, noise_amp):
    """
    given synaptic weights and noise amplitude, turn 3-neuron spiking time series
    """
    dt = 0.1  # time step in milliseconds
    timesteps = 30000  # total simulation steps

    # Neuron parameters
    tau = 10.0  # membrane time constant
    v_rest = -65.0  # resting membrane potential
    v_threshold = -50.0  # spike threshold
    v_reset = -65.0  # reset potential after a spike

    # Synaptic weight matrix
    # synaptic_weights = np.array([[0, 1, -2],  # Neuron 0 connections
    #                              [1, 0, -2],  # Neuron 1 connections
    #                              [1, 1, 0]])*20  #20  # Neuron 2 connections
    # synaptic_weights = (np.random.rand(3,3)+1)*20
    # sign = np.random.randn(3,3); sign[sign>0]=1; sign[sign<0] = -1
    # synaptic_weights = synaptic_weights*sign
    S = synaptic_weights*1
    np.fill_diagonal(S, np.zeros(3))
    # noise_amp = 2

    # Synaptic filtering parameters
    tau_synaptic = 5.0  # synaptic time constant

    # Initialize neuron membrane potentials and synaptic inputs
    v_neurons = np.zeros((3, timesteps))
    synaptic_inputs = np.zeros((3, timesteps))
    spike_times = []
    firing = []
    firing.append((np.array([]), np.array([]))) # init

    # Simulation loop
    for t in range(1, timesteps):

        # Update neuron membrane potentials using leaky integrate-and-fire model
        v_neurons[:, t] = v_neurons[:, t - 1] + dt/tau*(v_rest - v_neurons[:, t - 1]) + np.random.randn(3)*noise_amp
        
        # Check for spikes
        spike_indices = np.where(v_neurons[:, t] > v_threshold)[0]
        
        # Apply synaptic connections with synaptic filtering
        synaptic_inputs[:, t] = synaptic_inputs[:, t-1] + dt*( \
                                -synaptic_inputs[:, t-1]/tau_synaptic + np.sum(synaptic_weights[:, spike_indices], axis=1))
        # synaptic_inputs[:, t] = np.sum(synaptic_weights[:, spike_indices], axis=1)
        # Update membrane potentials with synaptic inputs
        v_neurons[:, t] += synaptic_inputs[:, t]*dt
        
        # record firing
        firing.append([t+0*spike_indices, spike_indices])
        
        # reset and record spikes
        v_neurons[spike_indices, t] = v_reset  # Reset membrane potential for neurons that spiked
        if len(spike_indices) > 0:
            spike_times.append(t * dt)
    
    return firing

# %% baiscs
N = 3
nc = 2**3
num_params = int((N*2**N)) 
spins = [0,1]  # binary patterns
combinations = list(itertools.product(spins, repeat=N))  # possible configurations
lt = 30000

# %% scanning loop
w_s = np.array([5,10,20,40])
n_s = np.array([2,4,8,16])
R2s = np.zeros((len(w_s), len(n_s)))
Wij = np.array([[0, 1, -2],  # Neuron 0 connections
                [1, 0, -2],  # Neuron 1 connections
                [1, 1, 0]])
P0 = np.ones((nc,nc))  # uniform prior
np.fill_diagonal(P0, np.zeros(nc))
np.fill_diagonal(P0, -np.sum(P0,1))
dofs_all = nc**2 + nc
Cp_condition = np.ones(dofs_all)
# cutoff = 20
# Cp_condition[cutoff:] = np.zeros(dofs_all-cutoff) # test with low-d dof constraints!!!

for ww in range(len(w_s)):
    for nn in range(len(n_s)):
        print(ww); print(nn)
        S = Wij*w_s[ww]
        firing = LIF_firing(S, n_s[nn])
        minisi = compute_min_isi(firing)
        adapt_window = int(minisi*10)  #100
        spk_states, spk_times = spk2statetime(firing, adapt_window)  # embedding states
        tau,C = compute_tauC(spk_states, spk_times)  # emperical measurements
        ### not scanning for now...
        rank_tau = np.argsort(tau)[::-1]  # ranking occupency
        rank_C = np.argsort(C.reshape(-1))[::-1]  # ranking transition
        tau_, C_ = tau/lt, C/lt # correct normalization
        observations = np.concatenate((tau_, C_.reshape(-1))) 
        
        ### run max-cal!
        constraints = ({'type': 'eq', 'fun': eq_constraint, 'args': (observations, Cp_condition)})
        bounds = [(.0, 100)]*num_params

        # Perform optimization using SLSQP method
        param0 = np.ones(num_params)*.1 + np.random.rand(num_params)*0.0
        result = minimize(objective_param, param0, args=(P0), method='SLSQP', constraints=constraints, bounds=bounds)
        
        # computed and record the corresponding KL
        param_temp = result.x
        M_inf, pi_inf = param2M(param_temp)
        f1,f2,f3 = M_inf[0,4], M_inf[0,2], M_inf[0,1]
        w12,w13,w21 = np.log(M_inf[4,6]/f2), np.log(M_inf[4,5]/f3), np.log(M_inf[2,6]/f1)
        w23,w32,w31 = np.log(M_inf[2,3]/f3), np.log(M_inf[1,3]/f2), np.log(M_inf[1,5]/f1)
        inf_w = np.array([w12,w13,w21,w23,w32,w31])
        true_s = np.array([S[1,0],S[2,0],S[0,1],S[2,1],S[1,2],S[0,2]])

        correlation_coefficient, _ = pearsonr(inf_w, true_s)
        R2s[ww,nn] = correlation_coefficient
        print(correlation_coefficient)
        
# %%
plt.figure()
img = plt.imshow(R2s, aspect='auto',  origin='lower');
cbar = plt.colorbar(img)
# img.set_clim(0.5, .7) 
plt.xticks(np.arange(len(n_s)), n_s)
plt.yticks(np.arange(len(w_s)), w_s)
plt.xlabel('noise', fontsize=20); plt.ylabel('network', fontsize=20); 
plt.title('R2', fontsize=20)
# plt.title('R2 dof:{}'.format(cutoff), fontsize=20)
# plt.savefig('spk_scan.pdf')

# %% IDEAS
# alter window according to ISI in data
# early cutoff for constraints!
#... next:
# play with motif (cyclic and mutual inhibition)
# download retina

