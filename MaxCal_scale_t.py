# -*- coding: utf-8 -*-
"""
Created on Fri May 31 20:11:30 2024

@author: kevin
"""

from maxcal_functions import spk2statetime, compute_tauC, param2M, eq_constraint, \
                            MaxCal_D, objective_param, compute_min_isi, corr_param, sign_corr, P_frw_ctmc, C_P,\
                            word_id, sim_Q

from statsmodels.tsa.stattools import acf
import random
import scipy.io
import numpy as np
from matplotlib import pyplot as plt
import itertools
from scipy.optimize import minimize
from scipy.stats import pearsonr
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

# %% parameter setting outside
N = 3
dt = 0.1  # time step in milliseconds
timesteps = 1000000  # total simulation steps
lt = timesteps*1
window = 200

# vanilla choice...
synaptic_weights = np.array([[0, 1, -2],  # Neuron 0 connections
                              [1, 0, -2],  # Neuron 1 connections
                              [1, 1, 0]])*20  #20  # Neuron 2 connections

# cyclic circuit
# synaptic_weights = np.array([[0, 0, 1],  # Neuron 0 connections
#                               [1, 0, 0],  # Neuron 1 connections
#                               [0, 1, 0]])*20  #20  # Neuron 2 connections

# common circuit
# synaptic_weights = np.array([[0, 0, 0],  # Neuron 0 connections
#                               [1, 0, 0],  # Neuron 1 connections
#                               [1, 0, 0]])*20  #20  # Neuron 2 connections

# chain circuit
synaptic_weights = np.array([[0, 0, 0],  # Neuron 0 connections
                              [1, 0, 0],  # Neuron 1 connections
                              [0, 1, 0]])*20  #20  # Neuron 2 connections

S = synaptic_weights*1

# %% spiking function
def LIF_firing(lt):
    """
    given synaptic weights and noise amplitude, turn 3-neuron spiking time series
    """
    dt = 0.1  # time step in milliseconds
    timesteps = lt*1  #30000  # total simulation steps

    # Neuron parameters
    tau = 10.0  # membrane time constant
    v_rest = -65.0  # resting membrane potential
    # v_threshold = -50.0  # spike threshold
    v_threshold = np.array([-50, -50, -50])  # spike threshold for each neuron
    v_reset = -65.0  # reset potential after a spike

    # Synaptic weight matrix
    S = synaptic_weights*1
    np.fill_diagonal(S, np.zeros(3))
    noise_amp = 2 #1.7

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
        # Update membrane potentials with synaptic inputs
        v_neurons[:, t] += synaptic_inputs[:, t]*dt
        
        # record firing
        firing.append([t+0*spike_indices, spike_indices])
        
        # reset and record spikes
        v_neurons[spike_indices, t] = v_reset  # Reset membrane potential for neurons that spiked
        if len(spike_indices) > 0:
            spike_times.append(t * dt)
    
    return firing

def invf(x):
    output = np.log(x)
    return output

def cos_ang(v1,v2):
    return np.dot(v1,v2) / (np.linalg.norm(v1)* np.linalg.norm(v2))

# firing = LIF_firing()

# %% scaling in time
reps = 10
scalet = np.array([5000 ,10000, 50000, 100000, 500000, 1000000])
weights_time = np.zeros((reps,len(scalet),6))  # repeat x weights x scale
cos_angs_t = np.zeros((reps,len(scalet)))
true_s = np.array([S[1,0],S[2,0],S[0,1],S[2,1],S[1,2],S[0,2]])
save_minC = np.zeros((reps, len(scalet)))

for ss in range(len(scalet)):
    print(ss)
    lti = scalet[ss]
    for rr in range(reps):
        # print(rr)
        firing = LIF_firing(lti)
        ### sub sample
        spk_states, spk_times = spk2statetime(firing, window, lt=lti, N=N)
        tau,C = compute_tauC(spk_states, spk_times, lt=lti)
        tau_, C_ = tau/lti, C/lti
        ### inference of w
        M_inf = (C_/tau_[:,None])
        f1,f2,f3 = M_inf[0,1], M_inf[0,2], M_inf[0,4]
        w12,w13,w21 = invf(M_inf[4,6]/f2), invf(M_inf[4,5]/f3), invf(M_inf[2,6]/f1)
        w23,w32,w31 = invf(M_inf[2,3]/f3), invf(M_inf[1,3]/f2), invf(M_inf[1,5]/f1)
        weights_time[rr,ss,:] = np.array([w12,w13,w21,w23,w32,w31])
        
        ### measure angle
        cos_angs_t[rr,ss] = cos_ang(true_s, np.array([w12,w13,w21,w23,w32,w31]))
        
        ### save counts
        save_minC[rr,ss] = np.min([C[4,6], C[4,5], C[2,6], C[2,3], C[1,3], C[1,5]])
        
# %% plotting
plt.figure()
plt.errorbar(scalet/10000, np.nanmean(cos_angs_t,0), np.nanstd(cos_angs_t,0))
plt.xlabel('time', fontsize=20)
plt.ylabel('cos-ang', fontsize=20); plt.legend(fontsize=20)
plt.xscale('log')
### plt.savefig('cos_ang_time.pdf')

# %%
fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_xlabel('time', fontsize=20)
ax1.set_ylabel('cos-ang', color=color, fontsize=20)
ax1.errorbar(scalet/10000, np.nanmean(cos_angs_t,0), np.nanstd(cos_angs_t,0),color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Create a second y-axis sharing the same x-axis
ax2 = ax1.twinx()
# Plot data for the second y-axis
color = 'tab:red'
ax2.set_ylabel('min flux', color=color, fontsize=20)
ax2.errorbar(scalet/10000, np.nanmean(save_minC,0), np.nanstd(save_minC,0),color=color) # count
# ax2.errorbar(scalet/10000, np.nanmean(save_minC,0)/(scalet/10000), np.nanstd(save_minC,0)/(scalet/10000),color=color) #flux
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_yscale('log')
plt.xscale('log')
# plt.savefig('cos_ang_time_c_chain.pdf')

# %%
# %% saving...
# import pickle

# filename = "cos_ang_time_c_ei.pkl"

# # Store variables in a dictionary
# data = {'cos_angs_t': cos_angs_t, 'save_minC': save_minC}

# # Save variables to a file
# with open(filename, 'wb') as f:
#     pickle.dump(data, f)

# print("Variables saved successfully.")