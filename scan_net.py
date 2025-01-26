# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 18:02:44 2024

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
def LIF_firing(synaptic_weights, noise_amp, syn_delay=None, syn_ratio=None):
    """
    given synaptic weights and noise amplitude, turn 3-neuron spiking time series
    """
    dt = 0.1  # time step in milliseconds
    timesteps = 100000  # total simulation steps
    if syn_delay is not None:
        delay_buffer = np.zeros((1, syn_delay))

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
    
    ### rescaling
    if syn_ratio is not None:
        synaptic_weights[2,0]*=syn_ratio
        # kk = list(test.keys())[0]
        # vv = list(test.values())[0]
        # if kk=='EI':
        #     synaptic_weights[2,1]*=vv
        # if kk=='comm':
        #     synaptic_weights[2,0]*=vv
        # if kk=='conv':
        #     synaptic_weights[2,0]*=vv
            
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
        synaptic_input = synaptic_weights @ spikes[:, t-1]
        if syn_delay is not None:
            ### place buffer to handle delayed spikes
            delay_buffer = np.roll(delay_buffer, -1)
            delay_buffer[0, -1] = spikes[1, t-1]  # Add latest spike from Neuron 2
            synaptic_input[2] += 1. * delay_buffer[0, 0]  ### delayed to the third neuron
        synaptic_inputs[:, t] = synaptic_inputs[:, t-1] + dt*( \
                                -synaptic_inputs[:, t-1]/tau_synaptic + synaptic_input)
        # synaptic_inputs[:, t] = synaptic_inputs[:, t-1] + dt*( \
        #                         -synaptic_inputs[:, t-1]/tau_synaptic + np.sum(synaptic_weights[:, spike_indices], axis=1))
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
lt = 100000

# %% scanning loop
w_s = np.array([1,2,4,8,16])*1  ### for network strength
n_s = np.array([0,2,4,8,16])*1  ### for noist stength
d_s = np.array([1,30,60,90, 120])*1  ### for synaptic delay  (pick one neuron for delay)
h_s = np.array([1,2,4,8,16])  ### for heterogeneious-ness (pick a pair to rescale)

fault_w = np.array([16,16,16])+4 #20*1  ### defualt variables if not tuned
fault_n = np.array([2,2,2])
fault_d = 0
fault_h = 0

R2s = np.zeros((3, len(n_s)))  ### measure 3 motifs as a function of control
signs = R2s*0
coss = R2s*0

### EI balanced
Wij_ei = np.array([[0, 1, -2],  # Neuron 0 connections
                [1, 0, -2],  # Neuron 1 connections
                [1, 1, 0]])*2
### common-input
Wij_common = np.array([[0, 0, 0],  # Neuron 0 connections
                [1, 0, 0],  # Neuron 1 connections
                [1, 0, 0]])*5
## convernged with different strength
# Wij_converge = np.array([[0, 0, 0],  # Neuron 0 connections
#                 [0, 0, 0],  # Neuron 1 connections
#                 [10, 1, 0]])
Wij_converge = np.array([[0, 0, 0],  # Neuron 0 connections
                [0, 0, 0],  # Neuron 1 connections
                [1, 1, 0]])*5

Ws = (Wij_ei, Wij_common, Wij_converge)  # list of motif types

P0 = np.ones((nc,nc))  # uniform prior
np.fill_diagonal(P0, np.zeros(nc))
np.fill_diagonal(P0, -np.sum(P0,1))
dofs_all = nc**2 + nc
Cp_condition = np.ones(dofs_all)


for ww in range(3):
    Wij = Ws[ww]  ### asign motif
    for ii in range(len(d_s)):
        print(ww); print(ii)
        S = Wij*fault_w[ww] #w_s[ii]  #
        firing = LIF_firing(S, fault_n[ww]+w_s[ii]*0, syn_delay=None, syn_ratio=None)  ### tune noise, delay, or ratio
        minisi = compute_min_isi(firing)
        adapt_window = 100 #int(minisi*10)  #100
        spk_states, spk_times = spk2statetime(firing, adapt_window)  # embedding states
        tau,C = compute_tauC(spk_states, spk_times)  # emperical measurements
        ### not scanning for now...
        rank_tau = np.argsort(tau)[::-1]  # ranking occupency
        rank_C = np.argsort(C.reshape(-1))[::-1]  # ranking transition
        tau_, C_ = (tau + 1)/lt, (C + 1)/lt # correct normalization  #######################################
        observations = np.concatenate((tau_, C_.reshape(-1))) 
        
        ### run max-cal!
        # constraints = ({'type': 'eq', 'fun': eq_constraint, 'args': (observations, Cp_condition)})
        # bounds = [(.0, 100)]*num_params

        # # Perform optimization using SLSQP method
        # param0 = np.ones(num_params)*.1 + np.random.rand(num_params)*0.0
        # result = minimize(objective_param, param0, args=(P0), method='SLSQP', constraints=constraints, bounds=bounds)
        # param_temp = result.x
        
        # # computed and record the corresponding KL
        M_inf = (C_/tau_[:,None]) #+ 1/lt  # to prevent exploding
        # M_inf, pi_inf = param2M(param_temp)
        f1,f2,f3 = M_inf[0,4], M_inf[0,2], M_inf[0,1]
        w12,w13,w21 = np.log(M_inf[4,6]/f2), np.log(M_inf[4,5]/f3), np.log(M_inf[2,6]/f1)
        w23,w32,w31 = np.log(M_inf[2,3]/f3), np.log(M_inf[1,3]/f2), np.log(M_inf[1,5]/f1)
        inf_w = np.array([w12,w13,w21,w23,w32,w31])
        true_s = np.array([S[1,0],S[2,0],S[0,1],S[2,1],S[1,2],S[0,2]])

        correlation_coefficient, _ = pearsonr(inf_w, true_s)
        R2s[ww, ii] = correlation_coefficient
        signs[ww, ii] = corr_param(true_s, inf_w, 'binary')
        coss[ww, ii] = cos_ang(inf_w, true_s)
        print(cos_ang(inf_w, true_s))
        
# %%
# scan_x = h_s*1
scan_x = np.array([1, 2, 4, 8, 16])
plt.figure()
plt.subplot(131); plt.semilogx(scan_x, R2s.T,'-o', label=['EI', 'comm', 'conv']); plt.title('R2'); plt.legend()
plt.subplot(132); plt.semilogx(scan_x, signs.T,'-o', label=['EI', 'comm', 'conv']); plt.title('signed corr');plt.xlabel('repeats...')
plt.subplot(133); plt.semilogx(scan_x, coss.T,'-o', label=['EI', 'comm', 'conv']); plt.title('cosine')

#### TO-DO #####
# tune untill the initial point is close, by tuning weights and maybe noise

