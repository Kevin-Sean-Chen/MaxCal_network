# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 00:47:21 2025

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
import random
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

np.random.seed(1) #1, 37

# %%
def LIF_firing(synaptic_weights, noise_amp, syn_delay=None, syn_ratio=None, stim=None, counter=0, lt=100000):
    """
    given synaptic weights and noise amplitude, turn 3-neuron spiking time series
    """
    #############################
    ### perturbation parameters
    stim_dur = 10 #10 # 5
    stim_inter = 200  # 100
    It = 50  #50
    if stim is None:
        It = 0
    #############################
    
    # time settings
    dt = 0.1  # time step in milliseconds
    timesteps = lt  # total simulation steps
    if syn_delay is not None:
        delay_buffer = np.zeros((1, syn_delay))

    # Neuron parameters
    tau = 10.0  # membrane time constant
    v_rest = -65.0  # resting membrane potential
    v_threshold = -50.0  # spike threshold
    v_reset = -65.0  # reset potential after a spike
    
    ### rescaling
    if syn_ratio is not None:
        synaptic_weights[2,0]*=syn_ratio
            
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
            
        # Update membrane potentials with synaptic inputs
        v_neurons[:, t] += synaptic_inputs[:, t]*dt
        
        ### single-neuron perturbation  
        if stim is True:
            if t % stim_inter == 0:
                pert_neuron = 2 #random.randint(0, 2) # pick neuron
                # print(pert_neuron,'++++++++++++++')
                counter = 1
            # if t % stim_inter == 0 and counter < stim_dur:  # logic for perturbation
            #     # for nn in range(3):
            #         # v_neurons[nn, t] -= It*dt
            #         # v_neurons[pert_neuron, t] = v_rest
            #     # v_neurons[pert_neuron, t] += It*dt
            #     v_neurons[pert_neuron, t] = v_reset#
            #     # v_neurons[pert_neuron, t] -= It*dt
            #     counter += 1
    
            if counter > 0 and counter < stim_dur:  # logic for perturbation  ### edit this!!!
                # for nn in range(3):
                    # v_neurons[nn, t] -= It*dt
                    # v_neurons[nn, t] = v_rest
                # v_neurons[:, t] += It*dt
                v_neurons[pert_neuron, t] = v_reset#
                # v_neurons[:, t] -= It*dt
                # v_neurons[pert_neuron, t] = v_neurons[pert_neuron, t]*1 ## control
                counter += 1 *dt
                # print(t)
                
            elif counter>= stim_dur:
                counter = 0
        
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
lt = 50000 #100000
reps = 20

# %% scanning loop
w_s = np.array([1,2,4,8,16])*2  ### for network strength
w_s = np.array([2, 4, 16])*2  ### for network strength

n_s = np.array([1,2,4,8,16])*1  ### for noist stength
d_s = np.array([0.1,30,60,90, 120])*5  ### for synaptic delay  (pick one neuron for delay)
h_s = np.array([1,2,4,8,16])  ### for heterogeneious-ness (pick a pair to rescale)

fault_w = np.array([20,20,10])+0 #20*1  ### defualt variables if not tuned
fault_n = np.array([2,2,2])
fault_d = 0
fault_h = 0

R2s = np.zeros((3, len(w_s), reps, 2))  ### 3 motifs x scan-params x repeats x stim-condition
signs = R2s*0
coss = R2s*0


### EI balanced
Wij_ei = np.array([[0, 1, -2],  # Neuron 0 connections
                [1, 0, -2],  # Neuron 1 connections
                [1, 1, 0]])*1
### common-input
Wij_common = np.array([[0, 0, 0],  # Neuron 0 connections
                [1, 0, 0],  # Neuron 1 connections
                [1, 0, 0]])*5
## convernged with different strength
Wij_converge = np.array([[0, 0, 0],  # Neuron 0 connections
                [0, 0, 0],  # Neuron 1 connections
                [1, 1, 0]])*1

Ws = (Wij_ei, Wij_common, Wij_converge)  # list of motif types

P0 = np.ones((nc,nc))  # uniform prior
np.fill_diagonal(P0, np.zeros(nc))
np.fill_diagonal(P0, -np.sum(P0,1))
dofs_all = nc**2 + nc
Cp_condition = np.ones(dofs_all)

stim_conds = (None, True)


for rr in range(reps): ### repears
    for ss in range(2):  ### stim condition
        for ww in range(0,1):  ### motifs
            Wij = Ws[ww]  # asign motif
            for ii in range(len(w_s)):  ### scanned-params
                print(ww); print(ii)
                S = Wij* w_s[ii]  # *fault_w[ww] #
                firing = LIF_firing(S, fault_n[ww]*1+w_s[ii]*0, syn_delay=None, syn_ratio=None, stim=stim_conds[ss],lt=lt)  ### tune noise, delay, or ratio
                minisi = compute_min_isi(firing, lt=lt)
                adapt_window = 150 #int(minisi*10)  #100
                spk_states, spk_times = spk2statetime(firing, adapt_window,lt=lt)  # embedding states
                tau,C = compute_tauC(spk_states, spk_times, lt=lt)  # emperical measurements
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
                R2s[ww, ii, rr,ss] = correlation_coefficient
                signs[ww, ii, rr,ss] = corr_param(true_s, inf_w, 'binary')
                coss[ww, ii, rr,ss] = cos_ang(inf_w, true_s)
                print(cos_ang(inf_w, true_s))

# %%
plt.figure()
scan_x = w_s #n_s #w_s #d_s*.1
for ii in range(0,1):  ### pick one motif
    for ss in range(2):  ### stim condition
        # plt.subplot(131); 
        # plt.errorbar(scan_x, np.mean(R2s[ii,:,:, ss],1), np.std(R2s[ii,:,:, ss],1)); plt.title('R2'); plt.xscale('log');
        # plt.subplot(132); 
        # plt.errorbar(scan_x, np.mean(signs[ii,:,:, ss],1), np.std(signs[ii,:,:, ss],1)); plt.title('sign'); plt.xscale('log');
        # plt.xlabel('weights',fontsize=20)
        # plt.subplot(133); 
        plt.errorbar(scan_x, np.mean(coss[ii,:,:, ss],1), np.std(coss[ii,:,:, ss],1)); #plt.title('cos'); #plt.xscale('log');

plt.xlabel('weights', fontsize=20); plt.ylabel('cos', fontsize=20) ;plt.ylim([0,1])
# %%
# scan_x = h_s*1
# scan_x = np.array([1, 2, 4, 8, 16])
# plt.figure()
# plt.subplot(131); plt.semilogx(scan_x, R2s.T,'-o', label=['EI', 'comm', 'conv']); plt.title('R2'); plt.legend()
# plt.subplot(132); plt.semilogx(scan_x, signs.T,'-o', label=['EI', 'comm', 'conv']); plt.title('signed corr');plt.xlabel('repeats...')
# plt.subplot(133); plt.semilogx(scan_x, coss.T,'-o', label=['EI', 'comm', 'conv']); plt.title('cosine')

# %%
######
# total excite!!! or total ingibut --> this comes from the fact that 0,0,0 is too little
# write a Raster function!

# %% saving data for plot later
###############################################################################
# %% saving...
# import pickle

# pre_text = 'remedy_stim'
# filename = pre_text + ".pkl"

# # Store variables in a dictionary
# data = {'coss': coss, 'scan_x': scan_x,\
#         'reps': reps, 'lt': lt}

# # Save variables to a file
# with open(filename, 'wb') as f:
#     pickle.dump(data, f)

# print("Variables saved successfully.")