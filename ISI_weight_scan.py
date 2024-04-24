#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:38:39 2024

@author: kschen
"""
from maxcal_functions import spk2statetime, compute_tauC, param2M, eq_constraint, objective_param, compute_min_isi,\
                                sim_Q, word_id, corr_param

from statsmodels.tsa.stattools import acf

import numpy as np
from matplotlib import pyplot as plt
import itertools
from scipy.optimize import minimize
from scipy.stats import pearsonr
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)


# %% LIF model
###############################################################################
# %% simulation settings
N = 3
dt = 0.1  # time step in milliseconds
timesteps = 2000000  # total simulation steps  60 s
lt = timesteps*1

# %% functions
def LIF_motifs(synaptic_weights, noise_amp):
    # Neuron parameters
    tau = 10.0  # membrane time constant
    v_rest = -65.0  # resting membrane potential
    v_threshold = np.array([-50, -50, -50])  # spike threshold for each neuron
    v_reset = -65.0  # reset potential after a spike
    
    S = synaptic_weights*1
    np.fill_diagonal(S, np.zeros(3))
    # noise_amp = 1.5  # 2.0 or 2.5 (1.5, 3.0)
    
    # Synaptic filtering parameters
    tau_synaptic = 5.0  # synaptic time constant
    
    # Initialize neuron membrane potentials and synaptic inputs
    v_neurons = np.zeros((3, timesteps))
    synaptic_inputs = np.zeros((3, timesteps))
    spike_times = []
    spike_id = []
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
            spike_id.append(spike_indices[0])
    
    return firing, spike_id, spike_times

def kl_divergence(p, q, bins):
    # Compute histograms
    hist_p, _ = np.histogram(p, bins=bins, density=True)
    hist_q, _ = np.histogram(q, bins=bins, density=True)

    # Remove zeros from histograms to avoid division by zero
    hist_p[hist_p == 0] = 1e-10
    hist_q[hist_q == 0] = 1e-10

    # Compute KL divergence
    kl_div = np.sum(hist_p * np.log(hist_p / hist_q))

    return kl_div,hist_p, hist_q

def make_bin(indices):
    binary_tuple = (0, 0, 0)
    indices = [index - 1 for index in indices]
    for index in indices:
        # Check if index is within the valid range
        if 0 <= index < 3:
            # Convert tuple to list to modify the element
            binary_list = list(binary_tuple)
            binary_list[index] = 1
            # Convert back to tuple
            binary_tuple = tuple(binary_list)
    return binary_tuple

def coarse_grain_tauC(ijk, tau, C):
    """
    return coupling that is i->j ignoring k
    """
    j,i,k = ijk
    gnd = (0,0,0)
    f = (C[word_id(gnd) , word_id(make_bin([i])) ] + C[word_id(make_bin([k])) , word_id(make_bin([i,k])) ]) \
          /(tau[word_id(gnd)]+tau[word_id(make_bin([k]))])
    fexpw = (C[word_id(make_bin([j])) , word_id(make_bin([i,j])) ] + C[word_id(make_bin([j,k])) , word_id((1,1,1)) ]) \
          /(tau[word_id(make_bin([j]))]+tau[word_id(make_bin([j,k]))])
    weff = np.log(fexpw/f)
    return weff

def cos_ang(v1,v2):
    return np.dot(v1,v2) / (np.linalg.norm(v1)* np.linalg.norm(v2))

# %% motifs and noise
motifs = []
# E-I circuit
synaptic_weights = np.array([[0, 1, -2],  # Neuron 0 connections
                             [1, 0, -2],  # Neuron 1 connections
                             [1, 1,  0]])*20  #20  # Neuron 2 connections
motifs.append(synaptic_weights)
# cyclic circuit
synaptic_weights = np.array([[0, 0, 1],  # Neuron 0 connections
                              [1, 0, 0],  # Neuron 1 connections
                              [0, 1, 0]])*20  #20  # Neuron 2 connections
motifs.append(synaptic_weights)
# common circuit
synaptic_weights = np.array([[0, 0, 0],  # Neuron 0 connections
                              [1, 0, 0],  # Neuron 1 connections
                              [1, 0, 0]])*20  #20  # Neuron 2 connections
motifs.append(synaptic_weights)
# chain circuit
synaptic_weights = np.array([[0, 0, 0],  # Neuron 0 connections
                              [1, 0, 0],  # Neuron 1 connections
                              [0, 1, 0]])*20  #20  # Neuron 2 connections
motifs.append(synaptic_weights)

# noise_amps = np.arange(1.,  5.,  0.5)  # scanning noise strength
noise_amps = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 8, 16, 32])

# %% MaxCal inference
###############################################################################
# %% baiscs
N = 3
nc = 2**3
num_params = int((N*2**N)) 
spins = [0,1]  # binary patterns
combinations = list(itertools.product(spins, repeat=N))  # possible configurations
lt = timesteps*1 #30000
adapt_window = 200 #int(minisi*10)  #100
isi_bins = np.arange(0,1,.05)*500

# %% looping
###############################################################################
# weight_isi_scan = []
# KL_isi = np.zeros((len(motifs), len(noise_amps)))
# weight_corr = KL_isi*1
# cg_corr = KL_isi*1

# for mm in range(len(motifs)):
#     row = []
#     for ni in range(len(noise_amps)):
#         print('m:'+str(mm) + ', n:'+str(ni))
#         ### generate LIF spikes
#         S = motifs[mm]*1
#         firing, spike_id, spike_times = LIF_motifs(S, noise_amps[ni])
        
#         ### measure LIF isi
#         LIF_isi = []
#         for nn in range(N): #(N):
#             pos = np.where(np.array(spike_id)==nn)[0]
#             spks = np.array(spike_times)[pos]
#             if len(spks)>2:
#                 spks = spks[np.where(spks<lt)[0]]
#                 isis = np.diff(spks)/dt
#                 LIF_isi.append(isis)
#         LIF_isi = np.concatenate(LIF_isi, axis=0)

#         ### do inference with CTMC
#         spk_states, spk_times = spk2statetime(firing, adapt_window, lt=lt)  # embedding states
#         tau,C = compute_tauC(spk_states, spk_times)  # emperical measurements
#         tau_, C_ = tau/lt, C/lt # correct normalization

#         ### infering M
#         M_inf = (C_/tau_[:,None]) ### hijack here
#         np.fill_diagonal(M_inf, np.zeros(nc))
#         Q = M_inf*1 
#         np.fill_diagonal(Q, -np.sum(Q,1))
        
#         ### reconstruct CTMC spikes
#         reps_ctmc = 200        
#         ctmc_data, ctmc_ids = [], []
#         for rr in range(reps_ctmc):
#             ctmc_s, ctmc_t = sim_Q(Q, 100000, 1)
#             ctmc_spkt, ctmc_spki = [],[]
#             btt = []
#             for tt in range(1, len(ctmc_t)):
#                 state = ctmc_s[tt]
#                 time = ctmc_t[tt]
#                 word = np.array(combinations[state])
#                 for nn in range(N):
#                     if word[nn]==1:
#                         ctmc_spkt.append(time)
#                         ctmc_spki.append(nn)
#             ctmc_data.append(np.array(ctmc_spkt))
#             ctmc_ids.append(np.array(ctmc_spki))
            
#         ### measure CTMC isi
#         ctmc_isi = []
#         for rr in range(len(ctmc_data)):
#             spkt = ctmc_data[rr]
#             spki = ctmc_ids[rr]
#             for nn in range(N): #(N):
#                 pos = np.where(spki==nn)[0]
#                 spks = spkt[pos]
#                 if len(spks)>2:
#                     isis = np.diff(spks)
#                     ctmc_isi.append(isis)
#         ctmc_isi = np.concatenate(ctmc_isi, axis=0)
        
#         ### # CG weights
#         weff12,weff13,weff21 = coarse_grain_tauC((1,2,3),tau,C), coarse_grain_tauC((1,3,2),tau,C), coarse_grain_tauC((2,1,3),tau,C)
#         weff23,weff32,weff31 = coarse_grain_tauC((2,3,1),tau,C), coarse_grain_tauC((3,2,1),tau,C), coarse_grain_tauC((3,1,2),tau,C)
#         cg_weights = np.array([weff12,weff13,weff21,weff23,weff32,weff31])
        
#         ### infer weights
#         f1,f2,f3 = M_inf[0,4], M_inf[0,2], M_inf[0,1]
#         w12,w13,w21 = np.log(M_inf[4,6]/f2), np.log(M_inf[4,5]/f3), np.log(M_inf[2,6]/f1)
#         w23,w32,w31 = np.log(M_inf[2,3]/f3), np.log(M_inf[1,3]/f2), np.log(M_inf[1,5]/f1)
#         inf_w = np.array([w12,w13,w21,w23,w32,w31])
        
#         ### KL and weights
#         kl_isi_i, hist_LIF, hist_ctmc = kl_divergence(LIF_isi, ctmc_isi, isi_bins)
#         KL_isi[mm, ni] = kl_isi_i
#         true_s = np.array([S[1,0],S[2,0],S[0,1],S[2,1],S[1,2],S[0,2]])
#         cc_weight, _ = pearsonr(inf_w, true_s)
#         weight_corr[mm, ni] = cc_weight
#         cc_cg, _ = pearsonr(cg_weights, true_s)
#         cg_corr[mm, ni] = cc_cg
        
#         ### recording...
#         element = {'KL_isi': kl_isi_i,  'inf_w':inf_w, 'cg_w':cg_weights, \
#                    'weight_corr': cc_weight, 'gij_corr': cc_cg, 'true_s':true_s, 'hist_LIF':hist_LIF, 'hist_ctmc':hist_ctmc}
#         row.append(element)
#     weight_isi_scan.append(row)


# # %%
# plt.figure()
# # plt.plot(KL_isi.reshape(-1), weight_corr.reshape(-1),'.')
# plt.plot(KL_isi.reshape(-1), cg_corr.reshape(-1),'.')

# %% load data
import pickle
# Load variables from file
with open('weight_isi_scan3.pkl', 'rb') as f:
    loaded_data = pickle.load(f)
weight_isi_scan = loaded_data*1 

# %%
plt.figure()
cols = ['r','b','g','k']
msize = range(1,len(noise_amps)+1)
for mm in range(4):
    for ni in range(len(noise_amps)):
        temp_isi = weight_isi_scan[mm][ni]['KL_isi']
        # temp_corr = weight_isi_scan[mm][ni]['gij_corr']
        # temp_corr = weight_isi_scan[mm][ni]['weight_corr']
        # temp_corr = corr_param(weight_isi_scan[mm][ni]['inf_w'], weight_isi_scan[mm][ni]['true_s'])
        temp_corr = cos_ang(weight_isi_scan[mm][ni]['inf_w'], weight_isi_scan[mm][ni]['true_s'])
        plt.plot(temp_isi, temp_corr, 'o', markersize=msize[ni]*1.5+1, color=cols[mm])
plt.xlabel('KL of ISI', fontsize=20)
plt.ylabel('c.c. of weights', fontsize=20)

# %%
plt.figure()
mm = 3
temp_isi, temp_corr, temp_corr_s = [], [], []
for ni in range(len(noise_amps)):
    temp_isi.append(weight_isi_scan[mm][ni]['KL_isi'])
    # temp_corr.append(weight_isi_scan[mm][ni]['gij_corr']) 
    # temp_corr.append(weight_isi_scan[mm][ni]['weight_corr'])
    temp_corr.append(cos_ang(weight_isi_scan[mm][ni]['inf_w'], weight_isi_scan[mm][ni]['true_s']))
    temp_corr_s.append(corr_param(weight_isi_scan[mm][ni]['inf_w'], weight_isi_scan[mm][ni]['true_s']))
reorder = np.argsort(np.array(temp_isi))
temp_isi, temp_corr = np.array(temp_isi), np.array(temp_corr)
plt.plot(temp_isi[reorder], temp_corr[reorder], '-o')#, markersize=msize[ni]*0.5+1, color=cols[mm])
plt.xlabel('KL of ISI', fontsize=20)
plt.ylabel('c.c. of weights', fontsize=20)

# %% double-y
print(temp_isi)
plt.figure()
plt.plot(noise_amps, temp_isi, 'k-o')
plt.ylabel('KL of ISI', fontsize=20)
plt.twinx()
plt.plot(noise_amps, temp_corr, '-o', color='r')  # You can specify the color here if needed
# plt.plot(noise_amps, temp_corr_s, '--o', color='r') 
plt.ylabel('cc. of weights', color='r', fontsize=20); plt.ylim([-.7,1])
plt.gca().tick_params(axis='y', colors='r')
plt.xlabel('noise level', fontsize=20)
plt.xscale('log')
# plt.savefig('ISI_chain.pdf')

# %%
mm = 0
bb = (isi_bins[1:] + isi_bins[:-1])/2
select = [0,3,8]
for ii in range(3):
    plt.figure()
    hist_c = weight_isi_scan[mm][select[ii]]['hist_ctmc']
    hist_l = weight_isi_scan[mm][select[ii]]['hist_LIF']
    plt.plot(bb, hist_l, label='LIF')
    plt.plot(bb, hist_c, label='CTMC')
    plt.yscale('log'); plt.legend(fontsize=20)
    # plt.savefig('ISI_EI_'+ str(ii+1) +'.pdf')
    