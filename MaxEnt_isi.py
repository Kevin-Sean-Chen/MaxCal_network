# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:50:40 2024

@author: kevin
"""

from maxcal_functions import spk2statetime, compute_tauC, param2M, eq_constraint, objective_param, compute_min_isi,\
                                sim_Q, word_id, corr_param

from statsmodels.tsa.stattools import acf
import random

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
timesteps = 1000000  # total simulation steps  60 s
lt = timesteps*1

spins = [0,1]  # binary patterns
combinations = list(itertools.product(spins, repeat=N))  # possible configurations
nc = 2**N

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

def MaxEnt_states(firing, window, lt=lt, N=N, combinations=combinations):
    """
    given the firing (time and neuron that fired) data, we choose time window to slide through,
    then convert to network states and the timing of transition
    """
    nstates = int(lt/window)-1
    me_states = np.zeros(nstates)
    for tt in range(nstates):
        this_window = firing[tt*window:tt*window+window]
        word = np.zeros(N)  # template for the binary word
        for ti in range(window): # in time
            if len(this_window[ti][1])>0:
                for nspk in range(len(this_window[ti][1])):  ## NOT CTMC anymore
                    this_neuron = this_window[ti][1][nspk]  # the neuron that fired first
                    word[int(this_neuron)] = 1
        state_id = combinations.index(tuple(word))
        me_states[tt] = state_id
    return me_states

def tau4maxent(states, nc=nc, combinations=combinations, lt=None):
    """
    given the emperically measured states, measure occupency tau and the transitions C
    """
    tau = np.zeros(nc)
    for i in range(len(states)):
        this_state = int(states[i])
        tau[this_state] += 1
    return tau

def sim_maxent(tau, lt=lt, N=N, combinations=combinations):
    me_spk, me_t = [], []
    state_freq = tau/np.sum(tau)
    state_vec = np.arange(len(tau))
    for tt in range(lt):
        state_samp = random.choices(state_vec, state_freq)[0]  # equilibrium sampling!
        word = np.array(combinations[state_samp])
        for nn in range(N):
            if word[nn]==1:
                me_t.append(tt)
                me_spk.append(nn)
    return me_spk, me_t

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
noise_amps = np.array([1, 2.5, 32])

# %% MaxCal inference
###############################################################################
# %% baiscs
num_params = int((N*2**N)) 
lt = timesteps*1 #30000
adapt_window = 200 #int(minisi*10)  #100
isi_bins = np.arange(0,1,.02)*500  #0.05
isi_bins = np.arange(0, 521, 20)  #20  # use bin size or prime to avoid aliasing!

# %% looping
###############################################################################
maxent_isi_scan = []
KL_isi = np.zeros((len(motifs), len(noise_amps)))
KL_isi_m = KL_isi*1
weight_corr = KL_isi*1
cg_corr = KL_isi*1

mm = 2  # for EI network or common...
# for mm in range(len(motifs)):

row = []
for ni in range(len(noise_amps)):
    print('m:'+str(mm) + ', n:'+str(ni))
    ### generate LIF spikes
    S = motifs[mm]*1
    firing, spike_id, spike_times = LIF_motifs(S, noise_amps[ni])
    
    ## measure LIF isi
    LIF_isi = []
    for nn in range(N): #(N):
        pos = np.where(np.array(spike_id)==nn)[0]
        spks = np.array(spike_times)[pos]
        if len(spks)>2:
            spks = spks[np.where(spks<lt)[0]]
            isis = np.diff(spks)/dt
            LIF_isi.append(isis)
    LIF_isi = np.concatenate(LIF_isi, axis=0)

    ### do inference with CTMC
    spk_states, spk_times = spk2statetime(firing, adapt_window, lt=lt)  # embedding states
    tau,C = compute_tauC(spk_states, spk_times)  # emperical measurements
    tau_, C_ = tau/lt, C/lt # correct normalization

    ### infering M
    M_inf = (C_/tau_[:,None]) ### hijack here
    np.fill_diagonal(M_inf, np.zeros(nc))
    Q = M_inf*1 
    np.fill_diagonal(Q, -np.sum(Q,1))
    
    ### reconstruct CTMC spikes
    reps_ctmc = 200        
    ctmc_data, ctmc_ids = [], []
    for rr in range(reps_ctmc):
        ctmc_s, ctmc_t = sim_Q(Q, 100000, 1)
        ctmc_spkt, ctmc_spki = [],[]
        btt = []
        for tt in range(1, len(ctmc_t)):
            state = ctmc_s[tt]
            time = ctmc_t[tt]
            word = np.array(combinations[state])
            for nn in range(N):
                if word[nn]==1:
                    ctmc_spkt.append(time)
                    ctmc_spki.append(nn)
        ctmc_data.append(np.array(ctmc_spkt))
        ctmc_ids.append(np.array(ctmc_spki))
        
    ### measure CTMC isi
    ctmc_isi = []
    for rr in range(len(ctmc_data)):
        spkt = ctmc_data[rr]
        spki = ctmc_ids[rr]
        for nn in range(N): #(N):
            pos = np.where(spki==nn)[0]
            spks = spkt[pos]
            if len(spks)>2:
                isis = np.diff(spks)
                ctmc_isi.append(isis)
    ctmc_isi = np.concatenate(ctmc_isi, axis=0)
    
    ### inference with MaxEnt
    states_me = MaxEnt_states(firing, adapt_window)
    tau_me = tau4maxent(states_me)
    
    ### sample from MaxEnt
    reps_me = 10        
    me_data, me_ids = [], []
    for rr in range(reps_me):
        me_spk, me_t = sim_maxent(tau_me)
        me_data.append(np.array(me_t))
        me_ids.append(np.array(me_spk))
    
    ### measure MaxEnt isi
    me_isi = []
    for rr in range(len(me_data)):
        spkt = me_data[rr]
        spki = me_ids[rr]
        for nn in range(N): #(N):
            pos = np.where(spki==nn)[0]
            spks = spkt[pos]
            if len(spks)>2:
                isis = np.diff(spks) * (adapt_window*dt)
                me_isi.append(isis)
    me_isi = np.concatenate(me_isi, axis=0)
    
    ### KL
    kl_isi_i, hist_LIF, hist_ctmc = kl_divergence(LIF_isi, ctmc_isi, isi_bins)
    kl_isi_m, hist_LIF, hist_me = kl_divergence(LIF_isi, me_isi, isi_bins)
    KL_isi[mm, ni] = kl_isi_i
    KL_isi_m[mm, ni] = kl_isi_m
    
    ### recording...
    element = {'hist_LIF':hist_LIF, 'hist_ctmc':hist_ctmc, 'hist_maxent':hist_me}
    row.append(element)
maxent_isi_scan.append(row)

# %% load data
# import pickle

# with open('ctmc_me_lif_isi3.pkl', 'rb') as f:
#     loaded_data = pickle.load(f)
# maxent_isi_scan = loaded_data*1
# # isi_bins = np.arange(0,1,.02)*500  #0.05 or 0.02

# %%
mm = 0 
bb = (isi_bins[1:] + isi_bins[:-1])/2
# bb = isi_bins[1:]*1 
for ii in range(3):
    plt.figure()
    hist_c = maxent_isi_scan[mm][ii]['hist_ctmc']
    hist_l = maxent_isi_scan[mm][ii]['hist_LIF']
    hist_m = maxent_isi_scan[mm][ii]['hist_maxent']
    plt.plot(bb[:-1], hist_c[:-1], label='CTMC')
    plt.plot(bb[:-1], hist_l[:-1], label='LIF')
    ### plotting for MaxEnt
    leave = np.arange(1,len(hist_m)-1,1)
    leave = hist_m>1e-10  # remove empty
    # leave = np.arange(1,len(hist_m)-0)
    plt.plot(bb[leave], hist_m[leave], 'b--',label='MaxEnt')
    plt.yscale('log'); plt.legend(fontsize=20)
    # plt.savefig('ISI_EI_'+ str(ii+1) +'_me.pdf')