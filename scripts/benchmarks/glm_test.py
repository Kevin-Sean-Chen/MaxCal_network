# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 18:23:40 2025

@author: kevin
"""

from maxcal_network import spk2statetime, compute_tauC, cos_ang, corr_param

import pickle
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
import matplotlib 
import random
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

np.random.seed(1) #1, 37

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Support NumPy 1.24 and later with pyglmnet 1.1.
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int

from pyglmnet import GLM

# %%
def exponential_filter(events, tau, dt=0.1):
    N, T = events.shape
    filtered = np.zeros((N, T))
    alpha = dt / tau

    for n in range(N):
        for t in range(1, T):
            filtered[n, t] = filtered[n, t-1] * (1 - alpha) + events[n, t] * alpha

    return filtered

def LIF_firing_voltage(synaptic_weights, noise_amp, syn_delay=None, syn_ratio=None, stim=None, stim_inter=100, stim_dur = .5, counter=0):
    """
    given synaptic weights and noise amplitude, turn 3-neuron spiking time series
    """
    #############################
    ### perturbation parameters
    # stim_dur = .5  # 1
    # stim_inter = 100  # 100
    #############################
    
    # time settings
    dt = 0.1  # time step in milliseconds
    timesteps = 100000  # total simulation steps
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
            # print(t)
            if t % stim_inter == 0:
                pert_neuron = random.randint(0, 2) # pick neuron
                counter = dt  ### initiate a kick
            if counter > 0 and counter < stim_dur:  # logic for perturbation  ### edit this!!!
                # for nn in range(3):
                    # v_neurons[nn, t] -= It*dt
                    # v_neurons[pert_neuron, t] = v_rest
                # v_neurons[pert_neuron, t] += It*dt
                v_neurons[pert_neuron, t] = v_reset#
                # v_neurons[pert_neuron, t] -= It*dt
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
            
    spk_train = v_neurons*0
    spk_train[v_neurons==v_rest] = 1
    filt_spk_train = exponential_filter(spk_train, 1., dt=dt)
    
    return firing, filt_spk_train #v_neurons #synaptic_inputs

# %% baiscs
N = 3
lt = 100000
reps = 2

# %% parameter sets (not changing any of this)
w_s = np.array([2,4,8,16,32, 64])*2
# w_s = np.array([2, 4, 16])*2*2  ### for network strength

fault_n = np.array([2,2,2])


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

R2s = np.zeros((3, len(w_s), reps, 2))  ### 3 motifs x scan-params x repeats x stim-condition
signs = R2s*0
coss = R2s*0

# %% single test
firing,volt = LIF_firing_voltage(Wij_ei*32*2, 2, syn_delay=None, syn_ratio=None, stim=False, 
                    stim_dur=.5, stim_inter=100)
# Filter out entries where both arrays are empty
firing_filtered = [pair for pair in firing if pair[0].size > 0 or pair[1].size > 0]

# %%
lagt = 15
# for nn in range(N):  ### loop through neurons
nn = 0
X_voltage = np.zeros((lt, lagt*N))  ### time by lagxN
y_spk = np.zeros(lt)
### design matrix
for tt in range(len(firing)):   ### loop through time
    if tt>lagt:
        X_voltage[tt, :] = volt[:, tt-lagt: tt].reshape(-1)
### spike vector
for tt in range(len(firing_filtered)):
    this_n = firing_filtered[tt][1][0] # this neuron spiked
    this_t = firing_filtered[tt][0][0]
    if this_n==nn and tt>lagt:
        y_spk[this_t] = 1

# %%
glm = GLM(distr='binomial', alpha=0.0, score_metric='pseudo_R2',
          learning_rate=1., tol=1e-4, verbose=True)

# fit model
glm.fit(X_voltage, y_spk)

# %%
betas = glm.beta_
beta0 = glm.beta0_
betas = betas.reshape(N,lagt)
plt.figure()
plt.plot(betas.T + beta0*1)

# %% functional for GLM
def GLM_weights(firing, volt, lagt=50):
    firing_filtered = [pair for pair in firing if pair[0].size > 0 or pair[1].size > 0]
    X_voltage = np.zeros((lt, lagt*N))  ### time by lagxN
    weight_matrix = np.zeros((N,N))
    ### design matrix
    for tt in range(len(firing)):   ### loop through time
        if tt>lagt:
            X_voltage[tt, :] = volt[:, tt-lagt: tt].reshape(-1)
    
    ### if we use spikes directly! ####
    # for nn in range(N):
    #     y_spk = np.zeros(lt)
    #     ### spike vector
    #     for tt in range(len(firing_filtered)):
    #         this_n = firing_filtered[tt][1][0] # this neuron spiked
    #         this_t = firing_filtered[tt][0][0]
    #         if this_n==nn and tt>lagt:
    #             X_voltage[this_t, nn] = 1
    
    ### loop through cells
    for nn in range(N):
        y_spk = np.zeros(lt)
        ### spike vector
        for tt in range(len(firing_filtered)):
            this_n = firing_filtered[tt][1][0] # this neuron spiked
            this_t = firing_filtered[tt][0][0]
            if this_n==nn and tt>lagt:
                y_spk[this_t] = 1

        ### do GLM!
        glm = GLM(distr='binomial', alpha=0.0, score_metric='pseudo_R2',
                  learning_rate=1., tol=1e-4, verbose=True)
        glm.fit(X_voltage, y_spk)
        
        ### compute effects
        betas = glm.beta_
        beta0 = glm.beta0_
        betas = betas.reshape(N,lagt)
        weight_matrix[nn, :] = np.mean(betas,1) + beta0*0
    return weight_matrix

# %% looping
for rr in range(reps): ### repears
    for ww in range(0,1):  ### motifs
        Wij = Ws[ww]  # asign motif
        for ii in range(len(w_s)):  ### scan stim params
            print('repeat: ', rr); print('scan: ', ii)
            
            ### simulate LIF spkes
            S = Wij* w_s[ii]  # *fault_w[ww] #
            firing, volt = LIF_firing_voltage(S, fault_n[ww]*1+w_s[ii]*0, syn_delay=None, syn_ratio=None, stim=None, 
                                stim_dur=0.1, stim_inter=100)  ### tune noise, delay, or ratio
            
            ### MaxCal inference
            adapt_window = 150 #int(minisi*10)  #100
            spk_states, spk_times = spk2statetime(firing, adapt_window)  # embedding states
            tau,C = compute_tauC(spk_states, spk_times)  # emperical measurements
            ### not scanning for now...
            tau_, C_ = (tau + 1)/lt, (C + 1)/lt # correct normalization  #######################################
            
            # # computed and record the corresponding KL
            M_inf = (C_/tau_[:,None]) #+ 1/lt  # to prevent exploding
            # M_inf, pi_inf = param2M(param_temp)
            f1,f2,f3 = M_inf[0,4], M_inf[0,2], M_inf[0,1]
            w12,w13,w21 = np.log(M_inf[4,6]/f2), np.log(M_inf[4,5]/f3), np.log(M_inf[2,6]/f1)
            w23,w32,w31 = np.log(M_inf[2,3]/f3), np.log(M_inf[1,3]/f2), np.log(M_inf[1,5]/f1)
            inf_w = np.array([w12,w13,w21,w23,w32,w31])
            true_s = np.array([S[1,0],S[2,0],S[0,1],S[2,1],S[1,2],S[0,2]])
            
            correlation_coefficient, _ = pearsonr(inf_w, true_s)
            R2s[ww, ii, rr, 0] = correlation_coefficient
            signs[ww, ii, rr, 0] = corr_param(true_s, inf_w, 'binary')
            coss[ww, ii, rr, 0] = cos_ang(inf_w, true_s)
            
            ### GLM inference
            wm = GLM_weights(firing, volt, lagt=lagt)
            inf_gc = np.array([wm[0,1],wm[2,0],wm[0,1],wm[2,1],wm[1,2],wm[0,2]]) ### confirm this is correct!
            cceff,_ = pearsonr(inf_gc, true_s)
            R2s[ww, ii, rr, 1] = cceff
            signs[ww, ii, rr, 1] = corr_param(true_s, inf_gc, 'binary')
            coss[ww, ii, rr, 1] = cos_ang(inf_gc, true_s)
            
                
# %% plotting
plt.figure()
scan_x = w_s*1 #w_s #d_s*.1
for ii in range(0,1):  ### pick one motif
    for ss in range(2):  ### inference condition
        plt.subplot(131); 
        plt.errorbar(scan_x, np.mean(R2s[ii,:,:, ss],1), np.std(R2s[ii,:,:, ss],1)); plt.title('R2'); plt.xscale('log');
        plt.subplot(132); 
        plt.errorbar(scan_x, np.mean(signs[ii,:,:, ss],1), np.std(signs[ii,:,:, ss],1)); plt.title('sign'); plt.xscale('log');
        plt.xlabel('weight',fontsize=20)
        plt.subplot(133); 
        plt.errorbar(scan_x, np.mean(coss[ii,:,:, ss],1), np.std(coss[ii,:,:, ss],1)); plt.title('cos'); plt.xscale('log');
        
# %% bar plots

selected_motif = 0
n_groups = 6
n_measurements = 3

meas = [R2s, signs, coss]

# group_labels = ['w=2', '4', '8', '16', '32']
group_labels = ['w=4', '8', '16', '32', '64', '128']
condition_labels = ['MaxCal', 'GLM', 'GC']
measurement_titles = ['R2', 'sign', 'cos']

# Step 2: Setup figure
plt.figure(figsize=(15, 6))

bar_width = 0.35
index = np.arange(n_groups)

for i in range(n_measurements):
    
    meas_i = meas[i]
    mean_i = np.nanmean(meas_i[selected_motif,:,:, 0],1)
    std_i = np.nanstd(meas_i[selected_motif,:,:, 0],1)
    mean_j = np.nanmean(meas_i[selected_motif,:,:, 1],1)
    std_j = np.nanstd(meas_i[selected_motif,:,:, 1],1)
    ax = plt.subplot(1, 3, i+1)

    # Plot both conditions
    ax.bar(index - bar_width/2, mean_i, bar_width,
                    label=condition_labels[0],
                    yerr=std_i, capsize=5)
    
    ax.bar(index + bar_width/2, mean_j, bar_width,
                    label=condition_labels[1],
                    yerr=std_j, capsize=5)

    # Set labels and title
    # ax.set_xlabel('Groups')
    # ax.set_ylabel('Mean Value')
    ax.set_title(measurement_titles[i], fontsize=30)
    ax.set_xticks(index)
    ax.set_xticklabels(group_labels)
    if i == 0:
        ax.legend(fontsize=20)

plt.tight_layout()
plt.show()



# %%
def plot_perturbed(purt='noise'):
    fname = DATA_DIR / "GC_comparison3.pkl"
    with open(fname, 'rb') as f:
        loaded_data = pickle.load(f)
    coss = loaded_data['coss']
    
    bar_width = 0.2
    index = np.arange(n_groups)
    plt.figure()
    meas_i = coss*1
    mean_i = np.nanmean(meas_i[selected_motif,:,:, 0],1)
    std_i = np.nanstd(meas_i[selected_motif,:,:, 0],1)
    mean_j = np.nanmean(meas_i[selected_motif,:,:, 1],1)
    std_j = np.nanstd(meas_i[selected_motif,:,:, 1],1)
    
    fname = DATA_DIR / "glm_comparison3.pkl"
    with open(fname, 'rb') as f:
        loaded_data = pickle.load(f)
    coss = loaded_data['coss']
    meas_i = coss*1
    mean_k = np.nanmean(meas_i[selected_motif,:,:, 1],1)
    std_k = np.nanstd(meas_i[selected_motif,:,:, 1],1)
    
    
    ax = plt.subplot(1, 1, 1)
    # Plot both conditions
    ax.bar(index - bar_width, mean_i, bar_width,
           label=condition_labels[0],
           yerr=std_i, capsize=5)
    
    ax.bar(index + bar_width, mean_j, bar_width,
           label=condition_labels[2],
           yerr=std_j, capsize=5)
    
    ax.bar(index, mean_k, bar_width,
           label=condition_labels[1],
           yerr=std_k, capsize=5)

    ax.set_xticks(index)
    ax.set_xticklabels(group_labels)
    ax.legend()

# plot_perturbed(''); plt.xlabel('weights',fontsize=20); plt.ylabel('cos',fontsize=20); #plt.xscale('log')   

# Optional pickle save example.
# filename = "glm_comparison3.pkl"
# data = {"coss": coss, "scan_x": scan_x, "reps": reps, "lt": lt}
# with open(filename, "wb") as file:
#     pickle.dump(data, file)

plt.show()
