# -*- coding: utf-8 -*-
"""
Created on Fri May 31 00:23:13 2024

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
timesteps = 100000 # 300000  # total simulation steps
lt = timesteps*1
window = 200

# vanilla choice...
# synaptic_weights = np.array([[0, 1, -2],  # Neuron 0 connections
#                               [1, 0, -2],  # Neuron 1 connections
#                               [1, 1, 0]])*20  #20  # Neuron 2 connections

# Synaptic weight matrix
# E-I balanced circuit
synaptic_weights = np.array([[0, 1.7, -2.],  # Neuron 0 connections
                              [.3, 0, -2],  # Neuron 1 connections
                              [.3, 1.7,  0]])*20  #20  # Neuron 2 connections
# cyclic circuit
# synaptic_weights = np.array([[0, 0, 1],  # Neuron 0 connections
#                               [1, 0, 0],  # Neuron 1 connections
#                               [0, 1, 0]])*20  #20  # Neuron 2 connections

# common circuit
# synaptic_weights = np.array([[0, 0, 0],  # Neuron 0 connections
#                               [1, 0, 0],  # Neuron 1 connections
#                               [1, 0, 0]])*20  #20  # Neuron 2 connections

# chain circuit
# synaptic_weights = np.array([[0, 0, 0],  # Neuron 0 connections
#                               [1, 0, 0],  # Neuron 1 connections
#                               [0, 1, 0]])*20  #20  # Neuron 2 connections

S = synaptic_weights*1
true_s = np.array([S[1,0],S[2,0],S[0,1],S[2,1],S[1,2],S[0,2]])

# %% for C5_3
hidden_stength = 40  # 2 10 20 30 40

# %% spiking function
def LIF_firing():
    """
    given synaptic weights and noise amplitude, turn 3-neuron spiking time series
    """
    dt = 0.1  # time step in milliseconds
    timesteps = lt*1  #30000  # total simulation steps

    # Neuron parameters
    tau = 10.0  # membrane time constant
    v_rest = -65.0  # resting membrane potential
    # v_threshold = -50.0  # spike threshold
    v_threshold = np.array([-50, -60, -50])  # spike threshold for each neuron
    v_reset = -65.0  # reset potential after a spike

    # Synaptic weight matrix
    S = synaptic_weights*1
    np.fill_diagonal(S, np.zeros(3))
    noise_amp = 1.7  #2

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

def LIF_C5_3():
    N5 = 5  # for 5-neuron similation

    # Neuron parameters
    tau = 10.0  # membrane time constant
    v_rest = -65.0  # resting membrane potential
    v_threshold = np.array([-50, -50, -50, -50, -50])  # spike threshold for each neuron
    v_reset = -65.0  # reset potential after a spike

    # Synaptic weight matrix
    # E-I balanced circuit
    synaptic_3neuron = np.array([[0, 1, -2],  # Neuron 0 connections
                                  [1, 0, -2],  # Neuron 1 connections
                                  [1, 1,  0]])*20  #20  # Neuron 2 connections

    ### EI structured connections
    synaptic_weights = np.ones((N5,N5))*hidden_stength
    sign = np.ones((N5,N5)); sign[:,-1] *= -2  # make on I cell
    synaptic_weights[3:5,2] = np.array([-2,-2])*20

    synaptic_weights = synaptic_weights*sign
    S = synaptic_3neuron[:3,:3]*1
    synaptic_weights[:3,:3] = synaptic_3neuron*1  # fix for 3 neurons

    np.fill_diagonal(S, np.zeros(N))
    np.fill_diagonal(synaptic_weights, np.zeros(N5))
    # synaptic_weights = np.random.randn(3,3)*.8
    noise_amp = 2

    # Synaptic filtering parameters
    tau_synaptic = 5.0  # synaptic time constant

    # Initialize neuron membrane potentials and synaptic inputs
    v_neurons = np.zeros((N5, timesteps))
    synaptic_inputs = np.zeros((N5, timesteps))
    spike_times = []
    firing = []
    firing.append((np.array([]), np.array([]))) # init

    # Simulation loop
    for t in range(1, timesteps):

        # Update neuron membrane potentials using leaky integrate-and-fire model
        v_neurons[:, t] = v_neurons[:, t - 1] + dt/tau*(v_rest - v_neurons[:, t - 1]) + np.random.randn(N5)*noise_amp
        
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

# %% repeat inference...
reps = 10
inf_ws = np.zeros((reps, 6))  # ws
eff_ws = inf_ws*1  # effective coupling
inf_us = inf_ws*1  # us
cgs = inf_ws*1  # couasre-graining
r_curves = np.zeros((reps, 2, 3, 4))  # repeats x (I,phi) x neurons x points

for rr in range(reps):
    print(rr)
    ### simulation
    firing = LIF_firing()
    # firing = LIF_C5_3()
    ### counting
    spk_states, spk_times = spk2statetime(firing, window, lt=lt, N=N)
    tau,C = compute_tauC(spk_states, spk_times, lt=lt)
    tau_, C_ = tau/lt, C/lt
    ### inference of w
    M_inf = (C_/tau_[:,None])
    f1,f2,f3 = M_inf[0,1], M_inf[0,2], M_inf[0,4]
    w12,w13,w21 = invf(M_inf[4,6]/f2), invf(M_inf[4,5]/f3), invf(M_inf[2,6]/f1)
    w23,w32,w31 = invf(M_inf[2,3]/f3), invf(M_inf[1,3]/f2), invf(M_inf[1,5]/f1)
    inf_ws[rr,:] = np.array([w12,w13,w21,w23,w32,w31])
    ### refractory
    r1,r2,r3 = M_inf[4,0], M_inf[2,0], M_inf[1,0]
    u12,u13,u21 = -invf(M_inf[6,4]/r2), -invf(M_inf[5,4]/r3), -invf(M_inf[6,2]/r1)
    u23,u32,u31 = -invf(M_inf[3,2]/r3), -invf(M_inf[3,1]/r2), -invf(M_inf[5,1]/r1)
    inf_us[rr,:] = np.array([u12,u13,u21,u23,u32,u31])
    ### higher order
    w231, w132, w123 = invf(M_inf[3,7]/f1), invf(M_inf[5,7]/f2), invf(M_inf[6,7]/f3)
    u231, u132, u123 = -invf(M_inf[7,3]/r1), -invf(M_inf[7,5]/r2), -invf(M_inf[7,6]/r3)
    ### effective w'
    eff31 = w231-w21
    eff32 = w132-w12
    eff21 = w231-w31
    eff23 = w123-w13
    eff12 = w132-w32
    eff13 = w123-w23
    eff_ws[rr,:] = np.array([eff12, eff13, eff21  ,eff23, eff31, eff32])
    ### coarse graining
    weff12,weff13,weff21 = coarse_grain_tauC((1,2,3),tau,C), coarse_grain_tauC((1,3,2),tau,C), coarse_grain_tauC((2,1,3),tau,C)
    weff23,weff32,weff31 = coarse_grain_tauC((2,3,1),tau,C), coarse_grain_tauC((3,2,1),tau,C), coarse_grain_tauC((3,1,2),tau,C)
    cgs[rr,:] = np.array([weff12,weff13,weff21,weff23,weff32,weff31])
    
    ### response curves
    ws = np.array([0, w21, w31, w21+w31])
    phis = np.array([f1, M_inf[2,6], M_inf[1,5], M_inf[3,7]])
    sort_id = np.argsort(ws)
    r_curves[rr,0,0,:] = ws[sort_id]
    r_curves[rr,1,0,:] = phis[sort_id]
    ws = np.array([0, w12, w32, w12+w32])
    phis = np.array([f2, M_inf[4,6], M_inf[1,3], M_inf[5,7]])
    sort_id = np.argsort(ws)
    r_curves[rr,0,1,:] = ws[sort_id]
    r_curves[rr,1,1,:] = phis[sort_id]
    ws = np.array([0, w13, w23, w13+w23])
    phis = np.array([f3, M_inf[4,5], M_inf[2,3], M_inf[6,7]])
    sort_id = np.argsort(ws)
    r_curves[rr,0,2,:] = ws[sort_id]
    r_curves[rr,1,2,:] = phis[sort_id]
    
# %% plot
def plot_inf(inf_ws, what):
    bar_width = 0.35
    bar_positions_group2 = np.arange(6)
    if what=='w':
        bar_positions_group1 = ['w12','w13','w21','w23','w32','w31']
    elif what=='u':
        bar_positions_group1 = ['u12','u13','u21','u23','u32','u31']
    elif what=='eff':
        bar_positions_group1 = ['eff12','eff13','eff21','eff23','eff32','eff31']
    elif what=='cg':
        bar_positions_group1 = ['cg12','cg13','cg21','cg23','cg32','cg31']
    plt.figure()
    plt.subplot(211)
    plt.bar( ['w12','w13','w21','w23','w32','w31'], [S[1,0],S[2,0],S[0,1],S[2,1],S[1,2],S[0,2]], width=bar_width)
    plt.bar(np.arange(4),[S[1,0],S[2,0],S[0,1],S[2,1]],width=bar_width,color='orange')
    plt.plot( ['w12','w13','w21','w23','w32','w31'], bar_positions_group2*0, 'k')
    plt.ylabel('true weights', fontsize=20)
    plt.subplot(212)
    plt.bar(bar_positions_group1, np.mean(inf_ws,0), width=bar_width)
    plt.bar(np.arange(4), np.mean(inf_ws[:,:4],0), width=bar_width,color='orange') ## for E cells
    plt.errorbar(np.arange(6), np.mean(inf_ws,0), np.std(inf_ws,0)/np.sqrt(1), fmt='none', ecolor='black')
    plt.plot(bar_positions_group1, bar_positions_group2*0, 'k')
    plt.ylabel('MaxCal inferred', fontsize=20)

# %%
plot_inf(inf_ws, 'w')
plot_inf(inf_us, 'u')
plot_inf(eff_ws, 'eff')
plot_inf(cgs, 'cg')
# plt.savefig('C53_w_40.pdf')

# %% response curve
cols = ['orange','orange','blue']
fig, ax = plt.subplots()
for nn in range(3):
    temp_I = r_curves[:,0,nn,:].squeeze()
    temp_phi = r_curves[:,1,nn,:].squeeze()
    ax.errorbar(np.mean(temp_I,0), np.mean(temp_phi,0), \
                xerr=np.std(temp_I,0)/np.sqrt(1), yerr=np.std(temp_phi,0)/np.sqrt(1), color=cols[nn])
plt.xlabel('I',fontsize=20); plt.ylabel('phi',fontsize=20); plt.legend(fontsize=15)
# plt.savefig('I_phi_error2.pdf')

# %% to-do
# add function to extract u and effective w... for EI circuit
# run for motifs with w and cg
# run for C5_3 with w

# %% saving...
# import pickle

# pre_text = 'C5_3_scan'
# filename = pre_text + "_" + str(hidden_stength) + ".pkl"

# # Store variables in a dictionary
# data = {'c53_w_2': c53_w_2, 'c53_w_10': c53_w_10, 'c53_w_20': c53_w_20,\
#         'c53_w_30': c53_w_30, 'c53_w_40': c53_w_40}

# # Save variables to a file
# with open(filename, 'wb') as f:
#     pickle.dump(data, f)

# print("Variables saved successfully.")

# %%
# import pickle

# pre_text = 'EI_fig4_curve'
# filename = pre_text + "_" + str(hidden_stength) + ".pkl"

# # Store variables in a dictionary
# data = {'r_curves': r_curves, 'cgs': cgs, 'eff_ws': eff_ws,\
#         'inf_us': inf_us, 'inf_ws': inf_ws}

# # Save variables to a file
# with open(filename, 'wb') as f:
#     pickle.dump(data, f)

# %% loading
# import pickle

# h_str = 40  #2,10,20,30,40
# # Load variables from file
# with open('C5_3_scan_2.pkl', 'rb') as f:
#     loaded_data = pickle.load(f)
# c53_w_2,c53_w_10,c53_w_20,c53_w_30,c53_w_40 = loaded_data['c53_w_2'],loaded_data['c53_w_10'],loaded_data['c53_w_20'],loaded_data['c53_w_30'],loaded_data['c53_w_40']

# %% plotting curve
# hs_ws = np.zeros((10,6,5))  # reps x weights x condition
# hs_ws[:,:,0], hs_ws[:,:,1], hs_ws[:,:,2], hs_ws[:,:,3], hs_ws[:,:,4] = c53_w_2,c53_w_10,c53_w_20,c53_w_30,c53_w_40
# true_s = np.array([S[1,0],S[2,0],S[0,1],S[2,1],S[1,2],S[0,2]])

# hs_cosang = np.zeros((10,5)) # reps x condition
# for hh in range(5):
#     this_w = hs_ws[:,:,hh].squeeze()
#     for rr in range(10):
#         hs_cosang[rr,hh] = cos_ang(true_s, this_w[rr,:])

# plt.figure()
# # plt.plot(np.array([2,10,20,30,40])/20, np.mean(hs_cosang,0))
# plt.errorbar(np.array([2,10,20,30,40])/20, np.mean(hs_cosang,0), np.std(hs_cosang,0))
# plt.xlabel('h/s', fontsize=20)
# plt.ylabel('cos-ang', fontsize=20); plt.legend(fontsize=20)
#### plt.savefig('c53_error_cos.pdf')

# %% plotting cosine angle for inferred weights
# import pickle

# with open(file, "rb") as f:   # 'rb' = read binary
#     data = pickle.load(f)

# print(type(data))
# print(data)

# items = ['inf_ws', 'inf_us', 'eff_ws']
# inff = ['w', 'u','w\'']
# for ii in range(len(items)):
#     temp = data[items[ii]]
#     coss = []
#     for jj in range(temp.shape[0]):
#         coss.append(cos_ang(true_s, temp[jj,:]))
#     coss = np.array(coss)
#     plt.bar(inff[ii], np.nanmean(coss), color='k', alpha=0.5)
#     plt.errorbar(inff[ii], np.nanmean(coss), yerr=np.nanstd(coss),color='k')
# plt.ylabel('cosine angle', fontsize=20)
### plt.savefig('EI_compare_wu.pdf')