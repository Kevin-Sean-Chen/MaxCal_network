# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 17:50:11 2024

@author: kevin
"""

import numpy as np
from matplotlib import pyplot as plt
import itertools
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

# %% Izhikevich spiking circuit
lt = 10000
Ne = 2  # 2 excitation
Ni = 1  # 1 inhibition
N = Ne + Ni
re = np.random.rand(Ne)
ri = np.random.rand(Ni)
a = np.concatenate((0.02*np.ones(Ne) , 0.02+0.08*ri))
b = np.concatenate((0.2*np.ones(Ne), 0.25-0.05*ri))
c = np.concatenate((-65+15*re**2 , -65*np.ones(Ni)))
d = np.concatenate((8-6*re**2 , 2*np.ones(Ni)))
S = np.concatenate((0.5*np.random.rand(Ne+Ni, Ne), -np.random.rand(Ne+Ni, Ni)), axis=1)
v = -65*np.ones(Ne+Ni)
u = b*v
firing = []#np.zeros((lt,2))
for tt in range(lt):
    I = np.concatenate((5*np.random.randn(Ne) , 2*np.random.randn(Ni)))
    fired = np.where(v>=30)[0]
    firing.append([tt+0*fired, fired])
    v[fired] = c[fired]
    u[fired] = u[fired] + d[fired]
    I = I + np.sum(S[:,fired])
    v = v + 0.5*(0.04*v**2 + 5*v + 140 - u + I) # step 0.5 ms
    v = v + 0.5*(0.04*v**2 + 5*v + 140 - u + I) # for numerical
    u = u + a*(b*v - u) # stability

plt.figure()
for tt in range(lt):
    plt.plot(firing[tt][0], firing[tt][1],'k.')

# %% extract spike time per neuron... can later check ISI distribution
isi = np.zeros(N)
for nn in range(N):
    # spk_i = np.array([firing[ii][0] for ii in range(lt) if len(firing[ii][0])>0 and firing[ii][1]==nn]).squeeze()
    spk_i = []
    for tt in range(lt):
        if len(firing[tt][0])>0 and len(firing[tt][0])<2 and firing[tt][1]==nn:
            spk_i.append(firing[tt][0])
        elif len(firing[tt][0])>1:  # if we happen to have synchronous spike breaking CTMC!
            spk_i.append(np.array([firing[tt][0][0]]))  # random for the first one~
    spk_i = np.array(spk_i).squeeze()
    isi[nn] = np.min(np.diff(spk_i))
min_isi = np.min(isi)
print(min_isi)

# %%
# scan through spikes to find smallest step
# sliding window to compute tau and C
def spk2statetime(firing, window=20, N=N):
    """
    given the firing (time and neuron that fired) data, we choose time window to slide through,
    then convert to network states and the timing of transition
    """
    spins = [0,1]  # binary patterns
    combinations = list(itertools.product(spins, repeat=N))  # possible configurations
    states_spk = np.zeros(lt-window)
    for tt in range(lt-window):
        this_window = firing[tt:tt+window]
        word = np.zeros(N)  # template for the binary word
        for ti in range(window): # in time
            if len(this_window[ti][1])>0:
                this_neuron = this_window[ti][1][0]  # the neuron that fired
                word[this_neuron] = 1
        state_id = combinations.index(tuple(word))
        states_spk[tt] = state_id
    
    # now compute all the transitions
    trans_temp = np.diff(states_spk)  # find transitions
    spk_times = np.where(np.abs(trans_temp)>0)[0]  # spike timing
    spk_states = states_spk[spk_times].astype(int)   # spiking states
    return spk_states, spk_times

spk_states, spk_times = spk2statetime(firing)
plt.figure()
plt.plot(spk_states)

# %% NEXT
# put code together
# check if Wij roughly matches M
# check how to interpret higher-order w_ijk -> response curve!
# come back to thinkig about the time window

# %%
# # Simulation parameters
# dt = 0.1  # time step in milliseconds
# timesteps = 1000  # total simulation steps

# # Neuron parameters
# tau = 10.0  # membrane time constant
# v_rest = -65.0  # resting membrane potential
# v_threshold = -50.0  # spike threshold
# v_reset = -65.0  # reset potential after a spike

# # Synaptic weight matrix
# synaptic_weights = np.array([[0, 1, -1],  # Neuron 0 connections
#                              [5, 0, 5],  # Neuron 1 connections
#                              [-2, 3, 0]])*30 # Neuron 2 connections
# # synaptic_weights = np.random.randn(3,3)*.8

# # Synaptic filtering parameters
# tau_synaptic = 2.0  # synaptic time constant

# # Initialize neuron membrane potentials and synaptic inputs
# v_neurons = np.zeros((3, timesteps))
# synaptic_inputs = np.zeros((3, timesteps))
# spike_times = []

# # Simulation loop
# for t in range(1, timesteps):

#     # Update neuron membrane potentials using leaky integrate-and-fire model
#     v_neurons[:, t] = v_neurons[:, t - 1] + dt/tau*(v_rest - v_neurons[:, t - 1]) + np.random.randn(3)*.5
    
#     # Check for spikes
#     spike_indices = np.where(v_neurons[:, t] > v_threshold)[0]
    
#     # Apply synaptic connections with synaptic filtering
#     synaptic_inputs[:, t] = synaptic_inputs[:, t-1] + dt*( \
#                             -synaptic_inputs[:, t-1]/tau_synaptic + np.sum(synaptic_weights[:, spike_indices], axis=1))

#     # Update membrane potentials with synaptic inputs
#     v_neurons[:, t] += synaptic_inputs[:, t]*dt
    
#     # reset and record spikes
#     v_neurons[spike_indices, t] = v_reset  # Reset membrane potential for neurons that spiked
#     if len(spike_indices) > 0:
#         spike_times.append(t * dt)

# # Plotting
# plt.figure(figsize=(10, 5))
# for i in range(3):
#     plt.plot(np.arange(timesteps) * dt, v_neurons[i, :], label=f'Neuron {i}')

# plt.scatter(spike_times, np.ones_like(spike_times) * v_threshold, color='red', marker='o', label='Spikes')
# plt.xlabel('Time (ms)')
# plt.ylabel('Membrane Potential')
# plt.title('Neural Circuit Simulation')
# plt.legend()
# plt.show()
