# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 17:50:11 2024

@author: kevin
"""

import numpy as np
from matplotlib import pyplot as plt
import itertools
from scipy.optimize import minimize
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)


# %% LIF model
# Simulation parameters
N = 3
dt = 0.1  # time step in milliseconds
timesteps = 30000  # total simulation steps
lt = timesteps*1

# Neuron parameters
tau = 10.0  # membrane time constant
v_rest = -65.0  # resting membrane potential
v_threshold = -50.0  # spike threshold
v_reset = -65.0  # reset potential after a spike

# Synaptic weight matrix
synaptic_weights = np.array([[0, 1, -2],  # Neuron 0 connections
                             [1, 0, -2],  # Neuron 1 connections
                             [1, 1, 0]])*20  #20  # Neuron 2 connections
S = synaptic_weights*1
# synaptic_weights = np.random.randn(3,3)*.8
noise_amp = 2

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
    

# Plotting
plt.figure(figsize=(10, 5))
for i in range(3):
    plt.plot(np.arange(timesteps) * dt, v_neurons[i, :], label=f'Neuron {i}')

plt.scatter(spike_times, np.ones_like(spike_times) * v_threshold, color='red', marker='o', label='Spikes')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential')
plt.title('Neural Circuit Simulation')
plt.legend()
plt.show()

plt.figure()
for tt in range(timesteps-1):
    plt.plot(firing[tt][0], firing[tt][1],'k.')

# %% bursting analysis
allspktime = []
for tt in range(timesteps-1):
    allspktime.append(firing[tt][0])
allspktime = np.concatenate(allspktime).squeeze()
isi = np.diff(allspktime)
plt.figure()
aa,bb = np.histogram(isi,30)
plt.plot(bb[1:],aa,'-o')
# plt.yscale('log')

# %% with Firing data in hand...
###############################################################################

# %% extract spike time per neuron... can later check ISI distribution
isi = np.zeros(N)
for nn in range(N):
    # spk_i = np.array([firing[ii][0] for ii in range(lt) if len(firing[ii][0])>0 and firing[ii][1]==nn]).squeeze()
    spk_i = []
    for tt in range(lt):
        if len(firing[tt][0])>0 and len(firing[tt][0])<2 and firing[tt][1]==nn:
            spk_i.append(firing[tt][0])
        elif len(firing[tt][0])>1:  # if we happen to have synchronous spike breaking CTMC!
            spk_i.append(np.array([firing[tt][0][0]]))  # random for the first one!!??
    spk_i = np.array(spk_i).squeeze()
    isi[nn] = np.min(np.diff(spk_i))
min_isi = np.min(isi)
print(min_isi)

spins = [0,1]  # binary patterns
combinations = list(itertools.product(spins, repeat=N))  # possible configurations

# %%
# scan through spikes to find smallest step
# sliding window to compute tau and C
def spk2statetime(firing, window, N=N, combinations=combinations):
    """
    given the firing (time and neuron that fired) data, we choose time window to slide through,
    then convert to network states and the timing of transition
    """
    states_spk = np.zeros(lt-window)
    for tt in range(lt-window):
        this_window = firing[tt:tt+window]
        word = np.zeros(N)  # template for the binary word
        for ti in range(window): # in time
            if len(this_window[ti][1])>0:
                this_neuron = this_window[ti][1][0]  # the neuron that fired first
                word[this_neuron] = 1
        state_id = combinations.index(tuple(word))
        states_spk[tt] = state_id
    
    # now compute all the transitions
    trans_temp = np.diff(states_spk)  # find transitions
    spk_times = np.where(np.abs(trans_temp)>0)[0]  # spike timing
    spk_states = states_spk[spk_times].astype(int)   # spiking states
    return spk_states, spk_times

spk_states, spk_times = spk2statetime(firing, window=100)
plt.figure()
plt.plot(spk_states)

# %% NEXT
# put code together
# check if Wij roughly matches M
# check how to interpret higher-order w_ijk -> response curve!
# come back to thinkig about the time window
#... retina!!

###############################################################################
# IMPORTANT
# find a bin/window method that guarantees CTMC transitions!
###############################################################################
###
# phase portait of W strength vs. noise strength, and measure correlation of reconstruction
# play with motif
# download retina
###

# %% CTMC setup
# N = 3  # number of neurons
num_params = int((N*2**N))  # number of parameters in model without refractory assumption
nc = 2**N  # number of total states of 3-neuron

# %% counting and ranking analysis
def param2M(param, N=N, combinations=combinations):
    """
    given array of parameters with length N*2**N, network size N, return transition matrix
    the matrix is a general CTMC form for 
    """
    nc = 2**N  # number of states
    
    ### idea: M = mask*FR, with mask for ctmc, FR is the rest of the transitions
    mask = np.ones((nc,nc))  # initialize the tilted matrix
    FR = mask*1
    # make the mask
    for ii in range(nc):
        for jj in range(nc):
            # Only allow one flip logic in ctmc
            if sum(x != y for x, y in zip(combinations[ii], combinations[jj])) != 1:
                mask[ii,jj] = 0
    
    # now make F matrix!
    kk = 0
    for ii in range(nc):
        for jj in range(nc):
            if mask[ii,jj]==1: #only check those that generates one spike
                FR[ii,jj] = param[kk]
                kk = kk+1  # marching forward to fill in f*exp(wij) parts... need to later invert this!!
    # print(kk)
    M = mask*FR  

    ### compute steady-state
    np.fill_diagonal(M, -np.sum(M,1))  # fill diagonal for continuous time Markov transition Q (is this correct?!)
    uu,vv = np.linalg.eig(M.T)
    zeros_eig_id = np.argmin(np.abs(uu-1))
    pi_ss = vv[:,zeros_eig_id] / np.sum(vv[:,zeros_eig_id])
    
    return M, np.real(pi_ss)

def compute_tauC(states, times, nc=nc, combinations=combinations):
    """
    given the emperically measured states, measure occupency tau and the transitions C
    """
    tau = np.zeros(nc)
    C = np.zeros((nc,nc))
    # compute occupency time
    for i in range(len(states)-1):
        this_state = states[i]
        tau[this_state] += times[i+1]-times[i]  ### total time occupancy
        
    # compute transitions
    for t in range(len(states)-1):
        ii,jj = states[t], states[t+1]
        if ii != jj:
            if sum(x != y for x, y in zip(combinations[ii], combinations[jj])) == 1:  # ignore those not CTMC for now!
                C[ii,jj] += 1  ### counting the transtion
    return tau, C    

tau,C = compute_tauC(spk_states, spk_times)  # emperical measurements

# %% new tau C computation from firing
def firing2tauC(firing):
    
    return tau, C

# %% computing statistics for infinite data given parameter
def P_frw_ctmc(param):
    """
    get joint probability given parameters (need to change for ctmc??)
    """
    k, pi = param2M(param)  # calling asymmetric network
    nc = len(pi)
    Pxy = np.zeros((nc,nc))
    for ii in range(nc):
        Pxy[ii,:] = k[ii,:]*pi[ii]  # compute joint from transition k and steady-state pi (this is wrong using Q!?)
    return Pxy

def edge_flux_inf(param):
    """
    compute edge flux with infinite data using pi_i k_ij
    """
    kij, pi = param2M(param)
    nc = len(pi)
    flux_ij = np.zeros((nc,nc))
    for ii in range(nc):
        for jj in range(nc):
            if ii is not jj:
                flux_ij[ii,jj] = pi[ii]*kij[ii,jj]
    return flux_ij

# %% Maxcal functions (should write better code and import once confirmed...)
def MaxCal_D(Pij, kij0, param):
    """
    KL devergence term, with transition Pij and prior rate kij0 as input
    This term can be unstable in log!
    """
    pi = get_stationary(Pij)
    # kij = Pij / pi[:,None]
    eps = 1e-11
    kl = 0
    n = len(pi)
    for ii in range(n):
        for jj in range(n):
            if ii is not jj:
                kl += Pij[ii,jj]*(np.log(Pij[ii,jj]+eps)-np.log(pi[ii]*kij0[ii,jj]+eps)) \
                      + pi[ii]*kij0[ii,jj] - Pij[ii,jj]
    return kl

def get_stationary(M):
    """
    get stationary state distribution given a transition matrix M
    """
    uu,vv = np.linalg.eig(M.T)
    zeros_eig_id = np.argmin(np.abs(uu-1))
    pix = vv[:,zeros_eig_id] / np.sum(vv[:,zeros_eig_id])
    return np.real(pix)


def C_P(P, observations, param, Cp_condition):
    """
    The C(P,f,r,w) function for constraints
    """
    _,pi = param2M(param)
    flux_ij = edge_flux_inf(param)
    # trick to make all observable a vector and shield with Cp conditions!
    obs_all = np.concatenate((pi, flux_ij.reshape(-1)))
    # obs_all = np.concatenate((flux_ij.reshape(-1), pi))
    cp_dof = obs_all * Cp_condition
    return cp_dof - (observations * Cp_condition)

def objective_param(param, kij0):
    """
    objective in the parameter space, using frw and adding extra constraints
    """
    Pyx,_ = param2M(param)
    D = MaxCal_D(Pyx, kij0, param)
    return D

def eq_constraint(param, observations, Cp_condition):
    Pxy = P_frw_ctmc(param)
    cp = C_P(Pxy, observations, param, Cp_condition)
    return 0.5*np.sum(cp**2) ### not sure if this hack is legit, but use MSE for now


# %% building constraints
dofs = num_params*1
dofs_all = nc**2 + nc
target_dof = dofs + nc
kls = np.zeros(target_dof) # measure KL
Cp_condition = np.zeros(dofs_all)  # mask... problem: this is ncxnc not dof???
rank_tau = np.argsort(tau)[::-1]  # ranking occupency
rank_C = np.argsort(C.reshape(-1))[::-1]  # ranking transition
tau_, C_ = tau/lt, C/lt # correct normalization
observations = np.concatenate((tau_, C_.reshape(-1)))  # observation from simulation!
# observations = np.concatenate((C_.reshape(-1), tau_))

P0 = np.ones((nc,nc))  # uniform prior
np.fill_diagonal(P0, np.zeros(nc))
np.fill_diagonal(P0, -np.sum(P0,1))
ii = 0
### scan through all dof
while ii < target_dof:
    ### work on tau first
    if ii<nc-1:
        Cp_condition[rank_tau[ii]] = 1
    else:
        Cp_condition[rank_C[ii-(nc-1)]+(nc)] = 1
    
    ### run max-cal!
    constraints = ({'type': 'eq', 'fun': eq_constraint, 'args': (observations, Cp_condition)})
    bounds = [(.0, 100)]*num_params

    # Perform optimization using SLSQP method
    param0 = np.ones(num_params)*.1 + np.random.rand(num_params)*0.0
    result = minimize(objective_param, param0, args=(P0), method='SLSQP', constraints=constraints, bounds=bounds)
    
    # computed and record the corresponding KL
    param_temp = result.x
    Pyx,_ = param2M(param_temp)
    kls[ii] = MaxCal_D(Pyx, P0, param_temp)
    print(ii)    
    ii = ii+1


# %%
plt.figure()
plt.plot(np.log(kls[:]),'-o')
plt.xlabel('ranked dof', fontsize=20)
plt.ylabel('KL', fontsize=20)
# plt.ylim([0,10])

# %% attempt to compare inferred M and nework W... ask Peter~
M_inf, pi_inf = param2M(param_temp)
plt.figure()
plt.imshow(M_inf, aspect='auto')
plt.colorbar()

# %%
# param_order = np.arange(1, len(param_temp)+1)
# M_order,_ = param2M(param_order)

# M = np.array([[0,    f3,   f2,   0,              f1,   0,               0,               0],
#               [r3,   0,    0,    f2*np.exp(w32), 0,    f1*np.exp(w31),  0,               0],
#               [r2,   0,    0,    f3*np.exp(w23), 0,    0,               f1*np.exp(w21),  0],
#               [0,    r2,   r3,   0,              0,    0,               0,               f1*np.exp(w231)],
#               [r1,   0,    0,    0,              0,    f3*np.exp(w13),  f2*np.exp(w12),  0],
#               [0,    r1,   0,    0,              r3,   0,               0,               f2*np.exp(w132)],
#               [0,    0,    r1,   0,              r2,   0,               0,               f3*np.exp(w123)],
#               [0,    0,    0,    r1,             0,    r2,              r3,              0]]) 

def invf(x):
    output = np.log(x)
    # output = np.log(-1+np.exp(-x))
    return output

f1,f2,f3 = M_inf[0,4], M_inf[0,2], M_inf[0,1]
# w12,w13,w21 = np.log(M_inf[4,6]/f2), np.log(M_inf[4,5]/f3), np.log(M_inf[2,6]/f1)
# w23,w32,w31 = np.log(M_inf[2,3]/f3), np.log(M_inf[1,3]/f2), np.log(M_inf[1,5]/f1)
w12,w13,w21 = invf(M_inf[4,6]/f2), invf(M_inf[4,5]/f3), invf(M_inf[2,6]/f1)
w23,w32,w31 = invf(M_inf[2,3]/f3), invf(M_inf[1,3]/f2), invf(M_inf[1,5]/f1)

bar_width = 0.35
bar_positions_group1 = np.arange(6)
bar_positions_group2 = bar_positions_group1 + bar_width*0
plt.figure()
plt.subplot(211)
plt.bar(bar_positions_group1,[S[1,0],S[2,0],S[0,1],S[2,1],S[1,2],S[0,2]],width=bar_width)
# plt.bar(bar_positions_group1,[S[0,1],S[0,2],S[1,0],S[1,2],S[2,1],S[2,0]],width=bar_width)
plt.plot(bar_positions_group1, bar_positions_group1*0, 'k')
plt.subplot(212)
plt.bar(bar_positions_group2,np.array([w12,w13,w21,w23,w32,w31])+0,width=bar_width)
plt.plot(bar_positions_group1, bar_positions_group1*0, 'k')


# %%
def sim_Q(Q, total_time, time_step):
    """
    Simulate Markov chain given rate matrix Q, time length and steps
    reading this: https://www.columbia.edu/~ww2040/6711F13/CTMCnotes120413.pdf
    """
    nc = Q.shape[0]
    initial_state = np.random.randint(nc) # uniform to begin with
    states = [initial_state]
    times = [0.0]

    current_state = initial_state
    current_time = 0.0

    while current_time < total_time:
        rate = -Q[current_state, current_state]
        next_time = current_time + np.random.exponential(scale=1/rate)
        if next_time > total_time:
            break

        transition_probabilities = Q[current_state,:]*1#expm(Q * (next_time - current_time))[current_state, :]
        transition_probabilities[current_state] = 0  # remove diagonal
        transition_probabilities /= transition_probabilities.sum()  # Normalize probabilities
        next_state = np.random.choice(len(Q), p=transition_probabilities)
        # print(transition_probabilities)
        
        #####
        # # Generate exponentially distributed time until the next event 
        # rate = abs(Q[current_state, current_state]) 
        # time_to_next_event = np.random.exponential(scale=1/rate) # Update the time and state 
        # time_points.append(time_points[-1] + time_to_next_event) # Determine the next state based on transition probabilities 
        # transition_probs = Q[current_state, :] / rate transition_probs[transition_probs < 0] = 0  # Ensure non-negative probabilities 
        # transition_probs /= np.sum(transition_probs)  # Normalize probabilities to sum to 1 
        # next_state = np.random.choice(len(Q), p=transition_probs) 
        # state_sequence.append(next_state)
        #####
        
        states.append(next_state)
        times.append(next_time)

        current_state = next_state
        current_time = next_time

    return np.array(states), np.array(times)

total_time = 10000
time_step = 1  # check with Peter if this is ok... THIS is OK
states_sim, times_sim = sim_Q(M_inf, total_time, time_step)

# %%
# %% Izhikevich spiking circuit
# lt = 30000
# Ne = 2  # 2 excitation
# Ni = 1  # 1 inhibition
# N = Ne + Ni
# re = np.random.rand(Ne)
# ri = np.random.rand(Ni)
# # a = np.concatenate((0.02*np.ones(Ne) , 0.02+0.08*ri))
# # b = np.concatenate((0.2*np.ones(Ne), 0.25-0.05*ri))
# # c = np.concatenate((-65+15*re**2 , -65*np.ones(Ni)))
# # d = np.concatenate((8-6*re**2 , 2*np.ones(Ni)))
# a = np.concatenate((0.02*np.ones(Ne) , 0.02+0.0*ri))
# b = np.concatenate((0.2*np.ones(Ne), 0.2-0.0*ri))
# c = np.concatenate((-65+15*re**2*0 , -65*np.ones(Ni)))
# d = np.concatenate((8-6*re**2*0 , 8*np.ones(Ni)))
# S = np.concatenate((5*np.random.rand(Ne+Ni, Ne), 10*-np.random.rand(Ne+Ni, Ni)), axis=1)*10
# np.fill_diagonal(S, np.zeros(N))  # remove self-coupling
# v = -65*np.ones(Ne+Ni)
# u = b*v
# dt = 0.5
# tau_s = 10
# I = np.zeros(N)
# firing = []#np.zeros((lt,2))
# for tt in range(lt):
#     # I = np.concatenate((5*np.random.randn(Ne) , 2*np.random.randn(Ni)))*1
#     I = np.concatenate((1*np.random.randn(Ne) , 1*np.random.randn(Ni)))*4
#     fired = np.where(v>=30)[0]
#     firing.append([tt+0*fired, fired])
#     v[fired] = c[fired]
#     u[fired] = u[fired] + d[fired]
#     dI = np.zeros(N)
#     if len(fired)>0:
#         for ii in range(len(fired)):  # identifying synaptic input in one step
#             I = I + S[:,fired[ii]]*1
#             # dI = dI + S[:,fired[ii]]
#     # I = I + dt*(-I/tau_s +  dI)  # filtering
#     v = v + dt*(0.04*v**2 + 5*v + 140 - u + I) #+ I_th # step 0.5 ms
#     v = v + dt*(0.04*v**2 + 5*v + 140 - u + I) # for numerical
#     u = u + a*(b*v - u) # stability

# plt.figure()
# for tt in range(lt):
#     plt.plot(firing[tt][0], firing[tt][1],'k.')
