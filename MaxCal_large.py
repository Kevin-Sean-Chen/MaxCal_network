# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 21:33:04 2024

@author: kevin
"""
from maxcal_functions import spk2statetime, compute_tauC, param2M, eq_constraint, \
    objective_param, compute_min_isi, sim_Q, MaxCal_D, edge_flux_inf, EP, corr_param

import maxcal_functions

import numpy as np
from matplotlib import pyplot as plt
import itertools
from scipy.optimize import minimize

import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

# %% same as MaxCal_dof
# but for larger network
# subsampling interesting dof to speed up the process

# %% algotrithm to infer the DOF of transition matrix via MaxCal
# generate spike trains
# the setup is the general N*2**N dof matrix
# rank the order of state sylables
# scanning through and appending them to make DOF-KL curve for finite data
# compare KL, correlation, and EP for analytic and finite-data inferences

# %% network setting
N = 5  # number of neurons
num_params = int((N*2**N))  # number of parameters in model without refractory assumption
spins = [0,1]  # binary patterns
combinations = list(itertools.product(spins, repeat=N))  # possible configurations
nc = 2**N  # number of total states of 3-neuron
param_true = np.random.rand(num_params)*1  # use rand for now...

# %% network simulation
total_time = 500
time_step = 1  # check with Peter if this is ok... THIS is OK
M,pi_ss = param2M(param_true, N, combinations)
states, times = sim_Q(M, total_time, time_step)

# %% counting and ranking analysis 
#### for simulation
tau_finite, C_finite = compute_tauC(states, times, nc, combinations)  # emperical measurements

# %% for infinite data
#### for ground-truth (infinite data!)
tau_infinite = pi_ss*total_time
flux_ij_true = edge_flux_inf(param_true, N, combinations)
C_infinite = flux_ij_true* total_time

# %% building constraints -- for three specital points
dofs = num_params*1
dofs_all = nc**2 + nc
target_dof = 3 #dofs + nc

### recordings
kls_inf = np.zeros(target_dof) # for infinite data
kls_fin = np.zeros(target_dof)  # for finite data
r2_inf = np.zeros(target_dof)
r2_fin = np.zeros(target_dof)
ep_inf = np.zeros(target_dof)
ep_fin = np.zeros(target_dof)

### sort according to inifinite data
Cp_condition = np.zeros(dofs_all)
rank_tau = np.argsort(tau_infinite)[::-1]  # ranking occupency
rank_C = np.argsort(C_infinite.reshape(-1))[::-1]  # ranking transition
tau_i, C_i = tau_infinite/total_time, C_infinite/total_time # correct normalization
observations_inf = np.concatenate((tau_i, C_i.reshape(-1)))  # observation from simulation!
tau_f, C_f = tau_finite/total_time, C_finite/total_time # correct normalization
observations_fin = np.concatenate((tau_f, C_f.reshape(-1))) 
nonz_pos = np.where(observations_fin!=0)[0]  # find non-zeros observations

### choosing targets
target_dof_id = np.array([nc, len(nonz_pos), dofs+nc]) #last one for infinite

P0 = np.ones((nc,nc))  # uniform prior
np.fill_diagonal(P0, np.zeros(nc))
np.fill_diagonal(P0, -np.sum(P0,1))

for ii in range(len(target_dof_id)):
    
    if ii==0:
        Cp_condition[rank_tau] = np.ones(len(rank_tau))
    
    elif ii==1:
        Cp_condition[nonz_pos] = np.ones(len(nonz_pos))
        
    elif ii==2:
        Cp_condition = np.ones(dofs_all)
        
    ### work on tau first
    # if ii<nc-1:
    #     Cp_condition[rank_tau[ii]] = 1
    # else:
    #     Cp_condition[rank_C[ii-(nc-1)]+(nc)] = 1
          
    ### run max-cal!
    constraints_inf = ({'type': 'eq', 'fun': eq_constraint, 'args': (observations_inf, Cp_condition)})
    constraints_fin = ({'type': 'eq', 'fun': eq_constraint, 'args': (observations_fin, Cp_condition)})
    bounds = [(.0, 100)]*len(param_true)

    # Perform optimization using SLSQP method
    param0 = np.ones(num_params)*.1 + np.random.rand(num_params)*0 + param_true*0
    result_inf = minimize(objective_param, param0, args=(P0), method='SLSQP', constraints=constraints_inf, bounds=bounds)
    result_fin = minimize(objective_param, param0, args=(P0), method='SLSQP', constraints=constraints_fin, bounds=bounds)
    
    # computed and record the corresponding KL
    param_inf = result_inf.x
    param_fin = result_fin.x
    kij_inf,_ = param2M(param_inf)
    kij_fin,_ = param2M(param_fin)
    kls_inf[ii] = MaxCal_D(kij_inf, P0, param_inf)
    kls_fin[ii] = MaxCal_D(kij_fin, P0, param_fin)
    ep_inf[ii] = EP(kij_inf)
    ep_fin[ii] = EP(kij_fin)
    r2_inf[ii] = corr_param(param_true, param_inf, '0')
    r2_fin[ii] = corr_param(param_true, param_fin, '0')
    
    print(ii)    

# %% compare 
M_inf, pi_inf = param2M(param_fin, N, combinations)
flux_ij = edge_flux_inf(param_fin, N, combinations)
plt.figure()
plt.subplot(211)
plt.plot(flux_ij_true.reshape(-1), flux_ij.reshape(-1),'.')
plt.xlabel('C analytic', fontsize=20); plt.ylabel('C finite', fontsize=20)

plt.subplot(212)
plt.plot(pi_ss, pi_inf, '.')
plt.xlabel('tau analytic', fontsize=20); plt.ylabel('tau finite', fontsize=20)

# %% line plots
x_lab = ['tau','non-zero','all']

plt.figure()
plt.subplot(131)
plt.plot( kls_inf, '-o', label='analytic')
plt.plot(kls_fin, '-o', label='finite-data')
plt.ylabel('KL', fontsize=20); #plt.legend(fontsize=20); 
plt.xticks(np.arange(3), x_lab)

# plt.figure()
plt.subplot(132)
plt.plot(r2_inf, '-o', label='analytic')
plt.plot(r2_fin, '-o', label='finite-data')
plt.ylabel('corr', fontsize=20); #plt.legend(fontsize=20); 
plt.xticks(np.arange(3), x_lab)

# plt.figure()
plt.subplot(133)
plt.plot(ep_inf, '-o', label='analytic')
plt.plot(ep_fin, '-o', label='finite-data')
plt.legend(fontsize=15); 
plt.ylabel('EP', fontsize=20)
plt.xticks(np.arange(3), x_lab)

# %% compare finit and inifinite data KL
x_tik2 = np.arange(3)-0.2
x_tik = np.arange(3)+0.2
x_lab = ['tau','non-zero','all']
bar_width = 0.35

plt.figure()
plt.subplot(311)
plt.bar(x_tik, kls_inf, label='analytic',width=bar_width)
plt.bar(x_tik2, kls_fin, label='finite-data',width=bar_width)
plt.ylabel('KL', fontsize=20); #plt.legend(fontsize=20); 

# plt.figure()
plt.subplot(312)
plt.bar(x_tik, r2_inf, label='analytic',width=bar_width)
plt.bar(x_tik2, r2_fin, label='finite-data',width=bar_width)
plt.ylabel('corr', fontsize=20); #plt.legend(fontsize=20); 

# plt.figure()
plt.subplot(313)
plt.bar(x_tik, ep_inf, label='analytic',width=bar_width)
plt.bar(x_tik2, ep_fin, label='finite-data',width=bar_width)
plt.legend(fontsize=20); plt.ylabel('EP', fontsize=20)
plt.xticks(np.arange(3), x_lab)

# %%
print('infinite diff: ', np.sum(C_i,0) - np.sum(C_i,1))
print('finite diff: ', np.sum(C_f,0) - np.sum(C_f,1))
