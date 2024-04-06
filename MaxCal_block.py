# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 13:13:38 2024

@author: kevin
"""

from maxcal_functions import spk2statetime, compute_tauC, param2M, eq_constraint, \
        objective_param, compute_min_isi, sim_Q, MaxCal_D, edge_flux_inf, EP, corr_param,\
        constraint_blocks_3N, LIF_firing

import maxcal_functions

import numpy as np
from matplotlib import pyplot as plt
import itertools
from scipy.optimize import minimize

import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

# %% same as MaxCal_dof but for blocks
###############################################################################
# tau, f, fw, fw_ijk, r, ru, ru_ijk
###############################################################################

# %% network setting
N = 3  # number of neurons
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

# %% M2w(M)
def invf(x):
    output = np.log(x)
    # output = np.log(-1+np.exp(-x))
    return output

def param2weight(param):
    M_inf, pi_inf = param2M(param)
    f1,f2,f3 = M_inf[0,4], M_inf[0,2], M_inf[0,1]
    w12,w13,w21 = invf(M_inf[4,6]/f2), invf(M_inf[4,5]/f3), invf(M_inf[2,6]/f1)
    w23,w32,w31 = invf(M_inf[2,3]/f3), invf(M_inf[1,3]/f2), invf(M_inf[1,5]/f1)
    wij = np.array([w12,w13,w21, w23,w32,w31])
    return wij

true_w = param2weight(param_true)

# %% building constraints -- for three specital points
dofs = num_params*1
dofs_all = nc**2 + nc
target_dof = 7 #dofs + nc

### recordings
kls_inf = np.zeros(target_dof) # for infinite data
kls_fin = np.zeros(target_dof)  # for finite data
r2_inf = np.zeros(target_dof)
r2_fin = np.zeros(target_dof)
ep_inf = np.zeros(target_dof)
ep_fin = np.zeros(target_dof)

### sort according to inifinite data
tau_i, C_i = tau_infinite/total_time, C_infinite/total_time # correct normalization
observations_inf = np.concatenate((tau_i, C_i.reshape(-1)))  # observation from simulation!
tau_f, C_f = tau_finite/total_time, C_finite/total_time # correct normalization
observations_fin = np.concatenate((tau_f, C_f.reshape(-1))) 
nonz_pos = np.where(observations_fin!=0)[0]  # find non-zeros observations

### choosing targets
# target_dof_id = np.array([nc, len(nonz_pos), dofs+nc]) #last one for infinite

P0 = np.ones((nc,nc))  # uniform prior
# P0 = np.random.rand(nc,nc)  # random one
np.fill_diagonal(P0, np.zeros(nc))
np.fill_diagonal(P0, -np.sum(P0,1))
tol = 1e-40

for ii in range(target_dof):  #3
    
    ### designed block of constraints!
    Cp_condition = constraint_blocks_3N(ii+1)
    # print(Cp_condition.sum())
    
    ### run max-cal!
    constraints_inf = ({'type': 'eq', 'fun': eq_constraint, 'args': (observations_inf, Cp_condition)})
    constraints_fin = ({'type': 'eq', 'fun': eq_constraint, 'args': (observations_fin, Cp_condition)})
    bounds = [(.0, 100)]*len(param_true)

    # Perform optimization using SLSQP method
    param0 = np.ones(num_params)*.1 + np.random.rand(num_params)*.0 + param_true*0 
    result_inf = minimize(objective_param, param0, args=(P0), method='SLSQP', constraints=constraints_inf, bounds=bounds,  tol=tol)
    result_fin = minimize(objective_param, param0, args=(P0), method='SLSQP', constraints=constraints_fin, bounds=bounds,  tol=tol)
    
    # computed and record the corresponding KL
    param_inf = result_inf.x
    param_fin = result_fin.x
    kij_inf,pi_ = param2M(param_inf)
    # print(pi_@kij_inf)
    kij_fin,_ = param2M(param_fin)
    kls_inf[ii] = MaxCal_D(kij_inf, P0, param_inf)
    kls_fin[ii] = MaxCal_D(kij_fin, P0, param_fin)
    ep_inf[ii] = EP(kij_inf)
    ep_fin[ii] = EP(kij_fin)
    inf_weight = param2weight(param_inf)
    r2_inf[ii] = corr_param(true_w, inf_weight, '0')
    r2_fin[ii] = corr_param(true_w, inf_weight, '0')
    
    print(ii)    

# %% DEBUGGIING
kij,pi_ = param2M(param_true)
print('true KL: ',MaxCal_D(kij,P0, param_true))

kij,pi_ = param2M(param_inf)
print('truncate KL: ', MaxCal_D(kij,P0, param_inf))

print('true eq-cst: ', eq_constraint(param_true, observations_inf, Cp_condition))
print('truncate eq-cst: ', eq_constraint(param_inf, observations_inf, Cp_condition))

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
x_lab = ['tau','f','fw','fw_ijk','r','ru','ru_ijk']

plt.figure()
plt.subplot(131)
plt.plot( kls_inf, '-o', label='analytic')
# plt.plot(kls_fin, '-o', label='finite-data')
plt.ylabel('KL', fontsize=20); #plt.legend(fontsize=20); 
plt.xticks(np.arange(len(x_lab)), x_lab)

# plt.figure()
plt.subplot(132)
plt.plot(r2_inf, '-o', label='analytic')
# plt.plot(r2_fin, '-o', label='finite-data'); #plt.ylim([0., 1])
plt.ylabel('corr', fontsize=20); #plt.legend(fontsize=20); 
plt.xticks(np.arange(len(x_lab)), x_lab)

# plt.figure()
plt.subplot(133)
plt.plot(ep_inf, '-o', label='analytic')
# plt.plot(ep_fin, '-o', label='finite-data')
plt.legend(fontsize=15); 
plt.ylabel('EP', fontsize=20)
plt.xticks(np.arange(len(x_lab)), x_lab)

# %% saving
# import pickle

# pre_text = 'learning_block_ctmc'
# filename = pre_text + ".pkl"

# # Store variables in a dictionary
# data = {'kls_inf': kls_inf, 'kls_fin': kls_fin, 'r2_inf': r2_inf,\
#         'r2_fin': r2_fin, 'ep_inf': ep_inf, 'ep_fin':ep_fin, 'param_true':param_true}

# # Save variables to a file
# with open(filename, 'wb') as f:
#     pickle.dump(data, f)

# print("Variables saved successfully.")

# %% compare finit and inifinite data KL
x_tik2 = np.arange(len(x_lab))-0.2
x_tik = np.arange(len(x_lab))+0.2
bar_width = 0.35

plt.figure()
plt.subplot(311)
plt.bar(x_tik, kls_inf, label='analytic',width=bar_width)
plt.bar(x_tik2, kls_fin, label='finite-data',width=bar_width)
plt.ylabel('KL', fontsize=20); #plt.ylim([4,5.])#plt.legend(fontsize=20); 

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
plt.xticks(np.arange(len(x_lab)), x_lab)

# %% for LIF
# %%
###############################################################################
# %%
firing = LIF_firing(100000)
lif_weights = np.array([1,1,1,1,-2,-2])*20   # check that this is same as in the function
wind = 150  
N = 3  # number of neurons
num_params = int((N*2**N)) 
nc = 2**N

lt = len(firing)
spk_states, spk_times = spk2statetime(firing, wind)  # embedding states
tau_count, C_count = compute_tauC(spk_states, spk_times, lt=lt)  # emperical measurements
C_base,_ = param2M(np.ones(num_params))
np.fill_diagonal(C_base, np.zeros(nc))
# C_count = C_count+C_base ##### cheeting here!!!
lt = np.sum(tau_count)
tau_spk, C_spk = tau_count/lt, C_count/lt # correct normalization
observations_spk = np.concatenate((tau_spk, C_spk.reshape(-1))) 

# %% try retina! # run from retina_maxcal script
# C_all_ = C_all#+C_base
# new_lt = np.sum(tau_all)
# tau_, C_ = tau_all/new_lt, C_all_/new_lt*1 # time_norm  #lt/reps # correct normalization
# observations_spk = np.concatenate((tau_, C_.reshape(-1))) 
### use the last one for "ground truth"

# %%
target_dof = 7 #dofs + nc

### recordings
kls_spk = np.zeros(target_dof) # for infinite data
r2_spk = np.zeros(target_dof)
sign_spk = np.zeros(target_dof)
ep_spk = np.zeros(target_dof)

P0 = np.ones((nc,nc))  # uniform prior
np.fill_diagonal(P0, np.zeros(nc))
np.fill_diagonal(P0, -np.sum(P0,1))

for ii in range(target_dof):
    
    ### designed block of constraints!
    Cp_condition = constraint_blocks_3N(ii+1)
    
    ### run max-cal!
    constraints_spk = ({'type': 'eq', 'fun': eq_constraint, 'args': (observations_spk, Cp_condition)})
    bounds = [(.0, 100)]*num_params

    # Perform optimization using SLSQP method
    param0 = np.ones(num_params)*.1
    result_spk = minimize(objective_param, param0, args=(P0), method='SLSQP', constraints=constraints_spk, bounds=bounds)
    
    # computed and record the corresponding KL
    param_spk = result_spk.x
    kij_spk,_ = param2M(param_spk)
    kls_spk[ii] = MaxCal_D(kij_spk, P0, param_spk)
    ep_spk[ii] = EP(kij_spk)
    M_inf, pi_inf = param2M(param_spk)
    f1,f2,f3 = M_inf[0,4], M_inf[0,2], M_inf[0,1]
    w12,w13,w21 = invf(M_inf[4,6]/f2), invf(M_inf[4,5]/f3), invf(M_inf[2,6]/f1)
    w23,w32,w31 = invf(M_inf[2,3]/f3), invf(M_inf[1,3]/f2), invf(M_inf[1,5]/f1)
    infer_wij = np.array([w12,w13,w21, w23,w32,w31])
    r2_spk[ii] = corr_param(lif_weights, infer_wij, '0')  # bin correlation of weights
    sign_spk[ii] = corr_param(lif_weights, infer_wij, 'binary')
    
    print(ii)  
    print('truncate eq-cst: ', eq_constraint(param_spk, observations_spk, Cp_condition))

# %% hijack
M_inf = (C_spk/tau_spk[:,None])
f1,f2,f3 = M_inf[0,4], M_inf[0,2], M_inf[0,1]
w12,w13,w21 = invf(M_inf[4,6]/f2), invf(M_inf[4,5]/f3), invf(M_inf[2,6]/f1)
w23,w32,w31 = invf(M_inf[2,3]/f3), invf(M_inf[1,3]/f2), invf(M_inf[1,5]/f1)
hijack_wij = np.array([w12,w13,w21, w23,w32,w31])
corr_param(lif_weights, hijack_wij, '0')

# %%
x_lab = ['tau','f','fw','fw_ijk','r','ru','ru_ijk']
x_tik = np.arange(len(x_lab))
plt.figure()
plt.subplot(311)
plt.plot(x_tik, kls_spk ,'-o')
plt.ylabel('KL', fontsize=20); #plt.ylim([4,5.])#plt.legend(fontsize=20); 
plt.title('LIF', fontsize=20)

# plt.figure()
plt.subplot(312)
plt.plot(x_tik, r2_spk, '-o', label='corr')
plt.plot(x_tik, sign_spk,'-o', label='signed corr')
plt.ylabel('corr', fontsize=20); plt.legend(fontsize=10); 

# plt.figure()
plt.subplot(313)
plt.plot(x_tik, ep_spk,'-o')
plt.ylabel('EP', fontsize=20)
plt.xticks(np.arange(len(x_lab)), x_lab)

# %% saving
# import pickle

# pre_text = 'learning_block_LIF'
# filename = pre_text + ".pkl"

# # Store variables in a dictionary
# data = {'kls_spk': kls_spk, 'r2_spk': r2_spk, 'sign_spk': sign_spk,\
#         'ep_spk': ep_spk, 'firing': firing, 'wind':wind}

# # Save variables to a file
# with open(filename, 'wb') as f:
#     pickle.dump(data, f)

# print("Variables saved successfully.")