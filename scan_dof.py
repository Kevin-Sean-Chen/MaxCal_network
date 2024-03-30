# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:21:55 2024

@author: kevin
"""

from maxcal_functions import spk2statetime, compute_tauC, param2M, eq_constraint, objective_param,\
                            compute_min_isi, sim_Q, edge_flux_inf, MaxCal_D, EP, corr_param

import numpy as np
from matplotlib import pyplot as plt
import itertools
from scipy.optimize import minimize
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

# %% baiscs
rep_finite = 10  #10 repeat finit oprimization runs
N = 3
nc = 2**3
num_params = int((N*2**N)) 
spins = [0,1]  # binary patterns
combinations = list(itertools.product(spins, repeat=N))  # possible configurations
param_true = np.random.rand(num_params)*1  # use rand for now...
total_time = 50
time_step = 1

# %% for inifinite
#### for ground-truth (infinite data!)
M,pi_ss = param2M(param_true)
tau_infinite = pi_ss*total_time
flux_ij_true = edge_flux_inf(param_true)
C_infinite = flux_ij_true* total_time

# %% building constraints
dofs = num_params*1
dofs_all = nc**2 + nc
target_dof = dofs + nc

### recordings
kls_inf = np.zeros(target_dof) # for infinite data
kls_fin = np.zeros((target_dof, rep_finite))  # for finite data
r2_inf = np.zeros(target_dof)
r2_fin = np.zeros((target_dof, rep_finite))
ep_inf = np.zeros(target_dof)
ep_fin = np.zeros((target_dof, rep_finite))

### sort according to inifinite data
rank_tau = np.argsort(tau_infinite)[::-1]  # ranking occupency
rank_C = np.argsort(C_infinite.reshape(-1))[::-1]  # ranking transition
tau_i, C_i = tau_infinite/total_time, C_infinite/total_time # correct normalization
observations_inf = np.concatenate((tau_i, C_i.reshape(-1)))  # observation from simulation!

P0 = np.ones((nc,nc))  # uniform prior
np.fill_diagonal(P0, np.zeros(nc))
np.fill_diagonal(P0, -np.sum(P0,1))

def run_dof(observations):
    """
    given finite data, run dof curve
    return KL, correlation, entropy, and parameters
    """
    kls = np.zeros(target_dof)
    r2s = np.zeros(target_dof)
    eps = np.zeros(target_dof)
    Cp_condition = np.zeros(dofs_all)
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
        bounds = [(.0, 100)]*len(param_true)

        # Perform optimization using SLSQP method
        param0 = np.ones(num_params)*.1 + np.random.rand(num_params)*0 + param_true*0
        result = minimize(objective_param, param0, args=(P0), method='SLSQP', constraints=constraints, bounds=bounds)
        
        # computed and record the corresponding KL
        param = result.x
        kij_inf,_ = param2M(param)
        kls[ii] = MaxCal_D(kij_inf, P0, param)
        eps[ii] = EP(kij_inf)
        r2s[ii] = corr_param(param_true, param, '0')  # non-biniary version!
        
        print(ii)    
        ii = ii+1
    
    return kls, r2s, eps, param  # return KL, corr, entropy, and parameters

# %% for inifinite data
kls_inf, r2_inf, ep_inf, param_inf = run_dof(observations_inf)

# %% scanning finite data repeats

for rr in range(rep_finite):
    #### for simulation
    states, times = sim_Q(M, total_time, time_step)
    tau_finite, C_finite = compute_tauC(states, times)  # emperical measurements
    tau_f, C_f = tau_finite/total_time, C_finite/total_time # correct normalization
    observations_fin = np.concatenate((tau_f, C_f.reshape(-1))) 
    
    ### run and record
    kls, r2s, eps, param = run_dof(observations_fin)
    kls_fin[:,rr] = kls
    r2_fin[:,rr] = r2s
    ep_fin[:,rr] = eps
    print('loop: ', rr)
    
# %%
plt.figure()
plt.plot(kls_fin,'-o', alpha=0.2, color='k')
plt.plot(kls_inf,'-o', label='analytic')
plt.legend(fontsize=20); plt.ylabel('KL', fontsize=20)
# plt.savefig('KL.pdf')

plt.figure()
plt.plot(r2_fin,'-o',  alpha=0.2, color='k')
plt.plot(r2_inf,'-o', label='analytic')
plt.legend(fontsize=20); plt.ylabel('corr', fontsize=20)
# plt.savefig('R2.pdf')

plt.figure()
plt.plot(ep_fin,'-o',  alpha=0.2, color='k')
plt.plot(ep_inf,'-o', label='analytic')
plt.legend(fontsize=20); plt.ylabel('EP', fontsize=20)
# plt.savefig('EP.pdf')

    
# %%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# scan for finit data versus KL rate
# %%
data_len = np.array([25,50,100,200,400,800])
kl_dist = np.zeros(len(data_len))
Cp_condition = np.ones(dofs_all)

for dd in range(len(data_len)):
    #### for simulation
    time_i = data_len[dd]
    states, times = sim_Q(M, time_i, time_step)
    tau_finite, C_finite = compute_tauC(states, times)  # emperical measurements
    tau_f, C_f = tau_finite/time_i, C_finite/time_i # correct normalization
    observations_fin = np.concatenate((tau_f, C_f.reshape(-1))) 
    
    ### run max-cal!
    constraints = ({'type': 'eq', 'fun': eq_constraint, 'args': (observations_fin, Cp_condition)})
    bounds = [(.0, 100)]*len(param_true)

    # Perform optimization using SLSQP method
    param0 = np.ones(num_params)*.1 + np.random.rand(num_params)*0 + param_true*0
    result = minimize(objective_param, param0, args=(P0), method='SLSQP', constraints=constraints, bounds=bounds)
    
    # computed and record the corresponding KL
    param = result.x
    kij_fin,_ = param2M(param)
    kl_dist[dd] = MaxCal_D(kij_fin, M, param)
    
# %%
plt.figure()
plt.plot(data_len, kl_dist,'-o')
plt.xlabel('data length', fontsize=20)
plt.ylabel('KL(infer|true)', fontsize=20)
# plt.savefig('KL_scaling.pdf')
