# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 18:48:50 2024

@author: kevin
"""

from maxcal_functions import spk2statetime, compute_tauC, param2M, eq_constraint, \
                            MaxCal_D, objective_param, compute_min_isi

import scipy.io
import numpy as np
from matplotlib import pyplot as plt
import itertools
from scipy.optimize import minimize
from scipy.stats import pearsonr
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

# %% loading retina!
mat_data = scipy.io.loadmat('C:/Users/kevin/Downloads/Data_processed.mat')
dataset = 2  # 0-natural, 1-Brownian, 2-repeats
nid = 3  # neuron example
spk_times = mat_data['spike_times'][0][dataset][0][nid].squeeze()
n_neurons = len(mat_data['spike_times'][0][dataset][0])
spk_data = mat_data['spike_times'][0][dataset][0]  # extract to use

plt.figure()
for nn in range(n_neurons):
    spki = spk_data[nn].squeeze()
    plt.plot(spki, np.ones(len(spki))+nn,'k.')
    
# %%
N = 3
temp = np.arange(0,n_neurons)
nids = np.random.choice(temp, size=N, replace=False)

# check isi
# the data is spike timing with 10kHz sampling, so .1ms time resolution
minisi = np.zeros(N)
max_spk_time = np.zeros(N)
for ii in range(N):
    isis = spk_data[nids[ii]].squeeze()
    max_spk_time[ii] = np.max(isis)
    isis = np.diff(isis)
    minisi[ii] = np.min(isis)
    
minisi_ = np.min(minisi)
max_lt = np.max(max_spk_time)
print(minisi_)
print(max_lt)

# %%
dt = 0.1
lt = int(max_lt/dt)
firing = []
firing.append((np.array([]), np.array([])))

for tt in range(lt):
    spike_indices = np.array([])
    for nn in range(N):
        temp_spk = spk_data[nids[nn]].squeeze()
        find_spk = np.where((temp_spk > tt*dt) & (temp_spk <= tt*dt+dt))[0]
        if len(find_spk)!=0:
            spike_indices = np.append(spike_indices, int(nn))
    firing.append([tt+0*spike_indices, spike_indices])  ## constuct firing tuple
    
# %% some tests!!
spk_states, spk_times = spk2statetime(firing, 200)  # embedding states
tau,C = compute_tauC(spk_states, spk_times)

plt.figure()
plt.plot(spk_states,'-o')
print(tau)
print(C)

# %% building constraints
num_params = int((N*2**N))  # number of parameters in model without refractory assumption
nc = 2**N  # number of total states of 3-neuron
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

# %%
M_inf, pi_inf = param2M(param_temp)
f1,f2,f3 = M_inf[0,4], M_inf[0,2], M_inf[0,1]
w12,w13,w21 = np.log(M_inf[4,6]/f2), np.log(M_inf[4,5]/f3), np.log(M_inf[2,6]/f1)
w23,w32,w31 = np.log(M_inf[2,3]/f3), np.log(M_inf[1,3]/f2), np.log(M_inf[1,5]/f1)
inf_w = np.array([w12,w13,w21,w23,w32,w31])

categories = ['w12','w13','w21','w23','w32','w31']
plt.figure()
plt.bar(np.arange(6), inf_w)
plt.xticks(np.arange(len(categories)), categories)