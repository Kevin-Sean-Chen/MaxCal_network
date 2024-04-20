# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 23:40:30 2024

@author: kevin
"""

from maxcal_functions import spk2statetime, compute_tauC, param2M, eq_constraint, \
                            MaxCal_D, objective_param, compute_min_isi, corr_param, sign_corr, P_frw_ctmc, C_P,\
                            word_id

import scipy.io
import numpy as np
from matplotlib import pyplot as plt
import itertools
from scipy.optimize import minimize
from scipy.stats import pearsonr
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

# %% loading retina and stimuli
spk_data = scipy.io.loadmat('C:/Users/kevin/Downloads/white_noise_checkerboard/data.mat')
stim_data = scipy.io.loadmat('C:/Users/kevin/Downloads/white_noise_checkerboard/aux_data/stim.mat')
sta_data = scipy.io.loadmat('C:/Users/kevin/Downloads/white_noise_checkerboard/aux_data/stas.mat')

spk_data = spk_data['data']['spike_times'][0][0][0]['all_spike_times'][0][0]  # all spike times!
peak_times = stim_data['peak_times']
frames = stim_data['frames']
stas = sta_data['all_stas']

# %% test STA
nid = 49 
delay = 15
# def find_closest_element_index(spk_t, peak_times=peak_times):
#     abs_diff = np.abs(peak_times - spk_t)
#     idx = np.argmin(abs_diff)
#     return idx

spk_i = spk_data[nid]
lt = len(spk_i)
sta = np.zeros((40,40))
for tt in range(lt):
    spk_t = spk_i[tt][0]
    abs_diff = np.abs(peak_times - spk_t)
    frame_id,diff = np.argmin(abs_diff), np.min(abs_diff)
    if diff<delay:
        # frame_id = find_closest_element_index(spk_i[tt][0])
        sta += frames[:,:,max(frame_id-delay,0)].squeeze()
sta = sta/lt

plt.figure()
plt.imshow(sta)

# %% load STA
nid = 140 # 5(on), 30(off), 40(off), 32(on), 49(on), 70 (off), 120 (off), 140(on), 154 (on), 145(off)
delay = 20
plt.figure()
plt.imshow(stas[:,:,delay,nid])

# %% all candidates
N = 3
nids = np.array([5,30,40])  # 5,30,40
nids = np.array([30,5,70])  # 30, 5, 70
nids = np.array([49,120,32])

plt.figure()
for ii in range(N):
    plt.subplot(1,3,ii+1);plt.imshow(stas[:,:,delay,nids[ii]].squeeze())
    
# %%
###############################################################################
# %% building CTMC!!
###############################################################################
# %% loop across trial and time and neurons
dt = 0.1 # 1ms
lt = 220000  #100s #int(50000/dt)
    
firing = []
firing.append((np.array([]), np.array([])))

for tt in range(lt-1):  # time
    spike_indices = np.array([])
    
    for nn in range(N):  # neurons
        spkt = spk_data[nids[nn]]  # ith neuron spike train
        ### dt method
        spkt = spkt-spkt[0][0]
        find_spk = np.where((spkt*dt > tt) & (spkt*dt <= tt+1))[0]   #changing here!
        
        ### increment method
        # spkt = spkt-spkt[0][0]
        # find_spk = np.where((spkt*dt > tt*dt) & (spkt*dt <= tt*dt+dt))[0]
        
        if len(find_spk)!=0:
            spike_indices = np.append(spike_indices, int(nn))
    firing.append([tt+0*spike_indices, spike_indices])  ## constuct firing tuple
    

# %% some tests!!
window = 50 #int(20/dt)  # .1ms window
spk_state_all = []
spk_time_all = []
spk_states, spk_times = spk2statetime(firing, window, lt=lt)
tau_all, C_all = compute_tauC(spk_states, spk_times, lt=lt)

print(tau_all)
print(C_all)

# %% full-dof for checking performance
num_params = int((N*2**N))  # number of parameters in model without refractory assumption
nc = 2**N  # number of total states of 3-neuron
dofs = num_params*1
dofs_all = nc**2 + nc
target_dof = dofs + nc

new_lt = np.sum(tau_all)
C_base,_ = param2M(np.ones(num_params))
np.fill_diagonal(C_base, np.zeros(nc))
C_all = C_all+C_base
tau_, C_ = tau_all/new_lt, C_all/new_lt*1 # time_norm  #lt/reps # correct normalization
observations = np.concatenate((tau_, C_.reshape(-1)))  # observation from simulation!

def log_obk(param, kij0):
    param_ = np.exp(param)
    obj = objective_param(param_, kij0)
    return obj

def log_const(param, observations, Cp_condition):
    param_ = np.exp(param)
    Pxy = P_frw_ctmc(param_)
    cp = C_P(Pxy, observations, param_, Cp_condition)
    return cp #0.5*np.sum(cp**2)

P0 = np.ones((nc,nc))  # uniform prior
np.fill_diagonal(P0, np.zeros(nc))
np.fill_diagonal(P0, -np.sum(P0,1))
Cp_condition = np.ones(dofs_all)
### run max-cal!
constraints = ({'type': 'eq', 'fun': eq_constraint, 'args': (observations, Cp_condition)})
bounds = [(.0, 100)]*num_params
# Perform optimization using SLSQP method
param0 = np.ones(num_params)*.01 + np.random.rand(num_params)*0.0

result = minimize(objective_param, param0, args=(P0), method='SLSQP', constraints=constraints, bounds=bounds)
param_temp = result.x

# %%
spins = [0,1]  # binary patterns
combinations = list(itertools.product(spins, repeat=N))
M_inf, pi_inf = param2M(param_temp, N, combinations)  #dof_cut
np.fill_diagonal(M_inf, np.zeros(nc))
############### hijack
M_inf = (C_/tau_[:,None])
##############
f1,f2,f3 = M_inf[0,4], M_inf[0,2], M_inf[0,1]
w12,w13,w21 = np.log(M_inf[4,6]/f2), np.log(M_inf[4,5]/f3), np.log(M_inf[2,6]/f1)
w23,w32,w31 = np.log(M_inf[2,3]/f3), np.log(M_inf[1,3]/f2), np.log(M_inf[1,5]/f1)
inf_w = np.array([w12,w13,w21,w23,w32,w31])

categories = ['w12','w13','w21','w23','w32','w31']
plt.figure()
plt.bar(np.arange(6), inf_w)
plt.xticks(np.arange(len(categories)), categories)
plt.title('retina (on-off-off)', fontsize=20)
# plt.ylim([-4.5, 1])
# plt.savefig('retina_120ms2.pdf')
