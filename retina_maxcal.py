# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 18:48:50 2024

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

# %% loading retina!
mat_data = scipy.io.loadmat('C:/Users/kevin/Downloads/Data_processed.mat')
dataset = 1  # 0-natural, 1-Brownian, 2-repeats
nid = 3  # neuron example
reps = 62 #len(mat_data['spike_times'][0][dataset][0])  #62, 50
spk_data = mat_data['spike_times'][0][dataset][0]  # extract timing
spk_ids = mat_data['cell_IDs'][0][dataset][0]  # extract cell ID

plt.figure()
for nn in range(reps):
    spkt = spk_data[nn].squeeze()
    spki = spk_ids[nn].squeeze()
    pos = np.where(spki==nid)[0]
    plt.plot(spkt[pos], np.ones(len(pos))+nn,'k.')

# %%
N = 3
nc = 2**N
spins = [0,1]  # binary patterns
combinations = list(itertools.product(spins, repeat=N))  # possible configurations
cell_ids = np.unique(spk_ids[0])
nids = np.random.choice(cell_ids, size=N, replace=False)  # random select three neurons
# nids = np.array([16, 40, 31])
# nids = np.array([24, 13, 15])  # ,,10
# nids = np.array([1,13,27])
# nids = np.array([50, 31, 13]) ###
### array([21,  2, 27], dtype=uint8)

nids = np.array([3, 34, 13])  #3,34,13  # for figure 7
# nids = np.array([40,1,31])   # for SI plot

# %% plot three neuron for Peter
trial_id = 0  
plt.figure()
for nn in range(3):
    # ni = nn*1 #
    ni = nids[nn]
    spkt = spk_data[trial_id].squeeze()
    spki = spk_ids[trial_id].squeeze()
    pos = np.where(spki==ni)[0]
    plt.plot(spkt[pos], np.ones(len(pos))+nn,'k.')
    plt.xlim([0,15000])
    # plt.xlim([8000,9000])
    # plt.xlim([4000,5000])
    # plt.xlim([3200,3700])
# plt.savefig('retina_spk_exp.pdf')

# %%
# check isi
# the data is spike timing with 10kHz sampling, so .1ms time resolution
# minisi = np.zeros((N,reps))
# max_spk_time = np.zeros((N,reps))
minisi = []; max_spk_time = []
for nn in range(N):
    for rr in range(reps):
        spkt = spk_data[rr].squeeze()
        spki = spk_ids[rr].squeeze()
        pos = np.where(spki==nids[nn])[0]
        spks = spkt[pos]
        if len(spks)>2:
            isis = np.diff(spks)
            max_spk_time.append(np.max(spks))
            minisi.append(np.min(isis))
    
minisi_ = np.min(minisi)
max_lt = np.max(max_spk_time)
print(minisi_)
print(max_lt)

# %% loop across trial and time and neurons
dt = 1 #.1
lt = int(10000/dt) #int(max_lt/dt)
firing_s = []  # across repeats!

for dd in range(2,3):   #### try all data!!
    spk_data = mat_data['spike_times'][0][dd][0]  # extract timing
    spk_ids = mat_data['cell_IDs'][0][dd][0]  # extract cell ID
    
    
    for rr in range(reps):  # repeats
        firing = []
        firing.append((np.array([]), np.array([])))
        
        for tt in range(lt):  # time
            spike_indices = np.array([])
            
            for nn in range(N):  # neurons
                spkt = spk_data[rr].squeeze()
                spki = spk_ids[rr].squeeze()
                pos = np.where(spki==nids[nn])[0]
                spks = spkt[pos]
                find_spk = np.where((spks > tt*dt) & (spks <= tt*dt+dt))[0]
                if len(find_spk)!=0:
                    spike_indices = np.append(spike_indices, int(nn))
            firing.append([tt+0*spike_indices, spike_indices])  ## constuct firing tuple
    
        firing_s.append(firing)

# %% checking raster
# plt.figure()
# temp = firing_s[0]
# for ii in range(len(temp)):
#     if len(temp[ii][1])>0:
#         plt.plot(ii, temp[ii][1], 'k.')
        
# %% some tests!!
window = int(20/dt)  # .1ms window
spk_state_all = []
spk_time_all = []
spk_states, spk_times = spk2statetime(firing_s[0], window, lt=lt)
tau_all, C_all = compute_tauC(spk_states, spk_times, lt=lt)
# (states, times, nc=nc, combinations=combinations, lt=None)

for rr in range(1, len(firing_s)):
    spk_states, spk_times = spk2statetime(firing_s[rr], window, lt=lt)  # embedding states
    spk_state_all.append(spk_states)
    spk_time_all.append(spk_times)
    
    tau,C = compute_tauC(spk_states, spk_times, lt=lt)
    tau_all += tau
    C_all += C

plt.figure()
plt.plot(spk_states,'-o')
print(tau_all)
print(C_all)

# %% building constraints
# num_params = int((N*2**N))  # number of parameters in model without refractory assumption
# nc = 2**N  # number of total states of 3-neuron
# dofs = num_params*1
# dofs_all = nc**2 + nc
# target_dof = dofs + nc
# kls = np.zeros(target_dof) # measure KL
# Cp_condition = np.zeros(dofs_all)  # mask... problem: this is ncxnc not dof???
# rank_tau = np.argsort(tau_all)[::-1]  # ranking occupency
# rank_C = np.argsort(C_all.reshape(-1))[::-1]  # ranking transition
# # time_norm = len(np.concatenate(spk_time_all))
# tau_, C_ = tau_all/lt/reps, C_all/lt/reps# time_norm  #lt/reps # correct normalization
# # tau_, C_ = tau/lt/1, C/lt/1 # correct normalization
# observations = np.concatenate((tau_, C_.reshape(-1)))  # observation from simulation!
# # observations = np.concatenate((C_.reshape(-1), tau_))

# P0 = np.ones((nc,nc))  # uniform prior
# np.fill_diagonal(P0, np.zeros(nc))
# np.fill_diagonal(P0, -np.sum(P0,1))
# ii = 0
# ### scan through all dof
# while ii < target_dof:
#     ### work on tau first
#     if ii<nc-1:
#         Cp_condition[rank_tau[ii]] = 1
#     else:
#         Cp_condition[rank_C[ii-(nc-1)]+(nc)] = 1
    
#     ### run max-cal!
#     constraints = ({'type': 'eq', 'fun': eq_constraint, 'args': (observations, Cp_condition)})
#     bounds = [(.0, 100)]*num_params

#     # Perform optimization using SLSQP method
#     param0 = np.ones(num_params)*.1 + np.random.rand(num_params)*0.0
#     result = minimize(objective_param, param0, args=(P0), method='SLSQP', constraints=constraints, bounds=bounds)
    
#     # computed and record the corresponding KL
#     param_temp = result.x
#     Pyx,_ = param2M(param_temp)
#     kls[ii] = MaxCal_D(Pyx, P0, param_temp)
#     print(ii)    
#     ii = ii+1

# %% full-dof for checking performance
num_params = int((N*2**N))  # number of parameters in model without refractory assumption
nc = 2**N  # number of total states of 3-neuron
dofs = num_params*1
dofs_all = nc**2 + nc
target_dof = dofs + nc
kls = np.zeros(target_dof) # measure KL
Cp_condition = np.zeros(dofs_all)  # mask... problem: this is ncxnc not dof???
rank_tau = np.argsort(tau_all)[::-1]  # ranking occupency
rank_C = np.argsort(C_all.reshape(-1))[::-1]  # ranking transition
# time_norm = len(np.concatenate(spk_time_all))
new_lt = np.sum(tau_all)
C_base,_ = param2M(np.ones(num_params))
np.fill_diagonal(C_base, np.zeros(nc))
# tau_all = tau_all+1  # remove zeros, like C_base logic 
# C_all = C_all+C_base
tau_, C_ = tau_all/new_lt, C_all/new_lt*1 # time_norm  #lt/reps # correct normalization
# tau_, C_ = tau/lt/1, C/lt/1 # correct normalization
observations = np.concatenate((tau_, C_.reshape(-1)))  # observation from simulation!
# observations = np.concatenate((C_.reshape(-1), tau_))


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
# bounds = [(-100, 100)]*num_params
# Perform optimization using SLSQP method
param0 = np.ones(num_params)*.01 + np.random.rand(num_params)*0.0
# good_init = (C_/tau_[:,None]).reshape(-1)
# pos = np.where(good_init!=0)[0]
# param0 = good_init[pos]

result = minimize(objective_param, param0, args=(P0), method='SLSQP', constraints=constraints, bounds=bounds)
# result = minimize(objective_param, param0, args=(P0), constraints=constraints)
# result = minimize(log_obk, param0, args=(P0), method='SLSQP', constraints=constraints, bounds=bounds)
# result = minimize(log_obk, param0, args=(P0), constraints=constraints, bounds=bounds)
# computed and record the corresponding KL
# param_temp = np.exp(result.x)
param_temp = result.x

# %%
# plt.figure()
# plt.plot(np.log(kls[:]),'-o')
# plt.xlabel('ranked dof', fontsize=20)
# plt.ylabel('KL', fontsize=20)
# plt.title('retina', fontsize=20)
# # plt.savefig('retina_KL.pdf')
# # plt.ylim([0,10])

# %% test cutoff
# dof_cut = 32
# ii = 0
# while ii < dof_cut:
#     if ii<nc-1:
#         Cp_condition[rank_tau[ii]] = 1
#     else:
#         Cp_condition[rank_C[ii-(nc-1)]+(nc)] = 1
#     ii+=1
# constraints = ({'type': 'eq', 'fun': eq_constraint, 'args': (observations, Cp_condition)})
# bounds = [(.0, 100)]*num_params
# param0 = np.ones(num_params)*.1 + np.random.rand(num_params)*0.0
# result = minimize(objective_param, param0, args=(P0), method='SLSQP', constraints=constraints, bounds=bounds)
# param_cut = result.x

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
plt.title('retina (B=100ms)', fontsize=20)
# plt.savefig('retina_wij_20.pdf')

# %% check biophysical correspondence
### (0, fi), (wji, fiewji ), (wki, fiewki ), (wji + wki, fiewjk,i )

plt.figure()
ws = np.array([0, w21, w31, w21+w31])
phis = np.array([f1, M_inf[2,6], M_inf[1,5], M_inf[3,7]])
plt.semilogy(ws, phis,'o', label='neuron1')
# plt.semilogy(ws[:3], phis[:3])
ws = np.array([0, w12, w32, w12+w32])
phis = np.array([f2, M_inf[4,6], M_inf[1,3], M_inf[5,7]])
plt.semilogy(ws, phis,'o', label='neuron2')
# plt.semilogy(ws[:3], phis[:3])
ws = np.array([0, w13, w23, w13+w23])
phis = np.array([f3, M_inf[4,5], M_inf[2,3], M_inf[6,7]])
plt.semilogy(ws, phis,'o', label='neuron3')
# plt.semilogy(ws[:3], phis[:3])
plt.xlabel('x',fontsize=20); plt.ylabel('phi',fontsize=20); plt.legend(fontsize=15)
# plt.savefig('retina_NL.pdf')

# %% scan effective model
# corr_final = np.zeros(target_dof)
# r2s = corr_final*0
# sign = corr_final*0
# ii = 0
# Cp_condition = np.zeros(dofs_all)
# while ii < target_dof:
#     ### work on tau first
#     if ii<nc-1:
#         Cp_condition[rank_tau[ii]] = 1
#     else:
#         Cp_condition[rank_C[ii-(nc-1)]+(nc)] = 1
    
#     ### run max-cal!
#     constraints = ({'type': 'eq', 'fun': eq_constraint, 'args': (observations, Cp_condition)})
#     bounds = [(.0, 100)]*num_params
#     param0 = np.ones(num_params)*.1 + np.random.rand(num_params)*0.0
#     result = minimize(objective_param, param0, args=(P0), method='SLSQP', constraints=constraints, bounds=bounds)
#     param_dof = result.x
#     r2s[ii] = corr_param(param_temp, param_dof, '0')  # non-biniary version!
#     sign[ii] = sign_corr(param_temp, param_dof)
#     print(ii)    
#     ii = ii+1

# %%
# plt.figure()
# plt.plot(r2s,'-o', label='corr')
# plt.plot(sign,'-o', label='signed corr')
# plt.xlabel('dof', fontsize=20); plt.legend(fontsize=20); plt.title('retina (compared to full dof)', fontsize=20)
# # plt.savefig('retina_dof.pdf')

# %% recover transitions from dof
def dof2trans(ranked_C, cutoff):
    Mid = np.arange(64).reshape(8,8)
    trans_list = []
    for ii in range(cutoff):
        rank = ranked_C[ii]
        ij = np.concatenate(np.where(Mid==rank))
        trans_list.append([combinations[ij[0]] , combinations[ij[1]]])
    return trans_list

ranked_trans = dof2trans(rank_C,24)

# %%
# M = np.array([[0,    f3,   f2,   0,              f1,   0,               0,               0],
#               [r3,   0,    0,    f2*np.exp(w32), 0,    f1*np.exp(w31),  0,               0],
#               [r2,   0,    0,    f3*np.exp(w23), 0,    0,               f1*np.exp(w21),  0],
#               [0,    r2,   r3,   0,              0,    0,               0,               f1*np.exp(w231)],
#               [r1,   0,    0,    0,              0,    f3*np.exp(w13),  f2*np.exp(w12),  0],
#               [0,    r1,   0,    0,              r3,   0,               0,               f2*np.exp(w132)],
#               [0,    0,    r1,   0,              r2,   0,               0,               f3*np.exp(w123)],
#               [0,    0,    0,    r1,             0,    r2,              r3,              0]]) 

r1,r2,r3 = M_inf[4,0],M_inf[2,0],M_inf[1,0]

# %% coarse graining
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

def coarse_grain_tauC(ijk, tau=tau_all, C=C_all):
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

# %% retrieve
weff12,weff13,weff21 = coarse_grain_tauC((1,2,3)), coarse_grain_tauC((1,3,2)), coarse_grain_tauC((2,1,3))
weff23,weff32,weff31 = coarse_grain_tauC((2,3,1)), coarse_grain_tauC((3,2,1)), coarse_grain_tauC((3,1,2))

# %%
bar_width = 0.35
bar_positions_group2 = np.arange(6)
bar_positions_group1 = ['cg12','cg13','cg21','cg23','cg32','cg31']

plt.figure()
plt.subplot(211)
plt.bar(categories, inf_w, width=bar_width)
plt.plot(categories, bar_positions_group2*0, 'k')
plt.ylabel('inferred', fontsize=20)
plt.ylim([-5.5,2.5])
# plt.ylim([-4.5, 2])

plt.subplot(212)
plt.bar(bar_positions_group1, np.array([weff12,weff13,weff21,weff23,weff32,weff31])+0, width=bar_width)
plt.plot(bar_positions_group1, bar_positions_group2*0, 'k')
plt.ylabel('coarse grain', fontsize=20)
plt.ylim([-5.5,2.5])
# plt.ylim([-4.5,2])
# plt.savefig('retina_infer_CG_B20_full.pdf')

# %% ideas
# window test
# response function
# effective model
# try 3 out of 5 neurons...

# %% new notes for 4/19
# modify fig7, revisit large window for Appendix
#######
# generate spike train from fitted rates
# measure ISI/IBI from the CTMC generated spike train (save MaxCal!)

# %%
###############################################################################
# %% custom M matrix
tau_new, C_new = (tau_all-0)/new_lt, (C_all-C_base*0)/new_lt*1
M_data = (C_new/tau_new[:,None])
M_data[np.isnan(M_data)] = 0
np.fill_diagonal(M_data, -np.sum(M_data,1))

# %% testing CTMC generative process!!
M_inf = (C_/tau_[:,None])
np.fill_diagonal(M_inf, np.zeros(nc))
Q = M_inf*1 
np.fill_diagonal(Q, -np.sum(Q,1))

# Q = M_data*1 
ctmc_s, ctmc_t = sim_Q(Q, 15000, 1)

# % state back to spikes
# ctmc_spkt, ctmc_spki = [],[]
plt.figure()
for tt in range(len(ctmc_t)):
    state = ctmc_s[tt]
    time = ctmc_t[tt]
    word = np.array(combinations[state])
    for nn in range(N):
        if word[nn]==1:
            plt.plot(time, nn, 'k.')
            # ctmc_spkt.append(tt)
            # ctmc_spki.append(nn)
# plt.savefig('retina_CTMC_spk.pdf')

# %% compute ISI, for RETINA
retina_isi = []
for dd in range(1,3):   #### try all data!!
    spk_data = mat_data['spike_times'][0][dd][0]  # extract timing
    spk_ids = mat_data['cell_IDs'][0][dd][0]  # extract cell ID
    for nn in range(N):
        for rr in range(reps):
            spkt = spk_data[rr].squeeze()
            spki = spk_ids[rr].squeeze()
            pos = np.where(spki==nids[nn])[0]
            spks = spkt[pos]
            if len(spks)>2:
                spks = spks[np.where(spks<lt)[0]]
                isis = np.diff(spks)
                retina_isi.append(isis)

retina_isi = np.concatenate(retina_isi, axis=0)
plt.figure()
plt.hist(retina_isi, np.arange(0,1,.02)*100, density=True, alpha=0.7)

# %% compute burst, for RETINA
Bt_retina = []
retina_burst = np.ones(4)
for dd in range(len(spk_state_all)):
    states = spk_state_all[dd]
    times = spk_time_all[dd]
    if len(states)>0:
        btt = []
        for tt in range(1, len(states)):
            word = np.array(combinations[states[tt]])
            idp = int(sum(word))
            retina_burst[idp] += 1
            btt.append(np.ones(times[tt]-times[tt-1])*idp)  # bust time series
    Bt_retina.append(np.concatenate((btt),axis=0))
    
# %% for CTMC
reps_ctmc = 500        
ctmc_data, ctmc_ids = [], []
ctmc_burst = np.ones(4)
Bt_ctmc = []
for rr in range(reps_ctmc):
    ctmc_s, ctmc_t = sim_Q(Q, 15000, 1)
    ctmc_spkt, ctmc_spki = [],[]
    btt = []
    for tt in range(1, len(ctmc_t)):
        state = ctmc_s[tt]
        time = ctmc_t[tt]
        word = np.array(combinations[state])
        idp = int(sum(word))
        ctmc_burst[idp] += 1
        btt.append(np.ones(int(ctmc_t[tt]-ctmc_t[tt-1]))*idp)
        for nn in range(N):
            if word[nn]==1:
                ctmc_spkt.append(time)
                ctmc_spki.append(nn)
    ctmc_data.append(np.array(ctmc_spkt))
    ctmc_ids.append(np.array(ctmc_spki))
    
    Bt_ctmc.append(np.concatenate((btt),axis=0))
# ctmc_spkt = np.array(ctmc_spkt)
# ctmc_spki = np.array(ctmc_spki)

# %% compare burst
plt.figure()
plt.bar([0,1,2,3], retina_burst/np.sum(retina_burst),  alpha=0.5, label='retina')
plt.bar([0,1,2,3], ctmc_burst/np.sum(ctmc_burst),  alpha=0.5, label='CTMC')
plt.legend(fontsize=20); plt.xlabel('burst size', fontsize=20); plt.ylabel('P(burst)', fontsize=20)
plt.yscale('log')

# %% compare burst-autocorr
def compute_average_acf(time_series_list, max_lag):
    total_acf = np.zeros(max_lag + 1)
    for time_series in time_series_list:
        if len(time_series)> max_lag:
            acf_values = acf(time_series, nlags=max_lag)
            total_acf += acf_values
    average_acf = total_acf / len(time_series_list)
    return average_acf

max_lag = 500
ctmc_acf = compute_average_acf(Bt_ctmc, max_lag)
retina_acf = compute_average_acf(Bt_retina, max_lag)
# Plot average autocorrelation function
lag = np.arange(max_lag + 1)
plt.figure()
plt.plot(lag, ctmc_acf, label='CTMC')
plt.plot(lag, retina_acf, label='retina')
plt.xlabel('Lag'); plt.ylabel('acf'); plt.legend(fontsize=20)

# %% compare ISI
ctmc_isi = []
for rr in range(reps_ctmc):
    spkt = ctmc_data[rr]
    spki = ctmc_ids[rr]
    for nn in range(N):
        pos = np.where(spki==nn)[0]
        spks = spkt[pos]
        if len(spks)>2:
            isis = np.diff(spks)
            ctmc_isi.append(isis)

ctmc_isi = np.concatenate(ctmc_isi, axis=0)
plt.figure()
plt.hist(ctmc_isi, np.arange(0,1,.05)*500, density=True, alpha=0.5, label='CTMC')
plt.hist(retina_isi, np.arange(0,1,.05)*500, density=True, alpha=0.5, label='retina')
plt.legend(fontsize=20); plt.xlabel('ISI (ms)')
plt.yscale('log')

# %%
plt.figure() 
aa_ctmc,bb = np.histogram(ctmc_isi, np.arange(0,1,.02)*500, density=True)
aa_retina,bb = np.histogram(retina_isi, np.arange(0,1,.02)*500, density=True)
bb = (bb[:-1]+bb[1:])/2
plt.plot(bb, aa_ctmc, label='CTMC')
plt.plot(bb,aa_retina, label='retina')
plt.yscale('log'); plt.legend(fontsize=20); plt.xlabel('ISI (ms)', fontsize=20)
# plt.savefig('retina_ctmc_ISI.pdf')

# %% Max Ent!!!
###############################################################################
# %%
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

# %% compute ISI
### inference with MaxEnt
adapt_window = window*1  # ms
states_me = MaxEnt_states(firing, adapt_window)
tau_me = tau4maxent(states_me)

### sample from MaxEnt
reps_me = 100        
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

# %%
isi_max = 520  # 1500
plt.figure() 
aa_ctmc,bb = np.histogram(ctmc_isi, np.arange(0,isi_max, adapt_window), density=True)
aa_retina,bb = np.histogram(retina_isi, np.arange(0,isi_max, adapt_window), density=True)
aa_me,bb = np.histogram(me_isi, np.arange(0,isi_max, adapt_window), density=True)
bb = (bb[:-1]+bb[1:])/2
leave = np.arange(1,len(aa_me)-1,1)
plt.plot(bb[:-1], aa_ctmc[:-1], label='CTMC')
plt.plot(bb[:-1], aa_retina[:-1], label='retina')
plt.plot(bb[leave],aa_me[leave], label='Max Ent')
plt.yscale('log'); plt.legend(fontsize=20); plt.xlabel('ISI (ms)', fontsize=20)
# plt.savefig('retina_ctmc_ISI2.pdf')
