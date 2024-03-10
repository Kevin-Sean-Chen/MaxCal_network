# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 18:48:50 2024

@author: kevin
"""

from maxcal_functions import spk2statetime, compute_tauC, param2M, eq_constraint, objective_param, compute_min_isi

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
dataset = 0  # 0-natural, 1-Brownian, 2-repeats
nid = 3  # neuron
spk_times = mat_data['spike_times'][dataset][0][0][nid].squeeze()
n_neurons = len(mat_data['spike_times'][dataset][0][0])
spk_data = mat_data['spike_times'][dataset][0][0]  # extract to use

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
spk_states, spk_times = spk2statetime(firing, 150)  # embedding states
tau,C = compute_tauC(spk_states, spk_times)



        