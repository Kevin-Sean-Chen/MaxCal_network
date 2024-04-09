# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 15:55:12 2024

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

# %% load data
import pickle

motif_type = 'common'  # common, chain, cyclic

with open('data_pkl/motif_'+motif_type+'.pkl', 'rb') as f:  
    loaded_data = pickle.load(f)
    
firing = loaded_data['firing']
S = loaded_data['S']
M_inf = loaded_data['M_inf']

# %% compute C and tau
window = 150  #0.1ms  # 120, 150, 170, 200
lt = len(firing)
spk_states, spk_times = spk2statetime(firing, window, lt=lt)
tau,C = compute_tauC(spk_states, spk_times, lt=lt)

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

def coarse_grain_tauC(ijk, tau=tau, C=C.T):
    """
    return coupling that is i->j ignoring k
    """
    i,j,k = ijk
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
plt.bar(bar_positions_group1, [S[1,0],S[2,0],S[0,1],S[2,1],S[1,2],S[0,2]], width=bar_width)
plt.plot(bar_positions_group1, bar_positions_group2*0, 'k')
plt.ylabel('true weights', fontsize=20)

plt.subplot(212)
plt.bar(bar_positions_group1, np.array([weff12,weff13,weff21,weff23,weff32,weff31])+0, width=bar_width)
plt.plot(bar_positions_group1, bar_positions_group2*0, 'k')
plt.ylabel('MaxCal inferred', fontsize=20)
# plt.savefig('3I_common_B20.pdf')

# %% debug
f = (C[word_id(((0,0,0))) , word_id((1,0,0)) ] + C[word_id((0,0,1)) , word_id((1,0,1)) ]) \
      /(tau[word_id((0,0,0))]+tau[word_id((0,0,1))])
fexpw = (C[word_id((0,1,0)) , word_id((1,1,0)) ] + C[word_id((0,1,1)) , word_id((1,1,1)) ]) \
      /(tau[word_id((0,1,0))]+tau[word_id((0,1,1))])
weff = np.log(fexpw/f)

# %%
states = np.array([0,1,0,2,3,0,4])
times = np.array([2,4,6,7,10,13,15])
tau_,C_ = compute_tauC(states, times)
coarse_grain_tauC((3,2,1), tau=tau_+1, C=C_)

# %% plotting wij
def invf(x):
    return np.log(x)

f1,f2,f3 = M_inf[0,4], M_inf[0,2], M_inf[0,1]
w12,w13,w21 = invf(M_inf[4,6]/f2), invf(M_inf[4,5]/f3), invf(M_inf[2,6]/f1)
w23,w32,w31 = invf(M_inf[2,3]/f3), invf(M_inf[1,3]/f2), invf(M_inf[1,5]/f1)

bar_width = 0.35
bar_positions_group2 = np.arange(6)
bar_positions_group1 = ['w12','w13','w21','w23','w32','w31']
plt.figure()
plt.subplot(211)
plt.bar(bar_positions_group1, [S[1,0],S[2,0],S[0,1],S[2,1],S[1,2],S[0,2]], width=bar_width)
plt.plot(bar_positions_group1, bar_positions_group2*0, 'k')
plt.ylabel('true weights', fontsize=20)
plt.subplot(212)
plt.bar(bar_positions_group1, np.array([w12,w13,w21,w23,w32,w31])+0, width=bar_width)
plt.plot(bar_positions_group1, bar_positions_group2*0, 'k')
plt.ylabel('MaxCal inferred', fontsize=20)
# plt.savefig('infer_w_chain.pdf')