# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 17:29:54 2026

@author: kevin
"""
###########################################
# loading all results and comparing methods
###########################################
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from maxcal_functions import (
    spk2statetime,
    compute_tauC,
    cos_ang,
    corr_param,
)
import pickle

matplotlib.rc("xtick", labelsize=20)
matplotlib.rc("ytick", labelsize=20)

# %% setup params
selected_motif = 0
n_groups = 6
n_conditions = 2
n_measurements = 3

group_labels = ['w=4', '8', '16', '32', '64', '128']
condition_labels = ['MaxCal', 'GLM', 'GC', 'spk_GC']


# %%
def plot_perturbed(purt='noise'):
    ### for GC
    fname = 'C:/Users/kevin/Documents/github/MaxCal_network/GC_comparison3.pkl'
    with open(fname, 'rb') as f:
        loaded_data = pickle.load(f)
    coss, scan_x = loaded_data['coss'], loaded_data['scan_x']
    # return coss, scan_x
    
    bar_width = 0.18
    index = np.arange(n_groups)
    plt.figure()
    meas_i = coss*1
    mean_i = np.nanmean(meas_i[selected_motif,:,:, 0],1)
    std_i = np.nanstd(meas_i[selected_motif,:,:, 0],1)
    mean_j = np.nanmean(meas_i[selected_motif,:,:, 1],1)
    std_j = np.nanstd(meas_i[selected_motif,:,:, 1],1)
    
    ### for GLM
    fname = 'C:/Users/kevin/Documents/github/MaxCal_network/glm_comparison3.pkl'
    with open(fname, 'rb') as f:
        loaded_data = pickle.load(f)
    coss, scan_x = loaded_data['coss'], loaded_data['scan_x']
    meas_i = coss*1
    mean_k = np.nanmean(meas_i[selected_motif,:,:, 1],1)
    std_k = np.nanstd(meas_i[selected_motif,:,:, 1],1)
    
    ### for spk_GC
    fname = 'C:/Users/kevin/Documents/github/MaxCal_network/spk_GC_comparison.pkl'
    with open(fname, 'rb') as f:
        loaded_data = pickle.load(f)
    coss = loaded_data['coss']
    meas_i = coss*1
    mean_l = np.nanmean(meas_i[selected_motif,:,:, 1],1)
    std_l = np.nanstd(meas_i[selected_motif,:,:, 1],1)
    
    ax = plt.subplot(1, 1, 1)
    # Plot all methods side by side
    rects1 = ax.bar(index - bar_width, mean_i, bar_width,
                    label=condition_labels[0],
                    yerr=std_i, capsize=5)
    
    rects2 = ax.bar(index, mean_k, bar_width,
                    label=condition_labels[1],
                    yerr=std_k, capsize=5)

    rects3 = ax.bar(index + bar_width, mean_j, bar_width,
                    label=condition_labels[2],
                    yerr=std_j, capsize=5)

    rects4 = ax.bar(index + 2 * bar_width, mean_l, bar_width,
                    label=condition_labels[3],
                    yerr=std_l, capsize=5)

    ax.set_xticks(index)
    ax.set_xticklabels(group_labels)
    ax.legend()

plot_perturbed(''); plt.xlabel('weights',fontsize=20); plt.ylabel('cos',fontsize=20); #plt.xscale('log')   
# plt.savefig("comparison.pdf", format="pdf", bbox_inches="tight")