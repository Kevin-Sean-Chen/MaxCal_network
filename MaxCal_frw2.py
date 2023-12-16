# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 21:13:30 2023

@author: kevin
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
from scipy.optimize import minimize
from scipy.optimize import fsolve

import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

#####
# Code for MaxCal in the parameter coordinate, with asymmetric neurons (different f,r,w)
#####
# %% initialize structure and parameters
f1,r1,w12 = 0.2, 0.9, 0.5
f2,r2,w21 = 0.6, 0.2, 0.9
nc = 2**2  # number of total states of 2-neuron
param_true = (f1,r1,w12,\
              f2,r2,w21)  # store ground-truth parameters

# %%
def asym_M(param):
    """
    With asymmetric 2-neuron circuit, given parameters f12,r12,w12,21
    return transition matrix and steady-state distirubtion
    """
    f1,r1,w12,  f2,r2,w21 = param
    M = np.array([[(1-f1)*(1-f2),  f2*(1-f1),      f1*(1-f2),      f1*f2],
                  [(1-w21)*r2,     (1-w21)*(1-r2), w21*r2,         w21*(1-r2)],
                  [(1-w12)*r1,     w12*r1,         (1-w12)*(1-r1), w12*(1-r1)],
                  [r1*r2,          r1*(1-r2),      (1-r2)*r1,      (1-r1)*(1-r2)]]) ## 00,01,10,11
    uu,vv = np.linalg.eig(M.T)
    zeros_eig_id = np.argmin(np.abs(uu-1))
    pi_ss = vv[:,zeros_eig_id] / np.sum(vv[:,zeros_eig_id])
    return M, np.real(pi_ss)

def MaxCal_D(Pyx, P0, param):
    """
    KL devergence term, with transition P and prior P0 as input
    This term can be unstable in log!
    """
    eps = 1e-10
    Pxy = P_frw2(param)
    kl = np.sum(Pxy*(np.log((Pyx + eps)) - np.log(P0 + eps)))  # MaxCal objective
    return kl

def P_frw2(param):
    """
    get joint probability given parameters
    """
    k, pi = asym_M(param)  # calling asymmetric network
    Pxy = np.zeros((nc,nc))
    for ii in range(nc):
        Pxy[ii,:] = k[ii,:]*pi[ii]  # compute joint from transition k and steady-state pi
    return Pxy
    
def comp_etas(P):
    return P + P.T

def comp_Js(P):
    return P - P.T

def bursting(P, param):
    """
    Bursting observations
    """
    etas = comp_etas(P)
    Js = comp_Js(P)
    _,pi = asym_M(param)
    piB0 = pi[0]*1
    piB2 = pi[3]*1
    etaB01 = etas[0,1] + etas[0,2]
    etaB12 = etas[1,3] + etas[2,3]
    etaB02 = etas[0,3]*1
    JB = Js[0,3]*1
    return piB0, piB2, etaB01, etaB12, etaB02, JB

def marginalization(P, param):
    """
    Marginal observations
    """
    etas = comp_etas(P)
#    Js = comp_Js(P)
    _,pi = asym_M(param)
    piX0 = pi[0] + pi[1]
    etaX = etas[1,3] + etas[1,2] + etas[0,3] + etas[0,2]
    piY0 = pi[0] + pi[2]
    etaY = etas[2,3] + etas[0,1] + etas[1,2] + etas[0,3]
    return piX0, etaX, piY0, etaY

def obs_given_frw(param, Cp_condition='all'):
    """
    concatenating different observables as constraints
    """
    Pxy = P_frw2(param)
    burst_obs = bursting(Pxy, param)
    marg_obs = marginalization(Pxy, param)
    if Cp_condition=='burst':
        obs = burst_obs*1
    elif Cp_condition=='marginal':
        obs = marg_obs*1
    elif Cp_condition=='all':
        obs = np.concatenate((burst_obs, marg_obs))
    return obs

def C_P(P, observations, param, Cp_condition='all'):
    """
    The C(P,f,r,w) function for constraints
    """
#    f,r,w = param
#    piB0,piB2,etaB01,etaB12,etaB02,JB,piX0,etaX,piY0,etaY = observations
    etas = comp_etas(P)
    Js = comp_Js(P)
    _,pi = asym_M(param)
    c1 = pi[0]
    c2 = pi[3]
    c3 = etas[0,1] + etas[0,2]
    c4 = etas[1,3] + etas[2,3]
    c5 = etas[0,3]
    c6 = Js[0,3]
    c7 = pi[0] + pi[1]
    c8 = etas[1,3] + etas[1,2] + etas[0,3] + etas[0,2]
    c9 = pi[0] + pi[2]
    c10 = etas[2,3] + etas[0,1] + etas[1,2] + etas[0,3]
    if Cp_condition=='burst':
        c_P_frw = np.array([c1,c2,c3,c4,c5,c6]) # burst only
    elif Cp_condition=='marginal':
        c_P_frw = np.array([c7,c8,c9,c10]) # marginal only
    elif Cp_condition=='all':
        c_P_frw = np.array([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10])  # all constraints
    return c_P_frw - observations


# %%
###############################################################################
# %% THIS is a nonlinear constrain optimization problem!
def objective(param, P0):
    Pyx,_ = asym_M(param)
    D = MaxCal_D(Pyx, P0, param)
    return D
def eq_constraint(param, observations, Cp_condition):
    Pxy = P_frw2(param)
    cp = C_P(Pxy, observations, param, Cp_condition)
    return 0.5*np.sum(cp**2) ### not sure if this hack is legit, but use MSE for now

# Constraints
Cp_condition = 'all'  #'burst', 'marginal', 'all'  ### choose one of the here ###
observations = obs_given_frw(param_true, Cp_condition)
constraints = ({'type': 'eq', 'fun': eq_constraint, 'args': (observations, Cp_condition)})
bounds = [(0, 1)]*6

# Perform optimization using SLSQP method
P0 = np.random.rand(nc,nc)
P0 = P0 / P0.sum(axis=1, keepdims=True)
param0 = np.random.rand(6)*1.
result = minimize(objective, param0, args=(P0), method='SLSQP', constraints=constraints, bounds=bounds)
frw_inf = result.x

print('true frw:', param_true)
print('inferred frw:', frw_inf)

# %% systemetic analysis
###############################################################################
# %%
param_true = np.random.rand(6)
conditions = ['all','burst','marginal']
frw_inference = np.zeros((3,6))
for ii in range(3):
    Cp_condition = conditions[ii]
    observations = obs_given_frw(param_true, Cp_condition)
    constraints = ({'type': 'eq', 'fun': eq_constraint, 'args': (observations, Cp_condition)})
    
    # Perform optimization using SLSQP method
    P0 = np.random.rand(nc,nc)
    P0 = P0 / P0.sum(axis=1, keepdims=True)
    param0 = np.random.rand(6)*1.
    result = minimize(objective, param0, args=(P0), method='SLSQP', constraints=constraints, bounds=bounds)
    frw_inf = result.x
    frw_inference[ii,:] = frw_inf
    
# %%
bars = np.concatenate((param_true[None,:], frw_inference))
num_rows, num_cols = bars.shape
bar_positions = np.arange(num_cols)
labels = ['true', 'all', 'burst',  'marginal']
col_labels = ['f1','r1','w12','f2','r2','w21']

plt.figure(figsize=(10, 6))
for i in range(num_rows):
    plt.bar(bar_positions + i * 0.2, bars[i, :], width=0.2, label=labels[i])
    
plt.legend(fontsize=15)
plt.xticks(bar_positions + 0.2, [col_labels[col] for col in range(num_cols)])

    