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
# Code for MaxCal in the parameter coordinate
# begin with test function, then exploration optimization algorithm
#####
# %% initialize structure and parameters
f,r,w = 0.3, 0.9, 0.5  # keep this symmetric for the first test
#f,r,w1,w2 = 0.3, 0.8, 0.5
nc = 2**2  # number of total states of 2-neuron
param_true = (f,r,w)  # store ground-truth parameters

# %%
def sym_M(param):
    """
    With symmetric (homogeneous) 2-neuron circuit, given parameters f,r,w
    return transition matrix and steady-state distirubtion
    """
    f,r,w = param
    M = np.array([[(1-f)**2,  f*(1-f),  f*(1-f),  f**2],
                  [(1-w)*r,  (1-w)*(1-r),  w*r,  w*(1-r)],
                  [(1-w)*r,  w*r,  (1-w)*(1-r),  w*(1-r)],
                  [r**2,  r*(1-r),  (1-r)*r,  (1-r)**2]]) ## 00,01,10,11
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
    Pxy = P_frw(param)
    kl = np.sum(Pxy*(np.log((Pyx + eps)) - np.log(P0 + eps)))  # MaxCal objective
    return kl

def P_frw(param):
    """
    get joint probability given parameters
    """
#    f,r,w = param
    k, pi = sym_M(param)
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
    _,pi = sym_M(param)
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
    _,pi = sym_M(param)
    piX0 = pi[0] + pi[1]
    etaX = etas[1,3] + etas[1,2] + etas[0,3] + etas[0,2]
    piY0 = pi[0] + pi[2]
    etaY = etas[2,3] + etas[0,1] + etas[1,2] + etas[0,3]
    return piX0, etaX, piY0, etaY

def obs_given_frw(param):
    """
    concatenating all observables as constraints
    """
    Pxy = P_frw(param)
    burst_obs = bursting(Pxy, param)
    marg_obs = marginalization(Pxy, param)
    obs = np.concatenate((burst_obs, marg_obs))
    return obs

def C_P(P, observations, param):
    """
    The C(P,f,r,w) function for constraints
    """
#    f,r,w = param
#    piB0,piB2,etaB01,etaB12,etaB02,JB,piX0,etaX,piY0,etaY = observations
    etas = comp_etas(P)
    Js = comp_Js(P)
    _,pi = sym_M(param)
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
    c_P_frw = np.array([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10])  # all constraints
#    c_P_frw = np.array([c1,c2,c3,c4,c5,c6]) # burst only
#    c_P_frw = np.array([c7,c8,c9,c10]) # martginal only
    return c_P_frw - observations

def objective_fix_beta(param, beta_, observations, P0):
    Pxy = P_frw(param)
    Pyx,_ = sym_M(param)
    D = MaxCal_D(Pyx, P0, param)
    obs_diff = C_P(Pxy, observations, param)
    obj = D - np.dot(beta_, obs_diff)
    return obj

def objective_fix_param(beta, param_, observations, P0):
    Pxy = P_frw(param_)
#    Pyx,_ = sym_M(param_)
#    D = MaxCal_D(Pyx, P0, param_)
    obs_diff = C_P(Pxy, observations, param_)
#    obj = D - np.dot(beta, obs_diff)
#    obj = -np.dot(beta, obs_diff)  # try MSE solver
    obj = np.sum(beta*(obs_diff)**2)
    return obj

def objective_full(x, observations, P0):
    param = x[:3]
    beta = x[3:]
    Pxy = P_frw(param)
    D = MaxCal_D(Pxy, P0, param)
    obs_diff = C_P(Pxy, observations, param)
    obj = D - np.dot(beta, obs_diff)
    return obj

def C__P_solve(param, observations):
    """
    The C(P,f,r,w) function for constraints
    WARNING: currently does not work!
    """
#    f,r,w = param
#    piB0,piB2,etaB01,etaB12,etaB02,JB,piX0,etaX,piY0,etaY = observations
    root = fsolve(func, param, args=(observations))
    return root

def func(param, observations):
    P = P_frw(param)
    etas = comp_etas(P)
    Js = comp_Js(P)
    _,pi = sym_M(param)
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
    c_P_frw = np.array([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10]) - observations
    return c_P_frw

# %%
###############################################################################
# %% THIS is a nonlinear constrain optimization problem!
def objective(param, P0):
    Pyx,_ = sym_M(param)
    D = MaxCal_D(Pyx, P0, param)
    return D
def eq_constraint(param, observations):
    Pxy = P_frw(param)
    cp = C_P(Pxy, observations, param)
    return 0.5*np.sum(cp**2) ### not sure if this hack is legit, but use MSE for now

# Constraints
observations = obs_given_frw(param_true)#[6:]
constraints = ({'type': 'eq', 'fun': eq_constraint, 'args': (observations, )})
bounds = [(0, 1), (0, 1), (0, 1)]

# Perform optimization using SLSQP method
P0 = np.random.rand(nc,nc)
P0 = P0 / P0.sum(axis=1, keepdims=True)
param0 = np.random.rand(3)*1.
result = minimize(objective, param0, args=(P0), method='SLSQP', constraints=constraints, bounds=bounds)
frw_inf = result.x

print('true frw:', param_true)
print('inferred frw:', frw_inf)

# %%
###############################################################################

# %% init optimization
n_contraints = 10
observations = obs_given_frw(param_true)
beta0 = np.random.randn(n_contraints)
P0 = np.random.rand(nc,nc)
P0 = P0 / P0.sum(axis=1, keepdims=True)
param0 = (0,0,0) + param_true*0 + np.random.rand(3)*1.
x0 = np.concatenate((param0, beta0))
#objective_full(x0, observations, P0)
#objective_fix_beta(param0, beta0, observations, P0)

# %% test with direct
#result = minimize(objective_full, x0, args=(observations, P0),tol=10**-10,method='Powell')  #Powell #L-BFGS-B
#param_beta_inf = result.x

# %% test solver
#frw_ = C__P_solve(param_true, observations)

# %% coordinate method
### is this even correct??!!!??
#n_step = 10
#frw_ = param0*1
#beta_ = beta0*1
#loss = np.zeros(n_step)
#for ii in range(n_step):
#    ### use graditn
#    result_frw = minimize(objective_fix_beta, frw_, args=(beta_, observations, P0),tol=10**-2,method='L-BFGS-B',bounds=[(0,1)]*3)
#    frw_ = result_frw.x*1
#    ### use solver
##    frw_ = C__P_solve(frw_, observations)
#    
#    result_beta = minimize(objective_fix_param, beta_, args=(frw_, observations, P0),tol=10**-2,method='L-BFGS-B')
#    beta_ = result_beta.x*1
#    
#    loss[ii] = result_frw.fun
    
# %%
#####
# If this code has the correct setup, next step would be to exlore better optimization methods
# a candidate is to you pytorch to compute gradient for coordinate desciending
#####
    