# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 21:13:52 2023

@author: kevin
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
import itertools
from scipy.optimize import minimize

import matplotlib 
matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15)

# %%
# %% pseudo-code for now
N = 2  # number of neurons
spins = [0,1]  # binary patterns
combinations = list(itertools.product(spins, repeat=N))  # possible configurations
nc = len(combinations)  # number of patterns
Mxy = np.random.rand(nc, nc) + np.eye(nc)*0  # all patterns to all patterns transition matrix
Mxy = Mxy / Mxy.sum(1)[:,None]  # normalize as transition matrix
## delta example
gs = [(1,0), (0,1)]  #x,y,g(x,y)
num_const = 3#1  # number of constraints (length of beta vector)
def g_function(x,y, constraint_i=None):
    """
    test function for g(x,y) as the constraints
    """
    if constraint_i is None or constraint_i==0:
        if np.sum(x+y) == N*2:  # bust detection
            gxy = 1
        elif (x==(1,1)) and (y==(0,0)):  # burst
            gxy = 0.5
        elif (x==(1,0)) and (y==(0,1)):  # transition
            gxy = 0.1
        else:
            gxy = 0.
    elif constraint_i == 1:
        if np.sum(x+y) == N*2:  # bust detection
            gxy = .0
        elif (x==(0,0)) and (y==(1,1)):  # burst
            gxy = 0.1
        elif (x==(0,1)) and (y==(1,0)):  # transition
            gxy = 0.5
        else:
            gxy = 0.
    elif constraint_i == 2:
        if np.sum(x+y) == N*2:  # bust detection
            gxy = .0
        elif (x==(1,1)) and (y==(0,1)):  # burst
            gxy = 1
        elif (x==(1,1)) and (y==(1,0)):  # transition
            gxy = .5
        else:
            gxy = 0.
    return gxy

beta = np.array([1,2,.4]) #np.array([1]) # the multiplier
Mxy_ = Mxy*1
Gxy = np.zeros((nc, nc, num_const))
for ii in range(nc):
    for jj in range(nc):
        for cc in range(num_const):
            Gxy[ii,jj,cc] = g_function(combinations[ii] , combinations[jj], cc)
        Mxy_[ii,jj] = Mxy[ii,jj] * np.exp(Gxy[ii,jj,:]@beta)  # eqn.1 in Peter's note

uu, vr = np.linalg.eig(Mxy_)  # right vector
u2, vl = np.linalg.eig(Mxy_.T)  # left vectors
lamb,lp = np.max(np.real(uu)), np.argmax(np.real(uu))  # max real eigen value
vrx, vlx = np.real(vr[:,lp]), np.real(vl[:,lp])
vec_dot_norm = np.sqrt(np.abs(np.dot(vrx, vlx)))  # normalization factor for unit dot product
vrx, vlx = np.abs(vrx)/vec_dot_norm, np.abs(vlx)/vec_dot_norm  # normalize eigen vectors
Pyx = np.zeros((nc,nc))
for ii in range(nc):
    for jj in range(nc):
        Pyx[ii,jj] = 1/lamb*vrx[jj]/vrx[ii]*Mxy_[ii,jj]  # transition matrix
#Pyx = (vrx)/(lamb*(vlx))*Mxy_  # posterior.... not sure about this?  #ry/rx, not l!!
#(vry)/(lamb*(vrx))*Mxy_
Pxy = np.zeros((nc,nc))
for ii in range(nc):
    Pxy[ii,:] = Pyx[ii,:]*vrx[ii]*vlx[ii]
#    Pxy = np.outer(vrx,vlx) * Pyx  # joint probability
#Pxy = vlx[:,None]@Mxy_@vrx[:,None]/lamb  # <l|M_bar|r>/lamb
        
# %% write functions for optimization
def lamb_beta(beta):
    # return largest eigen value given beta vector
    Mxy_ = Mxy*1
    Gxy = np.zeros((nc, nc, num_const))
    for ii in range(nc):
        for jj in range(nc):
            for cc in range(num_const):
                Gxy[ii,jj,cc] = g_function(combinations[ii] , combinations[jj], cc)
            Mxy_[ii,jj] = Mxy[ii,jj] * np.exp(Gxy[ii,jj,:]@beta)
    uu, vr = np.linalg.eig(Mxy_)  # right vector
    u2, vl = np.linalg.eig(Mxy_.T)  # left vectors
    lamb,lp = np.max(np.real(uu)), np.argmax(np.real(uu))  # max real eigen value
    vrx, vlx = np.real(vr[:,lp]), np.real(vl[:,lp])
    vec_dot_norm = np.sqrt(np.abs(np.dot(vrx, vlx)))
    vrx, vlx = np.abs(vrx)/vec_dot_norm, np.abs(vlx)/vec_dot_norm  # normalize eigen vectors
    Pyx = get_transition(vrx,lamb,Mxy_)
    Pxy = get_joint(vrx,vlx,Pyx)
    return lamb, Pxy, Pyx

def expect_g(Pxy, Gxy=Gxy):
    # put in posterior calculation and stationary probability
    if num_const>1:
        g_bar = np.zeros(num_const)
        for ii in range(num_const):
            g_bar[ii] = np.sum(Pxy * Gxy[:,:,ii])  # calculating expected g separately
    else:
        g_bar = np.sum(Pxy * Gxy)
    return g_bar

def objective(beta, g_bar):
    """
    a function of beta and g_bar, here we optimize for beta*
    """
    lamb,_,_ = lamb_beta(beta)
#    g_bar = expect_g(Pxy)
    obj = np.dot(beta, g_bar) - np.log(lamb)
    return -obj # do scipy.minimization on this

def posterior_Pxy(beta_star):
    # return posterior transition matrix given beta*
    _,Pxy,_ = lamb_beta(beta_star)
    return Pxy

def get_transition(vrx,lamb,Mxy_):
    # get transition matrix with element-wise computation
    Pyx = np.zeros((nc,nc))
    for ii in range(nc):
        for jj in range(nc):
            Pyx[ii,jj] = 1/lamb*vrx[jj]/vrx[ii]*Mxy_[ii,jj]  # transition matrix
    return Pyx

def get_joint(vrx,vlx,Pyx):
    # get joint distribution with element-wise computation
    Pxy = np.zeros((nc,nc))
    for ii in range(nc):
        Pxy[ii,:] = Pyx[ii,:]*vrx[ii]*vlx[ii]
    return Pxy

def get_stationary(M):
    # check stationary of mstrix M
    # pix@M should be pix, lamb should be one
    uu, vr = np.linalg.eig(M)  # right vector
    u2, vl = np.linalg.eig(M.T)  # left vectors
    lamb,lp = np.max(np.real(uu)), np.argmax(np.real(uu))  # max real eigen value
    vrx, vlx = np.real(vr[:,lp]), np.real(vl[:,lp])
    vec_dot_norm = np.sqrt(np.abs(np.dot(vrx, vlx)))  # normalization factor for unit dot product
    vrx, vlx = np.abs(vrx)/vec_dot_norm, np.abs(vlx)/vec_dot_norm  # normalize eigen vectors
    pix = (vrx)*vlx/(lamb)
    return pix, lamb

#def init_g_bar(beta0):
#    g_bars = beta0*0
#    for nn in range(num_const):
#        Q = np.random.rand(nc, nc)  # random transition matrix
#        Q = Q / Q.sum(1)[:,None] 
#        pix,_ = get_stationary(Pyx)
#        g_bars[nn] = expect_g(pix[:,None]*Q)
#    return g_bars

# %%
beta0 = beta*0 + np.random.randn()*beta  # initialize around the true parameter
lamb,Pxy,_ = lamb_beta(beta0)
Q = np.random.rand(nc, nc)  # random transition matrix
Q = Q / Q.sum(1)[:,None] 
pix,_ = get_stationary(Pyx) # or Q for perturbation?
g_bar = expect_g(pix[:,None]*Q) #feed in a random matrix, not Pxy: would be differnt but same average if it works
#g_bar = init_g_bar(beta0)
# Minimize the function
result = minimize(objective, beta0, args=(g_bar),method='SLSQP',bounds=[(0,500)]*num_const)
print('beta*:'+str(result.x))

post_tran = posterior_Pxy(result.x)  # compute posterior transition matrix given inferred beta*
#post_tran = posterior_Pyx(beta)

plt.figure()
plt.subplot(121)
plt.imshow(Pyx)
plt.title('True Py|x',fontsize=20)
plt.xticks(range(len(combinations)), combinations);
plt.yticks(range(len(combinations)), combinations);
plt.subplot(122)
plt.imshow(post_tran)
plt.title('MaxCal inferred',fontsize=20)
plt.xticks(range(len(combinations)), combinations);
plt.yticks(range(len(combinations)), combinations);

# %% check that the average is the same
print('observed g:'+str(g_bar))
infer_g = expect_g(post_tran)
print('inferred g:'+str(infer_g))

# check row-sum of M, steady state, and g_bar
# vectorize beta
# play with M matrix ...