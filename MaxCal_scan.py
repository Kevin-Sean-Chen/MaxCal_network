# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 17:56:42 2023

@author: kevin
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
from scipy.optimize import minimize

import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

###
# 2-neuron example with MaxCal inference; 
# numericlly scanning for parameter and prior effects
###

###
# debugging:
# 13 constraints, find the largest error over all 13 observations (absolute or the noramlized one)
# find a way to tune the tolerance
# plot tol vs. error; with a bad f,r,w

# if this doesn't work... return to constraint that are direct: pi, Pijs
tolerance = 10**-30

# %% 2-neuron setup
# firing, refractory, and coupling parameters
f,r,w = 0.6, 0.9, 0.3  # keep this symmetric for the first test
nc = 2**2  # number of total states of 2-neuron
param_true = (f,r,w)  # store ground-truth parameters

def sym_M(f,r,w):
    """
    With symmetric (homogeneous) 2-neuron circuit, given parameters f,r,w
    return transition matrix and steady-state distirubtion
    """
    M = np.array([[(1-f)**2,  f*(1-f),  f*(1-f),  f**2],
                  [(1-w)*r,  (1-w)*(1-r),  w*r,  w*(1-r)],
                  [(1-w)*r,  w*r,  (1-w)*(1-r),  w*(1-r)],
                  [r**2,  r*(1-r),  (1-r)*r,  (1-r)**2]]) ## 00,01,10,11
    uu,vv = np.linalg.eig(M.T)
    zeros_eig_id = np.argmin(np.abs(uu-1))
    pi_ss = vv[:,zeros_eig_id] / np.sum(vv[:,zeros_eig_id])
    return M, np.real(pi_ss)
M, pi_ss = sym_M(f,r,w)
print(pi_ss) # lesson: right eigen vector of a transition matrix is uniform

# %% Max-cal inference functions
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
    # check stationary of matrix M
    # pix@M should be pix, lamb should be one
    uu,vv = np.linalg.eig(M.T)
    zeros_eig_id = np.argmin(np.abs(uu-1))
    pix = vv[:,zeros_eig_id] / np.sum(vv[:,zeros_eig_id])
    return pix #, lamb


def M_tilt_comb(beta, param):
    """
    Titled matrix
    """
    ### unpack parameters
    f,r,w = param  # fill in wit ground-truth or prior guess
    if len(beta) >=13:  # full information
        alpha0, alpha2b, beta01, beta12, beta02, gamm,\
                        alpha1, alpha2, beta1, beta2,\
                        eta1, j1, j2 = beta  # conjugated variables, one etas and two Js
        eta2 = 0
    elif len(beta) == 10:  # bust and marginal observables
        alpha0, alpha2b, beta01, beta12, beta02, gamm,\
                        alpha1, alpha2, beta1, beta2 = beta
        eta1, eta2, j1, j2 = 0,0,0,0
    elif len(beta) == 6:  # only bursting observables
        alpha0, alpha2b, beta01, beta12, beta02, gamm = beta
        alpha1, alpha2, beta1, beta2 = 0,0,0,0
        eta1, eta2, j1, j2 = 0,0,0,0
    
    ### compute tilted matrix
    M_burst = np.array([[(1-f)**2*np.exp(alpha0), f*(1-f)*np.exp(alpha0+beta01), f*(1-f)*np.exp(alpha0+beta01), f**2*np.exp(alpha0+beta02+gamm)],
                   [(1-w)*r*np.exp(beta01), (1-w)*(1-r), w*r, w*(1-r)*np.exp(beta12)],
                   [(1-w)*r*np.exp(beta01), w*r, (1-w)*(1-r), w*(1-r)*np.exp(beta12)],
                   [r**2*np.exp(alpha2b+beta02-gamm), r*(1-r)*np.exp(alpha2b+beta12), r*(1-r)*np.exp(alpha2b+beta12), (1-r)**2*np.exp(alpha2b)]])
    
    M_marg = np.array([[np.exp(alpha1+alpha2), np.exp(alpha1+alpha2+beta2), np.exp(alpha1+alpha2+beta1), np.exp(beta1+alpha1+alpha2+beta2)],
                   [np.exp(alpha1+beta2), np.exp(alpha1), np.exp(alpha1+beta1+beta2), np.exp(alpha1+beta1)],
                   [np.exp(alpha2+beta1), np.exp(alpha2+beta1+beta2), np.exp(alpha2), np.exp(alpha2+beta2)],
                   [np.exp(beta1+beta2), np.exp(beta1), np.exp(beta2), np.exp(0)]])
    
    M_extra = np.exp(np.array([[0,0,-j2,0],
                               [0,0,eta1,j1],
                               [j2,eta1,0,0],
                               [0,-j1,0,0]]))
    
    ### compute eigenvalues
    M_ = M_burst*M_marg*M_extra
    uu, vr = np.linalg.eig(M_)  # right vector
    u2, vl = np.linalg.eig(M_.T)  # left vectors
    lamb,rp,lp = np.max(np.real(uu)), np.argmax(np.real(uu)), np.argmax(np.real(u2))  # max real eigen value
    vrx, vlx = np.real(vr[:,rp]), np.real(vl[:,lp])
    vec_dot_norm = np.sqrt(np.abs(np.dot(vrx, vlx)))  # normalization factor for unit dot product
    vrx, vlx = np.abs(vrx)/vec_dot_norm, np.abs(vlx)/vec_dot_norm  # normalize eigen vectors
    return M_, np.real(vrx), vlx, np.real(lamb)

def posterior_k_comb(beta, vrx, lamb, param):
    r1,r2,r3,r4 = vrx  # eigen vectors(?)
    M_,_,_,_ = M_tilt_comb(beta, param)
    pref = np.array([[1/lamb, r2/(r1*lamb), r3/(r1*lamb), r4/(r1*lamb)],
                     [r1/(r2*lamb), 1/lamb, r3/(r2*lamb), r4/(r2*lamb)],
                     [r1/(r3*lamb), r2/(r3*lamb), 1/lamb, r4/(r3*lamb)],
                     [r1/(r4*lamb), r2/(r4*lamb), r3/(r4*lamb), 1/lamb]])
    k_ = pref*M_
    return k_

def objective_comb(beta, g_bar, param):
    """
    a function of beta and g_bar, here we optimize for beta*
    """
    _,_,_, lamb = M_tilt_comb(beta, param)
    obj = np.dot(beta, g_bar) - np.log(lamb)
    return -obj # do scipy.minimization on thi


# %% compute observables
pi_00, pi_01, pi_10, pi_11 = pi_ss  # steady-state distribution

eta_00_01 = pi_00*f*(1-f) + pi_01*(1-w)*r  # symmetric trafficing
eta_00_10 = pi_00*f*(1-f) + pi_10*(1-w)*r
eta_00_11 = pi_00*f*f + pi_11*r*r
eta_01_11 = pi_01*w*(1-r) + pi_11*r*(1-r)
eta_10_11 = pi_10*w*(1-r) + pi_11*(1-r)*r
eta_01_10 = w*r*(pi_01 + pi_10)

### bust observables
piB0 = pi_00*1
piB2 = pi_11*1
eta_01 = eta_00_01 + eta_00_10
eta_12 = eta_01_11 + eta_10_11
eta_02 = eta_00_11*1
J02 = pi_00*f*f - pi_11*r*r  # P_{0,2} - P_{2,0}, according to the notes

### marginal observables
# neuron 1
pi_x1 = pi_00 + pi_01
eta_x1 = eta_00_11 + eta_01_10 + eta_01_11 + eta_00_10
# neuron 2
pi_x2 = pi_00 + pi_10
eta_x2 = eta_00_01 + eta_01_10 + eta_10_11 + eta_00_11

### extra measurements
J_01_11 = pi_01*(1-r)*w - pi_11*r*(1-r)
J_10_00 = pi_10*(1-w)*r - pi_00*f*(1-f)
g_bar_extra = np.array([eta_01_10, eta_10_11, J_01_11, J_10_00])  # one more eta and two more Js

# %% a prior matrix with 'guessed' parameters
f_,r_,w_ = np.random.rand(3)*1
M_prior, pi_ss_prior = sym_M(f_,r_,w_)
param_prior = (f_,r_,w_)  # guessing parameters as prior

# %% MaxCal!!!
tolerance = 10**-40
g_bar_burst = np.array([piB0, piB2, eta_01, eta_12, eta_02, J02])  # 6 observables from bursting
g_bar_marg = np.array([pi_x1, pi_x2, eta_x1, eta_x2])  # marginal observables
g_bar_extra = np.array([eta_01_10, J_01_11, J_10_00])  # one more eta and two more Js
# eta_10_11

g_bar = np.concatenate((g_bar_burst, g_bar_marg, g_bar_extra))
#g_bar = np.concatenate((g_bar_burst, g_bar_marg))
#g_bar = g_bar_burst*1
num_const = len(g_bar)
beta0 = np.random.rand(num_const)*1 # initialize conjugates
result = minimize(objective_comb, beta0, args=(g_bar, param_prior),tol=tolerance,method='Powell',bounds=[(0,100)]*num_const)  #Powell #L-BFGS-B
beta_inf = result.x
print('beta*:'+str(beta_inf))
result = minimize(objective_comb, beta0, args=(g_bar, param_true),tol=tolerance,method='Powell',bounds=[(0,100)]*num_const)  #SLSQP
beta_true = result.x
print('beta-true:'+str(beta_true))
#### maybe try to decrease tolerance

# %% analyze inference
# inferred posterior matrix
M_comb, vrx, vlx, lamb = M_tilt_comb(beta_inf, param_prior)
post_k_comb = posterior_k_comb(beta_inf, vrx, lamb, param_prior)
pi_ss_inf_comb = get_stationary(post_k_comb)

# same calculation if we have the true parameters
post_k_true,_ = sym_M(f,r,w)
#M_comb, vrx, vlx, lamb = M_tilt_comb(beta_true, param_true)
#post_k_true = posterior_k_comb(beta_true, vrx, lamb, param_true)

plt.figure()
plt.subplot(121)
plt.imshow(post_k_comb, aspect='auto')
plt.title('inferred Pxy',fontsize=20)
plt.subplot(122)
plt.imshow(post_k_true, aspect='auto')
plt.title('true Pxy',fontsize=20)

plt.figure()
plt.plot(pi_ss, label='true')
plt.plot(pi_ss_inf_comb, '--',label='inferred') #???
plt.legend(fontsize=20)
plt.title('state $\pi$', fontsize=20)

# %% check with altered prior
reps = 10
pxy_diff = np.zeros((reps, 4,4))
for rr in range(reps):
    f_,r_,w_ = np.random.rand(3)*1 #+ (f,r,w)
    param_prior = (f_,r,w_)
    result = minimize(objective_comb, beta0, args=(g_bar, param_prior),method='Powell',bounds=[(0,100)]*num_const)  #SLSQP
    beta_inf = result.x
    M_comb, vrx, vlx, lamb = M_tilt_comb(beta_inf, param_prior)
    post_k_comb = posterior_k_comb(beta_inf, vrx, lamb, param_prior)
#    M_comb, vrx, vlx, lamb = M_tilt(beta_inf[4:10], param_prior)
#    post_k_comb = posterior_k(beta_inf[4:10], vrx, lamb, param_prior)
    pxy_diff[rr,:,:] = np.abs(post_k_comb - post_k_true) / post_k_true
#    pxy_diff[rr,:,:] = np.abs(post_k_comb - post_k_true)**2

plt.figure()
plt.imshow((np.mean(pxy_diff,0)), aspect='auto')
plt.colorbar()
#plt.title('<inferred-true Pxy>', fontsize=30)
plt.title('proportional error Pxy (ratio)', fontsize=30)


# %% debugging to check if observables are returned
### use inferred pi!!!!!
pi_00_, pi_01_, pi_10_, pi_11_ = pi_ss_inf_comb ## use inferred pi
Minf = post_k_comb*1
eta_00_10_inf = pi_00_*Minf[0,2] + pi_10_*Minf[2,0]
eta_00_01_inf = pi_00_*Minf[0,1] + pi_01_*Minf[1,0]
eta_01_11_inf = pi_01_*Minf[1,3] + pi_11_*Minf[3,1]
eta_10_01_inf = pi_10_*Minf[2,1] + pi_01_*Minf[1,2]
eta_10_11_inf = pi_10_*Minf[2,3] + pi_11_*Minf[3,2]

data = np.array([[eta_00_10_inf, eta_00_01_inf, eta_01_11_inf, eta_10_01_inf, eta_10_11_inf],
                 [eta_00_10, eta_00_01, eta_01_11, eta_01_10, eta_10_11]])

width = 0.35
x = np.arange(data.shape[1])
fig, ax = plt.subplots()
lab = ['true', 'inferred']
for i, row in enumerate(data):
    ax.bar(x + i * width, row, width, label=lab[i])#f'Group {i+1}')
  
ax.set_xticks(x + width * 0.5)
ax.set_xticklabels(['$\eta$_00_10','$\eta$_00_01','$\eta$_01_11', '$\eta$_01_10', '$\eta$_10_11'], rotation=45)
ax.legend(fontsize=20)

# %% compute Js
J_10_00_inf = pi_10_*Minf[2,0] - pi_00_*Minf[0,2]
J_01_11_inf = pi_01_*Minf[1,3] - pi_11_*Minf[3,1]
J_00_11_inf = pi_00_*Minf[0,3] - pi_11_*Minf[3,0]


data_j = np.array([[J_10_00_inf, J_01_11_inf, J_00_11_inf],
                  [J_10_00, J_01_11, J02]])

width = 0.35
x = np.arange((data_j).shape[1])
fig, ax = plt.subplots()
lab = ['true', 'inferred']
for i, row in enumerate(data_j):
    ax.bar(x + i * width, row, width, label=lab[i])#f'Group {i+1}')
  
ax.set_xticks(x + width * 0.5)
ax.set_xticklabels(['J_10_00','J_01_11','J02',], rotation=45)
ax.legend(fontsize=20)

# %%
###############################################################################
# %% error tracking
#log_power = np.arange(1,15,3)
#log_values_integer = np.round(log_power).astype(int)
#tols = 1/10**log_values_integer
#err_tol = np.zeros(len(tols))
#
#for ll in range(len(tols)):
#    ### optimization
#    beta0 = np.random.rand(num_const)*1 # initialize conjugates
#    result = minimize(objective_comb, beta0, args=(g_bar, param_prior),tol=tols[ll],method='L-BFGS-B',bounds=[(0,100)]*num_const)  #L-BFGS-B # Powell
#    beta_inf = result.x
#    result = minimize(objective_comb, beta0, args=(g_bar, param_true),tol=tols[ll],method='L-BFGS-B',bounds=[(0,100)]*num_const)  #SLSQP
#    beta_true = result.x
#    ### reconstruction
#    M_comb, vrx, vlx, lamb = M_tilt_comb(beta_inf, param_prior)
#    post_k_comb = posterior_k_comb(beta_inf, vrx, lamb, param_prior)
#    pi_ss_inf_comb = get_stationary(post_k_comb)
#    
#    err_tol[ll] = np.mean((post_k_comb - post_k_true)**2)
#
#
#plt.figure()
#plt.semilogy(-np.log(tols), err_tol,'-o')
#plt.xlabel('-log tolerance', fontsize=30)
#plt.ylabel('error', fontsize=30)

# %%
# Scan through priors and find worse or better ones if they have structure
# back calculating f,r,w from transition matrix might not be unique: use back calculation or numeric
# complete the inference: pick one more eta that is not 01-10 and two more J that is not

# Generate data and discuss fluctuation with finite data

# %%
fs = np.arange(0.1,1,.2)
ws = np.arange(0.1,1,.2)
errs = np.zeros((len(fs), len(ws)))
reps = 10

for ff in range(len(fs)):
    for ww in range(len(ws)):
        pxy_diff = np.zeros(reps)
        for rr in range(reps):
            f_,r_,w_ = fs[ff], r*1, ws[ww]
            param_prior = (f_,r,w_)
            result = minimize(objective_comb, beta0, args=(g_bar, param_prior),method='Powell',bounds=[(0,100)]*num_const)  #SLSQP
            beta_inf = result.x
            M_comb, vrx, vlx, lamb = M_tilt_comb(beta_inf, param_prior)
            post_k_comb = posterior_k_comb(beta_inf, vrx, lamb, param_prior)
#            pxy_diff[rr] = np.mean((post_k_comb - post_k_true)**2)
            pxy_diff[rr] = np.mean(np.abs(post_k_comb - post_k_true)/post_k_true)
        errs[ff,ww] = np.mean(pxy_diff)
        
# %%
plt.figure()
plt.imshow(errs, aspect='auto')
plt.xlabel('f', fontsize=20)
plt.ylabel('w', fontsize=20)
plt.title('error given f,w; burst only', fontsize=20)
#plt.title('error given f,w; bust+marginal', fontsize=20)
#plt.title('error given f,w; all observables', fontsize=20)
plt.colorbar()
plt.xticks(range(len(fs)), [round(num, 1) for num in fs])
plt.yticks(range(len(ws)), [round(num, 1) for num in ws])