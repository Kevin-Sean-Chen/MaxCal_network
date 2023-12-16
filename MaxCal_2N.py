# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 15:50:03 2023

@author: kevin
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
import itertools
from scipy.optimize import minimize

import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

###
# 2-neuron example with MaxCal inference
###

# %% 2-neuron setup
# firing, refractory, and coupling parameters
f,r,w = 0.3, 0.9, 0.5  # keep this symmetric for the first test
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
    return M, pi_ss
M, pi_ss = sym_M(f,r,w)
print(pi_ss) # lesson: right eigen vector of a transition matrix is uniform

# %% compute observables
pi_00, pi_01, pi_10, pi_11 = pi_ss  # steady-state distribution

eta_00_01 = pi_00*f*(1-f) + pi_01*(1-w)*r  # symmetric trafficing
eta_00_10 = pi_00*f*(1-f) + pi_10*(1-w)*r
eta_00_11 = pi_00*f*f + pi_11*r*r
eta_01_11 = pi_01*w*(1-r) + pi_11*r*(1-r)
eta_10_11 = pi_10*w*(1-r) + pi_11*(r-1)*r
eta_01_10 = w*r*(pi_01 + pi_10)

piB0 = pi_00*1
piB2 = pi_11*1
eta_01 = eta_00_01 + eta_00_10
eta_12 = eta_01_11 + eta_10_11
eta_02 = eta_00_11*1
J02 = pi_00*f*f - pi_11*r*r  # P_{0,2} - P_{2,0}, according to the notes

# %% a prior matrix with 'guessed' parameters
f_,r_,w_ = np.random.rand(3)*1
M_prior, pi_ss_prior = sym_M(f_,r_,w_)
param_prior = (f_,r_,w_)  # guessing parameters as prior

# %% posterior matrix
def M_tilt(beta, param):
    """
    Titled matrix
    """
    f,r,w = param  # fill in wit ground-truth or prior guess
    alpha0, alpha2, beta01, beta12, beta02, gamm = beta  # conjugated variables
    M_ = np.array([[(1-f)**2*np.exp(alpha0), f*(1-f)*np.exp(alpha0+beta01), f*(1-f)*np.exp(alpha0+beta01), f**2*np.exp(alpha0+beta02+gamm)],
                   [(1-w)*r*np.exp(beta01), (1-w)*(1-r), w*r, w*(1-r)*np.exp(beta12)],
                   [(1-w)*r*np.exp(beta01), w*r, (1-w)*(1-r), w*(1-r)*np.exp(beta12)],
                   [r**2*np.exp(alpha2+beta02-gamm), r*(1-r)*np.exp(alpha2+beta12), r*(1-r)*np.exp(alpha2+beta12), (1-r)**2*np.exp(alpha2)]])
    uu, vr = np.linalg.eig(M_)  # right vector
    u2, vl = np.linalg.eig(M_.T)  # left vectors
    lamb,rp,lp = np.max(np.real(uu)), np.argmax(np.real(uu)), np.argmax(np.real(u2))  # max real eigen value
    vrx, vlx = np.real(vr[:,rp]), np.real(vl[:,lp])
    vec_dot_norm = np.sqrt(np.abs(np.dot(vrx, vlx)))  # normalization factor for unit dot product
    vrx, vlx = np.abs(vrx)/vec_dot_norm, np.abs(vlx)/vec_dot_norm  # normalize eigen vectors
    return M_, vrx, vlx, lamb

def posterior_k(beta, vrx, lamb, param):
    """
    Posterior transition matrix
    """
    f,r,w = param
    alpha0, alpha2, beta01, beta12, beta02, gamm = beta  # conjugated variables
    r1,r2,r3,r4 = vrx  # eigen vectors(?)
    k_ = np.array([[(1-f)**2*np.exp(alpha0)/lamb, f*(1-f)*np.exp(alpha0+beta01)*r2/(r1*lamb), f*(1-f)*np.exp(alpha0+beta01)*r3/(r1*lamb), f**2*np.exp(alpha0+beta02+gamm)*r4/(r1*lamb)],
                   [(1-w)*r*np.exp(beta01)*r1/(r2*lamb), (1-w)*(1-r)/lamb, w*r*r3/(r2*lamb), w*(1-r)*np.exp(beta12)*r4/(r2*lamb)],
                   [(1-w)*r*np.exp(beta01)*r1/(r3*lamb), w*r*r2/(r3*lamb), (1-w)*(1-r)/lamb, w*(1-r)*np.exp(beta12)*r4/(r3*lamb)],
                   [r**2*np.exp(alpha2+beta02-gamm)*r1/(r4*lamb), r*(1-r)*np.exp(alpha2+beta12)*r2/(r4*lamb), r*(1-r)*np.exp(alpha2+beta12)*r3/(r4*lamb), (1-r)**2*np.exp(alpha2)/lamb]])
    return k_

def objective(beta, g_bar, param):
    """
    a function of beta and g_bar, here we optimize for beta*
    """
    _,_,_, lamb = M_tilt(beta, param)
    obj = np.dot(beta, g_bar) - np.log(lamb)
    return -obj # do scipy.minimization on this

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
#    uu, vr = np.linalg.eig(M)  # right vector
#    u2, vl = np.linalg.eig(M.T)  # left vectors
#    lamb,lp = np.max(np.real(uu)), np.argmax(np.real(uu))  # max real eigen value
#    vrx, vlx = np.real(vr[:,lp]), np.real(vl[:,lp])
#    vec_dot_norm = np.sqrt(np.abs(np.dot(vrx, vlx)))  # normalization factor for unit dot product
#    vrx, vlx = np.abs(vrx)/vec_dot_norm, np.abs(vlx)/vec_dot_norm  # normalize eigen vectors
#    pix = (vrx)*vlx/(lamb)
    uu,vv = np.linalg.eig(M.T)
    zeros_eig_id = np.argmin(np.abs(uu-1))
    pix = vv[:,zeros_eig_id] / np.sum(vv[:,zeros_eig_id])
    return pix #, lamb

# %% optimization of conjugates
g_bar = np.array([piB0, piB2, eta_01, eta_12, eta_02, J02])  # 6 observables
num_const = len(g_bar)
beta0 = np.random.rand(num_const)*1  # initialize conjugates
result = minimize(objective, beta0, args=(g_bar, param_prior),method='L-BFGS-B',bounds=[(0,100)]*num_const)  #SLSQP
beta_inf = result.x
print('beta*:'+str(beta_inf))
result = minimize(objective, beta0, args=(g_bar, param_true),method='L-BFGS-B',bounds=[(0,100)]*num_const)  #SLSQP
beta_true = result.x
print('beta-true:'+str(beta_true))

# %% analyze inference
# inferred posterior matrix
M_, vrx, vlx, lamb = M_tilt(beta_inf, param_prior)
post_k = posterior_k(beta_inf, vrx, lamb, param_prior)
pi_ss_inf = get_stationary(post_k)

# same calculation if we have the true parameters
#_, vrx, _, lamb = M_tilt(beta_inf, param_true)
#post_k_true = posterior_k(beta_inf, vrx, lamb, param_true)
post_k_true,_ = sym_M(f,r,w)

plt.figure()
plt.subplot(121)
plt.imshow(post_k, aspect='auto')
plt.title('inferred Pxy',fontsize=20)
plt.subplot(122)
plt.imshow(post_k_true, aspect='auto')
plt.title('true Pxy',fontsize=20)

plt.figure()
plt.plot(pi_ss, label='true')
plt.plot(pi_ss_inf, label='inferred') #???
plt.legend(fontsize=20)


### 08.24.23: seems that two observables are degenerate
### Peter would come back to me:)
# %% checking second posterior from the inferred parameters
### numerical method
def objective_param(param_inf, post_k1):
    f2,r2,w2 = param_inf
    post_k2,_ = sym_M(f2, r2, w2)
#    post_k2 = posterior_k(beta_inf, vrx, lamb, param_inf)
    loss = np.mean((post_k1-post_k2)**2)
    return loss
param0 = np.random.rand(3)
result = minimize(objective_param, param0, args=(post_k),method='L-BFGS-B',bounds=[(0,100)]*3)  #SLSQP
param_inf = result.x
f2,r2,w2 = param_inf
M2,_ = sym_M(f2, r2, w2)

### Peter's analytic method
f2 = np.sqrt((eta_02 + J02)/pi_00)
r2 = np.sqrt((eta_02 - J02)/pi_11)
piB1 = pi_01 + pi_10
w2 = (eta_12-J02)/piB1 * 1/(1-np.sqrt((eta_02-J02)/piB2))
M2,_ = sym_M(f2, r2, w2)

plt.figure()
plt.subplot(121)
plt.imshow(post_k, aspect='auto')
plt.title('k*',fontsize=20)
plt.subplot(122)
plt.imshow(M2, aspect='auto')
plt.title('M*',fontsize=20)

# %% Marginal observables
###############################################################################
# %%
# new functions
def M_tilt_marg(beta, param):
    """
    Titled matrix
    """
    f,r,w = param  # fill in wit ground-truth or prior guess
    alpha1, alpha2, beta1, beta2 = beta  # conjugate variable
    M_ = np.array([[(1-f)**2*np.exp(alpha1+alpha2), f*(1-f)*np.exp(alpha1+alpha2+beta2), f*(1-f)*np.exp(alpha1+alpha2+beta1), f**2*np.exp(beta1+alpha1+alpha2+beta2)],
                   [(1-w)*r*np.exp(alpha1+beta2), (1-w)*(1-r)*np.exp(alpha1), w*r*np.exp(alpha1+beta1+beta2), w*(1-r)*np.exp(alpha1+beta1)],
                   [(1-w)*r*np.exp(alpha2+beta1), w*r*np.exp(alpha2+beta1+beta2), (1-w)*(1-r)*np.exp(alpha2), w*(1-r)*np.exp(alpha2+beta2)],
                   [r**2*np.exp(beta1+beta2), r*(1-r)*np.exp(beta1), r*(1-r)*np.exp(beta2), (1-r)**2]])
    uu, vr = np.linalg.eig(M_)  # right vector
    u2, vl = np.linalg.eig(M_.T)  # left vectors
    lamb,rp,lp = np.max(np.real(uu)), np.argmax(np.real(uu)), np.argmax(np.real(u2))  # max real eigen value
    vrx, vlx = np.real(vr[:,rp]), np.real(vl[:,lp])
    vec_dot_norm = np.sqrt(np.abs(np.dot(vrx, vlx)))  # normalization factor for unit dot product
    vrx, vlx = np.abs(vrx)/vec_dot_norm, np.abs(vlx)/vec_dot_norm  # normalize eigen vectors
    return M_, vrx, vlx, lamb

def posterior_k_marg(beta, vrx, lamb, param):
    r1,r2,r3,r4 = vrx  # eigen vectors(?)
    M_,_,_,_ = M_tilt_marg(beta, param)
    pref = np.array([[1/lamb, r2/(r1*lamb), r3/(r1*lamb), r4/(r1*lamb)],
                     [r1/(r2*lamb), 1/lamb, r3/(r2*lamb), r4/(r2*lamb)],
                     [r1/(r3*lamb), r2/(r3*lamb), 1/lamb, r4/(r3*lamb)],
                     [r1/(r4*lamb), r2/(r4*lamb), r3/(r4*lamb), 1/lamb]])
    k_ = pref*M_
    return k_

def objective_marg(beta, g_bar, param):
    """
    a function of beta and g_bar, here we optimize for beta*
    """
    _,_,_, lamb = M_tilt_marg(beta, param)
    obj = np.dot(beta, g_bar) - np.log(lamb)
    return -obj # do scipy.minimization on this
#
## %% MaxCal inference
# neuron 1
pi_x1 = pi_00 + pi_01
eta_x1 = eta_00_11 + eta_01_10 + eta_01_11 + eta_00_10
# neuron 2
pi_x2 = pi_00 + pi_10
eta_x2 = eta_00_01 + eta_01_10 + eta_10_11 + eta_00_11

g_bar = np.array([pi_x1, pi_x2, eta_x1, eta_x2])  # marginal observables
num_const = len(g_bar)
beta0 = np.random.rand(num_const)*.1 # initialize conjugates
result = minimize(objective_marg, beta0, args=(g_bar, param_prior),method='L-BFGS-B',bounds=[(0,100)]*num_const)  #SLSQP
beta_inf = result.x
print('beta*:'+str(beta_inf))
result = minimize(objective_marg, beta0, args=(g_bar, param_true),method='L-BFGS-B',bounds=[(0,100)]*num_const)  #SLSQP
beta_true = result.x
print('beta-true:'+str(beta_true))

# %% analyze inference
# inferred posterior matrix
M_marg, vrx, vlx, lamb = M_tilt_marg(beta_inf, param_prior)
post_k_marg = posterior_k_marg(beta_inf, vrx, lamb, param_prior)
pi_ss_inf_marg = get_stationary(post_k_marg)

# same calculation if we have the true parameters
post_k_true,_ = sym_M(f,r,w)

plt.figure()
plt.subplot(121)
plt.imshow(post_k_marg, aspect='auto')
plt.title('inferred Pxy',fontsize=20)
plt.subplot(122)
plt.imshow(post_k_true, aspect='auto')
plt.title('true Pxy',fontsize=20)

plt.figure()
plt.plot(pi_ss, label='true')
plt.plot(pi_ss_inf_marg, label='inferred') #???
plt.legend(fontsize=20)

# %% combine all (burst and marginals)??
def M_tilt_comb(beta, param):
    """
    Titled matrix
    """
    f,r,w = param  # fill in wit ground-truth or prior guess
    alpha0, alpha2b, beta01, beta12, beta02, gamm,\
                    alpha1, alpha2, beta1, beta2,\
                    eta1, eta2, j1, j2 = beta  # conjugated variables, one etas and two Js
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
                               [j2,eta1,0,eta2],
                               [0,-j1,eta2,0]]))  ### figure this out
    M_ = M_burst*M_marg*M_extra
    uu, vr = np.linalg.eig(M_)  # right vector
    u2, vl = np.linalg.eig(M_.T)  # left vectors
    lamb,rp,lp = np.max(np.real(uu)), np.argmax(np.real(uu)), np.argmax(np.real(u2))  # max real eigen value
    vrx, vlx = np.real(vr[:,rp]), np.real(vl[:,lp])
    vec_dot_norm = np.sqrt(np.abs(np.dot(vrx, vlx)))  # normalization factor for unit dot product
    vrx, vlx = np.abs(vrx)/vec_dot_norm, np.abs(vlx)/vec_dot_norm  # normalize eigen vectors
    return M_, vrx, vlx, lamb

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

# %% MaxCal!!!
g_bar_burst = np.array([piB0, piB2, eta_01, eta_12, eta_02, J02])  # 6 observables from bursting
g_bar_marg = np.array([pi_x1, pi_x2, eta_x1, eta_x2])  # marginal observables

J_01_11 = pi_01*(1-r)*w - pi_11*r*(1-r)
J_10_00 = pi_10*(1-r)*w - pi_00*f*(1-f)
g_bar_extra = np.array([eta_01_10, eta_10_11, J_01_11, J_10_00])  # one more eta and two more Js

g_bar = np.concatenate((g_bar_burst, g_bar_marg, g_bar_extra))
num_const = len(g_bar)
beta0 = np.random.rand(num_const)*1 # initialize conjugates
result = minimize(objective_comb, beta0, args=(g_bar, param_prior),method='L-BFGS-B',bounds=[(0,100)]*num_const)  #SLSQP
beta_inf = result.x
print('beta*:'+str(beta_inf))
result = minimize(objective_comb, beta0, args=(g_bar, param_true),method='L-BFGS-B',bounds=[(0,100)]*num_const)  #SLSQP
beta_true = result.x
print('beta-true:'+str(beta_true))

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
reps = 50
pxy_diff = np.zeros((reps, 4,4))
for rr in range(reps):
    f_,r_,w_ = np.random.rand(3)*1 #+ (f,r,w)
    param_prior = (f_,r_,w_)
    result = minimize(objective_comb, beta0, args=(g_bar, param_prior),method='L-BFGS-B',bounds=[(0,100)]*num_const)  #SLSQP
    beta_inf = result.x
    M_comb, vrx, vlx, lamb = M_tilt_comb(beta_inf, param_prior)
    post_k_comb = posterior_k_comb(beta_inf, vrx, lamb, param_prior)
#    M_comb, vrx, vlx, lamb = M_tilt(beta_inf[4:10], param_prior)
#    post_k_comb = posterior_k(beta_inf[4:10], vrx, lamb, param_prior)
    pxy_diff[rr,:,:] = (post_k_comb - post_k_true) #/ post_k_true

plt.figure()
plt.imshow((np.mean(pxy_diff,0)), aspect='auto')
plt.colorbar()
plt.title('<inferred-true Pxy>', fontsize=30)


# %% debugging to check if observables are returned
Minf = post_k_comb*1
eta_00_10_inf = pi_00*Minf[0,2] + pi_10*Minf[2,0]
eta_00_01_inf = pi_00*Minf[0,1] + pi_01*Minf[1,0]
eta_01_11_inf = pi_01*Minf[1,3] + pi_11*Minf[3,1]
eta_10_01_inf = pi_10*Minf[2,1] + pi_01*Minf[1,2]
eta_10_11_inf = pi_10*Minf[2,3] + pi_11*Minf[3,2]

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

# %%
# Scan through priors and find worse or better ones if they have structure
# back calculating f,r,w from transition matrix might not be unique: use back calculation or numeric
# complete the inference: pick one more eta that is not 01-10 and two more J that is not

# %%
###############################################################################
# %%
###############################################################################
# %% non-analytic MaxCal test with other observables
# using functions from MaxCal_sens for now... (if it matters I should start creating import functions)

#def g_xy_exp1(x, y, a=0):
#    """
#    constraint: pi_00 * u - P_00,10 = 0
#    """
#    f_ij, g_ij_kl = 0,0
#    if (x==(0,0)):
#        f_ij = f*(1-f)
#    if (x==(0,0)) and (y==(1,0)):
#        g_ij_kl = -1
#    gxy = f_ij + g_ij_kl - a
#    return gxy
#
#### gather observables Os and functions g_i(x,y)
#beta = np.array([.1, .1, .1])#,   .1,.1,.1]) # an arbitrary multiplier for now
#num_const = len(beta)
#constraint_list = [g_xy_exp1, g_xy_exp2, g_xy_exp3]#,   g_xy_exp1_2,g_xy_exp1_3,g_xy_exp1_4]
##obs_list = []  ### fill this out: 0, pi_{Y=0},  pi_Y0*r_eff + pi_Y1*l_eff
#pi_y0 = pi_00 + pi_10
#traffic = eta_00_01 + eta_10_11
#obs_list = [0, pi_y0, traffic]#,   0,0,0]
#
#### computing with exp(beta*g(x,y))
##Mxy_ = post_k*1
##Gxy = np.zeros((nc, nc, num_const))
##for ii in range(nc):
##    for jj in range(nc):
##        for cc in range(num_const):
##            Gxy[ii,jj,cc] = constraint_list[cc](combinations[ii] , combinations[jj], obs_list[cc])
##        Mxy_[ii,jj] = Mxy[ii,jj] * np.exp(Gxy[ii,jj,:]@beta)  # eqn.1 in Peter's note
##lamb, Pxy, Pyx = lamb_beta(beta)
#
## %% optimization
#beta0 = beta*1 + np.random.randn()*beta  # initialize around the true parameter
#lamb,Pxy,_ = lamb_beta(beta0)
#g_bar = np.array(obs_list)
##expect_g(Pxy)#(pix[:,None]*Q) #feed in a random matrix, not Pxy: would be differnt but same average if it works
##g_bar = init_g_bar(beta0)
## Minimize the function
#result = minimize(objective, beta0, args=(g_bar),method='SLSQP',bounds=[(0,500)]*num_const)  #SLSQP
#print('beta*:'+str(result.x))
#
#post_joint = posterior_Pxy(result.x)  # compute posterior transition matrix given inferred beta*
#post_tran = posterior_Pyx(result.x)
#
## %% checking results
#plt.figure()
#plt.subplot(121)
#plt.imshow(M)
#plt.title('True transition',fontsize=20)
#plt.xticks(range(len(combinations)), combinations);
#plt.yticks(range(len(combinations)), combinations);
#plt.subplot(122)
#plt.imshow(post_tran)
#plt.title('MaxCal inferred',fontsize=20)
#plt.xticks(range(len(combinations)), combinations);
#plt.yticks(range(len(combinations)), combinations);