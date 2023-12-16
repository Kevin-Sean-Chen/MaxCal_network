# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 19:14:11 2023

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

# %% model parameter setup
### rate parameters
u, d = 0.3, 0.7
l0, r0 = 0.15, 0.1  # l0,r0 < 1-u
l1, r1 = 0.1, 0.15  # l1,r1 < 1-d
### network state
N = 2  # number of neurons
spins = [0,1]  # binary patterns
combinations = list(itertools.product(spins, repeat=N))  # possible configurations
nc = len(combinations)  # number of patterns

# %% compute steady-state observations  WRONG!!!!
M = np.array([[-u-r0, d, l0, 0],
              [r0, 0, -u-l0, d],
              [u, -d-r1, 0, l1],
              [0, r1, u, -d-l1],
              [1,1,1,1],
              [1,0,1,0],
              [0,1,0,1]])  # design matrix, with rate eqn, sum conservation, and steady-state conditions
xx = np.array([0,0,0,0,1, d/(u+d), u/(u+d)])
def solve_non_square_linear_equation(A, b):
    x, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return x
### compute pi
pi_ss = solve_non_square_linear_equation(M, xx)  # meet all constraints for steady-state

# %% eigenvalue problem for steadt-state (Discrete-time formulism!)
M = np.array([[1-r0-u, r0, u, 0],
              [l0, 1-u-l0, 0, u],
              [d, 0, 1-d-r1, r1],
              [0, d, l1, 1-d-l1]]).T
uu,vv = np.linalg.eig(M)
zeros_eig_id = np.argmin(np.abs(uu-1))
pi_ss = vv[:,zeros_eig_id] / np.sum(vv[:,zeros_eig_id])  # this matches pi_ss from the systems eqn. above

### asymmetry J
pi_00, pi_01, pi_10, pi_11 = pi_ss  # note that this is the correct order of 'combinations'!
J = pi_00*u - pi_10*d

### compute eta
eta_00_10 = (pi_00*u + pi_10*d)#/(u+d)
eta_01_11 = (pi_01*u + pi_11*d)#/(u+d)
eta_00_01 = (pi_00*r0 + pi_01*l0)#/(r0+l0)
eta_10_11 = (pi_10*r1 + pi_11*l1)#/(r1+l1)

# %% construct MaxCal constraints
def g_xy_exp1(x, y, a=0):
    """
    constraint: pi_00 * u - P_00,10 = 0
    """
    f_ij, g_ij_kl = 0,0
    if (x==(0,0)):
        f_ij = u
    if (x==(0,0)) and (y==(1,0)):
        g_ij_kl = -1
    gxy = f_ij + g_ij_kl - a
    return gxy

def g_xy_exp1_2(x, y, a=0):
    f_ij, g_ij_kl = 0,0
    if (x==(0,1)):
        f_ij = u
    if (x==(0,1)) and (y==(1,1)):
        g_ij_kl = -1
    gxy = f_ij + g_ij_kl - a
    return gxy
def g_xy_exp1_3(x, y, a=0):
    f_ij, g_ij_kl = 0,0
    if (x==(1,0)):
        f_ij = d
    if (x==(1,0)) and (y==(0,0)):
        g_ij_kl = -1
    gxy = f_ij + g_ij_kl - a
    return gxy
def g_xy_exp1_4(x, y, a=0):
    f_ij, g_ij_kl = 0,0
    if (x==(1,1)):
        f_ij = d
    if (x==(1,1)) and (y==(0,1)):
        g_ij_kl = -1
    gxy = f_ij + g_ij_kl - a
    return gxy

def g_xy_exp2(x, y, a):
    """
    constraint: pi_00 + pi_10 = pi_{Y=0}
    """
    f_ij, g_ij_kl = 0,0
    if (x==(0,0)) or (x==(1,0)):
        f_ij = 1
    gxy = f_ij + g_ij_kl - a
    return gxy

def g_xy_exp3(x, y, a):
    """
    constraint: eta_00,01 + eta_10,11 = pi_Y0 * r_eff + pi_Y1 * l_eff
    """
    f_ij, g_ij_kl = 0,0
    if (x==(0,0)) and (y==(0,1)):
        g_ij_kl = 1
    if (x==(0,1)) and (y==(0,0)):
        g_ij_kl = 1
    if (x==(1,0)) and (y==(1,1)):
        g_ij_kl = 1
    if (x==(1,1)) and (y==(1,0)):
        g_ij_kl = 1
    gxy = f_ij + g_ij_kl - a
    return gxy

### MaxCal functions ### (only changed lamb_beta, compared to MaxCal_beta tests)
def lamb_beta(beta):
    # return largest eigen value given beta vector
    Mxy_ = Mxy*1
    Gxy = np.zeros((nc, nc, num_const))
    for ii in range(nc):
        for jj in range(nc):
            for cc in range(num_const):
                ### gathering the list of patterns and observables, and feed to the function list
                Gxy[ii,jj,cc] = constraint_list[cc](combinations[ii] , combinations[jj], obs_list[cc])
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
    obj = np.dot(beta, g_bar) - np.log(lamb)
    return -obj # do scipy.minimization on this

def posterior_Pxy(beta_star):
    # return posterior joist matrix given beta*
    _,Pxy,_ = lamb_beta(beta_star)
    return Pxy

def posterior_Pyx(beta_star):
    # return posterior transition matrix given beta*
    _,Pxy,Pyx = lamb_beta(beta_star)
    return Pyx

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
    uu, vr = np.linalg.eig(M)  # right vector
    u2, vl = np.linalg.eig(M.T)  # left vectors
    lamb,lp = np.max(np.real(uu)), np.argmax(np.real(uu))  # max real eigen value
    vrx, vlx = np.real(vr[:,lp]), np.real(vl[:,lp])
    vec_dot_norm = np.sqrt(np.abs(np.dot(vrx, vlx)))  # normalization factor for unit dot product
    vrx, vlx = np.abs(vrx)/vec_dot_norm, np.abs(vlx)/vec_dot_norm  # normalize eigen vectors
    pix = (vrx)*vlx/(lamb)
    return pix, lamb

# %% try MaxCal optimization!!
### but maybe check with Peter first~ (done!)
### setup to prior matrix
Mxy = np.random.rand(nc, nc) + np.eye(nc)*0  #  4x4-pattern transition matrix
Mxy = Mxy / Mxy.sum(1)[:,None]  # normalize as transition matrix

### gather observables Os and functions g_i(x,y)
beta = np.array([.1, .1, .1])#,   .1,.1,.1]) # an arbitrary multiplier for now
num_const = len(beta)
constraint_list = [g_xy_exp1, g_xy_exp2, g_xy_exp3]#,   g_xy_exp1_2,g_xy_exp1_3,g_xy_exp1_4]
#obs_list = []  ### fill this out: 0, pi_{Y=0},  pi_Y0*r_eff + pi_Y1*l_eff
pi_y0 = (pi_10*d+J)/u + pi_10
traffic_prob = eta_00_01 +eta_１0_11 # which equals pi_Y0*r_eff + pi_Y1*l_eff
obs_list = [0, pi_y0, traffic_prob]#,   0,0,0]

### computing with exp(beta*g(x,y))

Mxy_ = Mxy*1
Gxy = np.zeros((nc, nc, num_const))
for ii in range(nc):
    for jj in range(nc):
        for cc in range(num_const):
            Gxy[ii,jj,cc] = constraint_list[cc](combinations[ii] , combinations[jj], obs_list[cc])
        Mxy_[ii,jj] = Mxy[ii,jj] * np.exp(Gxy[ii,jj,:]@beta)  # eqn.1 in Peter's note
lamb, Pxy, Pyx = lamb_beta(beta)

# %% optimization
beta0 = beta*1 + np.random.randn()*beta  # initialize around the true parameter
lamb,Pxy,_ = lamb_beta(beta0)
Q = np.random.rand(nc, nc)  # random transition matrix
Q = Q / Q.sum(1)[:,None] 
pix,_ = get_stationary(Pyx) # or Q for perturbation?
g_bar = np.array(obs_list)
#expect_g(Pxy)#(pix[:,None]*Q) #feed in a random matrix, not Pxy: would be differnt but same average if it works
#g_bar = init_g_bar(beta0)
# Minimize the function
result = minimize(objective, beta0, args=(g_bar),method='SLSQP',bounds=[(0,500)]*num_const)  #SLSQP
print('beta*:'+str(result.x))

post_joint = posterior_Pxy(result.x)  # compute posterior transition matrix given inferred beta*
post_tran = posterior_Pyx(result.x)

# %% checking results
plt.figure()
plt.subplot(121)
plt.imshow(Pyx)
plt.title('True Pxy',fontsize=20)
plt.xticks(range(len(combinations)), combinations);
plt.yticks(range(len(combinations)), combinations);
plt.subplot(122)
plt.imshow(post_tran)
plt.title('MaxCal inferred',fontsize=20)
plt.xticks(range(len(combinations)), combinations);
plt.yticks(range(len(combinations)), combinations);

print('observed g:'+str(g_bar))
infer_g = expect_g(post_joint)
print('inferred g:'+str(infer_g))

plt.figure()
plt.plot(g_bar,'-o', label='true g')
plt.plot(infer_g,'--o', label='inferred g')
plt.legend(fontsize=20)

plt.figure()
plt.plot(pi_ss,'-o', label=r'$\pi$')
plt.plot(post_tran@pi_ss,'--o', label='$P\pi$')
plt.legend(fontsize=20)
plt.ylim([0,1])
# %%
### pseudo-code
# set prior matix
# compute M_xy with exp(beta*g(x,y))
# compute posterior
# optimize for beta...
