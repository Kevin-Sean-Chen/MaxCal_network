# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 00:48:21 2024

@author: kevin
"""

from matplotlib import pyplot as plt
import scipy as sp
import numpy as np

import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

# %% F(kappa) analysis
N = 30
chi = 50
M = np.random.randn(N,N)
uu,ss,vv = np.linalg.svd(M)

m = uu[:,0]
n = uu[:,1] + uu[:,0]*1.
I = uu[:,3]*0 + uu[:,1]*1 #np.random.randn(N) #np.random.randn(N) #uu[:,3]*1 + uu[:,1]*1
m,n,I = m/np.linalg.norm(m)*chi, n/np.linalg.norm(n)*chi, I/np.linalg.norm(I)*chi

def F_k(kappa):
    output = np.dot(n, np.tanh(m*kappa + I))/N
    return output

kappa = np.arange(-10,10, .1)
fk = np.array([F_k(k) for k in kappa])
plt.figure()
plt.plot(kappa, fk)
plt.plot(fk,fk)

# %% dynamics
### revisiting bistable low-rank RNN
# setup
J_lr = 1/N * m[:,None]@n[None,:]
sig = 50
tau = 1
tau_n = 10

# network dynamics
T = 2000
dt = 0.1
time = np.arange(0,T,dt)
lt = len(time)
xt = np.zeros((N,lt))
rt = np.zeros((N,lt))
eta = np.zeros(lt)
for tt in range(lt-1):
    xt[:,tt+1] = xt[:,tt] + dt/tau*(-xt[:,tt] + J_lr @ np.tanh(xt[:,tt]) + I*eta[tt])
                                    #I*np.random.randn(N)*sig*dt**0.5)
    rt[:,tt] = xt[:,tt]
    eta[tt+1] = eta[tt] + dt/tau_n*(-eta[tt] + np.random.randn()*sig*dt**0.5)

plt.figure()
plt.imshow(xt, aspect='auto')

kappa = m @ rt/N/chi
plt.figure()
plt.plot(time, kappa)

# %% dwell time analysis for states
# scale noise or overlap with dwell time...


# %% connect to psychometric curve and noise

# %% connect to state-space models

