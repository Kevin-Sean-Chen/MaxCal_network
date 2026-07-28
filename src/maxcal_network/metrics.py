"""Metrics for inferred network parameters."""

import numpy as np
from scipy.stats import pearsonr

from .dynamics import get_stationary, param2M

def EP(kij):
    """
    given transition matrix, compute entropy production
    """
    pi = get_stationary(kij)
    # kij = Pij / pi[:,None]
    eps = 1e-20
    ep = 0
    n = len(pi)
    for ii in range(n):
        for jj in range(n):
            Pij = pi[ii]*kij[ii,jj]
            Pji = pi[jj]*kij[jj,ii]
            if ii is not jj:
                ep += Pij*(np.log(Pij+eps)-np.log(Pji+eps))
    return ep 

def corr_param(param_true, param_infer, mode='binary'):
    """
    Peerson's  correlation berween true and inferred parameters
    """
    if mode=='binary':
        true_temp = param_true*0 - 0
        infer_temp = param_infer*0 - 0
        true_temp[param_true>0] = 1
        infer_temp[param_infer>0] = 1
        true_temp[param_true<0] = -1
        infer_temp[param_infer<0] = -1
        corr = np.dot(true_temp, infer_temp)/np.linalg.norm(true_temp)/np.linalg.norm(infer_temp)
        # corr, _ = pearsonr(true_temp, infer_temp)
        # corr = np.mean(true_temp == infer_temp)
        return corr
    else:
        true_temp = param_true*1
        infer_temp = param_infer*1        
        correlation_coefficient, _ = pearsonr(true_temp, infer_temp)
        return correlation_coefficient

def sign_corr(param_true, param_infer):
    """
    given parameters for CTMC, turn it into matric, then into the effective coupling weights
    """
    w_true = M2weights(param_true)
    w_infer = M2weights(param_infer)
    corr = corr_param(w_true, w_infer, 'binary')
    return corr
    
def M2weights(param):
    M_inf, _ = param2M(param)
    f1,f2,f3 = M_inf[0,4], M_inf[0,2], M_inf[0,1]
    w12,w13,w21 = np.log(M_inf[4,6]/f2), np.log(M_inf[4,5]/f3), np.log(M_inf[2,6]/f1)
    w23,w32,w31 = np.log(M_inf[2,3]/f3), np.log(M_inf[1,3]/f2), np.log(M_inf[1,5]/f1)
    weights = np.array([w12,w13,w21,w23,w32,w31])
    return weights

def cos_ang(v1,v2):
    return np.dot(v1,v2) / (np.linalg.norm(v1)* np.linalg.norm(v2))

