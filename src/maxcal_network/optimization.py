"""Maximum-caliber objectives and constraints."""

import numpy as np

from .dynamics import P_frw_ctmc, edge_flux_inf, get_stationary, param2M

def MaxCal_D(kij, kij0, param):
    """
    KL devergence term, with transition Pij and prior rate kij0 as input
    This term can be unstable in log!
    """
    pi = get_stationary(kij)
    # kij = Pij / pi[:,None]
    eps = 1e-30
    kl = 0
    n = len(pi)
    for ii in range(n):
        for jj in range(n):
            Pij = pi[ii]*kij[ii,jj]
            if ii is not jj:
                kl += Pij*(np.log(Pij+eps)-np.log(pi[ii]*kij0[ii,jj]+eps)) \
                      + pi[ii]*kij0[ii,jj] - Pij
    return kl

def C_P(P, observations, param, Cp_condition):
    """
    The C(P,f,r,w) function for constraints
    """
    _,pi = param2M(param)
    flux_ij = edge_flux_inf(param)
    # trick to make all observable a vector and shield with Cp conditions!
    obs_all = np.concatenate((pi, flux_ij.reshape(-1)))
    # obs_all = np.concatenate((flux_ij.reshape(-1), pi))
    cp_dof = obs_all * Cp_condition
    return cp_dof - (observations * Cp_condition)

def objective_param(param, kij0):
    """
    objective in the parameter space, using frw and adding extra constraints
    """
    kij,_ = param2M(param)
    D = MaxCal_D(kij, kij0, param)
    return D

def eq_constraint(param, observations, Cp_condition):
    Pxy = P_frw_ctmc(param)
    cp = C_P(Pxy, observations, param, Cp_condition)
    return 0.5*np.sum(cp**2)


