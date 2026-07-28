"""State conversion and continuous-time Markov chain operations."""

import itertools

import numpy as np

# %% basics
N = 3
nc = 2**N
spins = [0,1]  # binary patterns
combinations = list(itertools.product(spins, repeat=N))  # possible configurations
lt = 100000

# %% make blocks for contraints
def word_id(word):
    return combinations.index(word)

def constraint_blocks_3N(ith):
    Mid = np.arange(nc**2).reshape(nc,nc)
    dofs_all = nc**2 + nc
    Cp_condition = np.zeros(dofs_all)
    ii = 1
    ### tau
    Cp_condition[:nc] = np.ones(nc)
    if ii==ith:
        return Cp_condition
    ii = ii+1
    ### f
    Cp_condition[Mid[ word_id((0,0,0)) , word_id((0,0,1)) ]+nc] = 1
    Cp_condition[Mid[ word_id((0,0,0)) , word_id((0,1,0)) ]+nc] = 1
    Cp_condition[Mid[ word_id((0,0,0)) , word_id((1,0,0)) ]+nc] = 1
    if ii==ith:
        return Cp_condition
    ii = ii+1
    ### fw
    Cp_condition[Mid[ word_id((0,0,1)) , word_id((0,1,1)) ]+nc] = 1
    Cp_condition[Mid[ word_id((0,0,1)) , word_id((1,0,1)) ]+nc] = 1
    Cp_condition[Mid[ word_id((0,1,0)) , word_id((0,1,1)) ]+nc] = 1
    Cp_condition[Mid[ word_id((0,1,0)) , word_id((1,1,0)) ]+nc] = 1
    Cp_condition[Mid[ word_id((1,0,0)) , word_id((1,0,1)) ]+nc] = 1
    Cp_condition[Mid[ word_id((1,0,0)) , word_id((1,1,0)) ]+nc] = 1
    if ii==ith:
        return Cp_condition
    ii = ii+1
    ### fw_ijk
    Cp_condition[Mid[ word_id((0,1,1)) , word_id((1,1,1)) ]+nc] = 1
    Cp_condition[Mid[ word_id((1,0,1)) , word_id((1,1,1)) ]+nc] = 1
    Cp_condition[Mid[ word_id((1,1,0)) , word_id((1,1,1)) ]+nc] = 1
    if ii==ith:
        return Cp_condition
    ii = ii+1
    ### r
    Cp_condition[Mid[ word_id((0,0,1)) , word_id((0,0,0)) ]+nc] = 1
    Cp_condition[Mid[ word_id((0,1,0)) , word_id((0,0,0)) ]+nc] = 1
    Cp_condition[Mid[ word_id((1,0,0)) , word_id((0,0,0)) ]+nc] = 1
    if ii==ith:
        return Cp_condition
    ii = ii+1
    ### ru
    Cp_condition[Mid[ word_id((0,1,1)) , word_id((0,1,0)) ]+nc] = 1
    Cp_condition[Mid[ word_id((0,1,1)) , word_id((0,0,1)) ]+nc] = 1
    Cp_condition[Mid[ word_id((1,0,1)) , word_id((0,0,1)) ]+nc] = 1
    Cp_condition[Mid[ word_id((1,0,1)) , word_id((1,0,0)) ]+nc] = 1
    Cp_condition[Mid[ word_id((1,1,0)) , word_id((1,0,0)) ]+nc] = 1
    Cp_condition[Mid[ word_id((1,1,0)) , word_id((0,1,0)) ]+nc] = 1
    if ii==ith:
        return Cp_condition
    ii = ii+1
    ### ru_ijk
    Cp_condition[Mid[ word_id((1,1,1)) , word_id((1,1,0)) ]+nc] = 1
    Cp_condition[Mid[ word_id((1,1,1)) , word_id((1,0,1)) ]+nc] = 1
    Cp_condition[Mid[ word_id((1,1,1)) , word_id((0,1,1)) ]+nc] = 1
    if ii==ith:
        return Cp_condition
    ii = ii+1
    

# %%
def spk2statetime(
    firing,
    window,
    lt=None,
    N=N,
    combinations=combinations,
    stride=1,
):
    """
    Convert firing data into network states.

    Parameters
    ----------
    firing : list
        firing[t] = [time, spike_indices]
    window : int
        Window size in bins, e.g. 20 for 20 ms if dt=1 ms.
    lt : int, optional
        Trial length in bins. The default is the firing-data length.
    stride : int
        Step size between windows.
        stride=1 gives sliding windows. default for CTMC
        stride=window gives non-overlapping windows.
    """

    if lt is None:
        lt = len(firing)

    starts = np.arange(0, lt - window + 1, stride)
    states_spk = np.zeros(len(starts), dtype=int)

    for ii, tt in enumerate(starts):
        this_window = firing[tt:tt + window]
        word = np.zeros(N)

        for ti in range(window):
            if len(this_window[ti][1]) > 0:
                this_neuron = np.random.choice(this_window[ti][1])
                if this_neuron < N:
                    word[int(this_neuron)] = 1

        state_id = combinations.index(tuple(word))
        states_spk[ii] = state_id

    # transitions between consecutive sampled states
    trans_temp = np.diff(states_spk)
    trans_idx = np.where(np.abs(trans_temp) > 0)[0]

    spk_states = states_spk[trans_idx].astype(int)

    # return transition times in original bin units
    spk_times = starts[trans_idx]

    return spk_states, spk_times

# def spk2statetime(firing, window, lt=lt, N=N, combinations=combinations):
#     """
#     given the firing (time and neuron that fired) data, we choose time window to slide through,
#     then convert to network states and the timing of transition
#     """
#     states_spk = np.zeros(lt-window)
#     for tt in range(lt-window):
#         this_window = firing[tt:tt+window]
#         word = np.zeros(N)  # template for the binary word
#         for ti in range(window): # in time
#             if len(this_window[ti][1])>0:
#                 # this_neuron = this_window[ti][1][0]  # the neuron that fired first
#                 this_neuron = np.random.choice(this_window[ti][1])
#                 if this_neuron<N:  # for only oberved!
#                     word[int(this_neuron)] = 1
#         state_id = combinations.index(tuple(word))
#         states_spk[tt] = state_id
    
#     # now compute all the transitions
#     trans_temp = np.diff(states_spk)  # find transitions
#     spk_times = np.where(np.abs(trans_temp)>0)[0]  # spike timing
#     spk_states = states_spk[spk_times].astype(int)   # spiking states
#     return spk_states, spk_times

def spk2statetime_4N(firing, window, lt=lt):
    """
    4-neuron version of spk2statetime: convert firing list into state ids and transition times.
    """
    N4 = 4
    combinations4 = list(itertools.product(spins, repeat=N4))
    states_spk = np.zeros(lt-window)
    for tt in range(lt-window):
        this_window = firing[tt:tt+window]
        word = np.zeros(N4)
        for ti in range(window):
            if len(this_window[ti][1])>0:
                this_neuron = np.random.choice(this_window[ti][1])
                if this_neuron < N4:
                    word[int(this_neuron)] = 1
        state_id = combinations4.index(tuple(word))
        states_spk[tt] = state_id

    trans_temp = np.diff(states_spk)
    spk_times = np.where(np.abs(trans_temp)>0)[0]
    spk_states = states_spk[spk_times].astype(int)
    return spk_states, spk_times

def param2M(param, N=N, combinations=combinations):
    """
    given array of parameters with length N*2**N, network size N, return transition matrix
    the matrix is a general CTMC form for 
    """
    nc = 2**N  # number of states
    
    ### idea: M = mask*FR, with mask for ctmc, FR is the rest of the transitions
    mask = np.ones((nc,nc))  # initialize the tilted matrix
    FR = mask*1
    # make the mask
    for ii in range(nc):
        for jj in range(nc):
            # Only allow one flip logic in ctmc
            if sum(x != y for x, y in zip(combinations[ii], combinations[jj])) != 1:
                mask[ii,jj] = 0
    
    # now make F matrix!
    kk = 0
    for ii in range(nc):
        for jj in range(nc):
            if mask[ii,jj]==1: #only check those that generates one spike
                FR[ii,jj] = param[kk]
                kk = kk+1  # marching forward to fill in f*exp(wij) parts... need to later invert this!!
    # print(kk)
    M = mask*FR  

    ### compute steady-state
    np.fill_diagonal(M, -np.sum(M,1))  # fill diagonal for continuous time Markov transition Q (is this correct?!)
    uu,vv = np.linalg.eig(M.T)
    zeros_eig_id = np.argmin(np.abs(uu-1))
    pi_ss = vv[:,zeros_eig_id] / np.sum(vv[:,zeros_eig_id])
    
    return M, np.real(pi_ss)

def compute_tauC(states, times, nc=nc, combinations=combinations, lt=None):
    """
    given the emperically measured states, measure occupency tau and the transitions C
    """
    tau = np.zeros(nc)
    C = np.zeros((nc,nc))
    # compute occupency time
    for i in range(len(states)-1):
        this_state = states[i]
        # if i==0:
        #     tau[this_state] += times[i]  # correct for starting
        # #### check this~~~
        # elif lt is not None and i==len(states)-1:
        #     tau[this_state] += lt - times[i+1]
        # else:
        tau[this_state] += times[i+1]-times[i]  ### total time occupancy
        
    # compute transitions
    for t in range(len(states)-1):
        ii,jj = states[t], states[t+1]
        if ii != jj:
            if sum(x != y for x, y in zip(combinations[ii], combinations[jj])) == 1:  # ignore those not CTMC for now!
                C[ii,jj] += 1  ### counting the transtion
    return tau, C    

def compute_tauC_4N(states, times, lt=None):
    """
    4-neuron version of compute_tauC: measure occupancy tau and transitions C.
    """
    N4 = 4
    nc4 = 2**N4
    combinations4 = list(itertools.product(spins, repeat=N4))
    tau = np.zeros(nc4)
    C = np.zeros((nc4,nc4))
    for i in range(len(states)-1):
        this_state = states[i]
        tau[this_state] += times[i+1]-times[i]

    for t in range(len(states)-1):
        ii,jj = states[t], states[t+1]
        if ii != jj:
            if sum(x != y for x, y in zip(combinations4[ii], combinations4[jj])) == 1:
                C[ii,jj] += 1
    return tau, C

def compute_min_isi(firing, N=N, lt=lt):
    isi = np.zeros(N)
    for nn in range(N):
        # spk_i = np.array([firing[ii][0] for ii in range(lt) if len(firing[ii][0])>0 and firing[ii][1]==nn]).squeeze()
        spk_i = []
        for tt in range(lt):
            if len(firing[tt][0])>0 and len(firing[tt][0])<2 and firing[tt][1]==nn:
                spk_i.append(firing[tt][0])
            elif len(firing[tt][0])>1:  # if we happen to have synchronous spike breaking CTMC!
                spk_i.append(np.array([firing[tt][0][0]]))  # random for the first one!!??
        spk_i = np.array(spk_i).squeeze()
        isi[nn] = np.min(np.diff(spk_i))
    min_isi = np.min(isi)
    return min_isi

# %% computing statistics for infinite data given parameter
def P_frw_ctmc(param):
    """
    get joint probability given parameters (need to change for ctmc??)
    """
    k, pi = param2M(param)  # calling asymmetric network
    nc = len(pi)
    Pxy = np.zeros((nc,nc))
    for ii in range(nc):
        Pxy[ii,:] = k[ii,:]*pi[ii]  # compute joint from transition k and steady-state pi (this is wrong using Q!?)
    return Pxy

def edge_flux_inf(param, N=N, combinations=combinations):
    """
    compute edge flux with infinite data using pi_i k_ij
    """
    kij, pi = param2M(param, N, combinations)
    nc = len(pi)
    flux_ij = np.zeros((nc,nc))
    for ii in range(nc):
        for jj in range(nc):
            if ii is not jj:
                flux_ij[ii,jj] = pi[ii]*kij[ii,jj]
    return flux_ij

def get_stationary(M):
    """
    get stationary state distribution given a transition matrix M
    """
    uu,vv = np.linalg.eig(M.T)
    zeros_eig_id = np.argmin(np.abs(uu-1))
    pix = vv[:,zeros_eig_id] / np.sum(vv[:,zeros_eig_id])
    return np.real(pix)
