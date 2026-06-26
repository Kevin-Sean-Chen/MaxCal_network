# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 01:16:58 2026

@author: kevin
"""

#############################################################
# max-cal inference for retina triplets,
# but here we have samples of the third neuron
#############################################################

import itertools
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)


from maxcal_functions import spk2statetime, compute_tauC, word_id

# %% clean functions, modified from retina_maxcal.py, to scale up for triplet sampling
def load_dataset(mat_path, dataset):
    mat_data = scipy.io.loadmat(mat_path)
    spk_data = mat_data["spike_times"][0][dataset][0]
    spk_ids = mat_data["cell_IDs"][0][dataset][0]
    return spk_data, spk_ids


def count_spikes_by_cell(spk_ids):
    all_ids = np.unique(np.concatenate([x.squeeze() for x in spk_ids]))
    counts = []

    for nid in all_ids:
        count = 0
        for rr in range(len(spk_ids)):
            count += np.sum(spk_ids[rr].squeeze() == nid)
        counts.append(count)

    counts = np.asarray(counts)
    order = np.argsort(counts)[::-1]
    return all_ids[order], counts[order]


def build_firing_trials(spk_data, spk_ids, nids, dt=1.0):
    """
    Build firing_s for all repeats.
    Each repeat can have its own length.

    Returns
    -------
    firing_s : list
        firing_s[rr][tt] = [tt, spike_indices] or [empty, empty]
    lts : array
        length of each repeat in bins
    """
    N = len(nids)
    firing_s = []
    lts = []

    for rr in range(len(spk_data)):
        spkt = spk_data[rr].squeeze()
        spki = spk_ids[rr].squeeze()

        if len(spkt) == 0:
            continue

        lt_trial = int(np.ceil(np.max(spkt) / dt))
        lts.append(lt_trial)

        spikes_by_bin = {}

        for nn in range(N):
            spks = spkt[spki == nids[nn]]

            # same convention:
            # (0, dt] -> bin 0
            # (dt, 2dt] -> bin 1
            bins = np.ceil(spks / dt).astype(int) - 1

            valid = (bins >= 0) & (bins < lt_trial)
            bins = bins[valid]

            for b in bins:
                if b not in spikes_by_bin:
                    spikes_by_bin[b] = []
                spikes_by_bin[b].append(nn)

        firing = []
        for tt in range(lt_trial):
            if tt in spikes_by_bin:
                spike_indices = np.asarray(spikes_by_bin[tt], dtype=int)
                firing.append([tt, spike_indices])
            else:
                firing.append([np.array([]), np.array([])])

        firing_s.append(firing)

    return firing_s, np.asarray(lts)

def aggregate_tauC(firing_s, window, lts):
    """
    Aggregate tau and C across repeats with different lengths.
    """
    tau_all = None
    C_all = None
    states_all = []
    times_all = []

    for rr, firing in enumerate(firing_s):
        lt_trial = int(lts[rr])

        states, times = spk2statetime(firing, window, lt=lt_trial)
        tau, C = compute_tauC(states, times, lt=lt_trial)

        states_all.append(states)
        times_all.append(times)

        if tau_all is None:
            tau_all = tau.copy()
            C_all = C.copy()
        else:
            tau_all += tau
            C_all += C

    return tau_all, C_all, states_all, times_all


def infer_M(C, tau, eps=1e-12):
    """
    Empirical CTMC transition matrix estimate.
    """
    return C / np.maximum(tau[:, None], eps)


def infer_triplet_couplings(M, eps=1e-12):
    """
    For state ordering:
    000=0, 001=1, 010=2, 011=3,
    100=4, 101=5, 110=6, 111=7
    """
    M = np.maximum(M, eps)

    f1 = M[0, 4]
    f2 = M[0, 2]
    f3 = M[0, 1]

    w12 = np.log(M[4, 6] / f2)
    w13 = np.log(M[4, 5] / f3)
    w21 = np.log(M[2, 6] / f1)
    w23 = np.log(M[2, 3] / f3)
    w32 = np.log(M[1, 3] / f2)
    w31 = np.log(M[1, 5] / f1)

    labels = ["w12", "w13", "w21", "w23", "w32", "w31"]
    ws = np.array([w12, w13, w21, w23, w32, w31])

    return labels, ws


def make_bin(indices, N=3):
    x = [0] * N
    for idx in indices:
        x[idx - 1] = 1
    return tuple(x)


def coarse_grain_tauC(ijk, tau, C, eps=1e-12):
    """
    Estimate effective i -> j coupling while marginalizing over k.

    ijk uses 1-based neuron labels.
    """
    j, i, k = ijk

    gnd = (0, 0, 0)

    f = (
        C[word_id(gnd), word_id(make_bin([i]))]
        + C[word_id(make_bin([k])), word_id(make_bin([i, k]))]
    ) / np.maximum(
        tau[word_id(gnd)] + tau[word_id(make_bin([k]))],
        eps,
    )

    fexpw = (
        C[word_id(make_bin([j])), word_id(make_bin([i, j]))]
        + C[word_id(make_bin([j, k])), word_id((1, 1, 1))]
    ) / np.maximum(
        tau[word_id(make_bin([j]))] + tau[word_id(make_bin([j, k]))],
        eps,
    )

    return np.log(np.maximum(fexpw, eps) / np.maximum(f, eps))


def infer_coarse_grained_couplings(tau, C):
    labels = ["cg12", "cg13", "cg21", "cg23", "cg32", "cg31"]

    ws = np.array([
        coarse_grain_tauC((1, 2, 3), tau, C),
        coarse_grain_tauC((1, 3, 2), tau, C),
        coarse_grain_tauC((2, 1, 3), tau, C),
        coarse_grain_tauC((2, 3, 1), tau, C),
        coarse_grain_tauC((3, 2, 1), tau, C),
        coarse_grain_tauC((3, 1, 2), tau, C),
    ])

    return labels, ws


def plot_couplings(full_labels, full_ws, cg_labels, cg_ws):
    plt.figure(figsize=(7, 6))

    plt.subplot(2, 1, 1)
    plt.bar(full_labels, full_ws)
    plt.axhline(0, color="k", linewidth=1)
    plt.ylabel("inferred")

    plt.subplot(2, 1, 2)
    plt.bar(cg_labels, cg_ws)
    plt.axhline(0, color="k", linewidth=1)
    plt.ylabel("coarse-grained")

    plt.tight_layout()
    plt.show()


def plot_nonlinearity(M, eps=1e-12):
    M = np.maximum(M, eps)

    f1, f2, f3 = M[0, 4], M[0, 2], M[0, 1]

    w12 = np.log(M[4, 6] / f2)
    w13 = np.log(M[4, 5] / f3)
    w21 = np.log(M[2, 6] / f1)
    w23 = np.log(M[2, 3] / f3)
    w32 = np.log(M[1, 3] / f2)
    w31 = np.log(M[1, 5] / f1)

    plt.figure(figsize=(6, 5))

    ws = np.array([0, w21, w31, w21 + w31])
    phis = np.array([f1, M[2, 6], M[1, 5], M[3, 7]])
    plt.semilogy(ws, phis, "o", label="neuron 1")

    ws = np.array([0, w12, w32, w12 + w32])
    phis = np.array([f2, M[4, 6], M[1, 3], M[5, 7]])
    plt.semilogy(ws, phis, "o", label="neuron 2")

    ws = np.array([0, w13, w23, w13 + w23])
    phis = np.array([f3, M[4, 5], M[2, 3], M[6, 7]])
    plt.semilogy(ws, phis, "o", label="neuron 3")

    plt.xlabel("input x")
    plt.ylabel("transition rate phi")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    mat_path = "C:/Users/kevin/Downloads/Data_processed.mat"

    dataset = 1
    nids = np.array([3, 34, 13])

    dt = 1.0
    T = 10000
    window_ms = 20
    window = int(window_ms / dt)

    spk_data, spk_ids = load_dataset(mat_path, dataset)

    firing_s, lt = build_firing_trials(
        spk_data=spk_data,
        spk_ids=spk_ids,
        nids=nids,
        dt=dt,
        T=T,
    )

    tau_all, C_all, states_all, times_all = aggregate_tauC(
        firing_s=firing_s,
        window=window,
        lt=lt,
    )

    total_time = np.sum(tau_all)
    tau_norm = tau_all / total_time
    C_norm = C_all / total_time

    M = infer_M(C_norm, tau_norm)

    full_labels, full_ws = infer_triplet_couplings(M)
    cg_labels, cg_ws = infer_coarse_grained_couplings(tau_all, C_all)

    print("Triplet:", nids)
    print(dict(zip(full_labels, full_ws)))
    print(dict(zip(cg_labels, cg_ws)))

    plot_couplings(full_labels, full_ws, cg_labels, cg_ws)
    plot_nonlinearity(M)


if __name__ == "__main__":
    ### for origna result
    # main()
    
    # ##### sampling test #####
    # ##### sampling test #####
    mat_path = "C:/Users/kevin/Downloads/Data_processed.mat"
    dataset = 1
    dt = 1.0
    window_ms = 20
    window = int(window_ms / dt)
    
    spk_data, spk_ids = load_dataset(mat_path, dataset)
    
    # Correct sorting by firing rate
    sorted_cell_ids, sorted_counts = count_spikes_by_cell(spk_ids)
    list_of_ID = sorted_cell_ids
    print(f"Sorted IDs: {list_of_ID}")
    
    # %% iterations
    # fix two and loop the third
    # fix_id = np.array([25, 44])  # highest rate
    fix_id = np.array([34, 13])  # old choice
    # fix_id = np.array([2, 4])    # low rate
    
    possible_third_neurons = [i for i in list_of_ID if i not in fix_id]
    
    w_samples = []
    third_ids_used = []
    
    for ii in possible_third_neurons:
        nids = np.append(fix_id, ii)
        print(f"Testing third neuron: {ii}, out of {len(possible_third_neurons)}")
    
        firing_s, lts = build_firing_trials(
            spk_data=spk_data,
            spk_ids=spk_ids,
            nids=nids,
            dt=dt,
        )
    
        tau_all, C_all, states_all, times_all = aggregate_tauC(
            firing_s=firing_s,
            lts=lts,
            window=window,
        )
    
        if np.sum(tau_all) == 0:
            continue
    
        tau_norm = tau_all / np.sum(tau_all)
        C_norm = C_all / np.sum(tau_all)
    
        M = infer_M(C_norm, tau_norm)
    
        # Full conditional triplet weights
        full_labels, full_ws = infer_triplet_couplings(M)
    
        # Coarse-grained effective pairwise weights
        cg_labels, cg_ws = infer_coarse_grained_couplings(tau_all, C_all)
    
        # choose which one to store
        # w_samples.append(full_ws)
        w_samples.append(cg_ws)
    
        third_ids_used.append(ii)
    
    w_samples = np.asarray(w_samples)
    third_ids_used = np.asarray(third_ids_used)
    
    # %% plotting
    coupling_names = cg_labels  # or full_labels if using full_ws

    plt.figure(figsize=(14, 10))
    for idx, name in enumerate(coupling_names):
        plt.subplot(3, 2, idx + 1)
        plt.plot(third_ids_used, w_samples[:, idx], 'o')
        plt.xlabel('Third Neuron ID', fontsize=14)
        plt.ylabel('Coupling Strength', fontsize=14)
        plt.title(name, fontsize=14)
        plt.axhline(0, color='k', linewidth=0.5)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    means = np.nanmean(w_samples, axis=0)
    sems = np.nanstd(w_samples, axis=0) / np.sqrt(w_samples.shape[0])
    x = np.arange(len(coupling_names))
    
    plt.figure(figsize=(10, 5))
    plt.bar(x, means, yerr=sems, capsize=4, alpha=0.85)
    plt.xticks(x, coupling_names)
    plt.ylabel('coarse-grained weight')
    plt.axhline(0, color='k', linewidth=0.5)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()