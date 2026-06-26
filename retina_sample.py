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


def build_firing_trials(spk_data, spk_ids, nids, dt=1.0, T=10000):
    N = len(nids)
    lt = int(T / dt)
    firing_s = []

    for rr in range(len(spk_data)):
        spkt = spk_data[rr].squeeze()
        spki = spk_ids[rr].squeeze()

        spikes_by_bin = {}

        for nn in range(N):
            spks = spkt[spki == nids[nn]]

            bins = np.floor(spks / dt).astype(int)

            # match your original condition:
            # spks > tt*dt and spks <= tt*dt + dt
            # means spike exactly at t=dt goes to bin 0
            bins = np.ceil(spks / dt).astype(int) - 1

            valid = (bins >= 0) & (bins < lt)
            bins = bins[valid]

            for b in bins:
                if b not in spikes_by_bin:
                    spikes_by_bin[b] = []
                spikes_by_bin[b].append(nn)

        firing = []
        firing.append((np.array([]), np.array([])))

        for tt in range(lt):
            if tt in spikes_by_bin:
                spike_indices = np.asarray(spikes_by_bin[tt], dtype=int)
                firing.append([tt, spike_indices])
            else:
                firing.append([np.array([]), np.array([])])

        firing_s.append(firing)

    return firing_s, lt


def aggregate_tauC(firing_s, window, lt):
    tau_all = None
    C_all = None
    states_all = []
    times_all = []

    for firing in firing_s:
        states, times = spk2statetime(firing, window, lt=lt)
        tau, C = compute_tauC(states, times, lt=lt)

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
    # same data and param setup
    mat_path = "C:/Users/kevin/Downloads/Data_processed.mat"
    dataset = 2
    dt = 1.0
    T = 10000
    window_ms = 20  ### 80,20
    window = int(window_ms / dt)
    spk_data, spk_ids = load_dataset(mat_path, dataset)
    
    # %%
    # sort by firing rate, from output of count_spikes_by_cell
    firing_rates,_ = count_spikes_by_cell(spk_ids)
    sorted_ids = np.argsort(firing_rates)[::-1]
    print(f"Sorted IDs: {sorted_ids}")
        
    # %% hand chose examples (old triplet, low, and high firing picka)
    # fix two and loop the third
    nids = np.array([3, 34, 13])
    fix_id = np.array([34, 13]) ### old choice
    fix_id = np.array([25,44])  ### highest rate
    # fix_id = np.array([2, 4]) ### lowest rate
    ### list of possible third neurons to test, from the dataset (1 to the largest ID, excluding the fixed ones)
    list_of_ID = np.unique(np.hstack(spk_ids)[0]) ### unsorted
    list_of_ID = list_of_ID[sorted_ids] ### sorted
    possible_third_neurons = [i for i in list_of_ID if i not in fix_id]

    # loop and record variables
    w12s = []
    for ii in possible_third_neurons:
        ### append the third neuron to the fixed ones
        nids = np.append(fix_id, ii)
        print(f"Testing third neuron: {ii}, out of {len(possible_third_neurons)}")
        ### making firing
        firing_s, lt = build_firing_trials(
            spk_data=spk_data,
            spk_ids=spk_ids,
            nids=nids,
            dt=dt,
            T=T,
        )
        ### making tau and C
        tau_all, C_all, states_all, times_all = aggregate_tauC(
            firing_s=firing_s,
            window=window,
            lt=lt,
        )
        ### norm them
        total_time = np.sum(tau_all)
        tau_norm = tau_all / total_time
        C_norm = C_all / total_time
        ### get M and do inference
        M = infer_M(C_norm, tau_norm)
        full_labels, full_ws = infer_triplet_couplings(M)
        # cg_labels, cg_ws = infer_coarse_grained_couplings(tau_all, C_all)
        w12s.append(full_ws)
    # %% plotting - show distribution of 6 coupling strengths
    w12s_array = np.array(w12s)
    coupling_names = ['w12', 'w13', 'w21', 'w23', 'w32', 'w31']
    
    plt.figure(figsize=(14, 10))
    for idx, name in enumerate(coupling_names):
        plt.subplot(3, 2, idx + 1)
        plt.plot(possible_third_neurons, w12s_array[:, idx], 'o')
        plt.xlabel('Third Neuron ID', fontsize=14)
        plt.ylabel('Coupling Strength', fontsize=14)
        plt.title(name, fontsize=14)
        plt.axhline(0, color='k', linewidth=0.5)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
# %% plot g_ij with error bar
###############################################################################
###############################################################################
# %% iterate pairs in triplet
nids = np.array([3, 34, 13])
pair = np.array([nids[0], nids[1]])

list_of_ID = np.unique(np.hstack(spk_ids)[0])
list_of_ID = list_of_ID[sorted_ids]

possible_third_neurons = [i for i in list_of_ID if i not in pair]

w_samples = []
for ii in possible_third_neurons:
    nids = np.append(pair, ii)
    print(f"Testing pair {pair} with third neuron: {ii}, out of {len(possible_third_neurons)}")

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
    w_samples.append(full_ws)

pair_weights = np.asarray(w_samples)
pair_third_ids = np.asarray(possible_third_neurons)


# %%
### plotting - one bar plot with six weights
coupling_names = ['w12', 'w13', 'w21', 'w23', 'w32', 'w31']
all_w_samples = pair_weights

means = np.mean(all_w_samples, axis=0)
stds = np.std(all_w_samples, axis=0)/np.sqrt(len(possible_third_neurons))
x = np.arange(len(coupling_names))

plt.figure(figsize=(10, 5))
plt.bar(x, means, yerr=stds, capsize=4, alpha=0.85)
plt.xticks(x, coupling_names)
plt.ylabel('inferred weight')
plt.axhline(0, color='k', linewidth=0.5)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()