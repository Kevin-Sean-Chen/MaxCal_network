# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 01:16:58 2026

@author: kevin
"""

#############################################################
# max-cal inference for retina triplets,
# but here we have samples of the third neuron
#############################################################

# -*- coding: utf-8 -*-
import itertools
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib
import pickle

matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)

from maxcal_functions import spk2statetime, compute_tauC, word_id


# %% setup
N = 3
spins = [0, 1]
combinations = list(itertools.product(spins, repeat=N))


# %% functions
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

    Local neuron identities:
        local neuron 1 = nids[0]
        local neuron 2 = nids[1]
        local neuron 3 = nids[2]
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

            # convention: (0, dt] -> bin 0, (dt, 2dt] -> bin 1
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


def make_bin(indices, N=3):
    x = [0] * N
    for idx in indices:
        x[idx - 1] = 1
    return tuple(x)


def coarse_grain_tauC(ijk, tau, C, eps=1e-12):
    """
    Estimate effective i -> j coupling while marginalizing over k.

    ijk uses local 1-based labels.
    Example:
        if nids = [3, 34, sampled],
        then local 1 = ID 3,
             local 2 = ID 34,
             local 3 = sampled.

        coarse_grain_tauC((1,2,3), tau, C) gives 1 -> 2.
        coarse_grain_tauC((2,1,3), tau, C) gives 2 -> 1.
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


def full_and_prime_coupling(ijk, M, eps=1e-12):
    """
    Compute both:
        w      = effect j -> i when k = 0
        wprime = effect j -> i when k = 1

    ijk uses local 1-based labels.
    Example:
        full_and_prime_coupling((1,2,3), M)
        returns w12 and w12_prime, where prime means k=3 is ON.
    """

    j, i, k = ijk
    M = np.maximum(M, eps)

    # baseline i firing when j=0, k=0:
    # 000 -> i
    f0 = M[word_id((0, 0, 0)), word_id(make_bin([i]))]

    # i firing when j=1, k=0:
    # j -> i+j
    f_j = M[word_id(make_bin([j])), word_id(make_bin([i, j]))]

    # baseline i firing when j=0, k=1:
    # k -> i+k
    f_k = M[word_id(make_bin([k])), word_id(make_bin([i, k]))]

    # i firing when j=1, k=1:
    # j+k -> 111
    f_jk = M[word_id(make_bin([j, k])), word_id((1, 1, 1))]

    w = np.log(f_j / f0)
    wprime = np.log(f_jk / f_k)

    return w, wprime

def infer_full_and_prime_couplings(M):
    """
    Return w and wprime for all directed pairwise effects.

    w12       = 1 -> 2 when 3 is OFF
    w12prime  = 1 -> 2 when 3 is ON
    """

    labels = ["w12", "w13", "w21", "w23", "w32", "w31"]

    ijks = [
        (1, 2, 3),  # 1 -> 2, third = 3
        (1, 3, 2),  # 1 -> 3, third = 2
        (2, 1, 3),  # 2 -> 1, third = 3
        (2, 3, 1),  # 2 -> 3, third = 1
        (3, 2, 1),  # 3 -> 2, third = 1
        (3, 1, 2),  # 3 -> 1, third = 2
    ]

    w = []
    wp = []

    for ijk in ijks:
        wi, wpi = full_and_prime_coupling(ijk, M)
        w.append(wi)
        wp.append(wpi)

    return labels, np.asarray(w), np.asarray(wp)


def infer_cg12_cg21(tau, C):
    """
    For nids = [A, B, sampled],
    return:
        cg12 = A -> B
        cg21 = B -> A
    """

    cg12 = coarse_grain_tauC((1, 2, 3), tau, C)
    cg21 = coarse_grain_tauC((2, 1, 3), tau, C)

    return cg12, cg21


# %% main analysis ############################################################
###############################################################################
mat_path = "C:/Users/kevin/Downloads/Data_processed.mat"
dataset = 1

dt = 1.0
window_ms = 20 ### 20,  40,80
window = int(window_ms / dt)
top_K = 7  ### -1 for all

spk_data, spk_ids = load_dataset(mat_path, dataset)

# Fixed biological identity:
# neuron 1 = ID 3
# neuron 2 = ID 34
# neuron 3 = ID 13
base_triplet = np.array([3, 34, 13])
only_pair12 = True

sorted_cell_ids, sorted_counts = count_spikes_by_cell(spk_ids)
list_of_ID = sorted_cell_ids

print("Base triplet:")
print("neuron 1 = ID", base_triplet[0])
print("neuron 2 = ID", base_triplet[1])
print("neuron 3 = ID", base_triplet[2])


# %% pair definitions
# Each pair is [first neuron, second neuron].
# For each pair, we iterate the sampled third neuron.
# cg12 gives first -> second.
# cg21 gives second -> first.

pair_jobs = [
    {
        "pair": np.array([base_triplet[0], base_triplet[1]]),
        "forward_label": "w12",
        "reverse_label": "w21",
    },
]

if not only_pair12:
    pair_jobs.extend([
        {
            "pair": np.array([base_triplet[0], base_triplet[2]]),
            "forward_label": "w13",
            "reverse_label": "w31",
        },
        {
            "pair": np.array([base_triplet[1], base_triplet[2]]),
            "forward_label": "w23",
            "reverse_label": "w32",
        },
    ])

active_labels = []
for job in pair_jobs:
    active_labels.extend([job["forward_label"], job["reverse_label"]])


# %% sample over third neurons
all_samples = {label: [] for label in active_labels}

all_third_ids = {label: [] for label in active_labels}

all_w_primes = {label: [] for label in active_labels}

for job in pair_jobs:
    pair = job["pair"]
    forward_label = job["forward_label"]
    reverse_label = job["reverse_label"]
    
    ### top K selection, to speed up #####
    possible_third_neurons = [nid for nid in list_of_ID if nid not in pair][:top_K]

    print("\nSampling pair:", pair)
    print("Forward:", forward_label, "Reverse:", reverse_label)
    print("Number of sampled third neurons:", len(possible_third_neurons))

    for third_id in possible_third_neurons:
        # local neuron 1 = pair[0]
        # local neuron 2 = pair[1]
        # local neuron 3 = sampled third neuron
        nids = np.array([pair[0], pair[1], third_id])

        print(f"  third neuron = {third_id}")

        firing_s, lts = build_firing_trials(
            spk_data=spk_data,
            spk_ids=spk_ids,
            nids=nids,
            dt=dt,
        )

        tau_all, C_all, states_all, times_all = aggregate_tauC(
            firing_s=firing_s,
            window=window,
            lts=lts,
        )

        if tau_all is None or np.sum(tau_all) == 0:
            continue

        cg12, cg21 = infer_cg12_cg21(tau_all, C_all)
        
        # check w-prime
        tau_norm = (tau_all+1) / np.sum(tau_all)
        C_norm = (C_all+1) / np.sum(tau_all)
        M = (C_norm/tau_norm[:,None])
        labels, w, wp = infer_full_and_prime_couplings(M)

        # Store forward and reverse effects for this fixed pair
        all_samples[forward_label].append(cg12)
        all_samples[reverse_label].append(cg21)

        all_third_ids[forward_label].append(third_id)
        all_third_ids[reverse_label].append(third_id)
        
        all_w_primes[forward_label].append(wp[0])
        all_w_primes[reverse_label].append(wp[2])

# %% if we just load pkl
save = False
if save==True:
    with open("retina_triplet_sampling.pkl", "rb") as f:
        data = pickle.load(f)
    
    all_samples   = data["all_samples"]
    all_third_ids = data["all_third_ids"]
    all_w_primes  = data['all_w_primes']
    base_triplet  = data["base_triplet"]
    window_ms     = data["window_ms"]
    dataset       = data["dataset"]
    dt            = data["dt"]
    
    print("Loaded retina_triplet_sampling.pkl")

# %% convert to arrays
for key in all_samples:
    all_samples[key] = np.asarray(all_samples[key], dtype=float)
    all_w_primes[key] = np.asarray(all_w_primes[key], dtype=float)
    all_third_ids[key] = np.asarray(all_third_ids[key])


# %% MAIN PLOT: six wij bars with error bars over sampled third neurons
wij_order = [label for label in ["w12", "w13", "w21", "w23", "w32", "w31"] if label in all_samples]

means = np.array([np.nanmean(all_samples[k]) for k in wij_order])
sems = np.array([
    np.nanstd(all_samples[k]) / np.sqrt(len(all_samples[k]))
    for k in wij_order
])

x = np.arange(len(wij_order))

plt.figure(figsize=(8, 5))
plt.bar(x, means, yerr=sems, capsize=4, alpha=0.85)
plt.xticks(x, wij_order, fontsize=14)
plt.ylabel("coarse-grained weight", fontsize=16)
plt.axhline(0, color="k", linewidth=0.8)
plt.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.show()


# %% DEBUG PLOT: raw samples across third neurons

plt.figure(figsize=(12, 8))

for idx, key in enumerate(wij_order):
    plt.subplot(len(wij_order), 1, idx + 1)
    plt.plot(all_third_ids[key], all_samples[key], "o", alpha=0.8)
    plt.plot(all_third_ids[key], all_w_primes[key], "ro", alpha=0.8)
    plt.axhline(0, color="k", linewidth=0.5)
    plt.xlabel("sampled third neuron ID", fontsize=11)
    plt.ylabel("weight", fontsize=11)
    plt.title(key, fontsize=14)
    plt.grid(True, alpha=0.3)
### label o as wij and ro as wij-prime
plt.legend(["wij", "wij-prime"], loc="upper right", fontsize=10)
plt.tight_layout()
plt.show()

# %% print summary
print("\nFinal wij summary: mean ± SEM")
for key, mu, se in zip(wij_order, means, sems):
    print(f"{key}: {mu:.4f} ± {se:.4f}  | n={len(all_samples[key])}")
    
# %% Saving for later!
save = False

if save==True:
    save_data = {
        "all_samples": all_samples,
        "all_third_ids": all_third_ids,
        "all_w_primes": all_w_primes,
        "base_triplet": base_triplet,
        "window_ms": window_ms,
        "dataset": dataset,
        "dt": dt,
    }
    
    with open("retina_triplet_sampling.pkl", "wb") as f:
        pickle.dump(save_data, f)
    
    print("Saved to retina_triplet_sampling.pkl")