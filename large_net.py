#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 23:43:54 2026

@author: kschen
"""

"""
Random recurrent spiking network in Brian2.

What it does:
- Creates a population of excitatory and inhibitory LIF neurons
- Connects them randomly with sparse synapses
- Drives them with constant current plus weak noise
- Simulates spike trains
- Plots a spike raster and population firing rate

Requirements:
    pip install brian2 matplotlib numpy
"""

from brian2 import *
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# Simulation settings
# ----------------------------
seed(123)  # for reproducibility

defaultclock.dt = 0.1 * ms
duration = 20.0*1 * second

# ----------------------------
# Network size
# ----------------------------
N = 100
frac_exc = 0.8
N_exc = int(N * frac_exc)
N_inh = N - N_exc

# ----------------------------
# Neuron parameters
# ----------------------------
tau_m = 10 * ms
tau_ref = 1 * ms
v_rest = -65 * mV
v_reset = -65 * mV
v_thresh = -50 * mV

# Synaptic time constants
tau_e = 5 * ms
tau_i = 10 * ms
 
# Connection strengths
w_e = 0.6 * mV   # excitatory jump
w_i = -2.0 * mV  # inhibitory jump

# Random connectivity probability
p_connect = 0.2

# External drive and noise
mu_exc = 15.0 * mV
mu_inh = 15.0 * mV
sigma_noise = 1.5 * mV

# ----------------------------
# Model equations
# ----------------------------
# LIF with exponentially decaying synaptic currents and white noise
eqs = """
dv/dt = (-(v - v_rest) + I_ext + I_syn) / tau_m + sigma_noise * xi * sqrt(2 / tau_m) : volt (unless refractory)
I_syn = I_e + I_i : volt
dI_e/dt = -I_e / tau_e : volt
dI_i/dt = -I_i / tau_i : volt
I_ext : volt
"""

# ----------------------------
# Create neurons
# ----------------------------
neurons = NeuronGroup(
    N,
    model=eqs,
    threshold="v > v_thresh",
    reset="v = v_reset",
    refractory=tau_ref,
    method="euler",
    name="neurons",
)

# Random initial conditions
neurons.v = "v_rest + rand() * (v_thresh - v_rest)"
neurons.I_e = 0 * mV
neurons.I_i = 0 * mV

# External current differs slightly across neurons
neurons.I_ext[:N_exc] = mu_exc + 1.0 * mV * np.random.randn(N_exc)
neurons.I_ext[N_exc:] = mu_inh + 1.0 * mV * np.random.randn(N_inh)

# Define excitatory and inhibitory subgroups
exc = neurons[:N_exc]
inh = neurons[N_exc:]

# ----------------------------
# Synapses
# ----------------------------
# Excitatory synapses increase I_e
S_exc = Synapses(exc, neurons, on_pre="I_e += w_e", name="S_exc")
S_exc.connect(p=p_connect)

# Inhibitory synapses decrease membrane potential through I_i
# Since I_i is part of I_syn, and we want inhibition, w_i is negative.
S_inh = Synapses(inh, neurons, on_pre="I_i += w_i", name="S_inh")
S_inh.connect(p=p_connect)

# ----------------------------
# Monitors
# ----------------------------
spike_mon = SpikeMonitor(neurons, name="spike_mon")
rate_mon = PopulationRateMonitor(neurons, name="rate_mon")
state_mon = StateMonitor(neurons, variables=["v"], record=range(5), name="state_mon")

# ----------------------------
# Run simulation
# ----------------------------
print("Running simulation...")
run(duration)
print("Done.")

# ----------------------------
# Basic summary stats
# ----------------------------
n_spikes = spike_mon.num_spikes
mean_rate = n_spikes / N / duration
print(f"Total spikes: {n_spikes}")
print(f"Mean firing rate: {mean_rate / Hz:.2f} Hz")

# ----------------------------
# Plot results
# ----------------------------
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)

# Spike raster
axes[0].plot(spike_mon.t / ms, spike_mon.i, ".k", markersize=2)
axes[0].set_ylabel("Neuron index")
axes[0].set_title("Spike raster")

# Population firing rate
# Smooth for readability
window = int(5 * ms / defaultclock.dt)
window = max(window, 1)
smoothed_rate = np.convolve(
    rate_mon.rate / Hz,
    np.ones(window) / window,
    mode="same"
)
axes[1].plot(rate_mon.t / ms, smoothed_rate)
axes[1].set_ylabel("Rate (Hz)")
axes[1].set_title("Population firing rate")

# Example membrane potentials
for k in range(5):
    axes[2].plot(state_mon.t / ms, state_mon.v[k] / mV, label=f"Neuron {k}")
axes[2].axhline(v_thresh / mV, linestyle="--", linewidth=1)
axes[2].set_xlabel("Time (ms)")
axes[2].set_ylabel("V (mV)")
axes[2].set_title("Example membrane potentials")

plt.tight_layout()
plt.show()


# %% simple linear analysis to show some relation
"""
Brian2 random spiking network + simple analysis:
1) simulate a random E/I recurrent network
2) bin spikes into a neuron x time binary matrix
3) compute zero-lag correlation matrix
4) extract connectivity matrix
5) compare correlation with connectivity

Requirements:
    pip install brian2 matplotlib numpy
"""

from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. SIMULATE A RANDOM SPIKING NETWORK
# ============================================================

# seed(42)
# np.random.seed(42)

# defaultclock.dt = 0.1 * ms
# duration = 10.0 * second

# Network size
# N = 100
# frac_exc = 0.8
# N_exc = int(N * frac_exc)
# N_inh = N - N_exc

# # Neuron parameters
# tau_m = 10 * ms
# tau_ref = 1 * ms
# v_rest = -65 * mV
# v_reset = -65 * mV
# v_thresh = -50 * mV

# tau_e = 5 * ms
# tau_i = 10 * ms

# # Synaptic weights
# w_e = .6 * mV
# w_i = -2.0 * mV

# # Connection probability
# p_connect = 0.2

# # External drive
# mu_exc = 18. * mV
# mu_inh = 18. * mV
# sigma_noise = .5 * mV

# LIF with exponentially decaying synaptic currents
eqs = """
dv/dt = (-(v - v_rest) + I_ext + I_syn)/tau_m + sigma_noise*xi*sqrt(2/tau_m) : volt (unless refractory)
I_syn = I_e + I_i : volt
dI_e/dt = -I_e/tau_e : volt
dI_i/dt = -I_i/tau_i : volt
I_ext : volt
"""

neurons = NeuronGroup(
    N,
    model=eqs,
    threshold="v > v_thresh",
    reset="v = v_reset",
    refractory=tau_ref,
    method="euler",
    name="neurons",
)

neurons.v = "v_rest + rand() * (v_thresh - v_rest)"
neurons.I_e = 0 * mV
neurons.I_i = 0 * mV

# Slight heterogeneity in input current
neurons.I_ext[:N_exc] = mu_exc + (1. * mV) * np.random.randn(N_exc)
neurons.I_ext[N_exc:] = mu_inh + (1. * mV) * np.random.randn(N_inh)

exc = neurons[:N_exc]
inh = neurons[N_exc:]

# Synapses
S_exc = Synapses(exc, neurons, on_pre="I_e += w_e", name="S_exc")
S_exc.connect(p=p_connect)

S_inh = Synapses(inh, neurons, on_pre="I_i += w_i", name="S_inh")
S_inh.connect(p=p_connect)

# Monitors
spike_mon = SpikeMonitor(neurons, name="spike_mon")
rate_mon = PopulationRateMonitor(neurons, name="rate_mon")
state_mon = StateMonitor(neurons, "v", record=np.arange(min(5, N)), name="state_mon")

print("Running simulation...")
run(duration)
print("Done.")

# ============================================================
# 2. BUILD BINARY SPIKE MATRIX: X[neuron, time_bin]
# ============================================================

dt_bin = 1 * ms
T_bins = int(np.ceil((duration / dt_bin)))
X = np.zeros((N, T_bins), dtype=np.int8)

spike_i = np.asarray(spike_mon.i)
spike_t = np.asarray(spike_mon.t / dt_bin).astype(int)

valid = spike_t < T_bins
spike_i = spike_i[valid]
spike_t = spike_t[valid]

# Count spikes in bins, then binarize
np.add.at(X, (spike_i, spike_t), 1)
X = (X > 0).astype(np.int8)

print("Binary spike matrix shape:", X.shape)

# ============================================================
# 3. COMPUTE ZERO-LAG CORRELATION MATRIX
# ============================================================

# Convert to float and z-score each neuron's binned activity
Xf = X.astype(float)
X_centered = Xf - Xf.mean(axis=1, keepdims=True)
X_std = X_centered.std(axis=1, keepdims=True)

# Avoid divide-by-zero for silent neurons
silent = X_std.squeeze() < 1e-12
X_std[silent, :] = 1.0

X_norm = X_centered / X_std

# Correlation matrix
C = (X_norm @ X_norm.T) / X.shape[1]

# By convention, set diagonal to nan for plotting/comparison
np.fill_diagonal(C, np.nan)

# ============================================================
# 4. EXTRACT CONNECTIVITY MATRIX
#    W[i, j] = effect of neuron i on neuron j
#    +1 for excitatory, -1 for inhibitory, 0 for no direct connection
# ============================================================

W = np.zeros((N, N), dtype=np.int8)

# Excitatory: source indices already correspond to rows 0 ... N_exc-1
exc_i = np.asarray(S_exc.i[:], dtype=int)
exc_j = np.asarray(S_exc.j[:], dtype=int)
W[exc_i, exc_j] = 1

# Inhibitory: shift source indices by N_exc
inh_i = np.asarray(S_inh.i[:], dtype=int) + N_exc
inh_j = np.asarray(S_inh.j[:], dtype=int)
W[inh_i, inh_j] = -1

np.fill_diagonal(W, 0)

# ============================================================
# 5. COMPARE CORRELATION VS CONNECTIVITY
# ============================================================

mask_offdiag = ~np.eye(N, dtype=bool)

C_flat = C[mask_offdiag]
W_flat = W[mask_offdiag]

# Remove nan entries just in case
valid = np.isfinite(C_flat)
C_flat = C_flat[valid]
W_flat = W_flat[valid]

connected = W_flat != 0
unconnected = W_flat == 0
exc_connected = W_flat == 1
inh_connected = W_flat == -1

print("\n===== Summary =====")
print(f"Total spikes: {spike_mon.num_spikes}")
print(f"Mean population firing rate: {spike_mon.num_spikes / N / (duration/second):.2f} Hz")
print(f"Number of connected pairs: {connected.sum()}")
print(f"Number of unconnected pairs: {unconnected.sum()}")

if connected.sum() > 0:
    print(f"Mean corr (connected):   {np.nanmean(C_flat[connected]):.4f}")
if unconnected.sum() > 0:
    print(f"Mean corr (unconnected): {np.nanmean(C_flat[unconnected]):.4f}")
if exc_connected.sum() > 0:
    print(f"Mean corr (exc conn):    {np.nanmean(C_flat[exc_connected]):.4f}")
if inh_connected.sum() > 0:
    print(f"Mean corr (inh conn):    {np.nanmean(C_flat[inh_connected]):.4f}")

# ============================================================
# 6. OPTIONAL: PAIRWISE LAGGED CROSS-CORRELATION FUNCTION
# ============================================================

def normalized_crosscorr(x, y, max_lag_bins=20):
    """
    Compute normalized cross-correlation between two 1D binned spike trains.
    Returns lags and correlation values.

    Positive lag means y is shifted later relative to x.
    """
    x = x.astype(float)
    y = y.astype(float)

    x = x - x.mean()
    y = y - y.mean()

    sx = x.std()
    sy = y.std()
    if sx < 1e-12 or sy < 1e-12:
        lags = np.arange(-max_lag_bins, max_lag_bins + 1)
        return lags, np.zeros_like(lags, dtype=float)

    x = x / sx
    y = y / sy

    lags = np.arange(-max_lag_bins, max_lag_bins + 1)
    cc = np.zeros(len(lags), dtype=float)

    for k, lag in enumerate(lags):
        if lag < 0:
            cc[k] = np.mean(x[:lag] * y[-lag:])
        elif lag > 0:
            cc[k] = np.mean(x[lag:] * y[:-lag])
        else:
            cc[k] = np.mean(x * y)

    return lags, cc

# Example pair
example_i = 0
example_j = min(1, N - 1)
lags, cc_vals = normalized_crosscorr(X[example_i], X[example_j], max_lag_bins=20)

# ============================================================
# 7. PLOTS
# ============================================================

fig = plt.figure(figsize=(14, 10))

# (a) Spike raster
ax1 = plt.subplot(2, 3, 1)
ax1.plot(spike_mon.t / ms, spike_mon.i, ".k", markersize=2)
ax1.set_xlabel("Time (ms)")
ax1.set_ylabel("Neuron index")
ax1.set_title("Spike raster")

# (b) Population rate
ax2 = plt.subplot(2, 3, 2)
rate = np.asarray(rate_mon.rate / Hz)
time_rate = np.asarray(rate_mon.t / ms)
window = max(1, int((5 * ms) / defaultclock.dt))
kernel = np.ones(window) / window
rate_smooth = np.convolve(rate, kernel, mode="same")
ax2.plot(time_rate, rate_smooth)
ax2.set_xlabel("Time (ms)")
ax2.set_ylabel("Rate (Hz)")
ax2.set_title("Population firing rate")

# (c) Example membrane potentials
ax3 = plt.subplot(2, 3, 3)
n_show = min(5, N)
for k in range(n_show):
    ax3.plot(state_mon.t / ms, state_mon.v[k] / mV, label=f"{k}")
ax3.axhline(v_thresh / mV, linestyle="--", linewidth=1)
ax3.set_xlabel("Time (ms)")
ax3.set_ylabel("V (mV)")
ax3.set_title("Example membrane potentials")

# (d) Correlation matrix
ax4 = plt.subplot(2, 3, 4)
im1 = ax4.imshow(C, aspect="auto", interpolation="nearest")
ax4.set_title("Zero-lag correlation matrix")
ax4.set_xlabel("Neuron j")
ax4.set_ylabel("Neuron i")
plt.colorbar(im1, ax=ax4, fraction=0.046, pad=0.04)

# (e) Connectivity matrix
ax5 = plt.subplot(2, 3, 5)
im2 = ax5.imshow(W, aspect="auto", interpolation="nearest")
ax5.set_title("Connectivity matrix\n(+1 exc, -1 inh, 0 none)")
ax5.set_xlabel("Postsynaptic neuron j")
ax5.set_ylabel("Presynaptic neuron i")
plt.colorbar(im2, ax=ax5, fraction=0.046, pad=0.04)

# (f) Compare connected vs unconnected correlations
ax6 = plt.subplot(2, 3, 6)
bins = np.linspace(np.nanmin(C_flat), np.nanmax(C_flat), 50)
if unconnected.sum() > 0:
    ax6.hist(C_flat[unconnected], bins=bins, alpha=0.6, label="unconnected", density=True)
if exc_connected.sum() > 0:
    ax6.hist(C_flat[exc_connected], bins=bins, alpha=0.6, label="exc connected", density=True)
if inh_connected.sum() > 0:
    ax6.hist(C_flat[inh_connected], bins=bins, alpha=0.6, label="inh connected", density=True)
ax6.set_xlabel("Zero-lag correlation")
ax6.set_ylabel("Density")
ax6.set_title("Correlation vs connectivity")
ax6.legend()

plt.tight_layout()
plt.show()

# ============================================================
# 8. EXAMPLE LAGGED CROSS-CORRELATION PLOT
# ============================================================

plt.figure(figsize=(6, 4))
plt.plot(lags * float(dt_bin / ms), cc_vals)
plt.axvline(0, linestyle="--", linewidth=1)
plt.xlabel("Lag (ms)")
plt.ylabel("Normalized cross-correlation")
plt.title(f"Cross-correlation: neuron {example_i} vs {example_j}")
plt.tight_layout()
plt.show()


# %% MaxCal subsample
##########################################################################
# %% sub-sample 3 neural motifs, do MaxCal inference, and compare to ground truth, iteratively
##########################################################################
# %% setup
from maxcal_functions import spk2statetime, compute_tauC, spk2statetime_4N, compute_tauC_4N, cos_ang, corr_param
import os
import pickle


def spike_monitor_to_firing(spike_mon, timesteps):
    """
    Convert Brian2 SpikeMonitor output into scan_net-style `firing` format:
    firing[t] = [array_of_times_at_t, array_of_spiking_neuron_ids_at_t]
    """
    spikes_per_t = [[] for _ in range(timesteps)]
    spike_i = np.asarray(spike_mon.i, dtype=int)
    dt_sim_sec = 1e-4  # 0.1 ms
    spike_t = np.asarray(np.round(np.asarray(spike_mon.t) / dt_sim_sec), dtype=int)

    valid = (spike_t >= 0) & (spike_t < timesteps)
    for ii, tt in zip(spike_i[valid], spike_t[valid]):
        spikes_per_t[tt].append(ii)

    firing = [(np.array([]), np.array([]))]
    for tt in range(1, timesteps):
        ids = np.asarray(spikes_per_t[tt], dtype=int)
        if ids.size > 0:
            times = np.full(ids.shape, tt, dtype=int)
        else:
            times = np.array([], dtype=int)
        firing.append([times, ids])
    return firing


def project_triplet_firing(firing_full, triplet_ids):
    """
    Extract spikes from 3 chosen neurons and remap IDs to {0,1,2}
    so it is directly compatible with `spk2statetime(..., N=3)`.
    """
    triplet_ids = np.asarray(triplet_ids, dtype=int)
    mapper = {int(triplet_ids[k]): k for k in range(3)}
    in_triplet = set(triplet_ids.tolist())

    firing_triplet = []
    for rec in firing_full:
        t_arr = np.asarray(rec[0], dtype=int)
        i_arr = np.asarray(rec[1], dtype=int)

        if i_arr.size == 0:
            firing_triplet.append([np.array([], dtype=int), np.array([], dtype=int)])
            continue

        keep = np.array([ii in in_triplet for ii in i_arr], dtype=bool)
        if np.any(keep):
            i_sub = i_arr[keep]
            t_sub = t_arr[keep]
            i_remap = np.array([mapper[int(ii)] for ii in i_sub], dtype=int)
            firing_triplet.append([t_sub, i_remap])
        else:
            firing_triplet.append([np.array([], dtype=int), np.array([], dtype=int)])

    return firing_triplet


def project_quartet_firing(firing_full, quartet_ids):
    """
    Extract spikes from 4 chosen neurons and remap IDs to {0,1,2,3}
    so it is directly compatible with `spk2statetime_4N`.
    """
    quartet_ids = np.asarray(quartet_ids, dtype=int)
    mapper = {int(quartet_ids[k]): k for k in range(4)}
    in_quartet = set(quartet_ids.tolist())

    firing_quartet = []
    for rec in firing_full:
        t_arr = np.asarray(rec[0], dtype=int)
        i_arr = np.asarray(rec[1], dtype=int)

        if i_arr.size == 0:
            firing_quartet.append([np.array([], dtype=int), np.array([], dtype=int)])
            continue

        keep = np.array([ii in in_quartet for ii in i_arr], dtype=bool)
        if np.any(keep):
            i_sub = i_arr[keep]
            t_sub = t_arr[keep]
            i_remap = np.array([mapper[int(ii)] for ii in i_sub], dtype=int)
            firing_quartet.append([t_sub, i_remap])
        else:
            firing_quartet.append([np.array([], dtype=int), np.array([], dtype=int)])

    return firing_quartet


def infer_triplet_weights_from_firing(firing_triplet, S_sub, lt_steps, adapt_window=150):
    """
    Run the same inference-style summary used in scan_net.
    Returns inferred couplings and quality metrics.
    """
    spk_states, spk_times = spk2statetime(firing_triplet, adapt_window, lt=lt_steps, N=3)
    if len(spk_states) < 5:
        return None

    tau, C = compute_tauC(spk_states, spk_times)
    tau_ = (tau + 1.0) / lt_steps
    C_ = (C + 1.0) / lt_steps
    M_inf = C_ / tau_[:, None]

    eps = 1e-12
    f1 = max(M_inf[0, 4], eps)
    f2 = max(M_inf[0, 2], eps)
    f3 = max(M_inf[0, 1], eps)

    w12 = np.log(max(M_inf[4, 6], eps) / f2)
    w13 = np.log(max(M_inf[4, 5], eps) / f3)
    w21 = np.log(max(M_inf[2, 6], eps) / f1)
    w23 = np.log(max(M_inf[2, 3], eps) / f3)
    w32 = np.log(max(M_inf[1, 3], eps) / f2)
    w31 = np.log(max(M_inf[1, 5], eps) / f1)

    # Reverse-direction inferred couplings (u-like terms from MaxCal_motif)
    r1 = max(M_inf[4, 0], eps)
    r2 = max(M_inf[2, 0], eps)
    r3 = max(M_inf[1, 0], eps)

    u12 = -np.log(max(M_inf[6, 4], eps) / r2)
    u13 = -np.log(max(M_inf[5, 4], eps) / r3)
    u21 = -np.log(max(M_inf[6, 2], eps) / r1)
    u23 = -np.log(max(M_inf[3, 2], eps) / r3)
    u32 = -np.log(max(M_inf[3, 1], eps) / r2)
    u31 = -np.log(max(M_inf[5, 1], eps) / r1)

    inf_w = np.array([w12, w13, w21, w23, w32, w31], dtype=float)
    inf_u = np.array([u12, u13, u21, u23, u32, u31], dtype=float)
    true_s = np.array([
        S_sub[1, 0], S_sub[2, 0],
        S_sub[0, 1], S_sub[2, 1],
        S_sub[1, 2], S_sub[0, 2],
    ], dtype=float)

    if np.std(inf_w) < 1e-12 or np.std(true_s) < 1e-12:
        r_val = np.nan
    else:
        r_val = np.corrcoef(inf_w, true_s)[0, 1]

    if np.std(inf_u) < 1e-12 or np.std(true_s) < 1e-12:
        r_u_val = np.nan
    else:
        r_u_val = np.corrcoef(inf_u, true_s)[0, 1]

    return {
        "inf_w": inf_w,
        "inf_u": inf_u,
        "true_w": true_s,
        "R": r_val,
        "R_u": r_u_val,
        "sign": corr_param(true_s, inf_w, mode='binary'),
        "sign_u": corr_param(true_s, inf_u, mode='binary'),
        "cos": cos_ang(inf_w, true_s),
        "cos_u": cos_ang(inf_u, true_s),
    }


def infer_quartet_weights_from_firing(firing_quartet, S_sub, lt_steps, adapt_window=150):
    """
    4-neuron inference from firing: estimate pairwise couplings using
    the same log-ratio construction as the 3-neuron case.
    """
    spk_states, spk_times = spk2statetime_4N(firing_quartet, adapt_window, lt=lt_steps)
    if len(spk_states) < 5:
        return None

    tau, C = compute_tauC_4N(spk_states, spk_times)
    tau_ = (tau + 1.0) / lt_steps
    C_ = (C + 1.0) / lt_steps
    M_inf = C_ / tau_[:, None]

    eps = 1e-12
    f1 = max(M_inf[0, 8], eps)
    f2 = max(M_inf[0, 4], eps)
    f3 = max(M_inf[0, 2], eps)
    f4 = max(M_inf[0, 1], eps)

    w12 = np.log(max(M_inf[8, 12], eps) / f2)
    w13 = np.log(max(M_inf[8, 10], eps) / f3)
    w14 = np.log(max(M_inf[8,  9], eps) / f4)

    w21 = np.log(max(M_inf[4, 12], eps) / f1)
    w23 = np.log(max(M_inf[4,  6], eps) / f3)
    w24 = np.log(max(M_inf[4,  5], eps) / f4)

    w31 = np.log(max(M_inf[2, 10], eps) / f1)
    w32 = np.log(max(M_inf[2,  6], eps) / f2)
    w34 = np.log(max(M_inf[2,  3], eps) / f4)

    w41 = np.log(max(M_inf[1,  9], eps) / f1)
    w42 = np.log(max(M_inf[1,  5], eps) / f2)
    w43 = np.log(max(M_inf[1,  3], eps) / f3)

    inf_w = np.array(
        [w12, w13, w14, w21, w23, w24, w31, w32, w34, w41, w42, w43],
        dtype=float,
    )
    true_s = np.array(
        [
            S_sub[1, 0], S_sub[2, 0], S_sub[3, 0],
            S_sub[0, 1], S_sub[2, 1], S_sub[3, 1],
            S_sub[0, 2], S_sub[1, 2], S_sub[3, 2],
            S_sub[0, 3], S_sub[1, 3], S_sub[2, 3],
        ],
        dtype=float,
    )

    if np.std(inf_w) < 1e-12 or np.std(true_s) < 1e-12:
        r_val = np.nan
    else:
        r_val = np.corrcoef(inf_w, true_s)[0, 1]

    return {
        "inf_w": inf_w,
        "true_w": true_s,
        "R": r_val,
        "sign": corr_param(true_s, inf_w, mode='binary'),
        "cos": cos_ang(inf_w, true_s),
    }


# %% iterate random 3-neuron motifs from 100-neuron spike data
CACHE_MAXCAL = True
CACHE_PATH = "maxcal_inference_cache.pkl"

# shared inference parameters
adapt_window = 200
n_triplet_samples = 100
n_quartet_samples = 100
max_attempts_triplet = n_triplet_samples * 5
max_attempts_quartet = n_quartet_samples * 5
rng = np.random.default_rng(123)

triplet_records = []
quartet_records = []

if CACHE_MAXCAL and os.path.isfile(CACHE_PATH):
    with open(CACHE_PATH, "rb") as f:
        cached = pickle.load(f)
    triplet_records = cached.get("triplet_records", [])
    quartet_records = cached.get("quartet_records", [])
else:
    timesteps = int(np.ceil(float(duration) / 1e-4))  # duration / 0.1 ms
    firing_full = spike_monitor_to_firing(spike_mon, timesteps=timesteps)

if not triplet_records:
    attempts = 0
    while len(triplet_records) < n_triplet_samples and attempts < max_attempts_triplet:
        attempts += 1
        triplet = np.sort(rng.choice(N, size=3, replace=False))

        # Ground-truth local 3x3 connectivity (rows: post, cols: pre)
        S_sub = W[np.ix_(triplet, triplet)].astype(float)
        np.fill_diagonal(S_sub, 0.0)
        if np.count_nonzero(S_sub) == 0:
            continue

        firing_3 = project_triplet_firing(firing_full, triplet)

        out = infer_triplet_weights_from_firing(
            firing_triplet=firing_3,
            S_sub=S_sub,
            lt_steps=timesteps,
            adapt_window=adapt_window,
        )
        if out is None:
            continue

        triplet_records.append({
            "triplet": triplet,
            "R": out["R"],
            "R_u": out["R_u"],
            "sign": out["sign"],
            "sign_u": out["sign_u"],
            "cos": out["cos"],
            "cos_u": out["cos_u"],
            "inf_w": out["inf_w"],
            "inf_u": out["inf_u"],
            "true_w": out["true_w"],
        })

        if len(triplet_records) % 50 == 0:
            print(f"Processed {len(triplet_records)}/{n_triplet_samples} connected triplets")

    print(f"\nCompleted triplet inference: {len(triplet_records)} valid triplets")
    if attempts >= max_attempts_triplet and len(triplet_records) < n_triplet_samples:
        print(f"Stopped after {attempts} attempts; collected {len(triplet_records)} connected triplets.")
triplet_R = None
triplet_Ru = None
if len(triplet_records) > 0:
    R_vals = np.array([rr["R"] for rr in triplet_records], dtype=float)
    R_u_vals = np.array([rr["R_u"] for rr in triplet_records], dtype=float)
    sign_vals = np.array([rr["sign"] for rr in triplet_records], dtype=float)
    sign_u_vals = np.array([rr["sign_u"] for rr in triplet_records], dtype=float)
    cos_vals = np.array([rr["cos"] for rr in triplet_records], dtype=float)
    cos_u_vals = np.array([rr["cos_u"] for rr in triplet_records], dtype=float)

    triplet_R = R_vals
    triplet_Ru = R_u_vals

    print(f"Mean Pearson R: {np.nanmean(R_vals):.4f}")
    print(f"Mean Pearson R (u): {np.nanmean(R_u_vals):.4f}")
    print(f"Mean sign-corr: {np.nanmean(sign_vals):.4f}")
    print(f"Mean sign-corr (u): {np.nanmean(sign_u_vals):.4f}")
    print(f"Mean cosine: {np.nanmean(cos_vals):.4f}")
    print(f"Mean cosine (u): {np.nanmean(cos_u_vals):.4f}")

    plt.figure(figsize=(12, 3.5))
    plt.subplot(131)
    plt.hist(R_vals[np.isfinite(R_vals)], bins=30, alpha=0.8, density=True)
    plt.title("Triplet Pearson R")
    plt.xlabel("R")

    plt.subplot(132)
    plt.hist(sign_vals[np.isfinite(sign_vals)], bins=30, alpha=0.8, density=True)
    plt.title("Triplet sign-corr")
    plt.xlabel("sign correlation")

    plt.subplot(133)
    plt.hist(cos_vals[np.isfinite(cos_vals)], bins=30, alpha=0.8, density=True)
    plt.title("Triplet cosine")
    plt.xlabel("cosine") 

    plt.tight_layout()
    plt.show()

    # Compare cosine-angle distributions of w and u in one figure
    cos_w_finite = cos_vals[np.isfinite(cos_vals)]
    cos_u_finite = cos_u_vals[np.isfinite(cos_u_vals)]
    if cos_w_finite.size > 0 or cos_u_finite.size > 0:
        plt.figure(figsize=(6.2, 4.2))
        if cos_w_finite.size > 0:
            plt.hist(cos_w_finite, bins=30, alpha=0.45, density=True, label='w cosine', color='tab:blue')
        if cos_u_finite.size > 0:
            plt.hist(cos_u_finite, bins=30, alpha=0.45, density=True, label='u cosine', color='tab:orange')
        plt.xlabel("cosine")
        plt.ylabel("density")
        plt.title("Cosine angle distribution: w vs u")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Violin plots: inferred distributions grouped by categorical ground truth (-1, 0, 1)
    true_all_raw = np.concatenate([rr["true_w"] for rr in triplet_records]).astype(float)
    inf_all = np.concatenate([rr["inf_w"] for rr in triplet_records]).astype(float)
    finite_mask = np.isfinite(true_all_raw) & np.isfinite(inf_all)
    true_all = true_all_raw[finite_mask]
    inf_all = inf_all[finite_mask]
    u_all = np.concatenate([rr["inf_u"] for rr in triplet_records]).astype(float)
    finite_mask_u = np.isfinite(true_all_raw) & np.isfinite(u_all)
    true_u = true_all_raw[finite_mask_u]
    u_all = u_all[finite_mask_u]

    if true_all.size > 0 or true_u.size > 0:
        cats = np.array([-1.0, 0.0, 1.0])
        xticks = np.array([1, 2, 3], dtype=float)
        dx = 0.16

        w_data, w_pos = [], []
        for kk, cc in enumerate(cats):
            vals = inf_all[np.isclose(true_all, cc)]
            if vals.size > 0:
                w_data.append(vals)
                w_pos.append(xticks[kk] - dx)

        u_data, u_pos = [], []
        for kk, cc in enumerate(cats):
            vals = u_all[np.isclose(true_u, cc)]
            if vals.size > 0:
                u_data.append(vals)
                u_pos.append(xticks[kk] + dx)

        plt.figure(figsize=(7.2, 4.6))

        if len(w_data) > 0:
            vp_w = plt.violinplot(w_data, positions=w_pos, widths=0.26, showmeans=True, showextrema=False)
            for body in vp_w['bodies']:
                body.set_facecolor('tab:blue')
                body.set_alpha(0.35)
            vp_w['cmeans'].set_color('tab:blue')

        if len(u_data) > 0:
            vp_u = plt.violinplot(u_data, positions=u_pos, widths=0.26, showmeans=True, showextrema=False)
            for body in vp_u['bodies']:
                body.set_facecolor('tab:orange')
                body.set_alpha(0.35)
            vp_u['cmeans'].set_color('tab:orange')

        plt.xticks(xticks, ['-1', '0', '1'])
        plt.xlabel("Ground-truth weight category")
        plt.ylabel("Inferred value")
        plt.title("Inferred distributions by true weight (w vs u)")
        plt.scatter([], [], color='tab:blue', label='w inference')
        plt.scatter([], [], color='tab:orange', label='u inference')
        plt.legend()
        plt.tight_layout()
        plt.show()

# %% iterate random 4-neuron motifs from 100-neuron spike data
if not quartet_records:
    attempts = 0
    while len(quartet_records) < n_quartet_samples and attempts < max_attempts_quartet:
        attempts += 1
        quartet = np.sort(rng.choice(N, size=4, replace=False))

        S_sub = W[np.ix_(quartet, quartet)].astype(float)
        np.fill_diagonal(S_sub, 0.0)
        if np.count_nonzero(S_sub) == 0:
            continue

        firing_4 = project_quartet_firing(firing_full, quartet)

        out = infer_quartet_weights_from_firing(
            firing_quartet=firing_4,
            S_sub=S_sub,
            lt_steps=timesteps,
            adapt_window=adapt_window,
        )
        if out is None:
            continue

        quartet_records.append({
            "quartet": quartet,
            "R": out["R"],
            "sign": out["sign"],
            "cos": out["cos"],
            "inf_w": out["inf_w"],
            "true_w": out["true_w"],
        })

        if len(quartet_records) % 25 == 0:
            print(f"Processed {len(quartet_records)}/{n_quartet_samples} connected quartets")

    print(f"\nCompleted quartet inference: {len(quartet_records)} valid quartets")
    if attempts >= max_attempts_quartet and len(quartet_records) < n_quartet_samples:
        print(f"Stopped after {attempts} attempts; collected {len(quartet_records)} connected quartets.")

if CACHE_MAXCAL and (triplet_records or quartet_records):
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(
            {
                "triplet_records": triplet_records,
                "quartet_records": quartet_records,
            },
            f,
        )

quartet_R = None
if len(quartet_records) > 0:
    R4_vals = np.array([rr["R"] for rr in quartet_records], dtype=float)
    sign4_vals = np.array([rr["sign"] for rr in quartet_records], dtype=float)
    cos4_vals = np.array([rr["cos"] for rr in quartet_records], dtype=float)
    quartet_R = R4_vals

    print(f"Mean Pearson R (4N): {np.nanmean(R4_vals):.4f}")
    print(f"Mean sign-corr (4N): {np.nanmean(sign4_vals):.4f}")
    print(f"Mean cosine (4N): {np.nanmean(cos4_vals):.4f}")

    plt.figure(figsize=(12, 3.5))
    plt.subplot(131)
    plt.hist(R4_vals[np.isfinite(R4_vals)], bins=30, alpha=0.8, density=True)
    plt.title("Quartet Pearson R")
    plt.xlabel("R")

    plt.subplot(132)
    plt.hist(sign4_vals[np.isfinite(sign4_vals)], bins=30, alpha=0.8, density=True)
    plt.title("Quartet sign-corr")
    plt.xlabel("sign correlation")

    plt.subplot(133)
    plt.hist(cos4_vals[np.isfinite(cos4_vals)], bins=30, alpha=0.8, density=True)
    plt.title("Quartet cosine")
    plt.xlabel("cosine")

    plt.tight_layout()
    plt.show()

# %% compare 2N (u), 3N (w), and 4N inference
if (triplet_Ru is not None) or (triplet_R is not None) or (quartet_R is not None):
    plt.figure(figsize=(6.6, 4.4))
    if triplet_Ru is not None:
        plt.hist(triplet_Ru[np.isfinite(triplet_Ru)], bins=30, alpha=0.4, label='2N (u)', color='tab:orange', density=True)
    if triplet_R is not None:
        plt.hist(triplet_R[np.isfinite(triplet_R)], bins=30, alpha=0.4, label='3N (w)', color='tab:blue', density=True)
    if quartet_R is not None:
        plt.hist(quartet_R[np.isfinite(quartet_R)], bins=30, alpha=0.4, label='4N (w)', color='tab:green', density=True)
    plt.xlabel("Pearson R")
    plt.ylabel("density")
    plt.title("Inference quality: 2N vs 3N vs 4N")
    plt.legend()
    plt.tight_layout()
    plt.show()