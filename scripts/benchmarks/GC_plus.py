##################################################################
## a more solid likelihood-based method for GC
### Focus on spiking GC because transfer entropy requires trial structure.
##################################################################
"""
Minimal Kim-Brown-style point-process GLM baseline vs MaxCal.

Revision-purpose comparison:
    - MaxCal returns one signed coupling per directed edge.
    - Kim-Brown GLM baseline also returns one signed coupling per directed edge:
          beta_lag1 + beta_lag2
      from a logistic spike-history GLM with L=2 history bins.

The likelihood-ratio GC strength is intentionally not used for the signed-weight
comparison, because it is nonnegative and measures predictive importance rather
than signed coupling.

Edge-vector order throughout:
    [1->2, 1->3, 2->1, 2->3, 3->2, 3->1]

For the simulator, S[target, source] is the true synaptic weight.
"""

import random
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import pearsonr

from maxcal_network import (
    spk2statetime,
    compute_tauC,
    cos_ang,
)

matplotlib.rc("xtick", labelsize=20)
matplotlib.rc("ytick", labelsize=20)

np.random.seed(42)
random.seed(42)


# =============================================================================
# Simulation
# =============================================================================

def LIF_firing_voltage(
    synaptic_weights,
    noise_amp,
    syn_delay=None,
    syn_ratio=None,
    stim=None,
    stim_inter=100,
    stim_dur=0.5,
    counter=0,
    dt=0.1,
    timesteps=100000,
):
    """
    Simulate a 3-neuron LIF network.

    Parameters
    ----------
    synaptic_weights : array, shape (3, 3)
        S[target, source] is the synaptic effect from source to target.
    noise_amp : float
        Intrinsic voltage noise amplitude.
    dt : float
        Simulation time step.
    timesteps : int
        Number of simulation steps.

    Returns
    -------
    firing : list
        firing[t] = [time, spike_indices], compatible with spk2statetime.
    v_neurons : array, shape (3, timesteps)
        Membrane voltage traces.
    spikes : array, shape (3, timesteps)
        Binary spike raster in simulation bins.
    """
    if syn_delay is not None:
        delay_buffer = np.zeros((1, syn_delay))

    tau = 10.0
    v_rest = -65.0
    v_threshold = -50.0
    v_reset = -65.0

    S = np.array(synaptic_weights, dtype=float, copy=True)
    if syn_ratio is not None:
        S[2, 0] *= syn_ratio
    np.fill_diagonal(S, 0.0)

    tau_synaptic = np.array([5.0, 5.0, 5.0])

    v_neurons = np.zeros((3, timesteps))
    synaptic_inputs = np.zeros((3, timesteps))
    spikes = np.zeros((3, timesteps), dtype=np.int8)

    firing = [(np.array([]), np.array([]))]

    for t in range(1, timesteps):
        v_neurons[:, t] = (
            v_neurons[:, t - 1]
            + dt / tau * (v_rest - v_neurons[:, t - 1])
            + np.random.randn(3) * noise_amp
        )

        spike_indices = np.where(v_neurons[:, t] > v_threshold)[0]
        spikes[spike_indices, t] = 1

        synaptic_input = S @ spikes[:, t - 1]

        if syn_delay is not None:
            delay_buffer = np.roll(delay_buffer, -1)
            delay_buffer[0, -1] = spikes[1, t - 1]
            synaptic_input[2] += delay_buffer[0, 0]

        synaptic_inputs[:, t] = synaptic_inputs[:, t - 1] + dt * (
            -synaptic_inputs[:, t - 1] / tau_synaptic + synaptic_input
        )

        v_neurons[:, t] += synaptic_inputs[:, t] * dt

        if stim is True:
            if t % stim_inter == 0:
                pert_neuron = random.randint(0, 2)
                counter = dt

            if 0 < counter < stim_dur:
                v_neurons[pert_neuron, t] = v_reset
                counter += dt
            elif counter >= stim_dur:
                counter = 0

        # Preserve original firing-list convention.
        firing.append([t + 0 * spike_indices, spike_indices])

        v_neurons[spike_indices, t] = v_reset

    return firing, v_neurons, spikes


# =============================================================================
# MaxCal inference
# =============================================================================

def infer_maxcal_empirical_weights(firing, lt, adapt_window=150, pseudocount=1.0):
    """
    Infer signed MaxCal-style couplings from empirical transition rates.

    Returns directed couplings in order:
        [1->2, 1->3, 2->1, 2->3, 3->2, 3->1]
    """
    spk_states, spk_times = spk2statetime(firing, adapt_window)
    tau, C = compute_tauC(spk_states, spk_times)

    tau_ = (tau + pseudocount) / lt
    C_ = (C + pseudocount) / lt
    M_inf = C_ / tau_[:, None]

    # State convention:
    # 000=0, 001=1, 010=2, 011=3,
    # 100=4, 101=5, 110=6, 111=7.
    f1, f2, f3 = M_inf[0, 4], M_inf[0, 2], M_inf[0, 1]

    w12 = np.log(M_inf[4, 6] / f2)  # source 1 modulates target 2
    w13 = np.log(M_inf[4, 5] / f3)  # source 1 modulates target 3
    w21 = np.log(M_inf[2, 6] / f1)  # source 2 modulates target 1
    w23 = np.log(M_inf[2, 3] / f3)  # source 2 modulates target 3
    w32 = np.log(M_inf[1, 3] / f2)  # source 3 modulates target 2
    w31 = np.log(M_inf[1, 5] / f1)  # source 3 modulates target 1

    inf_w = np.array([w12, w13, w21, w23, w32, w31])
    return inf_w, tau, C, M_inf


# =============================================================================
# Kim-Brown-style point-process GLM baseline, minimal L=2 version
# =============================================================================

def firing_to_raster(firing, N=3, lt=None):
    """Convert firing list to binary spike raster X[t, n]."""
    if lt is None:
        lt = len(firing) - 1

    X = np.zeros((lt, N), dtype=np.int8)

    for t in range(1, min(len(firing), lt + 1)):
        spike_ids = np.asarray(firing[t][1], dtype=int)
        if spike_ids.size > 0:
            spike_ids = spike_ids[(spike_ids >= 0) & (spike_ids < N)]
            X[t - 1, spike_ids] = 1

    return X


def make_history_design(X, target, source_keep=None, L=2, include_intercept=True):
    """
    Build spike-history design matrix for target neuron.

    y[t] = X[t, target]
    predictors = X[t-lag, j] for lag=1..L and j in source_keep.
    """
    X = np.asarray(X, dtype=float)
    T, N = X.shape

    if T <= L:
        raise ValueError("Time series must be longer than history length L.")

    if source_keep is None:
        source_keep = list(range(N))

    y = X[L:, target].astype(float)

    cols = []
    names = []
    for j in source_keep:
        for lag in range(1, L + 1):
            cols.append(X[L - lag : T - lag, j])
            names.append((j, lag))

    Xdesign = np.column_stack(cols) if len(cols) > 0 else np.empty((len(y), 0))

    if include_intercept:
        Xdesign = np.column_stack([np.ones(len(y)), Xdesign])

    return y, Xdesign, names


def logistic_loglike(beta, Xdesign, y, eps=1e-12):
    """Bernoulli log likelihood under logistic GLM."""
    eta = Xdesign @ beta
    p = expit(eta)
    p = np.clip(p, eps, 1 - eps)
    return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_logistic_ridge(Xdesign, y, l2=1e-3, maxiter=500):
    """
    Numerically stable logistic GLM fit with a small ridge penalty.

    The intercept is not penalized.
    """
    Xdesign = np.asarray(Xdesign, dtype=float)
    y = np.asarray(y, dtype=float)

    p = Xdesign.shape[1]
    beta0 = np.zeros(p)

    # Rare-spike intercept initialization.
    ybar = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
    beta0[0] = np.log(ybar / (1 - ybar))

    def objective(beta):
        eta = Xdesign @ beta
        nll = np.sum(np.logaddexp(0, eta) - y * eta)
        penalty = 0.5 * l2 * np.sum(beta[1:] ** 2)
        return nll + penalty

    def gradient(beta):
        eta = Xdesign @ beta
        p_hat = expit(eta)
        grad = Xdesign.T @ (p_hat - y)
        grad[1:] += l2 * beta[1:]
        return grad

    res = minimize(
        objective,
        beta0,
        jac=gradient,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "disp": False},
    )

    if not res.success:
        warnings.warn(f"Logistic fit did not fully converge: {res.message}")

    beta = res.x
    ll = logistic_loglike(beta, Xdesign, y)
    return beta, ll, res


def kim_brown_signed_pair(X, source, target, L=2, l2=1e-3):
    """
    Minimal Kim-Brown-style signed coupling estimate for source -> target.

    Fit full point-process logistic GLM using all neurons' histories, then use
    the source-history filter coefficient sum over L=2 bins:

        signed_effect = beta(source, lag=1) + beta(source, lag=2)

    This is the signed quantity compared to true synaptic weights.
    """
    _, N = X.shape

    y, Xdesign, names = make_history_design(
        X,
        target=target,
        source_keep=list(range(N)),
        L=L,
    )
    beta, ll, _ = fit_logistic_ridge(Xdesign, y, l2=l2)

    source_coef_inds = [
        k + 1 for k, name in enumerate(names) if name[0] == source
    ]
    source_filter = beta[source_coef_inds]

    signed_effect = float(np.sum(source_filter[:L]))
    return signed_effect, source_filter, beta, names


def infer_kim_brown_signed(firing, N=3, lt=None, L=2, l2=1e-3):
    """
    Infer six directed signed Kim-Brown GLM effects.

    Output order:
        [1->2, 1->3, 2->1, 2->3, 3->2, 3->1]
    """
    X = firing_to_raster(firing, N=N, lt=lt)

    pairs = [
        (0, 1),  # 1 -> 2
        (0, 2),  # 1 -> 3
        (1, 0),  # 2 -> 1
        (1, 2),  # 2 -> 3
        (2, 1),  # 3 -> 2
        (2, 0),  # 3 -> 1
    ]

    signed = []
    filters = {}
    for source, target in pairs:
        val, source_filter, beta, names = kim_brown_signed_pair(
            X,
            source=source,
            target=target,
            L=L,
            l2=l2,
        )
        signed.append(val)
        filters[(source, target)] = source_filter

    return np.array(signed), X, filters


# =============================================================================
# Scoring and plotting
# =============================================================================

def safe_pearsonr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return pearsonr(x, y)[0]


def sign_accuracy(est, true):
    est = np.asarray(est, dtype=float)
    true = np.asarray(true, dtype=float)
    mask = true != 0
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(np.sign(est[mask]) == np.sign(true[mask]))


def true_weight_vector(S):
    """
    Return ground-truth weights in edge-vector order:
        [1->2, 1->3, 2->1, 2->3, 3->2, 3->1]
    where S[target, source].
    """
    return np.array([
        S[1, 0],
        S[2, 0],
        S[0, 1],
        S[2, 1],
        S[1, 2],
        S[0, 2],
    ])


def print_comparison(true_s, maxcal_w, kb_w):
    print("True weights:   ", true_s)
    print("MaxCal weights: ", maxcal_w)
    print("KB-GLM weights: ", kb_w)
    print("")
    print("corr(MaxCal,true): ", safe_pearsonr(maxcal_w, true_s))
    print("corr(KB,true):     ", safe_pearsonr(kb_w, true_s))
    print("signacc(MaxCal):   ", sign_accuracy(maxcal_w, true_s))
    print("signacc(KB):       ", sign_accuracy(kb_w, true_s))
    print("cos(MaxCal,true):  ", cos_ang(maxcal_w, true_s))
    print("cos(KB,true):      ", cos_ang(kb_w, true_s))


def plot_single_comparison(true_s, maxcal_w, kb_w, L_KB=2):
    labels = ["1→2", "1→3", "2→1", "2→3", "3→2", "3→1"]
    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(9, 4))
    plt.bar(x - width, true_s, width, label="true")
    plt.bar(x, maxcal_w, width, label="MaxCal")
    plt.bar(x + width, kb_w, width, label=f"KB-GLM L={L_KB}")
    plt.axhline(0, color="k", linewidth=1)
    # Show edge label plus the true weight under each tick
    labels_with_weights = [f"{lab}\n({val:.0f})" for lab, val in zip(labels, true_s)]
    plt.xticks(x, labels_with_weights)
    plt.ylabel("signed coupling")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# Main example and optional scan
# =============================================================================

def run_single_example():
    """Run one EI motif example and print the clean comparison."""
    lt = 100000
    L_KB = 2

    Wij_ei = np.array([
        [0, 1, -2],
        [1, 0, -2],
        [1, 1, 0],
    ])

    S = Wij_ei * 16

    firing, volt, spikes = LIF_firing_voltage(
        S,
        noise_amp=2,
        stim=False,
        timesteps=lt,
    )

    true_s = true_weight_vector(S)

    maxcal_w, tau, C, M_inf = infer_maxcal_empirical_weights(
        firing,
        lt=lt,
        adapt_window=150,
        pseudocount=1.0,
    )

    kb_w, X, kb_filters = infer_kim_brown_signed(
        firing,
        N=3,
        lt=lt,
        L=L_KB,
        l2=1e-3,
    )

    print_comparison(true_s, maxcal_w, kb_w)
    plot_single_comparison(true_s, maxcal_w, kb_w, L_KB=L_KB)

    return {
        "firing": firing,
        "volt": volt,
        "spikes": spikes,
        "tau": tau,
        "C": C,
        "M_inf": M_inf,
        "true_s": true_s,
        "maxcal_w": maxcal_w,
        "kb_w": kb_w,
        "kb_filters": kb_filters,
    }


def run_scan(L_KB=2):
    """
    Optional scan over network strength.

    Methods:
        0 = MaxCal signed coupling
        1 = Kim-Brown GLM signed coupling, L=2
    """
    N = 3
    lt = 100000
    reps = 3

    w_s = np.array([2, 4, 8, 16, 32, 64]) * 2

    Wij_ei = np.array([
        [0, 1, -2],
        [1, 0, -2],
        [1, 1, 0],
    ])

    Ws = (Wij_ei,)
    n_motifs = len(Ws)
    n_methods = 2

    Rcorr = np.full((n_motifs, len(w_s), reps, n_methods), np.nan)
    Sacc = np.full_like(Rcorr, np.nan)
    Coss = np.full_like(Rcorr, np.nan)

    for rr in range(reps):
        for ww, Wij in enumerate(Ws):
            for ii, scale in enumerate(w_s):
                print(f"repeat={rr}, motif={ww}, scale={scale}")

                S = Wij * scale
                firing, volt, spikes = LIF_firing_voltage(
                    S,
                    noise_amp=2,
                    stim=None,
                    timesteps=lt,
                )

                true_s = true_weight_vector(S)

                maxcal_w, tau, C, M_inf = infer_maxcal_empirical_weights(
                    firing,
                    lt=lt,
                    adapt_window=150,
                    pseudocount=1.0,
                )

                kb_w, X, kb_filters = infer_kim_brown_signed(
                    firing,
                    N=N,
                    lt=lt,
                    L=L_KB,
                    l2=1e-3,
                )

                vecs = [maxcal_w, kb_w]
                for mm, vec in enumerate(vecs):
                    Rcorr[ww, ii, rr, mm] = safe_pearsonr(vec, true_s)
                    Sacc[ww, ii, rr, mm] = sign_accuracy(vec, true_s)
                    Coss[ww, ii, rr, mm] = cos_ang(vec, true_s)

    return Rcorr, Sacc, Coss, w_s, L_KB


def plot_scan_results(Rcorr, Sacc, Coss, w_s, selected_motif=0, L_KB=2):
    meas = [Rcorr, Sacc, Coss]
    titles = ["Pearson R", "sign accuracy", "cosine"]
    method_labels = ["MaxCal", f"KB-GLM L={L_KB}"]

    w_s = np.asarray(w_s)
    n_w = len(w_s)

    x = np.arange(n_w)
    width = 0.35
    labels = [f"w={v:g}" for v in w_s]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)

    for k, ax in enumerate(axes):
        arr = meas[k]

        for mm in range(2):
            vals = arr[selected_motif, :, :, mm]

            # Protect against mismatch between w_s and array size
            vals = vals[:n_w, :]

            mu = np.nanmean(vals, axis=1)
            sd = np.nanstd(vals, axis=1)

            ax.bar(
                x + (mm - 0.5) * width,
                mu,
                width,
                yerr=sd,
                capsize=4,
                label=method_labels[mm],
            )

        ax.set_title(titles[k])
        ax.set_xlabel("weight scale")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
        ax.tick_params(axis="x", labelbottom=True)

        ax.axhline(0, color="k", linewidth=0.8, alpha=0.5)

        if k == 0:
            ax.legend(frameon=False)

    fig.tight_layout()
    plt.show()

# %%
if __name__ == "__main__":
    # results = run_single_example()

    # Uncomment for scan.
    L_KB = 10
    Rcorr, Sacc, Coss, w_s, L_KB = run_scan(L_KB=L_KB)
    # %% plotting bars
    plot_scan_results(Rcorr, Sacc, Coss, w_s, L_KB=10)

    # Optional pickle save example.
    # import pickle
    # filename = "spk_GC_comparison.pkl"
    # data = {"coss": Coss}
    # with open(filename, "wb") as file:
    #     pickle.dump(data, file)

plt.show()
