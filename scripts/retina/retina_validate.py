# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 21:22:12 2026

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
import scipy
from pathlib import Path
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


# %% loading retina
mat_path = DATA_DIR / "Data_processed.mat"
dataset = 1  # 0-natural, 1-Brownian, 2-repeats

mat_data = scipy.io.loadmat(mat_path)
spk_data = mat_data['spike_times'][0][dataset][0]
spk_ids  = mat_data['cell_IDs'][0][dataset][0]
reps = len(spk_data)


# %% optional raster check
nid = 1

plt.figure(figsize=(8, 4))
for rr in range(reps):
    spkt = spk_data[rr].squeeze()
    spki = spk_ids[rr].squeeze()
    pos = np.where(spki == nid)[0]
    plt.plot(spkt[pos], np.ones(len(pos)) + rr, 'k.')

plt.xlabel("time (ms)")
plt.ylabel("repeat")
plt.title(f"Raster for neuron {nid}")
plt.tight_layout()
plt.show()


# %% one-trial cross-correlation
def spike_xcorr_one_trial(
    spk_t,
    spk_id,
    nid1,
    nid2,
    bin_size=20,
    maxlag=10,
    subtract_mean=True,
    normalize=False,
):
    """
    Cross-correlation of two neurons for one repeat.

    Spike times are in ms.
    Trial length is inferred from that repeat's max spike time.

    maxlag is in bins.
    Example: bin_size=20, maxlag=10 gives +/-200 ms.
    """

    spk_t = np.asarray(spk_t).squeeze()
    spk_id = np.asarray(spk_id).squeeze()

    if len(spk_t) == 0:
        return None, None

    T = np.ceil(np.max(spk_t))
    nbins = int(np.ceil(T / bin_size))

    if nbins <= 1:
        return None, None

    x = np.zeros(nbins)
    y = np.zeros(nbins)

    # same convention as before:
    # (0, bin_size] -> bin 0
    # (bin_size, 2*bin_size] -> bin 1
    bins1 = np.ceil(spk_t[spk_id == nid1] / bin_size).astype(int) - 1
    bins2 = np.ceil(spk_t[spk_id == nid2] / bin_size).astype(int) - 1

    bins1 = bins1[(bins1 >= 0) & (bins1 < nbins)]
    bins2 = bins2[(bins2 >= 0) & (bins2 < nbins)]

    np.add.at(x, bins1, 1)
    np.add.at(y, bins2, 1)

    if subtract_mean:
        x = x - x.mean()
        y = y - y.mean()

    cc_full = np.correlate(x, y, mode='full')
    lags_full = np.arange(-nbins + 1, nbins) * bin_size

    keep = np.abs(lags_full) <= maxlag * bin_size

    lags = lags_full[keep]
    cc = cc_full[keep]

    if normalize:
        denom = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
        if denom > 0:
            cc = cc / denom

    return lags, cc


# %% average cross-correlation across repeats
def spike_xcorr_across_repeats(
    spk_data,
    spk_ids,
    nid1,
    nid2,
    bin_size=20,
    maxlag=10,
    subtract_mean=True,
    normalize=False,
):
    cc_all = []
    lags_ref = None

    for rr in range(len(spk_data)):
        lags, cc = spike_xcorr_one_trial(
            spk_t=spk_data[rr].squeeze(),
            spk_id=spk_ids[rr].squeeze(),
            nid1=nid1,
            nid2=nid2,
            bin_size=bin_size,
            maxlag=maxlag,
            subtract_mean=subtract_mean,
            normalize=normalize,
        )

        if lags is None:
            continue

        if lags_ref is None:
            lags_ref = lags

        if len(lags) == len(lags_ref) and np.all(lags == lags_ref):
            cc_all.append(cc)

    cc_all = np.asarray(cc_all)

    cc_mean = np.nanmean(cc_all, axis=0)
    cc_sem = np.nanstd(cc_all, axis=0) / np.sqrt(cc_all.shape[0])

    return lags_ref, cc_mean, cc_sem, cc_all


# %% run x-corr
nid1 = 3
nid2 = 34

bin_size = 20
maxlag = 10  # +/-10 bins = +/-200 ms

lags, cc_mean, cc_sem, cc_all = spike_xcorr_across_repeats(
    spk_data=spk_data,
    spk_ids=spk_ids,
    nid1=nid1,
    nid2=nid2,
    bin_size=bin_size,
    maxlag=maxlag,
    subtract_mean=True,
    normalize=False,
)


## %% plot
plt.figure(figsize=(5, 3))
plt.bar(lags, cc_mean, width=0.9 * bin_size)
plt.errorbar(lags, cc_mean, yerr=cc_sem, fmt='none', capsize=2)
plt.axhline(0, color='k', linewidth=0.5)
plt.xlabel("Lag (ms)")
plt.ylabel("Mean-subtracted x-corr")
plt.title(f"Neuron {nid1} vs {nid2}")
plt.tight_layout()
plt.show()


# %% optional: compare raw and mean-subtracted
lags_raw, cc_raw, cc_raw_sem, _ = spike_xcorr_across_repeats(
    spk_data=spk_data,
    spk_ids=spk_ids,
    nid1=nid1,
    nid2=nid2,
    bin_size=bin_size,
    maxlag=maxlag,
    subtract_mean=False,
    normalize=False,
)

plt.figure(figsize=(5, 3))
plt.bar(lags_raw, cc_raw, width=0.9 * bin_size)
plt.errorbar(lags_raw, cc_raw, yerr=cc_raw_sem, fmt='none', capsize=2)
plt.axhline(0, color='k', linewidth=0.5)
plt.xlabel("Lag (ms)")
plt.ylabel("Raw x-corr")
plt.title(f"Raw: neuron {nid1} vs {nid2}")
plt.tight_layout()
plt.show()

plt.show()
