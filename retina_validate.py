# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 21:22:12 2026

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
import scipy
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

# %% loading retina!
mat_data = scipy.io.loadmat('C:/Users/kevin/Downloads/Data_processed.mat')
dataset = 1  # 0-natural, 1-Brownian, 2-repeats
nid = 1  # neuron example
reps = 62 #len(mat_data['spike_times'][0][dataset][0])  #62, 50
spk_data = mat_data['spike_times'][0][dataset][0]  # extract timing
spk_ids = mat_data['cell_IDs'][0][dataset][0]  # extract cell ID

plt.figure()
for nn in range(reps):
    spkt = spk_data[nn].squeeze()
    spki = spk_ids[nn].squeeze()
    pos = np.where(spki==nid)[0]
    plt.plot(spkt[pos], np.ones(len(pos))+nn,'k.')

# %% x-corr
def spike_xcorr(spk_t, spk_id, nid1, nid2, T, bin_size=20, maxlag=20):
    """
    Cross-correlation of two neurons after binning into bin_size-ms windows.

    Parameters
    ----------
    spk_t : array
        Spike times (ms)
    spk_id : array
        Neuron IDs
    nid1, nid2 : int
        Neuron IDs
    T : float
        Recording duration (ms)
    bin_size : float
        Bin width (ms)
    maxlag : int
        Maximum lag to display (in bins)

    Returns
    -------
    lags : array
        Lag (ms)
    cc : array
        Cross-correlation
    """

    nbins = int(np.ceil(T / bin_size))

    x = np.zeros(nbins)
    y = np.zeros(nbins)

    # same binning convention as before:
    # (0,20] -> bin 0, (20,40] -> bin 1, ...
    bins1 = np.ceil(spk_t[spk_id == nid1] / bin_size).astype(int) - 1
    bins2 = np.ceil(spk_t[spk_id == nid2] / bin_size).astype(int) - 1

    bins1 = bins1[(bins1 >= 0) & (bins1 < nbins)]
    bins2 = bins2[(bins2 >= 0) & (bins2 < nbins)]

    # count spikes per bin
    np.add.at(x, bins1, 1)
    np.add.at(y, bins2, 1)

    cc_full = np.correlate(x, y, mode='full')
    lags = np.arange(-nbins + 1, nbins) * bin_size

    keep = np.abs(lags) <= maxlag * bin_size

    return lags[keep], cc_full[keep]

# %% x-corr
lags, cc = spike_xcorr(
    spk_data[0].squeeze(),
    spk_ids[0].squeeze(),
    nid1=13,
    nid2=1,
    T=10000,
    bin_size=20,
    maxlag=10      # ±10 bins = ±200 ms
)

plt.figure(figsize=(5,3))
plt.bar(lags, cc, width=18)
plt.xlabel("Lag (ms)")
plt.ylabel("Cross-correlation")
plt.show()