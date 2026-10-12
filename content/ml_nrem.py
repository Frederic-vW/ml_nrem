import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             ConfusionMatrixDisplay,
                             f1_score)
from sklearn.model_selection import (GridSearchCV,
                                     train_test_split,
                                     StratifiedShuffleSplit)


# global variables
data_dir = "./data"
sleep_stages = ['W', 'N1', 'N2', 'N3']


def load_meta():
    """Channel names, channel positions (standard 10-05, metres), sfreq."""
    with np.load(f"{data_dir:s}/meta.npz") as d:
        return {k: d[k] for k in d.files}


def load_subject(subject):
    """
    Load the raw EEG of one subject (e.g. 'S01' or 1).
    Returns: dict {stage: data (n_samples, n_channels) in volts}, fs, chs
    """
    if not isinstance(subject, str):
        subject = f"S{subject:02d}"
    with np.load(f"{data_dir:s}/{subject:s}.npz") as d:
        data = {
            stage: (d[stage] * d[f"{stage}_cal"] + d[f"{stage}_offsets"])
                   * d[f"{stage}_unit"]
            for stage in sleep_stages
        }
        fs = float(d['sfreq'])
        chs = list(d['ch_names'])
    return data, fs, chs


def load_spectra():
    """Welch spectra of all 10 sec epochs, psd: (n_epochs, n_freqs, n_ch)."""
    with np.load(f"{data_dir:s}/spectra.npz") as d:
        return {k: d[k] for k in d.files}


def make_features(delta=True, theta=True, alpha=True, beta=True,
                  from_raw=False):
    # selected frequency bands
    fbands_selected = []
    if delta:
        fbands_selected.append('delta')
    if theta:
        fbands_selected.append('theta')
    if alpha:
        fbands_selected.append('alpha')
    if beta:
        fbands_selected.append('beta')

    if from_raw:
        X_bands, y, bands, chs = _features_from_raw()
    else:
        # pre-computed band powers (see tools/edf_to_npz.py)
        with np.load(f"{data_dir:s}/spectral_features.npz") as d:
            X_bands, y = d['X_bands'], d['y']
            bands, chs = list(d['bands']), list(d['ch_names'])

    # X_bands: (n_epochs, n_bands, n_ch) -> X: (n_epochs, n_selected*n_ch)
    idx = [bands.index(fband) for fband in fbands_selected]
    X = X_bands[:, idx, :].reshape(len(y), -1)
    feature_names = [f"{fband:s}_{ch:s}"
                     for fband in fbands_selected for ch in chs]
    return X, y, feature_names


def _features_from_raw():
    """Re-compute the band power features from the raw EEG (slow)."""
    f_min, f_max = 0.5, 25
    freq_bands = {
        'delta' : (f_min,3.5),
        'theta' : (4,7.5),
        'alpha' : (8,12.5),
        'beta' : (13,f_max),
    }
    # frequency axis
    fs = 250 # Hz
    nperseg = 512 # Welch parameter
    freqs = np.linspace(start=0, stop=fs/2, num=nperseg//2+1, endpoint=True)

    # function returning the closest index of frequency f in freqs
    fidx = lambda f: np.argmin(np.abs(f-freqs))
    freq_idx = {
        fband : (fidx(freq_bands[fband][0]), fidx(freq_bands[fband][1]))
        for fband in freq_bands
    }

    subjects = load_meta()['subjects']
    raws = {subj: load_subject(subj) for subj in subjects}
    t_epoch = 10 # sec
    X = [] # initialize feature matrix X
    y = [] # initialize target labels y
    for i_stage, stage in enumerate(sleep_stages):
        print(f"Analyzing sleep stage: {stage:s}", end="\r")
        for subj in subjects:
            data_all, fs, chs = raws[subj]
            data = data_all[stage]
            n_samples_per_epoch = int(t_epoch*fs)
            # analyze epochs
            n_t, n_ch = data.shape
            n_ep = n_t // n_samples_per_epoch
            for i_ep in range(n_ep):
                t0 = i_ep*n_samples_per_epoch
                t1 = (i_ep+1)*n_samples_per_epoch
                _, psd = welch(data[t0:t1,:], fs=fs, axis=0, nperseg=nperseg)
                # normalize spectrum
                psd /= psd.sum()
                row = [[np.sum(psd[freq_idx[fband][0]:freq_idx[fband][1],i_ch])
                        for i_ch in range(n_ch)] for fband in freq_idx]
                X.append(row)
                y.append(i_stage)
    print()
    return np.array(X), np.array(y), list(freq_idx), chs


def show_random_data(channel = 'O1'):
    subjects = load_meta()['subjects']
    rnd_idx = np.random.choice(len(subjects))
    data, fs, chs = load_subject(subjects[rnd_idx])
    id_ch = chs.index(channel)
    data_W, fs_W = data['W'][:,id_ch], fs
    data_N1, fs_N1 = data['N1'][:,id_ch], fs

    # power spectral densities
    nperseg = 512 # 512 # Welch parameter
    freqs_W, psd_W = welch(data_W, fs=fs_W, nperseg=nperseg)
    freqs_N1, psd_N1 = welch(data_N1, fs=fs_N1, nperseg=nperseg)
    f_lo, f_hi = 1, 25
    i_lo = np.argmin(np.abs(freqs_W-f_lo))
    i_hi = np.argmin(np.abs(freqs_W-f_hi))

    # plot
    t_show = 8 # sec

    # select random epoch
    # W
    n_samples_per_ep_W = int(t_show*fs_W)
    nt_W = len(data_W)
    n_ep_W = nt_W // n_samples_per_ep_W  # number of available epochs
    i_ep_W = np.random.choice(n_ep_W)
    t0 = i_ep_W*n_samples_per_ep_W
    t1 = (i_ep_W+1)*n_samples_per_ep_W
    data_W = data_W[t0:t1]
    time_W = np.arange(len(data_W))/fs_W
    # N1
    n_samples_per_ep_N1 = int(t_show*fs_N1)
    nt_N1 = len(data_N1)
    n_ep_N1 = nt_N1 // n_samples_per_ep_N1  # number of available epochs
    i_ep_N1 = np.random.choice(n_ep_N1)
    t0 = i_ep_N1*n_samples_per_ep_N1
    t1 = (i_ep_N1+1)*n_samples_per_ep_N1
    data_N1 = data_N1[t0:t1]
    time_N1 = np.arange(len(data_N1))/fs_N1

    # time courses
    fsize = 16
    fig, ax = plt.subplots(3, 1, figsize=(12,9)) #, sharex=True, sharey=True)
    ax[0].plot(time_W, 1e6*data_W, '-k', lw=2)
    ax[0].set_title(f"Subject: {rnd_idx:d}", fontsize=fsize+2)
    ax[1].plot(time_N1, 1e6*data_N1, '-b', lw=2)
    # EEG limits
    ymn_W, ymx_W = ax[0].get_ylim()
    ymn_N1, ymx_N1 = ax[1].get_ylim()
    ymn = min(ymn_W, ymn_N1)
    ymx = max(ymx_W, ymx_N1)
    ax[0].set_ylim(ymn, ymx)
    ax[1].set_ylim(ymn, ymx)
    ax[1].set_xlabel("time (sec)", fontsize=fsize)
    ax[0].set_ylabel("voltage " + r"$(\mu V$)", fontsize=fsize)
    ax[1].set_ylabel("voltage " + r"$(\mu V$)", fontsize=fsize)
    # spectra
    ax[2].semilogy(freqs_W[i_lo:i_hi], psd_W[i_lo:i_hi], '-k', lw=2)
    ax[2].semilogy(freqs_N1[i_lo:i_hi], psd_N1[i_lo:i_hi], '-b', lw=2)
    ax[2].set_xlabel("freq. (Hz)", fontsize=fsize)
    ax[2].set_ylabel("power " + r"$\mu V^2/Hz$", fontsize=fsize)
    ax[2].set_title("Power spectral density", fontsize=fsize)
    ymn, ymx = ax[2].get_ylim()
    ax[2].text(x=1, y=1.2*ymn, s="delta", fontsize=fsize) # delta
    ax[2].text(x=5, y=1.2*ymn, s="theta", fontsize=fsize) # theta
    ax[2].text(x=8.5, y=1.2*ymn, s="alpha", fontsize=fsize) # alpha
    ax[2].text(x=13, y=1.2*ymn, s="beta", fontsize=fsize) # beta
    for freq in [4, 8, 12]:
        ax[2].axvline(freq, lw=2)
    plt.tight_layout()
    return fig


def plot_spectra(channel = 'O1', subject = None):
    """Average power spectrum of each sleep stage (all subjects or one)."""
    sp = load_spectra()
    chs = list(sp['ch_names'])
    id_ch = chs.index(channel)
    f_lo, f_hi = 0.5, 25
    i_f = (sp['freqs'] >= f_lo) & (sp['freqs'] <= f_hi)
    freqs = sp['freqs'][i_f]
    if subject is None:
        sel = np.ones(len(sp['y']), dtype=bool)
        title = "all subjects"
    else:
        sel = sp['subject'] == subject
        title = f"Subject: {subject:d}"

    fsize = 16
    fig, ax = plt.subplots(1, 1, figsize=(12,5))
    for i_stage, stage in enumerate(sleep_stages):
        psd = sp['psd'][sel & (sp['y'] == i_stage)][:, i_f, id_ch]
        ax.semilogy(freqs, 1e12*psd.mean(axis=0), lw=2, label=stage)
    ax.set_xlabel("freq. (Hz)", fontsize=fsize)
    ax.set_ylabel("power " + r"$\mu V^2/Hz$", fontsize=fsize)
    ax.set_title(f"Power spectral density, channel {channel:s}, {title:s}",
                 fontsize=fsize)
    for freq in [4, 8, 13]:
        ax.axvline(freq, color='gray', lw=1)
    ax.legend(fontsize=fsize)
    plt.tight_layout()
    return fig


def main():
    pass


if __name__ == "__main__":
    main()
