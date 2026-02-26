import matplotlib.pyplot as plt
import mne
import numpy as np
import os
import pickle

from mne.io import read_raw_edf
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


def load_files(verbose=False):
    all_files = os.listdir(data_dir)
    files = {
        stage: [f for f in all_files if f.endswith(f"_{stage:s}.edf")] 
        for stage in sleep_stages
    }
    
    if verbose:
        for stage in sleep_stages:
            print(f"\nStage: {stage:s}")
            for f in files[stage]:
                print(f)
    
    return files


def make_features(delta=True, theta=True, alpha=True, beta=True):
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
    
    # selected frequency band indices in freqs array
    freq_idx = { 
        fband : (fidx(freq_bands[fband][0]), fidx(freq_bands[fband][1])) 
        for fband in fbands_selected 
    }
    #print("Frequency axis: ", freqs.min(), freqs.max(), freqs.shape)
    #print(freq_idx)
    f_min_idx = fidx(f_min)
    f_max_idx = fidx(f_max)
    #print(f"f_min_idx: {f_min_idx:d}")
    #print(f"f_max_idx: {f_max_idx:d}")

    # extract spectral features from EEG files
    files = load_files()
    t_epoch = 10 # sec
    n_samples_per_epoch = int(t_epoch*fs)
    X = [] # initialize feature matrix X
    y = [] # initialize target labels y
    for i_stage, stage in enumerate(sleep_stages):
        print(f"Analyzing sleep stage: {stage:s}", end="\r")
        for i_file, f in enumerate(files[stage]):
            #print(f"File {i_file+1:d}/{len(files[stage]):d}: {f:s}", end="\r")
            # mini EDF reader
            #data, fs, chs = read_edf(f"{data_dir:s}/{f:s}")
            # pyedflib
            #data, data_headers, header = read_edf(f"{data_dir:s}/{f:s}")
            #fs = data_headers[0]['sample_rate']
            #chs = [h['label'] for h in data_headers]
            #data = data.T
            # MNE
            raw = read_raw_edf(f"{data_dir:s}/{f:s}", 
                               preload=True, verbose=False)
            data = raw.get_data().T
            fs = raw.info['sfreq']
            chs = raw.info.ch_names
            # analyze epochs
            n_t, n_ch = data.shape
            #print(f"n_t = {n_t:d}, n_ch = {n_ch:d}")
            n_ep = n_t // n_samples_per_epoch
            for i_ep in range(n_ep):
                #print(f"epoch: {i_ep+1:d}/{n_ep:d}", end="\r")
                t0 = i_ep*n_samples_per_epoch
                t1 = (i_ep+1)*n_samples_per_epoch
                _, psd = welch(data[t0:t1,:], fs=fs, axis=0, nperseg=nperseg)
                #psd = psd[f_min_idx:f_max_idx,:]
                # normalize spectrum
                psd /= psd.sum()
                #print(psd.shape)
                row = [np.sum(psd[freq_idx[fband][0]:freq_idx[fband][1],i_ch]) 
                       for fband in freq_idx for i_ch in range(n_ch)]
                X.append(row)
                y.append(i_stage)
    print()
    # NOTE: channels from last EEG loaded
    # this is only OK if all files have the same channels and order (!!!)
    #chs = raw.info.ch_names 
    feature_names = [f"{fband:s}_{chs[i_ch]:s}" 
                     for fband in freq_idx for i_ch in range(n_ch)]
                
    X = np.array(X)
    y = np.array(y)
    #print(X.shape)
    #print(y.shape)
    #np.savez("spectral_features.npz", X=X, y=y)
    return X, y, feature_names


def show_random_data(channel = 'O1'):
    files = load_files()
    n_subj = 15
    rnd_idx = np.random.choice(n_subj)
    #print(f"Random subject index: {rnd_idx:d}")

    # wake data
    f_W = f"{data_dir:s}/{files['W'][rnd_idx]:s}"
    # mini EDF reader
    #data_W, fs_W, ch_W = read_edf(f_W)
    # pyedflib
    #data_W, data_W_headers, header_W = read_edf(f_W)
    #fs_W = data_W_headers[0]['sample_rate']
    #ch_W = [h['label'] for h in data_W_headers]
    # MNE
    raw = read_raw_edf(f_W, preload=True, verbose=False)
    data_W = raw.get_data().T
    fs_W = raw.info['sfreq']
    ch_W = raw.info.ch_names
            
    id_ch_W = ch_W.index(channel)
    data_W = data_W[:,id_ch_W]
    
    # N1 data
    f_N1 = f"{data_dir:s}/{files['N1'][rnd_idx]:s}"
    
    # mini EDF reader
    # does not work with Numpy 2.x
    #data_N1, fs_N1, ch_N1 = read_edf(f_N1)
    
    # pyedflib
    #data_N1, data_N1_headers, header_N1 = read_edf(f_N1)
    #fs_N1 = data_N1_headers[0]['sample_rate']
    #ch_N1 = [h['label'] for h in data_N1_headers]

    # MNE
    raw = read_raw_edf(f_N1, preload=True, verbose=False)
    data_N1 = raw.get_data().T
    fs_N1 = raw.info['sfreq']
    ch_N1 = raw.info.ch_names
    
    id_ch_N1 = ch_N1.index(channel)
    data_N1 = data_N1[:,id_ch_N1]

    # power spectral densities
    #fs = 250 # Hz
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

    # start, stop, num=50, endpoint=True
    #freqs0 = np.linspace(start=0, stop=fs/2, num=nperseg//2+1, endpoint=True)
    #print(np.all(freqs_W == freqs0))
    #print(np.all(freqs_N1 == freqs0))

    # time courses
    # TODO: pick random segment
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


def main():
    pass


if __name__ == "__main__":
    main()