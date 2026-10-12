"""
Convert the ml_nrem EDF recordings into MNE-free .npz files for JupyterLite.

Run once, locally, in an environment with MNE installed:

    python tools/edf_to_npz.py --edf-dir ../data --out-dir content/data

Outputs (in --out-dir):
    meta.npz               channel names, 3D positions (standard_1005), sfreq
    S00.npz ... S14.npz    raw EEG per subject, one int16 array per sleep stage
    spectral_features.npz  band-power features per 10 s epoch (for ML training)
    spectra.npz            per-epoch Welch spectra 0-45 Hz (for plotting)

Raw EEG is stored as the original EDF integers; volts are recovered with
    x_V = (raw.astype(float) * cal + offsets) * unit
which is exactly what mne.io.read_raw_edf computes.
"""
import argparse
import os
import re

import mne
import numpy as np
from scipy.signal import welch

SLEEP_STAGES = ['W', 'N1', 'N2', 'N3']
FREQ_BANDS = {
    'delta': (0.5, 3.5),
    'theta': (4, 7.5),
    'alpha': (8, 12.5),
    'beta': (13, 25),
}
T_EPOCH = 10    # sec
NPERSEG = 512   # Welch parameter
F_SPECTRA_MAX = 45  # Hz, upper limit of stored spectra


def read_edf(fname):
    """Return (int16 data (n_t, n_ch), cal, offsets, unit, sfreq, ch_names)."""
    raw = mne.io.read_raw_edf(fname, preload=True, verbose=False)
    ex = raw._raw_extras[0]
    sel = [i for i, ch in enumerate(ex['ch_names']) if ch in raw.ch_names]
    cal = np.asarray(ex['cal'])[sel]
    offsets = np.asarray(ex['offsets'])[sel]
    unit = np.asarray(ex['units'])[sel]
    x = raw.get_data()  # volts, (n_ch, n_t)
    dig = np.round((x / unit[:, None] - offsets[:, None]) / cal[:, None])
    assert dig.min() >= -32768 and dig.max() <= 32767
    dig = dig.astype(np.int16)
    # lossless check against MNE
    x_rec = (dig * cal[:, None] + offsets[:, None]) * unit[:, None]
    assert np.allclose(x_rec, x, rtol=0, atol=1e-12), fname
    return dig.T, cal, offsets, unit, raw.info['sfreq'], list(raw.ch_names)


def epoch_spectra(data_V, fs):
    """Welch spectra of consecutive 10 s epochs: (n_ep, n_freq, n_ch)."""
    n = int(T_EPOCH * fs)
    n_ep = data_V.shape[0] // n
    psds = []
    for i_ep in range(n_ep):
        freqs, psd = welch(data_V[i_ep*n:(i_ep+1)*n, :], fs=fs, axis=0,
                           nperseg=NPERSEG)
        psds.append(psd)
    return freqs, np.array(psds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--edf-dir', required=True)
    p.add_argument('--out-dir', required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    edfs = sorted(f for f in os.listdir(args.edf_dir) if f.endswith('.edf'))
    subjects = sorted({re.match(r'(S\d+)_', f).group(1) for f in edfs})
    ch_names_ref = None

    # spectra/features, ordered by stage, then subject (as in the original)
    feats, psds_all, y, subj_idx, ep_idx = [], [], [], [], []
    psd_total = []
    raw_by_subject = {s: {} for s in subjects}

    for i_stage, stage in enumerate(SLEEP_STAGES):
        for i_subj, subj in enumerate(subjects):
            f = os.path.join(args.edf_dir, f"{subj}_{stage}.edf")
            dig, cal, offsets, unit, fs, ch_names = read_edf(f)
            if ch_names_ref is None:
                ch_names_ref, fs_ref = ch_names, fs
            assert ch_names == ch_names_ref and fs == fs_ref, f
            raw_by_subject[subj][stage] = (dig, cal, offsets, unit)

            data_V = (dig * cal + offsets) * unit
            freqs, psd = epoch_spectra(data_V, fs)
            fidx = lambda fr: np.argmin(np.abs(fr - freqs))
            for i_ep, ps in enumerate(psd):
                tot = ps.sum()
                ps_n = ps / tot
                feats.append([[ps_n[fidx(lo):fidx(hi), i_ch].sum()
                               for i_ch in range(len(ch_names))]
                              for lo, hi in FREQ_BANDS.values()])
                psd_total.append(tot)
                y.append(i_stage)
                subj_idx.append(i_subj)
                ep_idx.append(i_ep)
            psds_all.append(psd)
            print(f"{subj}_{stage}: {psd.shape[0]:d} epochs")

    ch_names = np.array(ch_names_ref)
    sleep_stages = np.array(SLEEP_STAGES)
    subjects = np.array(subjects)
    bands = np.array(list(FREQ_BANDS))
    band_edges = np.array(list(FREQ_BANDS.values()))

    # metadata incl. standard 10-05 electrode positions (head frame, metres)
    montage = mne.channels.make_standard_montage('standard_1005')
    pos = montage.get_positions()['ch_pos']
    ch_pos = np.array([pos[ch] for ch in ch_names])
    np.savez(os.path.join(args.out_dir, 'meta.npz'),
             ch_names=ch_names, ch_pos=ch_pos, sfreq=fs_ref,
             sleep_stages=sleep_stages, subjects=subjects,
             montage='standard_1005', coord_frame='head', pos_unit='m',
             data_unit='V')

    # raw EEG per subject
    for subj in subjects:
        d = {}
        for stage in SLEEP_STAGES:
            dig, cal, offsets, unit = raw_by_subject[subj][stage]
            d[stage] = dig
            d[f'{stage}_cal'] = cal
            d[f'{stage}_offsets'] = offsets
            d[f'{stage}_unit'] = unit
        np.savez(os.path.join(args.out_dir, f'{subj}.npz'),
                 sfreq=fs_ref, ch_names=ch_names, subject=subj, **d)

    # band-power features: X_bands (n_ep, n_bands, n_ch)
    common = dict(y=np.array(y), subject=np.array(subj_idx),
                  epoch=np.array(ep_idx), subjects=subjects,
                  sleep_stages=sleep_stages, ch_names=ch_names)
    np.savez(os.path.join(args.out_dir, 'spectral_features.npz'),
             X_bands=np.array(feats), bands=bands, band_edges=band_edges,
             t_epoch=T_EPOCH, nperseg=NPERSEG, **common)

    # per-epoch spectra (0-F_SPECTRA_MAX Hz) and total power (all freqs)
    i_max = np.searchsorted(freqs, F_SPECTRA_MAX, side='right')
    np.savez(os.path.join(args.out_dir, 'spectra.npz'),
             psd=np.concatenate(psds_all)[:, :i_max, :].astype(np.float32),
             freqs=freqs[:i_max], psd_total=np.array(psd_total),
             t_epoch=T_EPOCH, nperseg=NPERSEG, psd_unit='V^2/Hz', **common)


if __name__ == '__main__':
    main()
