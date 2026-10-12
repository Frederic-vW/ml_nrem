Practical: Machine Learning NREM sleep data from EEG spectral features

Run in your browser (no installation, JupyterLite):  
[![JupyterLite](https://jupyterlite.rtfd.io/en/latest/_static/badge.svg)](https://frederic-vw.github.io/ml_nrem/notebooks/index.html?path=ml_nrem.ipynb)

## Branch layout (`jupyterlite`)
- `content/` – everything students see: `ml_nrem.ipynb`, `ml_nrem.py`, `img/`, `data/`
- `content/data/`
  - `meta.npz` – channel names, electrode positions (MNE `standard_1005`, head frame, metres), sampling rate, subjects, sleep stages
  - `S00.npz` … `S14.npz` – raw EEG per subject, one `int16` array `(n_samples, 30)` per stage (`W`, `N1`, `N2`, `N3`) with per-channel `<stage>_cal`, `<stage>_offsets`, `<stage>_unit`; volts = `(raw * cal + offsets) * unit` (lossless w.r.t. the original EDF files)
  - `spectral_features.npz` – band powers per 10 s epoch, `X_bands (n_epochs, 4 bands, 30 ch)`, labels `y`, `subject`, `epoch`
  - `spectra.npz` – Welch spectra per 10 s epoch, `psd (n_epochs, n_freqs, 30)` in V²/Hz for 0–45 Hz, `psd_total` (sum over all frequencies, used for normalization)
- `tools/edf_to_npz.py` – regenerates `content/data/` from the EDF files (requires MNE; not used at runtime)
- ...

## Build locally
```
pip install -r requirements.txt
jupyter lite build --contents content --output-dir dist
jupyter lite serve --output-dir dist
```
The GitHub Action in `.github/workflows/deploy.yml` builds and deploys the site to GitHub Pages on every push to this branch.
