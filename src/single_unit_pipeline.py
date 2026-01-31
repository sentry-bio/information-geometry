#!/usr/bin/env python3
"""
Single-Unit (Neuropixels) Analysis Pipeline.

Processes Steinmetz et al. Neuropixels recordings:
  spikes → windowed neuron×neuron covariance (SPD) → Log-Euclidean → κ

Frozen parameters from calibration_manifest.json:
  - bin_s: 0.3s
  - window_s: [4.8, 6.0]s
  - hop_fraction: 0.5
  - neuron_caps: [180, 240]
  - fano_max: 2.0
  - nsamples: 1500, bootstrap: 500
  - nulls: trial_permutation, bin_shuffle
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from .spd_geometry import (
    mat_log,
    log_euclidean_distance,
    distance_matrix,
    tri_kappa_bootstrap,
)

# Frozen recipe parameters
BIN_SECS = [0.3]
WINDOW_SECS = [4.8, 6.0]
HOP_FRACTION = 0.5
NEURON_CAPS = [180, 240]
FANO_MAX = 2.0
TOP_NEURONS = 240
NSAMPLES = 1500
BOOTSTRAP = 500
MAX_WINDOWS = 120
NULL_KINDS = ["trial_perm", "bin_shuffle"]


def load_session(path: Path):
    """Load a Steinmetz session from .npy/.npz format."""
    arr = np.load(path, allow_pickle=True)
    if path.suffix == ".npy":
        return arr['dat'][0]
    elif path.suffix == ".npz":
        dat = arr['dat']
        if hasattr(dat, '__getitem__') and len(dat) > 0:
            return dat[0]
    raise RuntimeError(f"Cannot load session from {path}")


def select_stable_neurons(
    spks: np.ndarray,
    cap: int,
    fano_max: float,
) -> np.ndarray:
    """Select stable, high-firing neurons.

    Criteria: Fano factor < fano_max, active in both halves of session.
    Returns indices of top neurons by total spike count.
    """
    tot_spikes = spks.sum(axis=(1, 2))
    counts_flat = spks.reshape(spks.shape[0], -1)
    means = counts_flat.mean(axis=1)
    vars_ = counts_flat.var(axis=1)
    fano = np.where(means > 0, vars_ / np.maximum(means, 1e-12), np.inf)

    T = counts_flat.shape[1]
    half = T // 2
    has_both = (
        (counts_flat[:, :half].sum(axis=1) > 0)
        & (counts_flat[:, half:].sum(axis=1) > 0)
    )
    idx = np.where((fano < fano_max) & has_both)[0]
    if len(idx) == 0:
        idx = np.arange(spks.shape[0])
    idx_sorted = idx[np.argsort(tot_spikes[idx])]
    return idx_sorted[-min(cap, len(idx_sorted)):]


def build_spike_times(
    spks: np.ndarray,
    B: int,
    bin_size: float,
    Ttr: int,
) -> List[np.ndarray]:
    """Convert binned spike counts to spike times."""
    N = spks.shape[0]
    st_all = []
    for n in range(N):
        st = []
        for t in range(Ttr):
            base = t * B * bin_size
            cnt = spks[n, t]
            for b, c in enumerate(cnt):
                c = int(c)
                if c > 0:
                    for _ in range(c):
                        st.append(base + (b + 0.5) * bin_size / max(1, c))
        st_all.append(np.array(st))
    return st_all


def counts_from_spike_times(
    st_list: List[np.ndarray],
    edges: np.ndarray,
    Tb: int,
    N: int,
) -> np.ndarray:
    """Convert spike times to count matrix (time_bins x neurons)."""
    Xc = np.zeros((Tb, N))
    for nn, st in enumerate(st_list):
        if len(st) == 0:
            continue
        idx = np.searchsorted(edges[:-1], st, side='right') - 1
        idx = np.clip(idx, 0, Tb - 1)
        np.add.at(Xc[:, nn], idx, 1)
    return Xc


def run_session(session_path: Path) -> Dict:
    """Full analysis pipeline for one Steinmetz session.

    Returns dict with κ estimates for real data and null models.
    """
    dat = load_session(session_path)
    spks_full = dat['spks']
    N_all, Ttr, B = spks_full.shape
    bin_size = 0.01

    results = []
    for cap in NEURON_CAPS:
        keep = select_stable_neurons(spks_full, cap=min(cap, TOP_NEURONS), fano_max=FANO_MAX)
        spks = spks_full[keep]
        N = spks.shape[0]
        st_all = build_spike_times(spks, B, bin_size, Ttr)

        for bin_s in BIN_SECS:
            edges = np.arange(0, Ttr * B * bin_size + bin_s, bin_s)
            Tb = len(edges) - 1
            X_full = counts_from_spike_times(st_all, edges, Tb, N)
            W = min(MAX_WINDOWS, Tb)

            for window_s in WINDOW_SECS:
                win_bins = max(3, int(round(window_s / bin_s)))
                covs, log_covs = [], []
                for i in range(0, W, max(1, win_bins // 2)):
                    seg = X_full[i:min(W, i + win_bins)]
                    if len(seg) > 2:
                        C = np.cov(seg.T) + 1e-6 * np.eye(N)
                        covs.append(C)
                        log_covs.append(mat_log(C))

                if len(log_covs) < 6:
                    continue

                D = distance_matrix(log_covs)
                k_med, k_ci = tri_kappa_bootstrap(D, ns=NSAMPLES, B=BOOTSTRAP, seed=1)

                entry = {
                    'bin_s': float(bin_s),
                    'window_s': float(window_s),
                    'n_neurons': int(N),
                    'n_windows': len(log_covs),
                    'k_real': float(k_med),
                    'k_real_ci': [float(k_ci[0]), float(k_ci[1])],
                }

                # Null models
                for nk in NULL_KINDS:
                    vals = []
                    for it in range(2):
                        seed = 300 + it
                        if nk == 'trial_perm':
                            from .null_models import trial_permutation
                            stn = trial_permutation(st_all, B, bin_size, Ttr, seed)
                        else:
                            from .null_models import bin_shuffle
                            stn = bin_shuffle(st_all, edges, Tb, seed)

                        Xn = counts_from_spike_times(stn, edges, Tb, N)
                        null_covs, null_logs = [], []
                        for i in range(0, W, max(1, win_bins // 2)):
                            seg = Xn[i:min(W, i + win_bins)]
                            if len(seg) > 2:
                                C = np.cov(seg.T) + 1e-6 * np.eye(N)
                                null_covs.append(C)
                                null_logs.append(mat_log(C))
                        if len(null_logs) < 6:
                            continue
                        Dn = distance_matrix(null_logs)
                        k_n, ci_n = tri_kappa_bootstrap(Dn, ns=NSAMPLES, B=BOOTSTRAP // 2, seed=seed)
                        vals.append((k_n, ci_n))

                    vals = [v for v in vals if np.isfinite(v[0])]
                    if vals:
                        k_nulls = np.array([v[0] for v in vals])
                        entry[f'k_null_mean_{nk}'] = float(k_nulls.mean())
                        entry[f'delta_{nk}'] = float(k_med - k_nulls.mean())

                results.append(entry)

    return {
        'session_file': str(session_path),
        'results': results,
    }


if __name__ == '__main__':
    import json
    import sys
    from pathlib import Path

    data_dir = Path(__file__).resolve().parent.parent / 'data'
    out_path = data_dir / 'su_cohort.csv'

    # Look for Steinmetz .npy files
    raw_dir = data_dir / 'steinmetz'
    if not raw_dir.exists():
        print(f"No data at {raw_dir}. Place Steinmetz .npy session files there.")
        sys.exit(1)

    sessions = sorted(raw_dir.glob('*.npy'))
    all_results = []
    for sp in sessions:
        print(f"Processing {sp.name}...")
        res = run_session(sp)
        all_results.append(res)

    # Flatten and write CSV
    import pandas as pd
    rows = []
    for r in all_results:
        for entry in r.get('results', []):
            entry['session_file'] = r['session_file']
            rows.append(entry)
    if rows:
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"Wrote {len(rows)} rows to {out_path}")
    else:
        print("No results produced.")
