#!/usr/bin/env python3
"""
fMRI Analysis Pipeline.

Processes ABIDE resting-state fMRI data:
  parcellated timeseries → windowed ROI×ROI covariance (SPD) →
  Log-Euclidean distance → triangle excess κ

Frozen parameters from calibration_manifest.json:
  - parcellation: Schaefer400
  - windows_s: [30.0, 15.0]
  - distance: AIRM (primary), log_euclidean (ablation)
  - nsamples: 3000
  - nulls: phase_randomization
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List

from .spd_geometry import (
    mat_log,
    distance_matrix,
    tri_kappa_bootstrap,
    windowed_covariances,
)


# Frozen parameters
WINDOWS_S = [30.0, 15.0]
NSAMPLES = 3000
TR = 2.0  # seconds, typical ABIDE TR


def parcellate_timeseries(
    bold_data: np.ndarray,
    atlas_labels: np.ndarray,
    n_rois: int = 400,
) -> np.ndarray:
    """Extract mean timeseries per ROI from BOLD data.

    Args:
        bold_data: 4D BOLD volume (x, y, z, time).
        atlas_labels: 3D integer label volume.
        n_rois: Number of ROIs in the atlas.

    Returns:
        Array of shape (n_timepoints, n_rois).
    """
    n_time = bold_data.shape[-1]
    ts = np.zeros((n_time, n_rois))
    for roi in range(1, n_rois + 1):
        mask = atlas_labels == roi
        if mask.sum() > 0:
            ts[:, roi - 1] = bold_data[mask].mean(axis=0)
    return ts


def run_subject(
    timeseries: np.ndarray,
    tr: float = TR,
    window_s: float = 30.0,
    eps: float = 1e-6,
) -> Dict:
    """Analyze one fMRI subject.

    Args:
        timeseries: (n_timepoints, n_rois) parcellated BOLD timeseries.
        tr: Repetition time in seconds.
        window_s: Window duration in seconds.
        eps: Covariance regularization.

    Returns:
        Dict with κ estimate and metadata.
    """
    n_time, n_rois = timeseries.shape
    win_bins = max(3, int(round(window_s / tr)))
    hop = max(1, win_bins // 2)

    covs, log_covs = windowed_covariances(timeseries, win_bins, hop, eps)

    if len(log_covs) < 6:
        return {'error': 'Too few windows', 'n_windows': len(log_covs)}

    D = distance_matrix(log_covs)
    k_med, k_ci = tri_kappa_bootstrap(D, ns=NSAMPLES, B=500, seed=42)

    return {
        'kappa': float(k_med),
        'kappa_ci': [float(k_ci[0]), float(k_ci[1])],
        'n_windows': len(log_covs),
        'n_rois': n_rois,
        'window_s': window_s,
    }


def run_cohort(
    subjects: List[np.ndarray],
    subject_ids: List[str],
    tr: float = TR,
) -> Dict:
    """Run analysis on full cohort.

    Args:
        subjects: List of (n_time, n_rois) timeseries arrays.
        subject_ids: List of subject identifiers.
        tr: Repetition time.

    Returns:
        Cohort summary with per-subject κ and aggregates.
    """
    results = []
    for sid, ts in zip(subject_ids, subjects):
        for ws in WINDOWS_S:
            res = run_subject(ts, tr=tr, window_s=ws)
            res['subject_id'] = sid
            results.append(res)

    valid = [r for r in results if 'error' not in r]
    kappas = [r['kappa'] for r in valid]

    return {
        'subjects': results,
        'n_valid': len(valid),
        'kappa_mean': float(np.mean(kappas)) if kappas else None,
        'kappa_std': float(np.std(kappas)) if kappas else None,
    }


if __name__ == '__main__':
    import json
    import sys
    from pathlib import Path

    data_dir = Path(__file__).resolve().parent.parent / 'data'
    out_path = data_dir / 'fmri_cohort.csv'

    # Look for parcellated timeseries
    ts_dir = data_dir / 'fmri_parcellated'
    if not ts_dir.exists():
        print(f"No data at {ts_dir}. Place AAL-parcellated .npy timeseries there.")
        sys.exit(1)

    cohort = run_cohort(ts_dir)
    import pandas as pd
    rows = [s for s in cohort['subjects'] if 'error' not in s]
    if rows:
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"Wrote {len(rows)} subjects to {out_path}")
        print(f"kappa_mean={cohort['kappa_mean']:.4f} +/- {cohort['kappa_std']:.4f}")
    else:
        print("No valid results.")
