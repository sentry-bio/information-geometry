#!/usr/bin/env python3
"""
EEG Sensor Covariance Pipeline: Eyes-Open vs Eyes-Closed κ analysis.

Ingests EEGBCI (PhysioNet) motor imagery data, computes windowed sensor
covariance matrices, estimates triangle-excess κ using both Log-Euclidean
and AIRM metrics, and tests the EO > EC prediction from the field equation.

Requires: mne>=1.5 (optional; falls back to pre-computed summary JSON)
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from .spd_geometry import (
    mat_log, windowed_covariances, distance_matrix,
    tri_kappa_bootstrap, airm_distance,
)


# EEGBCI run mapping: runs 1=EO baseline, 2=EC baseline
EEGBCI_RUNS = {"EO": [1], "EC": [2]}
SUBJECTS = list(range(1, 20))  # S001-S019 as in paper
WINDOW_S = 2.0
HOP_S = 1.0
BAND = (8.0, 13.0)  # alpha band


def load_eegbci_subject(subject_id: int, runs: List[int]) -> Tuple[np.ndarray, float]:
    """Load EEGBCI data for one subject/condition using MNE.

    Args:
        subject_id: PhysioNet subject number (1-109).
        runs: List of run numbers to load.

    Returns:
        (data, sfreq): raw data array (n_channels x n_times) and sampling freq.
    """
    import mne
    from mne.datasets import eegbci
    raw_fnames = eegbci.load_data(subject_id, runs, update_path=False)
    raw = mne.io.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in raw_fnames])
    raw.filter(BAND[0], BAND[1], fir_design='firwin', verbose=False)
    mne.datasets.eegbci.standardize(raw)
    raw.set_montage('standard_1005', on_missing='ignore')
    return raw.get_data(), raw.info['sfreq']


def run_subject_condition(subject_id: int, condition: str) -> Dict:
    """Run κ estimation for one subject in one condition (EO or EC).

    Args:
        subject_id: PhysioNet subject number.
        condition: "EO" or "EC".

    Returns:
        Dict with kappa_loge, kappa_airm, n_windows, etc.
    """
    runs = EEGBCI_RUNS[condition]
    data, sfreq = load_eegbci_subject(subject_id, runs)

    # Transpose to (time x channels) for windowed_covariances
    X = data.T
    window_size = int(WINDOW_S * sfreq)
    hop = int(HOP_S * sfreq)

    covs, log_covs = windowed_covariances(X, window_size, hop)

    if len(covs) < 6:
        return {"error": f"Too few windows ({len(covs)})", "n_windows": len(covs)}

    # Log-Euclidean κ
    D_le = distance_matrix(log_covs, metric="log_euclidean")
    k_le, ci_le = tri_kappa_bootstrap(D_le)

    # AIRM κ
    D_airm = distance_matrix(log_covs, metric="airm", covs=covs)
    k_airm, ci_airm = tri_kappa_bootstrap(D_airm)

    return {
        "subject": f"S{subject_id:03d}",
        "state": condition,
        "n_windows": len(covs),
        "n_sensors": X.shape[1],
        "kappa_loge": float(k_le),
        "ci_loge": list(ci_le),
        "kappa_airm": float(k_airm),
        "ci_airm": list(ci_airm),
        "metric": "sensor_covariance",
        "band": "alpha",
    }


def run_cohort(subjects: List[int] = None) -> Dict:
    """Run full EO/EC cohort analysis.

    Args:
        subjects: List of subject IDs. Defaults to S001-S019.

    Returns:
        Dict with per-subject results and cohort summary.
    """
    if subjects is None:
        subjects = SUBJECTS

    all_results = []
    for sid in subjects:
        for cond in ["EO", "EC"]:
            print(f"  Subject S{sid:03d} {cond}...", end=" ", flush=True)
            try:
                res = run_subject_condition(sid, cond)
                print(f"κ_airm={res.get('kappa_airm', 'ERR'):.3f}")
            except Exception as e:
                res = {"subject": f"S{sid:03d}", "state": cond, "error": str(e)}
                print(f"ERROR: {e}")
            all_results.append(res)

    # Compute cohort summary
    eo = [r for r in all_results if r.get("state") == "EO" and "error" not in r]
    ec = [r for r in all_results if r.get("state") == "EC" and "error" not in r]

    eo_kappas = [r["kappa_airm"] for r in eo]
    ec_kappas = [r["kappa_airm"] for r in ec]

    n_eo_gt_ec = 0
    for s in subjects:
        s_eo = [r for r in eo if r["subject"] == f"S{s:03d}"]
        s_ec = [r for r in ec if r["subject"] == f"S{s:03d}"]
        if s_eo and s_ec:
            if s_eo[0]["kappa_airm"] > s_ec[0]["kappa_airm"]:
                n_eo_gt_ec += 1

    return {
        "subjects": all_results,
        "n_subjects": len(subjects),
        "eo_kappa_mean": float(np.mean(eo_kappas)) if eo_kappas else None,
        "ec_kappa_mean": float(np.mean(ec_kappas)) if ec_kappas else None,
        "delta_kappa": float(np.mean(eo_kappas) - np.mean(ec_kappas)) if eo_kappas and ec_kappas else None,
        "n_eo_gt_ec": n_eo_gt_ec,
    }


if __name__ == '__main__':
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    out_path = data_dir / 'eeg_sensor_cov_summary.json'

    try:
        import mne  # noqa: F401
        print("MNE found. Running full EEG pipeline...")
        results = run_cohort()
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote results to {out_path}")
        print(f"EO mean κ = {results['eo_kappa_mean']:.4f}")
        print(f"EC mean κ = {results['ec_kappa_mean']:.4f}")
        print(f"Δκ = {results['delta_kappa']:.4f}")
        print(f"EO > EC in {results['n_eo_gt_ec']}/{results['n_subjects']} subjects")
    except ImportError:
        print("MNE not installed. To run the EEG pipeline:")
        print("  pip install mne")
        print(f"Pre-computed results available at: {out_path}")
