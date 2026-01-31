#!/usr/bin/env python3
"""
Null Models for Curvature Testing.

Implements null models for both neural and AI pipelines:

Neural null models:
  - trial_permutation: Shuffle spike trains across neurons within trials
  - bin_shuffle: Permute time bins within each neuron's spike train

AI null models:
  - shuffle_windows: Shuffle the temporal order of activation windows
  - permute_tokens: Permute tokens within each window
  - shuffle_features: Independently shuffle each feature column

All null models preserve specific statistical properties while destroying
the structure that generates curvature, enabling controlled hypothesis
testing.
"""
import numpy as np
from typing import List


# ═══════════════════════════════════════════════════════════════════════
# NEURAL NULL MODELS
# ═══════════════════════════════════════════════════════════════════════

def trial_permutation(
    st_list: List[np.ndarray],
    B: int,
    bin_size: float,
    Ttr: int,
    seed: int,
) -> List[np.ndarray]:
    """Shuffle spike trains across neurons within each trial.

    Preserves: per-trial firing rates, overall rate distributions.
    Destroys: neuron-specific temporal patterns, cross-neuron correlations.

    Args:
        st_list: List of spike time arrays, one per neuron.
        B: Number of bins per trial.
        bin_size: Bin duration in seconds.
        Ttr: Number of trials.
        seed: Random seed.

    Returns:
        Null spike time arrays with shuffled trial assignments.
    """
    rng = np.random.default_rng(seed)
    n = len(st_list)
    trial_dur = B * bin_size

    # Segment spike times into trials
    trials = []
    for st in st_list:
        parts = []
        for tt in range(Ttr):
            s = tt * trial_dur
            e = (tt + 1) * trial_dur
            parts.append(st[(st >= s) & (st < e)] - s)
        trials.append(parts)

    # Reassemble with shuffled neuron assignments per trial
    out = []
    for i in range(n):
        new = []
        for tt in range(Ttr):
            src = rng.integers(0, n)
            new.append(trials[src][tt] + tt * trial_dur)
        combined = np.concatenate(new) if new else np.array([])
        out.append(np.sort(combined))
    return out


def bin_shuffle(
    st_list: List[np.ndarray],
    edges: np.ndarray,
    Tb: int,
    seed: int,
) -> List[np.ndarray]:
    """Shuffle time-bin assignments of spikes.

    Preserves: total spike count per neuron, spike count distribution.
    Destroys: temporal structure, within-neuron autocorrelation.

    Args:
        st_list: List of spike time arrays.
        edges: Bin edges.
        Tb: Total number of bins.
        seed: Random seed.

    Returns:
        Null spike time arrays with shuffled bin placements.
    """
    rng = np.random.default_rng(seed)
    out = []
    for st in st_list:
        if len(st) == 0:
            out.append(st)
            continue
        idx = np.searchsorted(edges[:-1], st, side='right') - 1
        idx = np.clip(idx, 0, Tb - 1)
        counts = np.bincount(idx, minlength=Tb)
        shuffled_counts = rng.permutation(counts)
        new_st = []
        for bin_idx, count in enumerate(shuffled_counts):
            if count > 0:
                bin_start = edges[bin_idx]
                bin_end = edges[bin_idx + 1]
                new_st.extend(rng.uniform(bin_start, bin_end, size=int(count)))
        out.append(np.sort(np.array(new_st)))
    return out


# ═══════════════════════════════════════════════════════════════════════
# AI NULL MODELS
# ═══════════════════════════════════════════════════════════════════════

def shuffle_windows(
    activation_windows: List[np.ndarray],
    seed: int = 99,
) -> List[np.ndarray]:
    """Shuffle temporal order of activation windows.

    Preserves: each window's internal structure and covariance.
    Destroys: temporal ordering, inter-window dynamics.

    Args:
        activation_windows: List of (seq_len, feat_dim) arrays.
        seed: Random seed.

    Returns:
        Reordered list of activation windows.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(activation_windows))
    return [activation_windows[i] for i in indices]


def permute_tokens(
    activation_windows: List[np.ndarray],
    seed: int = 99,
) -> List[np.ndarray]:
    """Permute tokens within each window.

    Preserves: marginal feature distributions per window.
    Destroys: within-window sequential structure.

    Args:
        activation_windows: List of (seq_len, feat_dim) arrays.
        seed: Random seed.

    Returns:
        Windows with permuted token order.
    """
    rng = np.random.default_rng(seed)
    nulled = []
    for X in activation_windows:
        perm = rng.permutation(X.shape[0])
        nulled.append(X[perm])
    return nulled


def shuffle_features(
    activation_windows: List[np.ndarray],
    seed: int = 99,
) -> List[np.ndarray]:
    """Independently shuffle each feature column.

    Preserves: marginal distribution of each feature.
    Destroys: all cross-feature correlations (covariance → diagonal).

    Args:
        activation_windows: List of (seq_len, feat_dim) arrays.
        seed: Random seed.

    Returns:
        Windows with independently shuffled feature columns.
    """
    rng = np.random.default_rng(seed)
    nulled = []
    for X in activation_windows:
        X_null = X.copy()
        for col in range(X.shape[1]):
            X_null[:, col] = rng.permutation(X_null[:, col])
        nulled.append(X_null)
    return nulled
