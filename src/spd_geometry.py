#!/usr/bin/env python3
"""
SPD Manifold Geometry: Core functions for curvature estimation.

Implements the Log-Euclidean framework on the manifold of Symmetric
Positive Definite (SPD) matrices, plus triangle-excess curvature κ
estimation with bootstrap confidence intervals.

The key insight: information-processing structure lives in the
*interaction geometry* (covariance manifold), not in marginal statistics.

References:
    Arsigny et al. (2007). Geometric means in a novel vector space
    structure on symmetric positive-definite matrices. SIAM J. Matrix
    Anal. Appl.
"""
import numpy as np
from numpy.linalg import eigh
from typing import List, Tuple


def mat_log(C: np.ndarray) -> np.ndarray:
    """Matrix logarithm via eigendecomposition for SPD matrices.

    Args:
        C: Symmetric positive definite matrix.

    Returns:
        log(C) in the matrix sense.
    """
    w, V = eigh(C)
    w = np.clip(w, 1e-10, None)
    return V @ np.diag(np.log(w)) @ V.T


def log_euclidean_distance(L1: np.ndarray, L2: np.ndarray) -> float:
    """Log-Euclidean distance: ||log(C1) - log(C2)||_F.

    Args:
        L1, L2: Matrix logarithms of SPD matrices.

    Returns:
        Frobenius norm of their difference.
    """
    D = L1 - L2
    return float(np.sqrt(np.sum(D * D)))


def mat_sqrt_inv(C: np.ndarray) -> np.ndarray:
    """Matrix C^{-1/2} via eigendecomposition for SPD matrices."""
    w, V = eigh(C)
    w = np.clip(w, 1e-10, None)
    return V @ np.diag(1.0 / np.sqrt(w)) @ V.T


def airm_distance(C1: np.ndarray, C2: np.ndarray) -> float:
    """Affine-Invariant Riemannian Metric (AIRM) distance.

    d_AIRM(C1, C2) = ||log(C1^{-1/2} C2 C1^{-1/2})||_F

    This is the geodesic distance on the SPD manifold under
    the Fisher-Rao / affine-invariant metric.

    Args:
        C1, C2: Symmetric positive definite matrices.

    Returns:
        AIRM distance between C1 and C2.
    """
    C1_isqrt = mat_sqrt_inv(C1)
    M = C1_isqrt @ C2 @ C1_isqrt
    w = eigh(M)[0]
    w = np.clip(w, 1e-10, None)
    return float(np.sqrt(np.sum(np.log(w) ** 2)))


def distance_matrix(log_covs: List[np.ndarray], metric: str = "log_euclidean",
                    covs: List[np.ndarray] = None) -> np.ndarray:
    """Pairwise distance matrix on the SPD manifold.

    Args:
        log_covs: List of matrix logarithms (used for log_euclidean).
        metric: "log_euclidean" or "airm".
        covs: List of raw SPD matrices (required for airm).

    Returns:
        Symmetric distance matrix D where D[i,j] = d(C_i, C_j).
    """
    if metric == "airm":
        if covs is None:
            raise ValueError("covs required for AIRM metric")
        n = len(covs)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = airm_distance(covs[i], covs[j])
                D[i, j] = D[j, i] = d
        return D
    n = len(log_covs)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = log_euclidean_distance(log_covs[i], log_covs[j])
            D[i, j] = D[j, i] = d
    return D


def tri_kappa(D: np.ndarray, ns: int = 1500, seed: int = 0) -> float:
    """Triangle-excess curvature estimator.

    Samples random triangles from the distance matrix and computes
    the median excess: κ = median((a + b - c) / 2) for sorted sides
    a ≤ b ≤ c of sampled triangles.

    Positive κ indicates positive (hyperbolic-like) curvature in the
    triangle-excess sense.

    Args:
        D: Pairwise distance matrix (median-normalized internally).
        ns: Number of triangle samples.
        seed: Random seed.

    Returns:
        Median triangle excess κ.
    """
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    if n < 3:
        return float('nan')
    med = np.median(D[np.triu_indices(n, 1)])
    if med > 1e-10:
        D = D / med
    ex = []
    for _ in range(ns):
        i, j, k = rng.choice(n, 3, replace=False)
        a, b, c = sorted([D[i, j], D[j, k], D[i, k]])
        if a + b > c:
            ex.append((a + b - c) / 2)
    return float(np.median(ex)) if ex else float('nan')


def tri_kappa_bootstrap(
    D: np.ndarray,
    ns: int = 1500,
    B: int = 500,
    seed: int = 0,
) -> Tuple[float, Tuple[float, float]]:
    """Bootstrap confidence interval for triangle-excess κ.

    Args:
        D: Pairwise distance matrix.
        ns: Triangle samples per bootstrap iteration.
        B: Number of bootstrap iterations.
        seed: Random seed.

    Returns:
        (median_kappa, (ci_lower, ci_upper)) at 95% level.
    """
    n = D.shape[0]
    if n < 3:
        return float('nan'), (float('nan'), float('nan'))
    med = np.median(D[np.triu_indices(n, 1)])
    if med > 1e-10:
        D = D / med
    ests = []
    for b in range(B):
        rr = np.random.default_rng(seed + b)
        ex = []
        for _ in range(ns):
            i, j, k = rr.choice(n, 3, replace=False)
            a, b_, c = sorted([D[i, j], D[j, k], D[i, k]])
            if a + b_ > c:
                ex.append((a + b_ - c) / 2)
        ests.append(float(np.median(ex)) if ex else float('nan'))
    ests = np.array([e for e in ests if np.isfinite(e)])
    if len(ests) == 0:
        return float('nan'), (float('nan'), float('nan'))
    return float(np.median(ests)), (
        float(np.percentile(ests, 2.5)),
        float(np.percentile(ests, 97.5)),
    )


def windowed_covariances(
    X: np.ndarray,
    window_size: int,
    hop: int,
    eps: float = 1e-6,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Compute windowed covariance matrices and their matrix logarithms.

    Args:
        X: Data matrix (time_bins x features).
        window_size: Number of time bins per window.
        hop: Hop size in bins.
        eps: Regularization for SPD guarantee.

    Returns:
        (covs, log_covs): Lists of covariance matrices and their logs.
    """
    T, d = X.shape
    covs = []
    log_covs = []
    for start in range(0, T - window_size + 1, hop):
        seg = X[start:start + window_size]
        if len(seg) > 2:
            C = np.cov(seg.T) + eps * np.eye(d)
            covs.append(C)
            log_covs.append(mat_log(C))
    return covs, log_covs


def state_equation(h: float, n: int = 2) -> float:
    """Theoretical curvature from the geometric state equation.

    κ = (h * ln(2) / (n - 1))²

    Args:
        h: Entropy rate in bits/symbol.
        n: Embedding dimension (default 2).

    Returns:
        Predicted curvature κ.
    """
    return (h * np.log(2) / (n - 1)) ** 2
