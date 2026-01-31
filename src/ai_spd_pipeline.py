#!/usr/bin/env python3
"""
AI (GPT-2) SPD Covariance Pipeline.

Resolves the measurement mismatch between neural and AI pipelines:
  - Neural: (time_bins × neurons) → neuron×neuron covariance (SPD)
  - AI:     (tokens × features)   → feature×feature covariance (SPD)

The critical insight is that hierarchical structure lives in the
*interaction geometry* (covariance manifold), not in individual
embeddings. Using cosine distance on last-token vectors misses this.

Frozen parameters (matching neural pipeline):
  - pca_dim: 64
  - meta_window: 12
  - nsamples: 1500, bootstrap: 500
  - cov_eps: 1e-6
"""
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .spd_geometry import (
    mat_log,
    log_euclidean_distance,
    distance_matrix,
    tri_kappa,
    tri_kappa_bootstrap,
)

# Frozen parameters
PCA_DIM = 64
META_WINDOW = 12
META_HOP = 1
NSAMPLES = 1500
BOOTSTRAP = 500
COV_EPS = 1e-6
MIN_COVS = 6
SEED = 42

# GPT-2 extraction parameters
TEXT_WINDOW_SIZE = 128
TEXT_STRIDE = 64
NUM_WINDOWS = 400


def extract_gpt2_activations(
    num_windows: int = NUM_WINDOWS,
    window_size: int = TEXT_WINDOW_SIZE,
    stride: int = TEXT_STRIDE,
    layer_idx: int = -1,
    pca_dim: int = PCA_DIM,
    seed: int = SEED,
) -> Tuple[List[np.ndarray], Dict]:
    """Extract full activation matrices from GPT-2.

    Returns (seq_len, pca_dim) arrays -- one per text window.
    This is the critical difference: full activation matrices,
    not just last-token embeddings.

    Requires: torch, transformers, sklearn.

    Args:
        num_windows: Number of text windows to extract.
        window_size: Tokens per window.
        stride: Token stride between windows.
        layer_idx: Which transformer layer (-1 = last).
        pca_dim: PCA reduction target.
        seed: Random seed.

    Returns:
        (activations, metadata) where activations is a list of
        (seq_len, pca_dim) arrays.
    """
    import torch
    from transformers import GPT2Model, GPT2Tokenizer
    from sklearn.decomposition import PCA

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True)
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    # Diverse text corpus
    text_corpus = (
        "The rapid advancement of artificial intelligence has transformed "
        "our understanding of computation and cognition. Large language "
        "models demonstrate emergent capabilities that suggest complex "
        "internal representations and dynamics. These systems process "
        "information through high-dimensional activation spaces that "
        "exhibit geometric structure. The study of consciousness has long "
        "sought to understand how subjective experience emerges from "
        "neural computation. Recent work in computational neuroscience "
        "suggests that geometric properties of neural state spaces may be "
        "fundamental. Hyperbolic geometry appears in biological systems "
        "from evolution to neural networks, suggesting universal "
        "principles. The relationship between information processing, "
        "geometric structure, and conscious experience remains an open "
        "question. Machine learning models provide a unique window into "
        "the dynamics of complex information processing systems. "
        "Understanding the geometric principles underlying these dynamics "
        "could illuminate fundamental aspects of intelligence. The "
        "emergence of coherent behavior from high-dimensional neural "
        "dynamics represents a central challenge in neuroscience. "
    ) * 50

    all_tokens = tokenizer(text_corpus, add_special_tokens=False)["input_ids"]
    tokens = torch.tensor(all_tokens, dtype=torch.long)

    all_activations = []
    metadata = {"windows_extracted": 0, "failed": 0}

    with torch.no_grad():
        for start in range(0, len(tokens) - window_size, stride):
            if len(all_activations) >= num_windows:
                break
            try:
                window_tokens = tokens[start:start + window_size].unsqueeze(0)
                outputs = model(
                    window_tokens,
                    attention_mask=torch.ones_like(window_tokens),
                    output_hidden_states=True,
                )
                H = outputs.hidden_states[layer_idx].squeeze(0).cpu().numpy()
                all_activations.append(H)
                metadata["windows_extracted"] += 1
            except Exception:
                metadata["failed"] += 1

    if not all_activations:
        raise ValueError("No activations extracted")

    # Global PCA (analogous to neuron cap in SU pipeline)
    stacked = np.concatenate(all_activations, axis=0)
    pca = PCA(n_components=pca_dim, random_state=seed)
    pca.fit(stacked)
    explained = pca.explained_variance_ratio_.sum()

    reduced = [pca.transform(H) for H in all_activations]
    metadata["pca_dim"] = pca_dim
    metadata["explained_variance"] = float(explained)
    metadata["source"] = "real_gpt2"

    return reduced, metadata


def activations_to_spd(
    activation_windows: List[np.ndarray],
    eps: float = COV_EPS,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Convert activation matrices to SPD covariances + logs.

    Each (seq_len, feat_dim) window → (feat_dim, feat_dim) covariance.
    Direct analog of neural pipeline:
      Neural: (time_bins, neurons) → neuron×neuron covariance
      AI:     (tokens, features)   → feature×feature covariance

    Args:
        activation_windows: List of (seq_len, feat_dim) arrays.
        eps: Regularization for SPD guarantee.

    Returns:
        (covs, log_covs): Covariance matrices and their logs.
    """
    d = activation_windows[0].shape[1]
    covs = []
    log_covs = []
    for X in activation_windows:
        C = np.cov(X.T) + eps * np.eye(d)
        covs.append(C)
        log_covs.append(mat_log(C))
    return covs, log_covs


def compute_global_kappa(
    log_covs: List[np.ndarray],
    ns: int = NSAMPLES,
    B: int = BOOTSTRAP,
    seed: int = SEED,
) -> Dict:
    """Compute global κ from all covariance matrices.

    Args:
        log_covs: List of matrix logarithms.
        ns: Triangle samples per bootstrap.
        B: Bootstrap iterations.
        seed: Random seed.

    Returns:
        Dict with kappa, CI, and distance statistics.
    """
    D = distance_matrix(log_covs)
    k_med, k_ci = tri_kappa_bootstrap(D, ns=ns, B=B, seed=seed)

    upper = D[np.triu_indices(len(D), 1)]
    return {
        "kappa": float(k_med),
        "kappa_ci": [float(k_ci[0]), float(k_ci[1])],
        "n_covariances": len(log_covs),
        "dist_median": float(np.median(upper)),
        "dist_std": float(np.std(upper)),
    }


def run_pipeline(use_real_gpt2: bool = True) -> Dict:
    """Execute the corrected AI SPD pipeline end-to-end.

    Args:
        use_real_gpt2: If True, extract from GPT-2 (requires GPU).
                       If False, use synthetic activations.

    Returns:
        Full results dict with kappa, nulls, and dynamics.
    """
    t0 = time.time()

    if use_real_gpt2:
        activations, meta = extract_gpt2_activations()
    else:
        activations, meta = _generate_synthetic(NUM_WINDOWS, TEXT_WINDOW_SIZE, PCA_DIM)

    covs, log_covs = activations_to_spd(activations)
    global_result = compute_global_kappa(log_covs)

    # Null models
    from .null_models import shuffle_windows, permute_tokens, shuffle_features

    null_results = {}
    for name, null_fn in [
        ("window_shuffle", shuffle_windows),
        ("token_permute", permute_tokens),
        ("feature_shuffle", shuffle_features),
    ]:
        null_acts = null_fn(activations, seed=99)
        _, null_logs = activations_to_spd(null_acts)
        null_global = compute_global_kappa(null_logs, B=100)
        null_results[name] = {
            "kappa": null_global["kappa"],
            "delta_kappa": global_result["kappa"] - null_global["kappa"],
        }

    return {
        "pipeline": "corrected_ai_spd",
        "extraction": meta,
        "global_kappa": global_result,
        "null_models": null_results,
        "elapsed_seconds": time.time() - t0,
    }


def _generate_synthetic(
    num_windows: int,
    seq_len: int,
    feat_dim: int,
    seed: int = SEED,
) -> Tuple[List[np.ndarray], Dict]:
    """Synthetic activations with hierarchical factor structure."""
    rng = np.random.default_rng(seed)
    n_factors_per_level = [2, 4, 8, 16]
    factor_strengths = [3.0, 1.5, 0.8, 0.4]

    all_activations = []
    for w in range(num_windows):
        doc_phase = w / num_windows * 2 * np.pi
        X = np.zeros((seq_len, feat_dim))

        for level, (nf, strength) in enumerate(zip(n_factors_per_level, factor_strengths)):
            factor_dim = feat_dim // nf
            for f_idx in range(nf):
                phase = doc_phase + level * 0.5 + f_idx * 0.3
                loading = np.zeros(feat_dim)
                start = f_idx * factor_dim
                end = min(start + factor_dim, feat_dim)
                loading[start:end] = strength * (1 + 0.3 * np.sin(phase))

                factor_scores = rng.normal(0, 1, seq_len)
                for t in range(1, seq_len):
                    factor_scores[t] = 0.7 * factor_scores[t - 1] + 0.3 * factor_scores[t]
                X += np.outer(factor_scores, loading)

        X += 0.2 * rng.normal(0, 1, (seq_len, feat_dim))
        all_activations.append(X)

    return all_activations, {"source": "synthetic_hierarchical", "n_windows": num_windows}


if __name__ == '__main__':
    import json
    from pathlib import Path

    data_dir = Path(__file__).resolve().parent.parent / 'data'
    out_path = data_dir / 'corrected_gpt2_results.json'

    try:
        results = run_pipeline(use_real_gpt2=True)
    except Exception:
        print("Real GPT-2 extraction failed, falling back to synthetic.")
        results = run_pipeline(use_real_gpt2=False)

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Wrote AI pipeline results to {out_path}")
