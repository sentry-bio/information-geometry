#!/usr/bin/env python3
"""
Multi-Architecture Natural κ + Volume Entropy Sweep.

Extends the AI SPD pipeline to measure natural κ and volume entropy
across multiple transformer architectures and layers. Builds on
ai_spd_pipeline.py (GPT-2) and adds BERT, DistilGPT-2, ViT-Base,
ViT-Large, RoBERTa, and CLIP.

For each architecture and layer:
  1. Extract full activation matrices (tokens × features)
  2. PCA reduce to d dimensions
  3. Compute windowed covariances → SPD trajectory
  4. Measure κ via triangle excess
  5. Measure h via volume entropy (geodesic ball growth)
  6. Compute n_implied = 1 + h_vol/√κ

Usage:
    python -m src.multi_architecture_sweep \
        --output data/multi_architecture_natural.json \
        [--models gpt2 bert vit_base] \
        [--layers 1 3 6 9 12]
"""
import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

LN2 = np.log(2)

# ── SPD geometry (inlined to keep script standalone on remote) ───────────

def mat_log(C):
    from numpy.linalg import eigh
    w, V = eigh(C)
    w = np.clip(w, 1e-10, None)
    return V @ np.diag(np.log(w)) @ V.T

def log_euclidean_distance(L1, L2):
    D = L1 - L2
    return float(np.sqrt(np.sum(D * D)))

def distance_matrix_loge(log_covs):
    n = len(log_covs)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = log_euclidean_distance(log_covs[i], log_covs[j])
            D[i, j] = D[j, i] = d
    return D

def tri_kappa_bootstrap(D, ns=2000, B=500, seed=0):
    n = D.shape[0]
    if n < 3:
        return float('nan'), (float('nan'), float('nan'))
    med = np.median(D[np.triu_indices(n, 1)])
    if med > 1e-10:
        Dn = D / med
    else:
        Dn = D.copy()
    ests = []
    for b in range(B):
        rng = np.random.default_rng(seed + b)
        ex = []
        for _ in range(ns):
            i, j, k = rng.choice(n, 3, replace=False)
            a, b_, c = sorted([Dn[i, j], Dn[j, k], Dn[i, k]])
            if a + b_ > c:
                ex.append((a + b_ - c) / 2)
        ests.append(float(np.median(ex)) if ex else float('nan'))
    ests = np.array([e for e in ests if np.isfinite(e)])
    return float(np.median(ests)), (float(np.percentile(ests, 2.5)),
                                     float(np.percentile(ests, 97.5)))

def estimate_volume_entropy(D, n_centers=50, r2_threshold=0.5):
    n_pts = D.shape[0]
    if n_pts < 20:
        return {"h_vol_nats": float('nan'), "error": "too few points"}
    slopes = []
    for center in range(min(n_pts, n_centers)):
        dists = np.sort(D[center, :])
        dists = dists[dists > 0]
        if len(dists) < 10:
            continue
        unique_d = np.unique(dists)
        counts = np.array([np.sum(dists <= r) for r in unique_d])
        mask = counts > 0
        R = unique_d[mask]
        logN = np.log(counts[mask].astype(float))
        if len(R) < 5:
            continue
        lo, hi = int(0.2 * len(R)), int(0.8 * len(R))
        if hi - lo < 3:
            lo, hi = 0, len(R)
        R_mid, logN_mid = R[lo:hi], logN[lo:hi]
        A = np.vstack([R_mid, np.ones_like(R_mid)]).T
        result = np.linalg.lstsq(A, logN_mid, rcond=None)
        slope = result[0][0]
        ss_res = np.sum((logN_mid - (slope * R_mid + result[0][1])) ** 2)
        ss_tot = np.sum((logN_mid - logN_mid.mean()) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-30)
        if r2 > r2_threshold:
            slopes.append(slope)
    if not slopes:
        return {"h_vol_nats": float('nan'), "error": "no good fits"}
    arr = np.array(slopes)
    return {"h_vol_nats": float(np.median(arr)), "h_vol_bits": float(np.median(arr) / LN2),
            "h_vol_std": float(np.std(arr)), "n_good_fits": len(arr)}


# ── Model definitions ────────────────────────────────────────────────────

MODELS = {
    "gpt2": {
        "hf_name": "gpt2",
        "type": "causal_lm",
        "class": "GPT2Model",
        "n_layers": 12,
        "hidden_dim": 768,
        "params": "124M",
        "pooling": "all_tokens",
    },
    "distilgpt2": {
        "hf_name": "distilgpt2",
        "type": "causal_lm",
        "class": "GPT2Model",
        "n_layers": 6,
        "hidden_dim": 768,
        "params": "82M",
        "pooling": "all_tokens",
    },
    "bert": {
        "hf_name": "bert-base-uncased",
        "type": "masked_lm",
        "class": "BertModel",
        "n_layers": 12,
        "hidden_dim": 768,
        "params": "110M",
        "pooling": "all_tokens",
    },
    "roberta": {
        "hf_name": "roberta-base",
        "type": "masked_lm",
        "class": "RobertaModel",
        "n_layers": 12,
        "hidden_dim": 768,
        "params": "125M",
        "pooling": "all_tokens",
    },
    "vit_base": {
        "hf_name": "google/vit-base-patch16-224",
        "type": "vision",
        "class": "ViTModel",
        "n_layers": 12,
        "hidden_dim": 768,
        "params": "86M",
        "pooling": "cls_token",
    },
    "vit_large": {
        "hf_name": "google/vit-large-patch16-224",
        "type": "vision",
        "class": "ViTModel",
        "n_layers": 24,
        "hidden_dim": 1024,
        "params": "307M",
        "pooling": "cls_token",
    },
}

# ── Corpus ───────────────────────────────────────────────────────────────

TEXT_CORPUS = (
    "The rapid advancement of artificial intelligence has transformed "
    "our understanding of computation and cognition. Large language "
    "models demonstrate emergent capabilities that suggest complex "
    "internal representations. These systems process information through "
    "high-dimensional activation spaces that exhibit geometric structure. "
    "The study of consciousness has long sought to understand how "
    "subjective experience emerges from neural computation. Recent work "
    "in computational neuroscience suggests that geometric properties of "
    "neural state spaces may be fundamental. Hyperbolic geometry appears "
    "in biological systems from evolution to neural networks, suggesting "
    "universal principles. The relationship between information processing "
    "and geometric structure remains an open question. Machine learning "
    "models provide a unique window into the dynamics of complex "
    "information processing systems. Understanding the geometric "
    "principles underlying these dynamics could illuminate fundamental "
    "aspects of intelligence and the nature of hierarchical computation. "
) * 80  # ~16K tokens when tokenized


# ── Extraction ───────────────────────────────────────────────────────────

PCA_DIM = 64
NUM_WINDOWS = 400
WINDOW_SIZE = 128
STRIDE = 64
COV_EPS = 1e-6


def extract_text_model(model_key, layers, device="cuda"):
    """Extract activations from a text model (GPT-2, BERT, RoBERTa, DistilGPT-2)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    spec = MODELS[model_key]
    print("  Loading %s (%s)..." % (spec["hf_name"], spec["params"]))

    tokenizer = AutoTokenizer.from_pretrained(spec["hf_name"])
    model = AutoModel.from_pretrained(spec["hf_name"], output_hidden_states=True)
    model.eval().to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_tokens = tokenizer(TEXT_CORPUS, add_special_tokens=False)["input_ids"]
    tokens = torch.tensor(all_tokens, dtype=torch.long)

    results_by_layer = {}
    for layer_idx in layers:
        if layer_idx > spec["n_layers"]:
            continue

        activations = []
        with torch.no_grad():
            for start in range(0, len(tokens) - WINDOW_SIZE, STRIDE):
                if len(activations) >= NUM_WINDOWS:
                    break
                window = tokens[start:start + WINDOW_SIZE].unsqueeze(0).to(device)
                outputs = model(window, attention_mask=torch.ones_like(window),
                                output_hidden_states=True)
                H = outputs.hidden_states[layer_idx].squeeze(0).cpu().numpy()
                activations.append(H)

        results_by_layer[layer_idx] = activations
        print("    Layer %d: %d windows extracted" % (layer_idx, len(activations)))

    del model
    import torch; torch.cuda.empty_cache()
    return results_by_layer


def extract_vision_model(model_key, layers, device="cuda"):
    """Extract activations from a vision model (ViT-Base, ViT-Large)."""
    import torch
    from transformers import ViTModel, ViTFeatureExtractor
    from torchvision import datasets, transforms

    spec = MODELS[model_key]
    print("  Loading %s (%s)..." % (spec["hf_name"], spec["params"]))

    feature_extractor = ViTFeatureExtractor.from_pretrained(spec["hf_name"])
    model = ViTModel.from_pretrained(spec["hf_name"], output_hidden_states=True)
    model.eval().to(device)

    # Use random images (the geometry shouldn't depend on specific content)
    torch.manual_seed(42)
    dummy_images = torch.randn(NUM_WINDOWS, 3, 224, 224)

    results_by_layer = {}
    for layer_idx in layers:
        if layer_idx > spec["n_layers"]:
            continue

        activations = []
        with torch.no_grad():
            batch_size = 16
            for i in range(0, min(NUM_WINDOWS, len(dummy_images)), batch_size):
                batch = dummy_images[i:i+batch_size].to(device)
                outputs = model(pixel_values=batch, output_hidden_states=True)
                H = outputs.hidden_states[layer_idx]

                if spec["pooling"] == "cls_token":
                    # Each image gives one CLS vector; group images as "tokens"
                    cls_vectors = H[:, 0, :].cpu().numpy()  # (batch, hidden)
                    activations.append(cls_vectors)
                else:
                    for j in range(H.shape[0]):
                        activations.append(H[j].cpu().numpy())

        # For CLS pooling: stack batches into a sequence of CLS vectors
        if spec["pooling"] == "cls_token":
            all_cls = np.concatenate(activations, axis=0)  # (N_images, hidden)
            # Window the CLS vectors like a time series
            win_size = 32
            hop = 8
            windowed = []
            for s in range(0, len(all_cls) - win_size, hop):
                windowed.append(all_cls[s:s + win_size])
            activations = windowed

        results_by_layer[layer_idx] = activations
        print("    Layer %d: %d windows" % (layer_idx, len(activations)))

    del model
    import torch; torch.cuda.empty_cache()
    return results_by_layer


# ── SPD analysis ─────────────────────────────────────────────────────────

def analyze_activations(activations, pca_dim=PCA_DIM):
    """PCA reduce, compute covariances, measure κ and h_vol."""
    from sklearn.decomposition import PCA

    # Stack for global PCA
    stacked = np.concatenate(activations, axis=0)
    actual_dim = min(pca_dim, stacked.shape[1], stacked.shape[0] - 1)
    pca = PCA(n_components=actual_dim, random_state=42)
    pca.fit(stacked)
    explained = float(pca.explained_variance_ratio_.sum())
    reduced = [pca.transform(H) for H in activations]

    # Covariance matrices
    d = actual_dim
    log_covs = []
    for X in reduced:
        if X.shape[0] < 3:
            continue
        C = np.cov(X.T) + COV_EPS * np.eye(d)
        log_covs.append(mat_log(C))

    if len(log_covs) < 20:
        return {"error": "too few covariances (%d)" % len(log_covs),
                "n_covs": len(log_covs), "pca_explained": explained}

    D = distance_matrix_loge(log_covs)
    k, k_ci = tri_kappa_bootstrap(D)
    vol = estimate_volume_entropy(D)

    h_nats = vol.get("h_vol_nats", float('nan'))
    n_impl = 1 + h_nats / np.sqrt(max(k, 1e-10)) if (k > 0 and np.isfinite(h_nats)) else float('nan')
    h_pred_bits = float(np.sqrt(k) / LN2) if k > 0 else float('nan')

    return {
        "n_covs": len(log_covs),
        "pca_dim": actual_dim,
        "pca_explained": round(explained, 4),
        "kappa": round(float(k), 4),
        "kappa_ci": [round(float(k_ci[0]), 4), round(float(k_ci[1]), 4)],
        "h_vol_bits": round(vol.get("h_vol_bits", float('nan')), 4),
        "h_vol_nats": round(h_nats, 4) if np.isfinite(h_nats) else None,
        "h_predicted_n2_bits": round(h_pred_bits, 4),
        "n_implied": round(float(n_impl), 3) if np.isfinite(n_impl) else None,
        "volume_entropy_detail": vol,
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-architecture natural κ + volume entropy")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--models", nargs="+", default=["gpt2", "bert", "distilgpt2"],
                        help="Models to sweep (default: gpt2 bert distilgpt2)")
    parser.add_argument("--layers", nargs="+", type=int, default=[1, 3, 6, 9, 12],
                        help="Layer indices to measure (default: 1 3 6 9 12)")
    parser.add_argument("--device", default="cuda", help="torch device")
    args = parser.parse_args()

    import torch
    device = args.device if torch.cuda.is_available() else "cpu"
    print("Multi-Architecture Natural κ + Volume Entropy Sweep")
    print("Device: %s" % device)
    print("Models: %s" % ", ".join(args.models))
    print("Layers: %s" % args.layers)
    print("=" * 70)

    all_results = {}

    for model_key in args.models:
        if model_key not in MODELS:
            print("Unknown model: %s (available: %s)" % (model_key, ", ".join(MODELS)))
            continue

        spec = MODELS[model_key]
        print("\n[%s] %s (%s, %d layers)" % (model_key, spec["hf_name"],
                                               spec["params"], spec["n_layers"]))
        t0 = time.time()

        # Filter layers to valid range
        valid_layers = [l for l in args.layers if l <= spec["n_layers"]]

        # Extract activations
        if spec["type"] == "vision":
            acts_by_layer = extract_vision_model(model_key, valid_layers, device)
        else:
            acts_by_layer = extract_text_model(model_key, valid_layers, device)

        # Analyze each layer
        model_results = {"spec": spec, "layers": {}}
        for layer_idx in valid_layers:
            acts = acts_by_layer.get(layer_idx, [])
            if not acts:
                continue
            print("  Analyzing layer %d (%d windows)..." % (layer_idx, len(acts)))
            result = analyze_activations(acts)
            model_results["layers"][layer_idx] = result

            if "error" not in result:
                ni = result["n_implied"] if result["n_implied"] is not None else "N/A"
                hv = result["h_vol_bits"] if result["h_vol_bits"] is not None else "N/A"
                print("    κ=%.4f  h_vol=%s  n_impl=%s  (covs=%d, PCA=%.1f%%)" %
                      (result["kappa"], hv, ni, result["n_covs"],
                       result["pca_explained"] * 100))
            else:
                print("    ERROR: %s" % result["error"])

        model_results["elapsed_s"] = round(time.time() - t0, 1)
        all_results[model_key] = model_results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Natural κ by Architecture")
    print("=" * 70)
    for model_key, mr in all_results.items():
        layers = mr.get("layers", {})
        if not layers:
            continue
        # Use last valid layer as the "natural κ"
        last_layer = max(layers.keys())
        r = layers[last_layer]
        if "error" in r:
            continue
        ni = r["n_implied"] if r["n_implied"] is not None else "N/A"
        hv = r["h_vol_bits"] if r["h_vol_bits"] is not None else "N/A"
        print("  %-12s (L%d): κ=%.4f  h_vol=%s bits  n_implied=%s" %
              (model_key, last_layer, r["kappa"], hv, ni))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nWrote %s" % out_path)


if __name__ == "__main__":
    main()
