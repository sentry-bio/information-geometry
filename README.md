# Information Geometry

**The geometry of hierarchical computation across neural and artificial systems.**

The geometric state equation

$$\kappa = \left(\frac{h \ln 2}{n-1}\right)^2$$

predicts the curvature of information-processing manifolds with **zero free parameters**. This repo validates the equation in the neural and AI domains: single-unit recordings, fMRI, EEG, and transformer architectures — all measured on symmetric positive-definite (SPD) covariance manifolds using a unified pipeline.

## Key Results

| Domain | System | κ | h (bits) | n_implied |
|--------|--------|---|----------|-----------|
| **Single-unit** | Steinmetz Neuropixels (39 sessions) | 0.485 ± 0.005 | 1.04 ± 0.36 | **2.03 ± 0.36** |
| **fMRI** | ABIDE Pitt (20 subjects) | 0.494 | — | — |
| **EEG** | EEGBCI (19 subjects) | 0.44–0.48 | — | — |
| **AI (GPT-2)** | SPD covariance, 12 layers | 0.348 | — | — |
| **AI (ViT-Base)** | CLS token covariance, 12 layers | 0.270 | — | — |

**Headline finding**: Volume entropy — the geodesic ball growth rate from Manning's theorem — is the correct entropy measure for SPD manifolds. The full 39-session Neuropixels cohort gives **n = 2.032 ± 0.359** (t-test vs n=2: p = 0.59), the first independent confirmation of n = 2 outside symbolic codes.

**Brain region prediction**: Sessions dominated by recurrent connectivity (thalamic relay, prefrontal cortex) show n > 2, as the theory predicts for non-tree-like information hierarchies. Spearman ρ = 0.362, p = 0.023.

## The Trilogy

| Repo | Domain | Paper | κ range |
|------|--------|-------|---------|
| [`active-geometry`](https://github.com/sentry-bio/active-geometry) | Genomic / evolutionary | Paper I | 1.23–1.34 |
| **`information-geometry`** | **Neural / cognitive / AI** | **Paper II** | **0.27–0.49** |
| [`convergent-alphabets`](https://github.com/sentry-bio/convergent-alphabets) | Linguistic | Paper III | 0.75–1.31 |

## Structure

```
src/
  spd_geometry.py              Core SPD manifold functions (Log-Euclidean, AIRM, triangle excess)
  volume_entropy.py            Manning volume entropy — the key h measurement
  single_unit_pipeline.py      Steinmetz Neuropixels → SPD → κ
  fmri_pipeline.py             ABIDE fMRI → SPD → κ
  eeg_pipeline.py              EEGBCI EEG → SPD → κ
  ai_spd_pipeline.py           Transformer activations → SPD → κ
  null_models.py               Trial permutation, bin shuffle, feature shuffle

validation/
  neuropixels/                 39-session volume entropy + brain region stratification
  ai/                          GPT-2 + ViT-Base natural κ + multi-architecture sweeps

data/                          Cohort CSVs, calibration manifest, multi-architecture results

lean/                          9 theorems in Lean 4 (523 lines, 0 sorry stubs)

constants.yaml                 Single source of truth
```

## The Volume Entropy Breakthrough

Previous attempts to measure h for neural data tested five entropy candidates (von Neumann, spike rate, spike marginal, VAR(1) innovation). All gave n ≈ 3–4. The problem: they measured the wrong quantity.

Manning's theorem (1979) relates the **volume entropy** of a negatively curved manifold to its sectional curvature:

```
h_vol = (n−1) · √κ   [nats]
```

Volume entropy is the exponential growth rate of geodesic balls — the geometrically native entropy for SPD manifolds. At 2.4-second covariance windows on 39 Steinmetz sessions:

- κ = 0.485 ± 0.005
- h_vol = 1.04 ± 0.36 bits
- **n = 2.03 ± 0.36** (p = 0.59 vs n = 2)

## Brain Region Stratification

The theory predicts n = 2 for tree-like hierarchies. Recurrent lateral connectivity pushes n above 2.

| Connectivity Type | Sessions | n_implied | Interpretation |
|---|---|---|---|
| Hierarchical cortex | 4 | 2.08 ± 0.22 | Feedforward → n ≈ 2 ✓ |
| Recurrent (thalamic/PFC) | 7 | 2.38 ± 0.28 | Relay loops → n > 2 ✓ |
| Subcortical | 11 | 1.78 ± 0.33 | Variable |
| Mixed | 17 | 2.04 ± 0.30 | Population average → n ≈ 2 ✓ |

## Multi-Architecture AI Comparison

| Architecture | Natural κ | Layer Pattern |
|---|---|---|
| GPT-2 (124M) | 0.348 | Flat across layers 0–10 |
| ViT-Base (86M) | 0.270 | 0.52 (L1) → 0.27 (L3–12 plateau) |

Null controls: token shuffle → κ = 0.07; feature whitening → κ = 0.91. The signal lives in temporal interaction structure.

## Lean Proofs

9 theorems machine-checked in Lean 4 (523 lines, 0 sorry stubs):

```bash
cd lean && lake build
```

## Authors

Rohit Fenn & Amit Fenn, Sentry Bio, Inc.

## License

MIT
