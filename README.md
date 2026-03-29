# Information Geometry

**Cognition as Active Geometry — the neural instantiation of a universal law.**

The geometric state equation

$$\kappa = \left(\frac{h \ln 2}{n-1}\right)^2$$

predicts the curvature of information-processing manifolds with **zero free parameters**. This repo validates the equation in the neural and AI domains: single-unit recordings, fMRI, EEG, and transformer architectures — all measured on symmetric positive-definite (SPD) covariance manifolds using a unified pipeline.

## The Hyperbolic Trilogy

One equation. Three papers. Three substrates.

| Paper | Title | Repo | Domain | $\kappa$ range |
|-------|-------|------|--------|----------------|
| I | [Evolution as Active Geometry](https://www.biorxiv.org/content/10.64898/2026.03.09.710612v2) | [`active-geometry`](https://github.com/sentry-bio/active-geometry) | Genomic / viral / proteomic | 1.23–16.4 |
| II | A Geometric State Equation for Information-Generating Hierarchies | **`information-geometry`** | **Neural / cognitive / AI** | **0.27–0.49** |
| III | Cognition as Active Geometry | [`convergent-alphabets`](https://github.com/sentry-bio/convergent-alphabets) | Linguistic | 0.75–1.31 |

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

## Effective Alphabet Convergence

The mechanism connecting all three papers: raw alphabet sizes vary wildly, but the **effective alphabet** — the number of accessible transitions per symbol — converges across substrates.

| System | Raw alphabet | $K_\text{eff}$ | $h$ (bits) | $\kappa$ ($n=2$) |
|--------|-------------|----------------|------------|-------------------|
| DNA | 4 bases | **2.83** | 1.50 | 1.08 |
| Language | ~170 IPA phonemes | **3.14** | 1.65 | 1.31 |
| Protein | 20 amino acids | 2.69 | 1.43 | 0.98 |
| Neural | N/A | N/A | 1.04 | 0.485 |
| AI (GPT-2) | N/A | N/A | 0.84 | 0.348 |

Five rows, one equation, zero free parameters. See [`convergent-alphabets`](https://github.com/sentry-bio/convergent-alphabets) for the full linguistic validation.

## The Criticality Corollary

The brain's distance from criticality is not a free parameter — it is set by the state equation.

$$1 - J = \frac{\kappa}{(h_0 \ln 2)^2} > 0 \quad \text{whenever } \kappa > 0$$

A system with tree-like hierarchy ($n = 2$) and positive information rate ($h > 0$) **cannot** be at criticality, because criticality means $\kappa = 0$ — flat geometry — no capacity to distinguish branches. The near-criticality that Beggs and Plenz discovered is not the brain finding an optimal edge. It is the shadow of a geometric constraint that has governed every information-generating hierarchy for 3.7 billion years.

## The Icosahedral Atlas

The state equation forces the optimal finite partition of consciousness to have **12 regions with icosahedral symmetry**. The proof chain (formalized in `lean/BiosphereCurvature/IcosahedralAtlas.lean`, 447 lines):

1. $\kappa^* > 0$ — from the state equation (machine-checked)
2. Geodesic spheres in $\mathbb{H}^2_{\kappa^*}$ carry round $S^2$ metric (axiomatized, textbook Riemannian geometry)
3. The optimal 12-point code on $S^2$ is icosahedral (axiomatized, [Cohn-Kumar 2007](https://doi.org/10.1090/S0894-0347-06-00546-7))
4. $N = 12$ is cardinality-optimal in a computable $\lambda$-window (machine-checked)

The Buddhist dvādasāyatana (twelve sense bases) — six sense organs paired with six sense objects — matches **10 of 12** vertices ($z = 1.91$, $p < 0.0001$). The two unmatched vertices correspond to **mano** (mind-sense) and **dhammā** (mental objects): the only āyatana pair defined as a taxonomic residual rather than grounded in a physical sense organ. The five physiologically anchored pairs (eye/form, ear/sound, nose/smell, tongue/taste, body/touch) match perfectly. The mismatch occurs exactly where introspective phenomenology loses the precision that dedicated sensory organs provide — a prediction about the limits of first-person observation, derived from the geometry.

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

data/                          Cohort CSVs, calibration manifest, multi-architecture results
lean/                          9 theorems in Lean 4 (523 lines, 0 sorry stubs)
constants.yaml                 Single source of truth
```

## Lean Proofs

9 theorems machine-checked in Lean 4 (523 lines, 0 sorry stubs):

```bash
cd lean && lake build
```

## Authors

Rohit Fenn & Amit Fenn, Sentry Bio, Inc.

## License

MIT
