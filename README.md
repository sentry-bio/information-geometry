# The Geometry of Hierarchical Computation

**A Parameter-Free Invariant Across Neural and Artificial Systems**

Submission repository for the AAAI 2026 Spring Symposium on Machine Consciousness: Integrating Theory, Technology, and Philosophy.

## Key Result

A zero-parameter state equation predicts the geometry of hierarchical information processing across five domains:

```
κ = (h ln 2 / (n-1))²
```

where κ is the triangle-excess curvature on SPD covariance manifolds, h is the entropy rate (bits/symbol), and n is the embedding dimension.

Validated across:
- **Evolution** (genomic embeddings): κ = 1.25, h = 1.6
- **Single-unit** (Neuropixels): κ = 0.43, h = 0.94
- **fMRI** (ABIDE): κ = 0.49, h = 1.01
- **EEG** (EEGBCI): κ = 0.18, h = 0.61
- **AI** (GPT-2 SPD): κ = 0.34, h = 0.84

Cross-domain correlation: ρ = 0.984.

## Repository Structure

```
aaai-sss26-consciousness/
├── paper/
│   ├── main.tex              # 6-8 page AAAI-format paper
│   ├── references.bib        # Bibliography
│   └── figures/               # Paper figures (generated from notebooks)
│
├── src/
│   ├── spd_geometry.py       # SPD covariance + Log-Euclidean + triangle excess
│   ├── single_unit_pipeline.py   # Steinmetz Neuropixels analysis
│   ├── fmri_pipeline.py      # ABIDE fMRI analysis
│   ├── ai_spd_pipeline.py    # GPT-2 SPD pipeline
│   └── null_models.py        # All null models
│
├── lean/
│   ├── lakefile.lean
│   ├── lean-toolchain
│   └── BiosphereCurvature/
│       └── KappaCurvature.lean   # Machine-checked proofs (523 lines)
│
├── data/
│   ├── su_cohort.csv             # 39 sessions, single-unit results
│   ├── fmri_cohort.csv           # 20 subjects, fMRI results
│   ├── eeg_sensor_cov_summary.json  # 19 subjects, EEG results
│   ├── corrected_gpt2_results.json  # SPD-corrected AI results
│   ├── robustness_results.json   # PCA/layer robustness
│   └── calibration_manifest.json # Frozen parameters
│
├── notebooks/
│   ├── 01_reproduce_results.ipynb  # End-to-end reproduction
│   └── 02_generate_figures.ipynb   # All four paper figures
│
└── scripts/
    ├── run_neural.sh            # Reproduce neural analysis
    └── run_ai.sh                # Reproduce AI analysis
```

## Quick Start

```bash
pip install -r requirements.txt

# Reproduce key results
jupyter notebook notebooks/01_reproduce_results.ipynb

# Generate paper figures
jupyter notebook notebooks/02_generate_figures.ipynb
```

## Lean Proofs

The 9 core theorems are machine-checked in Lean 4 with Mathlib:

```bash
cd lean && lake build
```

Key theorems: closed-form κ, uniqueness, monotonicity in h and n, Lyapunov stability.

## Authors

Rohit Fenn & Amit Fenn, Sentry Bio, Inc.

## License

MIT
