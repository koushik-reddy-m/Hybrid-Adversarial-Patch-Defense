# Hybrid Adversarial Patch Defense

**ORB/AKAZE + Feature Squeezing + GradCAM Masking**

A 3-stage hybrid defense against adversarial patch attacks on ImageNette/ResNet-50. Achieves 93.2% recovered accuracy with 0.4% false positive rate while maintaining 99.6% clean image accuracy — all without model retraining.

---

## Setup (macOS with uv)

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone the repositorygit clone
https://github.com/koushik-reddy-m/Hybrid-Adversarial-Patch-Defense.git

# 3. Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# 4. Verify MPS (Apple Silicon)
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# 5. Launch Jupyter Lab
jupyter lab
```

---

## Project Structure

```
├── main.ipynb              - Main execution notebook
├── requirements.txt        - Python dependencies
├── src/
│   ├── config.py           - ALL hyperparameters and settings
│   ├── data_loader.py      - ImageNette download + data transforms
│   ├── model.py            - ResNet-50 wrapper + ImageNette class mapping
│   ├── attack.py           - Adversarial patch generation (Brown et al.)
│   ├── classical.py        - Stage 1: Keypoint detection (SIFT, ORB, AKAZE)
│   ├── feature_squeeze.py  - Stage 2: Feature squeezing detector
│   ├── gradcam_defense.py  - Stage 3: GradCAM localization + masking
│   ├── pipeline.py         - Full 3-stage pipeline + 5-method evaluation
│   ├── evaluate.py         - Plots, LaTeX tables, statistical tests
│   └── stats.py            - Bootstrap CIs, chi-square significance tests
├── data/                   - ImageNette downloaded here (auto)
└── results/
    ├── patches/            - Trained patch saved as patch.pt
    ├── checkpoints/        - Thresholds, detector results, bboxes
    ├── plots/              - All figures and visualizations
    └── gradcam_examples/   - Per-image GradCAM visualizations
```

---

## Quick Start

| Step | Notebook Section | Key Output |
|------|-----------------|------------|
| 1 | Environment + Attack | `results/patches/patch.pt` |
| 2 | Stage 1: Classical Detection | `results/checkpoints/classical_results.pt` |
| 3 | Stage 2: Feature Squeezing | `results/checkpoints/fs_threshold.pt` |
| 4 | Stage 3: GradCAM Masking | `results/gradcam_examples/` |
| 5 | Full Pipeline + Evaluation | `results/results.pt` + all plots |

Every session starts by loading saved results from disk.

---

## Key Configuration (`src/config.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `PATCH_RATIO` | 0.40 | Patch covers 40% of image dimension (89×89 px) |
| `TARGET_CLASS` | 3 | ImageNette class 3 = chain saw |
| `PATCH_STEPS` | 1000 | Patch training iterations |
| `PATCH_LR` | 0.1 | Learning rate for patch optimizer |
| `GRID_CELLS` | 32 | Density grid dimensions for keypoint analysis |
| `FS_BIT_DEPTH` | 2 | Bit-depth reduction (4 quantization levels) |
| `FS_MAX_FPR` | 0.10 | Maximum acceptable false positive rate |
| `GRADCAM_THRESH` | 0.15 | Top-15% activation defines hot zone |
| `EVAL_TEST_N` | 1000 | Number of images for final evaluation |

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Clean Accuracy | Model accuracy on unmodified (clean) images |
| Attack Success Rate | % of correctly-classified images fooled by the patch |
| Recovered Accuracy | % of patched images correctly classified after defense |
| False Positive Rate | % of clean images incorrectly flagged as adversarial |

---

## Results (n = 1,000 paired images)

| Method | Clean Acc | Attack SR | Recovered Acc | FPR |
|--------|-----------|-----------|---------------|-----|
| No defense | 99.9% | 69.1% | 30.9% | 0.0% |
| JPEG compression | 99.9% | 23.0% | 77.0% | 0.1% |
| Feature squeezing | 91.0% | 11.5% | 88.5%* | 9.0% |
| GradCAM only | 95.5% | 3.6% | 96.4% | 4.5% |
| **Hybrid pipeline** | **99.6%** | **6.8%** | **93.2%** | **0.4%** |

*Detection rate only — images are rejected, not corrected.

---

## Architecture

```
Input Image
    │
    ▼
Stage 1: ORB Keypoint Detection
    ├─ No cluster found → Classify normally
    └─ Cluster found → Continue
         │
         ▼
Stage 2: Feature Squeezing
    ├─ Score < 0.15 → Classify normally
    └─ Score ≥ 0.15 → Continue
         │
         ▼
Stage 3: GradCAM Masking + Reclassification
    ├─ Intersect ORB bbox ∩ GradCAM bbox
    ├─ Mask intersection region
    └─ Reclassify masked image
```

---

## References

1. Brown, T., et al. (2017). Adversarial Patch. *arXiv:1712.09665*.
2. Xu, W., et al. (2018). Feature Squeezing. *NDSS 2018*.
3. Selvaraju, R., et al. (2017). Grad-CAM. *ICCV 2017*.
4. He, K., et al. (2016). Deep Residual Learning. *CVPR 2016*.
5. Lowe, D. (2004). SIFT. *IJCV 60(2)*.
6. Alcantarilla, P., et al. (2012). KAZE Features. *ECCV 2012*.
7. Rublee, E., et al. (2011). ORB. *ICCV 2011*.

---

## Author

**Madhadi Koushik Reddy** 
