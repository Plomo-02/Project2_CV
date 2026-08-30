# Project 2 compliance audit

This audit maps the official Fall/Winter 2026 Project 2 brief to repository
evidence. “Encouraged” items are distinguished from mandatory deliverables.

| Brief requirement | Evidence | Status |
|---|---|---|
| Python and PyTorch implementation | Kaggle notebook and `src/` modules | Complete |
| Code or notebook in GitHub | `notebooks/project_2_cores_mde_kaggle.ipynb` | Complete |
| Dataset or dataset links | Kaggle inputs and links in `README.md` | Complete |
| Detailed README and run instructions | `README.md`, Kaggle-first workflow | Complete; final proofreading pending |
| Project presentation in repository | `presentation/CORES_MDE_presentation.pptx` and editable `PRESENTATION.md` | Complete |
| Prepare and split NYU for training/validation | Deterministic 42,826/4,758 split; untouched 654-image official test | Complete |
| Collect and resize KITTI OOD samples | Official 1,000-image validation selection at 224 × 304 | Complete |
| Implement and train a lightweight depth model | 2.39M-parameter FastDepth-style MobileNetV2, 20 epochs | Complete |
| Compare two or more models | The brief says this is encouraged, not mandatory | Not performed; declared limitation |
| Integrate layer-wise CORES response statistics and score | Signed response hooks and four named stages | Complete |
| Evaluate OOD with AUROC and FPR95 | NYU official test versus KITTI | Complete |
| Evaluate depth with RMSE, AbsRel, δ1, δ2, δ3 on ID | NYU validation and official test | Complete |
| Evaluate depth metrics on OOD | Version 17: RMSE 9.8854, AbsRel 0.6912, δ1 0.0792 at 0–80 m | Complete |
| Ablate different layers/model depth | Early, middle, late encoder and late decoder | Complete |
| Compare convolutional architectures | One architecture plus trained/pretrained/random controls | Partial; declared limitation |
| Provide an interpretable analysis | Layer, component, RGB, training-state, corruption, threshold and stability controls | Complete |

## Additional work beyond the minimum

- leakage-free validation calibration and a paper-style synthetic calibration;
- full/magnitude/frequency/positive/negative CORES component ablations;
- per-sample 20% response-selected dense-prediction adaptation;
- uniform, encoder-only, leave-one-layer-out and synthetic-weighted aggregation;
- 500-replicate bootstrap confidence intervals;
- RGB Mahalanobis baseline and trained/ImageNet/random controls;
- near-OOD brightness, blur and Gaussian-noise experiments;
- five-seed, five-size calibration stability study;
- recovery checkpoints and private Kaggle checkpoint dataset;
- raw CSV/JSON outputs and reproducible report figures.

## Submission actions that still require the students

1. Insert both students’ names, IDs, and contribution statement.
2. Export the presentation to the format requested by the instructor.
3. Ensure both students can explain the CORES equations, the dense-prediction
   adaptation, dataset splits, leakage controls, and principal failure cases.
4. Replace any wording that does not reflect the students’ own understanding;
   the official brief explicitly warns against submitting generative-AI-derived
   material without ownership and comprehension.
