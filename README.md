# CORES for Monocular Depth Estimation

Computer Vision project investigating convolutional-response OOD detection in a monocular depth estimation network.

## Current status

The completed project is **Kaggle-first**. The executable notebook is in
[`notebooks/project_2_cores_mde_kaggle.ipynb`](notebooks/project_2_cores_mde_kaggle.ipynb).

Implemented so far:

- portable Kaggle/Colab/local environment detection;
- GPU and runtime diagnostics;
- reproducibility configuration;
- dataset discovery and early path validation;
- RMSE, AbsRel, δ1, δ2 and δ3 depth metrics;
- deterministic unit and notebook smoke tests for the depth metrics.
- optimized NYU/KITTI quick-mode loading;
- a modern FastDepth-style MobileNetV2 baseline with named CORES feature levels;
- a GPU forward-pass and output-shape smoke test;
- masked metric-depth L1 loss and a one-batch overfitting smoke test;
- mixed-precision train/validation loops, best-checkpoint saving, NYU depth metrics,
  per-epoch and emergency recovery checkpoints, learning curves and qualitative
  prediction panels;
- full NYU training configuration with a deterministic validation split, untouched
  official test set, learning-rate scheduling, early stopping, compatible resume
  and mid-epoch recovery checkpoints every 250 batches;
- unbuffered notebook logging and a private Kaggle checkpoint dataset for recovery
  across independent runs;
- a tested single-layer CORES implementation on signed pre-activation responses,
  layer-wise and normalized multi-layer NYU/KITTI evaluation, component ablations,
  bootstrap confidence intervals, RGB-statistics and untrained-network controls,
  synthetic near-OOD corruptions, leakage-free weighted aggregation, calibration
  stability analysis and CSV/JSON exports;
- eight reproducible report figures and a complete final Kaggle evaluation.
- an independently trained ResNet18 depth ablation, matched CORES analysis,
  separate checkpoints, raw outputs, and an architecture-comparison figure.

The experimental design and working checklist are documented in
[`PROJECT_GUIDELINES.md`](PROJECT_GUIDELINES.md). Confirmed metrics are collected
in [`RESULTS.md`](RESULTS.md), raw tables in [`results/`](results/), generated
figures in [`figures/`](figures/), and the report in [`REPORT.md`](REPORT.md).
The submission presentation is available as
[`presentation/CORES_MDE_presentation.pptx`](presentation/CORES_MDE_presentation.pptx),
with an editable text outline in [`PRESENTATION.md`](PRESENTATION.md). The
requirement-by-requirement audit is in [`COMPLIANCE.md`](COMPLIANCE.md).

## Intended experiment

- ID training/evaluation domain: NYU Depth v2;
- OOD domain: KITTI;
- initial MDE architecture: FastDepth;
- OOD method: CORES;
- OOD metrics: AUROC and FPR95;
- depth metrics: RMSE, AbsRel, δ1, δ2 and δ3.

## Running on Kaggle

1. Import `notebooks/project_2_cores_mde_kaggle.ipynb` into Kaggle.
2. Select a **T4 GPU** accelerator. The tested PyTorch build does not support
   the P100's older `sm_60` architecture.
3. Attach the three inputs listed below. The checkpoint input is needed only to
   reproduce evaluation without repeating the four-hour training run.
4. For a loader and forward-pass smoke test, set `quick_mode=True`,
   `train_model=False`, and `load_checkpoint=False` in `Config`.
5. For evaluation of the final model, use `quick_mode=False`,
   `train_model=False`, and `load_checkpoint=True`.
6. To retrain from scratch, use `quick_mode=False`, `train_model=True`, and
   `load_checkpoint=False`. The notebook saves best, last, mid-epoch, and
   emergency checkpoints under `/kaggle/working/cores-mde/checkpoints`.
7. Run all cells in order. CSV/JSON results and PNG figures are written under
   `/kaggle/working/cores-mde` and appear in the saved notebook output.

The baseline uses one GPU (`cuda:0`). Selecting T4 x2 does not require
distributed-training code; the second GPU can initially remain unused.

The final detector is the middle-encoder magnitude-only CORES score. It reaches
AUROC 0.999943 and FPR95 0.000 on NYU ID versus KITTI OOD and remains stable
across five calibration seeds and calibration sets as small as 16 images.

The separate ResNet18 notebook is
[`notebooks/resnet18_architecture_ablation_kaggle.ipynb`](notebooks/resnet18_architecture_ablation_kaggle.ipynb)
and its Kaggle kernel is `plomo02/project2-cv-resnet18-ablation`. It defaults to
loading `plomo02/project2-cv-resnet18-checkpoints`, preserving the original
MobileNetV2 notebook and checkpoints. ResNet18 reaches middle-magnitude AUROC
0.834523 and FPR95 0.499, demonstrating that CORES performance is strongly
architecture-dependent even when depth estimation remains competitive.

Recommended Kaggle inputs:

- [NYUv2 official split](https://www.kaggle.com/datasets/awsaf49/nyuv2-official-split-dataset) (`awsaf49/nyuv2-official-split-dataset`);
- [KITTI depth prediction evaluation](https://www.kaggle.com/datasets/artemmmtry/kitti-depth-prediction-evaluation) (`artemmmtry/kitti-depth-prediction-evaluation`).
- private final checkpoint dataset (`plomo02/project2-cv-checkpoints`).

Protocol references:

- [official NYU Depth v2 dataset documentation](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html);
- [KITTI depth benchmark devkit](https://github.com/joseph-zhong/KITTI-devkit).

The KITTI `val_selection_cropped` loader and the NYU train/official-test loader
are implemented. Both pair RGB and ground-truth depth by filename and preserve
native-resolution depth for evaluation.

The notebook resolves both Kaggle input layouts automatically:

- `/kaggle/input/<dataset-slug>`;
- `/kaggle/input/datasets/<owner>/<dataset-slug>`.

The checked Kaggle paths currently used as defaults are:

- `/kaggle/input/datasets/awsaf49/nyuv2-official-split-dataset`;
- `/kaggle/input/datasets/artemmmtry/kitti-depth-prediction-evaluation`.
