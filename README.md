# CORES for Monocular Depth Estimation

Computer Vision project investigating convolutional-response OOD detection in a monocular depth estimation network.

## Current status

The repository is being developed **Kaggle-first**. The executable notebook is in [`notebooks/project_2_cores_mde_kaggle.ipynb`](notebooks/project_2_cores_mde_kaggle.ipynb).

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
- masked metric-depth L1 loss and a one-batch overfitting smoke test for the training pipeline.

The experimental design and working checklist are documented in [`PROJECT_GUIDELINES.md`](PROJECT_GUIDELINES.md).

## Intended experiment

- ID training/evaluation domain: NYU Depth v2;
- OOD domain: KITTI;
- initial MDE architecture: FastDepth;
- OOD method: CORES;
- OOD metrics: AUROC and FPR95;
- depth metrics: RMSE, AbsRel, δ1, δ2 and δ3.

## Running on Kaggle

1. Create a Kaggle notebook with the **GPU T4 x2** accelerator. The current
   Kaggle PyTorch build does not support the P100's `sm_60` architecture.
2. Upload or import `notebooks/project_2_cores_mde_kaggle.ipynb`.
3. Attach the required datasets from the notebook's **Input** panel.
4. Set the dataset folder names in the `Config` cell.
5. Run the notebook from top to bottom, initially with `QUICK_MODE = True`.

The baseline uses one GPU (`cuda:0`). Selecting T4 x2 does not require
distributed-training code; the second GPU can initially remain unused.

Dataset-specific loaders and training are the next implementation milestone. Dataset names and layouts will be fixed only after selecting the exact Kaggle dataset sources.

Recommended Kaggle inputs:

- [NYUv2 official split](https://www.kaggle.com/datasets/awsaf49/nyuv2-official-split-dataset) (`awsaf49/nyuv2-official-split-dataset`);
- [KITTI depth prediction evaluation](https://www.kaggle.com/datasets/artemmmtry/kitti-depth-prediction-evaluation) (`artemmmtry/kitti-depth-prediction-evaluation`).

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
