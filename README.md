# CORES for Monocular Depth Estimation

Computer Vision project investigating convolutional-response OOD detection in a monocular depth estimation network.

## Current status

The repository is being developed **Kaggle-first**. The initial executable scaffold is in [`notebooks/project_2_cores_mde_kaggle.ipynb`](notebooks/project_2_cores_mde_kaggle.ipynb).

The experimental design and working checklist are documented in [`PROJECT_GUIDELINES.md`](PROJECT_GUIDELINES.md).

## Intended experiment

- ID training/evaluation domain: NYU Depth v2;
- OOD domain: KITTI;
- initial MDE architecture: FastDepth;
- OOD method: CORES;
- OOD metrics: AUROC and FPR95;
- depth metrics: RMSE, AbsRel, δ1, δ2 and δ3.

## Running on Kaggle

1. Create a Kaggle notebook with a GPU accelerator.
2. Upload or import `notebooks/project_2_cores_mde_kaggle.ipynb`.
3. Attach the required datasets from the notebook's **Input** panel.
4. Set the dataset folder names in the `Config` cell.
5. Run the notebook from top to bottom, initially with `QUICK_MODE = True`.

Dataset-specific loaders and training are the next implementation milestone. Dataset names and layouts will be fixed only after selecting the exact Kaggle dataset sources.

