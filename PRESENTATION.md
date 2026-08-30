# CORES for Monocular Depth Estimation

Computer Vision — Project 2  
_Names and student IDs_

---

## 1. Problem and research question

- Monocular depth models may remain confident outside their training domain.
- Train FastDepth-style MDE on indoor NYU Depth v2.
- Treat outdoor KITTI as OOD.
- Question: can signed convolutional responses detect this domain shift?

---

## 2. Pipeline

NYU RGB → FastDepth-style MobileNetV2 → signed layer responses → CORES → ID/OOD score

- ID: NYU official test, 654 images.
- OOD: KITTI validation selection, 1,000 images.
- Input: 224 × 304.
- Metrics: RMSE, AbsRel, δ1–δ3; AUROC and FPR95.

---

## 3. Depth baseline

- 2.39M parameters.
- MobileNetV2 encoder and depthwise FastDepth-style decoder.
- 42,826 training / 4,758 validation samples.
- AdamW, masked L1, AMP, validation checkpoints, 20 epochs.
- Best checkpoint: epoch 19.
- KITTI OOD: RMSE 9.8854, AbsRel 0.6912, δ1 0.0792 (0–80 m).

![Training curves](figures/01_fastdepth_training.png)

---

## 4. CORES in one slide

- For each kernel, extract maximum and minimum signed spatial response.
- Magnitude: distance beyond positive/negative thresholds.
- Frequency: fraction of kernels crossing each threshold.
- Higher log CORES score means more ID-like.
- Original CORES backtracks classifier kernels; dense regression has no class
  logits, so we evaluate all kernels and a 20% extreme-channel variant.

---

## 5. Where is the OOD signal?

| Layer | AUROC | FPR95 |
|---|---:|---:|
| Early encoder | 0.9897 | 0.044 |
| Middle encoder | **0.9999** | **0.001** |
| Late encoder | 0.9028 | 0.466 |
| Late decoder | 0.5144 | 0.798 |

![ROC curves](figures/04_cores_roc_curves.png)

---

## 6. Main result

- Middle encoder, magnitude-only: AUROC **0.999943**, FPR95 **0.000**.
- Response frequency slightly hurts performance.
- The decoder is close to random.
- Interpretation: intermediate encoder responses retain domain information that
  is altered during task-specific depth reconstruction.

![Component ablation](figures/05_cores_component_ablation.png)

---

## 7. Is this just colour or random features?

| Control | AUROC | FPR95 |
|---|---:|---:|
| Trained FastDepth | 0.99994 | 0.000 |
| ImageNet encoder only | 0.91886 | 0.324 |
| Random encoder | 0.45062 | 1.000 |
| RGB Mahalanobis | 0.82283 | 0.759 |

Depth training creates most of the useful OOD separation.

---

## 8. Multi-layer contribution

- All-layer mean: 0.999737 / 0.001.
- Encoder-only mean: 0.999884 / 0.000.
- Removing middle encoder: 0.987777 / 0.052.
- No aggregation beats middle magnitude-only.
- Synthetic weighting saturates because pure noise is too easy for every layer.

![Aggregation](figures/07_cores_multilayer_aggregation.png)

---

## 9. Robustness and stability

- Synthetic threshold calibration: AUROC 0.99868, FPR95 0.007.
- Five seeds × five calibration sizes.
- Even 16 calibration images: mean AUROC 0.999943.
- FPR95 remains zero in all 25 trials.

![Calibration stability](figures/08_cores_calibration_stability.png)

---

## 10. Failure cases and honest limitations

- NYU indoor versus KITTI road scenes is a broad domain shift.
- The NYU model is capped at 10 m, so KITTI metric depth deteriorates sharply.
- Gaussian near-OOD noise: AUROC 0.94273, FPR95 0.388.
- One trained architecture and one checkpoint.
- FastDepth-style MobileNetV2, not exact original MobileNetV1 FastDepth.
- Dense-prediction channel selection is an adaptation of classifier CORES.
- Synthetic noise is too easy to calibrate meaningful layer weights.

---

## 11. Conclusions

- CORES transfers effectively from classification to dense depth estimation.
- The middle encoder is the best location for this model.
- Magnitude is sufficient; frequency and weak layers can dilute the score.
- The final detector is simple, post-hoc and stable, but near-OOD remains open.

---

## 12. Reproducibility and questions

- Complete Kaggle notebook and checkpoint recovery.
- Raw CSV/JSON results and eight generated figures.
- Deterministic splits, seeds and smoke tests.
- GitHub: <https://github.com/Plomo-02/Project2_CV>
