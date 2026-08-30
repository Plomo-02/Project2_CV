# CORES for Monocular Depth Estimation

## Abstract

This project studies whether convolutional responses from a monocular depth
estimation network can identify out-of-distribution inputs. A lightweight
FastDepth-style model is trained on NYU Depth v2 and evaluated for depth
prediction on its official test split. CORES is then adapted from image
classification to dense depth estimation and evaluated with NYU as the
in-distribution domain and KITTI as the out-of-distribution domain. The study
includes layer-wise and component ablations, RGB-statistics and training-state
controls, synthetic corruptions, leakage-free threshold calibration,
multi-layer aggregation, and calibration stability.

## 1. Introduction

Monocular depth estimators can produce plausible predictions even when their
inputs differ substantially from the training distribution. Detecting such
inputs is therefore important for interpreting model reliability. This work
investigates whether CORES, an OOD score based on signed convolutional
responses, remains useful when the underlying task is dense regression rather
than classification.

The main research questions are:

1. Can CORES distinguish NYU indoor images from KITTI outdoor images?
2. Which encoder or decoder stage contains the strongest OOD signal?
3. Which CORES components are responsible for the separation?
4. Does multi-layer aggregation improve over the best individual layer?
5. How stable are the conclusions across calibration protocols, seeds, and
   calibration-set sizes?

## 2. Related work

### 2.1 Monocular depth estimation and FastDepth

TODO: summarize monocular depth estimation and motivate the lightweight
FastDepth-style MobileNetV2 encoder-decoder baseline.

### 2.2 Out-of-distribution detection and CORES

TODO: introduce activation-based OOD detection and describe the original CORES
method, including response magnitude, response frequency, positive and negative
responses, and its original classification setting.

## 3. Method

### 3.1 Depth-estimation baseline

The baseline uses a MobileNetV2 encoder with a lightweight decoder and produces
one dense depth channel. RGB inputs are resized to 224 × 304. Invalid depth
pixels are excluded from the masked L1 training loss and from every evaluation
metric. The model contains 2,390,713 parameters and is trained on the official
NYU training split using a deterministic train/validation partition.

### 3.2 CORES adaptation to dense prediction

Signed pre-activation responses are captured at four locations: early, middle,
and late encoder stages and a late decoder stage. Since a depth estimator has
no class-specific output kernel, the baseline aggregates all convolutional
channels. This is an explicit adaptation of CORES rather than a claim of an
identical classification implementation.

For each layer, thresholds are calibrated using NYU validation responses only.
The score combines positive and negative response magnitudes and frequencies;
higher values indicate stronger agreement with the ID response pattern.

### 3.3 Multi-layer aggregation

Layer scores are centered and scaled using NYU validation statistics. Uniform,
encoder-only, early-middle, leave-one-layer-out, and synthetic-noise-weighted
aggregations are compared. Synthetic weights are estimated from clean NYU
validation samples and Gaussian/uniform noise only. KITTI is never used to
choose thresholds, normalization statistics, weights, or configurations.

## 4. Experimental protocol

### 4.1 Datasets and splits

- NYU Depth v2 official training split: depth training and internal validation.
- NYU Depth v2 official test split: final ID evaluation.
- KITTI depth-prediction evaluation split: final OOD evaluation.

The deterministic split contains 42,826 training, 4,758 validation, and 654 NYU
test samples. KITTI contributes 1,000 OOD samples.

### 4.2 Metrics

Depth quality is measured with L1, RMSE, AbsRel, and δ accuracy thresholds.
OOD ranking is measured with AUROC and FPR95, where lower FPR95 is better.
Selected comparisons include stratified percentile bootstrap confidence
intervals.

### 4.3 Controls and ablations

The experimental controls cover CORES components, response-selected channels,
an RGB Mahalanobis baseline, trained/ImageNet/random model states, synthetic
near-OOD corruptions, synthetic-noise threshold calibration, and calibration
subsampling across five random seeds.

## 5. Results

### 5.1 Depth-estimation baseline

The best validation checkpoint was obtained at epoch 19.

| Split | L1 | RMSE | AbsRel | δ1 | δ2 | δ3 |
|---|---:|---:|---:|---:|---:|---:|
| NYU validation | 0.1676 | 0.2778 | 0.0701 | 0.9530 | 0.9897 | 0.9971 |
| NYU official test | 0.5071 | 0.7901 | 0.2016 | 0.7181 | 0.9171 | 0.9698 |

### 5.2 Layer-wise OOD detection

| Layer | AUROC | FPR95 |
|---|---:|---:|
| Early encoder | 0.9897 | 0.044 |
| Middle encoder | **0.9999** | **0.001** |
| Late encoder | 0.9028 | 0.466 |
| Late decoder | 0.5144 | 0.798 |
| Uniform multi-layer mean | 0.9997 | **0.001** |

The middle encoder is the strongest individual feature stage. The decoder is
close to random, suggesting that depth reconstruction substantially changes or
discards the domain evidence available in intermediate encoder responses.

### 5.3 Ablations and controls

Magnitude-only middle-layer CORES reaches AUROC 0.99994 and FPR95 0.000. The RGB
baseline reaches only AUROC 0.82283 and FPR95 0.759. Performance decreases to
AUROC 0.91886 without depth training and to 0.45062 with a fully random encoder.
These controls show that the separation cannot be explained solely by simple
RGB statistics or arbitrary convolutional features.

Synthetic Gaussian/uniform threshold calibration reaches AUROC 0.99868 and
FPR95 0.007 without using KITTI for model selection. This small reduction
supports the robustness of the main conclusion to the threshold protocol.

### 5.4 Multi-layer aggregation and stability

| Aggregation | AUROC | FPR95 |
|---|---:|---:|
| All layers, uniform | 0.999737 | 0.001 |
| Synthetic-noise weighted | 0.999737 | 0.001 |
| Encoder only, uniform | **0.999884** | **0.000** |
| Early + middle encoder | 0.999881 | **0.000** |
| Leave out early encoder | 0.998469 | 0.005 |
| Leave out middle encoder | 0.987777 | 0.052 |
| Leave out late encoder | 0.999569 | 0.002 |
| Leave out late decoder | **0.999884** | **0.000** |

The synthetic-noise criterion produces equal weights because every layer
perfectly separates clean validation images from Gaussian and uniform noise.
It is therefore a saturated calibration proxy, not evidence that every layer
is equally useful on real OOD data. Excluding the decoder improves the uniform
mean, and excluding the middle encoder causes the largest degradation. The best
single detector remains middle-encoder magnitude-only CORES (AUROC 0.999943,
FPR95 0.000), which is also simpler than multi-layer inference.

| Calibration samples | Mean AUROC | AUROC std | Maximum FPR95 |
|---:|---:|---:|---:|
| 16 | 0.999943 | 0.00000167 | 0.000 |
| 32 | 0.999943 | 0.00000108 | 0.000 |
| 64 | 0.999943 | 0.00000068 | 0.000 |
| 128 | 0.999944 | 0.00000068 | 0.000 |
| 256 | 0.999943 | 0.00000000 | 0.000 |

Across five seeds per calibration size, FPR95 remains zero in all 25 trials.
The detector is therefore insensitive to calibration seed and sample count for
this broad cross-dataset shift, including when only 16 ID samples are used.

## 6. Discussion

The very large NYU/KITTI separation is both a strength and a limitation. It
demonstrates that intermediate depth-network responses contain a strong domain
signal, but NYU indoor scenes and KITTI road scenes represent a broad domain
shift. Synthetic corruptions provide a more difficult complementary test:
Gaussian noise yields AUROC 0.94273 but FPR95 0.388, showing that near-OOD
detection is not solved by the almost perfect cross-dataset result.

Further limitations include the single depth architecture, a single trained
checkpoint, the absence of semantic class kernels in dense regression, and the
use of synthetic rather than naturally occurring near-OOD datasets.

## 7. Conclusion

Intermediate convolutional responses from a depth estimator provide a very
strong signal for distinguishing NYU from KITTI. The most effective and
efficient configuration is the magnitude component from the middle encoder,
which reaches AUROC 0.999943 and FPR95 0.000. Multi-layer aggregation does not
improve this result, primarily because weaker late and decoder responses dilute
the useful middle-layer signal. Controls against RGB statistics, pretraining,
random features, threshold protocols, calibration seeds, and calibration sizes
support the robustness of the conclusion. At the same time, performance on
Gaussian near-OOD corruption and the large indoor/outdoor domain gap show that
near-distribution detection remains the more challenging direction for future
work.

## References

TODO: add complete bibliographic entries for CORES, FastDepth, MobileNetV2, NYU
Depth v2, KITTI, and the metric/protocol sources used by the project.
