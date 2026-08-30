# Experimental results

## FastDepth baseline

The best checkpoint was selected at epoch 19 using NYU validation L1.

| Split | L1 | RMSE | AbsRel | delta1 | delta2 | delta3 |
|---|---:|---:|---:|---:|---:|---:|
| NYU validation | 0.1676 | 0.2778 | 0.0701 | 0.9530 | 0.9897 | 0.9971 |
| NYU official test | 0.5071 | 0.7901 | 0.2016 | 0.7181 | 0.9171 | 0.9698 |

## CORES-MDE all-kernel baseline

Thresholds and layer normalization were estimated only on NYU validation.
NYU official test is ID and KITTI is OOD. A higher CORES score means ID.

| Configuration | AUROC | FPR95 |
|---|---:|---:|
| Encoder early | 0.9897 | 0.044 |
| Encoder middle | **0.9999** | **0.001** |
| Encoder late | 0.9028 | 0.466 |
| Decoder late | 0.5144 | 0.798 |
| Multi-layer mean | 0.9997 | **0.001** |

The middle encoder response is the strongest individual OOD indicator. The
decoder response is close to random, suggesting that distribution information
is substantially altered by task-specific depth reconstruction.

## CORES component and response-selection ablation

| Layer | Variant | AUROC | FPR95 |
|---|---|---:|---:|
| Encoder middle | Magnitude only | **0.99994** | **0.000** |
| Encoder middle | Full CORES | 0.99990 | 0.001 |
| Encoder middle | Positive only | 0.99948 | 0.002 |
| Encoder middle | Response-selected 20% | 0.99922 | 0.005 |
| Encoder middle | Negative only | 0.99862 | 0.007 |
| Encoder early | Negative only | 0.99769 | 0.013 |
| Encoder late | Full CORES | 0.90275 | 0.466 |
| Decoder late | Response-selected 20% | 0.71350 | 0.475 |
| Decoder late | Full CORES | 0.51443 | 0.798 |

For the middle encoder, response magnitude alone is sufficient and frequency
degrades FPR95 substantially. Per-sample extreme-response selection does not
improve the already near-perfect middle-layer result, although it makes the
otherwise weak decoder response more informative.

## Robustness and RGB-statistics baseline

Confidence intervals are stratified 95% percentile intervals from 500 bootstrap
replicates.

| Configuration | AUROC (95% CI) | FPR95 (95% CI) |
|---|---:|---:|
| CORES middle magnitude | **0.99994** [0.99987, 1.00000] | **0.000** [0.000, 0.000] |
| CORES middle full | 0.99990 [0.99976, 0.99999] | 0.001 [0.000, 0.003] |
| CORES multi-layer | 0.99974 [0.99948, 0.99992] | 0.001 [0.000, 0.004] |
| RGB Mahalanobis | 0.82283 [0.80352, 0.84269] | 0.759 [0.62548, 0.85052] |

Simple image colour and contrast statistics detect part of the indoor/outdoor
domain gap, but they are far below the middle-layer convolutional response.
This supports the claim that CORES captures substantially more informative
distribution evidence than trivial RGB appearance statistics.

## Effect of depth training

| Model state | AUROC (95% CI) | FPR95 (95% CI) |
|---|---:|---:|
| FastDepth trained on NYU | **0.99994** [0.99987, 0.99999] | **0.000** [0.000, 0.000] |
| ImageNet encoder, no depth training | 0.91886 [0.90715, 0.93182] | 0.324 [0.28248, 0.41153] |
| Fully random encoder | 0.45062 [0.41964, 0.48209] | 1.000 [0.99748, 1.000] |

ImageNet pretraining provides a useful domain signal, but NYU depth training is
responsible for most of the final separation. Random convolutional responses do
not distinguish NYU from KITTI.

## Synthetic near-OOD corruptions

| NYU corruption | AUROC (95% CI) | FPR95 (95% CI) |
|---|---:|---:|
| Brightness x0.50 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| Average blur 7x7 | 1.000 [0.99999, 1.000] | 0.000 [0.000, 0.000] |
| Gaussian noise sigma 0.10 | 0.94273 [0.92834, 0.95369] | 0.388 [0.31804, 0.47328] |

CORES reacts strongly to blur and illumination shift. Gaussian noise is a more
difficult near-OOD case: ranking remains good, but accepting 95% of clean NYU
requires a high false-positive rate.

## Threshold-calibration protocol

To test whether the near-perfect NYU/KITTI result depends on thresholds fitted
only from real ID validation samples, the middle-encoder magnitude score was
also calibrated with synthetic Gaussian and uniform noise, following the
calibration principle used in the CORES paper. KITTI was not used to choose the
thresholds or grid configuration.

| Calibration | tau positive | tau negative | AUROC | FPR95 |
|---|---:|---:|---:|---:|
| NYU validation median | 6.7341 | -5.2857 | **0.99994** | **0.000** |
| Gaussian/uniform tuned | 4.1906 | -6.6864 | 0.99868 | 0.007 |

The paper-style calibration loses only 0.00126 AUROC and 0.007 FPR95. This
supports the conclusion that the NYU/KITTI separation is robust to the
threshold-selection protocol, rather than an artefact of tuning on KITTI.
The synthetic grid itself separated clean NYU from synthetic noise perfectly
for every tested quantile pair, so the selected pair should not be interpreted
as uniquely optimal.

## Multi-layer aggregation

Every layer score was standardized with NYU validation statistics. Aggregation
weights and configurations were defined without using KITTI.

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

The synthetic weighting rule assigns 0.25 to every layer because all four
layers perfectly separate clean validation images from synthetic Gaussian and
uniform noise. This calibration proxy is therefore saturated and cannot rank
layer reliability. Removing the weak decoder improves the uniform mean, while
removing the middle encoder causes the largest degradation. Nevertheless, no
multi-layer configuration exceeds the middle-encoder magnitude-only score
(AUROC 0.999943, FPR95 0.000), so the simpler single-layer configuration remains
the recommended final detector.

## Calibration stability

Middle-encoder magnitude thresholds were recalibrated with five seeds at each
of five calibration-set sizes. All 25 trials retain FPR95 0.000.

| Calibration samples | Mean AUROC | AUROC std | Maximum FPR95 |
|---:|---:|---:|---:|
| 16 | 0.999943 | 0.00000167 | 0.000 |
| 32 | 0.999943 | 0.00000108 | 0.000 |
| 64 | 0.999943 | 0.00000068 | 0.000 |
| 128 | 0.999944 | 0.00000068 | 0.000 |
| 256 | 0.999943 | 0.00000000 | 0.000 |

The conclusion is insensitive to both seed and calibration size in this broad
NYU/KITTI shift. Even 16 calibration images are sufficient for effectively the
same ranking, although this should not be generalized to subtler OOD shifts.
