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
