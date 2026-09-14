# KFS⁴LAM

[Reconstruction Release](../../releases/tag/reconstructions-v1) · [Installation & Usage](USAGE.md)

![Project Page — coming soon](assets/project-page-pending.svg)

Fold-aware Gaussian SLAM with adaptive keyframe observation selection, evaluated with SplaTAM and EndoGSLAM.

![Pipeline](assets/pipeline.png)

## Main results

Sequence means over 14 C3VD and 11 C3VD-v2 sequences. Pose translation errors are in mm and rotation errors in degrees. Rendering follows each backbone’s evaluation protocol; depth values retain the source evaluator’s scale. [LaTeX](results/main_results.tex) · [Per-sequence results](results/main_per_sequence.json)

### C3VD

| Method | m ATE ↓ | m ARE ↓ | RPE-t ↓ | RPE-r ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Depth RMSE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GaussianSLAM | 4.0885 | 77.3899 | 0.3392 | 0.1930 | 25.7629 | 0.7498 | 0.5148 | 6.5036 |
| Endo-2DTAM | 1.5718 | 47.2588 | 0.1321 | 0.6578 | 20.2745 | 0.7867 | 0.3603 | 9.0555 |
| MonoGS | 2.7190 | 63.4965 | 0.3791 | 0.5640 | 29.6784 | 0.8482 | 0.3824 | 1430.0773 |
| EndoGSLAM | 0.5377 | 10.7657 | 0.0888 | 0.1617 | 21.1734 | 0.7319 | 0.3111 | 2.6904 |
| KFS⁴LAM (EndoGSLAM) | 0.5276 | 12.1956 | 0.0870 | 0.1605 | 21.7622 | 0.7379 | 0.3013 | 2.5875 |
| SplaTAM | 0.4544 | 10.3404 | 0.0625 | 0.1464 | 25.6437 | 0.8776 | 0.2416 | 1.1509 |
| KFS⁴LAM (SplaTAM) | 0.4476 | 9.3457 | 0.0624 | 0.1418 | 26.6164 | 0.8835 | 0.2206 | 1.0720 |

### C3VD-v2

| Method | m ATE ↓ | m ARE ↓ | RPE-t ↓ | RPE-r ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Depth RMSE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GaussianSLAM | 10.9947 | 104.4292 | 0.7494 | 0.4394 | 22.9059 | 0.6604 | 0.6127 | 10.1778 |
| Endo-2DTAM | 10.9641 | 93.3534 | 0.4446 | 2.4587 | 18.3417 | 0.6850 | 0.4917 | 14.5562 |
| MonoGS | 7.8102 | 69.8136 | 0.5809 | 1.6017 | 25.7714 | 0.7740 | 0.6115 | 1071.4941 |
| EndoGSLAM | 7.5037 | 60.1639 | 0.4411 | 1.2234 | 14.6158 | 0.5615 | 0.6484 | 14.8925 |
| KFS⁴LAM (EndoGSLAM) | 7.6249 | 56.0092 | 0.4277 | 0.9274 | 15.3012 | 0.5977 | 0.6082 | 11.9277 |
| SplaTAM | 7.6836 | 59.4920 | 0.3103 | 1.3969 | 20.1354 | 0.7053 | 0.4257 | 4.9467 |
| KFS⁴LAM (SplaTAM) | 7.2429 | 44.8062 | 0.3764 | 1.1134 | 20.9113 | 0.7163 | 0.4332 | 3.0472 |

## Reconstruction replay

Download a sequence archive from the Release, extract it, and run after [installation](USAGE.md):

```bash
python render_checkpoint.py --backbone splatam --checkpoint params.npz --frames all --output rendered
```

Use `--backbone endogslam` for EndoGSLAM. The project page with figures and synchronized camera/trajectory videos is coming soon.

## Code & ablations

- `final/splatam/`, `final/endogslam/`: backbone implementations and configurations.
- `modules/`: keyframe selection and fold supervision.
- `ablations/`: [component settings](ablations/README.md); [component results](results/component_results.tex).
- `docs/`: project-page source, figures and videos.

The released SplaTAM parameters include temporal deformation; the historical EndoGSLAM parameters do not. See the [checkpoint audit](CHECKPOINT_AUDIT.md) and [third-party notices](THIRD_PARTY.md).
