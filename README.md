# KFS⁴LAM: Adaptive KeyFrame Selection with Structure-Aware Mapping for Endoscopic Gaussian Splatting SLAM

KFS⁴LAM combines motion-driven keyframe selection, appearance-based redundancy suppression, fold-boundary supervision, and restricted temporal deformation for endoscopic Gaussian splatting SLAM.

[Project page](https://anonymous4submission123.github.io/KFS4LAM_Submission/) · [Reconstruction downloads](https://github.com/Anonymous4Submission123/KFS4LAM_Submission/releases/tag/reconstructions-v1) · [Installation and usage](USAGE.md)

## Code

The repository contains implementations for two Gaussian SLAM backbones together with the shared KFS⁴LAM modules:

- `final/splatam/`: SplaTAM implementation and experiment configurations.
- `final/endogslam/`: EndoGSLAM implementation and experiment configurations.
- `modules/`: adaptive keyframe selection, fold-boundary supervision, and restricted temporal deformation.
- `ablations/`: component configurations used in the ablation study.
- `render_checkpoint.py`: render RGB, depth, and camera trajectories from a saved reconstruction.
- `run.py`: entry point for running the supplied configurations.

Install the dependencies and render a downloaded reconstruction with:

```bash
pip install -r requirements.txt
python render_checkpoint.py \
  --backbone splatam \
  --checkpoint params.npz \
  --frames all \
  --output rendered
```

Instructions for new SLAM runs, checkpoint rendering, and evaluation are provided in [USAGE.md](USAGE.md).

## Qualitative results

### Method overview

![KFS⁴LAM pipeline](assets/pipeline.png)

The adaptive observation policy proposes keyframes from accumulated camera motion and suppresses visually redundant observations. Selected frames update a time-conditioned Gaussian map using fold-boundary distance supervision and fold-conditioned deformation regularization.

### RGB and depth reconstruction

![RGB and depth reconstruction compared with GaussianSLAM, Endo-2DTAM, MonoGS, EndoGSLAM, and SplaTAM](assets/render_comparison.png)

The figure compares RGB and depth reconstruction on `trans_t2_b` from C3VD and `c1_rectum_t2_v1` from C3VD-v2. The full-resolution image is available [here](assets/render_comparison.png).

### Videos

| Sequence | Setting | Video |
|---|---|---|
| C3VD-v2: `c2_sigmoid_t2_v1` | Main qualitative comparison | [Watch or download](https://anonymous4submission123.github.io/KFS4LAM_Submission/assets/c2_sigmoid_t2_v1__comparison_rgbd_surface.mp4) |
| C3VD: `trans_t2_b` | Main qualitative comparison | [Watch or download](https://anonymous4submission123.github.io/KFS4LAM_Submission/assets/trans_t2_b__comparison_rgbd_surface.mp4) |
| C3VD: `trans_t2_c` | EndoGSLAM implementation | [Watch or download](https://anonymous4submission123.github.io/KFS4LAM_Submission/assets/trans_t2_c__comparison_rgbd_surface.mp4) |

Additional render, mesh, and keyframe-trajectory comparisons are available on the [project page](https://anonymous4submission123.github.io/KFS4LAM_Submission/).

## Reconstruction downloads

The accompanying [reconstruction release](https://github.com/Anonymous4Submission123/KFS4LAM_Submission/releases/tag/reconstructions-v1) provides saved KFS⁴LAM reconstructions for the C3VD and C3VD-v2 evaluation sequences. Each sequence archive contains the Gaussian parameters, temporal deformation coefficients, estimated camera poses, intrinsics, keyframe indices, and configuration required by `render_checkpoint.py`.

## Datasets and acknowledgements

- **C3VD** — Bobrow et al., *Colonoscopy 3D Video Dataset with Paired Depth from 2D–3D Registration*, Medical Image Analysis, 2023. [Dataset](https://durrlab.github.io/C3VD/) · [Paper](https://arxiv.org/abs/2206.08903)
- **C3VD-v2** — Golhar et al., *C3VDv2: Colonoscopy 3D Video Dataset with Enhanced Realism*, Scientific Data, 2026. [Dataset and citation](https://durrlab.github.io/C3VDv2/)
- **SplaTAM** — Keetha et al., *SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM*, CVPR 2024. [Paper](https://arxiv.org/abs/2312.02126) · [Code](https://github.com/spla-tam/SplaTAM)
- **EndoGSLAM** — *Real-Time Dense Reconstruction and Tracking in Endoscopic Surgeries using Gaussian Splatting*, MICCAI 2024. [Paper](https://arxiv.org/abs/2403.15124) · [Code](https://github.com/Loping151/EndoGSLAM)

Third-party licenses and implementation attribution are listed in [THIRD_PARTY.md](THIRD_PARTY.md).
