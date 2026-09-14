# Usage

## Environment

The experiments use Python 3.10, CUDA-enabled PyTorch, and NVIDIA V100 32GB GPUs. Install a PyTorch/CUDA combination supported by the local driver before installing the requirements. The rasterizer must be compiled against that same environment.

```bash
python -m pip install -r requirements.txt
python -m pip install 'git+https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth.git@cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110'
python run.py --list
```

For the optional MonoGS selection ablation, compile its separate renderer before use:

```bash
(cd third_party/monogs_rasterizer && python setup.py build_ext --inplace)
```

Do not replace the main depth-enabled renderer with the MonoGS renderer; the adaptation loads the latter under a separate module name. Upstream attribution and notices are in [THIRD_PARTY.md](THIRD_PARTY.md).

## Input data

RGB-D input is required: depth is an observed input to initialization, tracking and mapping. Rendered depth is a map prediction, not a replacement for input depth. Use registered, undistorted RGB/depth and spatially aligned continuous fold responses.

```text
C3VD_ROOT/sequence/color/0000.png
C3VD_ROOT/sequence/depth/0000.tiff
C3VD_V2_ROOT/sequence/rgb/0000.png
C3VD_V2_ROOT/sequence/depth/0000_depth.tiff
FOLD_ROOT/sequence/0000_color.npy
```

Keep the original pose files expected by each dataset loader beside these images. The paper uses the camera calibration in `final/<backbone>/configs/data/`; depth uses the dataset scale 655.35 integer units per millimetre. See the dataset loaders for the exact source pose convention and filename handling.

Fold inputs are continuous arrays in [0,1]; the published runs use channel 0 of the four-channel float32 original-Docker outputs produced with the reproduced predictor. Fold inference is external to this repository. Resize to the mapping resolution with the exported nearest-neighbour convention; RGB/fold distortion and cropping must agree. No binary threshold is applied to the stored Gaussian fold attribute. Boundary-support thresholding is used only to construct the target distance transform.

## Run the final methods

```bash
python run.py --backbone splatam --dataset c3vd_v2 \
  --sequence c1_rectum_t2_v1 --dataset-root /path/to/C3VD_V2_ROOT \
  --fold-root /path/to/FOLD_ROOT --output outputs/splatam_rectum --gpu 0

python run.py --backbone endogslam --dataset c3vd_v2 \
  --sequence c1_rectum_t2_v1 --dataset-root /path/to/C3VD_V2_ROOT \
  --fold-root /path/to/FOLD_ROOT --output outputs/endogslam_rectum --gpu 0
```

Use `--dry-run` to inspect the resolved configuration. `--frames`, `--mapping-iters` and `--tracking-iters` are smoke-test overrides and do not reproduce paper numbers. `--cpu-offload` offloads autograd saved tensors without changing the loss definition; it was needed for one fixed-keyframe deformation control due to GPU memory exhaustion.

```bash
python run.py --backbone splatam --variant structure_guided_deformation \
  --dataset c3vd_v2 --sequence c1_rectum_t2_v1 \
  --dataset-root /path/to/C3VD_V2_ROOT --fold-root /path/to/FOLD_ROOT \
  --output outputs/fixed_keyframe_control --gpu 0
```

See [the ablation matrix](ablations/README.md) for all paper names. The final Full configuration is selected with `--variant full` (the default).

## Implementation conventions

- Poses are **world-to-camera**. The implemented accumulated-motion selector uses changes in the stored world-to-camera translation component as a **motion proxy**, not world-frame camera travel distance. Camera centers would instead be `-R.T @ t`.
- SplaTAM mapping runs at every input frame; retained-keyframe selection happens after mapping. Mapping replays current and retained historical observations.
- Fold supervision uses the one-sided mean product of rendered Sobel boundary magnitude and stop-gradient target Boundary-DT, capped at 10 pixels. The branch detaches stored fold attributes and the mapping pose.
- Restricted temporal deformation changes Gaussian means with six coefficients and the basis `[u, sin(pi*u)]`. Larger fold response reduces the displacement penalty through `1 - 0.5*f`; it does not enforce greater rigidity on folds.
- Tracking freezes the map and temporal coefficients and optimizes pose using the backbone RGB/depth objective.
- DynMap uses a 25–75 iteration budget. Equal budget rules do not imply equal realized iteration counts across different keyframe histories.

## Evaluation

```bash
python evaluate.py --run outputs/splatam_rectum --backbone splatam --dataset c3vd_v2 --gpu 0
```

Use `paper_metrics.json` for the paper protocol. SplaTAM's legacy native console depth labels are not the paper depth-RMSE computation; this evaluator rerenders the saved evaluation frames and computes true pixelwise RMSE in millimetres. Pose metrics use Sim(3) alignment and relative errors on adjacent evaluated poses. EndoGSLAM retains its native held-out rendering protocol and evaluates saved RGB/depth against dataset observations; it is not the same frame sampling as SplaTAM. This release preserves those evaluated protocols rather than silently standardizing them.


## Checkpoint replay

```bash
python render_checkpoint.py --backbone splatam --checkpoint params.npz --frames all --output rendered
```

Camera-and-trajectory videos can be composed with `tools/compose_camera_replay.py`.

## Checkpoint scope

The Release contains KFS⁴LAM reconstruction parameters, including Gaussian attributes, temporal deformation coefficients, camera poses, intrinsics and keyframe indices.
