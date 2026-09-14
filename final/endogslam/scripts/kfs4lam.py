import argparse
import csv
import os
import shutil
import sys
import time
from collections import OrderedDict
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)


import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

from datasets.gradslam_datasets import (
    load_dataset_config,
    EndoSLAMDataset,
    C3VDDataset
)
from utils.common_utils import seed_everything, save_params_ckpt, save_params, save_means3D
from utils.eval_helpers import report_progress, eval_save
from utils.keyframe_selection import (
    keyframe_selection_overlap,
    keyframe_selection_distance,
    keyframe_distance_probs,
    keyframe_selection_scorenet,
    keyframe_sampling_scorenet,
    keyframe_sampling_scorenet_with_current,
    keyframe_sampling_mixed_with_current,
)
from utils.scorenet_selection import (
    load_scorenet_wrapper,
    score_candidates,
    estimate_w2c_from_outputs,
)
from utils.pose_graph_scorenet import build_weighted_keyframe_edges, optimize_pose_graph_local
from utils.recon_helpers import setup_camera, energy_mask
from utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
    transform_to_frame, temporal_deformation_delta, l1_loss_v1, matrix_to_quaternion
)
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians, densify
from utils.vis_utils import plot_video
from utils.time_helper import Timer

from diff_gaussian_rasterization import GaussianRasterizer as Renderer


_KFG_PAIR_PROXY_FN = None
_KFG_PAIR_PROXY_ROOT = None
_FOLD_SEG_MISSING_WARNED = set()
_FOLD_BOUNDARY_DT_CACHE = OrderedDict()
_FOLD_BOUNDARY_DT_CACHE_MAX = 64
_FOLD_BINARY_DT_CACHE = OrderedDict()
_FOLD_BINARY_DT_CACHE_MAX = 64
_FOLD_TARGET_DT_CACHE = OrderedDict()
_FOLD_TARGET_DT_CACHE_MAX = 64


def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["endoslam_unity"]:
        return EndoSLAMDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["c3vd"]:
        return C3VDDataset(config_dict, basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


def get_pointcloud(color, depth, intrinsics, w2c, transform_pts=True,
                   mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective",
                   fold_seg=None):
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(),
                                    torch.arange(height).cuda().float(),
                                    indexing='xy')
    xx = (x_grid - CX)/FX
    yy = (y_grid - CY)/FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    if transform_pts:
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
    else:
        pts = pts_cam

    # Compute mean squared distance for initializing the scale of the Gaussians
    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
            scale_gaussian = depth_z / ((FX + FY)/2)
            mean3_sq_dist = scale_gaussian**2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
    
    # Colorize point cloud
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)
    point_cols = [pts, cols]
    if fold_seg is not None:
        fold_cols = torch.permute(fold_seg, (1, 2, 0)).reshape(-1, 1)
        point_cols.append(fold_cols)
    point_cld = torch.cat(point_cols, -1)
    # Image.fromarray(np.uint8((torch.permute(color, (1, 2, 0)) * mask.reshape(320, 320, 1)).detach().cpu().numpy()*255), 'RGB').save('gaussian.png')

    # Select points based on mask
    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    else:
        return point_cld


def initialize_params(init_pt_cld, num_frames, mean3_sq_dist, use_simplification=True):
    num_pts = init_pt_cld.shape[0]
    means3D = init_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 3]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    params = {
        'means3D': means3D,
        'rgb_colors': init_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1 if use_simplification else 3)),
    }
    if init_pt_cld.shape[1] > 6:
        params['fold_colors'] = init_pt_cld[:, 6:7]
    if os.environ.get("MPKFS_TEMPORAL_DEFORMATION_ENABLE", "0") == "1":
        num_basis = max(int(os.environ.get("MPKFS_TEMPORAL_DEFORMATION_BASIS", "2")), 1)
        params['deform_coeffs'] = torch.zeros(
            (num_pts, 3 * num_basis), dtype=torch.float, device="cuda"
        )
    if not use_simplification:
        params['feature_rest'] = torch.zeros(num_pts, 45) # set SH degree 3 fixed

    # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params['cam_unnorm_rots'] = cam_rots
    params['cam_trans'] = np.zeros((1, 3, num_frames))

    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'timestep': torch.zeros(params['means3D'].shape[0]).cuda().float()}

    return params, variables


def initialize_optimizer(params, lrs_dict):
    lrs = lrs_dict
    param_groups = [{'params': [v], 'name': k, 'lr': lrs.get(k, 0.0)} for k, v in params.items() if k != 'feature_rest']
    if 'feature_rest' in params:
        param_groups.append({'params': [params['feature_rest']], 'name': 'feature_rest', 'lr': lrs['rgb_colors'] / 20.0})
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def _dataset_frame_stem(dataset, index):
    try:
        return os.path.splitext(os.path.basename(dataset.color_paths[index]))[0]
    except Exception:
        return f"{int(index):04d}_color"


def _dataset_source_index(dataset, index):
    try:
        return int(dataset.retained_inds[index].item())
    except Exception:
        return int(index)


def _load_fold_seg(dataset, index, target_hw, cfg, device):
    if not bool(cfg.get("enable", False)):
        return None

    root = str(cfg.get("root", "")).strip()
    if not root:
        return None

    sequence = os.path.basename(getattr(dataset, "input_folder", "")).strip()
    stem = _dataset_frame_stem(dataset, index)
    src_idx = _dataset_source_index(dataset, index)
    candidates = [
        os.path.join(root, sequence, f"{stem}.npy"),
        os.path.join(root, sequence, f"{src_idx:04d}_color.npy"),
        os.path.join(root, sequence, f"{src_idx:05d}_color.npy"),
        os.path.join(root, sequence, f"{src_idx:06d}_color.npy"),
    ]
    pred_path = next((path for path in candidates if os.path.isfile(path)), None)
    if pred_path is None:
        warn_key = (sequence, stem)
        if warn_key not in _FOLD_SEG_MISSING_WARNED:
            print(f"[FoldSegLoss] missing prediction for {sequence}/{stem}; using all-background fold channel")
            _FOLD_SEG_MISSING_WARNED.add(warn_key)
        return torch.zeros((1, int(target_hw[0]), int(target_hw[1])), device=device, dtype=torch.float32)

    arr = np.load(pred_path)
    channel = int(cfg.get("channel", 0))
    if arr.ndim == 3:
        if arr.shape[-1] <= 16:
            arr = arr[..., channel]
        elif arr.shape[0] <= 16:
            arr = arr[channel, ...]
        else:
            arr = arr[..., channel]
    elif arr.ndim != 2:
        raise ValueError(f"[FoldSegLoss] expected 2D or 3D prediction, got {arr.shape} from {pred_path}")

    fold = np.asarray(arr, dtype=np.float32)
    if bool(cfg.get("probability_clip", True)):
        fold = np.clip(fold, 0.0, 1.0)
    low = float(cfg.get("low", 0.0))
    high = float(cfg.get("high", 1.0))
    fold = low + fold * (high - low)
    fold = torch.from_numpy(fold).to(device=device, dtype=torch.float32)[None, None]
    if tuple(fold.shape[-2:]) != (int(target_hw[0]), int(target_hw[1])):
        fold = F.interpolate(fold, size=(int(target_hw[0]), int(target_hw[1])), mode="nearest")
    return fold[0].contiguous()


def _transformed_params2foldrendervar(params, transformed_pts):
    fold_colors = params['fold_colors'].detach().clamp(0.0, 1.0).repeat(1, 3)
    rendervar = {
        'means3D': transformed_pts,
        'colors_precomp': fold_colors,
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    if params['log_scales'].shape[1] == 1:
        rendervar['scales'] = torch.exp(torch.tile(params['log_scales'], (1, 3)))
    else:
        rendervar['scales'] = torch.exp(params['log_scales'])
    return rendervar


def _fold_boundary_map(x):
    x = x[:1].unsqueeze(0)
    dtype = x.dtype
    device = x.device
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3) / 8.0
    sobel_y = sobel_x.transpose(-1, -2)
    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)[0]


def _fold_boundary_distance_target(fold_target, dmax: float = 10.0):
    """Distance-to-target-boundary map for boundary-distance fold supervision."""
    global _FOLD_BOUNDARY_DT_CACHE
    target = fold_target[:1].detach().float().contiguous()
    key = (
        int(target.data_ptr()),
        tuple(int(v) for v in target.shape),
        str(target.device),
        float(dmax),
    )
    cached = _FOLD_BOUNDARY_DT_CACHE.get(key)
    if cached is not None:
        _FOLD_BOUNDARY_DT_CACHE.move_to_end(key)
        return cached

    boundary = _fold_boundary_map(target).detach()
    q90 = torch.quantile(boundary.flatten(), 0.90).clamp_min(1e-6)
    edge = (boundary / q90) > 0.20
    if not bool(edge.any()):
        edge = boundary >= torch.quantile(boundary.flatten(), 0.99)
    edge_np = edge[0].detach().cpu().numpy().astype(bool)
    if not edge_np.any():
        dist_np = np.ones(edge_np.shape, dtype=np.float32)
    else:
        dist_np = distance_transform_edt(~edge_np).astype(np.float32)
        dist_np = np.clip(dist_np, 0.0, float(dmax)) / max(float(dmax), 1e-6)
    dist = torch.from_numpy(dist_np)[None].to(device=fold_target.device, dtype=fold_target.dtype)

    _FOLD_BOUNDARY_DT_CACHE[key] = dist
    if len(_FOLD_BOUNDARY_DT_CACHE) > _FOLD_BOUNDARY_DT_CACHE_MAX:
        _FOLD_BOUNDARY_DT_CACHE.popitem(last=False)
    return dist


def _fold_binary_ste(x, threshold: float = 0.5):
    x = x[:1].clamp(0.0, 1.0)
    hard = (x >= threshold).to(dtype=x.dtype)
    return hard + x - x.detach()


def _fold_binary_target(x, threshold: float = 0.5):
    return (x[:1].detach().float() >= threshold).to(device=x.device, dtype=x.dtype)


def _fold_target_dt_and_edge(
    fold_target,
    threshold: float = 0.5,
    dmax: float = 8.0,
    min_component_area: int = 8,
):
    """Binarize target edge probability and cache its truncated DT cost map."""
    global _FOLD_TARGET_DT_CACHE
    target = fold_target[:1].detach().float().contiguous()
    key = (
        int(target.data_ptr()),
        tuple(int(v) for v in target.shape),
        str(target.device),
        float(threshold),
        float(dmax),
        int(min_component_area),
    )
    cached = _FOLD_TARGET_DT_CACHE.get(key)
    if cached is not None:
        _FOLD_TARGET_DT_CACHE.move_to_end(key)
        return cached

    edge_np = (target[0].detach().cpu().numpy() >= float(threshold)).astype(np.uint8)
    if edge_np.any() and int(min_component_area) > 1:
        import cv2

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edge_np, connectivity=8)
        keep = np.zeros_like(edge_np, dtype=bool)
        for label_idx in range(1, num_labels):
            if int(stats[label_idx, cv2.CC_STAT_AREA]) >= int(min_component_area):
                keep |= labels == label_idx
        edge_np = keep.astype(np.uint8)

    if edge_np.any():
        dist_np = distance_transform_edt(edge_np == 0).astype(np.float32)
        dist_np = np.clip(dist_np, 0.0, float(dmax)) / max(float(dmax), 1e-6)
    else:
        dist_np = np.ones(edge_np.shape, dtype=np.float32)
    edge = torch.from_numpy(edge_np.astype(np.float32))[None].to(device=fold_target.device, dtype=fold_target.dtype)
    dist = torch.from_numpy(dist_np)[None].to(device=fold_target.device, dtype=fold_target.dtype)
    result = (edge, dist)

    _FOLD_TARGET_DT_CACHE[key] = result
    if len(_FOLD_TARGET_DT_CACHE) > _FOLD_TARGET_DT_CACHE_MAX:
        _FOLD_TARGET_DT_CACHE.popitem(last=False)
    return result


def _fold_target_dt_loss(
    fold_im,
    fold_target,
    reduction,
    mask=None,
    truncation_px: float = 8.0,
    coverage_radius: int = 3,
    coverage_weight: float = 0.5,
):
    pred = fold_im[:1].clamp(0.0, 1.0)
    target_edge, target_dt = _fold_target_dt_and_edge(fold_target, dmax=truncation_px)
    if mask is not None:
        if mask.ndim == 2:
            mask_ = mask[None].to(device=pred.device, dtype=pred.dtype)
        elif mask.ndim == 3:
            mask_ = mask[:1].to(device=pred.device, dtype=pred.dtype)
        else:
            mask_ = mask.to(device=pred.device, dtype=pred.dtype)
        pred = pred * mask_
        target_edge = target_edge * mask_
        normalizer = mask_.sum().clamp_min(1.0)
    else:
        normalizer = torch.as_tensor(float(pred.numel()), device=pred.device, dtype=pred.dtype)

    eps = torch.as_tensor(1e-6, device=pred.device, dtype=pred.dtype)
    pred_to_target = (pred * target_dt).sum() / (pred.sum() + eps)
    pred_nearby = F.max_pool2d(
        pred[None],
        kernel_size=2 * int(coverage_radius) + 1,
        stride=1,
        padding=int(coverage_radius),
    )[0]
    target_to_pred = (target_edge * (1.0 - pred_nearby)).sum() / (target_edge.sum() + eps)
    loss = pred_to_target + float(coverage_weight) * target_to_pred
    if reduction == "sum":
        return loss * normalizer
    if reduction == "mean":
        return loss
    raise ValueError(f"[FoldSegLoss] unknown reduction={reduction!r}")


def _dynamic_mapping_num_iters(config, time_idx, num_keyframes):
    dyn = config.get("dynamic_mapping_iters", {})
    if not bool(dyn.get("enable", False)):
        return int(config["mapping"]["num_iters"]), 1.0, int(config["mapping"]["num_iters"])

    base_iters = int(config["mapping"]["num_iters"])
    base_gap = max(int(dyn.get("base_gap", config.get("keyframe_every", 8))), 1)
    # Mapping samples from historical keyframes plus the current frame.  The
    # fair fixed-gap baseline pool is therefore expected historical KFs + curr.
    expected_history_kf = max(1.0, float(np.ceil((int(time_idx) + 1) / float(base_gap))))
    expected_units = expected_history_kf + 1.0
    active_units = max(1.0, float(int(num_keyframes) + 1))
    ratio = max(1.0, active_units / expected_units)
    scale = min(float(dyn.get("max_scale", 3.0)), ratio ** float(dyn.get("power", 1.0)))
    scaled_iters = int(np.ceil(base_iters * scale))
    min_iters = int(dyn.get("min_iters", base_iters))
    max_iters = int(dyn.get("max_iters", base_iters * 3))
    return int(np.clip(scaled_iters, min_iters, max_iters)), float(scale), base_iters


def _loss_to_float(value):
    try:
        if torch.is_tensor(value):
            return float(value.detach().cpu().item())
        return float(value)
    except Exception:
        return np.nan


def _open_iter_loss_logger(eval_dir, config):
    if not bool(config.get("iter_loss_logging", {}).get("enable", False)):
        return None, None
    path = config.get("iter_loss_logging", {}).get("path", "")
    if not path:
        path = os.path.join(eval_dir, "iter_losses.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    handle = open(path, "w", newline="", buffering=1)
    fields = [
        "phase",
        "online_frame",
        "loss_frame",
        "iter",
        "num_iters",
        "num_keyframes",
        "mapping_iter_scale",
        "selected_keyframe_slot",
        "selected_frame",
        "num_points",
        "loss",
        "im",
        "im_rgb",
        "fold",
        "depth",
        "def_mag",
        "def_temp",
        "def_mean",
        "kf_prior",
    ]
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    return handle, writer


def _log_iter_loss(
    writer,
    phase,
    online_frame,
    loss_frame,
    iter_idx,
    num_iters,
    losses,
    params,
    num_keyframes,
    mapping_iter_scale=1.0,
    selected_keyframe_slot="",
    selected_frame="",
):
    if writer is None:
        return
    row = {
        "phase": phase,
        "online_frame": int(online_frame),
        "loss_frame": int(loss_frame),
        "iter": int(iter_idx),
        "num_iters": int(num_iters),
        "num_keyframes": int(num_keyframes),
        "mapping_iter_scale": float(mapping_iter_scale),
        "selected_keyframe_slot": selected_keyframe_slot,
        "selected_frame": selected_frame,
        "num_points": int(params["means3D"].shape[0]) if "means3D" in params else -1,
    }
    for key in (
        "loss", "im", "im_rgb", "fold", "depth",
        "def_mag", "def_temp", "def_mean", "kf_prior",
    ):
        row[key] = _loss_to_float(losses[key]) if key in losses else np.nan
    writer.writerow(row)


def _binary_region_distance_map(binary, dmax: float = 10.0, cache: bool = False):
    """Boundary distance field like MONAI HausdorffDTLoss uses for binary masks."""
    global _FOLD_BINARY_DT_CACHE
    mask_tensor = binary[:1].detach().contiguous()
    key = None
    if cache:
        key = (
            int(mask_tensor.data_ptr()),
            tuple(int(v) for v in mask_tensor.shape),
            str(mask_tensor.device),
            float(dmax),
        )
        cached = _FOLD_BINARY_DT_CACHE.get(key)
        if cached is not None:
            _FOLD_BINARY_DT_CACHE.move_to_end(key)
            return cached

    mask_np = (mask_tensor[0].detach().cpu().numpy() > 0.5)
    if mask_np.any() and (~mask_np).any():
        dist_np = distance_transform_edt(mask_np).astype(np.float32)
        dist_np += distance_transform_edt(~mask_np).astype(np.float32)
    else:
        dist_np = np.ones(mask_np.shape, dtype=np.float32) * float(dmax)
    dist_np = np.clip(dist_np, 0.0, float(dmax)) / max(float(dmax), 1e-6)
    dist = torch.from_numpy(dist_np)[None].to(device=binary.device, dtype=binary.dtype)

    if cache and key is not None:
        _FOLD_BINARY_DT_CACHE[key] = dist
        if len(_FOLD_BINARY_DT_CACHE) > _FOLD_BINARY_DT_CACHE_MAX:
            _FOLD_BINARY_DT_CACHE.popitem(last=False)
    return dist


def _soft_erode2d(x):
    p1 = -F.max_pool2d(-x, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-x, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.minimum(p1, p2)


def _soft_dilate2d(x):
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def _soft_open2d(x):
    return _soft_dilate2d(_soft_erode2d(x))


def _soft_skeletonize2d(x, iterations: int = 10):
    x = x.clamp(0.0, 1.0)
    skel = F.relu(x - _soft_open2d(x))
    for _ in range(iterations):
        x = _soft_erode2d(x)
        opened = _soft_open2d(x)
        delta = F.relu(x - opened)
        skel = skel + F.relu(delta - skel * delta)
    return skel.clamp(0.0, 1.0)


def _binary_softdice_cldice_loss(fold_im, fold_target, reduction, mask=None):
    pred = _fold_binary_ste(fold_im)[None]
    target = _fold_binary_target(fold_target)[None]
    if mask is not None:
        if mask.ndim == 2:
            mask_ = mask[None, None].to(device=pred.device, dtype=pred.dtype)
        elif mask.ndim == 3:
            mask_ = mask[:1][None].to(device=pred.device, dtype=pred.dtype)
        else:
            mask_ = mask.to(device=pred.device, dtype=pred.dtype)
        pred = pred * mask_
        target = target * mask_
        normalizer = mask_.sum().clamp_min(1.0)
    else:
        normalizer = torch.as_tensor(float(pred.numel()), device=pred.device, dtype=pred.dtype)

    eps = 1e-6
    intersection = (pred * target).sum()
    dice = 1.0 - (2.0 * intersection + eps) / (pred.sum() + target.sum() + eps)

    pred_skel = _soft_skeletonize2d(pred)
    target_skel = _soft_skeletonize2d(target.detach())
    tprec = (pred_skel * target).sum() / (pred_skel.sum() + eps)
    tsens = (target_skel * pred).sum() / (target_skel.sum() + eps)
    cldice = 1.0 - (2.0 * tprec * tsens + eps) / (tprec + tsens + eps)
    loss = 0.5 * dice + 0.5 * cldice
    if reduction == "sum":
        return loss * normalizer
    return loss


def _fold_chamfer_loss(
    fold_im,
    fold_target,
    reduction,
    mask=None,
    direct: bool = False,
    sparse_topk: int = 0,
):
    """Soft symmetric Chamfer loss between rendered and target fold contours.

    Hard thresholded contour coordinates would make the rendered contour
    selection non-differentiable.  In boundary mode, the rendered Sobel
    boundary is used as a soft point weight field.  In direct mode, the fold
    probability itself is used as the soft point field, avoiding edge-of-edge
    behavior when the fold prediction is already line-like.
    """
    if direct:
        pred = fold_im[:1].clamp(0.0, 1.0)
        target = fold_target[:1].detach().clamp(0.0, 1.0)
    else:
        pred = _fold_boundary_map(fold_im).clamp_min(0.0)
        target = _fold_boundary_map(fold_target).detach().clamp_min(0.0)
    if mask is not None:
        if mask.ndim == 2:
            mask_ = mask[None].to(device=pred.device, dtype=pred.dtype)
        elif mask.ndim == 3:
            mask_ = mask[:1].to(device=pred.device, dtype=pred.dtype)
        else:
            mask_ = mask.to(device=pred.device, dtype=pred.dtype)
        pred = pred * mask_
        target = target * mask_
        normalizer = mask_.sum().clamp_min(1.0)
    else:
        normalizer = torch.as_tensor(float(pred.numel()), device=pred.device, dtype=pred.dtype)

    # Chamfer over a lower-resolution contour grid keeps per-iteration cost
    # bounded while preserving the large-structure contour displacement.
    max_side = 64
    h, w = pred.shape[-2:]
    scale = min(1.0, float(max_side) / float(max(h, w)))
    out_h = max(8, int(round(h * scale)))
    out_w = max(8, int(round(w * scale)))
    pred_lr = F.interpolate(pred[None], size=(out_h, out_w), mode="bilinear", align_corners=False)[0, 0]
    target_lr = F.interpolate(target[None], size=(out_h, out_w), mode="bilinear", align_corners=False)[0, 0]

    ys = torch.linspace(0.0, 1.0, out_h, device=pred.device, dtype=pred.dtype)
    xs = torch.linspace(0.0, 1.0, out_w, device=pred.device, dtype=pred.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack((yy.reshape(-1), xx.reshape(-1)), dim=1)
    pred_w = pred_lr.reshape(-1).clamp_min(0.0)
    target_w = target_lr.reshape(-1).clamp_min(0.0)

    eps = torch.as_tensor(1e-8, device=pred.device, dtype=pred.dtype)
    if float(target_w.detach().sum().cpu()) <= 1e-8:
        loss = pred_w.mean()
        return loss * normalizer if reduction == "sum" else loss
    if float(pred_w.detach().sum().cpu()) <= 1e-8:
        loss = target_w.mean()
        return loss * normalizer if reduction == "sum" else loss

    max_points = min(512, target_w.numel())
    target_k = min(max_points, int(sparse_topk)) if sparse_topk > 0 else max_points
    if target_w.numel() > target_k:
        target_vals, target_idx = torch.topk(target_w, k=target_k, largest=True, sorted=False)
    else:
        target_vals, target_idx = target_w, torch.arange(target_w.numel(), device=target_w.device)
    target_pts = coords[target_idx]
    target_prob = target_vals / target_vals.sum().clamp_min(eps)

    if sparse_topk > 0:
        pred_k = min(int(sparse_topk), pred_w.numel())
        if pred_w.numel() > pred_k:
            pred_vals, pred_idx = torch.topk(pred_w, k=pred_k, largest=True, sorted=False)
        else:
            pred_vals, pred_idx = pred_w, torch.arange(pred_w.numel(), device=pred_w.device)
        pred_pts = coords[pred_idx]
        pred_prob = pred_vals / pred_vals.sum().clamp_min(eps)
        dists = torch.cdist(pred_pts, target_pts, p=2)
    else:
        dists = torch.cdist(coords, target_pts, p=2)
        pred_prob = pred_w / pred_w.sum().clamp_min(eps)
    pred_to_target = (pred_prob * dists.min(dim=1).values).sum()

    # Soft nearest rendered contour distance for target points.  log(pred_w)
    # biases the softmin toward pixels that currently carry rendered contour.
    tau = torch.as_tensor(0.03, device=pred.device, dtype=pred.dtype)
    dists_t = dists.transpose(0, 1)
    logits = torch.log(pred_prob.clamp_min(eps))[None, :] - dists_t / tau
    soft_nn = (torch.softmax(logits, dim=1) * dists_t).sum(dim=1)
    target_to_pred = (target_prob * soft_nn).sum()

    loss = pred_to_target + target_to_pred
    if reduction == "sum":
        return loss * normalizer
    if reduction == "mean":
        return loss
    raise ValueError(f"[FoldSegLoss] unknown reduction={reduction!r}")


def _fold_channel_loss(fold_im, fold_target, loss_type, reduction, mask=None):
    if loss_type == "boundary":
        diff = torch.abs(_fold_boundary_map(fold_im) - _fold_boundary_map(fold_target))
    elif loss_type == "boundary_dt":
        # Boundary distance transform loss: rendered fold contours are penalized
        # by their distance to the target fold contour, instead of requiring
        # exact pixel-wise Sobel overlap.
        render_boundary = _fold_boundary_map(fold_im)
        target_dt = _fold_boundary_distance_target(fold_target)
        diff = render_boundary * target_dt
    elif loss_type == "binary_hausdorff_dt":
        pred = _fold_binary_ste(fold_im)
        target = _fold_binary_target(fold_target)
        pred_dt = _binary_region_distance_map((pred.detach() >= 0.5).to(dtype=pred.dtype), cache=False)
        target_dt = _binary_region_distance_map(target, cache=True)
        diff = (pred - target).pow(2) * (pred_dt.pow(2.0) + target_dt.pow(2.0))
    elif loss_type == "binary_softdice_cldice":
        return _binary_softdice_cldice_loss(fold_im, fold_target, reduction, mask)
    elif loss_type == "binary_hdt_softdice_cldice":
        pred = _fold_binary_ste(fold_im)
        target = _fold_binary_target(fold_target)
        pred_dt = _binary_region_distance_map((pred.detach() >= 0.5).to(dtype=pred.dtype), cache=False)
        target_dt = _binary_region_distance_map(target, cache=True)
        hdt_diff = (pred - target).pow(2) * (pred_dt.pow(2.0) + target_dt.pow(2.0))
        if mask is not None:
            if mask.ndim == hdt_diff.ndim:
                hdt_diff = hdt_diff[mask]
            else:
                hdt_diff = hdt_diff[:, mask]
        if reduction == "sum":
            hdt_loss = hdt_diff.sum()
        elif reduction == "mean":
            hdt_loss = hdt_diff.mean()
        else:
            raise ValueError(f"[FoldSegLoss] unknown reduction={reduction!r}")
        return hdt_loss + _binary_softdice_cldice_loss(fold_im, fold_target, reduction, mask)
    elif loss_type == "chamfer":
        return _fold_chamfer_loss(fold_im, fold_target, reduction, mask)
    elif loss_type == "chamfer_direct":
        return _fold_chamfer_loss(fold_im, fold_target, reduction, mask, direct=True)
    elif loss_type == "chamfer_direct_sparse":
        return _fold_chamfer_loss(fold_im, fold_target, reduction, mask, direct=True, sparse_topk=256)
    elif loss_type == "target_dt":
        return _fold_target_dt_loss(fold_im, fold_target, reduction, mask)
    elif loss_type == "l1":
        diff = torch.abs(fold_target - fold_im)
    else:
        raise ValueError(f"[FoldSegLoss] unknown loss_type={loss_type!r}")

    if mask is not None:
        if mask.ndim == diff.ndim:
            diff = diff[mask]
        else:
            diff = diff[:, mask]
    if reduction == "sum":
        return diff.sum()
    if reduction == "mean":
        return diff.mean()
    raise ValueError(f"[FoldSegLoss] unknown reduction={reduction!r}")


def initialize_first_timestep(dataset, num_frames, scene_radius_depth_ratio, mean_sq_dist_method, densify_dataset=None, use_simplification=True, fold_seg_cfg=None):
    # Get RGB-D Data & Camera Parameters
    color, depth, intrinsics, pose = dataset[0]

    # Process RGB-D Data
    color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
    
    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3]
    w2c = torch.linalg.inv(pose)

    # Setup Camera
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), w2c.detach().cpu().numpy(), use_simplification=use_simplification)

    if densify_dataset is not None:
        # Get Densification RGB-D Data & Camera Parameters
        color, depth, densify_intrinsics, _ = densify_dataset[0]
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
        densify_intrinsics = densify_intrinsics[:3, :3]
        densify_cam = setup_camera(color.shape[2], color.shape[1], densify_intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())
    else:
        densify_intrinsics = intrinsics
    fold_seg = _load_fold_seg(densify_dataset or dataset, 0, color.shape[1:], fold_seg_cfg or {}, color.device)

    # Get Initial Point Cloud (PyTorch CUDA Tensor)

    mask = (depth > 0) & energy_mask(color) # Mask out invalid depth values
    # Image.fromarray(np.uint8(mask[0].detach().cpu().numpy()*255), 'L').save('mask.png')
    mask = mask.reshape(-1)
    init_pt_cld, mean3_sq_dist = get_pointcloud(color, depth, densify_intrinsics, w2c, 
                                                mask=mask, compute_mean_sq_dist=True, 
                                                mean_sq_dist_method=mean_sq_dist_method,
                                                fold_seg=fold_seg)

    # Initialize Parameters
    params, variables = initialize_params(init_pt_cld, num_frames, mean3_sq_dist, use_simplification)

    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    variables['scene_radius'] = torch.max(depth)/scene_radius_depth_ratio # NOTE: change_here

    if densify_dataset is not None:
        return params, variables, intrinsics, w2c, cam, densify_intrinsics, densify_cam
    else:
        return params, variables, intrinsics, w2c, cam


def get_loss(params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss, 
             sil_thres, use_l1,ignore_outlier_depth_loss, tracking=False, 
             mapping=False, do_ba=False, plot_dir=None, visualize_tracking_loss=False, tracking_iteration=None):
    global w2cs, w2ci
    # Initialize Loss Dictionary
    losses = {}

    if tracking:
        # Get current frame Gaussians, where only the camera pose gets gradient
        transformed_pts = transform_to_frame(params, iter_time_idx, 
                                             gaussians_grad=False,
                                             camera_grad=True)
    elif mapping:
        if do_ba: # Bundle Adjustment
            # Get current frame Gaussians, where both camera pose and Gaussians get gradient
            transformed_pts = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=True)
        else:
            # Get current frame Gaussians, where only the Gaussians get gradient
            transformed_pts = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=False)
    else:
        # Get current frame Gaussians, where only the Gaussians get gradient
        transformed_pts = transform_to_frame(params, iter_time_idx,
                                             gaussians_grad=True,
                                             camera_grad=False)

    # Initialize Render Variables
    rendervar = transformed_params2rendervar(params, transformed_pts)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_pts)
    
    # Visualize the Rendered Images
    # online_render(curr_data, iter_time_idx, rendervar, dev_use_controller=False)
        
    # RGB Rendering
    rendervar['means2D'].retain_grad()
    im, radius, _ = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    variables['means2D'] = rendervar['means2D'] # Gradient only accum from colour render for densification


    # Depth & Silhouette Rendering
    depth_sil, _, _ = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    depth = depth_sil[0, :, :].unsqueeze(0)
    silhouette = depth_sil[1, :, :]
    presence_sil_mask = (silhouette > sil_thres)
    depth_sq = depth_sil[2, :, :].unsqueeze(0)
    uncertainty = depth_sq - depth**2
    uncertainty = uncertainty.detach()

    # Mask with valid depth values (accounts for outlier depth values)
    nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
    bg_mask = energy_mask(curr_data['im'])
    if ignore_outlier_depth_loss:
        depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
        mask = (depth_error < 20*depth_error.mean())
        mask = mask & (curr_data['depth'] > 0)
    else:
        mask = (curr_data['depth'] > 0)
    mask = mask & nan_mask & bg_mask
    # Mask with presence silhouette mask (accounts for empty space)
    if tracking and use_sil_for_loss:
        mask = mask & presence_sil_mask

    # Depth loss
    if use_l1:
        mask = mask.detach()
        if tracking:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
        else:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()
    # RGB Loss
    if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
        color_mask = torch.tile(mask, (3, 1, 1))
        color_mask = color_mask.detach()
        losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].sum()
    elif tracking:
        losses['im'] = torch.abs(curr_data['im'] - im).sum()
    else:
        losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
    im_rgb_loss_for_log = losses['im']
    fold_loss_for_log = None

    # Optional fold-segmentation channel loss.  This is rendered as a separate
    # grayscale Gaussian attribute so the RGB renderer/data path remains intact.
    fold_target = curr_data.get('fold_seg', None)
    fold_mapping_only = bool(curr_data.get('fold_seg_mapping_only', False))
    if (fold_target is not None and 'fold_colors' in params
            and (mapping or not fold_mapping_only)):
        fold_rendervar = _transformed_params2foldrendervar(params, transformed_pts)
        fold_im_rgb, _, _ = Renderer(raster_settings=curr_data['cam'])(**fold_rendervar)
        fold_im = fold_im_rgb[:1, :, :]
        if tuple(fold_target.shape[-2:]) != tuple(fold_im.shape[-2:]):
            fold_target = F.interpolate(
                fold_target[None].float(),
                size=fold_im.shape[-2:],
                mode="nearest",
            )[0]
        fold_target = fold_target.to(device=fold_im.device, dtype=fold_im.dtype)
        fold_weight = float(curr_data.get('fold_seg_weight', 1.0))
        fold_loss_type = str(curr_data.get('fold_seg_loss_type', 'l1')).lower()
        if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
            fold_mask = mask.detach()
            fold_loss = _fold_channel_loss(fold_im, fold_target, fold_loss_type, "sum", fold_mask)
        elif tracking:
            fold_loss = _fold_channel_loss(fold_im, fold_target, fold_loss_type, "sum")
        else:
            fold_loss = _fold_channel_loss(fold_im, fold_target, fold_loss_type, "mean")
        fold_loss_for_log = fold_weight * fold_loss
        losses['im'] = losses['im'] + fold_loss_for_log

    # Full restricted deformation: mapping-only regularization of position
    # offsets. Fold probability is a soft relaxation of the magnitude prior,
    # never a hard deformation mask.
    if mapping and 'deform_coeffs' in params:
        delta = temporal_deformation_delta(params, iter_time_idx, True)
        prev_delta = temporal_deformation_delta(
            params, max(int(iter_time_idx) - 1, 0), True
        )
        fold_prob = params.get('fold_colors')
        if fold_prob is None:
            magnitude_weight = torch.ones_like(delta[:, :1])
        else:
            alpha = float(curr_data.get('deformation_fold_soft_alpha', 0.5))
            magnitude_weight = 1.0 - alpha * fold_prob.detach().clamp(0.0, 1.0)
        losses['def_mag'] = (
            magnitude_weight * torch.sum(delta * delta, dim=-1, keepdim=True)
        ).mean()
        losses['def_temp'] = torch.sum((delta - prev_delta) ** 2, dim=-1).mean()
        losses['def_mean'] = torch.sum(delta.mean(dim=0) ** 2)

    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(weighted_losses.values())

    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    weighted_losses['loss'] = loss
    weighted_losses['im_rgb'] = im_rgb_loss_for_log * loss_weights['im']
    if fold_loss_for_log is not None:
        weighted_losses['fold'] = fold_loss_for_log * loss_weights['im']

    return loss, variables, weighted_losses


def initialize_new_params(new_pt_cld, mean3_sq_dist, use_simplification):
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 3]
    logit_opacities = torch.ones((num_pts, 1), dtype=torch.float, device="cuda") * 0.5
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1 if use_simplification else 3)),
    }
    if new_pt_cld.shape[1] > 6:
        params['fold_colors'] = new_pt_cld[:, 6:7]
    if os.environ.get("MPKFS_TEMPORAL_DEFORMATION_ENABLE", "0") == "1":
        num_basis = max(int(os.environ.get("MPKFS_TEMPORAL_DEFORMATION_BASIS", "2")), 1)
        params['deform_coeffs'] = torch.zeros(
            (num_pts, 3 * num_basis), dtype=torch.float, device="cuda"
        )
    if not use_simplification:
        params['feature_rest'] = torch.zeros(num_pts, 45) # set SH degree 3 fixed
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    return params


def add_new_gaussians(params, variables, curr_data, sil_thres, time_idx, mean_sq_dist_method, use_simplification=True):
    # Silhouette Rendering
    transformed_pts = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_pts)
    depth_sil, _, _ = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    silhouette = depth_sil[1, :, :]
    non_presence_sil_mask = (silhouette < sil_thres)
    # Check for new foreground objects by using GT depth
    gt_depth = curr_data['depth'][0, :, :]
    render_depth = depth_sil[0, :, :]
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 20*depth_error.mean())
    # Determine non-presence mask
    non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)

    # Get the new frame Gaussians based on the Silhouette
    if torch.sum(non_presence_mask) > 0:
        # Get the new pointcloud in the world frame
        curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        curr_cam_tran = params['cam_trans'][..., time_idx].detach()
        curr_w2c = torch.eye(4).cuda().float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        valid_depth_mask = (curr_data['depth'][0, :, :] > 0) & (curr_data['depth'][0, :, :] < 1e10)
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
        valid_color_mask = energy_mask(curr_data['im']).squeeze()
        non_presence_mask = non_presence_mask & valid_color_mask.reshape(-1)        
        new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'], 
                                    curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                    mean_sq_dist_method=mean_sq_dist_method,
                                    fold_seg=curr_data.get('fold_seg', None))
        new_params = initialize_new_params(new_pt_cld, mean3_sq_dist, use_simplification)
        for k, v in new_params.items():
            params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
        num_pts = params['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
        variables['denom'] = torch.zeros(num_pts, device="cuda").float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
        new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device="cuda").float()
        variables['timestep'] = torch.cat((variables['timestep'], new_timestep),dim=0)
    return params, variables


def initialize_camera_pose(params, curr_time_idx, forward_prop):
    with torch.no_grad():
        if curr_time_idx > 1 and forward_prop:
            # Initialize the camera pose for the current frame based on a constant velocity model
            # Rotation
            prev_rot1 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-1].detach())
            prev_rot2 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-2].detach())
            new_rot = F.normalize(prev_rot1 + (prev_rot1 - prev_rot2))
            params['cam_unnorm_rots'][..., curr_time_idx] = new_rot.detach()
            # Translation
            prev_tran1 = params['cam_trans'][..., curr_time_idx-1].detach()
            prev_tran2 = params['cam_trans'][..., curr_time_idx-2].detach()
            new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
            params['cam_trans'][..., curr_time_idx] = new_tran.detach()
        else:
            # Initialize the camera pose for the current frame
            params['cam_unnorm_rots'][..., curr_time_idx] = params['cam_unnorm_rots'][..., curr_time_idx-1].detach()
            params['cam_trans'][..., curr_time_idx] = params['cam_trans'][..., curr_time_idx-1].detach()
    
    return params


def load_tracking_init_poses(config, num_frames, device):
    tcfg = config.get("tracking", {})
    if not bool(tcfg.get("init_pose_enable", False)):
        return None
    pose_path = str(tcfg.get("init_pose_path", "")).strip()
    if not pose_path:
        print("[InitPose] disabled: init_pose_path is empty")
        return None
    if not os.path.isfile(pose_path):
        print(f"[InitPose] disabled: file not found: {pose_path}")
        return None

    loaded = np.load(pose_path, allow_pickle=True)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        pose_key = str(tcfg.get("init_pose_key", "w2c"))
        if pose_key not in loaded.files:
            pose_key = "poses" if "poses" in loaded.files else loaded.files[0]
        poses = loaded[pose_key]
    else:
        poses = loaded
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim == 2 and poses.shape[1] == 16:
        poses = poses.reshape(-1, 4, 4)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"[InitPose] expected (N,4,4) poses, got {poses.shape} from {pose_path}")

    convention = str(tcfg.get("init_pose_convention", "w2c")).lower()
    relative = bool(tcfg.get("init_pose_relative", True))
    if convention == "c2w":
        if relative:
            poses = np.linalg.inv(poses)
        else:
            first_c2w_inv = np.linalg.inv(poses[0])
            rel_c2w = first_c2w_inv[None, ...] @ poses
            poses = np.linalg.inv(rel_c2w)
    elif convention == "w2c":
        if not relative:
            poses = poses @ np.linalg.inv(poses[0])[None, ...]
    else:
        raise ValueError(f"[InitPose] unsupported convention: {convention}")

    if poses.shape[0] < num_frames:
        print(f"[InitPose] warning: {poses.shape[0]} init poses for {num_frames} frames; tail frames use default init")
    elif poses.shape[0] > num_frames:
        poses = poses[:num_frames]

    print(
        f"[InitPose] loaded {poses.shape[0]} poses from {pose_path} "
        f"convention={convention} relative={relative}"
    )
    return torch.from_numpy(poses).to(device=device, dtype=torch.float32)


def apply_tracking_init_pose(params, init_poses, time_idx, apply_mode="all"):
    if init_poses is None or time_idx >= init_poses.shape[0]:
        return False
    mode = str(apply_mode or "all").lower()
    if mode in ("first", "once", "initial") and time_idx != 1:
        return False
    if mode not in ("all", "every", "first", "once", "initial"):
        raise ValueError(f"[InitPose] unsupported init_pose_apply_mode: {apply_mode}")
    init_w2c = init_poses[time_idx]
    if not torch.isfinite(init_w2c).all():
        return False
    rot_q = matrix_to_quaternion(init_w2c[:3, :3].unsqueeze(0))
    with torch.no_grad():
        params['cam_unnorm_rots'][..., time_idx] = rot_q.detach()
        params['cam_trans'][..., time_idx] = init_w2c[:3, 3].detach()
    return True


def _scheduled_mix_alpha(time_idx: int, scfg: dict) -> float:
    base_alpha = float(scfg.get("mapping_mix_alpha", 0.5))
    if not bool(scfg.get("mapping_mix_alpha_sched_enable", False)):
        return float(np.clip(base_alpha, 0.0, 1.0))
    start_f = int(scfg.get("mapping_mix_alpha_sched_start_frame", scfg.get("mapping_warmup_frames", 0)))
    end_f = int(scfg.get("mapping_mix_alpha_sched_end_frame", start_f + 300))
    a0 = float(scfg.get("mapping_mix_alpha_sched_start", min(0.95, max(base_alpha, 0.85))))
    a1 = float(scfg.get("mapping_mix_alpha_sched_end", base_alpha))
    if end_f <= start_f:
        return float(np.clip(a1, 0.0, 1.0))
    t = float(np.clip((time_idx - start_f) / float(end_f - start_f), 0.0, 1.0))
    curve = str(scfg.get("mapping_mix_alpha_sched_curve", "linear")).lower()
    if curve == "cosine":
        t = 0.5 - 0.5 * np.cos(np.pi * t)
    return float(np.clip((1.0 - t) * a0 + t * a1, 0.0, 1.0))


def _tracking_residual_gate(
    time_idx: int,
    tracking_residual: float,
    tracking_residual_ema: float,
    scfg: dict,
) -> float:
    if not bool(scfg.get("tracking_residual_gate_enable", False)):
        return 1.0
    warmup = int(scfg.get("tracking_residual_gate_warmup_frames", 30))
    if (time_idx < warmup) or (tracking_residual_ema <= 1e-12):
        return 1.0
    ratio = float(tracking_residual / (tracking_residual_ema + 1e-12))
    hi = max(float(scfg.get("tracking_residual_gate_ratio_hi", 2.0)), 1.0001)
    gmin = float(np.clip(scfg.get("tracking_residual_gate_min", 0.2), 0.0, 1.0))
    if ratio <= 1.0:
        return 1.0
    if ratio >= hi:
        return gmin
    # Linear decay from 1.0 -> gmin as residual ratio rises from 1.0 -> hi.
    t = (ratio - 1.0) / (hi - 1.0)
    return float(np.clip(1.0 - t * (1.0 - gmin), gmin, 1.0))


def _score_confidence(scores: torch.Tensor | np.ndarray, temperature: float = 1.0) -> tuple[float, float]:
    """Return (top probability, normalized entropy) for candidate scores."""
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().float().view(-1).numpy()
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size <= 0:
        return 0.0, 1.0
    logits = scores / max(float(temperature), 1e-6)
    logits = logits - np.max(logits)
    probs = np.exp(np.clip(logits, -50.0, 50.0))
    probs = probs / (probs.sum() + 1e-12)
    top_prob = float(np.max(probs))
    if probs.size <= 1:
        return top_prob, 0.0
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    norm_entropy = entropy / float(np.log(probs.size))
    return top_prob, float(np.clip(norm_entropy, 0.0, 1.0))


def _mapping_hard_gate_should_fallback(
    time_idx: int,
    scores: torch.Tensor | np.ndarray,
    tracking_gate: float,
    pose_step_ratio: float | None,
    scfg: dict,
) -> tuple[bool, list[str]]:
    """Decide whether to bypass KFS and use exact baseline distance sampling."""
    if not bool(scfg.get("mapping_hard_gate_enable", False)):
        return False, []
    warmup = int(scfg.get("mapping_hard_gate_warmup_frames", scfg.get("mapping_warmup_frames", 0)))
    if time_idx < warmup:
        return True, ["warmup"]

    reasons = []
    top_prob, norm_entropy = _score_confidence(scores, scfg.get("sampling_temperature", 1.0))
    min_top = float(scfg.get("mapping_hard_gate_min_top_prob", 0.0))
    max_entropy = float(scfg.get("mapping_hard_gate_max_entropy", 1.0))
    min_tracking_gate = float(scfg.get("mapping_hard_gate_min_tracking_gate", 0.0))
    max_pose_step_ratio = float(scfg.get("mapping_hard_gate_max_pose_step_ratio", 1e9))

    if top_prob < min_top:
        reasons.append(f"low_top_prob={top_prob:.3f}")
    if norm_entropy > max_entropy:
        reasons.append(f"high_entropy={norm_entropy:.3f}")
    if tracking_gate < min_tracking_gate:
        reasons.append(f"tracking_gate={tracking_gate:.3f}")
    if pose_step_ratio is not None and np.isfinite(pose_step_ratio) and pose_step_ratio > max_pose_step_ratio:
        reasons.append(f"pose_step_ratio={pose_step_ratio:.3f}")
    return bool(reasons), reasons


def _pose_demand_image_change(curr_color: torch.Tensor, prev_color: torch.Tensor | None) -> float:
    """Cheap appearance-change signal in [0, inf), using RGB tensors in [0, 1]."""
    if prev_color is None:
        return 0.0
    try:
        with torch.no_grad():
            return float((curr_color.detach() - prev_color.detach()).abs().mean().cpu().item())
    except Exception:
        return 0.0


def _pose_demand_should_update(
    time_idx: int,
    curr_color: torch.Tensor,
    prev_color: torch.Tensor | None,
    frames_since_last_update: int,
    residual_ratio: float,
    pose_uncertainty: float,
    accumulated_residual: float,
    scfg: dict,
) -> tuple[bool, float, list[str], dict]:
    """Adaptive trigger for learned pose initialization."""
    if not bool(scfg.get("pose_demand_enable", False)):
        return True, 1.0, ["disabled"], {}
    warmup = int(scfg.get("pose_demand_warmup_frames", 1))
    if time_idx < warmup:
        return False, 0.0, ["warmup"], {}

    image_change = _pose_demand_image_change(curr_color, prev_color)
    motion_th = max(float(scfg.get("pose_demand_motion_threshold", 0.03)), 1e-8)
    residual_th = max(float(scfg.get("pose_demand_residual_ratio_threshold", 1.35)), 1e-8)
    uncertainty_th = max(float(scfg.get("pose_demand_uncertainty_threshold", 1.8)), 1e-8)
    accum_th = max(float(scfg.get("pose_demand_accumulated_residual_threshold", 2.0)), 1e-8)
    max_gap = max(int(scfg.get("pose_demand_max_gap", 12)), 1)

    motion_signal = image_change / motion_th
    residual_signal = max(float(residual_ratio), 0.0) / residual_th
    uncertainty_signal = max(float(pose_uncertainty), 0.0) / uncertainty_th
    time_signal = max(float(frames_since_last_update), 0.0) / float(max_gap)
    accum_signal = max(float(accumulated_residual), 0.0) / accum_th

    weights = (
        float(scfg.get("pose_demand_w_motion", 1.0)),
        float(scfg.get("pose_demand_w_residual", 1.0)),
        float(scfg.get("pose_demand_w_uncertainty", 1.0)),
        float(scfg.get("pose_demand_w_time", 1.0)),
        float(scfg.get("pose_demand_w_accumulated", 1.0)),
    )
    score = (
        weights[0] * motion_signal
        + weights[1] * residual_signal
        + weights[2] * uncertainty_signal
        + weights[3] * time_signal
        + weights[4] * accum_signal
    ) / max(sum(abs(w) for w in weights), 1e-8)

    score_th = float(scfg.get("pose_demand_score_threshold", 1.0))
    reasons = []
    if image_change >= motion_th:
        reasons.append(f"motion={image_change:.4f}")
    if residual_ratio >= residual_th:
        reasons.append(f"residual_ratio={residual_ratio:.3f}")
    if pose_uncertainty >= uncertainty_th:
        reasons.append(f"uncertainty={pose_uncertainty:.3f}")
    if frames_since_last_update >= max_gap:
        reasons.append(f"max_gap={frames_since_last_update}")
    if accumulated_residual >= accum_th:
        reasons.append(f"accum_residual={accumulated_residual:.3f}")
    if score >= score_th:
        reasons.append(f"score={score:.3f}")

    flags = {
        "motion": bool(image_change >= motion_th),
        "residual": bool(residual_ratio >= residual_th),
        "uncertainty": bool(pose_uncertainty >= uncertainty_th),
        "max_gap": bool(frames_since_last_update >= max_gap),
        "accumulated": bool(accumulated_residual >= accum_th),
        "score": bool(score >= score_th),
    }
    debug = {
        "image_change": image_change,
        "residual_ratio": float(residual_ratio),
        "pose_uncertainty": float(pose_uncertainty),
        "frames_since_last_update": int(frames_since_last_update),
        "accumulated_residual": float(accumulated_residual),
        "flags": flags,
    }
    return bool(reasons), float(score), reasons, debug


def _pose_demand_quality_gate_reason(tier: str, demand_debug: dict, scfg: dict) -> str | None:
    """Suppress extra tracking on low-information frames while preserving baseline tracking."""
    if not bool(scfg.get("pose_demand_quality_gate_enable", False)):
        return None
    if not isinstance(demand_debug, dict):
        return None

    gated_tiers = {
        x.strip().lower()
        for x in str(scfg.get("pose_demand_quality_gate_tiers", "medium,weak")).split(",")
        if x.strip()
    }
    if tier.lower() not in gated_tiers:
        return None

    flags = demand_debug.get("flags", {}) if isinstance(demand_debug.get("flags", {}), dict) else {}
    if flags.get("residual", False) or flags.get("accumulated", False):
        return None

    motion_th = max(float(scfg.get("pose_demand_motion_threshold", 0.03)), 1e-8)
    motion_signal = float(demand_debug.get("image_change", 0.0)) / motion_th
    residual_ratio = float(demand_debug.get("residual_ratio", 0.0))
    accumulated_residual = float(demand_debug.get("accumulated_residual", 0.0))
    frames_since = int(demand_debug.get("frames_since_last_update", 0))
    max_gap = max(int(scfg.get("pose_demand_max_gap", 12)), 1)

    near_dup_motion = float(scfg.get("pose_demand_quality_gate_near_duplicate_motion_ratio", 0.6))
    low_residual_ratio = float(scfg.get("pose_demand_quality_gate_low_residual_ratio", 1.10))
    low_accumulated = float(scfg.get("pose_demand_quality_gate_low_accumulated_residual", 0.50))
    low_info = (
        motion_signal <= near_dup_motion
        and residual_ratio < low_residual_ratio
        and accumulated_residual < low_accumulated
    )

    if low_info and frames_since < max_gap:
        return (
            "gated_low_info"
            f"(m={motion_signal:.2f},r={residual_ratio:.2f},a={accumulated_residual:.2f},gap={frames_since})"
        )

    block_score_only = bool(scfg.get("pose_demand_quality_gate_block_score_only", True))
    if (
        block_score_only
        and tier.lower() == "weak"
        and flags.get("score", False)
        and not flags.get("max_gap", False)
        and residual_ratio < low_residual_ratio
        and accumulated_residual < low_accumulated
    ):
        return (
            "gated_score_only"
            f"(m={motion_signal:.2f},r={residual_ratio:.2f},a={accumulated_residual:.2f},gap={frames_since})"
        )

    return None


def _pose_demand_tracking_iters(base_iters: int, demand_debug: dict, scfg: dict) -> tuple[int, str]:
    """Choose tracking budget by trigger class: residual > uncertainty/motion > max-gap."""
    flags = demand_debug.get("flags", {}) if isinstance(demand_debug, dict) else {}
    max_iters = int(scfg.get("pose_demand_tracking_max_iters", 60))
    max_iters = max(base_iters, max_iters)

    if flags.get("residual", False) or flags.get("accumulated", False):
        extra = int(scfg.get("pose_demand_tracking_strong_extra_iters", scfg.get("pose_demand_tracking_extra_iters", 10)))
        mult = float(scfg.get("pose_demand_tracking_strong_iter_multiplier", scfg.get("pose_demand_tracking_iter_multiplier", 1.0)))
        tier = "strong"
    elif flags.get("uncertainty", False) or flags.get("motion", False):
        extra = int(scfg.get("pose_demand_tracking_medium_extra_iters", 5))
        mult = float(scfg.get("pose_demand_tracking_medium_iter_multiplier", 1.0))
        tier = "medium"
    elif flags.get("max_gap", False) or flags.get("score", False):
        extra = int(scfg.get("pose_demand_tracking_weak_extra_iters", 2))
        mult = float(scfg.get("pose_demand_tracking_weak_iter_multiplier", 1.0))
        tier = "weak"
    else:
        return base_iters, "none"

    gate_reason = _pose_demand_quality_gate_reason(tier, demand_debug, scfg)
    if gate_reason is not None:
        return base_iters, gate_reason

    iters = int(round(base_iters * mult)) + extra
    return min(max_iters, max(base_iters, iters)), tier


def _float_loss_value(loss_value) -> float:
    try:
        if torch.is_tensor(loss_value):
            return float(loss_value.detach().cpu().item())
        return float(loss_value)
    except Exception:
        return float("inf")


def _tracking_extra_should_accept(
    base_loss: float,
    extra_loss: float,
    base_tran: torch.Tensor,
    extra_tran: torch.Tensor,
    prev_tran: torch.Tensor,
    pose_step_ema: float | None,
    scfg: dict,
) -> tuple[bool, str]:
    """Accept extra tracking only when it improves loss and keeps pose step sane."""
    min_rel = float(scfg.get("pose_demand_tracking_rollback_min_rel_improve", 0.005))
    min_abs = float(scfg.get("pose_demand_tracking_rollback_min_abs_improve", 0.0))
    required_loss = base_loss * (1.0 - min_rel) - min_abs
    if not np.isfinite(extra_loss) or extra_loss > required_loss:
        return False, f"loss {extra_loss:.6f} !< {required_loss:.6f}"

    try:
        base_t = base_tran.detach().cpu().numpy()
        extra_t = extra_tran.detach().cpu().numpy()
        prev_t = prev_tran.detach().cpu().numpy()
        base_step = float(np.linalg.norm(base_t - prev_t))
        extra_step = float(np.linalg.norm(extra_t - prev_t))
    except Exception as exc:
        return False, f"bad_step={exc}"
    ratio = float(scfg.get("pose_demand_tracking_rollback_max_step_ratio", 2.5))
    max_abs = float(scfg.get("pose_demand_tracking_rollback_max_abs_step", 1.0))
    ref_step = max(base_step, float(pose_step_ema or 0.0), 1e-4)
    step_limit = max_abs + ratio * ref_step
    if extra_step > step_limit:
        return False, f"step {extra_step:.6f} > {step_limit:.6f}"
    return True, f"loss {base_loss:.6f}->{extra_loss:.6f}, step {base_step:.6f}->{extra_step:.6f}"


def _pose_translation_np(params: dict, time_idx: int) -> np.ndarray:
    try:
        return params["cam_trans"][..., time_idx].detach().cpu().numpy().reshape(-1)[:3].astype(np.float64)
    except Exception:
        return np.zeros((3,), dtype=np.float64)


def _pose_rotation_np(params: dict, time_idx: int) -> np.ndarray:
    try:
        q = F.normalize(params["cam_unnorm_rots"][..., time_idx].detach())
        r = build_rotation(q).detach().cpu().numpy()
        if r.ndim == 3:
            r = r[0]
        return r.astype(np.float64)
    except Exception:
        return np.eye(3, dtype=np.float64)


def _current_w2c_tensor(params: dict, time_idx: int) -> torch.Tensor:
    """Differentiable current-frame w2c from optimizable camera pose params."""
    cam_rot = F.normalize(params["cam_unnorm_rots"][..., time_idx])
    cam_tran = params["cam_trans"][..., time_idx].reshape(-1)[:3]
    w2c = torch.eye(4, device=cam_rot.device, dtype=cam_rot.dtype)
    w2c[:3, :3] = build_rotation(cam_rot)[0]
    w2c[:3, 3] = cam_tran
    return w2c


def _rotation_angle_deg(r0: np.ndarray, r1: np.ndarray) -> float:
    try:
        rel = r1 @ r0.T
        c = float(np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0))
        return float(np.degrees(np.arccos(c)))
    except Exception:
        return 0.0


def _vector_angle_deg(v0: np.ndarray, v1: np.ndarray) -> float:
    n0 = float(np.linalg.norm(v0))
    n1 = float(np.linalg.norm(v1))
    if n0 <= 1e-12 or n1 <= 1e-12:
        return 0.0
    c = float(np.clip(np.dot(v0, v1) / (n0 * n1), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _image_proxy_gray_grad(color: torch.Tensor, downsample: int) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Small grayscale/gradient proxy for current-vs-reference keyframe gating."""
    x = color.detach()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    x = x.float().clamp(0.0, 1.0)
    size = int(downsample)
    if size > 0:
        h, w = x.shape[-2:]
        if max(h, w) > size:
            if h >= w:
                out_h = size
                out_w = max(8, int(round(w * size / max(h, 1))))
            else:
                out_w = size
                out_h = max(8, int(round(h * size / max(w, 1))))
            x = F.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=False)
    gray = (0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]).contiguous()
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=gray.device,
        dtype=gray.dtype,
    ).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    grad = torch.sqrt(gx * gx + gy * gy + 1e-6)
    return gray, grad, float(grad.mean().detach().cpu().item()), float(gray.std().detach().cpu().item())


def _keyframe_energy_stats(curr_color: torch.Tensor, fold_seg: torch.Tensor | None, config: dict) -> dict:
    scfg = config.get("scorenet", {})
    downsample = int(scfg.get("keyframe_energy_downsample", 128))
    x = curr_color.detach()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    x = x.float().clamp(0.0, 1.0)
    if downsample > 0:
        h, w = x.shape[-2:]
        if max(h, w) > downsample:
            if h >= w:
                out_h = downsample
                out_w = max(8, int(round(w * downsample / max(h, 1))))
            else:
                out_w = downsample
                out_h = max(8, int(round(h * downsample / max(w, 1))))
            x = F.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=False)

    gray = (0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]).contiguous()
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=gray.device,
        dtype=gray.dtype,
    ).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    grad = torch.sqrt(F.conv2d(gray, kx, padding=1) ** 2 + F.conv2d(gray, ky, padding=1) ** 2 + 1e-6)
    grad_norm = (grad / (torch.quantile(grad.flatten(), 0.90).clamp_min(1e-6))).clamp(0.0, 1.0)

    if fold_seg is not None:
        fold = fold_seg.detach().float().clamp(0.0, 1.0)
        if fold.ndim == 3:
            fold = fold.unsqueeze(0)
        if tuple(fold.shape[-2:]) != tuple(gray.shape[-2:]):
            fold = F.interpolate(fold, size=gray.shape[-2:], mode="bilinear", align_corners=False)
        fold = fold[:, :1]
        boundary = _fold_boundary_map(fold[0])[None]
        boundary_norm = (boundary / (torch.quantile(boundary.flatten(), 0.90).clamp_min(1e-6))).clamp(0.0, 1.0)
    else:
        fold = torch.zeros_like(gray)
        boundary_norm = torch.zeros_like(gray)

    spec_th = float(scfg.get("keyframe_energy_specular_threshold", 0.92))
    spec = ((x.max(dim=1, keepdim=True).values > spec_th) | (gray > spec_th)).float()

    alpha = float(scfg.get("keyframe_energy_grad_weight", 1.0))
    beta = float(scfg.get("keyframe_energy_fold_weight", 0.75))
    gamma = float(scfg.get("keyframe_energy_boundary_weight", 1.0))
    delta = float(scfg.get("keyframe_energy_specular_weight", 1.0))
    energy = (alpha * grad_norm + beta * fold + gamma * boundary_norm - delta * spec).clamp(0.0, 1.0)

    energy_th = float(scfg.get("keyframe_energy_threshold", 0.25))
    grad_th = float(scfg.get("keyframe_energy_grad_threshold", 0.20))
    fold_th = float(scfg.get("keyframe_energy_fold_threshold", 0.15))
    boundary_th = float(scfg.get("keyframe_energy_boundary_threshold", 0.20))
    return {
        "energy_ratio": float((energy > energy_th).float().mean().detach().cpu().item()),
        "texture_ratio": float((grad_norm > grad_th).float().mean().detach().cpu().item()),
        "fold_ratio": float((fold > fold_th).float().mean().detach().cpu().item()),
        "fold_boundary_ratio": float((boundary_norm > boundary_th).float().mean().detach().cpu().item()),
        "specular_ratio": float(spec.mean().detach().cpu().item()),
        "energy_mean": float(energy.mean().detach().cpu().item()),
    }


def _energy_keyframe_gate(
    proposed_add: bool,
    motion_debug: dict,
    curr_color: torch.Tensor,
    curr_data: dict,
    config: dict,
) -> tuple[bool, dict]:
    """Suppress keyframe insertion when a frame has too little reliable signal."""
    scfg = config.get("scorenet", {})
    if not bool(scfg.get("keyframe_energy_gate_enable", False)):
        return proposed_add, {}

    stats = _keyframe_energy_stats(curr_color, curr_data.get("fold_seg", None), config)
    reason = str(motion_debug.get("reason", "unknown"))
    ref_reason = str(motion_debug.get("ref_reason", ""))
    protected = reason in ("init", "terminal", "max_gap")
    min_energy = float(scfg.get("keyframe_energy_min_ratio", 0.12))
    min_boundary = float(scfg.get("keyframe_energy_min_boundary_ratio", 0.0))
    max_specular = float(scfg.get("keyframe_energy_max_specular_ratio", 0.45))
    reliable = (
        stats["energy_ratio"] >= min_energy
        and stats["fold_boundary_ratio"] >= min_boundary
        and stats["specular_ratio"] <= max_specular
    )
    mode = str(scfg.get("keyframe_energy_gate_mode", "block")).lower()
    out = {
        "energy_reason": "protected" if protected else ("keep" if reliable else "low_reliability"),
        "energy_gate_mode": mode,
        "energy_reliable": bool(reliable),
        "energy_min_ratio": min_energy,
        "energy_min_boundary_ratio": min_boundary,
        "energy_max_specular_ratio": max_specular,
        **stats,
    }
    if protected or (not proposed_add) or reliable:
        return proposed_add, out

    if mode == "soft":
        # Soft mode targets novelty-driven insertions first, while leaving strong
        # motion triggers to the motion proxy.
        if reason == "wait" or ref_reason == "add_ref_novel":
            out["energy_reason"] = "soft_block_ref_novel"
            return False, out
        out["energy_reason"] = "soft_keep_motion"
        return proposed_add, out

    out["energy_reason"] = "block_low_reliability"
    return False, out


def _resize_color_for_pair_proxy(color: torch.Tensor, downsample: int) -> torch.Tensor:
    x = color.detach()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    x = x.float().clamp(0.0, 1.0)
    size = int(downsample)
    if size <= 0:
        return x
    h, w = x.shape[-2:]
    if max(h, w) <= size:
        return x
    if h >= w:
        out_h = size
        out_w = max(8, int(round(w * size / max(h, 1))))
    else:
        out_w = size
        out_h = max(8, int(round(h * size / max(w, 1))))
    return F.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=False)


def _get_kfg_pair_proxy_fn(scfg: dict):
    global _KFG_PAIR_PROXY_FN, _KFG_PAIR_PROXY_ROOT
    root = str(scfg.get("keyframe_graph_root", os.path.abspath(os.path.join(_BASE_DIR, '..', '..', 'modules'))))
    if _KFG_PAIR_PROXY_FN is not None and _KFG_PAIR_PROXY_ROOT == root:
        return _KFG_PAIR_PROXY_FN
    if root and root not in sys.path:
        sys.path.insert(0, root)
    from networks.multiwrapper import compute_pair_proxies_pose_free
    _KFG_PAIR_PROXY_FN = compute_pair_proxies_pose_free
    _KFG_PAIR_PROXY_ROOT = root
    return _KFG_PAIR_PROXY_FN


def _feature_pair_keyframe_gate(
    proposed_add: bool,
    motion_debug: dict,
    curr_color: torch.Tensor,
    keyframe_list: list,
    config: dict,
) -> tuple[bool, dict]:
    """Gate motion-proxy keyframes using KeyFrameGraph ref-current 8D pair features."""
    scfg = config.get("scorenet", {})
    if len(keyframe_list) == 0:
        return proposed_add, {"feat_reason": "no_ref"}

    reason = str(motion_debug.get("reason", "unknown"))
    if reason in ("init", "terminal", "max_gap"):
        return proposed_add, {"feat_reason": "protected"}
    if reason == "min_gap":
        return proposed_add, {"feat_reason": "min_gap"}

    recent_n = max(int(scfg.get("keyframe_feature_pair_recent_refs", 3)), 1)
    refs = keyframe_list[-recent_n:]
    downsample = int(scfg.get("keyframe_feature_pair_downsample", 128))
    max_disp = int(scfg.get("keyframe_feature_pair_max_disp", scfg.get("pair_max_disp", 4)))
    temp = float(scfg.get("keyframe_feature_pair_corr_temp", scfg.get("pair_corr_temp", 0.5)))

    try:
        proxy_fn = _get_kfg_pair_proxy_fn(scfg)
        target = _resize_color_for_pair_proxy(curr_color, downsample).to(curr_color.device)
        ref_imgs = [_resize_color_for_pair_proxy(ref["color"], downsample).to(curr_color.device) for ref in refs]
        ref_stack = torch.stack(ref_imgs, dim=1)
        with torch.no_grad():
            pair_vec_t = proxy_fn(target, ref_stack, max_disp=max_disp, temp=temp)
        if pair_vec_t.ndim == 2:
            pair_vec_t = pair_vec_t.unsqueeze(0)
        pair_vec = pair_vec_t[0].detach().cpu().float().numpy()
    except Exception as exc:
        return proposed_add, {"feat_reason": f"proxy_error={exc}"}

    # Columns from KeyFrameGraph: mae, contrast, edge_t, edge_s, grad_overlap,
    # flow_mag, prob_max, entropy.
    mae = pair_vec[:, 0]
    contrast = pair_vec[:, 1]
    edge_t = pair_vec[:, 2]
    edge_s = pair_vec[:, 3]
    grad_overlap = pair_vec[:, 4]
    flow_mag = pair_vec[:, 5]
    prob_max = pair_vec[:, 6]
    entropy = pair_vec[:, 7]

    best_i = int(np.argmin(mae))
    last_i = len(refs) - 1
    min_mae = float(mae[best_i])
    last_mae = float(mae[last_i])
    min_flow = float(flow_mag[best_i])
    last_flow = float(flow_mag[last_i])
    max_grad = float(np.max(grad_overlap))
    last_grad = float(grad_overlap[last_i])
    last_prob = float(prob_max[last_i])
    last_entropy = float(entropy[last_i])
    edge_ratio = float(np.median(edge_t) / (np.median(edge_s) + 1e-6))

    dup_mae = float(scfg.get("keyframe_feature_pair_duplicate_mae", 0.025))
    dup_flow = float(scfg.get("keyframe_feature_pair_duplicate_flow", 0.10))
    dup_grad = float(scfg.get("keyframe_feature_pair_duplicate_grad", 0.92))
    min_edge_ratio = float(scfg.get("keyframe_feature_pair_min_edge_ratio", 0.65))
    min_prob = float(scfg.get("keyframe_feature_pair_min_prob_max", 0.015))
    max_entropy = float(scfg.get("keyframe_feature_pair_max_entropy", 0.985))
    block_duplicate = bool(scfg.get("keyframe_feature_pair_block_duplicate", True))
    block_low_texture = bool(scfg.get("keyframe_feature_pair_block_low_texture", True))
    block_bad_corr = bool(scfg.get("keyframe_feature_pair_block_bad_corr", False))

    near_duplicate = bool(min_mae <= dup_mae and min_flow <= dup_flow and max_grad >= dup_grad)
    low_texture = bool(edge_ratio < min_edge_ratio)
    bad_corr = bool(last_prob < min_prob or last_entropy > max_entropy)

    out = {
        "feat_reason": "keep",
        "feat_best": int(refs[best_i]["id"]),
        "feat_mae": min_mae,
        "feat_last_mae": last_mae,
        "feat_flow": min_flow,
        "feat_last_flow": last_flow,
        "feat_grad": max_grad,
        "feat_last_grad": last_grad,
        "feat_prob": last_prob,
        "feat_entropy": last_entropy,
        "feat_edge_ratio": edge_ratio,
        "feat_contrast": float(contrast[last_i]),
        "feat_duplicate": near_duplicate,
        "feat_low_texture": low_texture,
        "feat_bad_corr": bad_corr,
    }

    if proposed_add and block_low_texture and low_texture:
        out["feat_reason"] = "block_low_texture"
        return False, out
    if proposed_add and block_bad_corr and bad_corr:
        out["feat_reason"] = "block_bad_corr"
        return False, out
    if proposed_add and block_duplicate and near_duplicate:
        out["feat_reason"] = "block_duplicate"
        return False, out
    return proposed_add, out


def _reference_pair_keyframe_gate(
    proposed_add: bool,
    motion_debug: dict,
    curr_color: torch.Tensor,
    keyframe_list: list,
    config: dict,
) -> tuple[bool, dict]:
    """Gate motion-proxy keyframe decisions using current-vs-reference image proxies."""
    scfg = config.get("scorenet", {})
    if len(keyframe_list) == 0:
        return proposed_add, {"ref_reason": "no_ref"}

    reason = str(motion_debug.get("reason", "unknown"))
    if reason in ("init", "terminal", "max_gap"):
        return proposed_add, {"ref_reason": "protected"}
    if reason == "min_gap":
        return proposed_add, {"ref_reason": "min_gap"}

    recent_n = max(int(scfg.get("keyframe_ref_pair_recent_refs", 3)), 1)
    downsample = int(scfg.get("keyframe_ref_pair_downsample", 96))
    refs = keyframe_list[-recent_n:]

    try:
        curr_gray, curr_grad, curr_edge, curr_contrast = _image_proxy_gray_grad(curr_color, downsample)
        stats = []
        for ref in refs:
            ref_gray, ref_grad, ref_edge, ref_contrast = _image_proxy_gray_grad(ref["color"], downsample)
            im_l1 = float(torch.mean(torch.abs(curr_gray - ref_gray)).detach().cpu().item())
            grad_l1 = float(torch.mean(torch.abs(curr_grad - ref_grad)).detach().cpu().item())
            stats.append(
                {
                    "id": int(ref["id"]),
                    "im_l1": im_l1,
                    "grad_l1": grad_l1,
                    "edge": ref_edge,
                    "contrast": ref_contrast,
                }
            )
    except Exception as exc:
        return proposed_add, {"ref_reason": f"proxy_error={exc}"}

    last = stats[-1]
    best = min(stats, key=lambda item: item["im_l1"])
    min_l1 = float(best["im_l1"])
    min_grad_l1 = float(min(item["grad_l1"] for item in stats))
    last_l1 = float(last["im_l1"])
    last_grad_l1 = float(last["grad_l1"])
    ref_edge = float(np.median([item["edge"] for item in stats]))
    ref_contrast = float(np.median([item["contrast"] for item in stats]))
    edge_ratio = curr_edge / (ref_edge + 1e-6)
    contrast_ratio = curr_contrast / (ref_contrast + 1e-6)

    dup_l1 = float(scfg.get("keyframe_ref_pair_duplicate_l1", 0.025))
    dup_grad_l1 = float(scfg.get("keyframe_ref_pair_duplicate_grad_l1", 0.020))
    add_l1 = float(scfg.get("keyframe_ref_pair_add_l1", 0.060))
    add_grad_l1 = float(scfg.get("keyframe_ref_pair_add_grad_l1", 0.035))
    min_edge_ratio = float(scfg.get("keyframe_ref_pair_min_edge_ratio", 0.65))
    min_contrast_ratio = float(scfg.get("keyframe_ref_pair_min_contrast_ratio", 0.60))
    allow_add_ref_novel = bool(scfg.get("keyframe_ref_pair_allow_add", True))

    near_duplicate = bool(min_l1 <= dup_l1 and min_grad_l1 <= dup_grad_l1)
    low_quality = bool(edge_ratio < min_edge_ratio or contrast_ratio < min_contrast_ratio)
    ref_novel = bool(last_l1 >= add_l1 or last_grad_l1 >= add_grad_l1)

    out = {
        "ref_reason": "keep",
        "ref_best": int(best["id"]),
        "ref_l1": min_l1,
        "ref_last_l1": last_l1,
        "ref_grad": min_grad_l1,
        "ref_last_grad": last_grad_l1,
        "ref_edge_ratio": edge_ratio,
        "ref_contrast_ratio": contrast_ratio,
        "ref_duplicate": near_duplicate,
        "ref_low_quality": low_quality,
        "ref_novel": ref_novel,
    }

    if low_quality and proposed_add:
        out["ref_reason"] = "block_low_quality"
        return False, out
    if near_duplicate and proposed_add:
        out["ref_reason"] = "block_duplicate"
        return False, out
    if allow_add_ref_novel and (not proposed_add) and ref_novel and not low_quality:
        out["ref_reason"] = "add_ref_novel"
        return True, out
    return proposed_add, out


def _motion_proxy_keyframe_decision(
    time_idx: int,
    num_frames: int,
    params: dict,
    keyframe_list: list,
    config: dict,
) -> tuple[bool, dict]:
    """Adaptive keyframe insertion from pose motion proxy."""
    scfg = config.get("scorenet", {})
    mode = str(scfg.get("keyframe_motion_proxy_mode", "instant")).lower()
    base_gap = max(int(scfg.get("keyframe_motion_proxy_base_gap", config.get("keyframe_every", 8))), 1)
    min_gap = max(int(scfg.get("keyframe_motion_proxy_min_gap", max(2, base_gap // 2))), 1)
    max_gap = max(int(scfg.get("keyframe_motion_proxy_max_gap", max(base_gap + 1, 2 * base_gap))), min_gap)
    terminal = bool(time_idx == num_frames - 2)

    if time_idx == 0 or len(keyframe_list) == 0:
        scfg["_keyframe_motion_accum_state"] = {"last_time_idx": int(time_idx), "motion": 0.0, "rot": 0.0}
        return True, {"reason": "init", "gap": 0}

    last_kf_id = int(keyframe_list[-1]["id"])
    gap = int(time_idx - last_kf_id)

    curr_t = _pose_translation_np(params, time_idx)
    curr_r = _pose_rotation_np(params, time_idx)
    prev_idx = max(0, time_idx - 1)
    prev_t = _pose_translation_np(params, prev_idx)
    prev_r = _pose_rotation_np(params, prev_idx)
    step_dist = float(np.linalg.norm(curr_t - prev_t))
    step_rot_deg = _rotation_angle_deg(prev_r, curr_r)

    accum_state = scfg.setdefault("_keyframe_motion_accum_state", {"last_time_idx": None, "motion": 0.0, "rot": 0.0})
    if accum_state.get("last_time_idx") != int(time_idx):
        accum_state["motion"] = float(accum_state.get("motion", 0.0)) + step_dist
        accum_state["rot"] = float(accum_state.get("rot", 0.0)) + step_rot_deg
        accum_state["last_time_idx"] = int(time_idx)

    accum_motion = float(accum_state.get("motion", 0.0))
    accum_rot_deg = float(accum_state.get("rot", 0.0))

    last_t = _pose_translation_np(params, last_kf_id)
    net_dist = float(np.linalg.norm(curr_t - last_t))

    hist_start = max(0, time_idx - max(max_gap, 2 * base_gap))
    hist_positions = [_pose_translation_np(params, i) for i in range(hist_start, time_idx + 1)]
    recent_steps = np.array(
        [np.linalg.norm(hist_positions[i] - hist_positions[i - 1]) for i in range(1, len(hist_positions))],
        dtype=np.float64,
    )
    positive_steps = recent_steps[recent_steps > 1e-8]
    ref_step = float(np.median(positive_steps)) if positive_steps.size > 0 else 0.0
    min_target_dist = float(scfg.get("keyframe_motion_proxy_min_target_dist", 0.03))
    target_dist = max(
        min_target_dist,
        ref_step * float(base_gap) * float(scfg.get("keyframe_motion_proxy_target_dist_scale", 1.0)),
    )

    rot_th = float(scfg.get("keyframe_motion_proxy_rot_angle_deg", 18.0))
    accum_threshold = target_dist * float(scfg.get("keyframe_motion_proxy_accum_threshold_scale", 1.0))
    accum_rot_weight = float(scfg.get("keyframe_motion_proxy_accum_rot_weight", 1.0))
    rot_equiv = target_dist * accum_rot_weight * (accum_rot_deg / max(rot_th, 1e-6))
    accum_score = accum_motion + (rot_equiv if mode == "accum_motion_rot" else 0.0)

    seg_positions = [_pose_translation_np(params, i) for i in range(last_kf_id, time_idx + 1)]
    path_len = float(sum(np.linalg.norm(seg_positions[i] - seg_positions[i - 1]) for i in range(1, len(seg_positions))))
    linearity = float(net_dist / (path_len + 1e-12)) if path_len > 0 else 1.0

    mid = last_kf_id + max(gap // 2, 1)
    mid_t = _pose_translation_np(params, min(mid, time_idx))
    segment_turn = _vector_angle_deg(mid_t - last_t, curr_t - mid_t)
    recent_turn = 0.0
    if time_idx >= 2:
        t0 = _pose_translation_np(params, time_idx - 2)
        t1 = _pose_translation_np(params, time_idx - 1)
        recent_turn = _vector_angle_deg(t1 - t0, curr_t - t1)
    turn_deg = max(segment_turn, recent_turn)
    rot_deg = _rotation_angle_deg(_pose_rotation_np(params, last_kf_id), curr_r)

    turn_th = float(scfg.get("keyframe_motion_proxy_turn_angle_deg", 20.0))
    straight_turn_th = float(scfg.get("keyframe_motion_proxy_straight_angle_deg", 6.0))
    straight_rot_th = float(scfg.get("keyframe_motion_proxy_straight_rot_deg", 6.0))
    straight_linearity = float(scfg.get("keyframe_motion_proxy_straight_linearity", 0.92))
    straight_dist_scale = float(scfg.get("keyframe_motion_proxy_straight_dist_scale", 1.5))
    jitter_dist_ratio = float(scfg.get("keyframe_motion_proxy_jitter_dist_ratio", 0.35))
    jitter_rot_th = float(scfg.get("keyframe_motion_proxy_jitter_rot_deg", 5.0))
    turn_min_dist_ratio = float(scfg.get("keyframe_motion_proxy_turn_min_dist_ratio", 0.25))

    high_turn = turn_deg >= turn_th and net_dist >= target_dist * turn_min_dist_ratio
    high_rot = rot_deg >= rot_th and net_dist >= target_dist * turn_min_dist_ratio
    straight = turn_deg <= straight_turn_th and rot_deg <= straight_rot_th and linearity >= straight_linearity
    jitter = net_dist <= target_dist * jitter_dist_ratio and rot_deg <= jitter_rot_th and gap < max_gap
    if mode in ("accum_motion", "accum_motion_rot"):
        jitter = jitter and accum_score < accum_threshold * float(scfg.get("keyframe_motion_proxy_accum_jitter_hold_ratio", 0.50))

    if terminal and gap >= min_gap:
        add, reason = True, "terminal"
    elif gap < min_gap:
        add, reason = False, "min_gap"
    elif gap >= max_gap:
        add, reason = True, "max_gap"
    elif jitter:
        add, reason = False, "jitter_hold"
    elif mode in ("accum_motion", "accum_motion_rot"):
        add = accum_score >= accum_threshold
        reason = mode if add else "accum_wait"
    elif high_turn:
        add, reason = True, "turn"
    elif high_rot:
        add, reason = True, "rotation"
    else:
        dist_threshold = target_dist * (straight_dist_scale if straight else 1.0)
        add = net_dist >= dist_threshold
        reason = "straight_distance" if straight and add else ("distance" if add else "wait")

    if add:
        accum_state["motion"] = 0.0
        accum_state["rot"] = 0.0

    return bool(add), {
        "mode": mode,
        "reason": reason,
        "gap": gap,
        "step_dist": step_dist,
        "step_rot_deg": step_rot_deg,
        "accum_motion": accum_motion,
        "accum_rot_deg": accum_rot_deg,
        "accum_score": accum_score,
        "accum_threshold": accum_threshold,
        "net_dist": net_dist,
        "target_dist": target_dist,
        "dist_threshold": dist_threshold if "dist_threshold" in locals() else target_dist,
        "path_len": path_len,
        "linearity": linearity,
        "turn_deg": turn_deg,
        "rot_deg": rot_deg,
        "straight": bool(straight),
        "jitter": bool(jitter),
        "high_turn": bool(high_turn),
        "high_rot": bool(high_rot),
        "turn_threshold": turn_th,
        "rot_threshold": rot_th,
        "straight_turn_threshold": straight_turn_th,
        "straight_rot_threshold": straight_rot_th,
        "straight_linearity_threshold": straight_linearity,
        "straight_dist_scale": straight_dist_scale,
        "jitter_dist_ratio": jitter_dist_ratio,
        "jitter_rot_threshold": jitter_rot_th,
        "turn_min_dist_ratio": turn_min_dist_ratio,
    }


def _format_motion_kfs_debug(dbg: dict) -> str:
    msg = (
        f"{dbg.get('reason', 'unknown')} mode={dbg.get('mode', 'instant')} gap={dbg.get('gap', -1)} "
        f"net={dbg.get('net_dist', 0.0):.4f}/{dbg.get('target_dist', 0.0):.4f} "
        f"acc={dbg.get('accum_score', 0.0):.4f}/{dbg.get('accum_threshold', 0.0):.4f} "
        f"turn={dbg.get('turn_deg', 0.0):.1f} rot={dbg.get('rot_deg', 0.0):.1f} "
        f"lin={dbg.get('linearity', 0.0):.2f} jitter={int(bool(dbg.get('jitter', False)))}"
    )
    if "ref_reason" in dbg:
        msg += (
            f" ref={dbg.get('ref_reason', 'na')}"
            f" l1={dbg.get('ref_l1', 0.0):.4f}/{dbg.get('ref_last_l1', 0.0):.4f}"
            f" gl1={dbg.get('ref_grad', 0.0):.4f}/{dbg.get('ref_last_grad', 0.0):.4f}"
            f" er={dbg.get('ref_edge_ratio', 0.0):.2f}"
            f" cr={dbg.get('ref_contrast_ratio', 0.0):.2f}"
        )
    if "feat_reason" in dbg:
        msg += (
            f" feat={dbg.get('feat_reason', 'na')}"
            f" mae={dbg.get('feat_mae', 0.0):.4f}/{dbg.get('feat_last_mae', 0.0):.4f}"
            f" flow={dbg.get('feat_flow', 0.0):.3f}/{dbg.get('feat_last_flow', 0.0):.3f}"
            f" grad={dbg.get('feat_grad', 0.0):.2f}"
            f" p={dbg.get('feat_prob', 0.0):.3f}"
            f" ent={dbg.get('feat_entropy', 0.0):.3f}"
            f" er={dbg.get('feat_edge_ratio', 0.0):.2f}"
        )
    if "energy_reason" in dbg:
        msg += (
            f" energy={dbg.get('energy_reason', 'na')}"
            f" eratio={dbg.get('energy_ratio', 0.0):.3f}/{dbg.get('energy_min_ratio', 0.0):.3f}"
            f" tex={dbg.get('texture_ratio', 0.0):.3f}"
            f" fold={dbg.get('fold_ratio', 0.0):.3f}"
            f" bnd={dbg.get('fold_boundary_ratio', 0.0):.3f}/{dbg.get('energy_min_boundary_ratio', 0.0):.3f}"
            f" spec={dbg.get('specular_ratio', 0.0):.3f}/{dbg.get('energy_max_specular_ratio', 0.0):.3f}"
        )
    return msg


_FAILURE_DEBUG_FIELDS = [
    "frame_id",
    "is_keyframe",
    "trigger_reason",
    "kfs_mode",
    "attribution_guess",
    "tracking_loss",
    "tracking_residual_ema",
    "tracking_residual_ratio",
    "photometric_loss",
    "ssim_loss",
    "overlap_ratio",
    "num_visible_gaussians",
    "pose_delta_trans",
    "pose_delta_rot",
    "mean_brightness",
    "brightness_diff_to_kf",
    "blur_score",
    "specular_ratio",
    "texture_score",
    "gradient_energy",
    "optical_flow_magnitude",
    "reprojection_error",
    "residual_specular_overlap",
    "kf_gap",
    "kfs_step_dist",
    "kfs_step_rot_deg",
    "kfs_accum_motion",
    "kfs_accum_rot_deg",
    "kfs_accum_score",
    "kfs_accum_threshold",
    "kfs_net_dist",
    "kfs_target_dist",
    "kfs_path_len",
    "kfs_linearity",
    "kfs_turn_deg",
    "kfs_rot_deg",
    "kfs_dist_threshold",
    "kfs_straight",
    "kfs_jitter",
    "kfs_high_turn",
    "kfs_high_rot",
    "kfs_turn_threshold",
    "kfs_rot_threshold",
    "kfs_straight_turn_threshold",
    "kfs_straight_rot_threshold",
    "kfs_straight_linearity_threshold",
    "kfs_straight_dist_scale",
    "kfs_jitter_dist_ratio",
    "kfs_jitter_rot_threshold",
    "kfs_turn_min_dist_ratio",
    "ref_reason",
    "ref_best",
    "ref_l1",
    "ref_last_l1",
    "ref_grad",
    "ref_last_grad",
    "ref_edge_ratio",
    "ref_contrast_ratio",
    "ref_duplicate",
    "ref_low_quality",
    "ref_novel",
    "feat_reason",
    "feat_best",
    "feat_mae",
    "feat_last_mae",
    "feat_flow",
    "feat_last_flow",
    "feat_grad",
    "feat_last_grad",
    "feat_prob",
    "feat_entropy",
    "feat_edge_ratio",
    "feat_contrast",
    "feat_duplicate",
    "feat_low_texture",
    "feat_bad_corr",
    "energy_reason",
    "energy_gate_mode",
    "energy_reliable",
    "energy_ratio",
    "energy_min_ratio",
    "energy_mean",
    "texture_ratio",
    "fold_ratio",
    "fold_boundary_ratio",
    "energy_min_boundary_ratio",
    "energy_max_specular_ratio",
]


def _failure_float(value, default=np.nan) -> float:
    try:
        if torch.is_tensor(value):
            value = value.detach().cpu().item()
        value = float(value)
        if not np.isfinite(value):
            return default
        return value
    except Exception:
        return default


def _failure_to_np_img(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().float().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


def _failure_gray_np(image: torch.Tensor) -> np.ndarray:
    rgb = image.detach().float().clamp(0.0, 1.0)
    gray = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return gray.cpu().numpy()


def _failure_gradient_and_blur(gray: torch.Tensor) -> tuple[float, float]:
    gray = gray.detach().float()[None, None]
    device = gray.device
    dtype = gray.dtype
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype).view(1, 1, 3, 3) / 8.0
    sobel_y = sobel_x.transpose(-1, -2)
    lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=device, dtype=dtype).view(1, 1, 3, 3)
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    grad = torch.sqrt(gx * gx + gy * gy + 1e-12)
    lap_im = F.conv2d(gray, lap, padding=1)
    return _failure_float(grad.mean()), _failure_float(lap_im.var())


def _failure_flow_map(curr_gray: np.ndarray, ref_gray: np.ndarray, downsample: int) -> tuple[float, np.ndarray]:
    try:
        import cv2

        h, w = curr_gray.shape
        scale = max(1, int(downsample))
        small_size = (max(8, w // scale), max(8, h // scale))
        curr_small = cv2.resize((curr_gray * 255).astype(np.uint8), small_size, interpolation=cv2.INTER_AREA)
        ref_small = cv2.resize((ref_gray * 255).astype(np.uint8), small_size, interpolation=cv2.INTER_AREA)
        flow = cv2.calcOpticalFlowFarneback(
            ref_small,
            curr_small,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        mag = np.linalg.norm(flow, axis=-1)
        mag_full = cv2.resize(mag, (w, h), interpolation=cv2.INTER_LINEAR)
        return float(np.nanmean(mag)), mag_full
    except Exception:
        return np.nan, np.zeros_like(curr_gray, dtype=np.float32)


def _failure_pose_delta(curr_w2c: torch.Tensor, ref_w2c: torch.Tensor | None) -> tuple[float, float]:
    if ref_w2c is None:
        return 0.0, 0.0
    try:
        curr = curr_w2c.detach().cpu().numpy()
        ref = ref_w2c.detach().cpu().numpy()
        trans = float(np.linalg.norm(curr[:3, 3] - ref[:3, 3]))
        rel = curr[:3, :3] @ ref[:3, :3].T
        c = float(np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0))
        rot = float(np.degrees(np.arccos(c)))
        return trans, rot
    except Exception:
        return np.nan, np.nan


def _failure_append_csv(csv_path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FAILURE_DEBUG_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in _FAILURE_DEBUG_FIELDS})


def _failure_attribution_guess(row: dict) -> str:
    photo = row.get("photometric_loss", np.nan)
    overlap = row.get("overlap_ratio", np.nan)
    visible = row.get("num_visible_gaussians", 0)
    pose_t = row.get("pose_delta_trans", np.nan)
    bright = row.get("brightness_diff_to_kf", np.nan)
    spec_overlap = row.get("residual_specular_overlap", np.nan)
    if np.isfinite(overlap) and (overlap < 0.25 or visible < 200):
        return "geometry_low_overlap"
    if np.isfinite(pose_t) and np.isfinite(photo) and pose_t > 0.12 and photo > 0.08:
        return "geometry_pose_jump"
    if np.isfinite(photo) and photo > 0.08 and np.isfinite(spec_overlap) and spec_overlap > 0.35:
        return "appearance_specular"
    if np.isfinite(photo) and photo > 0.08 and np.isfinite(bright) and bright > 0.08:
        return "appearance_brightness"
    if np.isfinite(photo) and photo > 0.08 and np.isfinite(overlap) and overlap > 0.45:
        return "appearance_or_local_texture"
    return "normal_or_mixed"


def _failure_save_panel(panel_path: str, images: dict, row: dict, recent_rows: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(os.path.dirname(panel_path), exist_ok=True)
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
        axes = axes.ravel()
        axes[0].imshow(images["current"])
        axes[0].set_title("current RGB")
        axes[1].imshow(images["rendered"])
        axes[1].set_title("rendered RGB")
        im2 = axes[2].imshow(images["residual"], cmap="magma")
        axes[2].set_title("|current-rendered|")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.02)
        axes[3].imshow(images["specular"], cmap="gray", vmin=0, vmax=1)
        axes[3].set_title("specular mask")
        im4 = axes[4].imshow(images["flow"], cmap="viridis")
        axes[4].set_title("flow magnitude")
        fig.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.02)

        ax = axes[5]
        factor_text = "\n".join(
            [
                f"decision: kf={int(bool(row.get('is_keyframe', False)))} reason={row.get('trigger_reason', '')}",
                f"motion: gap={row.get('kf_gap')} net={_failure_float(row.get('kfs_net_dist')):.4f} "
                f"target={_failure_float(row.get('kfs_target_dist')):.4f} "
                f"dist_th={_failure_float(row.get('kfs_dist_threshold')):.4f}",
                f"angles: turn={_failure_float(row.get('kfs_turn_deg')):.1f}/"
                f"{_failure_float(row.get('kfs_turn_threshold')):.1f} "
                f"rot={_failure_float(row.get('kfs_rot_deg')):.1f}/"
                f"{_failure_float(row.get('kfs_rot_threshold')):.1f} "
                f"pose_rot={_failure_float(row.get('pose_delta_rot')):.1f}",
                f"shape: path={_failure_float(row.get('kfs_path_len')):.4f} "
                f"lin={_failure_float(row.get('kfs_linearity')):.2f} "
                f"straight={row.get('kfs_straight')} jitter={row.get('kfs_jitter')}",
                f"render: photo={_failure_float(row.get('photometric_loss')):.4f} "
                f"ssim_loss={_failure_float(row.get('ssim_loss')):.4f} "
                f"overlap={_failure_float(row.get('overlap_ratio')):.3f} "
                f"visible={row.get('num_visible_gaussians')}",
                f"appearance: bright={_failure_float(row.get('mean_brightness')):.3f} "
                f"diff_kf={_failure_float(row.get('brightness_diff_to_kf')):.3f} "
                f"spec={_failure_float(row.get('specular_ratio')):.3f} "
                f"resid_spec={_failure_float(row.get('residual_specular_overlap')):.3f}",
                f"texture/flow: grad={_failure_float(row.get('gradient_energy')):.4f} "
                f"blur={_failure_float(row.get('blur_score')):.6f} "
                f"flow={_failure_float(row.get('optical_flow_magnitude')):.3f}",
                f"ref gate: {row.get('ref_reason', '')} l1={_failure_float(row.get('ref_l1')):.4f}/"
                f"{_failure_float(row.get('ref_last_l1')):.4f} er={_failure_float(row.get('ref_edge_ratio')):.2f} "
                f"cr={_failure_float(row.get('ref_contrast_ratio')):.2f}",
                f"feat gate: {row.get('feat_reason', '')} mae={_failure_float(row.get('feat_mae')):.4f}/"
                f"{_failure_float(row.get('feat_last_mae')):.4f} "
                f"prob={_failure_float(row.get('feat_prob')):.3f} ent={_failure_float(row.get('feat_entropy')):.3f}",
                f"energy gate: {row.get('energy_reason', '')} "
                f"eratio={_failure_float(row.get('energy_ratio')):.3f}/"
                f"{_failure_float(row.get('energy_min_ratio')):.3f} "
                f"tex={_failure_float(row.get('texture_ratio')):.3f} "
                f"fold={_failure_float(row.get('fold_ratio')):.3f} "
                f"bnd={_failure_float(row.get('fold_boundary_ratio')):.3f}/"
                f"{_failure_float(row.get('energy_min_boundary_ratio')):.3f}",
                f"guess: {row.get('attribution_guess', '')}",
            ]
        )
        ax.axis("off")
        ax.text(0.02, 0.98, factor_text, va="top", ha="left", fontsize=8, family="monospace")
        for ax in axes[:5]:
            ax.axis("off")
        fig.suptitle(
            f"frame {row.get('frame_id')} kf={int(bool(row.get('is_keyframe', False)))} "
            f"photo={_failure_float(row.get('photometric_loss')):.4f} "
            f"overlap={_failure_float(row.get('overlap_ratio')):.3f} "
            f"visible={row.get('num_visible_gaussians')}",
            fontsize=12,
        )
        fig.savefig(panel_path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"[FailureDebug] panel save failed: {exc}")


def _failure_collect_frame_debug(
    params: dict,
    curr_data: dict,
    variables: dict,
    time_idx: int,
    keyframe_list: list,
    add_keyframe: bool,
    motion_debug: dict | None,
    tracking_residual_curr: float | None,
    tracking_residual_ema: float | None,
    config: dict,
):
    cfg = config.get("failure_debug", {})
    with torch.no_grad():
        transformed_pts = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
        rendervar = transformed_params2rendervar(params, transformed_pts)
        im, radius, _ = Renderer(raster_settings=curr_data["cam"])(**rendervar)
        depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data["w2c"], transformed_pts)
        depth_sil, _, _ = Renderer(raster_settings=curr_data["cam"])(**depth_sil_rendervar)
        render_depth = depth_sil[0:1]
        silhouette = depth_sil[1]

        current = curr_data["im"].detach().float().clamp(0.0, 1.0)
        rendered = im.detach().float().clamp(0.0, 1.0)
        residual = torch.abs(current - rendered).mean(dim=0)
        gray = 0.299 * current[0] + 0.587 * current[1] + 0.114 * current[2]
        spec_th = float(cfg.get("specular_threshold", 0.92))
        spec_mask = (current.max(dim=0).values > spec_th) | (gray > spec_th)
        residual_hi = residual > torch.quantile(residual.flatten(), 0.90)
        residual_specular_overlap = (
            (residual_hi & spec_mask).float().sum() / (residual_hi.float().sum() + 1e-6)
        )
        grad_energy, blur_score = _failure_gradient_and_blur(gray)
        ref_color = keyframe_list[-1]["color"] if len(keyframe_list) > 0 else current
        ref_gray_np = _failure_gray_np(ref_color)
        curr_gray_np = gray.detach().cpu().numpy()
        flow_mag, flow_map = _failure_flow_map(
            curr_gray_np,
            ref_gray_np,
            int(cfg.get("flow_downsample", 4)),
        )
        curr_w2c = _current_w2c_tensor(params, time_idx)
        ref_w2c = keyframe_list[-1].get("est_w2c", None) if len(keyframe_list) > 0 else None
        pose_delta_trans, pose_delta_rot = _failure_pose_delta(curr_w2c, ref_w2c)
        valid_depth = curr_data["depth"] > 0
        reproj_mask = valid_depth & (silhouette[None] > float(config["tracking"].get("sil_thres", 0.99)))
        if bool(reproj_mask.any()):
            reproj_error = torch.abs(curr_data["depth"] - render_depth)[reproj_mask].mean()
        else:
            reproj_error = torch.abs(curr_data["depth"] - render_depth)[valid_depth].mean() if bool(valid_depth.any()) else torch.tensor(float("nan"), device=current.device)

    motion_debug = motion_debug or {}
    tracking_loss = _failure_float(tracking_residual_curr)
    ema = _failure_float(tracking_residual_ema)
    ratio = tracking_loss / (ema + 1e-12) if np.isfinite(tracking_loss) and np.isfinite(ema) and ema > 0 else np.nan
    ref_brightness = float(ref_gray_np.mean()) if ref_gray_np.size > 0 else float(curr_gray_np.mean())
    row = {
        "frame_id": int(time_idx),
        "is_keyframe": bool(add_keyframe),
        "trigger_reason": motion_debug.get("reason", "fixed_interval" if add_keyframe else "hold"),
        "kfs_mode": motion_debug.get("mode", ""),
        "tracking_loss": tracking_loss,
        "tracking_residual_ema": ema,
        "tracking_residual_ratio": ratio,
        "photometric_loss": _failure_float(residual.mean()),
        "ssim_loss": _failure_float(1.0 - calc_ssim(rendered, current)),
        "overlap_ratio": _failure_float((silhouette > float(config["tracking"].get("sil_thres", 0.99))).float().mean()),
        "num_visible_gaussians": int((radius > 0).sum().detach().cpu().item()),
        "pose_delta_trans": pose_delta_trans,
        "pose_delta_rot": pose_delta_rot,
        "mean_brightness": float(curr_gray_np.mean()),
        "brightness_diff_to_kf": abs(float(curr_gray_np.mean()) - ref_brightness),
        "blur_score": blur_score,
        "specular_ratio": _failure_float(spec_mask.float().mean()),
        "texture_score": grad_energy,
        "gradient_energy": grad_energy,
        "optical_flow_magnitude": flow_mag,
        "reprojection_error": _failure_float(reproj_error),
        "residual_specular_overlap": _failure_float(residual_specular_overlap),
        "kf_gap": motion_debug.get("gap", np.nan),
        "kfs_step_dist": motion_debug.get("step_dist", np.nan),
        "kfs_step_rot_deg": motion_debug.get("step_rot_deg", np.nan),
        "kfs_accum_motion": motion_debug.get("accum_motion", np.nan),
        "kfs_accum_rot_deg": motion_debug.get("accum_rot_deg", np.nan),
        "kfs_accum_score": motion_debug.get("accum_score", np.nan),
        "kfs_accum_threshold": motion_debug.get("accum_threshold", np.nan),
        "kfs_net_dist": motion_debug.get("net_dist", np.nan),
        "kfs_target_dist": motion_debug.get("target_dist", np.nan),
        "kfs_dist_threshold": motion_debug.get("dist_threshold", np.nan),
        "kfs_path_len": motion_debug.get("path_len", np.nan),
        "kfs_linearity": motion_debug.get("linearity", np.nan),
        "kfs_turn_deg": motion_debug.get("turn_deg", np.nan),
        "kfs_rot_deg": motion_debug.get("rot_deg", np.nan),
        "kfs_straight": int(bool(motion_debug.get("straight", False))),
        "kfs_jitter": int(bool(motion_debug.get("jitter", False))),
        "kfs_high_turn": int(bool(motion_debug.get("high_turn", False))),
        "kfs_high_rot": int(bool(motion_debug.get("high_rot", False))),
        "kfs_turn_threshold": motion_debug.get("turn_threshold", np.nan),
        "kfs_rot_threshold": motion_debug.get("rot_threshold", np.nan),
        "kfs_straight_turn_threshold": motion_debug.get("straight_turn_threshold", np.nan),
        "kfs_straight_rot_threshold": motion_debug.get("straight_rot_threshold", np.nan),
        "kfs_straight_linearity_threshold": motion_debug.get("straight_linearity_threshold", np.nan),
        "kfs_straight_dist_scale": motion_debug.get("straight_dist_scale", np.nan),
        "kfs_jitter_dist_ratio": motion_debug.get("jitter_dist_ratio", np.nan),
        "kfs_jitter_rot_threshold": motion_debug.get("jitter_rot_threshold", np.nan),
        "kfs_turn_min_dist_ratio": motion_debug.get("turn_min_dist_ratio", np.nan),
        "ref_reason": motion_debug.get("ref_reason", ""),
        "ref_best": motion_debug.get("ref_best", ""),
        "ref_l1": motion_debug.get("ref_l1", np.nan),
        "ref_last_l1": motion_debug.get("ref_last_l1", np.nan),
        "ref_grad": motion_debug.get("ref_grad", np.nan),
        "ref_last_grad": motion_debug.get("ref_last_grad", np.nan),
        "ref_edge_ratio": motion_debug.get("ref_edge_ratio", np.nan),
        "ref_contrast_ratio": motion_debug.get("ref_contrast_ratio", np.nan),
        "ref_duplicate": int(bool(motion_debug.get("ref_duplicate", False))),
        "ref_low_quality": int(bool(motion_debug.get("ref_low_quality", False))),
        "ref_novel": int(bool(motion_debug.get("ref_novel", False))),
        "feat_reason": motion_debug.get("feat_reason", ""),
        "feat_best": motion_debug.get("feat_best", ""),
        "feat_mae": motion_debug.get("feat_mae", np.nan),
        "feat_last_mae": motion_debug.get("feat_last_mae", np.nan),
        "feat_flow": motion_debug.get("feat_flow", np.nan),
        "feat_last_flow": motion_debug.get("feat_last_flow", np.nan),
        "feat_grad": motion_debug.get("feat_grad", np.nan),
        "feat_last_grad": motion_debug.get("feat_last_grad", np.nan),
        "feat_prob": motion_debug.get("feat_prob", np.nan),
        "feat_entropy": motion_debug.get("feat_entropy", np.nan),
        "feat_edge_ratio": motion_debug.get("feat_edge_ratio", np.nan),
        "feat_contrast": motion_debug.get("feat_contrast", np.nan),
        "feat_duplicate": int(bool(motion_debug.get("feat_duplicate", False))),
        "feat_low_texture": int(bool(motion_debug.get("feat_low_texture", False))),
        "feat_bad_corr": int(bool(motion_debug.get("feat_bad_corr", False))),
        "energy_reason": motion_debug.get("energy_reason", ""),
        "energy_gate_mode": motion_debug.get("energy_gate_mode", ""),
        "energy_reliable": int(bool(motion_debug.get("energy_reliable", False))),
        "energy_ratio": motion_debug.get("energy_ratio", np.nan),
        "energy_min_ratio": motion_debug.get("energy_min_ratio", np.nan),
        "energy_mean": motion_debug.get("energy_mean", np.nan),
        "texture_ratio": motion_debug.get("texture_ratio", np.nan),
        "fold_ratio": motion_debug.get("fold_ratio", np.nan),
        "fold_boundary_ratio": motion_debug.get("fold_boundary_ratio", np.nan),
        "energy_min_boundary_ratio": motion_debug.get("energy_min_boundary_ratio", np.nan),
        "energy_max_specular_ratio": motion_debug.get("energy_max_specular_ratio", np.nan),
    }
    row["attribution_guess"] = _failure_attribution_guess(row)
    images = {
        "current": _failure_to_np_img(current),
        "rendered": _failure_to_np_img(rendered),
        "residual": residual.detach().cpu().numpy(),
        "specular": spec_mask.detach().cpu().numpy().astype(np.float32),
        "flow": flow_map,
    }
    return row, images


def _failure_should_save_panel(row: dict, cfg: dict, panel_count: int) -> bool:
    max_panels = int(cfg.get("max_panels", 80))
    if max_panels >= 0 and panel_count >= max_panels:
        return False
    panel_every = int(cfg.get("panel_every", 0))
    if panel_every > 0 and int(row.get("frame_id", 0)) % panel_every == 0:
        return True
    if bool(row.get("is_keyframe", False)):
        return True
    ratio_th = float(cfg.get("residual_ratio_threshold", 1.8))
    ratio = _failure_float(row.get("tracking_residual_ratio"))
    if np.isfinite(ratio) and ratio > ratio_th:
        return True
    photo_th = float(cfg.get("photometric_loss_threshold", 0.0))
    photo = _failure_float(row.get("photometric_loss"))
    if photo_th > 0.0 and np.isfinite(photo) and photo > photo_th:
        return True
    return False


def _tracking_kf_prior_loss(params: dict, time_idx: int, prior_w2c: torch.Tensor, scfg: dict) -> tuple[torch.Tensor, dict]:
    """Weak pose prior that lets selected keyframes directly condition tracking."""
    curr_w2c = _current_w2c_tensor(params, time_idx)
    prior_w2c = prior_w2c.to(curr_w2c.device, curr_w2c.dtype).detach()

    trans_scale = max(float(scfg.get("tracking_kf_prior_trans_scale", 0.05)), 1e-8)
    trans_delta = (curr_w2c[:3, 3] - prior_w2c[:3, 3]) / trans_scale
    trans_loss = F.smooth_l1_loss(trans_delta, torch.zeros_like(trans_delta), reduction="mean")

    q_curr = F.normalize(params["cam_unnorm_rots"][..., time_idx]).reshape(-1)[:4]
    q_prior = matrix_to_quaternion(prior_w2c[:3, :3].unsqueeze(0)).reshape(-1)[:4].detach()
    q_prior = F.normalize(q_prior.to(q_curr.device, q_curr.dtype), dim=0)
    rot_cos = torch.abs(torch.sum(q_curr * q_prior)).clamp(0.0, 1.0)
    rot_loss = 1.0 - rot_cos

    w_t = float(scfg.get("tracking_kf_prior_trans_weight", 1.0))
    w_r = float(scfg.get("tracking_kf_prior_rot_weight", 0.1))
    loss = w_t * trans_loss + w_r * rot_loss
    stats = {
        "trans_loss": float(trans_loss.detach().cpu().item()),
        "rot_loss": float(rot_loss.detach().cpu().item()),
        "trans_norm": float((curr_w2c[:3, 3] - prior_w2c[:3, 3]).detach().norm().cpu().item()),
        "rot_cos": float(rot_cos.detach().cpu().item()),
    }
    return loss, stats


def _tracking_kf_prior_gate(
    time_idx: int,
    iter_idx: int,
    base_loss,
    tracking_residual_ema: float | None,
    pose_demand_debug: dict,
    accumulated_residual: float,
    scfg: dict,
) -> tuple[float, str]:
    mode = str(scfg.get("tracking_kf_prior_gate_mode", "always")).lower()
    if mode in ("always", "all"):
        return 1.0, "always"
    if mode in ("off", "none", "disabled"):
        return 0.0, "disabled"

    start_iter = int(scfg.get("tracking_kf_prior_start_iter", 0))
    if iter_idx < start_iter:
        return 0.0, f"iter<{start_iter}"
    warmup = int(scfg.get("tracking_kf_prior_gate_warmup_frames", 30))
    if time_idx < warmup:
        return 0.0, f"warmup<{warmup}"

    base_loss_value = _float_loss_value(base_loss)
    ema = float(tracking_residual_ema or 0.0)
    if not np.isfinite(base_loss_value) or ema <= 1e-12:
        return 0.0, "no_residual_ema"
    residual_ratio = float(base_loss_value / (ema + 1e-12))

    ratio_th = float(scfg.get("tracking_kf_prior_residual_ratio_threshold", scfg.get("pose_demand_residual_ratio_threshold", 1.35)))
    accum_th = float(scfg.get("tracking_kf_prior_accum_threshold", scfg.get("pose_demand_accumulated_residual_threshold", 2.0)))
    flags = pose_demand_debug.get("flags", {}) if isinstance(pose_demand_debug, dict) else {}
    flag_residual = bool(flags.get("residual", False))
    flag_accum = bool(flags.get("accumulated", False))
    residual_hit = residual_ratio >= ratio_th
    accum_hit = float(accumulated_residual) >= accum_th

    if mode in ("residual", "residual_only"):
        triggered = residual_hit
    elif mode in ("accum", "accumulated"):
        triggered = accum_hit
    elif mode in ("pose_demand", "demand"):
        triggered = flag_residual or flag_accum
    else:
        triggered = residual_hit or accum_hit or flag_residual or flag_accum

    if not triggered:
        return 0.0, f"low_residual(r={residual_ratio:.2f},a={accumulated_residual:.2f})"

    hi = max(float(scfg.get("tracking_kf_prior_residual_ratio_hi", ratio_th + 1.0)), ratio_th + 1e-4)
    min_scale = float(np.clip(scfg.get("tracking_kf_prior_gate_min_scale", 0.25), 0.0, 1.0))
    max_scale = float(np.clip(scfg.get("tracking_kf_prior_gate_max_scale", 1.0), 0.0, 1.0))
    if residual_hit:
        t = float(np.clip((residual_ratio - ratio_th) / (hi - ratio_th), 0.0, 1.0))
        scale = min_scale + t * (max_scale - min_scale)
    else:
        scale = min_scale
    return float(scale), f"residual_gate(r={residual_ratio:.2f},a={accumulated_residual:.2f})"


@torch.no_grad()
def _build_tracking_kf_prior(
    scorenet_model,
    color: torch.Tensor,
    keyframe_list: list,
    params: dict,
    time_idx: int,
    pose_step_ema: float | None,
    config: dict,
    device: torch.device,
) -> tuple[torch.Tensor | None, dict]:
    """Estimate a fused current pose from selected keyframes for tracking-time prior."""
    scfg = config.get("scorenet", {})
    if (not bool(scfg.get("tracking_kf_prior_enable", False))) or scorenet_model is None:
        return None, {"reason": "disabled"}
    if len(keyframe_list) == 0:
        return None, {"reason": "no_keyframes"}

    candidate_window = max(int(scfg.get("tracking_kf_prior_candidate_window", scfg.get("candidate_window", 20))), 1)
    cand_kf = keyframe_list[-candidate_window:]
    ref_ids = [int(kf["id"]) for kf in cand_kf]
    ref_imgs = [kf["color"].unsqueeze(0).to(device) for kf in cand_kf]
    target = color.unsqueeze(0).to(device)

    try:
        outputs = score_candidates(scorenet_model, target, ref_imgs, ref_ids)
        ref_w2c_list = [kf["est_w2c"].to(device) for kf in cand_kf]
        fused_w2c = estimate_w2c_from_outputs(
            outputs,
            ref_ids,
            ref_w2c_list,
            invert_relative=bool(scfg.get("pose_relative_invert_for_w2c", True)),
        )
    except Exception as exc:
        return None, {"reason": f"error={exc}"}

    if fused_w2c is None:
        return None, {"reason": "no_pose_edges", "ref_ids": ref_ids}

    proposal_ok, proposal_reason = _pose_proposal_is_sane(
        fused_w2c=fused_w2c,
        params=params,
        time_idx=time_idx,
        pose_step_ema=pose_step_ema,
        scfg=scfg,
    )
    selected = []
    if "selected_frame_ids" in outputs:
        selected = [int(x) for x in outputs["selected_frame_ids"][0].detach().cpu().numpy().tolist()]
    debug = {
        "reason": proposal_reason,
        "ok": bool(proposal_ok),
        "ref_ids": ref_ids,
        "selected": selected,
    }
    if not proposal_ok and bool(scfg.get("tracking_kf_prior_sanity_gate", True)):
        return None, debug
    return fused_w2c.detach(), debug


def _pose_proposal_is_sane(
    fused_w2c: torch.Tensor,
    params: dict,
    time_idx: int,
    pose_step_ema: float | None,
    scfg: dict,
) -> tuple[bool, str]:
    """Reject learned pose proposals that jump far beyond recent odometry scale."""
    if time_idx <= 0:
        return True, "first"
    try:
        proposal_t = fused_w2c[:3, 3].detach().cpu().numpy()
        prev_t = params["cam_trans"][..., time_idx - 1].detach().cpu().numpy()
        init_t = params["cam_trans"][..., time_idx].detach().cpu().numpy()
        proposal_step = float(np.linalg.norm(proposal_t - prev_t))
        init_step = float(np.linalg.norm(init_t - prev_t))
    except Exception as exc:
        return False, f"bad_tensor={exc}"
    ratio = float(scfg.get("pose_demand_proposal_max_step_ratio", 5.0))
    max_abs = float(scfg.get("pose_demand_proposal_max_abs_step", 1.0))
    ref_step = max(float(pose_step_ema or 0.0), init_step, 1e-4)
    limit = max_abs + ratio * ref_step
    if proposal_step > limit:
        return False, f"proposal_step={proposal_step:.4f}>limit={limit:.4f}"
    return True, f"proposal_step={proposal_step:.4f}"


def convert_params_to_store(params):
    params_to_store = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            params_to_store[k] = v.detach().clone()
        else:
            params_to_store[k] = v
    return params_to_store


def rgbd_slam(config: dict):
    # timer = Timer()
    # timer.start()
    
    # Print Config
    print("Loaded Config:")
    if "use_depth_loss_thres" not in config['tracking']:
        config['tracking']['use_depth_loss_thres'] = False
        config['tracking']['depth_loss_thres'] = 100000
    if "visualize_tracking_loss" not in config['tracking']:
        config['tracking']['visualize_tracking_loss'] = False
    if "scorenet" not in config:
        config["scorenet"] = dict(enable=False)
    if "enable" not in config["scorenet"]:
        config["scorenet"]["enable"] = False
    if "topk" not in config["scorenet"]:
        config["scorenet"]["topk"] = 5
    if "candidate_window" not in config["scorenet"]:
        config["scorenet"]["candidate_window"] = 20
    if "use_for_tracking_init" not in config["scorenet"]:
        config["scorenet"]["use_for_tracking_init"] = True
    if "use_for_keyframe_select" not in config["scorenet"]:
        config["scorenet"]["use_for_keyframe_select"] = True
    if "use_for_mapping_sampling" not in config["scorenet"]:
        config["scorenet"]["use_for_mapping_sampling"] = True
    if "sampling_temperature" not in config["scorenet"]:
        config["scorenet"]["sampling_temperature"] = 1.0
    if "mapping_candidate_scope" not in config["scorenet"]:
        # "all" keeps candidate pool consistent with baseline distance sampling.
        config["scorenet"]["mapping_candidate_scope"] = "all"
    if "mapping_use_mixed_sampling" not in config["scorenet"]:
        config["scorenet"]["mapping_use_mixed_sampling"] = True
    if "mapping_mix_alpha" not in config["scorenet"]:
        # mixed = alpha * distance + (1-alpha) * scorenet
        config["scorenet"]["mapping_mix_alpha"] = 0.5
    if "mapping_warmup_frames" not in config["scorenet"]:
        # During warmup, mapping sampling falls back to baseline distance sampling.
        config["scorenet"]["mapping_warmup_frames"] = 0
    if "mapping_dedup_keyframes" not in config["scorenet"]:
        # If True, each keyframe is sampled at most once per mapping window before replacement fallback.
        config["scorenet"]["mapping_dedup_keyframes"] = False
    if "mapping_score_keep_ratio" not in config["scorenet"]:
        # Keep top ratio of score candidates before softmax. 1.0 means no truncation.
        config["scorenet"]["mapping_score_keep_ratio"] = 1.0
    if "mapping_temporal_min_gap" not in config["scorenet"]:
        # Enforce temporal diversity in one mapping cycle: sampled keyframes should be >= this frame gap.
        config["scorenet"]["mapping_temporal_min_gap"] = 0
    if "pair_log1p_enable" not in config["scorenet"]:
        # Stabilize heavy-tailed proxy features before ScoreNet.
        config["scorenet"]["pair_log1p_enable"] = False
    if "pair_zscore_enable" not in config["scorenet"]:
        # Per-target z-score along candidate dimension T.
        config["scorenet"]["pair_zscore_enable"] = False
    if "pair_clip_value" not in config["scorenet"]:
        # Symmetric clipping after optional normalization.
        config["scorenet"]["pair_clip_value"] = 0.0
    if "mapping_mix_alpha_sched_enable" not in config["scorenet"]:
        # If enabled, alpha is scheduled over time (early: more baseline distance prior).
        config["scorenet"]["mapping_mix_alpha_sched_enable"] = False
    if "mapping_mix_alpha_sched_start_frame" not in config["scorenet"]:
        config["scorenet"]["mapping_mix_alpha_sched_start_frame"] = config["scorenet"]["mapping_warmup_frames"]
    if "mapping_mix_alpha_sched_end_frame" not in config["scorenet"]:
        config["scorenet"]["mapping_mix_alpha_sched_end_frame"] = config["scorenet"]["mapping_warmup_frames"] + 300
    if "mapping_mix_alpha_sched_start" not in config["scorenet"]:
        config["scorenet"]["mapping_mix_alpha_sched_start"] = min(0.95, max(config["scorenet"]["mapping_mix_alpha"], 0.85))
    if "mapping_mix_alpha_sched_end" not in config["scorenet"]:
        config["scorenet"]["mapping_mix_alpha_sched_end"] = config["scorenet"]["mapping_mix_alpha"]
    if "mapping_mix_alpha_sched_curve" not in config["scorenet"]:
        config["scorenet"]["mapping_mix_alpha_sched_curve"] = "linear"
    if "tracking_residual_gate_enable" not in config["scorenet"]:
        # If enabled, high tracking residual frames down-weight score contribution in mapping sampling.
        config["scorenet"]["tracking_residual_gate_enable"] = False
    if "tracking_residual_gate_ema_beta" not in config["scorenet"]:
        config["scorenet"]["tracking_residual_gate_ema_beta"] = 0.9
    if "tracking_residual_gate_ratio_hi" not in config["scorenet"]:
        config["scorenet"]["tracking_residual_gate_ratio_hi"] = 2.0
    if "tracking_residual_gate_min" not in config["scorenet"]:
        config["scorenet"]["tracking_residual_gate_min"] = 0.2
    if "tracking_residual_gate_warmup_frames" not in config["scorenet"]:
        config["scorenet"]["tracking_residual_gate_warmup_frames"] = 30
    if "pose_demand_enable" not in config["scorenet"]:
        config["scorenet"]["pose_demand_enable"] = False
    if "pose_demand_warmup_frames" not in config["scorenet"]:
        config["scorenet"]["pose_demand_warmup_frames"] = 1
    if "pose_demand_motion_threshold" not in config["scorenet"]:
        config["scorenet"]["pose_demand_motion_threshold"] = 0.03
    if "pose_demand_residual_ratio_threshold" not in config["scorenet"]:
        config["scorenet"]["pose_demand_residual_ratio_threshold"] = 1.35
    if "pose_demand_uncertainty_threshold" not in config["scorenet"]:
        config["scorenet"]["pose_demand_uncertainty_threshold"] = 1.8
    if "pose_demand_accumulated_residual_threshold" not in config["scorenet"]:
        config["scorenet"]["pose_demand_accumulated_residual_threshold"] = 2.0
    if "pose_demand_max_gap" not in config["scorenet"]:
        config["scorenet"]["pose_demand_max_gap"] = 12
    if "pose_demand_score_threshold" not in config["scorenet"]:
        config["scorenet"]["pose_demand_score_threshold"] = 1.0
    if "pose_demand_accum_decay" not in config["scorenet"]:
        config["scorenet"]["pose_demand_accum_decay"] = 0.85
    if "pose_demand_log_every" not in config["scorenet"]:
        config["scorenet"]["pose_demand_log_every"] = 25
    if "pose_demand_action" not in config["scorenet"]:
        config["scorenet"]["pose_demand_action"] = "pose_init"
    if "pose_demand_tracking_extra_iters" not in config["scorenet"]:
        config["scorenet"]["pose_demand_tracking_extra_iters"] = 0
    if "pose_demand_tracking_iter_multiplier" not in config["scorenet"]:
        config["scorenet"]["pose_demand_tracking_iter_multiplier"] = 1.0
    if "pose_demand_tracking_strong_extra_iters" not in config["scorenet"]:
        config["scorenet"]["pose_demand_tracking_strong_extra_iters"] = config["scorenet"]["pose_demand_tracking_extra_iters"]
    if "pose_demand_tracking_medium_extra_iters" not in config["scorenet"]:
        config["scorenet"]["pose_demand_tracking_medium_extra_iters"] = 5
    if "pose_demand_tracking_weak_extra_iters" not in config["scorenet"]:
        config["scorenet"]["pose_demand_tracking_weak_extra_iters"] = 2
    if "pose_demand_tracking_strong_iter_multiplier" not in config["scorenet"]:
        config["scorenet"]["pose_demand_tracking_strong_iter_multiplier"] = config["scorenet"]["pose_demand_tracking_iter_multiplier"]
    if "pose_demand_tracking_medium_iter_multiplier" not in config["scorenet"]:
        config["scorenet"]["pose_demand_tracking_medium_iter_multiplier"] = 1.0
    if "pose_demand_tracking_weak_iter_multiplier" not in config["scorenet"]:
        config["scorenet"]["pose_demand_tracking_weak_iter_multiplier"] = 1.0
    if "pose_demand_tracking_max_iters" not in config["scorenet"]:
        config["scorenet"]["pose_demand_tracking_max_iters"] = 60
    if "pose_demand_tracking_rollback_enable" not in config["scorenet"]:
        config["scorenet"]["pose_demand_tracking_rollback_enable"] = False
    if "pose_demand_tracking_rollback_min_rel_improve" not in config["scorenet"]:
        config["scorenet"]["pose_demand_tracking_rollback_min_rel_improve"] = 0.005
    if "pose_demand_tracking_rollback_min_abs_improve" not in config["scorenet"]:
        config["scorenet"]["pose_demand_tracking_rollback_min_abs_improve"] = 0.0
    if "pose_demand_tracking_rollback_max_step_ratio" not in config["scorenet"]:
        config["scorenet"]["pose_demand_tracking_rollback_max_step_ratio"] = 2.5
    if "pose_demand_tracking_rollback_max_abs_step" not in config["scorenet"]:
        config["scorenet"]["pose_demand_tracking_rollback_max_abs_step"] = 1.0
    if "pose_demand_quality_gate_enable" not in config["scorenet"]:
        config["scorenet"]["pose_demand_quality_gate_enable"] = False
    if "pose_demand_quality_gate_tiers" not in config["scorenet"]:
        config["scorenet"]["pose_demand_quality_gate_tiers"] = "medium,weak"
    if "pose_demand_quality_gate_near_duplicate_motion_ratio" not in config["scorenet"]:
        config["scorenet"]["pose_demand_quality_gate_near_duplicate_motion_ratio"] = 0.6
    if "pose_demand_quality_gate_low_residual_ratio" not in config["scorenet"]:
        config["scorenet"]["pose_demand_quality_gate_low_residual_ratio"] = 1.10
    if "pose_demand_quality_gate_low_accumulated_residual" not in config["scorenet"]:
        config["scorenet"]["pose_demand_quality_gate_low_accumulated_residual"] = 0.50
    if "pose_demand_quality_gate_block_score_only" not in config["scorenet"]:
        config["scorenet"]["pose_demand_quality_gate_block_score_only"] = True
    if "keyframe_motion_proxy_enable" not in config["scorenet"]:
        config["scorenet"]["keyframe_motion_proxy_enable"] = False
    if "keyframe_motion_proxy_base_gap" not in config["scorenet"]:
        config["scorenet"]["keyframe_motion_proxy_base_gap"] = config.get("keyframe_every", 8)
    if "keyframe_motion_proxy_min_gap" not in config["scorenet"]:
        config["scorenet"]["keyframe_motion_proxy_min_gap"] = max(2, int(config.get("keyframe_every", 8)) // 2)
    if "keyframe_motion_proxy_max_gap" not in config["scorenet"]:
        config["scorenet"]["keyframe_motion_proxy_max_gap"] = max(2, int(config.get("keyframe_every", 8)) * 2)
    if "keyframe_motion_proxy_target_dist_scale" not in config["scorenet"]:
        config["scorenet"]["keyframe_motion_proxy_target_dist_scale"] = 1.0
    if "keyframe_motion_proxy_min_target_dist" not in config["scorenet"]:
        config["scorenet"]["keyframe_motion_proxy_min_target_dist"] = 0.03
    if "keyframe_motion_proxy_turn_angle_deg" not in config["scorenet"]:
        config["scorenet"]["keyframe_motion_proxy_turn_angle_deg"] = 20.0
    if "keyframe_motion_proxy_rot_angle_deg" not in config["scorenet"]:
        config["scorenet"]["keyframe_motion_proxy_rot_angle_deg"] = 18.0
    if "keyframe_motion_proxy_straight_angle_deg" not in config["scorenet"]:
        config["scorenet"]["keyframe_motion_proxy_straight_angle_deg"] = 6.0
    if "keyframe_motion_proxy_straight_rot_deg" not in config["scorenet"]:
        config["scorenet"]["keyframe_motion_proxy_straight_rot_deg"] = 6.0
    if "keyframe_motion_proxy_straight_linearity" not in config["scorenet"]:
        config["scorenet"]["keyframe_motion_proxy_straight_linearity"] = 0.92
    if "keyframe_motion_proxy_straight_dist_scale" not in config["scorenet"]:
        config["scorenet"]["keyframe_motion_proxy_straight_dist_scale"] = 1.5
    if "keyframe_motion_proxy_jitter_dist_ratio" not in config["scorenet"]:
        config["scorenet"]["keyframe_motion_proxy_jitter_dist_ratio"] = 0.35
    if "keyframe_motion_proxy_jitter_rot_deg" not in config["scorenet"]:
        config["scorenet"]["keyframe_motion_proxy_jitter_rot_deg"] = 5.0
    if "keyframe_motion_proxy_turn_min_dist_ratio" not in config["scorenet"]:
        config["scorenet"]["keyframe_motion_proxy_turn_min_dist_ratio"] = 0.25
    if "keyframe_motion_proxy_log_every" not in config["scorenet"]:
        config["scorenet"]["keyframe_motion_proxy_log_every"] = 25
    if "keyframe_ref_pair_gate_enable" not in config["scorenet"]:
        config["scorenet"]["keyframe_ref_pair_gate_enable"] = False
    if "keyframe_ref_pair_recent_refs" not in config["scorenet"]:
        config["scorenet"]["keyframe_ref_pair_recent_refs"] = 3
    if "keyframe_ref_pair_downsample" not in config["scorenet"]:
        config["scorenet"]["keyframe_ref_pair_downsample"] = 96
    if "keyframe_ref_pair_duplicate_l1" not in config["scorenet"]:
        config["scorenet"]["keyframe_ref_pair_duplicate_l1"] = 0.025
    if "keyframe_ref_pair_duplicate_grad_l1" not in config["scorenet"]:
        config["scorenet"]["keyframe_ref_pair_duplicate_grad_l1"] = 0.020
    if "keyframe_ref_pair_add_l1" not in config["scorenet"]:
        config["scorenet"]["keyframe_ref_pair_add_l1"] = 0.060
    if "keyframe_ref_pair_add_grad_l1" not in config["scorenet"]:
        config["scorenet"]["keyframe_ref_pair_add_grad_l1"] = 0.035
    if "keyframe_ref_pair_min_edge_ratio" not in config["scorenet"]:
        config["scorenet"]["keyframe_ref_pair_min_edge_ratio"] = 0.65
    if "keyframe_ref_pair_min_contrast_ratio" not in config["scorenet"]:
        config["scorenet"]["keyframe_ref_pair_min_contrast_ratio"] = 0.60
    if "keyframe_ref_pair_allow_add" not in config["scorenet"]:
        config["scorenet"]["keyframe_ref_pair_allow_add"] = True
    if "keyframe_feature_pair_gate_enable" not in config["scorenet"]:
        config["scorenet"]["keyframe_feature_pair_gate_enable"] = False
    if "keyframe_graph_root" not in config["scorenet"]:
        config["scorenet"]["keyframe_graph_root"] = os.path.abspath(os.path.join(_BASE_DIR, '..', '..', 'modules'))
    if "keyframe_feature_pair_recent_refs" not in config["scorenet"]:
        config["scorenet"]["keyframe_feature_pair_recent_refs"] = 3
    if "keyframe_feature_pair_downsample" not in config["scorenet"]:
        config["scorenet"]["keyframe_feature_pair_downsample"] = 128
    if "keyframe_feature_pair_max_disp" not in config["scorenet"]:
        config["scorenet"]["keyframe_feature_pair_max_disp"] = config["scorenet"].get("pair_max_disp", 4)
    if "keyframe_feature_pair_corr_temp" not in config["scorenet"]:
        config["scorenet"]["keyframe_feature_pair_corr_temp"] = config["scorenet"].get("pair_corr_temp", 0.5)
    if "keyframe_feature_pair_duplicate_mae" not in config["scorenet"]:
        config["scorenet"]["keyframe_feature_pair_duplicate_mae"] = 0.025
    if "keyframe_feature_pair_duplicate_flow" not in config["scorenet"]:
        config["scorenet"]["keyframe_feature_pair_duplicate_flow"] = 0.10
    if "keyframe_feature_pair_duplicate_grad" not in config["scorenet"]:
        config["scorenet"]["keyframe_feature_pair_duplicate_grad"] = 0.92
    if "keyframe_feature_pair_min_edge_ratio" not in config["scorenet"]:
        config["scorenet"]["keyframe_feature_pair_min_edge_ratio"] = 0.65
    if "keyframe_feature_pair_min_prob_max" not in config["scorenet"]:
        config["scorenet"]["keyframe_feature_pair_min_prob_max"] = 0.015
    if "keyframe_feature_pair_max_entropy" not in config["scorenet"]:
        config["scorenet"]["keyframe_feature_pair_max_entropy"] = 0.985
    if "keyframe_feature_pair_block_duplicate" not in config["scorenet"]:
        config["scorenet"]["keyframe_feature_pair_block_duplicate"] = True
    if "keyframe_feature_pair_block_low_texture" not in config["scorenet"]:
        config["scorenet"]["keyframe_feature_pair_block_low_texture"] = True
    if "keyframe_feature_pair_block_bad_corr" not in config["scorenet"]:
        config["scorenet"]["keyframe_feature_pair_block_bad_corr"] = False
    if "tracking_kf_prior_enable" not in config["scorenet"]:
        config["scorenet"]["tracking_kf_prior_enable"] = False
    if "tracking_kf_prior_candidate_window" not in config["scorenet"]:
        config["scorenet"]["tracking_kf_prior_candidate_window"] = config["scorenet"].get("candidate_window", 20)
    if "tracking_kf_prior_trans_weight" not in config["scorenet"]:
        config["scorenet"]["tracking_kf_prior_trans_weight"] = 1.0
    if "tracking_kf_prior_rot_weight" not in config["scorenet"]:
        config["scorenet"]["tracking_kf_prior_rot_weight"] = 0.1
    if "tracking_kf_prior_trans_scale" not in config["scorenet"]:
        config["scorenet"]["tracking_kf_prior_trans_scale"] = 0.05
    if "tracking_kf_prior_sanity_gate" not in config["scorenet"]:
        config["scorenet"]["tracking_kf_prior_sanity_gate"] = True
    if "tracking_kf_prior_log_every" not in config["scorenet"]:
        config["scorenet"]["tracking_kf_prior_log_every"] = 25
    if "tracking_kf_prior_gate_mode" not in config["scorenet"]:
        config["scorenet"]["tracking_kf_prior_gate_mode"] = "always"
    if "tracking_kf_prior_start_iter" not in config["scorenet"]:
        config["scorenet"]["tracking_kf_prior_start_iter"] = 0
    if "tracking_kf_prior_gate_warmup_frames" not in config["scorenet"]:
        config["scorenet"]["tracking_kf_prior_gate_warmup_frames"] = config["scorenet"].get("tracking_residual_gate_warmup_frames", 30)
    if "tracking_kf_prior_residual_ratio_threshold" not in config["scorenet"]:
        config["scorenet"]["tracking_kf_prior_residual_ratio_threshold"] = config["scorenet"].get("pose_demand_residual_ratio_threshold", 1.35)
    if "tracking_kf_prior_accum_threshold" not in config["scorenet"]:
        config["scorenet"]["tracking_kf_prior_accum_threshold"] = config["scorenet"].get("pose_demand_accumulated_residual_threshold", 2.0)
    if "tracking_kf_prior_residual_ratio_hi" not in config["scorenet"]:
        config["scorenet"]["tracking_kf_prior_residual_ratio_hi"] = 2.0
    if "tracking_kf_prior_gate_min_scale" not in config["scorenet"]:
        config["scorenet"]["tracking_kf_prior_gate_min_scale"] = 0.25
    if "tracking_kf_prior_gate_max_scale" not in config["scorenet"]:
        config["scorenet"]["tracking_kf_prior_gate_max_scale"] = 1.0
    if "pose_relative_invert_for_w2c" not in config["scorenet"]:
        config["scorenet"]["pose_relative_invert_for_w2c"] = True
    if "pose_demand_proposal_max_step_ratio" not in config["scorenet"]:
        config["scorenet"]["pose_demand_proposal_max_step_ratio"] = 5.0
    if "pose_demand_proposal_max_abs_step" not in config["scorenet"]:
        config["scorenet"]["pose_demand_proposal_max_abs_step"] = 1.0
    for _k in ("pose_demand_w_motion", "pose_demand_w_residual", "pose_demand_w_uncertainty",
               "pose_demand_w_time", "pose_demand_w_accumulated"):
        if _k not in config["scorenet"]:
            config["scorenet"][_k] = 1.0
    if "mapping_hard_gate_enable" not in config["scorenet"]:
        # If enabled, bypass KFS and use exact baseline distance sampling when score/tracking/pose confidence is poor.
        config["scorenet"]["mapping_hard_gate_enable"] = False
    if "mapping_hard_gate_warmup_frames" not in config["scorenet"]:
        config["scorenet"]["mapping_hard_gate_warmup_frames"] = config["scorenet"]["mapping_warmup_frames"]
    if "mapping_hard_gate_min_top_prob" not in config["scorenet"]:
        config["scorenet"]["mapping_hard_gate_min_top_prob"] = 0.0
    if "mapping_hard_gate_max_entropy" not in config["scorenet"]:
        config["scorenet"]["mapping_hard_gate_max_entropy"] = 1.0
    if "mapping_hard_gate_min_tracking_gate" not in config["scorenet"]:
        config["scorenet"]["mapping_hard_gate_min_tracking_gate"] = 0.0
    if "mapping_hard_gate_max_pose_step_ratio" not in config["scorenet"]:
        config["scorenet"]["mapping_hard_gate_max_pose_step_ratio"] = 1e9
    if "mapping_hard_gate_log_every" not in config["scorenet"]:
        config["scorenet"]["mapping_hard_gate_log_every"] = 25
    if "use_pose_graph" not in config["scorenet"]:
        config["scorenet"]["use_pose_graph"] = False
    if "pose_graph_store_keyframe_edges" not in config["scorenet"]:
        config["scorenet"]["pose_graph_store_keyframe_edges"] = bool(config["scorenet"].get("use_pose_graph", False))
    if "pose_graph_use_stored_edges" not in config["scorenet"]:
        config["scorenet"]["pose_graph_use_stored_edges"] = bool(config["scorenet"].get("pose_graph_store_keyframe_edges", False))
    if "pose_graph_refresh_latest_edges" not in config["scorenet"]:
        config["scorenet"]["pose_graph_refresh_latest_edges"] = not bool(config["scorenet"].get("pose_graph_use_stored_edges", False))
    if "pose_graph_predict_missing_edges" not in config["scorenet"]:
        config["scorenet"]["pose_graph_predict_missing_edges"] = False
    if "pose_graph_log_edges" not in config["scorenet"]:
        config["scorenet"]["pose_graph_log_edges"] = False
    if "pose_graph_window" not in config["scorenet"]:
        config["scorenet"]["pose_graph_window"] = 20
    if "pose_graph_iters" not in config["scorenet"]:
        config["scorenet"]["pose_graph_iters"] = 20
    if "pose_graph_lr" not in config["scorenet"]:
        config["scorenet"]["pose_graph_lr"] = 0.005
    if "pose_graph_odom_w_t" not in config["scorenet"]:
        config["scorenet"]["pose_graph_odom_w_t"] = 5.0
    if "pose_graph_odom_w_r" not in config["scorenet"]:
        config["scorenet"]["pose_graph_odom_w_r"] = 5.0
    if "pose_graph_long_w_t" not in config["scorenet"]:
        config["scorenet"]["pose_graph_long_w_t"] = 1.0
    if "pose_graph_long_w_r" not in config["scorenet"]:
        config["scorenet"]["pose_graph_long_w_r"] = 1.0
    if "pose_graph_anchor_w_t" not in config["scorenet"]:
        config["scorenet"]["pose_graph_anchor_w_t"] = 20.0
    if "pose_graph_anchor_w_r" not in config["scorenet"]:
        config["scorenet"]["pose_graph_anchor_w_r"] = 20.0
    if "pose_graph_every_n_kf" not in config["scorenet"]:
        # Run local pose-graph optimization every N keyframe insertions.
        config["scorenet"]["pose_graph_every_n_kf"] = 1
    if "pose_graph_score_weight_gamma" not in config["scorenet"]:
        # Score-to-information scaling exponent for long edges.
        config["scorenet"]["pose_graph_score_weight_gamma"] = 1.0
    if "pose_graph_score_weight_min" not in config["scorenet"]:
        config["scorenet"]["pose_graph_score_weight_min"] = 1.0
    if "pose_graph_score_weight_max" not in config["scorenet"]:
        config["scorenet"]["pose_graph_score_weight_max"] = 1.0
    if "dynamic_mapping_iters" not in config:
        config["dynamic_mapping_iters"] = dict(enable=False)
    if "enable" not in config["dynamic_mapping_iters"]:
        config["dynamic_mapping_iters"]["enable"] = False
    if "base_gap" not in config["dynamic_mapping_iters"]:
        config["dynamic_mapping_iters"]["base_gap"] = config.get("keyframe_every", 8)
    if "power" not in config["dynamic_mapping_iters"]:
        config["dynamic_mapping_iters"]["power"] = 1.0
    if "max_scale" not in config["dynamic_mapping_iters"]:
        config["dynamic_mapping_iters"]["max_scale"] = 3.0
    if "min_iters" not in config["dynamic_mapping_iters"]:
        config["dynamic_mapping_iters"]["min_iters"] = int(config["mapping"]["num_iters"])
    if "max_iters" not in config["dynamic_mapping_iters"]:
        config["dynamic_mapping_iters"]["max_iters"] = int(config["mapping"]["num_iters"]) * 3
    if "iter_loss_logging" not in config:
        config["iter_loss_logging"] = dict(enable=False, path="")
    if "enable" not in config["iter_loss_logging"]:
        config["iter_loss_logging"]["enable"] = False
    if "path" not in config["iter_loss_logging"]:
        config["iter_loss_logging"]["path"] = ""
    print(f"{config}")

    # Create Output Directories
    output_dir = os.path.join(config["workdir"], config["run_name"])
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    iter_loss_handle, iter_loss_writer = _open_iter_loss_logger(eval_dir, config)

    # Get Device
    device = torch.device(config["primary_device"])

    # Load Dataset
    print("Loading Dataset ...")
    dataset_config = config["data"]
    if 'distance_keyframe_selection' not in config:
        config['distance_keyframe_selection'] = False
    if config['distance_keyframe_selection']:
        print("Using CDF Keyframe Selection. Note that \'mapping window size\' is useless.")
        if 'distance_current_frame_prob' not in config:
            config['distance_current_frame_prob'] = 0.5
    if 'gaussian_simplification' not in config:
        config['gaussian_simplification'] = True # simplified in paper
    if not config['gaussian_simplification']:
        print("Using Full Gaussian Representation, which may cause unstable optimization if not fully optimized.")
    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {}
        gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
    if "train_or_test" not in dataset_config:
        dataset_config["train_or_test"] = 'all'
    if "preload" not in dataset_config:
        dataset_config["preload"] = False
    if "ignore_bad" not in dataset_config:
        dataset_config["ignore_bad"] = False
    if "use_train_split" not in dataset_config:
        dataset_config["use_train_split"] = True
    if "densification_image_height" not in dataset_config:
        dataset_config["densification_image_height"] = dataset_config["desired_image_height"]
        dataset_config["densification_image_width"] = dataset_config["desired_image_width"]
        seperate_densification_res = False
    else:
        if dataset_config["densification_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["densification_image_width"] != dataset_config["desired_image_width"]:
            seperate_densification_res = True
        else:
            seperate_densification_res = False
    if "tracking_image_height" not in dataset_config:
        dataset_config["tracking_image_height"] = dataset_config["desired_image_height"]
        dataset_config["tracking_image_width"] = dataset_config["desired_image_width"]
    else:
        pass
    tracking_basedir = dataset_config.get("tracking_basedir", dataset_config["basedir"])
    seperate_tracking_res = (
        dataset_config["tracking_image_height"] != dataset_config["desired_image_height"]
        or dataset_config["tracking_image_width"] != dataset_config["desired_image_width"]
        or os.path.abspath(str(tracking_basedir)) != os.path.abspath(str(dataset_config["basedir"]))
    )
    # Poses are relative to the first frame
    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=device,
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
        train_or_test=dataset_config["train_or_test"]
    )
    num_frames = dataset_config["num_frames"]
    if num_frames == -1:
        num_frames = len(dataset)

    if dataset_config["train_or_test"] == 'train': # kind of ill implementation here. train_or_test should be 'all' or 'train'. If 'test', you view test set as full dataset.
        eval_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["desired_image_height"], # if you eval, you should keep reso as raw image.
            desired_width=dataset_config["desired_image_width"],
            device=device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
            train_or_test='test'
        )
    # Init seperate dataloader for densification if required
    if seperate_densification_res:
        densify_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["densification_image_height"],
            desired_width=dataset_config["densification_image_width"],
            device=device,
            relative_pose=True,
            preload = dataset_config["preload"],
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
            train_or_test=dataset_config["train_or_test"]
        )
        # Initialize Parameters, Canonical & Densification Camera parameters
        params, variables, intrinsics, first_frame_w2c, cam, \
            densify_intrinsics, densify_cam = initialize_first_timestep(dataset, num_frames,
                                                                        config['scene_radius_depth_ratio'],
                                                                        config['mean_sq_dist_method'],
                                                                        densify_dataset=densify_dataset, 
                                                                        use_simplification=config['gaussian_simplification'],
                                                                        fold_seg_cfg=config.get('fold_seg_loss', {}))
    else:
        # Initialize Parameters & Canoncial Camera parameters
        params, variables, intrinsics, first_frame_w2c, cam = initialize_first_timestep(dataset, num_frames, 
                                                                                        config['scene_radius_depth_ratio'],
                                                                                        config['mean_sq_dist_method'], 
                                                                                        use_simplification=config['gaussian_simplification'],
                                                                                        fold_seg_cfg=config.get('fold_seg_loss', {}))
    # Load ScoreNet wrapper if enabled
    scorenet_model = None
    if config["scorenet"]["enable"]:
        scorenet_model = load_scorenet_wrapper(config["scorenet"])
        scorenet_model = scorenet_model.to(device)
    
    # Init seperate dataloader for tracking if required
    if seperate_tracking_res:
        print(f"Using separate tracking dataset root: {tracking_basedir}")
        tracking_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=tracking_basedir,
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["tracking_image_height"],
            desired_width=dataset_config["tracking_image_width"],
            device=device,
            relative_pose=True,
            preload = dataset_config["preload"],
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
            train_or_test=dataset_config["train_or_test"]
        )
        tracking_color, _, tracking_intrinsics, _ = tracking_dataset[0]
        tracking_color = tracking_color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        tracking_intrinsics = tracking_intrinsics[:3, :3]
        tracking_cam = setup_camera(tracking_color.shape[2], tracking_color.shape[1], 
                                    tracking_intrinsics.cpu().numpy(), first_frame_w2c.detach().cpu().numpy(), 
                                    use_simplification=config['gaussian_simplification'])
    
    # Initialize list to keep track of Keyframes
    keyframe_list = []
    keyframe_time_indices = []
    
    # Init Variables to keep track of ground truth poses and runtimes
    gt_w2c_all_frames = []
    tracking_iter_time_sum = 0
    tracking_iter_time_count = 0
    mapping_iter_time_sum = 0
    mapping_iter_time_count = 0
    tracking_frame_time_sum = 0
    tracking_frame_time_count = 0
    mapping_frame_time_sum = 0
    mapping_frame_time_count = 0
    tracking_residual_ema = None
    tracking_residual_curr = None
    pose_step_ema = None
    pose_step_curr = None
    pose_demand_prev_color = None
    pose_demand_last_update_idx = 0
    pose_demand_accumulated_residual = 0.0
    pose_demand_last_residual_ratio = 0.0
    pose_demand_last_uncertainty = 0.0
    tracking_init_poses = load_tracking_init_poses(config, num_frames, device)
    failure_debug_cfg = config.get("failure_debug", {})
    failure_debug_enable = bool(failure_debug_cfg.get("enable", False))
    failure_debug_output_root = str(failure_debug_cfg.get("output_root", "") or "").strip()
    if failure_debug_output_root:
        failure_debug_dir = os.path.join(failure_debug_output_root, config["run_name"], "failure_debug")
    else:
        failure_debug_dir = os.path.join(config["workdir"], config["run_name"], "failure_debug")
    failure_debug_csv = os.path.join(failure_debug_dir, "frame_attribution.csv")
    failure_debug_recent_rows = []
    failure_debug_panel_count = 0

    # Load Checkpoint
    if config['load_checkpoint']:
        checkpoint_time_idx = config['checkpoint_time_idx']
        print(f"Loading Checkpoint for Frame {checkpoint_time_idx}")
        ckpt_path = os.path.join(config['workdir'], config['run_name'], f"params{checkpoint_time_idx}.npz")
        params = dict(np.load(ckpt_path, allow_pickle=True))
        params = {k: torch.tensor(params[k]).cuda().float().requires_grad_(True) for k in params.keys()}
        variables['max_2D_radius'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['means2D_gradient_accum'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['denom'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['timestep'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        # Load the keyframe time idx list
        keyframe_time_indices = np.load(os.path.join(config['workdir'], config['run_name'], f"keyframe_time_indices{checkpoint_time_idx}.npy"))
        keyframe_time_indices = keyframe_time_indices.tolist()
        # Update the ground truth poses list
        for time_idx in range(checkpoint_time_idx):
            # Load RGBD frames incrementally instead of all frames
            color, depth, _, gt_pose = dataset[time_idx]
            # Process poses
            gt_w2c = torch.linalg.inv(gt_pose)
            gt_w2c_all_frames.append(gt_w2c)
            # Initialize Keyframe List
            if time_idx in keyframe_time_indices:
                # Get the estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                color = color.permute(2, 0, 1) / 255
                depth = depth.permute(2, 0, 1)
                fold_seg = _load_fold_seg(dataset, time_idx, color.shape[1:], config.get('fold_seg_loss', {}), color.device)
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                if fold_seg is not None:
                    curr_keyframe['fold_seg'] = fold_seg
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
    else:
        checkpoint_time_idx = 0
    
    # timer.lap("all the config")
    
    # Iterate over Scan
    for time_idx in tqdm(range(checkpoint_time_idx, num_frames)):
        
        # timer.lap("iterating over frame "+str(time_idx), 0)
        
        print() # always show global iteration
        # Load RGBD frames incrementally instead of all frames
        color, depth, _, gt_pose = dataset[time_idx]
        # Process poses
        gt_w2c = torch.linalg.inv(gt_pose)
        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255
        depth = depth.permute(2, 0, 1)
        fold_seg = _load_fold_seg(dataset, time_idx, color.shape[1:], config.get('fold_seg_loss', {}), color.device)
        gt_w2c_all_frames.append(gt_w2c)
        curr_gt_w2c = gt_w2c_all_frames
        # Optimize only current time step for tracking
        iter_time_idx = time_idx
        # Initialize Mapping Data for selected frame
        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': iter_time_idx, 'intrinsics': intrinsics, 
                     'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
        curr_data['fold_seg_mapping_only'] = bool(
            config.get('fold_seg_loss', {}).get('mapping_only', False)
        )
        curr_data['deformation_fold_soft_alpha'] = float(
            config.get('temporal_deformation', {}).get('fold_soft_alpha', 0.5)
        )
        if fold_seg is not None:
            curr_data['fold_seg'] = fold_seg
            curr_data['fold_seg_weight'] = float(config.get('fold_seg_loss', {}).get('weight', 1.0))
            curr_data['fold_seg_loss_type'] = str(config.get('fold_seg_loss', {}).get('loss_type', 'l1'))
        tracking_residual_curr = None
        pose_step_curr = None
        
        # Initialize Data for Tracking
        if seperate_tracking_res:
            tracking_color, tracking_depth, _, _ = tracking_dataset[time_idx]
            tracking_color = tracking_color.permute(2, 0, 1) / 255
            tracking_depth = tracking_depth.permute(2, 0, 1)
            tracking_fold_seg = _load_fold_seg(
                tracking_dataset,
                time_idx,
                tracking_color.shape[1:],
                config.get('fold_seg_loss', {}),
                tracking_color.device,
            )
            tracking_curr_data = {'cam': tracking_cam, 'im': tracking_color, 'depth': tracking_depth, 'id': iter_time_idx,
                                  'intrinsics': tracking_intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
            tracking_curr_data['fold_seg_mapping_only'] = bool(
                config.get('fold_seg_loss', {}).get('mapping_only', False)
            )
            tracking_curr_data['deformation_fold_soft_alpha'] = float(
                config.get('temporal_deformation', {}).get('fold_soft_alpha', 0.5)
            )
            if tracking_fold_seg is not None:
                tracking_curr_data['fold_seg'] = tracking_fold_seg
                tracking_curr_data['fold_seg_weight'] = float(config.get('fold_seg_loss', {}).get('weight', 1.0))
                tracking_curr_data['fold_seg_loss_type'] = str(config.get('fold_seg_loss', {}).get('loss_type', 'l1'))
        else:
            tracking_curr_data = curr_data

        # Optimization Iterations
        num_iters_mapping = int(config['mapping']['num_iters'])
        mapping_iter_scale = 1.0
        
        # Initialize the camera pose for the current frame
        pose_demand_run = False
        pose_demand_score = 0.0
        pose_demand_reasons = []
        pose_demand_frames_since = 0
        pose_demand_debug = {}
        if time_idx > 0:
            params = initialize_camera_pose(params, time_idx, forward_prop=config['tracking']['forward_prop'])
            if apply_tracking_init_pose(
                params,
                tracking_init_poses,
                time_idx,
                config["tracking"].get("init_pose_apply_mode", "all"),
            ):
                log_every = int(config["tracking"].get("init_pose_log_every", 25))
                if log_every <= 0 or time_idx % log_every == 0:
                    print(f"\n[InitPose] applied external init at Frame {time_idx}")
            if bool(config["scorenet"].get("pose_demand_enable", False)):
                pose_demand_frames_since = int(time_idx - pose_demand_last_update_idx)
                pose_demand_run, pose_demand_score, pose_demand_reasons, pose_demand_debug = _pose_demand_should_update(
                    time_idx=time_idx,
                    curr_color=color,
                    prev_color=pose_demand_prev_color,
                    frames_since_last_update=pose_demand_frames_since,
                    residual_ratio=pose_demand_last_residual_ratio,
                    pose_uncertainty=pose_demand_last_uncertainty,
                    accumulated_residual=pose_demand_accumulated_residual,
                    scfg=config["scorenet"],
                )
                if pose_demand_run:
                    pose_demand_last_update_idx = time_idx
                log_every = int(config["scorenet"].get("pose_demand_log_every", 25))
                if log_every <= 0 or time_idx % log_every == 0:
                    status = "high" if pose_demand_run else "low"
                    print(
                        f"\n[PoseDemand] {status} at Frame {time_idx}: "
                        f"score={pose_demand_score:.3f}, frames_since={pose_demand_frames_since}, "
                        f"{', '.join(pose_demand_reasons) if pose_demand_reasons else 'no_trigger'}"
                    )
            else:
                pose_demand_run = True

            action = str(config["scorenet"].get("pose_demand_action", "pose_init")).lower()
            if (
                pose_demand_run
                and action == "pose_init"
                and scorenet_model is not None
                and config["scorenet"]["use_for_tracking_init"]
            ):
                # Learned pose init is kept for legacy ablations.  The safer
                # tracking-only variant sets pose_demand_action="tracking_iters".
                if True:
                    # Build candidate refs from recent keyframes
                    cand_kf = keyframe_list[-config["scorenet"]["candidate_window"]:]
                    if len(cand_kf) > 0:
                        ref_ids = [kf["id"] for kf in cand_kf]
                        ref_imgs = [kf["color"].unsqueeze(0).to(device) for kf in cand_kf]
                        target = color.unsqueeze(0).to(device)
                        outputs = score_candidates(scorenet_model, target, ref_imgs, ref_ids)
                        # Build ref w2c list
                        ref_w2c_list = [kf["est_w2c"].to(device) for kf in cand_kf]
                        fused_w2c = estimate_w2c_from_outputs(
                            outputs,
                            ref_ids,
                            ref_w2c_list,
                            invert_relative=bool(config["scorenet"].get("pose_relative_invert_for_w2c", True)),
                        )
                        if fused_w2c is not None:
                            proposal_ok, proposal_reason = _pose_proposal_is_sane(
                                fused_w2c=fused_w2c,
                                params=params,
                                time_idx=time_idx,
                                pose_step_ema=pose_step_ema,
                                scfg=config["scorenet"],
                            )
                            log_every = int(config["scorenet"].get("pose_demand_log_every", 25))
                            if proposal_ok:
                                # Update params cam pose from fused w2c
                                rot_q = matrix_to_quaternion(fused_w2c[:3, :3].unsqueeze(0))
                                with torch.no_grad():
                                    params['cam_unnorm_rots'][..., time_idx] = rot_q.detach()
                                    params['cam_trans'][..., time_idx] = fused_w2c[:3, 3].detach()
                                pose_demand_accumulated_residual = 0.0
                                if log_every <= 0 or time_idx % log_every == 0:
                                    print(
                                        f"\n[PoseDemand] pose init at Frame {time_idx}: "
                                        f"score={pose_demand_score:.3f}, {proposal_reason}"
                                    )
                            elif log_every <= 0 or time_idx % log_every == 0:
                                print(
                                    f"\n[PoseDemand] reject pose init at Frame {time_idx}: "
                                    f"score={pose_demand_score:.3f}, {proposal_reason}"
                                )

        # timer.lap("initialized data", 1)

        tracking_kf_prior_w2c = None
        tracking_kf_prior_debug = {}
        if time_idx > 0:
            tracking_kf_prior_w2c, tracking_kf_prior_debug = _build_tracking_kf_prior(
                scorenet_model=scorenet_model,
                color=color,
                keyframe_list=keyframe_list,
                params=params,
                time_idx=time_idx,
                pose_step_ema=pose_step_ema,
                config=config,
                device=device,
            )
            if bool(config["scorenet"].get("tracking_kf_prior_enable", False)):
                log_every = int(config["scorenet"].get("tracking_kf_prior_log_every", 25))
                if tracking_kf_prior_w2c is not None and (log_every <= 0 or time_idx % log_every == 0):
                    print(
                        f"\n[TrackingKFPrior] use at Frame {time_idx}: "
                        f"selected={tracking_kf_prior_debug.get('selected', [])} "
                        f"{tracking_kf_prior_debug.get('reason', '')}"
                    )
                elif log_every <= 0 or time_idx % log_every == 0:
                    print(
                        f"\n[TrackingKFPrior] skip at Frame {time_idx}: "
                        f"{tracking_kf_prior_debug.get('reason', 'unknown')}"
                    )

        # Tracking
        tracking_start_time = time.time()
        if time_idx > 0 and not config['tracking']['use_gt_poses']:
            # Reset Optimizer & Learning Rates for tracking
            optimizer = initialize_optimizer(params, config['tracking']['lrs'])
            # Keep Track of Best Candidate Rotation & Translation
            candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
            candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
            current_min_loss = float(1e20)
            # Tracking Optimization
            iter = 0
            do_continue_slam = False
            base_num_iters_tracking = config['tracking']['num_iters']
            num_iters_tracking = base_num_iters_tracking
            demand_tier = "none"
            if pose_demand_run and str(config["scorenet"].get("pose_demand_action", "")).lower() in ("tracking_iters", "tracking", "tracking_only"):
                num_iters_tracking, demand_tier = _pose_demand_tracking_iters(
                    num_iters_tracking,
                    pose_demand_debug,
                    config["scorenet"],
                )
                log_every = int(config["scorenet"].get("pose_demand_log_every", 25))
                if log_every <= 0 or time_idx % log_every == 0:
                    print(f"\n[PoseDemand] tracking tier={demand_tier} iters={num_iters_tracking} at Frame {time_idx}")
            rollback_tracking = (
                bool(config["scorenet"].get("pose_demand_tracking_rollback_enable", False))
                and num_iters_tracking > base_num_iters_tracking
            )
            base_cam_unnorm_rot = None
            base_cam_tran = None
            base_min_loss_value = None
            current_min_base_loss_for_residual = float(1e20)
            tracking_kf_prior_gate_logged = False
            progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
            while True:
                iter_start_time = time.time()
                # Loss for current frame
                loss, variables, losses = get_loss(params, tracking_curr_data, variables, iter_time_idx, config['tracking']['loss_weights'],
                                                   config['tracking']['use_sil_for_loss'], config['tracking']['sil_thres'],
                                                   config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'], tracking=True, 
                                                   plot_dir=eval_dir, visualize_tracking_loss=config['tracking']['visualize_tracking_loss'],
                                                   tracking_iteration=iter)
                base_tracking_loss = loss
                if tracking_kf_prior_w2c is not None:
                    prior_gate, prior_gate_reason = _tracking_kf_prior_gate(
                        time_idx=time_idx,
                        iter_idx=iter,
                        base_loss=base_tracking_loss,
                        tracking_residual_ema=tracking_residual_ema,
                        pose_demand_debug=pose_demand_debug,
                        accumulated_residual=pose_demand_accumulated_residual,
                        scfg=config["scorenet"],
                    )
                    if prior_gate > 0.0:
                        kf_prior_loss, kf_prior_stats = _tracking_kf_prior_loss(
                            params=params,
                            time_idx=time_idx,
                            prior_w2c=tracking_kf_prior_w2c,
                            scfg=config["scorenet"],
                        )
                        kf_prior_loss = prior_gate * kf_prior_loss
                        loss = loss + kf_prior_loss
                        losses["kf_prior"] = kf_prior_loss.detach()
                        log_every = int(config["scorenet"].get("tracking_kf_prior_log_every", 25))
                        if (not tracking_kf_prior_gate_logged) and (log_every <= 0 or time_idx % log_every == 0):
                            print(
                                f"\n[TrackingKFPrior] apply at Frame {time_idx}: "
                                f"gate={prior_gate:.2f} {prior_gate_reason} "
                                f"trans_norm={kf_prior_stats.get('trans_norm', 0.0):.4f}"
                            )
                            tracking_kf_prior_gate_logged = True
                _log_iter_loss(
                    iter_loss_writer,
                    "tracking",
                    time_idx,
                    iter_time_idx,
                    iter,
                    num_iters_tracking,
                    losses,
                    params,
                    len(keyframe_list),
                )
                # Backprop
                loss.backward()
                # Optimizer Update
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    # Save the best candidate rotation & translation
                    if loss < current_min_loss:
                        current_min_loss = loss
                        current_min_base_loss_for_residual = base_tracking_loss.detach().clone()
                        candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
                        candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
                    # Report Progress
                    if config['report_iter_progress']:
                        report_progress(params, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True)
                    else:
                        progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                tracking_iter_time_sum += iter_end_time - iter_start_time
                tracking_iter_time_count += 1
                # Check if we should stop tracking
                iter += 1
                if rollback_tracking and iter == base_num_iters_tracking:
                    base_cam_unnorm_rot = candidate_cam_unnorm_rot.detach().clone()
                    base_cam_tran = candidate_cam_tran.detach().clone()
                    base_min_loss_value = _float_loss_value(current_min_loss)
                if iter == num_iters_tracking:
                    if losses['depth'] < config['tracking']['depth_loss_thres'] and config['tracking']['use_depth_loss_thres']:
                        break
                    elif config['tracking']['use_depth_loss_thres'] and not do_continue_slam:
                        do_continue_slam = True
                        progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
                        num_iters_tracking = 2*num_iters_tracking
                    else:
                        break

            progress_bar.close()
            if rollback_tracking and base_cam_unnorm_rot is not None and base_cam_tran is not None:
                extra_min_loss_value = _float_loss_value(current_min_loss)
                prev_cam_tran = params['cam_trans'][..., time_idx - 1].detach()
                accept_extra, rollback_reason = _tracking_extra_should_accept(
                    base_loss=base_min_loss_value,
                    extra_loss=extra_min_loss_value,
                    base_tran=base_cam_tran,
                    extra_tran=candidate_cam_tran,
                    prev_tran=prev_cam_tran,
                    pose_step_ema=pose_step_ema,
                    scfg=config["scorenet"],
                )
                log_every = int(config["scorenet"].get("pose_demand_log_every", 25))
                if not accept_extra:
                    candidate_cam_unnorm_rot = base_cam_unnorm_rot
                    candidate_cam_tran = base_cam_tran
                    current_min_loss = base_min_loss_value
                if log_every <= 0 or time_idx % log_every == 0:
                    decision = "accept" if accept_extra else "rollback"
                    print(f"\n[PoseDemand] {decision} extra tracking at Frame {time_idx}: {rollback_reason}")
            # Copy over the best candidate rotation & translation
            with torch.no_grad():
                params['cam_unnorm_rots'][..., time_idx] = candidate_cam_unnorm_rot
                params['cam_trans'][..., time_idx] = candidate_cam_tran
            # Track per-frame residual for online score gating (lower score contribution on bad tracking frames).
            try:
                residual_source = current_min_base_loss_for_residual
                if torch.is_tensor(residual_source):
                    tracking_residual_curr = float(residual_source.detach().cpu().item())
                else:
                    tracking_residual_curr = float(residual_source)
            except Exception:
                tracking_residual_curr = None
            if tracking_residual_curr is not None and np.isfinite(tracking_residual_curr):
                beta = float(config["scorenet"]["tracking_residual_gate_ema_beta"])
                beta = float(np.clip(beta, 0.0, 0.9999))
                if tracking_residual_ema is None:
                    tracking_residual_ema = tracking_residual_curr
                else:
                    tracking_residual_ema = beta * tracking_residual_ema + (1.0 - beta) * tracking_residual_curr
                if tracking_residual_ema is not None and tracking_residual_ema > 1e-12:
                    pose_demand_last_residual_ratio = float(tracking_residual_curr / (tracking_residual_ema + 1e-12))
                    excess = max(0.0, pose_demand_last_residual_ratio - 1.0)
                    decay = float(np.clip(config["scorenet"].get("pose_demand_accum_decay", 0.85), 0.0, 0.9999))
                    pose_demand_accumulated_residual = decay * pose_demand_accumulated_residual + excess
            if time_idx > 0:
                try:
                    curr_tran = params['cam_trans'][..., time_idx].detach().cpu().numpy()
                    prev_tran = params['cam_trans'][..., time_idx - 1].detach().cpu().numpy()
                    pose_step_curr = float(np.linalg.norm(curr_tran - prev_tran))
                except Exception:
                    pose_step_curr = None
                if pose_step_curr is not None and np.isfinite(pose_step_curr):
                    beta = float(config["scorenet"].get("tracking_residual_gate_ema_beta", 0.9))
                    beta = float(np.clip(beta, 0.0, 0.9999))
                    if pose_step_ema is None:
                        pose_step_ema = pose_step_curr
                    else:
                        pose_step_ema = beta * pose_step_ema + (1.0 - beta) * pose_step_curr
                    if pose_step_ema is not None and pose_step_ema > 1e-12:
                        pose_demand_last_uncertainty = float(pose_step_curr / (pose_step_ema + 1e-12))
        elif time_idx > 0 and config['tracking']['use_gt_poses']:
            with torch.no_grad():
                # Get the ground truth pose relative to frame 0
                rel_w2c = curr_gt_w2c[-1]
                rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                rel_w2c_tran = rel_w2c[:3, 3].detach()
                # Update the camera parameters
                params['cam_unnorm_rots'][..., time_idx] = rel_w2c_rot_quat
                params['cam_trans'][..., time_idx] = rel_w2c_tran
            tracking_residual_curr = None
        # Update the runtime numbers
        tracking_end_time = time.time()
        tracking_frame_time_sum += tracking_end_time - tracking_start_time
        tracking_frame_time_count += 1

        # timer.lap("tracking done", 2)

        # Densification & KeyFrame-based Mapping
        if time_idx == 0 or (time_idx+1) % config['map_every'] == 0:
            num_iters_mapping, mapping_iter_scale, base_num_iters_mapping = _dynamic_mapping_num_iters(
                config,
                time_idx,
                len(keyframe_list),
            )
            if bool(config.get("dynamic_mapping_iters", {}).get("enable", False)):
                dyn_log_every = int(config.get("dynamic_mapping_iters", {}).get("log_every", 25))
                if dyn_log_every <= 0 or time_idx % dyn_log_every == 0:
                    print(
                        f"\n[DynamicMappingIters] Frame {time_idx}: "
                        f"kf={len(keyframe_list)} base={base_num_iters_mapping} "
                        f"iters={num_iters_mapping} scale={mapping_iter_scale:.3f}"
                    )
            # Densification
            if config['mapping']['add_new_gaussians'] and time_idx > 0:
                # Setup Data for Densification
                if seperate_densification_res:
                    # Load RGBD frames incrementally instead of all frames
                    densify_color, densify_depth, _, _ = densify_dataset[time_idx]
                    densify_color = densify_color.permute(2, 0, 1) / 255
                    densify_depth = densify_depth.permute(2, 0, 1)
                    densify_fold_seg = _load_fold_seg(
                        densify_dataset,
                        time_idx,
                        densify_color.shape[1:],
                        config.get('fold_seg_loss', {}),
                        densify_color.device,
                    )
                    densify_curr_data = {'cam': densify_cam, 'im': densify_color, 'depth': densify_depth, 'id': time_idx, 
                                 'intrinsics': densify_intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
                    if densify_fold_seg is not None:
                        densify_curr_data['fold_seg'] = densify_fold_seg
                        densify_curr_data['fold_seg_weight'] = float(config.get('fold_seg_loss', {}).get('weight', 1.0))
                        densify_curr_data['fold_seg_loss_type'] = str(config.get('fold_seg_loss', {}).get('loss_type', 'l1'))
                else:
                    densify_curr_data = curr_data

                # delete floating gaussians
                # params, variables = remove_floating_gaussians(params, variables, densify_curr_data, time_idx)
                
                # Add new Gaussians to the scene based on the Silhouette
                params, variables = add_new_gaussians(params, variables, densify_curr_data, 
                                                      config['mapping']['sil_thres'], time_idx,
                                                      config['mean_sq_dist_method'], 
                                                      config['gaussian_simplification'])
                post_num_pts = params['means3D'].shape[0]
            
            if not config['distance_keyframe_selection']:
                with torch.no_grad():
                    # Get the current estimated rotation & translation
                    curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                    curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                    curr_w2c = torch.eye(4).cuda().float()
                    curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                    curr_w2c[:3, 3] = curr_cam_tran
                    # Select Keyframes for Mapping
                    num_keyframes = config['mapping_window_size']-2
                    selected_keyframes = keyframe_selection_overlap(depth, curr_w2c, intrinsics, keyframe_list[:-1], num_keyframes)
                    selected_time_idx = [keyframe_list[frame_idx]['id'] for frame_idx in selected_keyframes]
                    if len(keyframe_list) > 0:
                        # Add last keyframe to the selected keyframes
                        selected_time_idx.append(keyframe_list[-1]['id'])
                        selected_keyframes.append(len(keyframe_list)-1)
                    # Add current frame to the selected keyframes
                    selected_time_idx.append(time_idx)
                    selected_keyframes.append(-1)
                    # Print the selected keyframes
                    print(f"\nSelected Keyframes at Frame {time_idx}: {selected_time_idx}")

            # Reset Optimizer & Learning Rates for Full Map Optimization
            optimizer = initialize_optimizer(params, config['mapping']['lrs']) 

            # timer.lap("Densification Done at frame "+str(time_idx), 3)

            # Mapping
            mapping_start_time = time.time()
            if num_iters_mapping > 0:
                progress_bar = tqdm(range(num_iters_mapping), desc=f"Mapping Time Step: {time_idx}")
                
            mapping_scorenet_cache = None
            if scorenet_model is not None and config["scorenet"]["use_for_mapping_sampling"] and (not config['distance_keyframe_selection']):
                candidate_indices = [i for i in selected_keyframes if i != -1]
                cand_kf = [keyframe_list[i] for i in candidate_indices]
                if len(cand_kf) > 0:
                    ref_ids = [kf["id"] for kf in cand_kf]
                    ref_imgs = [kf["color"].unsqueeze(0).to(device) for kf in cand_kf]
                    target = color.unsqueeze(0).to(device)
                    outputs = score_candidates(scorenet_model, target, ref_imgs, ref_ids)
                    mapping_scorenet_cache = (candidate_indices, ref_ids, outputs["scores"][0])

            actural_keyframe_ids = []
            for iter in range(num_iters_mapping):
                iter_start_time = time.time()
                selected_keyframe_slot = ""
                selected_frame_for_log = ""
                if not config['distance_keyframe_selection']:
                    # Randomly select a frame until current time step amongst keyframes
                    if scorenet_model is not None and config["scorenet"]["use_for_mapping_sampling"]:
                        if mapping_scorenet_cache is not None:
                            candidate_indices, ref_ids, scores = mapping_scorenet_cache
                            sampled = keyframe_sampling_scorenet(
                                ref_ids,
                                scores,
                                n_samples=1,
                                temperature=config["scorenet"]["sampling_temperature"],
                            )
                            selected_rand_keyframe_idx = candidate_indices[sampled[0]]
                        else:
                            selected_rand_keyframe_idx = -1
                    else:
                        rand_idx = np.random.randint(0, len(selected_keyframes))
                        selected_rand_keyframe_idx = selected_keyframes[rand_idx]
                    actural_keyframe_ids.append(selected_rand_keyframe_idx)
                    selected_keyframe_slot = selected_rand_keyframe_idx
                    if selected_rand_keyframe_idx == -1:
                        # Use Current Frame Data
                        iter_time_idx = time_idx
                        iter_color = color
                        iter_depth = depth
                        iter_fold_seg = fold_seg
                        selected_frame_for_log = time_idx
                    else:
                        # Use Keyframe Data
                        iter_time_idx = keyframe_list[selected_rand_keyframe_idx]['id']
                        iter_color = keyframe_list[selected_rand_keyframe_idx]['color']
                        iter_depth = keyframe_list[selected_rand_keyframe_idx]['depth']
                        iter_fold_seg = keyframe_list[selected_rand_keyframe_idx].get('fold_seg', None)
                        selected_frame_for_log = iter_time_idx
                else:
                    if len(actural_keyframe_ids) == 0:
                        if len(keyframe_list) > 0:
                            if scorenet_model is not None and config["scorenet"]["use_for_mapping_sampling"]:
                                if config["scorenet"]["mapping_candidate_scope"] == "all":
                                    cand_kf = keyframe_list
                                    cand_indices = list(range(len(keyframe_list)))
                                else:
                                    cand_kf = keyframe_list[-config["scorenet"]["candidate_window"]:]
                                    cand_indices = list(range(len(keyframe_list) - len(cand_kf), len(keyframe_list)))
                                ref_ids = [kf["id"] for kf in cand_kf]
                                ref_imgs = [kf["color"].unsqueeze(0).to(device) for kf in cand_kf]
                                target = color.unsqueeze(0).to(device)
                                outputs = score_candidates(scorenet_model, target, ref_imgs, ref_ids)
                                scores = outputs["scores"][0]
                                curr_position = params['cam_trans'][..., time_idx].detach().cpu()
                                warmup_frames = int(config["scorenet"]["mapping_warmup_frames"])
                                tracking_gate = _tracking_residual_gate(
                                    time_idx=time_idx,
                                    tracking_residual=(tracking_residual_curr if tracking_residual_curr is not None else 0.0),
                                    tracking_residual_ema=(tracking_residual_ema if tracking_residual_ema is not None else 0.0),
                                    scfg=config["scorenet"],
                                )
                                pose_step_ratio = None
                                if pose_step_curr is not None and pose_step_ema is not None and pose_step_ema > 1e-12:
                                    pose_step_ratio = float(pose_step_curr / (pose_step_ema + 1e-12))
                                hard_fallback, hard_reasons = _mapping_hard_gate_should_fallback(
                                    time_idx=time_idx,
                                    scores=scores,
                                    tracking_gate=tracking_gate,
                                    pose_step_ratio=pose_step_ratio,
                                    scfg=config["scorenet"],
                                )
                                if hard_fallback:
                                    actural_keyframe_ids = keyframe_selection_distance(
                                        time_idx,
                                        curr_position,
                                        keyframe_list,
                                        config.get("distance_current_frame_prob", 0.1),
                                        num_iters_mapping,
                                    )
                                    log_every = int(config["scorenet"].get("mapping_hard_gate_log_every", 25))
                                    if log_every <= 0 or time_idx % log_every == 0:
                                        print(
                                            f"\n[HardGate] fallback exact baseline at Frame {time_idx}: "
                                            f"{', '.join(hard_reasons)}"
                                        )
                                elif time_idx < warmup_frames:
                                    sampled = keyframe_selection_distance(
                                        time_idx,
                                        curr_position,
                                        cand_kf,
                                        config.get("distance_current_frame_prob", 0.1),
                                        num_iters_mapping,
                                    )
                                elif config["scorenet"]["mapping_use_mixed_sampling"]:
                                    dis_probs = keyframe_distance_probs(time_idx, curr_position, cand_kf)
                                    alpha_sched = _scheduled_mix_alpha(time_idx, config["scorenet"])
                                    # Gate score contribution online: bad tracking residual -> rely more on baseline distance prior.
                                    alpha_eff = float(np.clip(1.0 - (1.0 - alpha_sched) * tracking_gate, 0.0, 1.0))
                                    sampled = keyframe_sampling_mixed_with_current(
                                        ref_ids,
                                        scores,
                                        distance_probs=dis_probs,
                                        n_samples=num_iters_mapping,
                                        current_frame_prob=config.get("distance_current_frame_prob", 0.1),
                                        alpha=alpha_eff,
                                        temperature=config["scorenet"]["sampling_temperature"],
                                        deduplicate_keyframes=config["scorenet"]["mapping_dedup_keyframes"],
                                        score_keep_ratio=config["scorenet"]["mapping_score_keep_ratio"],
                                        temporal_min_gap=config["scorenet"]["mapping_temporal_min_gap"],
                                    )
                                else:
                                    sampled = keyframe_sampling_scorenet_with_current(
                                        ref_ids,
                                        scores,
                                        n_samples=num_iters_mapping,
                                        current_frame_prob=config.get("distance_current_frame_prob", 0.1),
                                        temperature=config["scorenet"]["sampling_temperature"],
                                        deduplicate_keyframes=config["scorenet"]["mapping_dedup_keyframes"],
                                        score_keep_ratio=config["scorenet"]["mapping_score_keep_ratio"],
                                        temporal_min_gap=config["scorenet"]["mapping_temporal_min_gap"],
                                    )
                                if not hard_fallback:
                                    # map sampled indices back to keyframe_list indices.
                                    # token len(ref_ids) means selecting current frame.
                                    current_token = len(ref_ids)
                                    actural_keyframe_ids = [
                                        len(keyframe_list) if i == current_token else cand_indices[i]
                                        for i in sampled
                                    ]
                            else:
                                curr_position = params['cam_trans'][..., time_idx].detach().cpu()
                                actural_keyframe_ids = keyframe_selection_distance(time_idx, curr_position, keyframe_list, config['distance_current_frame_prob'], num_iters_mapping)
                        else:
                            actural_keyframe_ids = [0] * num_iters_mapping
                        print(f"\nUsed Frames for mapping at Frame {time_idx}: {[keyframe_list[i]['id'] if i != len(keyframe_list) else 'curr' for i in actural_keyframe_ids]}")

                    selected_keyframe_ids = actural_keyframe_ids[iter]
                    selected_keyframe_slot = selected_keyframe_ids

                    if selected_keyframe_ids == len(keyframe_list):
                        # Use Current Frame Data
                        iter_time_idx = time_idx
                        iter_color = color
                        iter_depth = depth
                        iter_fold_seg = fold_seg
                        selected_frame_for_log = time_idx
                    else:
                        # Use Keyframe Data
                        iter_time_idx = keyframe_list[selected_keyframe_ids]['id']
                        iter_color = keyframe_list[selected_keyframe_ids]['color']
                        iter_depth = keyframe_list[selected_keyframe_ids]['depth']
                        iter_fold_seg = keyframe_list[selected_keyframe_ids].get('fold_seg', None)
                        selected_frame_for_log = iter_time_idx
                    
                    
                iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx+1]
                iter_data = {'cam': cam, 'im': iter_color, 'depth': iter_depth, 'id': iter_time_idx, 
                             'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}
                iter_data['fold_seg_mapping_only'] = bool(
                    config.get('fold_seg_loss', {}).get('mapping_only', False)
                )
                iter_data['deformation_fold_soft_alpha'] = float(
                    config.get('temporal_deformation', {}).get('fold_soft_alpha', 0.5)
                )
                if iter_fold_seg is not None:
                    iter_data['fold_seg'] = iter_fold_seg
                    iter_data['fold_seg_weight'] = float(config.get('fold_seg_loss', {}).get('weight', 1.0))
                    iter_data['fold_seg_loss_type'] = str(config.get('fold_seg_loss', {}).get('loss_type', 'l1'))
                # Loss for current frame
                loss, variables, losses = get_loss(params, iter_data, variables, iter_time_idx, config['mapping']['loss_weights'],
                                                config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                                                config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'], mapping=True)
                _log_iter_loss(
                    iter_loss_writer,
                    "mapping",
                    time_idx,
                    iter_time_idx,
                    iter,
                    num_iters_mapping,
                    losses,
                    params,
                    len(keyframe_list),
                    mapping_iter_scale=mapping_iter_scale,
                    selected_keyframe_slot=selected_keyframe_slot,
                    selected_frame=selected_frame_for_log,
                )
                # Backprop
                loss.backward()
                with torch.no_grad():
                    # Prune Gaussians
                    if config['mapping']['prune_gaussians']:
                        params, variables = prune_gaussians(params, variables, optimizer, iter, config['mapping']['pruning_dict'])
                    # Gaussian-Splatting's Gradient-based Densification
                    if config['mapping']['use_gaussian_splatting_densification']:
                        params, variables = densify(params, variables, optimizer, iter, config['mapping']['densify_dict'])
                    # Optimizer Update
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    # Report Progress
                    if config['report_iter_progress']:
                        report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                        mapping=True, online_time_idx=time_idx)
                    else:
                        progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                mapping_iter_time_sum += iter_end_time - iter_start_time
                mapping_iter_time_count += 1
            if num_iters_mapping > 0:
                progress_bar.close()
            # Update the runtime numbers
            mapping_end_time = time.time()
            mapping_frame_time_sum += mapping_end_time - mapping_start_time
            mapping_frame_time_count += 1

            if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
                try:
                    # Report Mapping Progress
                    progress_bar = tqdm(range(1), desc=f"Mapping Result Time Step: {time_idx}")
                    with torch.no_grad():
                        report_progress(params, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                        mapping=True, online_time_idx=time_idx)
                    progress_bar.close()
                except:
                    ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                    save_params_ckpt(params, ckpt_output_dir, time_idx)
                    print('Failed to evaluate trajectory.')
        
        # timer.lap('Mapping Done.', 4)
        
        # Add frame to keyframe list.
        valid_keyframe_pose = (not torch.isinf(curr_gt_w2c[-1]).any()) and (not torch.isnan(curr_gt_w2c[-1]).any())
        motion_kfs_debug = None
        if bool(config["scorenet"].get("keyframe_motion_proxy_enable", False)):
            add_keyframe, motion_kfs_debug = _motion_proxy_keyframe_decision(
                time_idx=time_idx,
                num_frames=num_frames,
                params=params,
                keyframe_list=keyframe_list,
                config=config,
            )
            if bool(config["scorenet"].get("keyframe_ref_pair_gate_enable", False)):
                add_keyframe, ref_kfs_debug = _reference_pair_keyframe_gate(
                    proposed_add=bool(add_keyframe),
                    motion_debug=motion_kfs_debug,
                    curr_color=color,
                    keyframe_list=keyframe_list,
                    config=config,
                )
                motion_kfs_debug.update(ref_kfs_debug)
            if bool(config["scorenet"].get("keyframe_feature_pair_gate_enable", False)):
                add_keyframe, feat_kfs_debug = _feature_pair_keyframe_gate(
                    proposed_add=bool(add_keyframe),
                    motion_debug=motion_kfs_debug,
                    curr_color=color,
                    keyframe_list=keyframe_list,
                    config=config,
                )
                motion_kfs_debug.update(feat_kfs_debug)
            if bool(config["scorenet"].get("keyframe_energy_gate_enable", False)):
                add_keyframe, energy_kfs_debug = _energy_keyframe_gate(
                    proposed_add=bool(add_keyframe),
                    motion_debug=motion_kfs_debug,
                    curr_color=color,
                    curr_data=curr_data,
                    config=config,
                )
                motion_kfs_debug.update(energy_kfs_debug)
            add_keyframe = bool(add_keyframe) and bool(valid_keyframe_pose)
            log_every = int(config["scorenet"].get("keyframe_motion_proxy_log_every", 25))
            if add_keyframe or log_every <= 0 or time_idx % log_every == 0:
                decision = "add" if add_keyframe else "hold"
                print(f"\n[MotionKFS] {decision} at Frame {time_idx}: {_format_motion_kfs_debug(motion_kfs_debug)}")
        else:
            add_keyframe = ((time_idx == 0) or ((time_idx+1) % config['keyframe_every'] == 0) or \
                        (time_idx == num_frames-2)) and valid_keyframe_pose
        if (
            (not bool(config["scorenet"].get("keyframe_motion_proxy_enable", False)))
            and scorenet_model is not None
            and config["scorenet"]["use_for_keyframe_select"]
            and time_idx > 0
        ):
            cand_kf = keyframe_list[-config["scorenet"]["candidate_window"]:]
            if len(cand_kf) > 0:
                ref_ids = [kf["id"] for kf in cand_kf]
                ref_imgs = [kf["color"].unsqueeze(0).to(device) for kf in cand_kf]
                target = color.unsqueeze(0).to(device)
                outputs = score_candidates(scorenet_model, target, ref_imgs, ref_ids)
                scores = outputs["scores"][0]
                topk = keyframe_selection_scorenet(time_idx, ref_ids, scores, k=1)
                # Simple rule: add as keyframe if best score above median of candidates
                if len(topk) > 0:
                    score_best = scores[topk[0]].item()
                    score_med = scores.median().item()
                    if score_best > score_med:
                        add_keyframe = True
        if failure_debug_enable:
            try:
                debug_row, debug_images = _failure_collect_frame_debug(
                    params=params,
                    curr_data=curr_data,
                    variables=variables,
                    time_idx=time_idx,
                    keyframe_list=keyframe_list,
                    add_keyframe=bool(add_keyframe),
                    motion_debug=motion_kfs_debug,
                    tracking_residual_curr=tracking_residual_curr,
                    tracking_residual_ema=tracking_residual_ema,
                    config=config,
                )
                failure_debug_recent_rows.append(debug_row)
                if len(failure_debug_recent_rows) > 200:
                    failure_debug_recent_rows = failure_debug_recent_rows[-200:]
                _failure_append_csv(failure_debug_csv, debug_row)
                if (
                    bool(failure_debug_cfg.get("save_panels", True))
                    and _failure_should_save_panel(debug_row, failure_debug_cfg, failure_debug_panel_count)
                ):
                    panel_path = os.path.join(failure_debug_dir, "panels", f"frame_{time_idx:05d}.png")
                    _failure_save_panel(panel_path, debug_images, debug_row, failure_debug_recent_rows)
                    failure_debug_panel_count += 1
            except Exception as exc:
                print(f"[FailureDebug] frame {time_idx} failed: {exc}")
        if add_keyframe:
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                if fold_seg is not None:
                    curr_keyframe['fold_seg'] = fold_seg
                if (
                    scorenet_model is not None
                    and bool(config["scorenet"].get("pose_graph_store_keyframe_edges", False))
                    and len(keyframe_list) > 0
                ):
                    pg_ref_pool = keyframe_list[-int(config["scorenet"].get("candidate_window", 20)):]
                    pg_edge_stats = build_weighted_keyframe_edges(
                        curr_keyframe=curr_keyframe,
                        ref_pool=pg_ref_pool,
                        scorenet_model=scorenet_model,
                        device=device,
                        scorenet_cfg=config["scorenet"],
                    )
                    curr_keyframe["kfg_pose_edges"] = pg_edge_stats.get("edges", [])
                    if bool(config["scorenet"].get("pose_graph_log_edges", False)):
                        selected = pg_edge_stats.get("selected", [])
                        print(
                            f"[pose_graph] stored_edges frame={time_idx} "
                            f"edges={len(curr_keyframe['kfg_pose_edges'])} selected={selected[:8]}"
                        )
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
                keyframe_time_indices.append(time_idx)
            if scorenet_model is not None and config["scorenet"]["use_pose_graph"]:
                try:
                    every_n = max(int(config["scorenet"].get("pose_graph_every_n_kf", 1)), 1)
                    if len(keyframe_list) % every_n == 0:
                        pg_stats = optimize_pose_graph_local(
                            params=params,
                            keyframe_list=keyframe_list,
                            scorenet_model=scorenet_model,
                            device=device,
                            scorenet_cfg=config["scorenet"],
                        )
                        if pg_stats.get("optimized", False):
                            print(
                                f"[pose_graph] optimized nodes={pg_stats.get('nodes', 0)} "
                                f"edges={pg_stats.get('edges', 0)} "
                                f"stored={pg_stats.get('stored_long_edges', 0)} "
                                f"loss={pg_stats.get('loss', -1.0):.6f}"
                            )
                except Exception as e:
                    print(f"[pose_graph] skipped at frame {time_idx}: {e}")
        pose_demand_prev_color = color.detach()
        
        # Checkpoint every iteration
        if time_idx % config["checkpoint_interval"] == 0 and config['save_checkpoints']:
            ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
            save_params_ckpt(params, ckpt_output_dir, time_idx)
            np.save(os.path.join(ckpt_output_dir, f"keyframe_time_indices{time_idx}.npy"), np.array(keyframe_time_indices))
        

        torch.cuda.empty_cache()

    # timer.end()

    # Compute Average Runtimes
    if tracking_iter_time_count == 0:
        tracking_iter_time_count = 1
        tracking_frame_time_count = 1
    if mapping_iter_time_count == 0:
        mapping_iter_time_count = 1
        mapping_frame_time_count = 1
    tracking_iter_time_avg = tracking_iter_time_sum / tracking_iter_time_count
    tracking_frame_time_avg = tracking_frame_time_sum / tracking_frame_time_count
    mapping_iter_time_avg = mapping_iter_time_sum / mapping_iter_time_count
    mapping_frame_time_avg = mapping_frame_time_sum / mapping_frame_time_count
    print(f"\nAverage Tracking/Iteration Time: {tracking_iter_time_avg*1000} ms")
    print(f"Average Tracking/Frame Time: {tracking_frame_time_avg} s")
    print(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000} ms")
    print(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s")
    with open(os.path.join(output_dir, "runtimes.txt"), "w") as f:
        f.write(f"Average Tracking/Iteration Time: {tracking_iter_time_avg*1000} ms\n")
        f.write(f"Average Tracking/Frame Time: {tracking_frame_time_avg} s\n")
        f.write(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000} ms\n")
        f.write(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s\n")
        f.write(f"Frame Time: {tracking_frame_time_avg + mapping_frame_time_avg} s\n")
    
    # Evaluate Final Parameters
    dataset = [dataset, eval_dataset, 'C3VD'] if dataset_config["train_or_test"] == 'train' else dataset
    with torch.no_grad():
        eval_save(dataset, params, eval_dir, sil_thres=config['mapping']['sil_thres'],
                mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'])

    # Add Camera Parameters to Save them
    params['timestep'] = variables['timestep']
    params['intrinsics'] = intrinsics.detach().cpu().numpy()
    params['w2c'] = first_frame_w2c.detach().cpu().numpy()
    params['org_width'] = dataset_config["desired_image_width"]
    params['org_height'] = dataset_config["desired_image_height"]
    params['gt_w2c_all_frames'] = []
    for gt_w2c_tensor in gt_w2c_all_frames:
        params['gt_w2c_all_frames'].append(gt_w2c_tensor.detach().cpu().numpy())
    params['gt_w2c_all_frames'] = np.stack(params['gt_w2c_all_frames'], axis=0)
    params['keyframe_time_indices'] = np.array(keyframe_time_indices)
    
    # Save Parameters
    save_params(params, output_dir)
    save_means3D(params['means3D'], output_dir)
    if iter_loss_handle is not None:
        iter_loss_handle.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")
    # parser.add_argument("--online_vis", action="store_true", help="Visualize mapping renderings while running")

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    # Prepare dir for visualization
    # if args.online_vis:
    #     vis_dir = './online_vis'
    #     os.makedirs(vis_dir, exist_ok=True)
    #     for filename in os.listdir(vis_dir):
    #         os.unlink(os.path.join(vis_dir, filename))

    # Set Experiment Seed
    seed_everything(seed=experiment.config['seed'])
    
    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    if not experiment.config['load_checkpoint']:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))

    rgbd_slam(experiment.config)
    
    # Output identity is defined by config, including frozen queue configs that
    # do not expose the legacy module-level group_name/scene_name variables.
    plots_dir = os.path.join(results_dir, 'eval', 'plots')
    if os.path.isdir(plots_dir) and any(name.endswith('.png') for name in os.listdir(plots_dir)):
        plot_video(plots_dir, os.path.join(results_dir, 'keyframes'))
    else:
        print('[video] No saved plot frames; skipping optional video export.')
