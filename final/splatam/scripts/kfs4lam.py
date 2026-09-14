import argparse
import os
import shutil
import sys
import time
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)


import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from datasets.gradslam_datasets import (load_dataset_config, ICLDataset, ReplicaDataset, ReplicaV2Dataset, AzureKinectDataset,
                                        ScannetDataset, Ai2thorDataset, Record3DDataset, RealsenseDataset, TUMDataset,
                                        ScannetPPDataset, NeRFCaptureDataset, C3VDDataset)
from utils.common_utils import seed_everything, save_params_ckpt, save_params
from utils.eval_helpers import report_loss, report_progress, eval
from utils.keyframe_selection import keyframe_selection_overlap
from utils.recon_helpers import setup_camera
from utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
    transform_to_frame, temporal_deformation_delta, l1_loss_v1, matrix_to_quaternion
)
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians, densify

from diff_gaussian_rasterization import GaussianRasterizer as Renderer


_FOLD_SEG_MISSING_WARNED = set()


def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["icl"]:
        return ICLDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replicav2"]:
        return ReplicaV2Dataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["azure", "azurekinect"]:
        return AzureKinectDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet"]:
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["ai2thor"]:
        return Ai2thorDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["record3d"]:
        return Record3DDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["realsense"]:
        return RealsenseDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["tum"]:
        return TUMDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannetpp"]:
        return ScannetPPDataset(basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["nerfcapture"]:
        return NeRFCaptureDataset(basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["c3vd"]:
        return C3VDDataset(config_dict, basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


def dynamic_mapping_num_iters(config, time_idx, num_keyframes):
    """Mirror EndoGSLAM's adaptive-keyframe-density mapping budget."""
    dyn = config.get("dynamic_mapping_iters", {})
    base_iters = int(config["mapping"]["num_iters"])
    if not bool(dyn.get("enable", False)):
        return base_iters, 1.0, base_iters

    base_gap = max(int(dyn.get("base_gap", config.get("keyframe_every", 8))), 1)
    expected_history_kf = max(1.0, float(np.ceil((int(time_idx) + 1) / float(base_gap))))
    expected_units = expected_history_kf + 1.0
    active_units = max(1.0, float(int(num_keyframes) + 1))
    ratio = max(1.0, active_units / expected_units)
    scale = min(float(dyn.get("max_scale", 3.0)), ratio ** float(dyn.get("power", 1.0)))
    scaled_iters = int(np.ceil(base_iters * scale))
    min_iters = int(dyn.get("min_iters", base_iters))
    max_iters = int(dyn.get("max_iters", base_iters * 3))
    return int(np.clip(scaled_iters, min_iters, max_iters)), float(scale), base_iters


def _kfs_pose(params, time_idx):
    t = params['cam_trans'][..., time_idx].detach().cpu().numpy().reshape(-1)[:3].astype(np.float64)
    q = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
    r = build_rotation(q).detach().cpu().numpy()
    return t, (r[0] if r.ndim == 3 else r).astype(np.float64)


def _kfs_rotation_deg(r0, r1):
    rel = r1 @ r0.T
    return float(np.degrees(np.arccos(np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0))))


def _kfs_vector_angle_deg(v0, v1):
    n0, n1 = float(np.linalg.norm(v0)), float(np.linalg.norm(v1))
    if n0 <= 1e-12 or n1 <= 1e-12:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v0, v1) / (n0 * n1), -1.0, 1.0))))


def _endogslam_v4_motion_decision(time_idx, num_frames, params, keyframe_list, config):
    """Original EndoGSLAM v4 instant-motion proposal on the SplaTAM poses."""
    kcfg = config['accum_motion_redundancy']
    base_gap = max(int(kcfg.get('base_gap', config.get('keyframe_every', 8))), 1)
    min_gap = max(int(kcfg.get('min_gap', 4)), 1)
    max_gap = max(int(kcfg.get('max_gap', 16)), min_gap)
    if time_idx == 0 or not keyframe_list:
        return True, {'reason': 'init', 'gap': 0}
    last_kf = int(keyframe_list[-1]['id'])
    gap = int(time_idx - last_kf)
    curr_t, curr_r = _kfs_pose(params, time_idx)
    last_t, last_r = _kfs_pose(params, last_kf)
    net_dist = float(np.linalg.norm(curr_t - last_t))

    hist_start = max(0, time_idx - max(max_gap, 2 * base_gap))
    hist_positions = [_kfs_pose(params, i)[0] for i in range(hist_start, time_idx + 1)]
    steps = np.asarray([np.linalg.norm(hist_positions[i] - hist_positions[i - 1])
                        for i in range(1, len(hist_positions))], dtype=np.float64)
    positive = steps[steps > 1e-8]
    ref_step = float(np.median(positive)) if positive.size else 0.0
    target = max(float(kcfg.get('min_target_dist', 0.03)), ref_step * base_gap *
                 float(kcfg.get('target_dist_scale', 1.0)))

    segment = [_kfs_pose(params, i)[0] for i in range(last_kf, time_idx + 1)]
    path_len = float(sum(np.linalg.norm(segment[i] - segment[i - 1])
                         for i in range(1, len(segment))))
    linearity = float(net_dist / (path_len + 1e-12)) if path_len > 0 else 1.0
    mid = min(last_kf + max(gap // 2, 1), time_idx)
    mid_t = _kfs_pose(params, mid)[0]
    segment_turn = _kfs_vector_angle_deg(mid_t - last_t, curr_t - mid_t)
    recent_turn = 0.0
    if time_idx >= 2:
        t0 = _kfs_pose(params, time_idx - 2)[0]
        t1 = _kfs_pose(params, time_idx - 1)[0]
        recent_turn = _kfs_vector_angle_deg(t1 - t0, curr_t - t1)
    turn_deg = max(segment_turn, recent_turn)
    rot_deg = _kfs_rotation_deg(last_r, curr_r)

    high_turn = (turn_deg >= float(kcfg.get('turn_angle_deg', 20.0)) and
                 net_dist >= target * float(kcfg.get('turn_min_dist_ratio', 0.25)))
    high_rot = (rot_deg >= float(kcfg.get('rot_angle_deg', 18.0)) and
                net_dist >= target * float(kcfg.get('turn_min_dist_ratio', 0.25)))
    straight = (turn_deg <= float(kcfg.get('straight_angle_deg', 6.0)) and
                rot_deg <= float(kcfg.get('straight_rot_deg', 6.0)) and
                linearity >= float(kcfg.get('straight_linearity', 0.92)))
    jitter = (net_dist <= target * float(kcfg.get('jitter_dist_ratio', 0.35)) and
              rot_deg <= float(kcfg.get('jitter_rot_deg', 5.0)) and gap < max_gap)
    if time_idx == num_frames - 2 and gap >= min_gap:
        add, reason = True, 'terminal'
    elif gap < min_gap:
        add, reason = False, 'min_gap'
    elif gap >= max_gap:
        add, reason = True, 'max_gap'
    elif jitter:
        add, reason = False, 'jitter_hold'
    elif high_turn:
        add, reason = True, 'turn'
    elif high_rot:
        add, reason = True, 'rotation'
    else:
        dist_threshold = target * (float(kcfg.get('straight_dist_scale', 1.5)) if straight else 1.0)
        add = net_dist >= dist_threshold
        reason = 'straight_distance' if straight and add else ('distance' if add else 'wait')
    return bool(add), {'reason': reason, 'gap': gap, 'net_dist': net_dist,
                       'target_dist': target, 'turn_deg': turn_deg, 'rot_deg': rot_deg,
                       'linearity': linearity, 'straight': straight}


def _endogslam_v4_ref_feature_gate(proposed, debug, color, keyframe_list, config):
    """Original v4 ref-novel insertion followed by feature-based suppression."""
    if not keyframe_list:
        return proposed, 'no_ref'
    reason = debug.get('reason', 'unknown')
    if reason in ('init', 'terminal', 'max_gap'):
        return proposed, 'protected'
    if reason == 'min_gap':
        return proposed, 'min_gap'
    kcfg = config['accum_motion_redundancy']
    refs = keyframe_list[-max(int(kcfg.get('recent_refs', 3)), 1):]
    cur_g, cur_d, cur_edge, cur_contrast = _gray_grad_proxy(color, int(kcfg.get('downsample', 96)))
    stats = []
    for ref in refs:
        ref_g, ref_d, ref_edge, ref_contrast = _gray_grad_proxy(ref['color'], int(kcfg.get('downsample', 96)))
        stats.append((float(torch.mean(torch.abs(cur_g-ref_g))),
                      float(torch.mean(torch.abs(cur_d-ref_d))), ref_edge, ref_contrast))
    edge_ratio = cur_edge / (float(np.median([x[2] for x in stats])) + 1e-6)
    contrast_ratio = cur_contrast / (float(np.median([x[3] for x in stats])) + 1e-6)
    low_quality = (edge_ratio < float(kcfg.get('min_edge_ratio', 0.65)) or
                   contrast_ratio < float(kcfg.get('min_contrast_ratio', 0.60)))
    duplicate = (min(x[0] for x in stats) <= float(kcfg.get('duplicate_l1', 0.025)) and
                 min(x[1] for x in stats) <= float(kcfg.get('duplicate_grad_l1', 0.020)))
    ref_novel = (stats[-1][0] >= float(kcfg.get('ref_add_l1', 0.060)) or
                 stats[-1][1] >= float(kcfg.get('ref_add_grad_l1', 0.035)))
    if proposed and low_quality:
        proposed, gate_reason = False, 'block_low_quality'
    elif proposed and duplicate:
        proposed, gate_reason = False, 'block_ref_duplicate'
    elif (not proposed) and ref_novel and not low_quality:
        proposed, gate_reason = True, 'add_ref_novel'
    else:
        gate_reason = 'ref_keep'

    # v4 applies the 8-D KeyFrameGraph pair proxy after the reference gate.
    try:
        root = kcfg.get('keyframe_graph_root', os.path.abspath(os.path.join(_BASE_DIR, '..', '..', 'modules')))
        if root not in sys.path:
            sys.path.insert(0, root)
        from networks.multiwrapper import compute_pair_proxies_pose_free
        size = int(kcfg.get('feature_downsample', 128))
        target = F.interpolate(color.unsqueeze(0), size=(size, size), mode='bilinear', align_corners=False)
        ref_images = [F.interpolate(ref['color'].unsqueeze(0), size=(size, size),
                                    mode='bilinear', align_corners=False) for ref in refs]
        pair = compute_pair_proxies_pose_free(
            target, torch.stack(ref_images, dim=1),
            max_disp=int(kcfg.get('feature_max_disp', 4)),
            temp=float(kcfg.get('feature_corr_temp', 0.5)))
        if pair.ndim == 3:
            pair = pair[0]
        pair = pair.detach().cpu().float().numpy()
        best = int(np.argmin(pair[:, 0]))
        near_duplicate = (float(pair[best, 0]) <= float(kcfg.get('feature_duplicate_mae', 0.025)) and
                          float(pair[best, 5]) <= float(kcfg.get('feature_duplicate_flow', 0.10)) and
                          float(np.max(pair[:, 4])) >= float(kcfg.get('feature_duplicate_grad', 0.92)))
        feature_edge_ratio = float(np.median(pair[:, 2]) / (np.median(pair[:, 3]) + 1e-6))
        if proposed and feature_edge_ratio < float(kcfg.get('min_edge_ratio', 0.65)):
            return False, 'block_feature_low_texture'
        if proposed and near_duplicate:
            return False, 'block_feature_duplicate'
    except Exception as exc:
        print(f"[EndoGSLAMv4KFS] feature gate fallback: {exc}")
    return bool(proposed), gate_reason


def _accum_motion_keyframe_decision(time_idx, num_frames, params, keyframe_list, config, state):
    """Accumulated-motion proposal used by the current KFS redundancy variant."""
    kcfg = config['accum_motion_redundancy']
    base_gap = max(int(kcfg.get('base_gap', config.get('keyframe_every', 8))), 1)
    min_gap = max(int(kcfg.get('min_gap', max(2, base_gap // 2))), 1)
    max_gap = max(int(kcfg.get('max_gap', max(base_gap + 1, 2 * base_gap))), min_gap)
    if time_idx == 0 or not keyframe_list:
        state.update(last_time_idx=time_idx, motion=0.0)
        return True, {'reason': 'init', 'gap': 0}
    last_kf = int(keyframe_list[-1]['id'])
    gap = time_idx - last_kf
    curr_t, curr_r = _kfs_pose(params, time_idx)
    prev_t, _ = _kfs_pose(params, max(0, time_idx - 1))
    if state.get('last_time_idx') != time_idx:
        state['motion'] = float(state.get('motion', 0.0)) + float(np.linalg.norm(curr_t - prev_t))
        state['last_time_idx'] = time_idx
    accum_motion = float(state.get('motion', 0.0))
    last_t, last_r = _kfs_pose(params, last_kf)
    net_dist = float(np.linalg.norm(curr_t - last_t))
    hist_start = max(0, time_idx - max(max_gap, 2 * base_gap))
    positions = [_kfs_pose(params, i)[0] for i in range(hist_start, time_idx + 1)]
    steps = np.asarray([np.linalg.norm(positions[i] - positions[i - 1]) for i in range(1, len(positions))])
    positive = steps[steps > 1e-8]
    ref_step = float(np.median(positive)) if positive.size else 0.0
    target = max(float(kcfg.get('min_target_dist', 0.03)), ref_step * base_gap * float(kcfg.get('target_dist_scale', 1.0)))
    threshold = target * float(kcfg.get('threshold_scale', 1.0))
    rot_deg = _kfs_rotation_deg(last_r, curr_r)
    jitter = (net_dist <= target * float(kcfg.get('jitter_dist_ratio', 0.35)) and
              rot_deg <= float(kcfg.get('jitter_rot_deg', 5.0)) and gap < max_gap and
              accum_motion < threshold * float(kcfg.get('jitter_hold_ratio', 0.5)))
    if time_idx == num_frames - 2 and gap >= min_gap:
        add, reason = True, 'terminal'
    elif gap < min_gap:
        add, reason = False, 'min_gap'
    elif gap >= max_gap:
        add, reason = True, 'max_gap'
    elif jitter:
        add, reason = False, 'jitter_hold'
    else:
        add = accum_motion >= threshold
        reason = 'accum_motion' if add else 'accum_wait'
    if add:
        state['motion'] = 0.0
    return bool(add), {'reason': reason, 'gap': gap, 'accum_motion': accum_motion,
                       'threshold': threshold, 'net_dist': net_dist, 'target_dist': target}


def _gray_grad_proxy(color, downsample):
    x = color.detach().float().clamp(0.0, 1.0)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    h, w = x.shape[-2:]
    if max(h, w) > downsample:
        size = ((downsample, max(8, round(w * downsample / h))) if h >= w
                else (max(8, round(h * downsample / w)), downsample))
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
    gray = 0.2989*x[:, 0:1] + 0.5870*x[:, 1:2] + 0.1140*x[:, 2:3]
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=x.device).view(1, 1, 3, 3)
    grad = torch.sqrt(F.conv2d(gray, kx, padding=1)**2 + F.conv2d(gray, kx.transpose(2, 3), padding=1)**2 + 1e-6)
    return gray, grad, float(grad.mean()), float(gray.std())


def _redundancy_gate(proposed, debug, color, keyframe_list, config):
    """Apply the reference duplicate/quality suppression from the latest policy."""
    if not proposed or debug['reason'] in ('init', 'terminal', 'max_gap') or not keyframe_list:
        return proposed, 'protected' if proposed else 'not_proposed'
    kcfg = config['accum_motion_redundancy']
    refs = keyframe_list[-max(int(kcfg.get('recent_refs', 3)), 1):]
    downsample = int(kcfg.get('downsample', 96))
    cur_g, cur_d, cur_edge, cur_contrast = _gray_grad_proxy(color, downsample)
    stats = []
    for ref in refs:
        ref_g, ref_d, ref_edge, ref_contrast = _gray_grad_proxy(ref['color'], downsample)
        stats.append((float(torch.mean(torch.abs(cur_g-ref_g))), float(torch.mean(torch.abs(cur_d-ref_d))), ref_edge, ref_contrast))
    edge_ratio = cur_edge / (float(np.median([x[2] for x in stats])) + 1e-6)
    contrast_ratio = cur_contrast / (float(np.median([x[3] for x in stats])) + 1e-6)
    if edge_ratio < float(kcfg.get('min_edge_ratio', 0.65)) or contrast_ratio < float(kcfg.get('min_contrast_ratio', 0.60)):
        return False, 'block_low_quality'
    if (min(x[0] for x in stats) <= float(kcfg.get('duplicate_l1', 0.025)) and
            min(x[1] for x in stats) <= float(kcfg.get('duplicate_grad_l1', 0.020))):
        return False, 'block_duplicate'
    # The latest redundancy variant also applies the KeyFrameGraph 8-D pair
    # proxy, but only blocks duplicates (not low texture or bad correlation).
    root = kcfg.get('keyframe_graph_root', os.path.abspath(os.path.join(_BASE_DIR, '..', '..', 'modules')))
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from networks.multiwrapper import compute_pair_proxies_pose_free
        feature_size = int(kcfg.get('feature_downsample', 128))
        target = F.interpolate(color.unsqueeze(0), size=(feature_size, feature_size), mode='bilinear', align_corners=False)
        ref_images = [F.interpolate(ref['color'].unsqueeze(0), size=(feature_size, feature_size),
                                    mode='bilinear', align_corners=False) for ref in refs]
        pair = compute_pair_proxies_pose_free(
            target, torch.stack(ref_images, dim=1),
            max_disp=int(kcfg.get('feature_max_disp', 4)),
            temp=float(kcfg.get('feature_corr_temp', 0.5)))
        if pair.ndim == 3:
            pair = pair[0]
        pair = pair.detach().cpu().float().numpy()
        duplicate = (float(np.min(pair[:, 0])) <= float(kcfg.get('feature_duplicate_mae', 0.025)) and
                     float(pair[int(np.argmin(pair[:, 0])), 5]) <= float(kcfg.get('feature_duplicate_flow', 0.10)) and
                     float(np.max(pair[:, 4])) >= float(kcfg.get('feature_duplicate_grad', 0.92)))
        if duplicate:
            return False, 'block_feature_duplicate'
    except Exception as exc:
        print(f"[AccumMotionRedundancyKFS] feature gate fallback: {exc}")
    return True, 'keep'


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
        point_cols.append(torch.permute(fold_seg, (1, 2, 0)).reshape(-1, 1))
    point_cld = torch.cat(point_cols, -1)

    # Select points based on mask
    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    else:
        return point_cld


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
    """Load one fold probability map without exposing it to tracking."""
    if not bool(cfg.get("enable", False)):
        return None
    root = str(cfg.get("root", "")).strip()
    sequence = os.path.basename(getattr(dataset, "input_folder", "")).strip()
    stem = _dataset_frame_stem(dataset, index)
    source_index = _dataset_source_index(dataset, index)
    candidates = [
        os.path.join(root, sequence, f"{stem}.npy"),
        os.path.join(root, sequence, f"{source_index:04d}_color.npy"),
        os.path.join(root, sequence, f"{source_index:05d}_color.npy"),
        os.path.join(root, sequence, f"{source_index:06d}_color.npy"),
    ]
    path = next((candidate for candidate in candidates if os.path.isfile(candidate)), None)
    if path is None:
        key = (sequence, stem)
        if key not in _FOLD_SEG_MISSING_WARNED:
            print(f"[FoldL1] missing {sequence}/{stem}; using background")
            _FOLD_SEG_MISSING_WARNED.add(key)
        return torch.zeros((1, int(target_hw[0]), int(target_hw[1])), device=device)
    array = np.load(path)
    channel = int(cfg.get("channel", 0))
    if array.ndim == 3:
        array = array[..., channel] if array.shape[-1] <= 16 else array[channel]
    if array.ndim != 2:
        raise ValueError(f"Fold prediction must be 2-D/3-D, got {array.shape}: {path}")
    fold = torch.from_numpy(np.asarray(array, dtype=np.float32))[None, None].to(device)
    if bool(cfg.get("probability_clip", True)):
        fold = fold.clamp(0.0, 1.0)
    fold = float(cfg.get("low", 0.0)) + fold * (float(cfg.get("high", 1.0)) - float(cfg.get("low", 0.0)))
    if tuple(fold.shape[-2:]) != tuple(int(v) for v in target_hw):
        fold = F.interpolate(fold, size=tuple(int(v) for v in target_hw), mode="nearest")
    return fold[0].contiguous()


def _fold_rendervar(params, transformed_gaussians):
    # The attribute is an observed anatomical label. Detaching prevents the
    # optimizer from satisfying Fold-L1 merely by recolouring Gaussians; the
    # gradient acts on Gaussian geometry instead.
    colors = params["fold_colors"].detach().clamp(0.0, 1.0).repeat(1, 3)
    out = {
        "means3D": transformed_gaussians["means3D"],
        "colors_precomp": colors,
        "rotations": F.normalize(transformed_gaussians["unnorm_rotations"]),
        "opacities": torch.sigmoid(params["logit_opacities"]),
        "means2D": torch.zeros_like(params["means3D"], requires_grad=True, device="cuda"),
    }
    out["scales"] = torch.exp(torch.tile(params["log_scales"], (1, 3))) if params["log_scales"].shape[1] == 1 else torch.exp(params["log_scales"])
    return out


def _fold_mapping_loss(params, curr_data, iter_time_idx):
    """Render Fold-L1 alone so its rasterizer graph need not overlap RGB-D."""
    transformed_gaussians = transform_to_frame(
        params, iter_time_idx, gaussians_grad=True, camera_grad=False
    )
    fold_rgb, _, _ = Renderer(raster_settings=curr_data['cam'])(
        **_fold_rendervar(params, transformed_gaussians)
    )
    fold_render = fold_rgb[:1]
    fold_target = curr_data['fold_seg'].to(fold_render).clamp(0.0, 1.0)
    if tuple(fold_target.shape[-2:]) != tuple(fold_render.shape[-2:]):
        fold_target = F.interpolate(
            fold_target[None], size=fold_render.shape[-2:], mode='nearest'
        )[0]
    return float(curr_data.get('fold_seg_weight', 1.0)) * torch.abs(
        fold_render - fold_target
    ).mean()


def initialize_params(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution,
                      temporal_deformation_cfg=None):
    num_pts = init_pt_cld.shape[0]
    means3D = init_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
    params = {
        'means3D': means3D,
        'rgb_colors': init_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }
    if init_pt_cld.shape[1] > 6:
        params['fold_colors'] = init_pt_cld[:, 6:7]
    deformation_cfg = temporal_deformation_cfg or {}
    if bool(deformation_cfg.get('enable', False)):
        num_basis = max(int(deformation_cfg.get('num_basis', 2)), 1)
        params['deform_coeffs'] = torch.zeros(
            (num_pts, 3 * num_basis), dtype=torch.float, device="cuda"
        )

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


def initialize_optimizer(params, lrs_dict, tracking):
    lrs = lrs_dict
    # Extra observed Gaussian attributes (currently fold_colors) are frozen by
    # default; their rendered loss still differentiates Gaussian geometry.
    param_groups = [{'params': [v], 'name': k, 'lr': lrs.get(k, 0.0)} for k, v in params.items()]
    if tracking:
        return torch.optim.Adam(param_groups)
    else:
        return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def initialize_first_timestep(dataset, num_frames, scene_radius_depth_ratio, 
                              mean_sq_dist_method, densify_dataset=None, gaussian_distribution=None,
                              fold_seg_cfg=None, temporal_deformation_cfg=None):
    # Get RGB-D Data & Camera Parameters
    color, depth, intrinsics, pose = dataset[0]

    # Process RGB-D Data
    color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
    
    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3]
    w2c = torch.linalg.inv(pose)

    # Setup Camera
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())

    if densify_dataset is not None:
        # Get Densification RGB-D Data & Camera Parameters
        color, depth, densify_intrinsics, _ = densify_dataset[0]
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
        densify_intrinsics = densify_intrinsics[:3, :3]
        densify_cam = setup_camera(color.shape[2], color.shape[1], densify_intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())
    else:
        densify_intrinsics = intrinsics

    fold_dataset = densify_dataset if densify_dataset is not None else dataset
    fold_seg = _load_fold_seg(fold_dataset, 0, color.shape[1:], fold_seg_cfg or {}, color.device)

    # Get Initial Point Cloud (PyTorch CUDA Tensor)
    mask = (depth > 0) # Mask out invalid depth values
    mask = mask.reshape(-1)
    init_pt_cld, mean3_sq_dist = get_pointcloud(color, depth, densify_intrinsics, w2c, 
                                                mask=mask, compute_mean_sq_dist=True, 
                                                mean_sq_dist_method=mean_sq_dist_method,
                                                fold_seg=fold_seg)

    # Initialize Parameters
    params, variables = initialize_params(
        init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution,
        temporal_deformation_cfg=temporal_deformation_cfg,
    )

    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    variables['scene_radius'] = torch.max(depth)/scene_radius_depth_ratio

    if densify_dataset is not None:
        return params, variables, intrinsics, w2c, cam, densify_intrinsics, densify_cam
    else:
        return params, variables, intrinsics, w2c, cam


def get_loss(params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss,
             sil_thres, use_l1, ignore_outlier_depth_loss, tracking=False, 
             mapping=False, do_ba=False, plot_dir=None, visualize_tracking_loss=False, tracking_iteration=None):
    # Initialize Loss Dictionary
    losses = {}

    if tracking:
        # Get current frame Gaussians, where only the camera pose gets gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx, 
                                             gaussians_grad=False,
                                             camera_grad=True)
    elif mapping:
        if do_ba:
            # Get current frame Gaussians, where both camera pose and Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=True)
        else:
            # Get current frame Gaussians, where only the Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=False)
    else:
        # Get current frame Gaussians, where only the Gaussians get gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                             gaussians_grad=True,
                                             camera_grad=False)

    # Initialize Render Variables
    rendervar = transformed_params2rendervar(params, transformed_gaussians)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_gaussians)

    # RGB Rendering
    rendervar['means2D'].retain_grad()
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    # Depth & Silhouette Rendering
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    depth = depth_sil[0, :, :].unsqueeze(0)
    silhouette = depth_sil[1, :, :]
    presence_sil_mask = (silhouette > sil_thres)
    depth_sq = depth_sil[2, :, :].unsqueeze(0)
    uncertainty = depth_sq - depth**2
    uncertainty = uncertainty.detach()

    # Mask with valid depth values (accounts for outlier depth values)
    nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
    if ignore_outlier_depth_loss:
        depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
        mask = (depth_error < 10*depth_error.median())
        mask = mask & (curr_data['depth'] > 0)
    else:
        mask = (curr_data['depth'] > 0)
    mask = mask & nan_mask
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

    # Fold supervision is mapping-only: it never contributes a camera-pose
    # gradient. The observed per-Gaussian fold attribute is fixed, so this
    # term supervises anatomical map geometry rather than label recolouring.
    if (mapping and curr_data.get('fold_seg') is not None and 'fold_colors' in params
            and not curr_data.get('defer_fold_backward', False)):
        losses['fold'] = _fold_mapping_loss(params, curr_data, iter_time_idx)

    # Restricted EndoGS-style temporal geometry: position offsets only.
    # Fold probability is a soft prior that relaxes the magnitude penalty in
    # fold regions; it never hard-disables deformation elsewhere.
    if mapping and 'deform_coeffs' in params:
        delta = temporal_deformation_delta(params, iter_time_idx, gaussians_grad=True)
        prev_delta = temporal_deformation_delta(
            params, max(int(iter_time_idx) - 1, 0), gaussians_grad=True
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

    # Visualize the Diff Images
    if tracking and visualize_tracking_loss:
        fig, ax = plt.subplots(2, 4, figsize=(12, 6))
        weighted_render_im = im * color_mask
        weighted_im = curr_data['im'] * color_mask
        weighted_render_depth = depth * mask
        weighted_depth = curr_data['depth'] * mask
        diff_rgb = torch.abs(weighted_render_im - weighted_im).mean(dim=0).detach().cpu()
        diff_depth = torch.abs(weighted_render_depth - weighted_depth).mean(dim=0).detach().cpu()
        viz_img = torch.clip(weighted_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[0, 0].imshow(viz_img)
        ax[0, 0].set_title("Weighted GT RGB")
        viz_render_img = torch.clip(weighted_render_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[1, 0].imshow(viz_render_img)
        ax[1, 0].set_title("Weighted Rendered RGB")
        ax[0, 1].imshow(weighted_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[0, 1].set_title("Weighted GT Depth")
        ax[1, 1].imshow(weighted_render_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[1, 1].set_title("Weighted Rendered Depth")
        ax[0, 2].imshow(diff_rgb, cmap="jet", vmin=0, vmax=0.8)
        ax[0, 2].set_title(f"Diff RGB, Loss: {torch.round(losses['im'])}")
        ax[1, 2].imshow(diff_depth, cmap="jet", vmin=0, vmax=0.8)
        ax[1, 2].set_title(f"Diff Depth, Loss: {torch.round(losses['depth'])}")
        ax[0, 3].imshow(presence_sil_mask.detach().cpu(), cmap="gray")
        ax[0, 3].set_title("Silhouette Mask")
        ax[1, 3].imshow(mask[0].detach().cpu(), cmap="gray")
        ax[1, 3].set_title("Loss Mask")
        # Turn off axis
        for i in range(2):
            for j in range(4):
                ax[i, j].axis('off')
        # Set Title
        fig.suptitle(f"Tracking Iteration: {tracking_iteration}", fontsize=16)
        # Figure Tight Layout
        fig.tight_layout()
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"tmp.png"), bbox_inches='tight')
        plt.close()
        plot_img = cv2.imread(os.path.join(plot_dir, f"tmp.png"))
        cv2.imshow('Diff Images', plot_img)
        cv2.waitKey(1)
        ## Save Tracking Loss Viz
        # save_plot_dir = os.path.join(plot_dir, f"tracking_%04d" % iter_time_idx)
        # os.makedirs(save_plot_dir, exist_ok=True)
        # plt.savefig(os.path.join(save_plot_dir, f"%04d.png" % tracking_iteration), bbox_inches='tight')
        # plt.close()

    weighted_losses = {
        k: v * (loss_weights['im'] if k == 'fold' else loss_weights[k])
        for k, v in losses.items()
    }
    loss = sum(weighted_losses.values())

    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    weighted_losses['loss'] = loss

    return loss, variables, weighted_losses


def initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution,
                          temporal_deformation_cfg=None):
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }
    if new_pt_cld.shape[1] > 6:
        params['fold_colors'] = new_pt_cld[:, 6:7]
    deformation_cfg = temporal_deformation_cfg or {}
    if bool(deformation_cfg.get('enable', False)):
        num_basis = max(int(deformation_cfg.get('num_basis', 2)), 1)
        params['deform_coeffs'] = torch.zeros(
            (num_pts, 3 * num_basis), dtype=torch.float, device="cuda"
        )
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    return params


def add_new_gaussians(params, variables, curr_data, sil_thres, 
                      time_idx, mean_sq_dist_method, gaussian_distribution,
                      temporal_deformation_cfg=None):
    # Silhouette Rendering
    transformed_gaussians = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_gaussians)
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    silhouette = depth_sil[1, :, :]
    non_presence_sil_mask = (silhouette < sil_thres)
    # Check for new foreground objects by using GT depth
    gt_depth = curr_data['depth'][0, :, :]
    render_depth = depth_sil[0, :, :]
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 50*depth_error.median())
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
        valid_depth_mask = (curr_data['depth'][0, :, :] > 0)
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
        new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'],
                                    curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                    mean_sq_dist_method=mean_sq_dist_method,
                                    fold_seg=curr_data.get('fold_seg'))
        new_params = initialize_new_params(
            new_pt_cld, mean3_sq_dist, gaussian_distribution,
            temporal_deformation_cfg=temporal_deformation_cfg,
        )
        for k, v in new_params.items():
            params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
        num_pts = params['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
        variables['denom'] = torch.zeros(num_pts, device="cuda").float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
        new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device="cuda").float()
        variables['timestep'] = torch.cat((variables['timestep'],new_timestep),dim=0)

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


def convert_params_to_store(params):
    params_to_store = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            params_to_store[k] = v.detach().clone()
        else:
            params_to_store[k] = v
    return params_to_store


def rgbd_slam(config: dict):
    # Print Config
    print("Loaded Config:")
    if "use_depth_loss_thres" not in config['tracking']:
        config['tracking']['use_depth_loss_thres'] = False
        config['tracking']['depth_loss_thres'] = 100000
    if "visualize_tracking_loss" not in config['tracking']:
        config['tracking']['visualize_tracking_loss'] = False
    if "gaussian_distribution" not in config:
        config['gaussian_distribution'] = "isotropic"
    print(f"{config}")

    # Create Output Directories
    output_dir = os.path.join(config["workdir"], config["run_name"])
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Init WandB
    if config['use_wandb']:
        wandb_time_step = 0
        wandb_tracking_step = 0
        wandb_mapping_step = 0
        wandb_run = wandb.init(project=config['wandb']['project'],
                               entity=config['wandb']['entity'],
                               group=config['wandb']['group'],
                               name=config['wandb']['name'],
                               config=config)

    # Get Device
    device = torch.device(config["primary_device"])

    # Load Dataset
    print("Loading Dataset ...")
    dataset_config = config["data"]
    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {}
        gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
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
        seperate_tracking_res = False
    else:
        if dataset_config["tracking_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["tracking_image_width"] != dataset_config["desired_image_width"]:
            seperate_tracking_res = True
        else:
            seperate_tracking_res = False
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
    )
    num_frames = dataset_config["num_frames"]
    if num_frames == -1:
        num_frames = len(dataset)

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
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
        )
        # Initialize Parameters, Canonical & Densification Camera parameters
        params, variables, intrinsics, first_frame_w2c, cam, \
            densify_intrinsics, densify_cam = initialize_first_timestep(dataset, num_frames,
                                                                        config['scene_radius_depth_ratio'],
                                                                        config['mean_sq_dist_method'],
                                                                        densify_dataset=densify_dataset,
                                                                        gaussian_distribution=config['gaussian_distribution'],
                                                                        fold_seg_cfg=config.get('fold_seg_loss', {}),
                                                                        temporal_deformation_cfg=config.get('temporal_deformation', {}))
    else:
        # Initialize Parameters & Canoncial Camera parameters
        params, variables, intrinsics, first_frame_w2c, cam = initialize_first_timestep(dataset, num_frames, 
                                                                                        config['scene_radius_depth_ratio'],
                                                                                        config['mean_sq_dist_method'],
                                                                                        gaussian_distribution=config['gaussian_distribution'],
                                                                                        fold_seg_cfg=config.get('fold_seg_loss', {}),
                                                                                        temporal_deformation_cfg=config.get('temporal_deformation', {}))
    
    # Init seperate dataloader for tracking if required
    if seperate_tracking_res:
        tracking_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["tracking_image_height"],
            desired_width=dataset_config["tracking_image_width"],
            device=device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
        )
        tracking_color, _, tracking_intrinsics, _ = tracking_dataset[0]
        tracking_color = tracking_color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        tracking_intrinsics = tracking_intrinsics[:3, :3]
        tracking_cam = setup_camera(tracking_color.shape[2], tracking_color.shape[1], 
                                    tracking_intrinsics.cpu().numpy(), first_frame_w2c.detach().cpu().numpy())
    
    # Initialize list to keep track of Keyframes
    keyframe_list = []
    keyframe_time_indices = []
    accum_motion_kfs_state = {'last_time_idx': None, 'motion': 0.0}
    
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
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                restored_fold = _load_fold_seg(dataset, time_idx, color.shape[1:], config.get('fold_seg_loss', {}), color.device)
                if restored_fold is not None:
                    curr_keyframe['fold_seg'] = restored_fold
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
    else:
        checkpoint_time_idx = 0
    
    # Iterate over Scan
    for time_idx in tqdm(range(checkpoint_time_idx, num_frames)):
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
        if fold_seg is not None:
            curr_data['fold_seg'] = fold_seg
            curr_data['fold_seg_weight'] = float(config.get('fold_seg_loss', {}).get('weight', 1.0))
        
        # Initialize Data for Tracking
        if seperate_tracking_res:
            tracking_color, tracking_depth, _, _ = tracking_dataset[time_idx]
            tracking_color = tracking_color.permute(2, 0, 1) / 255
            tracking_depth = tracking_depth.permute(2, 0, 1)
            tracking_curr_data = {'cam': tracking_cam, 'im': tracking_color, 'depth': tracking_depth, 'id': iter_time_idx,
                                  'intrinsics': tracking_intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
        else:
            tracking_curr_data = curr_data

        # Optimization Iterations
        num_iters_mapping = config['mapping']['num_iters']
        
        # Initialize the camera pose for the current frame
        if time_idx > 0:
            params = initialize_camera_pose(params, time_idx, forward_prop=config['tracking']['forward_prop'])

        # Tracking
        tracking_start_time = time.time()
        if time_idx > 0 and not config['tracking']['use_gt_poses']:
            # Reset Optimizer & Learning Rates for tracking
            optimizer = initialize_optimizer(params, config['tracking']['lrs'], tracking=True)
            # Keep Track of Best Candidate Rotation & Translation
            candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
            candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
            current_min_loss = float(1e20)
            # Tracking Optimization
            iter = 0
            do_continue_slam = False
            num_iters_tracking = config['tracking']['num_iters']
            progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
            while True:
                iter_start_time = time.time()
                # Loss for current frame
                loss, variables, losses = get_loss(params, tracking_curr_data, variables, iter_time_idx, config['tracking']['loss_weights'],
                                                   config['tracking']['use_sil_for_loss'], config['tracking']['sil_thres'],
                                                   config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'], tracking=True, 
                                                   plot_dir=eval_dir, visualize_tracking_loss=config['tracking']['visualize_tracking_loss'],
                                                   tracking_iteration=iter)
                if config['use_wandb']:
                    # Report Loss
                    wandb_tracking_step = report_loss(losses, wandb_run, wandb_tracking_step, tracking=True)
                # Backprop
                loss.backward()
                # Optimizer Update
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    # Save the best candidate rotation & translation
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
                        candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
                    # Report Progress
                    if config['report_iter_progress']:
                        if config['use_wandb']:
                            report_progress(params, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True,
                                            wandb_run=wandb_run, wandb_step=wandb_tracking_step, wandb_save_qual=config['wandb']['save_qual'])
                        else:
                            report_progress(params, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True)
                    else:
                        progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                tracking_iter_time_sum += iter_end_time - iter_start_time
                tracking_iter_time_count += 1
                # Check if we should stop tracking
                iter += 1
                if iter == num_iters_tracking:
                    if losses['depth'] < config['tracking']['depth_loss_thres'] and config['tracking']['use_depth_loss_thres']:
                        break
                    elif config['tracking']['use_depth_loss_thres'] and not do_continue_slam:
                        do_continue_slam = True
                        progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
                        num_iters_tracking = 2*num_iters_tracking
                        if config['use_wandb']:
                            wandb_run.log({"Tracking/Extra Tracking Iters Frames": time_idx,
                                        "Tracking/step": wandb_time_step})
                    else:
                        break

            progress_bar.close()
            # Copy over the best candidate rotation & translation
            with torch.no_grad():
                params['cam_unnorm_rots'][..., time_idx] = candidate_cam_unnorm_rot
                params['cam_trans'][..., time_idx] = candidate_cam_tran
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
        # Update the runtime numbers
        tracking_end_time = time.time()
        tracking_frame_time_sum += tracking_end_time - tracking_start_time
        tracking_frame_time_count += 1

        if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
            try:
                # Report Final Tracking Progress
                progress_bar = tqdm(range(1), desc=f"Tracking Result Time Step: {time_idx}")
                with torch.no_grad():
                    if config['use_wandb']:
                        report_progress(params, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True,
                                        wandb_run=wandb_run, wandb_step=wandb_time_step, wandb_save_qual=config['wandb']['save_qual'], global_logging=True)
                    else:
                        report_progress(params, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True)
                progress_bar.close()
            except:
                ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                save_params_ckpt(params, ckpt_output_dir, time_idx)
                print('Failed to evaluate trajectory.')

        # Densification & KeyFrame-based Mapping
        if time_idx == 0 or (time_idx+1) % config['map_every'] == 0:
            num_iters_mapping, mapping_iter_scale, base_num_iters_mapping = dynamic_mapping_num_iters(
                config, time_idx, len(keyframe_list)
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
                    densify_curr_data = {'cam': densify_cam, 'im': densify_color, 'depth': densify_depth, 'id': time_idx, 
                                 'intrinsics': densify_intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
                    densify_fold = _load_fold_seg(densify_dataset, time_idx, densify_color.shape[1:], config.get('fold_seg_loss', {}), densify_color.device)
                    if densify_fold is not None:
                        densify_curr_data['fold_seg'] = densify_fold
                else:
                    densify_curr_data = curr_data

                # Add new Gaussians to the scene based on the Silhouette
                params, variables = add_new_gaussians(params, variables, densify_curr_data, 
                                                      config['mapping']['sil_thres'], time_idx,
                                                      config['mean_sq_dist_method'], config['gaussian_distribution'],
                                                      temporal_deformation_cfg=config.get('temporal_deformation', {}))
                post_num_pts = params['means3D'].shape[0]
                if config['use_wandb']:
                    wandb_run.log({"Mapping/Number of Gaussians": post_num_pts,
                                   "Mapping/step": wandb_time_step})
            
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
            optimizer = initialize_optimizer(params, config['mapping']['lrs'], tracking=False) 

            # Mapping
            mapping_start_time = time.time()
            if num_iters_mapping > 0:
                progress_bar = tqdm(range(num_iters_mapping), desc=f"Mapping Time Step: {time_idx}")
            for iter in range(num_iters_mapping):
                iter_start_time = time.time()
                # Randomly select a frame until current time step amongst keyframes
                rand_idx = np.random.randint(0, len(selected_keyframes))
                selected_rand_keyframe_idx = selected_keyframes[rand_idx]
                if selected_rand_keyframe_idx == -1:
                    # Use Current Frame Data
                    iter_time_idx = time_idx
                    iter_color = color
                    iter_depth = depth
                    iter_fold = fold_seg
                else:
                    # Use Keyframe Data
                    iter_time_idx = keyframe_list[selected_rand_keyframe_idx]['id']
                    iter_color = keyframe_list[selected_rand_keyframe_idx]['color']
                    iter_depth = keyframe_list[selected_rand_keyframe_idx]['depth']
                    iter_fold = keyframe_list[selected_rand_keyframe_idx].get('fold_seg')
                iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx+1]
                iter_data = {'cam': cam, 'im': iter_color, 'depth': iter_depth, 'id': iter_time_idx, 
                             'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}
                if iter_fold is not None:
                    iter_data['fold_seg'] = iter_fold
                    iter_data['fold_seg_weight'] = float(config.get('fold_seg_loss', {}).get('weight', 1.0))
                    iter_data['defer_fold_backward'] = bool(
                        config.get('fold_seg_loss', {}).get('memory_efficient_backward', False)
                    )
                iter_data['deformation_fold_soft_alpha'] = float(
                    config.get('temporal_deformation', {}).get('fold_soft_alpha', 0.5)
                )
                # Loss for current frame
                loss, variables, losses = get_loss(params, iter_data, variables, iter_time_idx, config['mapping']['loss_weights'],
                                                config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                                                config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'], mapping=True)
                deferred_fold_loss = None
                if iter_data.get('defer_fold_backward', False):
                    # Backprop RGB-D first and release its two rasterizer graphs
                    # before constructing the Fold graph. The optimizer is not
                    # stepped between them, so this preserves the summed gradient.
                    loss.backward()
                    del loss
                    deferred_fold_loss = (
                        _fold_mapping_loss(params, iter_data, iter_time_idx)
                        * config['mapping']['loss_weights']['im']
                    )
                    losses['fold'] = deferred_fold_loss
                    losses['loss'] = losses['loss'].detach() + deferred_fold_loss.detach()
                if config['use_wandb']:
                    # Report Loss
                    wandb_mapping_step = report_loss(losses, wandb_run, wandb_mapping_step, mapping=True)
                # Backprop
                if deferred_fold_loss is None:
                    loss.backward()
                else:
                    deferred_fold_loss.backward()
                with torch.no_grad():
                    # Prune Gaussians
                    if config['mapping']['prune_gaussians']:
                        params, variables = prune_gaussians(params, variables, optimizer, iter, config['mapping']['pruning_dict'])
                        if config['use_wandb']:
                            wandb_run.log({"Mapping/Number of Gaussians - Pruning": params['means3D'].shape[0],
                                           "Mapping/step": wandb_mapping_step})
                    # Gaussian-Splatting's Gradient-based Densification
                    if config['mapping']['use_gaussian_splatting_densification']:
                        params, variables = densify(params, variables, optimizer, iter, config['mapping']['densify_dict'])
                        if config['use_wandb']:
                            wandb_run.log({"Mapping/Number of Gaussians - Densification": params['means3D'].shape[0],
                                           "Mapping/step": wandb_mapping_step})
                    # Optimizer Update
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    # Report Progress
                    if config['report_iter_progress']:
                        if config['use_wandb']:
                            report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            wandb_run=wandb_run, wandb_step=wandb_mapping_step, wandb_save_qual=config['wandb']['save_qual'],
                                            mapping=True, online_time_idx=time_idx)
                        else:
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

            deformation_cfg = config.get('temporal_deformation', {})
            deformation_log_every = int(deformation_cfg.get('log_every', 25))
            if ('deform_coeffs' in params and
                    (deformation_log_every <= 0 or time_idx % deformation_log_every == 0)):
                with torch.no_grad():
                    current_delta = temporal_deformation_delta(
                        params, time_idx, gaussians_grad=False
                    )
                    deformation_rms = torch.sqrt(
                        torch.mean(torch.sum(current_delta * current_delta, dim=-1))
                    )
                    deformation_mean = torch.linalg.norm(current_delta.mean(dim=0))
                    deformation_max = torch.linalg.norm(current_delta, dim=-1).max()
                print(
                    f"\n[RestrictedDeformation] Frame {time_idx}: "
                    f"rms={float(deformation_rms):.6f} "
                    f"mean={float(deformation_mean):.6f} "
                    f"max={float(deformation_max):.6f} "
                    f"gaussians={params['means3D'].shape[0]}"
                )

            if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
                try:
                    # Report Mapping Progress
                    progress_bar = tqdm(range(1), desc=f"Mapping Result Time Step: {time_idx}")
                    with torch.no_grad():
                        if config['use_wandb']:
                            report_progress(params, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            wandb_run=wandb_run, wandb_step=wandb_time_step, wandb_save_qual=config['wandb']['save_qual'],
                                            mapping=True, online_time_idx=time_idx, global_logging=True)
                        else:
                            report_progress(params, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            mapping=True, online_time_idx=time_idx)
                    progress_bar.close()
                except:
                    ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                    save_params_ckpt(params, ckpt_output_dir, time_idx)
                    print('Failed to evaluate trajectory.')
        
        selection_mode = config.get('keyframe_selection_mode', 'accum_motion_redundancy')
        if selection_mode == 'fixed':
            add_keyframe = ((time_idx == 0) or ((time_idx + 1) % config['keyframe_every'] == 0) or
                            (time_idx == num_frames - 2))
        elif selection_mode == 'endogslam_v4':
            add_keyframe, kfs_debug = _endogslam_v4_motion_decision(
                time_idx, num_frames, params, keyframe_list, config)
            add_keyframe, gate_reason = _endogslam_v4_ref_feature_gate(
                add_keyframe, kfs_debug, color, keyframe_list, config)
            log_every = int(config['accum_motion_redundancy'].get('log_every', 25))
            if add_keyframe or log_every <= 0 or time_idx % log_every == 0:
                print(f"\n[EndoGSLAMv4KFS] {'add' if add_keyframe else 'hold'} frame={time_idx} "
                      f"reason={kfs_debug['reason']} gate={gate_reason} stats={kfs_debug}")
        else:
            # The latest accumulated-motion + redundancy policy replaces fixed/distance KFS.
            add_keyframe, kfs_debug = _accum_motion_keyframe_decision(
                time_idx, num_frames, params, keyframe_list, config, accum_motion_kfs_state)
            add_keyframe, gate_reason = _redundancy_gate(add_keyframe, kfs_debug, color, keyframe_list, config)
            log_every = int(config['accum_motion_redundancy'].get('log_every', 25))
            if add_keyframe or log_every <= 0 or time_idx % log_every == 0:
                print(f"\n[AccumMotionRedundancyKFS] {'add' if add_keyframe else 'hold'} frame={time_idx} "
                      f"reason={kfs_debug['reason']} gate={gate_reason} stats={kfs_debug}")
        valid_keyframe_pose = (not torch.isinf(curr_gt_w2c[-1]).any()) and (not torch.isnan(curr_gt_w2c[-1]).any())
        if add_keyframe and valid_keyframe_pose:
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
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
                keyframe_time_indices.append(time_idx)
        
        # Checkpoint every iteration
        if time_idx % config["checkpoint_interval"] == 0 and config['save_checkpoints']:
            ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
            save_params_ckpt(params, ckpt_output_dir, time_idx)
            np.save(os.path.join(ckpt_output_dir, f"keyframe_time_indices{time_idx}.npy"), np.array(keyframe_time_indices))
        
        # Increment WandB Time Step
        if config['use_wandb']:
            wandb_time_step += 1

        torch.cuda.empty_cache()

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
    if config['use_wandb']:
        wandb_run.log({"Final Stats/Average Tracking Iteration Time (ms)": tracking_iter_time_avg*1000,
                       "Final Stats/Average Tracking Frame Time (s)": tracking_frame_time_avg,
                       "Final Stats/Average Mapping Iteration Time (ms)": mapping_iter_time_avg*1000,
                       "Final Stats/Average Mapping Frame Time (s)": mapping_frame_time_avg,
                       "Final Stats/step": 1})
    
    # Add Camera Parameters and persist the completed reconstruction before the
    # comparatively long final render evaluation. This keeps a usable result if
    # an external job scheduler interrupts evaluation.
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

    # Evaluate Final Parameters
    with torch.no_grad():
        if config['use_wandb']:
            eval(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                 wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                 mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                 eval_every=config['eval_every'])
        else:
            eval(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                 mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                 eval_every=config['eval_every'])

    # Close WandB Run
    if config['use_wandb']:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

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
