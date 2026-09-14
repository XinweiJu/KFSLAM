from collections import OrderedDict
import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt
_FOLD_BOUNDARY_DT_CACHE=OrderedDict()
_FOLD_TARGET_DT_CACHE=OrderedDict()
_FOLD_BOUNDARY_DT_CACHE_MAX=256
_FOLD_TARGET_DT_CACHE_MAX=256
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
