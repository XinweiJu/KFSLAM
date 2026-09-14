import glob
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.slam_helpers import matrix_to_quaternion
from utils.slam_external import build_rotation


def _add_keyframegraph_to_path(root: str) -> None:
    if root and root not in sys.path:
        sys.path.insert(0, root)


def load_scorenet_wrapper(cfg: dict) -> torch.nn.Module:
    """
    Load MultiPoseKeyFrameWrapper from KeyFrameGraph.
    Expects cfg keys:
      - keyframe_graph_root
      - weights_dir
      - topk
    """
    mode = str(cfg.get("mode", "model")).lower()
    if mode == "proxy_scalar":
        model = ProxyScalarWrapper(cfg)
        model.eval()
        return model
    if mode in ("mmr", "submodular"):
        model = DiversityWrapper(cfg)
        model.eval()
        return model

    kfg_root = cfg.get("keyframe_graph_root", "")
    weights_dir = cfg.get("weights_dir", "")
    topk = int(cfg.get("topk", 5))

    if not kfg_root:
        raise ValueError("scorenet.keyframe_graph_root is empty")
    if not weights_dir:
        raise ValueError("scorenet.weights_dir is empty")

    _add_keyframegraph_to_path(kfg_root)

    from pose_pred.model_loader import load_keyframe_model

    weights_path = _resolve_weights_path(weights_dir)
    model = load_keyframe_model(weights_path=weights_path, k_frames=topk)
    # Make selection deterministic in eval
    if hasattr(model, "selector"):
        model.selector.stochastic = False
    # Selection/pose toggles for ablation and bug isolation.
    if hasattr(model, "selection_policy"):
        model.selection_policy = str(cfg.get("selection_policy", "model_weighted")).lower()
    if hasattr(model, "pose_invert_by_fid_sign"):
        model.pose_invert_by_fid_sign = bool(cfg.get("pose_invert_by_fid_sign", True))
    # Optional proxy-feature stabilization switches.
    if hasattr(model, "pair_use_log1p"):
        model.pair_use_log1p = bool(cfg.get("pair_log1p_enable", False))
    if hasattr(model, "pair_use_zscore"):
        model.pair_use_zscore = bool(cfg.get("pair_zscore_enable", False))
    if hasattr(model, "pair_clip_value"):
        model.pair_clip_value = float(cfg.get("pair_clip_value", 0.0))
    if hasattr(model, "pair_max_disp"):
        model.pair_max_disp = int(cfg.get("pair_max_disp", 4))
    if hasattr(model, "pair_corr_temp"):
        model.pair_corr_temp = float(cfg.get("pair_corr_temp", 0.5))
    model.eval()
    return model


def _resolve_weights_path(weights_dir: str) -> str:
    if not weights_dir:
        return ""
    if os.path.isfile(weights_dir):
        return weights_dir
    if os.path.isdir(weights_dir):
        # 1) Strong preference: full wrapper checkpoint.
        direct_full = os.path.join(weights_dir, "multi_pose_wrapper.pth")
        if os.path.isfile(direct_full):
            return direct_full
        # 2) Then any wrapper-like checkpoint (recursive; newer first).
        wrapper_cands = sorted(
            glob.glob(os.path.join(weights_dir, "**", "multi_pose_wrapper*.pth"), recursive=True),
            key=lambda p: os.path.getmtime(p),
            reverse=True,
        )
        if wrapper_cands:
            return wrapper_cands[0]
        # 3) Fallback to any torch checkpoint (recursive; newer first).
        any_cands = sorted(
            glob.glob(os.path.join(weights_dir, "**", "*.pth"), recursive=True)
            + glob.glob(os.path.join(weights_dir, "**", "*.pt"), recursive=True),
            key=lambda p: os.path.getmtime(p),
            reverse=True,
        )
        if any_cands:
            return any_cands[0]
    return ""


def _to_gray(x: torch.Tensor) -> torch.Tensor:
    return (0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]).clamp(0.0, 1.0)


def _sobel_grad(x: torch.Tensor) -> torch.Tensor:
    # x: (B,1,H,W)
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def _proxy_scores(target: torch.Tensor, refs_st: torch.Tensor, ref_ids: List[int]) -> torch.Tensor:
    """
    Hand-crafted scalar score logits for each ref frame.
    target: (B,3,H,W), refs_st: (B,T,3,H,W)
    returns scores: (B,T)
    """
    b, t, _, _, _ = refs_st.shape
    t_gray = _to_gray(target)
    t_grad = _sobel_grad(t_gray)
    std_t = t_gray.std(dim=[1, 2, 3], keepdim=True) + 1e-6

    if len(ref_ids) == t:
        ids = torch.tensor(ref_ids, device=target.device, dtype=target.dtype).view(1, t)
        recency = (ids - ids.min()) / (ids.max() - ids.min() + 1e-6)
        recency = recency * 2.0 - 1.0
    else:
        recency = torch.linspace(-1.0, 1.0, t, device=target.device, dtype=target.dtype).view(1, t)
    recency = recency.expand(b, -1)

    outs = []
    for i in range(t):
        s = refs_st[:, i]
        s_gray = _to_gray(s)
        s_grad = _sobel_grad(s_gray)

        mae = (t_gray - s_gray).abs().mean(dim=[1, 2, 3])
        std_s = s_gray.std(dim=[1, 2, 3], keepdim=True) + 1e-6
        contrast = (std_t / std_s).log().abs().view(-1)

        tg = t_grad.view(b, -1)
        sg = s_grad.view(b, -1)
        grad_overlap = F.cosine_similarity(tg, sg, dim=1, eps=1e-6)

        edge_s = s_grad.mean(dim=[1, 2, 3])
        absdiff_median = (target - s).abs().mean(dim=1).reshape(b, -1).median(dim=1).values

        score = (
            -2.5 * mae
            -0.5 * contrast
            +1.0 * grad_overlap
            +0.8 * edge_s
            -1.2 * absdiff_median
            +0.2 * recency[:, i]
        )
        outs.append(score)
    return torch.stack(outs, dim=1)


class ProxyScalarWrapper(nn.Module):
    """
    Non-learned proxy scoring wrapper.
    - scores are from handcrafted scalar proxy
    - optional pose edges can be generated by KeyFrameGraph pose wrapper (no ScoreNet usage)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.topk = int(cfg.get("topk", 5))
        self.temperature = float(cfg.get("sampling_temperature", 1.0))
        self.proxy_use_pose_net = bool(cfg.get("proxy_use_pose_net", False))
        self.pose_wrapper = None

        if self.proxy_use_pose_net:
            kfg_root = cfg.get("keyframe_graph_root", "")
            if not kfg_root:
                raise ValueError("scorenet.keyframe_graph_root is required when proxy_use_pose_net=True")
            _add_keyframegraph_to_path(kfg_root)
            from pose_pred.model_loader import load_keyframe_model

            weights_path = _resolve_weights_path(str(cfg.get("weights_dir", "")))
            self.pose_wrapper = load_keyframe_model(weights_path=weights_path, k_frames=self.topk)
            if hasattr(self.pose_wrapper, "selector"):
                self.pose_wrapper.selector.stochastic = False
            self.pose_wrapper.eval()

    def forward(self, target: torch.Tensor, refs, ref_ids: List[int]) -> dict:
        refs_st = torch.stack(refs, dim=1) if isinstance(refs, (list, tuple)) else refs
        scores = _proxy_scores(target, refs_st, ref_ids)  # (B,T)
        weights = torch.softmax(scores / max(self.temperature, 1e-6), dim=1)
        k = min(max(self.topk, 1), scores.shape[1])
        topk = torch.topk(scores, k=k, dim=1).indices

        if torch.is_tensor(ref_ids):
            ids_tensor = ref_ids.to(target.device).long()
        else:
            ids_tensor = torch.tensor(ref_ids, device=target.device, dtype=torch.long)
        ids_expand = ids_tensor.unsqueeze(0).expand(scores.size(0), -1)
        selected_fids = torch.gather(ids_expand, 1, topk)

        outputs = {
            "scores": scores,
            "selection_weights": weights,
            "top_k_indices": topk,
            "selected_frame_ids": selected_fids,
        }

        # Optional relative pose edges (for pose-graph long edges)
        if self.proxy_use_pose_net and self.pose_wrapper is not None:
            with torch.no_grad():
                for b in range(scores.shape[0]):
                    for j in range(topk.shape[1]):
                        idx = int(topk[b, j])
                        fid = int(ids_tensor[idx].item())
                        t_b = target[b : b + 1].contiguous()
                        r_b = refs_st[b : b + 1, idx].contiguous()
                        axis, trans, T = self.pose_wrapper._predict_single_pose(t_b, r_b, fid)
                        outputs[("axisangle", 0, fid)] = axis
                        outputs[("translation", 0, fid)] = trans
                        outputs[("cam_T_cam", 0, fid)] = T

        return outputs


class DiversityWrapper(nn.Module):
    """
    Drop-in replacement for ScoreNet that uses MMR or Submodular Facility Location
    to select diverse reference frames.  No pose head — tracking init falls back to
    forward propagation (use_for_tracking_init must be False in config).

    Supported modes (set cfg["mode"]):
      "mmr"        — Maximum Marginal Relevance  (λ controls relevance vs diversity)
      "submodular" — Facility-Location submodular maximisation
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.mode = str(cfg.get("mode", "mmr")).lower()
        self.topk = int(cfg.get("topk", 5))
        self.mmr_lambda = float(cfg.get("mmr_lambda", 0.5))
        self.submod_alpha = float(cfg.get("submod_alpha", 0.3))
        self.blur_pct = float(cfg.get("blur_pct", 20.0))
        self.use_quality_filter = bool(cfg.get("use_quality_filter", True))
        # overlap_pct: if set (0-1), only candidates with MAE_rank <= overlap_pct are
        # eligible for diversity selection (enforces minimum photometric overlap).
        # Replaces the default hard-example band (0.10-0.85).
        _op = cfg.get("overlap_pct", None)
        self.overlap_pct: Optional[float] = float(_op) if _op is not None else None
        self.max_disp = int(cfg.get("pair_max_disp", 4))
        self.temp = float(cfg.get("pair_corr_temp", 0.5))

        kfg_root = cfg.get("keyframe_graph_root", "")
        if kfg_root:
            _add_keyframegraph_to_path(kfg_root)
        from networks.multiwrapper import compute_pair_proxies_pose_free
        self._proxies = compute_pair_proxies_pose_free

        # Optional learned pose estimator for pose graph edges.
        # cfg keys: pose_model ("monodepth2"|"af-sfm"), pose_weights_dir
        self._pose_net = None
        pose_model = cfg.get("pose_model", "")
        pose_weights = cfg.get("pose_weights_dir", "")
        if pose_model and pose_weights:
            from utils.learned_pose_wrapper import load_pose_net
            self._pose_net = load_pose_net(pose_model, pose_weights)

    # ── low-level helpers ────────────────────────────────────────────────

    @staticmethod
    def _norm_rows(x: np.ndarray) -> np.ndarray:
        return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)

    def _quality_mask(self, pv: np.ndarray) -> np.ndarray:
        if not self.use_quality_filter:
            return np.ones(len(pv), dtype=bool)
        edge_t = pv[:, 2]
        blur_ok = edge_t >= float(np.percentile(edge_t, self.blur_pct))
        mae = pv[:, 0]
        T = len(mae)
        if T >= 2:
            ranks = mae.argsort().argsort().astype(np.float32) / max(T - 1, 1)
            if self.overlap_pct is not None:
                # Overlap-constrained mode: keep only the most-overlapping frames
                # (lowest MAE = most similar to target = highest photometric overlap).
                hard_ok = ranks <= self.overlap_pct
            else:
                # Default: hard-example mining band (skip trivial and hopeless refs)
                hard_ok = (ranks >= 0.10) & (ranks <= 0.85)
        else:
            hard_ok = np.ones(T, dtype=bool)
        return blur_ok & hard_ok

    @staticmethod
    def _relevance(pv: np.ndarray) -> np.ndarray:
        """Normalised edge sharpness of reference frame (edge_s, index 3)."""
        e = pv[:, 3]
        rng = e.max() - e.min()
        return ((e - e.min()) / rng).astype(np.float32) if rng > 1e-8 else np.ones(len(e), np.float32)

    def _feat_matrix(self, pv: np.ndarray) -> np.ndarray:
        f = pv.astype(np.float32).copy()
        f = (f - f.mean(0)) / f.std(0).clip(1e-6)
        return self._norm_rows(f)

    def _mmr(self, feat_n, relevance, valid_mask):
        sim = feat_n @ feat_n.T
        eligible = [i for i in range(len(relevance)) if valid_mask[i]]
        selected: List[int] = []
        lam = self.mmr_lambda
        for _ in range(min(self.topk, len(eligible))):
            if not eligible:
                break
            if not selected:
                best = max(eligible, key=lambda i: relevance[i])
            else:
                best, bs = eligible[0], -1e9
                for c in eligible:
                    sc = lam * relevance[c] - (1 - lam) * max(sim[c, s] for s in selected)
                    if sc > bs:
                        bs, best = sc, c
            selected.append(best)
            eligible.remove(best)
        return selected

    def _submodular(self, feat_n, quality, valid_mask):
        T = feat_n.shape[0]
        sim = feat_n @ feat_n.T
        eligible = [i for i in range(T) if valid_mask[i]]
        coverage = np.zeros(T, dtype=np.float32)
        selected: List[int] = []
        for _ in range(min(self.topk, len(eligible))):
            if not eligible:
                break
            best, bg = eligible[0], -1e9
            for c in eligible:
                gain = float(np.maximum(coverage, sim[:, c]).sum() - coverage.sum()) \
                       + self.submod_alpha * float(quality[c])
                if gain > bg:
                    bg, best = gain, c
            coverage = np.maximum(coverage, sim[:, best])
            selected.append(best)
            eligible.remove(best)
        return selected

    # ── forward ─────────────────────────────────────────────────────────

    def _pose_feat_matrix(self, target: torch.Tensor, refs_st: torch.Tensor, T: int) -> Optional[np.ndarray]:
        """
        Estimate relative pose for all T candidates and return (T,6) feature matrix
        [t_x, t_y, t_z, aa_x, aa_y, aa_z], z-score normalized and row-normalized.
        Returns None if pose net unavailable or all estimates fail.
        """
        if self._pose_net is None:
            return None
        feats = []
        for i in range(T):
            try:
                T_edge = self._pose_net.estimate_pose(target, refs_st[:, i])  # (1,4,4)
                R = T_edge[0, :3, :3]
                t = T_edge[0, :3, 3].cpu().numpy()
                # axisangle from rotation matrix: aa = theta * axis
                cos_a = ((R.trace() - 1.0) / 2.0).clamp(-1.0, 1.0)
                theta = torch.acos(cos_a).item()
                if theta < 1e-6:
                    aa = np.zeros(3, np.float32)
                else:
                    skew = (R - R.T) / (2.0 * np.sin(theta) + 1e-8)
                    aa = np.array([skew[2,1].item(), skew[0,2].item(), skew[1,0].item()], np.float32) * theta
                feats.append(np.concatenate([t, aa]).astype(np.float32))
            except Exception:
                feats.append(np.zeros(6, np.float32))
        f = np.stack(feats, axis=0)          # (T,6)
        std = f.std(0).clip(1e-6)
        f = (f - f.mean(0)) / std            # z-score
        norms = np.linalg.norm(f, axis=1, keepdims=True).clip(1e-8)
        return (f / norms).astype(np.float32)

    def forward(self, target: torch.Tensor, refs, ref_ids: List[int]) -> dict:
        refs_st = torch.stack(refs, dim=1) if isinstance(refs, (list, tuple)) else refs
        with torch.no_grad():
            pv_t = self._proxies(target, refs_st, self.max_disp, self.temp)  # (1,T,8)
        pv = pv_t[0].detach().cpu().float().numpy()   # (T,8)
        T = len(ref_ids)

        valid_mask = self._quality_mask(pv)
        relevance  = self._relevance(pv)

        # Use pose-based feature matrix if requested and pose net available
        use_pose_feat = bool(self.cfg.get("use_pose_for_kfs", False)) and self._pose_net is not None
        if use_pose_feat:
            pose_feat = self._pose_feat_matrix(target, refs_st, T)
            feat_n = pose_feat if pose_feat is not None else self._feat_matrix(pv)
        else:
            feat_n = self._feat_matrix(pv)

        if self.mode == "mmr":
            ordinals = self._mmr(feat_n, relevance, valid_mask)
        else:
            ordinals = self._submodular(feat_n, relevance, valid_mask)

        # Pad to topk with temporal fallback if quality filter left too few
        if len(ordinals) < self.topk:
            used = set(ordinals)
            for i in sorted(range(T), key=lambda i: -relevance[i]):
                if len(ordinals) >= self.topk:
                    break
                if i not in used:
                    ordinals.append(i)
                    used.add(i)
        ordinals = ordinals[:self.topk]

        dev = target.device
        ids_t = torch.tensor(ref_ids, device=dev, dtype=torch.long)
        # Use logit-scale scores: 0.0 for MMR/submodular-selected frames, -1e9 for others.
        # After softmax in keyframe_sampling_mixed_with_current this gives uniform prob
        # over the selected top-k frames and ~0 elsewhere.
        scores = torch.full((1, T), fill_value=-1e9, device=dev)
        weights = torch.zeros(1, T, device=dev)
        for o in ordinals:
            scores[0, o] = 0.0
            weights[0, o] = 1.0 / max(len(ordinals), 1)
        topk_t = torch.tensor(ordinals, device=dev).unsqueeze(0)            # (1,k)
        sel_fids = ids_t[ordinals].unsqueeze(0)                             # (1,k)

        out = {
            "scores": scores,                                                # (1,T) — logit mask for sampler
            "selection_weights": weights,
            "top_k_indices": topk_t,
            "selected_frame_ids": sel_fids,
        }

        # Learned pose edges for selected keyframes (used by pose graph optimizer).
        if self._pose_net is not None:
            _max_rot  = float(self.cfg.get("pose_edge_max_rot_deg",  45.0))
            _max_tran = float(self.cfg.get("pose_edge_max_trans",    0.5))
            refs_list = [refs_st[:, i] for i in range(T)]  # each (1,3,H,W)
            for o in ordinals:
                fid = int(ids_t[o].item())
                try:
                    T_edge = self._pose_net.estimate_pose(target, refs_list[o])  # (1,4,4)
                    R = T_edge[0, :3, :3]
                    t = T_edge[0, :3, 3]
                    cos_angle = ((R.trace() - 1.0) / 2.0).clamp(-1.0, 1.0)
                    rot_deg = torch.acos(cos_angle).item() * (180.0 / 3.14159265)
                    trans_norm = t.norm().item()
                    if rot_deg > _max_rot or trans_norm > _max_tran:
                        continue
                    out[("cam_T_cam", 0, fid)] = T_edge
                except Exception:
                    pass

        return out


@torch.no_grad()
def score_candidates(
    model: torch.nn.Module,
    target: torch.Tensor,
    ref_imgs: List[torch.Tensor],
    ref_ids: List[int],
) -> dict:
    """
    target: (1,3,H,W)
    ref_imgs: list of (1,3,H,W)
    ref_ids: list of frame ids
    Returns model outputs dict.
    """
    return model(target, ref_imgs, ref_ids)


def _invert_se3(T: torch.Tensor) -> torch.Tensor:
    """Invert a 4x4 SE3 matrix (batchable)."""
    R = T[..., :3, :3]
    t = T[..., :3, 3:4]
    Rt = R.transpose(-1, -2)
    t_inv = -Rt @ t
    T_inv = torch.eye(4, device=T.device, dtype=T.dtype).expand(T.shape)
    T_inv[..., :3, :3] = Rt
    T_inv[..., :3, 3:4] = t_inv
    return T_inv


@torch.no_grad()
def estimate_w2c_from_outputs(
    outputs: dict,
    ref_ids: List[int],
    ref_w2c_list: List[torch.Tensor],
    weights: Optional[torch.Tensor] = None,
    invert_relative: bool = True,
) -> Optional[torch.Tensor]:
    """
    Fuse multiple candidate w2c from predicted relative poses.
    outputs: wrapper outputs
    ref_ids: list of candidate frame ids aligned with ref_w2c_list
    ref_w2c_list: list of (4,4) w2c for refs
    weights: (T,) optional selection weights
    Returns fused w2c (4,4) or None if no candidates.
    """
    if weights is None:
        if "selection_weights" in outputs:
            weights = outputs["selection_weights"][0]
        else:
            weights = None

    # Build candidate w2c from selected refs
    candidates = []
    cand_weights = []
    for i, fid in enumerate(ref_ids):
        key = ("cam_T_cam", 0, fid)
        if key not in outputs:
            continue
        T_ref_cur = outputs[key][0]  # (4,4)
        T_cur_ref = _invert_se3(T_ref_cur) if invert_relative else T_ref_cur
        w2c_ref = ref_w2c_list[i].to(T_ref_cur.device)
        w2c_cur = T_cur_ref @ w2c_ref
        candidates.append(w2c_cur)
        if weights is not None:
            cand_weights.append(weights[i])

    if len(candidates) == 0:
        return None

    w2c_stack = torch.stack(candidates, dim=0)  # (K,4,4)

    if len(cand_weights) == 0:
        w = torch.ones((w2c_stack.shape[0],), device=w2c_stack.device, dtype=w2c_stack.dtype)
    else:
        w = torch.stack(cand_weights).to(w2c_stack.device).clamp_min(1e-6)
    w = w / w.sum()

    # Weighted average translation
    t = w2c_stack[:, :3, 3]
    t_fused = (t * w[:, None]).sum(dim=0)

    # Weighted average rotation via quaternions
    R = w2c_stack[:, :3, :3]
    q = matrix_to_quaternion(R)  # (K,4) wxyz
    q = q * w[:, None]
    q_fused = q.sum(dim=0, keepdim=True)
    R_fused = build_rotation(q_fused)

    w2c_fused = torch.eye(4, device=w2c_stack.device, dtype=w2c_stack.dtype)
    w2c_fused[:3, :3] = R_fused[0]
    w2c_fused[:3, 3] = t_fused
    return w2c_fused
