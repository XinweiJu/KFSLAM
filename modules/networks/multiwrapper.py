import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .scorenet import ScoreNet, GumbelTopKSelector

# --- utility functions ---

def _to_gray(x):
    return (0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]).clamp(0.0, 1.0)


def _sobel_xy(x):  # x: (B,1,H,W)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return gx, gy


def _sobel_grad(x):  # x: (B,1,H,W)
    gx, gy = _sobel_xy(x)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def _build_grid(B, H, W, dx, dy, device, dtype):
    ix = torch.linspace(-1, 1, W, device=device, dtype=dtype).view(1, 1, 1, W).expand(B, 1, H, W)
    iy = torch.linspace(-1, 1, H, device=device, dtype=dtype).view(1, 1, H, 1).expand(B, 1, H, W)
    ndx = 2.0 * dx / max(W - 1, 1)
    ndy = 2.0 * dy / max(H - 1, 1)
    gx = ix - ndx.view(B, 1, 1, 1)
    gy = iy - ndy.view(B, 1, 1, 1)
    return torch.stack([gx.squeeze(1), gy.squeeze(1)], dim=-1)  # (B,H,W,2)


def _local_corr_volume(t_feat, s_feat, d=4, metric="cos"):
    B, C, H, W = t_feat.shape
    ds = torch.arange(-d, d + 1, device=t_feat.device)
    disp = torch.stack(torch.meshgrid(ds, ds, indexing='ij'), dim=-1).view(-1, 2)  # (K,2)
    if metric == "cos":
        if C == 1:
            # A single-channel "cosine over channels" collapses to a constant.
            # Use globally centered/scaled maps instead so shifted correlations
            # still carry structure for the softmax over local displacements.
            t_base = t_feat - t_feat.mean(dim=[2, 3], keepdim=True)
            s_base = s_feat - s_feat.mean(dim=[2, 3], keepdim=True)
            t_n = t_base / t_base.square().mean(dim=[2, 3], keepdim=True).sqrt().clamp_min(1e-6)
            s_n = s_base / s_base.square().mean(dim=[2, 3], keepdim=True).sqrt().clamp_min(1e-6)
        else:
            t_n = F.normalize(t_feat, dim=1)
            s_n = F.normalize(s_feat, dim=1)
    corrs = []
    for (dx, dy) in disp:
        grid = _build_grid(B, H, W, dx.expand(B, 1), dy.expand(B, 1), t_feat.device, t_feat.dtype)
        if metric == "cos":
            s_warp = F.grid_sample(s_n, grid, mode='bilinear', padding_mode='border', align_corners=True)
            sim = (t_n * s_warp).sum(1, keepdim=True)
        else:
            s_warp = F.grid_sample(s_feat, grid, mode='bilinear', padding_mode='border', align_corners=True)
            sim = - (t_feat - s_warp).abs().mean(1, keepdim=True)
        corrs.append(sim)
    corr = torch.cat(corrs, dim=1)  # (B,K,H,W)
    return corr, disp.to(t_feat)


def _corr_to_stats(corr, disp, temperature=0.5):
    B, K, H, W = corr.shape
    prob = F.softmax(corr / temperature, dim=1)
    dx = disp[:, 0].view(1, K, 1, 1).to(corr)
    dy = disp[:, 1].view(1, K, 1, 1).to(corr)
    ex = (prob * dx).sum(1, keepdim=True)
    ey = (prob * dy).sum(1, keepdim=True)
    flow_mu = torch.cat([ex, ey], dim=1)  # (B,2,H,W)
    prob_max, _ = prob.max(1, keepdim=True)
    entropy = -(prob.clamp_min(1e-9) * prob.clamp_min(1e-9).log()).sum(1, keepdim=True)
    return flow_mu, prob_max, entropy


def _stabilize_pair_vec(
    pair_vec,
    use_log1p=False,
    use_zscore=False,
    clip_value=0.0,
    eps=1e-6,
):
    """
    Stabilize proxy features before ScoreNet:
      - optional log1p for heavy-tailed positive channels
      - optional per-sample z-score on candidate dimension T
      - optional symmetric clipping
    pair_vec: (B,T,8)
    """
    x = pair_vec

    if use_log1p:
        x = x.clone()
        # positive heavy-tailed channels: mae, contrast, edge_t, edge_s, flow_mag, entropy
        pos_idx = [0, 1, 2, 3, 5, 7]
        x[..., pos_idx] = torch.log1p(torch.clamp(x[..., pos_idx], min=0.0, max=1e3))
        # bounded channels
        x[..., 4] = x[..., 4].clamp(-1.0, 1.0)  # grad overlap
        x[..., 6] = x[..., 6].clamp(0.0, 1.0)   # prob max

    if use_zscore:
        mu = x.mean(dim=1, keepdim=True)
        sigma = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
        x = (x - mu) / sigma

    if clip_value is not None and float(clip_value) > 0.0:
        x = x.clamp(min=-float(clip_value), max=float(clip_value))

    return x


def compute_pair_proxies_pose_free(target_rgb, ref_rgbs, max_disp=4, temp=0.5):
    """
    target_rgb: (B,3,H,W)
    ref_rgbs:   (B,T,3,H,W)
    return: pair_vec (B,T,8)
    """
    B, T, _, H, W = ref_rgbs.shape
    t_gray = _to_gray(target_rgb)
    t_gx, t_gy = _sobel_xy(t_gray)
    t_grad = _sobel_grad(t_gray)
    t_corr_feat = torch.cat([t_gray - t_gray.mean(dim=[2, 3], keepdim=True), t_gx, t_gy], dim=1)

    vecs = []
    disp_norm = float(max(max_disp, 1))
    for i in range(T):
        s = ref_rgbs[:, i]
        s_gray = _to_gray(s)
        s_gx, s_gy = _sobel_xy(s_gray)
        s_grad = _sobel_grad(s_gray)
        s_corr_feat = torch.cat([s_gray - s_gray.mean(dim=[2, 3], keepdim=True), s_gx, s_gy], dim=1)

        diff = (t_gray - s_gray).abs()
        mae = diff.mean([1, 2, 3], keepdim=True)

        std_t = t_gray.std([1, 2, 3], keepdim=True) + 1e-6
        std_s = s_gray.std([1, 2, 3], keepdim=True) + 1e-6
        contrast = (std_t / std_s).log().abs()

        tg_flat = torch.cat([t_gx, t_gy], dim=1).reshape(B, -1)
        sg_flat = torch.cat([s_gx, s_gy], dim=1).reshape(B, -1)
        grad_overlap = F.cosine_similarity(tg_flat, sg_flat, dim=1, eps=1e-6).view(B, 1, 1, 1)

        edge_t = t_grad.mean([1, 2, 3], keepdim=True)
        edge_s = s_grad.mean([1, 2, 3], keepdim=True)

        corr, disp = _local_corr_volume(t_corr_feat, s_corr_feat, d=max_disp, metric="cos")
        flow_mu, prob_max, entropy = _corr_to_stats(corr, disp, temperature=temp)
        flow_mag = torch.sqrt(flow_mu[:, 0:1] ** 2 + flow_mu[:, 1:2] ** 2)
        flow_mag_mean = (flow_mag / disp_norm).mean([2, 3], keepdim=True)
        prob_max_mean = prob_max.mean([2, 3], keepdim=True)
        # Normalize entropy by log(K) where K is correlation candidate count.
        # Channel-axis normalization would be degenerate here because entropy has shape (B,1,H,W).
        k_corr = float(max(corr.shape[1], 2))
        ent_norm = entropy / max(math.log(k_corr), 1e-6)
        entropy_mean = ent_norm.mean([2, 3], keepdim=True)

        vec = torch.cat(
            [mae, contrast, edge_t, edge_s, grad_overlap, flow_mag_mean, prob_max_mean, entropy_mean], dim=1
        )
        vecs.append(vec)

    # Keep shape stable as (B, T, 8), even when B==1 or T==1.
    # Each vec is (B,8,1,1) due keepdim=True reductions.
    return torch.stack(vecs, dim=1).reshape(B, T, 8)


def _edge_density(x):  # x: (B,3,H,W)
    g = x.mean(dim=1, keepdim=True)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(g, kx, padding=1)
    gy = F.conv2d(g, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy).mean(dim=[2, 3])


def build_pair_feats(target, refs, ref_ids):
    B = target.shape[0]
    t_edge = _edge_density(target)
    feats = []
    for ref, fid in zip(refs, ref_ids):
        absdiff = (target - ref).abs().mean(dim=1, keepdim=True)
        flow_proxy = absdiff.view(B, -1).median(dim=1).values.view(B, 1)
        ref_edge = _edge_density(ref)
        dt = torch.full((B, 1), float(abs(int(fid))), device=target.device)
        feat = torch.cat([dt, flow_proxy, flow_proxy, t_edge, ref_edge, flow_proxy], dim=1)
        feats.append(feat)
    return torch.stack(feats, dim=1)


class SimplePoseEncoder(nn.Module):
    def __init__(self, in_ch=6, base=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SimplePoseNet(nn.Module):
    def __init__(self, in_ch=16):
        super().__init__()
        self.axis_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(in_ch, 3))
        self.trans_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(in_ch, 3))

    def forward(self, feats_list):
        feats = feats_list[0]
        axis = self.axis_head(feats)
        trans = self.trans_head(feats)
        return axis, trans


def se3_from_params(axis, trans, invert=False):
    # axis, trans: (B,1,3)
    B = axis.shape[0]
    rot = axis.view(B, 3)
    angle = torch.linalg.norm(rot + 1e-9, dim=1, keepdim=True)
    axis_unit = rot / (angle + 1e-9)
    angle = angle.clamp(max=1e6)

    def skew(v):
        vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
        zeros = torch.zeros_like(vx)
        return torch.stack(
            [zeros, -vz, vy, vz, zeros, -vx, -vy, vx, zeros], dim=1
        ).view(-1, 3, 3)

    K = skew(axis_unit)
    I = torch.eye(3, device=axis.device, dtype=axis.dtype).unsqueeze(0).expand(B, -1, -1)
    angle = angle.view(B, 1, 1)
    R = I + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
    t = trans.view(B, 3, 1)

    T = torch.zeros((B, 4, 4), device=axis.device, dtype=axis.dtype)
    T[:, :3, :3] = R
    T[:, :3, 3:4] = t
    T[:, 3, 3] = 1.0
    if invert:
        Rinvt = R.transpose(1, 2)
        tinv = -Rinvt @ t
        T[:, :3, :3] = Rinvt
        T[:, :3, 3:4] = tinv
    return T


class MultiPoseKeyFrameWrapper(nn.Module):
    def __init__(self, pose_encoder, pose_net, score_net, selector, se3_from_params_fn):
        super().__init__()
        self.pose_encoder = pose_encoder
        self.pose_net = pose_net
        self.score_net = score_net
        self.selector = selector
        self.se3_from_params = se3_from_params_fn
        # Runtime-tunable feature stabilization flags (set by loader/config).
        self.pair_use_log1p = False
        self.pair_use_zscore = False
        self.pair_clip_value = 0.0
        self.pair_max_disp = 4
        self.pair_corr_temp = 0.5
        # Selection behavior:
        # - model_weighted: use learned selector output (default)
        # - topk_unweighted: choose top-k by score, equal weights within top-k
        # - constant_equal: deterministic recency top-k, equal weights over all candidates
        self.selection_policy = "model_weighted"
        # Pose direction convention flag.
        self.pose_invert_by_fid_sign = True

    def _predict_single_pose(self, target_b1, ref_b1, fid):
        pose_in = torch.cat([target_b1, ref_b1], dim=1)
        feat = self.pose_encoder(pose_in)
        axis, trans = self.pose_net([feat])
        axis = axis.flatten(start_dim=1)[..., :3].reshape(-1, 1, 3).contiguous()
        trans = trans.flatten(start_dim=1)[..., :3].reshape(-1, 1, 3).contiguous()
        invert_flag = bool(self.pose_invert_by_fid_sign) and (fid < 0)
        T = self.se3_from_params(axis, trans, invert=invert_flag)
        return axis, trans, T

    @staticmethod
    def _topk_by_recency(ids_tensor: torch.Tensor, batch_size: int, k_eff: int) -> torch.Tensor:
        # If ids are absolute increasing frame ids, larger means newer.
        # If ids are relative signed offsets, smaller |fid| means temporally closer.
        ids_f = ids_tensor.float()
        if float(ids_f.min().item()) >= 0.0:
            recency_score = ids_f
        else:
            recency_score = -ids_f.abs()
        recency_score = recency_score.unsqueeze(0).expand(batch_size, -1)
        return torch.topk(recency_score, k=k_eff, dim=1).indices

    def forward(self, target, refs, ref_ids):
        B = target.shape[0]
        refs_st = torch.stack(refs, dim=1) if isinstance(refs, (list, tuple)) else refs
        n_cand = refs_st.shape[1]

        pair_vec = compute_pair_proxies_pose_free(
            target_rgb=target,
            ref_rgbs=refs_st,
            max_disp=int(getattr(self, "pair_max_disp", 4)),
            temp=float(getattr(self, "pair_corr_temp", 0.5)),
        )
        if pair_vec.dim() == 2:
            pair_vec = pair_vec.unsqueeze(0)
        pair_vec = _stabilize_pair_vec(
            pair_vec,
            use_log1p=bool(self.pair_use_log1p),
            use_zscore=bool(self.pair_use_zscore),
            clip_value=float(self.pair_clip_value),
        )

        scores = self.score_net(pair_vec)
        if scores.dim() == 3:
            scores = scores.squeeze(-1)

        if torch.is_tensor(ref_ids):
            ids_tensor = ref_ids.to(target.device).long()
        else:
            ids_tensor = torch.tensor(ref_ids, device=target.device, dtype=torch.long)

        k_eff = min(max(int(getattr(self.selector, "k", 1)), 1), n_cand)
        policy = str(getattr(self, "selection_policy", "model_weighted")).lower()
        if policy == "constant_equal":
            topk = self._topk_by_recency(ids_tensor, B, k_eff)
            weights = torch.full_like(scores, 1.0 / float(max(n_cand, 1)))
        elif policy == "topk_unweighted":
            topk = torch.topk(scores, k=k_eff, dim=1).indices
            weights = torch.zeros_like(scores)
            weights.scatter_(1, topk, 1.0 / float(max(k_eff, 1)))
        else:
            weights, topk = self.selector(scores)

        ids_expand = ids_tensor.unsqueeze(0).expand(scores.size(0), -1)
        selected_fids = torch.gather(ids_expand, 1, topk)
        weights_prob = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-9)

        outputs = {
            "scores": scores,
            "selection_weights": weights,
            "top_k_indices": topk,
            "selected_frame_ids": selected_fids,
            "selection_policy": policy,
            "selection_entropy": (-(weights_prob * weights_prob.clamp_min(1e-9).log())).sum(1, keepdim=True),
            "delta_ts": ids_expand.abs(),
            "pair_vec": pair_vec,
        }

        # Optional pose predictions for top-k
        for b in range(B):
            for j in range(topk.shape[1]):
                idx = int(topk[b, j])
                fid = int(ids_tensor[idx])
                t_b = target[b : b + 1].contiguous()
                r_b = refs_st[b : b + 1, idx].contiguous()
                axis, trans, T = self._predict_single_pose(t_b, r_b, fid)
                outputs[("axisangle", b, fid)] = axis
                outputs[("translation", b, fid)] = trans
                outputs[("cam_T_cam", b, fid)] = T

        return outputs
