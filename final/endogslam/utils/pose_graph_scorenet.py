import torch
import torch.nn.functional as F

from utils.slam_helpers import matrix_to_quaternion
from utils.slam_external import build_rotation
from utils.scorenet_selection import score_candidates


def _se3_inverse(T: torch.Tensor) -> torch.Tensor:
    R = T[:3, :3]
    t = T[:3, 3]
    Rt = R.transpose(0, 1)
    out = torch.eye(4, device=T.device, dtype=T.dtype)
    out[:3, :3] = Rt
    out[:3, 3] = -(Rt @ t)
    return out


def _rel_from_w2c(w2c_ref: torch.Tensor, w2c_cur: torch.Tensor) -> torch.Tensor:
    # T_ref_cur
    return w2c_cur @ _se3_inverse(w2c_ref)


def _rot_angle_from_R(R: torch.Tensor) -> torch.Tensor:
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    cos = ((tr - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos)


def _rot_loss_residual_from_R(R: torch.Tensor) -> torch.Tensor:
    """
    Stable rotation residual for optimization:
      residual = 1 - cos(theta) in [0, 2]
    This avoids acos gradient explosion near theta ~= 0.
    """
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    cos = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    return 1.0 - cos


def _huber_scalar(x: torch.Tensor, delta: float) -> torch.Tensor:
    d = max(float(delta), 1e-8)
    ax = torch.abs(x)
    return torch.where(ax <= d, 0.5 * ax * ax, d * (ax - 0.5 * d))


def _score_to_info_scale(score: float, n_ref: int, gamma: float, w_min: float, w_max: float) -> float:
    """
    Map score probability to a multiplicative information-scale for long edges.
    Uses score relative to uniform probability (1 / n_ref) as confidence ratio.
    """
    n = max(int(n_ref), 1)
    g = max(float(gamma), 1e-6)
    ratio = max(float(score), 1e-9) * float(n)  # uniform => ~1.0
    scaled = ratio ** g
    return float(max(float(w_min), min(float(w_max), scaled)))


def _score_map_from_outputs(outputs: dict, ref_ids: list[int], score_temp: float, score_from_scores: bool) -> dict[int, float]:
    if score_from_scores and ("scores" in outputs):
        scores = outputs["scores"][0].detach()
        prob = torch.softmax(scores / max(score_temp, 1e-6), dim=0)
        return {int(fid): float(prob[i].item()) for i, fid in enumerate(ref_ids)}
    if "selection_weights" in outputs:
        w = outputs["selection_weights"][0].detach()
        w = w / w.sum().clamp_min(1e-9)
        return {int(fid): float(w[i].item()) for i, fid in enumerate(ref_ids)}
    if "scores" in outputs:
        scores = outputs["scores"][0].detach()
        prob = torch.softmax(scores / max(score_temp, 1e-6), dim=0)
        return {int(fid): float(prob[i].item()) for i, fid in enumerate(ref_ids)}
    uni = 1.0 / float(max(len(ref_ids), 1))
    return {int(fid): uni for fid in ref_ids}


def _selected_fids_from_outputs(
    outputs: dict,
    ref_ids: list[int],
    score_map: dict[int, float],
    use_all_candidates: bool,
    use_selected_only: bool,
    max_edges: int,
) -> list[int]:
    if use_all_candidates:
        selected = sorted((int(x) for x in ref_ids), key=lambda x: score_map.get(x, 0.0), reverse=True)
    elif use_selected_only and ("selected_frame_ids" in outputs):
        selected = []
        seen = set()
        for fid in outputs["selected_frame_ids"][0].detach().cpu().tolist():
            fid_i = int(fid)
            if fid_i in seen:
                continue
            seen.add(fid_i)
            selected.append(fid_i)
    else:
        selected = []

    if len(selected) == 0:
        selected = sorted((int(x) for x in ref_ids), key=lambda x: score_map.get(x, 0.0), reverse=True)
    return selected[: max(max_edges, 0)]


def _predict_pose_edge_on_demand(
    scorenet_model,
    target: torch.Tensor,
    ref_img: torch.Tensor,
    fid: int,
):
    """
    Predict one relative pose edge when wrapper output does not include it.
    Returns (4,4) tensor or None.
    """
    if hasattr(scorenet_model, "_predict_single_pose"):
        _, _, T = scorenet_model._predict_single_pose(
            target.contiguous(),
            ref_img.contiguous(),
            int(fid),
        )
        if torch.is_tensor(T):
            return T[0] if T.dim() == 3 else T

    pose_wrapper = getattr(scorenet_model, "pose_wrapper", None)
    if pose_wrapper is not None and hasattr(pose_wrapper, "_predict_single_pose"):
        _, _, T = pose_wrapper._predict_single_pose(
            target.contiguous(),
            ref_img.contiguous(),
            int(fid),
        )
        if torch.is_tensor(T):
            return T[0] if T.dim() == 3 else T

    return None


@torch.no_grad()
def build_weighted_keyframe_edges(curr_keyframe, ref_pool, scorenet_model, device, scorenet_cfg):
    """
    Build and store weighted KFG long edges when a keyframe is created.

    Records are intentionally small and detached so later pose-graph refinement can
    reuse the exact selected refs, scores, and pose edges that existed at insertion time.
    """
    if scorenet_model is None:
        return {"edges": [], "reason": "no_scorenet"}
    if len(ref_pool) == 0:
        return {"edges": [], "reason": "no_refs"}

    score_temp = float(scorenet_cfg.get("sampling_temperature", 1.0))
    max_long_edges = int(scorenet_cfg.get("pose_graph_max_long_edges", 8))
    use_selected_only = bool(scorenet_cfg.get("pose_graph_use_selected_only", True))
    use_all_candidates = bool(scorenet_cfg.get("pose_graph_use_all_candidates", False))
    score_from_scores = bool(scorenet_cfg.get("pose_graph_score_from_scores", False))
    min_long_score = float(scorenet_cfg.get("pose_graph_min_long_score", 0.03))
    predict_missing = bool(scorenet_cfg.get("pose_graph_predict_missing_edges", False))

    ref_ids = [int(kf["id"]) for kf in ref_pool]
    ref_imgs = [kf["color"].unsqueeze(0).to(device) for kf in ref_pool]
    target = curr_keyframe["color"].unsqueeze(0).to(device)
    outputs = score_candidates(scorenet_model, target, ref_imgs, ref_ids)

    score_map = _score_map_from_outputs(outputs, ref_ids, score_temp, score_from_scores)
    selected_fids = _selected_fids_from_outputs(
        outputs=outputs,
        ref_ids=ref_ids,
        score_map=score_map,
        use_all_candidates=use_all_candidates,
        use_selected_only=use_selected_only,
        max_edges=max_long_edges,
    )
    selected_set = set(selected_fids)
    fid_to_refidx = {int(fid): i for i, fid in enumerate(ref_ids)}

    records = []
    for fid in selected_fids:
        score = float(score_map.get(int(fid), 0.0))
        if score < min_long_score:
            continue
        key = ("cam_T_cam", 0, int(fid))
        if key in outputs:
            T_ref_cur = outputs[key][0].to(device).detach()
        elif predict_missing:
            ref_idx = fid_to_refidx.get(int(fid), None)
            if ref_idx is None:
                continue
            T_ref_cur = _predict_pose_edge_on_demand(
                scorenet_model=scorenet_model,
                target=target,
                ref_img=ref_imgs[ref_idx],
                fid=int(fid),
            )
            if T_ref_cur is None:
                continue
            T_ref_cur = T_ref_cur.to(device).detach()
        else:
            continue
        if not torch.isfinite(T_ref_cur).all():
            continue
        records.append(
            {
                "ref_id": int(fid),
                "target_id": int(curr_keyframe["id"]),
                "T_ref_cur": T_ref_cur.detach().cpu(),
                "score": score,
                "selected": int(fid) in selected_set,
                "n_ref": len(ref_ids),
            }
        )

    return {
        "edges": records,
        "selected": selected_fids,
        "scores": score_map,
        "reason": "ok" if records else "no_valid_edges",
    }


def optimize_pose_graph_local(params, keyframe_list, scorenet_model, device, scorenet_cfg):
    """
    Local pose-graph optimization on recent keyframes.

    Edges:
      - consecutive odometry edges from current estimates
      - ScoreNet long-range edges from recent keyframe to older keyframes
    """
    if len(keyframe_list) < 3:
        return {"optimized": False, "reason": "too_few_keyframes"}

    window = int(scorenet_cfg.get("pose_graph_window", 20))
    iters = int(scorenet_cfg.get("pose_graph_iters", 20))
    lr = float(scorenet_cfg.get("pose_graph_lr", 5e-3))
    odom_w_t = float(scorenet_cfg.get("pose_graph_odom_w_t", 5.0))
    odom_w_r = float(scorenet_cfg.get("pose_graph_odom_w_r", 5.0))
    long_w_t = float(scorenet_cfg.get("pose_graph_long_w_t", 1.0))
    long_w_r = float(scorenet_cfg.get("pose_graph_long_w_r", 1.0))
    anchor_w_t = float(scorenet_cfg.get("pose_graph_anchor_w_t", 20.0))
    anchor_w_r = float(scorenet_cfg.get("pose_graph_anchor_w_r", 20.0))
    score_temp = float(scorenet_cfg.get("sampling_temperature", 1.0))
    candidate_window = int(scorenet_cfg.get("candidate_window", 20))
    min_long_score = float(scorenet_cfg.get("pose_graph_min_long_score", 0.03))
    max_long_edges = int(scorenet_cfg.get("pose_graph_max_long_edges", 8))
    max_long_weight = float(scorenet_cfg.get("pose_graph_max_long_weight", 0.2))
    consistency_t = float(scorenet_cfg.get("pose_graph_consistency_trans_thresh", 0.15))
    consistency_r = float(scorenet_cfg.get("pose_graph_consistency_rot_thresh", 0.35))
    huber_t = float(scorenet_cfg.get("pose_graph_huber_trans", 0.05))
    huber_r = float(scorenet_cfg.get("pose_graph_huber_rot", 0.10))
    grad_clip = float(scorenet_cfg.get("pose_graph_grad_clip", 1.0))
    score_w_gamma = float(scorenet_cfg.get("pose_graph_score_weight_gamma", 1.0))
    score_w_min = float(scorenet_cfg.get("pose_graph_score_weight_min", 1.0))
    score_w_max = float(scorenet_cfg.get("pose_graph_score_weight_max", 1.0))
    use_score_weight = bool(scorenet_cfg.get("pose_graph_use_score_weight", True))
    use_selected_only = bool(scorenet_cfg.get("pose_graph_use_selected_only", True))
    equal_long_edge_weight = bool(scorenet_cfg.get("pose_graph_equal_long_edge_weight", False))
    use_all_candidates = bool(scorenet_cfg.get("pose_graph_use_all_candidates", False))
    score_from_scores = bool(scorenet_cfg.get("pose_graph_score_from_scores", False))
    use_stored_edges = bool(scorenet_cfg.get("pose_graph_use_stored_edges", False))
    refresh_latest_edges = bool(scorenet_cfg.get("pose_graph_refresh_latest_edges", not use_stored_edges))

    start = max(0, len(keyframe_list) - window)
    local_kfs = keyframe_list[start:]
    n = len(local_kfs)
    if n < 3:
        return {"optimized": False, "reason": "too_few_local_keyframes"}

    w2c_init = torch.stack([kf["est_w2c"].to(device) for kf in local_kfs], dim=0)
    q0 = matrix_to_quaternion(w2c_init[:, :3, :3]).detach()
    t0 = w2c_init[:, :3, 3].detach()

    q_param = torch.nn.Parameter(q0.clone())
    t_param = torch.nn.Parameter(t0.clone())
    optim = torch.optim.Adam([q_param, t_param], lr=lr)

    frameid_to_local = {kf["id"]: i for i, kf in enumerate(local_kfs)}

    edges = []
    for i in range(n - 1):
        T_ij = _rel_from_w2c(w2c_init[i], w2c_init[i + 1]).detach()
        if torch.isfinite(T_ij).all():
            edges.append((i, i + 1, T_ij, odom_w_t, odom_w_r))

    stored_long_edges = 0
    with torch.no_grad():
        if use_stored_edges:
            for j, kf in enumerate(local_kfs):
                per_target = 0
                for record in kf.get("kfg_pose_edges", []):
                    if per_target >= max_long_edges:
                        break
                    fid = int(record.get("ref_id", -1))
                    if fid not in frameid_to_local:
                        continue
                    i = frameid_to_local[fid]
                    if i == j:
                        continue
                    if use_selected_only and (not bool(record.get("selected", True))) and (not use_all_candidates):
                        continue
                    score = float(record.get("score", 0.0))
                    if use_score_weight and (score < min_long_score):
                        continue
                    T_ij = record.get("T_ref_cur", None)
                    if T_ij is None:
                        continue
                    T_ij = T_ij.to(device).detach()
                    if not torch.isfinite(T_ij).all():
                        continue

                    T_ij_est = _rel_from_w2c(w2c_init[i], w2c_init[j]).detach()
                    T_delta = _se3_inverse(T_ij_est) @ T_ij
                    if not torch.isfinite(T_delta).all():
                        continue
                    t_delta = torch.norm(T_delta[:3, 3], p=2).item()
                    r_delta = torch.abs(_rot_angle_from_R(T_delta[:3, :3])).item()
                    if t_delta > consistency_t or r_delta > consistency_r:
                        continue

                    if equal_long_edge_weight or (not use_score_weight):
                        edge_scale = 1.0
                    else:
                        w = min(max(score, min_long_score), max_long_weight)
                        info_scale = _score_to_info_scale(
                            score=score,
                            n_ref=int(record.get("n_ref", candidate_window)),
                            gamma=score_w_gamma,
                            w_min=score_w_min,
                            w_max=score_w_max,
                        )
                        edge_scale = w * info_scale

                    edges.append((i, j, T_ij, long_w_t * edge_scale, long_w_r * edge_scale))
                    stored_long_edges += 1
                    per_target += 1

        if refresh_latest_edges:
            curr_kf = local_kfs[-1]
            ref_pool = local_kfs[:-1][-candidate_window:]
        else:
            ref_pool = []

        if len(ref_pool) > 0:
            ref_ids = [kf["id"] for kf in ref_pool]
            ref_imgs = [kf["color"].unsqueeze(0).to(device) for kf in ref_pool]
            fid_to_refidx = {int(fid): i for i, fid in enumerate(ref_ids)}
            target = curr_kf["color"].unsqueeze(0).to(device)
            outputs = score_candidates(scorenet_model, target, ref_imgs, ref_ids)

            score_map = _score_map_from_outputs(outputs, ref_ids, score_temp, score_from_scores)
            selected_fids = _selected_fids_from_outputs(
                outputs=outputs,
                ref_ids=ref_ids,
                score_map=score_map,
                use_all_candidates=use_all_candidates,
                use_selected_only=use_selected_only,
                max_edges=max_long_edges,
            )

            for fid in selected_fids:
                key = ("cam_T_cam", 0, fid)
                if fid not in frameid_to_local:
                    continue
                i = frameid_to_local[fid]
                j = n - 1
                if key in outputs:
                    T_ij = outputs[key][0].to(device).detach()
                elif use_all_candidates:
                    ref_idx = fid_to_refidx.get(int(fid), None)
                    if ref_idx is None:
                        continue
                    T_ij = _predict_pose_edge_on_demand(
                        scorenet_model=scorenet_model,
                        target=target,
                        ref_img=ref_imgs[ref_idx],
                        fid=int(fid),
                    )
                    if T_ij is None:
                        continue
                    T_ij = T_ij.to(device).detach()
                else:
                    continue
                score = score_map.get(fid, 0.0)
                if use_score_weight and (score < min_long_score):
                    continue
                if not torch.isfinite(T_ij).all():
                    continue

                # Reject inconsistent long edges using current odometry as reference.
                T_ij_est = _rel_from_w2c(w2c_init[i], w2c_init[j]).detach()
                T_delta = _se3_inverse(T_ij_est) @ T_ij
                if not torch.isfinite(T_delta).all():
                    continue
                t_delta = torch.norm(T_delta[:3, 3], p=2).item()
                r_delta = torch.abs(_rot_angle_from_R(T_delta[:3, :3])).item()
                if t_delta > consistency_t or r_delta > consistency_r:
                    continue

                if equal_long_edge_weight or (not use_score_weight):
                    edge_scale = 1.0
                else:
                    w = min(max(score, min_long_score), max_long_weight)
                    info_scale = _score_to_info_scale(
                        score=score,
                        n_ref=len(ref_ids),
                        gamma=score_w_gamma,
                        w_min=score_w_min,
                        w_max=score_w_max,
                    )
                    edge_scale = w * info_scale

                edges.append((i, j, T_ij, long_w_t * edge_scale, long_w_r * edge_scale))

    if len(edges) == 0:
        return {"optimized": False, "reason": "no_edges", "edges_total": 0}

    last_loss = None
    valid_for_opt = False
    with torch.enable_grad():
        for _ in range(iters):
            optim.zero_grad(set_to_none=True)
            q = F.normalize(q_param, dim=1)
            R = build_rotation(q)
            w2c = torch.eye(4, device=device, dtype=t_param.dtype).unsqueeze(0).repeat(n, 1, 1)
            w2c[:, :3, :3] = R
            w2c[:, :3, 3] = t_param

            loss = torch.zeros((), device=device, dtype=t_param.dtype)
            n_valid_edges = 0
            n_nonfinite_Terr = 0
            n_nonfinite_r = 0
            for i, j, T_ij, wt_t, wt_r in edges:
                T_pred = _rel_from_w2c(w2c[i], w2c[j])
                T_err = _se3_inverse(T_ij) @ T_pred
                if not torch.isfinite(T_err).all():
                    n_nonfinite_Terr += 1
                    continue
                t_err = T_err[:3, 3]
                r_res = _rot_loss_residual_from_R(T_err[:3, :3])
                if not torch.isfinite(r_res):
                    n_nonfinite_r += 1
                    continue
                t_norm = torch.norm(t_err, p=2)
                loss = loss + wt_t * _huber_scalar(t_norm, huber_t) + wt_r * _huber_scalar(r_res, huber_r)
                n_valid_edges += 1

            if n_valid_edges == 0:
                return {
                    "optimized": False,
                    "reason": "no_finite_edges",
                    "edges_total": len(edges),
                    "nonfinite_T_err": n_nonfinite_Terr,
                    "nonfinite_rot_err": n_nonfinite_r,
                }
            valid_for_opt = True

            t_anchor = t_param[0] - t0[0]
            q_anchor = (q[0] * q0[0]).sum().abs().clamp(1e-6, 1.0)
            # Stable anchor residual; avoid acos singularity near 0-rotation.
            r_anchor = 1.0 - q_anchor
            loss = loss + anchor_w_t * (t_anchor * t_anchor).sum() + anchor_w_r * (r_anchor * r_anchor)
            if not torch.isfinite(loss):
                return {"optimized": False, "reason": "non_finite_loss"}

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([q_param, t_param], max_norm=grad_clip)
            optim.step()
            last_loss = float(loss.detach().cpu())

    if (not valid_for_opt) or (last_loss is None) or (not torch.isfinite(torch.tensor(last_loss))):
        return {"optimized": False, "reason": "invalid_optimization"}

    with torch.no_grad():
        q = F.normalize(q_param, dim=1)
        R = build_rotation(q)
        w2c_opt = torch.eye(4, device=device, dtype=t_param.dtype).unsqueeze(0).repeat(n, 1, 1)
        w2c_opt[:, :3, :3] = R
        w2c_opt[:, :3, 3] = t_param
        if not torch.isfinite(w2c_opt).all():
            return {"optimized": False, "reason": "non_finite_solution"}

        for local_idx, kf in enumerate(local_kfs):
            global_idx = start + local_idx
            tid = int(kf["id"])
            keyframe_list[global_idx]["est_w2c"] = w2c_opt[local_idx].detach()
            params["cam_unnorm_rots"][..., tid] = q[local_idx].view(1, 4).detach()
            params["cam_trans"][..., tid] = t_param[local_idx].view(1, 3).detach()

    return {
        "optimized": True,
        "nodes": n,
        "edges": len(edges),
        "stored_long_edges": stored_long_edges,
        "loss": last_loss if last_loss is not None else -1.0,
    }
