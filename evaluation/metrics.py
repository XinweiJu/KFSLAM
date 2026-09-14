from pathlib import Path
import re, math
import cv2, numpy as np, torch
from scipy.spatial.transform import Rotation
from skimage.metrics import structural_similarity
C3VD_ROOT=Path("data/c3vd")
C3VD_V2_ROOT=Path("data/c3vd_v2")

def read_w2c_txt(path: Path) -> np.ndarray:
    mats = []
    for line in path.read_text(encoding="utf-8").splitlines():
        vals = [float(x) for x in re.split(r"[,\s]+", line.strip()) if x]
        if vals:
            mats.append(np.asarray(vals[:16], dtype=np.float64).reshape(4, 4))
    return np.stack(mats, axis=0)

def w2c_from_splatam_params(params_path: Path) -> tuple[np.ndarray, np.ndarray]:
    archive = np.load(params_path, allow_pickle=True)
    quat = np.asarray(archive["cam_unnorm_rots"])[0].T.astype(np.float64)
    quat /= np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-12)
    rotations = Rotation.from_quat(quat[:, [1, 2, 3, 0]]).as_matrix()
    translations = np.asarray(archive["cam_trans"])[0].T.astype(np.float64)
    est = np.repeat(np.eye(4)[None], len(rotations), axis=0)
    est[:, :3, :3] = rotations
    est[:, :3, 3] = translations
    gt = np.asarray(archive["gt_w2c_all_frames"], dtype=np.float64)
    return est, gt

def centers_from_w2c(w2c: np.ndarray) -> np.ndarray:
    return -np.einsum("nij,nj->ni", w2c[:, :3, :3].transpose(0, 2, 1), w2c[:, :3, 3])

def c2w_from_w2c(w2c: np.ndarray) -> np.ndarray:
    out = np.repeat(np.eye(4)[None], len(w2c), axis=0)
    out[:, :3, :3] = w2c[:, :3, :3].transpose(0, 2, 1)
    out[:, :3, 3] = centers_from_w2c(w2c)
    return out

def align_umeyama(
    model_centers: np.ndarray, gt_centers: np.ndarray, with_scale: bool
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    n = min(len(model_centers), len(gt_centers))
    model = model_centers[:n]
    gt = gt_centers[:n]
    mu_m = model.mean(axis=0, keepdims=True)
    mu_g = gt.mean(axis=0, keepdims=True)
    mc = model - mu_m
    gc = gt - mu_g
    var_m = float((mc * mc).sum() / max(n, 1))
    cov = (gc.T @ mc) / max(n, 1)
    u, svals, vt = np.linalg.svd(cov)
    det = np.linalg.det(u) * np.linalg.det(vt)
    diag = np.diag([1.0, 1.0, det])
    rot = u @ diag @ vt
    scale = float((svals * np.array([1.0, 1.0, det])).sum() / max(var_m, 1e-14)) if with_scale else 1.0
    trans = mu_g.reshape(3) - scale * (rot @ mu_m.reshape(3))
    aligned = (scale * (rot @ model.T)).T + trans.reshape(1, 3)
    return aligned, scale, rot, trans

def rotation_angle_deg(rot: np.ndarray) -> float:
    cos = float(np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cos))

def pose_metrics(est_w2c: np.ndarray, gt_w2c: np.ndarray) -> dict[str, float | int]:
    n = min(len(est_w2c), len(gt_w2c))
    est_w2c = est_w2c[:n]
    gt_w2c = gt_w2c[:n]
    est_c2w = c2w_from_w2c(est_w2c)
    gt_c2w = c2w_from_w2c(gt_w2c)
    est_centers = est_c2w[:, :3, 3]
    gt_centers = gt_c2w[:, :3, 3]
    se3_centers, se3_scale, se3_rot, _se3_trans = align_umeyama(est_centers, gt_centers, with_scale=False)
    aligned_centers, scale, align_rot, _align_trans = align_umeyama(est_centers, gt_centers, with_scale=True)

    aligned_c2w = np.repeat(np.eye(4)[None], n, axis=0)
    aligned_c2w[:, :3, :3] = align_rot[None] @ est_c2w[:, :3, :3]
    aligned_c2w[:, :3, 3] = aligned_centers

    ate = np.linalg.norm(aligned_centers - gt_centers, axis=1)
    se3_ate = np.linalg.norm(se3_centers - gt_centers, axis=1)
    are = np.asarray(
        [
            rotation_angle_deg(gt_c2w[i, :3, :3].T @ aligned_c2w[i, :3, :3])
            for i in range(n)
        ],
        dtype=np.float64,
    )

    rpet, rper = [], []
    for i in range(n - 1):
        est_rel = np.linalg.inv(aligned_c2w[i]) @ aligned_c2w[i + 1]
        gt_rel = np.linalg.inv(gt_c2w[i]) @ gt_c2w[i + 1]
        err = np.linalg.inv(gt_rel) @ est_rel
        rpet.append(float(np.linalg.norm(err[:3, 3])))
        rper.append(rotation_angle_deg(err[:3, :3]))
    rpet_arr = np.asarray(rpet, dtype=np.float64)
    rper_arr = np.asarray(rper, dtype=np.float64)
    return {
        "frames": n,
        "rpe_pairs": max(n - 1, 0),
        "se3_ate_mean": float(se3_ate.mean()),
        "se3_ate_rmse": float(np.sqrt(np.mean(se3_ate * se3_ate))),
        "se3_ate_median": float(np.median(se3_ate)),
        "se3_scale": se3_scale,
        "sim3_ate_mean": float(ate.mean()),
        "sim3_ate_rmse": float(np.sqrt(np.mean(ate * ate))),
        "sim3_ate_median": float(np.median(ate)),
        "sim3_scale": scale,
        "are_deg_mean": float(are.mean()),
        "are_deg_rmse": float(np.sqrt(np.mean(are * are))),
        "rpet_mean": float(rpet_arr.mean()) if rpet_arr.size else math.nan,
        "rpet_rmse": float(np.sqrt(np.mean(rpet_arr * rpet_arr))) if rpet_arr.size else math.nan,
        "rper_deg_mean": float(rper_arr.mean()) if rper_arr.size else math.nan,
        "rper_deg_rmse": float(np.sqrt(np.mean(rper_arr * rper_arr))) if rper_arr.size else math.nan,
    }

def mean_txt(path: Path) -> float:
    if not path.is_file():
        return math.nan
    vals = []
    for token in path.read_text(errors="replace").split():
        try:
            val = float(token)
        except ValueError:
            continue
        if math.isfinite(val):
            vals.append(val)
    return float(np.mean(vals)) if vals else math.nan

def gt_rgb_path(dataset: str, scene: str, frame_id: int) -> Path:
    if dataset == "c3vd":
        return C3VD_ROOT / scene / "color" / f"{frame_id:04d}.png"
    return C3VD_V2_ROOT / scene / "rgb" / f"{frame_id:04d}.png"

def gt_depth_path(dataset: str, scene: str, frame_id: int) -> Path:
    if dataset == "c3vd":
        return C3VD_ROOT / scene / "depth" / f"{frame_id:04d}.tiff"
    return C3VD_V2_ROOT / scene / "depth" / f"{frame_id:04d}_depth.tiff"

def read_rgb(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

def read_gt_depth_mm(path: Path) -> np.ndarray | None:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    depth = depth.astype(np.float32)
    if depth.max() > 1000.0:
        return depth / 65535.0 * 100.0
    return depth

def read_render_depth_mm(path: Path) -> np.ndarray | None:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    return depth.astype(np.float32) / 655.35

def sampled_paths(paths: list[Path], max_count: int | None) -> list[Path]:
    if max_count is None or len(paths) <= max_count:
        return paths
    ids = np.linspace(0, len(paths) - 1, max_count).round().astype(int)
    return [paths[int(i)] for i in ids]

def endo_render_metrics(run_dir: Path, dataset: str, scene: str, lpips_model, device: str, max_frames: int | None) -> dict[str, float | int]:
    color_paths = sampled_paths(sorted((run_dir / "eval" / "color").glob("color_*.png")), max_frames)
    rgb_rmse, psnr, ssim, lpips_vals, depth_rmse = [], [], [], [], []
    for pred_path in color_paths:
        try:
            frame_id = int(pred_path.stem.split("_")[-1])
        except ValueError:
            continue
        pred = read_rgb(pred_path)
        gt = read_rgb(gt_rgb_path(dataset, scene, frame_id))
        if pred is None or gt is None:
            continue
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        diff = pred - gt
        rmse = float(np.sqrt(np.mean(diff * diff)))
        rgb_rmse.append(rmse)
        psnr.append(float(20.0 * math.log10(1.0 / max(rmse, 1e-8))))
        ssim.append(float(structural_similarity(gt, pred, channel_axis=2, data_range=1.0)))
        if lpips_model is not None:
            with torch.no_grad():
                pred_t = torch.from_numpy(pred).permute(2, 0, 1)[None].to(device) * 2.0 - 1.0
                gt_t = torch.from_numpy(gt).permute(2, 0, 1)[None].to(device) * 2.0 - 1.0
                lpips_vals.append(float(lpips_model(pred_t, gt_t).detach().cpu().item()))
        pred_depth = read_render_depth_mm(run_dir / "eval" / "depth" / f"depth_{frame_id:04d}.tiff")
        gt_depth = read_gt_depth_mm(gt_depth_path(dataset, scene, frame_id))
        if pred_depth is not None and gt_depth is not None:
            if pred_depth.shape != gt_depth.shape:
                pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]), interpolation=cv2.INTER_LINEAR)
            valid = np.isfinite(gt_depth) & (gt_depth > 0)
            if valid.any():
                depth_rmse.append(float(np.sqrt(np.mean((pred_depth[valid] - gt_depth[valid]) ** 2))))
    return {
        "render_frames": len(rgb_rmse),
        "rgb_rmse": float(np.mean(rgb_rmse)) if rgb_rmse else math.nan,
        "psnr": float(np.mean(psnr)) if psnr else math.nan,
        "ssim": float(np.mean(ssim)) if ssim else math.nan,
        "lpips": float(np.mean(lpips_vals)) if lpips_vals else math.nan,
        "depth_rmse": float(np.mean(depth_rmse)) if depth_rmse else math.nan,
    }