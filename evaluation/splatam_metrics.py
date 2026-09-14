from pathlib import Path
import sys,math,numpy as np
SPLATAM=Path(__file__).resolve().parents[1]/"final/splatam"

def sample_metric(path: Path, count: int | None = None) -> tuple[float, int]:
    if not path.is_file():
        return math.nan, 0
    vals = np.asarray([float(x) for x in path.read_text(errors="replace").split()], dtype=float)
    vals = vals[np.isfinite(vals)]
    if count is not None and len(vals) > count:
        vals = vals[np.linspace(0, len(vals) - 1, count).round().astype(int)]
    return (float(vals.mean()), len(vals)) if len(vals) else (math.nan, 0)

def selected_frame_ids(run: Path, maximum: int | None = None) -> list[int]:
    ids = []
    for p in sorted((run / "eval/plots").glob("*.png")):
        try:
            ids.append(int(p.stem))
        except ValueError:
            pass
    if maximum is not None and len(ids) > maximum:
        ids = [ids[int(i)] for i in np.linspace(0, len(ids) - 1, maximum).round().astype(int)]
    return ids

def true_depth_rmse(run: Path, config: dict, dataset_name: str, scene: str, frame_ids: list[int], device: str) -> tuple[float, int]:
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(SPLATAM))
    from datasets.gradslam_datasets import C3VDDataset, load_dataset_config
    from diff_gaussian_rasterization import GaussianRasterizer as Renderer
    from utils.recon_helpers import setup_camera
    from utils.slam_helpers import transform_to_frame, transformed_params2depthplussilhouette

    data = config["data"]
    basedir = data["basedir"]
    cfg_path = Path(data["gradslam_data_cfg"])
    if not cfg_path.is_file():
        # Saved queue configs may contain an absolute path derived from the
        # copied config's output location rather than the source repository.
        # Fall back to the authoritative, dataset-specific intrinsics file.
        cfg_path = SPLATAM / "configs/data" / (
            "c3vd_v2.yaml" if dataset_name == "c3vd_v2" else "c3vd.yaml"
        )
    ds_cfg = load_dataset_config(str(cfg_path))
    dataset = C3VDDataset(
        ds_cfg, basedir, scene,
        start=data["start"], end=data["end"], stride=data["stride"],
        desired_height=data["desired_image_height"], desired_width=data["desired_image_width"],
        device=torch.device(device), relative_pose=True,
    )
    raw = np.load(run / "params.npz", allow_pickle=True)
    params = {k: torch.tensor(raw[k], device=device).float() for k in raw.files if np.issubdtype(raw[k].dtype, np.number)}
    color0, _depth0, intr0, pose0 = dataset[0]
    first_w2c = torch.linalg.inv(pose0)
    cam = setup_camera(color0.shape[1], color0.shape[0], intr0[:3, :3].cpu().numpy(), first_w2c.cpu().numpy())
    vals = []
    with torch.no_grad():
        for idx in frame_ids:
            _color, gt_depth, _intr, _pose = dataset[idx]
            pts = transform_to_frame(params, idx, gaussians_grad=False, camera_grad=False)
            rv = transformed_params2depthplussilhouette(params, first_w2c, pts)
            rendered, _, _ = Renderer(raster_settings=cam)(**rv)
            pred = rendered[0]
            gt = gt_depth[..., 0] if gt_depth.ndim == 3 else gt_depth
            valid = torch.isfinite(gt) & (gt > 0)
            if valid.any():
                vals.append(float(torch.sqrt(torch.mean((pred[valid] - gt[valid]) ** 2)).cpu()))
    return (float(np.mean(vals)), len(vals)) if vals else (math.nan, 0)