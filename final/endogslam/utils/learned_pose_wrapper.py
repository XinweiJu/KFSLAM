"""
Learned relative-pose estimators for use with DiversityWrapper.

Supports MonoDepth2 and AF-SfMLearner pose networks pretrained on HK / C3VD.
Both share the same architecture (ResnetEncoder + PoseDecoder) and the same
transformation_from_parameters convention.

Usage:
    net = MonoDepth2PoseNet(weights_dir="/path/to/monodepth/hk/weights_19")
    net = AFSfMPoseNet(weights_dir="/path/to/af-sfm/hk/weights_19")
    T_ref_to_cur = net.estimate_pose(target_rgb, ref_rgb)   # (4,4) on same device
"""
import importlib.util
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── image normalization (matches monodepth2 / AF-SfM training) ───────────────
# Input images are expected in [0, 1] float range (from EndoGSLAM's color tensor).
_MEAN = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1)

# Resize target — both models are fully convolutional so any size works;
# use an endoscopy-friendly resolution that's close to HK training size.
_POSE_H, _POSE_W = 192, 256


def _normalize(x: torch.Tensor) -> torch.Tensor:
    mean = _MEAN.to(x.device, x.dtype)
    std  = _STD.to(x.device, x.dtype)
    return (x - mean) / std


def _resize(x: torch.Tensor, h: int = _POSE_H, w: int = _POSE_W) -> torch.Tensor:
    if x.shape[-2] == h and x.shape[-1] == w:
        return x
    return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)


def _load_module_from_file(name: str, filepath: str):
    """Load a Python module from an absolute file path, bypassing sys.modules cache."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_repo_modules(repo_dir: str):
    """
    Load resnet_encoder, pose_decoder, and layers from a monodepth2-style repo
    without touching sys.path or the global 'networks' package — so it doesn't
    conflict with KeyFrameGraph's own 'networks' package.
    """
    r = Path(repo_dir)
    layers         = _load_module_from_file(f"_lpw_layers_{r.name}",  str(r / "layers.py"))
    resnet_encoder = _load_module_from_file(f"_lpw_renc_{r.name}",    str(r / "networks" / "resnet_encoder.py"))
    pose_decoder   = _load_module_from_file(f"_lpw_pdec_{r.name}",    str(r / "networks" / "pose_decoder.py"))
    return layers, resnet_encoder, pose_decoder


class _PoseNetBase(nn.Module):
    """
    Base class. Subclasses load encoder + decoder and expose:
        estimate_pose(target: Tensor (B,3,H,W), ref: Tensor (B,3,H,W)) -> Tensor (B,4,4)
    where the returned matrix transforms points FROM ref INTO target coordinates
    (i.e. T_ref→target, or equivalently the w2c change needed to go from ref to target).
    """

    def __init__(self):
        super().__init__()
        self.pose_encoder: nn.Module = None
        self.pose_decoder: nn.Module = None

    @torch.no_grad()
    def estimate_pose(self, target: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        target, ref: (B,3,H,W) float in [0,1], same device.
        Returns: (B,4,4) SE3 matrix T such that P_target ≈ T @ P_ref
        """
        t = _normalize(_resize(target))
        r = _normalize(_resize(ref))
        # Convention: concat [ref, target] → T_ref→target
        pair = torch.cat([r, t], dim=1)           # (B, 6, H, W)
        features = [self.pose_encoder(pair)]
        axisangle, translation = self.pose_decoder(features)
        # axisangle: (B, 1, 1, 3), translation: (B, 1, 1, 3)
        T = self._transform(axisangle[:, 0], translation[:, 0])   # (B,4,4)
        return T

    def _transform(self, axisangle: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MonoDepth2PoseNet(_PoseNetBase):
    """
    MonoDepth2 pose network.
    weights_dir should contain: pose_encoder.pth, pose.pth
    """

    REPO = "third_party/optional_pose_model"

    def __init__(self, weights_dir: str, num_layers: int = 18, device: str = "cuda"):
        super().__init__()
        layers, resnet_encoder_mod, pose_decoder_mod = _load_repo_modules(self.REPO)
        self._tfp = layers.transformation_from_parameters

        wd = Path(weights_dir)
        pose_enc_state = torch.load(str(wd / "pose_encoder.pth"), map_location="cpu", weights_only=False)
        pose_dec_state = torch.load(str(wd / "pose.pth"),         map_location="cpu", weights_only=False)

        self.pose_encoder = resnet_encoder_mod.ResnetEncoder(num_layers, False, num_input_images=2)
        self.pose_decoder = pose_decoder_mod.PoseDecoder(self.pose_encoder.num_ch_enc, 1, 2)
        self.pose_encoder.load_state_dict(pose_enc_state)
        self.pose_decoder.load_state_dict(pose_dec_state)

        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        self.pose_encoder.to(dev).eval()
        self.pose_decoder.to(dev).eval()
        self._device = dev

    def _transform(self, axisangle, translation):
        return self._tfp(axisangle, translation, invert=False)


class AFSfMPoseNet(_PoseNetBase):
    """
    AF-SfMLearner pose network.
    weights_dir should contain: pose_encoder.pth, pose.pth
    (The extra transform/position networks are only needed for depth; not used here.)
    """

    REPO = "third_party/optional_pose_model"

    def __init__(self, weights_dir: str, num_layers: int = 18, device: str = "cuda"):
        super().__init__()
        layers, resnet_encoder_mod, pose_decoder_mod = _load_repo_modules(self.REPO)
        self._tfp = layers.transformation_from_parameters

        wd = Path(weights_dir)
        pose_enc_state = torch.load(str(wd / "pose_encoder.pth"), map_location="cpu", weights_only=False)
        pose_dec_state = torch.load(str(wd / "pose.pth"),         map_location="cpu", weights_only=False)

        self.pose_encoder = resnet_encoder_mod.ResnetEncoder(num_layers, False, num_input_images=2)
        self.pose_decoder = pose_decoder_mod.PoseDecoder(self.pose_encoder.num_ch_enc, 1, 2)
        self.pose_encoder.load_state_dict(pose_enc_state)
        self.pose_decoder.load_state_dict(pose_dec_state)

        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        self.pose_encoder.to(dev).eval()
        self.pose_decoder.to(dev).eval()
        self._device = dev

    def _transform(self, axisangle, translation):
        return self._tfp(axisangle, translation, invert=False)


def load_pose_net(model: str, weights_dir: str, device: str = "cuda") -> _PoseNetBase:
    """
    model: "monodepth2" | "af-sfm"
    """
    m = model.lower().replace("_", "-")
    if m in ("monodepth2", "monodepth"):
        return MonoDepth2PoseNet(weights_dir, device=device)
    if m in ("af-sfm", "afsfm", "af_sfm"):
        return AFSfMPoseNet(weights_dir, device=device)
    raise ValueError(f"Unknown pose model: {model!r}")
