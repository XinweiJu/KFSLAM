import glob
import os
from typing import Optional

import torch
from natsort import natsorted

from .basedataset import GradSLAMDataset


class C3VDDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 540,
        desired_width: Optional[int] = 675,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder, "pose.txt")
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/color/*.png"))
        if not color_paths:
            color_paths = natsorted(glob.glob(f"{self.input_folder}/rgb/*.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.tiff"))
        if not depth_paths:
            depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.tif"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        poses = []
        with open(self.pose_path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
        for i in range(self.num_imgs):
            pose = list(map(float, lines[i].split(sep=",")))
            poses.append(torch.Tensor(pose).reshape(4, 4).float().transpose(0, 1))
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)
