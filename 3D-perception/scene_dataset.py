import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

class SceneCaptionDataset(Dataset):
    def __init__(self, root_dir, num_points=40000, test=False):
        self.root_dir = root_dir
        self.num_points = num_points
        self.scene_dirs = [
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ]
        self.test = test

    def __len__(self):
        return len(self.scene_dirs)

    def __getitem__(self, idx):
        scene_path = self.scene_dirs[idx]
        pc_path = os.path.join(scene_path, "world_points.txt")
        caption_path = os.path.join(scene_path, "dataset.json")

        # read point cloud data
        points = np.loadtxt(pc_path, skiprows=1)  # skip "x y z"
        if points.shape[0] > self.num_points:
            choice = np.random.choice(points.shape[0], self.num_points, replace=False)
            points = points[choice]
        points = torch.tensor(points, dtype=torch.float32)
        
        if self.test == False:
            # read caption data
            with open(caption_path, "r") as f:
                data = json.load(f)
            caption = data["captions"][0]  # 或自訂取法
        else:
            caption = None
            with open(caption_path, "r") as f:
                data = json.load(f)
            caption = data["captions"][0]  # 或自訂取法
            
        pc_min = points.min(dim=0).values  
        pc_max = points.max(dim=0).values  

        return {
            "point_clouds": points,              # (N, 3)
            "point_cloud_dims_min": pc_min,      # (3,)
            "point_cloud_dims_max": pc_max,      # (3,)
            "caption": caption                   # str
        }
