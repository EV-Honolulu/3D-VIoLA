import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

class TraningDataset(Dataset):
    def __init__(self, root_dirs, tokenizer, feature_extractor, num_points=40000, max_length=128):
        # 支援單個或多個 root_dir
        if isinstance(root_dirs, str):
            self.root_dirs = [root_dirs]

        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor.eval()
        self.num_points = num_points
        self.max_length = max_length
        # self.scene_dirs = [
        #     os.path.join(root_dir, d)
        #     for d in os.listdir(root_dir)
        #     if os.path.isdir(os.path.join(root_dir, d))
        # ]
        self.scene_dirs = []
        for root_dir in root_dirs:
            dirs = [
                os.path.join(root_dir, d)
                for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ]
            self.scene_dirs.extend(dirs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.scene_dirs)

    def __getitem__(self, idx):
        scene_path = self.scene_dirs[idx]
        pc_path = os.path.join(scene_path, "world_points.txt")
        caption_path = os.path.join(scene_path, "dataset.json")

        points = np.loadtxt(pc_path, skiprows=1)
        if points.shape[0] > self.num_points:
            choice = np.random.choice(points.shape[0], self.num_points, replace=False)
            points = points[choice]
        points = torch.tensor(points, dtype=torch.float32) 
        pc_min = points.min(dim=0).values  
        pc_max = points.max(dim=0).values  

        point_inputs =  {
            "point_clouds": points.unsqueeze(0),       # [1, N, 3] where N is the number of points
            "point_cloud_dims_min": pc_min.unsqueeze(0),           # [1, 3]
            "point_cloud_dims_max": pc_max.unsqueeze(0),            
        }
        # Extract 3D feature (encoder + decoder)
        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in point_inputs.items()}
            features = self.feature_extractor(inputs)  # implement this wrapper
            x_enc = features["encoder_features"].permute(1, 0, 2)        # [1, 2048, 256]
            x_dec = features["decoder_features"].permute(1, 0, 2)        # [1, 256, 256]
            # print("x_enc", x_enc.shape, "x_dec", x_dec.shape)
            x = torch.cat([x_enc, x_dec], dim=1).squeeze(0)  # [2304, 256]
            # print("x", x.shape)
            
        with open(caption_path, "r") as f:
            caption = json.load(f)["captions"][0]

        # Tokenize
        tokens = self.tokenizer(
            caption,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

        return {
            "3d_feat": x,
            "input_ids": tokens.input_ids.squeeze(0),
            "attention_mask": tokens.attention_mask.squeeze(0),
        }


class TestingDataset(Dataset):
    def __init__(self, root_dir, feature_extractor, num_points=40000):
        self.root_dir = root_dir
        # self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor.eval()
        self.num_points = num_points
        # self.max_length = max_length
        self.scene_dirs = sorted(
            [
                os.path.join(root_dir, d)
                for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ]
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.scene_dirs)

    def __getitem__(self, idx):
        scene_path = self.scene_dirs[idx]
        caption_path = os.path.join(scene_path, "dataset.json")

        dir_name = os.path.basename(os.path.normpath(scene_path))  # 使用 os.path.normpath 去掉末尾的斜線

        pc_path = os.path.join(scene_path, "world_points.txt")
        # caption_path = os.path.join(scene_path, "dataset.json")

        points = np.loadtxt(pc_path, skiprows=1)
        if points.shape[0] > self.num_points:
            choice = np.random.choice(points.shape[0], self.num_points, replace=False)
            points = points[choice]
        points = torch.tensor(points, dtype=torch.float32) 
        pc_min = points.min(dim=0).values  
        pc_max = points.max(dim=0).values  

        point_inputs =  {
            "point_clouds": points.unsqueeze(0),       # [1, N, 3] where N is the number of points
            "point_cloud_dims_min": pc_min.unsqueeze(0),           # [1, 3]
            "point_cloud_dims_max": pc_max.unsqueeze(0),            
        }
        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in point_inputs.items()}
            features = self.feature_extractor(inputs)  # implement this wrapper
            x_enc = features["encoder_features"].permute(1, 0, 2)        # [1, 2048, 256]
            x_dec = features["decoder_features"].permute(1, 0, 2)        # [1, 256, 256]
            x = torch.cat([x_enc, x_dec], dim=1).squeeze(0)  # [2304, 256]
            
        with open(caption_path, "r") as f:
            caption = json.load(f)["captions"][0]
        # tokens = self.tokenizer(
        #     caption,
        #     return_tensors="pt",
        #     truncation=True,
        #     padding="max_length",
        #     max_length=self.max_length,
        # )

        return {"3d_feat": x,
                "dir": dir_name,
                "caption": caption,
                }  # Return only features for testing
