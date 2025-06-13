# usage: 
# python extract_feature.py --dataset_root_dir /path/to/dataset --test_ckpt /path/to/checkpoint.pth --batchsize_per_gpu 8

import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from scene_dataset import SceneCaptionDataset
from detr3d.models import build_model
import argparse
from transformers import T5Tokenizer, T5EncoderModel

def make_args_parser():
    parser = argparse.ArgumentParser("3D Detection Using Transformers", add_help=False)

    ##### Optimizer #####
    parser.add_argument("--base_lr", default=5e-4, type=float)
    parser.add_argument("--warm_lr", default=1e-6, type=float)
    parser.add_argument("--warm_lr_epochs", default=9, type=int)
    parser.add_argument("--final_lr", default=1e-6, type=float)
    parser.add_argument("--lr_scheduler", default="cosine", type=str)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--filter_biases_wd", default=False, action="store_true")
    parser.add_argument(
        "--clip_gradient", default=0.1, type=float, help="Max L2 norm of the gradient"
    )

    ##### Model #####
    parser.add_argument(
        "--model_name",
        default="3detr",
        type=str,
        help="Name of the model",
        choices=["3detr"],
    )
    ### Encoder
    parser.add_argument(
        "--enc_type", default="vanilla", choices=["masked", "maskedv2", "vanilla"]
    )
    # Below options are only valid for vanilla encoder
    parser.add_argument("--enc_nlayers", default=3, type=int)
    parser.add_argument("--enc_dim", default=256, type=int)
    parser.add_argument("--enc_ffn_dim", default=128, type=int)
    parser.add_argument("--enc_dropout", default=0.1, type=float)
    parser.add_argument("--enc_nhead", default=4, type=int)
    parser.add_argument("--enc_pos_embed", default=None, type=str)
    parser.add_argument("--enc_activation", default="relu", type=str)

    ### Decoder
    parser.add_argument("--dec_nlayers", default=8, type=int)
    parser.add_argument("--dec_dim", default=256, type=int)
    parser.add_argument("--dec_ffn_dim", default=256, type=int)
    parser.add_argument("--dec_dropout", default=0.1, type=float)
    parser.add_argument("--dec_nhead", default=4, type=int)

    ### MLP heads for predicting bounding boxes
    parser.add_argument("--mlp_dropout", default=0.3, type=float)
    parser.add_argument(
        "--nsemcls",
        default=-1,
        type=int,
        help="Number of semantic object classes. Can be inferred from dataset",
    )

    ### Other model params
    parser.add_argument("--preenc_npoints", default=2048, type=int)
    parser.add_argument(
        "--pos_embed", default="fourier", type=str, choices=["fourier", "sine"]
    )
    parser.add_argument("--nqueries", default=256, type=int)
    parser.add_argument("--use_color", default=False, action="store_true")

    ##### Set Loss #####
    ### Matcher
    parser.add_argument("--matcher_giou_cost", default=2, type=float)
    parser.add_argument("--matcher_cls_cost", default=1, type=float)
    parser.add_argument("--matcher_center_cost", default=0, type=float)
    parser.add_argument("--matcher_objectness_cost", default=0, type=float)

    ### Loss Weights
    parser.add_argument("--loss_giou_weight", default=0, type=float)
    parser.add_argument("--loss_sem_cls_weight", default=1, type=float)
    parser.add_argument(
        "--loss_no_object_weight", default=0.2, type=float
    )  # "no object" or "background" class for detection
    parser.add_argument("--loss_angle_cls_weight", default=0.1, type=float)
    parser.add_argument("--loss_angle_reg_weight", default=0.5, type=float)
    parser.add_argument("--loss_center_weight", default=5.0, type=float)
    parser.add_argument("--loss_size_weight", default=1.0, type=float)

    ##### Dataset #####
    parser.add_argument(
        "--dataset_name", default="scannet", type=str, #"sunrgbd"
    )
    parser.add_argument(
        "--dataset_root_dir",
        type=str,
        default="./ds_small",  # change to your dataset path
        help="Root directory containing the dataset files. \
              If None, default values from scannet.py/sunrgbd.py are used",
    )
    parser.add_argument(
        "--meta_data_dir",
        type=str,
        default=None,
        help="Root directory containing the metadata files. \
              If None, default values from scannet.py/sunrgbd.py are used",
    )
    parser.add_argument("--dataset_num_workers", default=4, type=int)
    parser.add_argument("--batchsize_per_gpu", default=8, type=int)

    ##### Training #####
    parser.add_argument("--start_epoch", default=-1, type=int)
    parser.add_argument("--max_epoch", default=720, type=int)
    parser.add_argument("--eval_every_epoch", default=10, type=int)
    parser.add_argument("--seed", default=0, type=int)

    ##### Testing #####
    parser.add_argument("--test_only", default=False, action="store_true")
    parser.add_argument("--test_ckpt", default="detr3d/weights/scannet_ep1080.pth", type=str)

    ##### I/O #####
    parser.add_argument("--checkpoint_dir", default=None, type=str)
    parser.add_argument("--log_every", default=10, type=int)
    parser.add_argument("--log_metrics_every", default=20, type=int)
    parser.add_argument("--save_separate_checkpoint_every_epoch", default=100, type=int)

    ##### Distributed Training #####
    parser.add_argument("--ngpus", default=1, type=int)
    parser.add_argument("--dist_url", default="tcp://localhost:12345", type=str)

    return parser

def load_pretrained_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    print(f"Loaded pretrained weights from {checkpoint_path}")

def extract_features(model, dataloader, device): 
    """
    Extract encoder and decoder features from point clouds.
    Args:
        model: The 3DETR model.
        dataloader: DataLoader for the point cloud dataset.
        device: Device to run the model on (e.g., 'cuda' or 'cpu').
    Returns:
        List of extracted features.
    """
    model.eval()
    all_features = []
    with torch.no_grad():
        for batch in dataloader:
            #caption = batch["caption"]
            #print(id, "Caption:", caption)
            inputs = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            # print("pc shape", inputs["point_clouds"].shape)  # (B, N, 3)

            outputs = model(inputs, encoder_only=False, return_features=True)
            outputs["encoder_features"] = outputs["encoder_features"].permute(1, 0, 2)  # (n_point, B, C) 變成 (B, n_point, C)
            outputs["decoder_features"] = outputs["decoder_features"].permute(1, 0, 2)  # (num_queries, B, C)變成 (B, num_queries, C)

            all_features.append(outputs)
            # encoder_features = outputs["encoder_features"]  # (N, B, C)
            # decoder_features = outputs["decoder_features"]  # (B, Q, C)

            # # 把每一筆拆出來儲存（對齊 caption）
            # B = decoder_features.shape[0]
            # for i in range(B):
            #     all_features.append({
            #         "encoder_features": encoder_features[:, i].cpu().numpy(),  # (N, C)
            #         "decoder_features": decoder_features[i].cpu().numpy(),      # (Q, C)
            #         "caption": batch["caption"][i]
            #     })
    return all_features

def main():
    # Parse arguments using the same parser as main.py
    parser = make_args_parser()
    args = parser.parse_args()

    # Configuration
    dataset_root =  args.dataset_root_dir
    checkpoint_path = args.test_ckpt
    num_points = 20000  # Default number of points to sample from each point cloud
    batch_size = 2 # args.batchsize_per_gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    #dataset = SimplePointCloudDataset(root_dir=dataset_root, num_points=num_points)
    dataset = SceneCaptionDataset(root_dir=dataset_root, num_points=num_points)
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    # Build model
    model, _ = build_model(args, dataset_config=None)
    #print(model)
    model.to(device)

    # Load pretrained weights
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    #load_pretrained_weights(model, checkpoint_path)

    features = extract_features(model, dataloader, device)
    # print(features[0]["encoder_features"].shape)  #  # (npoints, B, C) (2048, 1, 256)
    # print(features[0]["decoder_features"].shape)  # (num_queries, B, C) (256, 1, 256)
    print(len(features), "batches processed")
    print(features[0].keys())
    print(features[0]["encoder_features"].shape)  # (B, npoints, C)
    print(features[0]["decoder_features"].shape)  # (B, num_queries, C)


    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    text_encoder = T5EncoderModel.from_pretrained("t5-base").to(device)
    text_encoder.eval()
    MAX_LEN = 128
    for batch in dataloader:
        captions = batch["caption"]
        inputs = tokenizer(
            captions,
            padding="max_length",      # 統一 padding 長度
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = text_encoder(**inputs)
            text_embeddings = outputs.last_hidden_state  # shape: (B, seq_len, hidden_dim)

if __name__ == "__main__":
    main()