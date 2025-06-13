import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import numpy as np
import os
import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

# Load and preprocess example images (replace with your own image paths)
# image_names = ["/tmp2/tinger529/cosmos-predict1/frames/frame_0000.jpg", 
#                "/tmp2/tinger529/cosmos-predict1/frames/frame_0001.jpg", 
#                "/tmp2/tinger529/cosmos-predict1/frames/frame_0002.jpg"] 

# for color point cloud
# def process_images(image_names, output_path_txt):
#     images = load_and_preprocess_images(image_names).to(device)

#     with torch.no_grad():
#         with torch.cuda.amp.autocast(dtype=dtype):
#             # Predict attributes including cameras, depth maps, and point maps.
#             predictions = model(images)
#             predictions_cpu = {k: v.cpu().numpy() for k, v in predictions.items()}
#             # print("Predictions:", predictions_cpu.keys())
#             # print("World Points Shape:", predictions_cpu["world_points"].shape)
#             # print("Image Shape:", predictions_cpu["images"].shape)
#             world_points = predictions_cpu["world_points"]  # (1, 3, H, W, 3)
#             rgb_images = predictions_cpu["images"]  # (1, 3, H, W, 3)
#             # rgb_images = np.squeeze(rgb_images, axis=1)  # Now (1, 3, 518, 518)
#             rgb_images = rgb_images[0]  # Now (3, 518, 518, 3)

#             # Convert to (1, 518, 518, 3)
#             rgb_images = np.transpose(rgb_images, (0, 2, 3, 1))

#             # Flatten for RGB values: (N * H * W, 3)
#             rgb_flat = rgb_images.reshape(-1, 3)
#             B, N, H, W, _ = world_points.shape  # N = 3 (assuming 3 images), B=1

#             world_points_flat = world_points.reshape(-1, 3)          # (N * H * W, 3)

#             # Normalize RGB if needed (e.g., from [0, 1] to [0, 255])
#             if rgb_flat.max() <= 1.0:
#                 rgb_flat = (rgb_flat * 255).astype(np.uint8)

#             # Combine (r, g, b, x, y, z)
#             combined = np.hstack((rgb_flat, world_points_flat))  # shape: (N*H*W, 6)

#             # Save to txt
#             np.savetxt(output_path_txt, combined, fmt="%d %d %d %.6f %.6f %.6f",
#                     header="r g b x y z", comments='')

# for normal point cloud
def process_images(image_names, output_path_txt):
    images = load_and_preprocess_images(image_names).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = model(images)
            predictions_cpu = {k: v.cpu().numpy() for k, v in predictions.items()}
            # print("Predictions:", predictions_cpu.keys())
            # print("World Points Shape:", predictions_cpu["world_points"].shape)
            # print("Image Shape:", predictions_cpu["images"].shape)
            world_points = predictions_cpu["world_points"]  # (1, 3, H, W, 3)
            world_points_flat = world_points.reshape(-1, 3)          # (N * H * W, 3)

            # Save to txt
            np.savetxt(output_path_txt, world_points_flat, fmt="%.6f %.6f %.6f",
                    header="x y z", comments='')
        
# for dir in os.listdir("/project/aimm/ev-honolulu/dataset/new/ds_large/"):
for dir in tqdm.tqdm(os.listdir("/project/aimm/ev-honolulu/a/")):
    dir_path = os.path.join("/project/aimm/ev-honolulu/a/", dir)
    if os.path.isdir(dir_path):
        image_names = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.png')]
        if image_names:
            output_path_txt = os.path.join(dir_path, "world_points.txt")
            process_images(image_names, output_path_txt)
            print(f"Processed {len(image_names)} images in {dir_path}, saved world points to {output_path_txt}")

# output_path_txt = "world_points.txt"
# process_images(image_names, output_path_txt)
# print(f"Processed {len(image_names)} images, saved world points to {output_path_txt}")