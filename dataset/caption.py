from openai import OpenAI
import base64
import os
import json

def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def describe_multiple_images(image_paths):
    client = OpenAI()
    
    # Encode all image paths
    image_contents = [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + encode_image(path)}
        } for path in image_paths
    ]
    
    # Add the text prompt as the first content item
    content = [
        {"type": "text", "text": f"Please extract useful information from the following images."}
    ] + image_contents

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an advanced AI image captioner assisting another language model that performs object manipulation tasks in a 3D environment. Your job is to generate concise, informative captions based on three visual observations from an embodied camera moving through a static scene. You are provided with three sets of images, corresponding to time steps t-1, t, and t+1, captured from slightly different camera positions. Each set includes: An RGB image (the first-person camera view), a depth map (showing per-pixel depth), and a segmented feature map (identifying object classes and boundaries). Your caption should integrate spatial information from the three views and describe: Relative 3D positions of visible objects (e.g., “vase is on shelf, left side, near front edge”), Distinctive visual features to differentiate similar objects (e.g., “the left drawer has a metal handle, unlike the right one”), Object affordances (e.g., “safe is closed but has a visible handle”), when relevant. Do not only consider object relationships within each individual frame. Instead, please reason about object correspondences across frames. For example, if the camera is moving to the right, the middle object of the last frame should be on the right of the middle object of the first frame. However, please do not mention the camera in your response. Your caption should be short but detailed enough to assist the decision-making model in selecting and manipulating objects. Avoid generic or redundant descriptions."},
            {"role": "user", "content": content}
        ],
        max_tokens=200
    )

    return response.choices[0].message.content

def collect_image_paths(base_dir):
    segment_ids = ["segment_2.png", "segment_3.png", "segment_4.png"]
    image_paths = []

    for subfolder in ["high_res", "instance_masks", "depth_images"]:
        for seg_id in segment_ids:
            img_path = os.path.join(base_dir, subfolder, "000000000", seg_id)
            image_paths.append(img_path)
    
    return image_paths

def save_results_to_dataset(scene_path, caption):
    dataset_path = os.path.join(scene_path, "dataset.json")
    
    if not os.path.exists(dataset_path):
        data = {"captions": []}
    else:
        with open(dataset_path, "r") as f:
            data = json.load(f)

    data["captions"].append(caption)

    with open(dataset_path, "w") as f:
        json.dump(data, f, indent=4)

def process_all_scenes(root="../dataset/output"):
    for root_dir, dirs, _ in os.walk(root):
        for dir_name in dirs:
            trial_path = os.path.join(root_dir, dir_name)
            subdir = os.listdir(trial_path)
            scene_path = os.path.join(trial_path, subdir[0])
            segment_dir = os.path.join(scene_path, "high_res", "000000025")
            if not os.path.exists(segment_dir):
                continue  # Skip if path not found

            image_paths = collect_image_paths(scene_path)
            if image_paths is None:
                continue  # Incomplete data, skip

            caption = describe_multiple_images(image_paths)  # <-- Your function
            save_pth = os.path.join(root_dir, "ds_medium", dir_name+"_1")
            print(save_pth)
            if not os.path.exists(save_pth):
                os.makedirs(save_pth)
            # copy images to save_pth
            for img_path in image_paths[:3]:
                if os.path.exists(img_path):
                    os.system(f"cp {img_path} {save_pth}")
            
            save_results_to_dataset(save_pth, caption)
            print(f"Processed scene: {dir_name}, Caption: {caption}")

# Call this function to run the batch
process_all_scenes()

# image_paths = [
#     "../dataset/output/look_at_obj_in_light-CD-None-DeskLamp-314/trial_T20190907_114323_767231/high_res/000000000/segment_2.png",
#     "../dataset/output/look_at_obj_in_light-CD-None-DeskLamp-314/trial_T20190907_114323_767231/high_res/000000000/segment_3.png",
#     "../dataset/output/look_at_obj_in_light-CD-None-DeskLamp-314/trial_T20190907_114323_767231/high_res/000000000/segment_4.png",
#     "../dataset/output/look_at_obj_in_light-CD-None-DeskLamp-314/trial_T20190907_114323_767231/instance_masks/000000000/segment_2.png",
#     "../dataset/output/look_at_obj_in_light-CD-None-DeskLamp-314/trial_T20190907_114323_767231/instance_masks/000000000/segment_3.png",
#     "../dataset/output/look_at_obj_in_light-CD-None-DeskLamp-314/trial_T20190907_114323_767231/instance_masks/000000000/segment_4.png",
#     "../dataset/output/look_at_obj_in_light-CD-None-DeskLamp-314/trial_T20190907_114323_767231/depth_images/000000000/segment_2.png",
#     "../dataset/output/look_at_obj_in_light-CD-None-DeskLamp-314/trial_T20190907_114323_767231/depth_images/000000000/segment_3.png",
#     "../dataset/output/look_at_obj_in_light-CD-None-DeskLamp-314/trial_T20190907_114323_767231/depth_images/000000000/segment_4.png",
# ]

# res = describe_multiple_images(image_paths)
# print(res)


