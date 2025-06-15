# 3D-VIoLA: 3D Visual Information of Embodied Scene Views for Language-Action Prediction

We introduce 3D-VioLA, a system for Language-Action Prediction that enables high-level control in object management tasks using large language models (LLMs). 3D-VioLA converts 2D camera views into 3D point clouds using VGGT, then projects this visual information into text representations through a learned projector. With the 3D visual information, LLM can perform better spatial reasoning and provide more accurate control to the robot.

## Pipeline

![](asset/pipeline.png)

## Environment setup

We extracts 3D features from point cloud data using [3DETR](https://github.com/facebookresearch/3detr), with a little our own modification.

---
Navigate to the `3D-perception/detr3d` directory:

```
cd 3D-perception/detr3d
```

### 1. Install Dependencies

Install dependencies (requires **Python 3.8**, **PyTorch 1.10.0**, and **CUDA 11.3**):

```
pip install torch==1.10.0 torchvision==0.11.1 torchaudio==0.10.0 --index-url https://download.pytorch.org/whl/cu113
pip install transformers==4.20.0
pip install spacy==3.5.4
pip install SceneGraphParser --no-build-isolation
```

Then, follow the installation guide provided in the official [3DETR repository](https://github.com/facebookresearch/3detr) to complete the setup.

---

### 2. Download Pretrained Weights

Run the following script to download the pretrained model:

```
python utils/download_weights.py
```

### 3. Download Datasets

Due to the large file size, we only provide small dataset to test our code:

Dataset link: https://drive.google.com/drive/folders/1z9W2g07JyK2Z-WO5LfYg4hwvJeLu4whz?usp=sharing

After downloading the "ds_small" dataset, please move it to 3D-perception directory.

---
## Training
Under 3D-perception directory:
``` 
python train.py
```

## Inference and Evaluation
```
python inference.py
```

> For point cloud generation, please visit vggt official website and clone the repository. Our point cloud generation code is provided in dataset/pointcloud.py


## Link LLms to Alfworld

### meta-llama/Meta-Llama-3.1-8B-Instruct

- inference time : 3s per eval steps
 
## Install Source for alfworld

Installing from source is recommended for development.

Clone repo:

    git clone https://github.com/alfworld/alfworld.git alfworld
    cd alfworld

Install requirements:
```bash
# Note: Requires python 3.9 or higher
virtualenv -p $(which python3.9) --system-site-packages alfworld_env # or whichever package manager you prefer
source alfworld_env/bin/activate

pip install -e .[full]
```

Download PDDL & Game Files and pre-trained MaskRCNN detector:
```bash
export ALFWORLD_DATA=<storage_path>
python scripts/alfworld-download
```
Use `--extra` to download pre-trained checkpoints and seq2seq data.

## Inference 

### Modify Llama model 

- Look at [**Llama Agent**](3D-Llama/alfworld/agents/agent/eval_lama.py)

### INFO
- [**Agents**](3D-Lama/alfworld/agents/): Training and evaluating TextDAgger, TextDQN, VisionDAgger agents.

The enference script evaluates every `report_frequency` episodes. But additionally, you can also independently evaluate pre-trained agents:

```bash
$ python scripts/run_eval.py config/eval_config.yaml
```


## Citations

**ALFWorld**
```
@inproceedings{ALFWorld20,
  title ={{ALFWorld: Aligning Text and Embodied
           Environments for Interactive Learning}},
  author={Mohit Shridhar and Xingdi Yuan and
          Marc-Alexandre C\^ot\'e and Yonatan Bisk and
          Adam Trischler and Matthew Hausknecht},
  booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
  year = {2021},
  url = {https://arxiv.org/abs/2010.03768}
}
```

**ALFRED**
```
@inproceedings{ALFRED20,
  title ={{ALFRED: A Benchmark for Interpreting Grounded
           Instructions for Everyday Tasks}},
  author={Mohit Shridhar and Jesse Thomason and Daniel Gordon and Yonatan Bisk and
          Winson Han and Roozbeh Mottaghi and Luke Zettlemoyer and Dieter Fox},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2020},
  url  = {https://arxiv.org/abs/1912.01734}
}
```

**TextWorld**
```
@inproceedings{cote2018textworld,
  title={Textworld: A learning environment for text-based games},
  author={C{\^o}t{\'e}, Marc-Alexandre and K{\'a}d{\'a}r, {\'A}kos and Yuan, Xingdi and Kybartas, Ben and Barnes, Tavian and Fine, Emery and Moore, James and Hausknecht, Matthew and El Asri, Layla and Adada, Mahmoud and others},
  booktitle={Workshop on Computer Games},
  pages={41--75},
  year={2018},
  organization={Springer}
}
```

## License

- ALFWorld - MIT License
- TextWorld - MIT License
- Fast Downward - GNU General Public License (GPL) v3.0




 
