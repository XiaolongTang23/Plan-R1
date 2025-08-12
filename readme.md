# Plan-R1: Safe and Feasible Trajectory Planning as Language Modeling

<p align="left">
<a href='[https://arxiv.org/abs/2404.14327](https://arxiv.org/abs/2505.17659)' style='padding-left: 0.5rem;'>
    <img src='https://img.shields.io/badge/arXiv-PDF-red?style=flat&logo=arXiv&logoColor=wihte' alt='arXiv PDF'>
</a>
</p>

![Overview](assets/network.png)

---

## Table of Contents
- [Results](#results)
- [Setup](#setup)
- [Datasets](#datasets)
- [Training](#training)
- [Validation](#validation)
- [Simulation](#simulation)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

---

## Results
**Comparison with SOTAs on nuPlan benchmark.**   
NR/R: non-reactive/reactive mode. *: with rule-based post-processing.

| Type | Planner | Val14 NR | Val14 R | Test14-hard NR | Test14-hard R | Test14-random NR | Test14-random R |
|------|---------|----------|---------|----------------|---------------|------------------|-----------------|
| *Expert* | Log-Replay | 93.53 | 80.32 | 85.96 | 68.80 | 94.03 | 75.86 |
| **Rule-based & Hybrid** | IDM | 75.60 | 77.33 | 56.15 | 62.26 | 70.39 | 72.42 |
|  | PDM-Closed* | 92.84 | 92.12 | 65.08 | 75.19 | 90.05 | 91.64 |
|  | PDM-Hybrid* | 92.77 | 92.11 | 65.99 | 76.07 | 90.10 | 91.28 |
|  | Gameformer* | 79.94 | 79.78 | 68.70 | 67.05 | 83.88 | 82.05 |
|  | PLUTO* | 92.88 | 89.84 | 80.08 | 76.88 | 92.23 | 90.29 |
|  | PlanAgent* | 93.26 | 92.75 | 72.51 | 76.82 | - | - |
|  | Diffusion Planner* | 94.26 | 92.90 | 78.87 | 82.00 | 94.80 | 91.75 |
|  | Carplanner* | - | - | - | - | 94.07 | 91.10 |
|  | Plan-R1* (Ours) | 94.77 | 93.25 | 77.79 | 80.97 | 94.11 | 94.51 |
| **Learning-based** | UrbanDriver | 68.57 | 64.11 | 50.40 | 49.95 | 51.83 | 67.15 |
|  | PDM-Open | 53.53 | 54.24 | 33.51 | 35.83 | 52.81 | 57.23 |
|  | PlanTF | 84.27 | 76.95 | 69.70 | 61.61 | 85.62 | 79.58 |
|  | PLUTO | <u>88.89</u> | 78.11 | 70.03 | 59.74 | 89.90 | 78.62 |
|  | Diffusion Planner | 89.87 | 82.80 | 75.99 | 69.22 | 89.19 | 82.93 |
|  | Plan-R1 (Ours) | 87.99 | 84.97 | 75.31 | 73.18 | 89.46 | 88.89 |

---

## Setup

### 1. Create Environment
```
conda create -n planr1 python=3.9
conda activate planr1
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pytorch-lightning==2.0.3
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch_geometric==2.3.1
```
### 2. Download the pre-trained [model weights](https://drive.google.com/drive/folders/1E_A3NVgC3IZqCbs7G75A9oZcQHGL3I4j?usp=sharing) and organize the directory as:
```
Plan-R1
├── ckpts
│   ├── pre-training.ckpt
│   └── fine-tuning.ckpt
├── ...
```

---

## Datasets

### 1. Download [nuPlan Dataset](https://nuplan-devkit.readthedocs.io/en/latest/dataset_setup.html) and organize the directory as:
```
~
├── Plan-R1
└── nuplan
    └── dataset
        ├── maps
        │   ├── nuplan-maps-v1.0.json
        │   ├── sg-one-north
        │   │   └── 9.17.1964
        │   │       └── map.gpkg
        │   ├── ...
        └── nuplan-v1.1
            └── splits
            ├── train
            ├── val
            └── test
                ├── 2021.05.25.12.30.39_veh-25_00005_00215.db
                ├── ...
```

### 2. Install nuplan-devkit
```
cd ~/nuplan
git clone https://github.com/motional/nuplan-devkit.git && cd nuplan-devkit
pip install -e .
pip install pip==24.0
pip install -r requirements.txt
pip install tensorboard
pip install numpy==1.24.4
```

---

## Training

### 1. Preprocess Dataset
Preprocessing may take a long time (~30 hours).
```
python preprocess_dataset.py
```

### 2. Pre-training
```
python train.py --config config/train/pred.yaml
```
### 3. Fine-tuning
```
python train.py --config config/train/plan.yaml
```

---

## Validation
To visualize results during validation, set `val_visualization = True` in `config/val/pred.yaml` or `config/val/plan.yaml`.
```
# For Pre-training
python val.py --config config/val/pred.yaml
# For Fine-tuning
python val.py --config config/val/plan.yaml
```

---

## Simulation
```
bash simulation/run_simulation.sh <sim_type> <planner> <split> <ckpt_path>
```
### Config

| Argument     | Description / Options                                                               |
|--------------|-------------------------------------------------------------------------------------|
| `<sim_type>` | `closed_loop_nonreactive_agents`, `closed_loop_reactive_agents`, `open_loop_boxes`  |
| `<planner>`  | `planr1_planner`, `planr1_planner_with_refinement`                                  |
| `<split>`    | `val14`, `test14-random`, `test14-hard`                                             |

### Example:
```
bash simulation/run_simulation.sh closed_loop_nonreactive_agents planr1_planner test14-random ckpts/fine-tuning.ckpt
```

### Visualization with NuBoard:
```
python run_nuboard.py
```

---

## Acknowledgements
We thank the following works for their contributions and inspiration to this project:  
- [nuPlan](https://github.com/motional/nuplan-devkit)
- [SMART](https://github.com/rainmaker22/SMART)
- [PDM](https://github.com/autonomousvision/tuplan_garage)
- [PLUTO](https://github.com/jchengai/pluto)  
- [Diffusion Planner](https://github.com/ZhengYinan-AIR/Diffusion-Planner)
- [STR](https://github.com/Tsinghua-MARS-Lab/StateTransformer)
- [HPNet](https://github.com/XiaolongTang23/HPNet)

---

## Citation

If Plan-R1 has been helpful in your research, please consider citing our work:

```
@article{tang2025plan,
  title={Plan-R1: Safe and Feasible Trajectory Planning as Language Modeling},
  author={Tang, Xiaolong and Kan, Meina and Shan, Shiguang and Chen, Xilin},
  journal={arXiv preprint arXiv:2505.17659},
  year={2025}
}
```
