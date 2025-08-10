## 🛠️ Setup

### Create Environment
```
conda create -n planr1 python=3.9
conda activate planr1
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pytorch-lightning==2.0.3
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch_geometric==2.3.1
```

## 📦 Datasets

### 1. Download [nuPlan Dataset](https://nuplan-devkit.readthedocs.io/en/latest/dataset_setup.html) and organize the directory as:
```
~/nuplan
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

## 🚀 Training

### 1. Preprocess Dataset
```
python preprocess_dataset.py
```
🕒: Preprocessing may take a long time (~30 hours).

### 2. Pre-training
```
python train.py --config config/train/pred.yaml
```
### 3. Fine-tuning
```
python train.py --config config/train/plan.yaml
```

## 📈 Evaluation
```
bash simulation/run_simulation.sh <sim_type> <planner> <split> <ckpt_path>
```
### Parameters

| Argument     | Description / Options                                                       |
|--------------|------------------------------------------------------------------------------|
| `<sim_type>` | `closed_loop_nonreactive_agents`, `closed_loop_reactive_agents`, `open_loop_boxes` |
| `<planner>`  | `planr1_planner`, `planr1_planner_with_refinement`                          |
| `<split>`    | `val14`, `test14-random`, `test14-hard`                                     |


### Example:
```
bash simulation/run_simulation.sh closed_loop_nonreactive_agents planr1_planner test14-random ckpts/fine-tuning.ckpt
```

## Tips
During pre-training, the final architecture for fine-tuning had not yet been finalized. As a result, the model structure saved in pre-training.ckpt differs slightly from the current architecture. Therefore, when fine-tuning from this checkpoint, the model must be loaded using the logic in lines 23–34 of train.py, rather than directly using torch.load.