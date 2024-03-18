# DeepQPF Codebase for Research
This repository is for research use. All the contents aim to build AI models which can predict the rainfall amount in the next three hours. In current stage, only *rain rate (mm/h)* and *radar reflectivity (dBZ)* are considered as input. 

![](./visualization/tswp_example.png)
*Figure 1. DeepQPF rainfall prediction from TSWP website at 2024/03/14 19:10 pm. TSWP is a private website under Center for Weather Climate and Disaster Research, Taiwan*

## Build Environment
Recommand to use a virtual environment like conda env
```bash
# update conda in base env
conda update conda

# create a new env for this project
conda create --name deepQPF_research python=3.10 -y

# turn off the auto-activate
conda config --set auto_activate_base false

# install PyTorch regarding to your CUDA driver version
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# install other depandencies
pip install -r requirement_pypi.txt

# There is no available `datatable` supporting Python 3.10 from pypi repo
# (waiting for v1.1.0). The temporary workaround is downloading 
# from S3 directly.
pip install https://h2o-release.s3.amazonaws.com/datatable/dev%2Fdatatable-1.1.0a2132%2Fdatatable-1.1.0a2132-cp310-cp310-manylinux_2_12_x86_64.whl#sha256=db998c9bdba371e4bd6861282c60744c8ac0ba2c1ca1f2aa1fe1857d48f1d413
```

## Quick Start
step 1. set hyperparameters
1. `exp_config/your_experiment_settings.yml`
2. `src/const.py`
3. `set_lat_range` & `set_lon_range` in all data loaders

step 2. start training
```bash
python train.py
```

step 3. monitoring on tensorboard
```bash
# only http can work, not https
tensorboard --logdir ./deepQPF_research/logs/ --port 5000 --bind_all --load_fast=false
```

![](./visualization/tb_example.png)
*figure 2. Tensorboard example*

## Data Loader
1. Load all `datetime` from **ALL DataLoader Class** listed in `config.yaml`, including input data and output data.
2. Make the intersection of `datetime` from different data.
3. Eliminate invalid time/blacklist/greylist
4. Split data into train/valid/test
5. Initialize `AdoptedDataset` inherited from `PyTorch.Dataset`

## Model Director
1. Choose certain model builder
2. Prepare all elements the model needs and `build_model()`

|Model Name|Total Params|Gird Size|Batch Size|GPU Mem Consumption|Min Valid Loss|
|:----:|:----:|:----:|:----:|:----:|:----:|
|ConvGRU|8.7M|(540, 420)|64|18,767 MiB|1.102|
