# DeepQPF Codebase for Research
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

## Hyperparameters
1. `exp_config/exp.yml`
2. `src/blacklist.Blacklist.BLACKLIST_PATH`
3. `set_lat_range` & `set_lon_range` in all data loaders
4. `src/model_architectures/utils.py`
5. `loss fn`

## Data Loader
1. Load all `datetime` from **ALL DataLoader Class** in `src/data_loaders`, including input data and output data.
2. Make the intersection of `datetime` from different data.
3. Eliminate invalid time/blacklist/greylist
4. Split data into train/valid/test
5. Initialize `AdoptedDataset` inherited from `PyTorch.Dataset`

## Model Director
1. Choose certain model builder
2. Prepare all elements the model needs and `build_model()`

## Training
Well prepare all the hyperparameters and execute `python train.py`.

## Tensorboard
```Bash
tensorboard --logdir ./deepQPF_research/logs/ --port 5000 --bind_all --load_fast=false
```