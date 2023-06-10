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
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# install other depandencies
pip install -r requirement_pypi.txt

# There is no available `datatable` supporting Python 3.10 from pypi repo
# (waiting for v1.1.0). The temporary workaround is downloading 
# from S3 directly.
pip install https://h2o-release.s3.amazonaws.com/datatable/dev%2Fdatatable-1.1.0a2132%2Fdatatable-1.1.0a2132-cp310-cp310-manylinux_2_12_x86_64.whl#sha256=db998c9bdba371e4bd6861282c60744c8ac0ba2c1ca1f2aa1fe1857d48f1d413
```