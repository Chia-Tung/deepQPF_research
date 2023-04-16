# DeepQPF Codebase for Research
## Build Environment
Recommand to use a virtual environment like conda env
```bash
# update conda in base env
conda update conda

# create a new env for this project
conda create --name deepQPF_research python=3.8 -y

# turn off the auto-activate
conda config --set auto_activate_base false

# install PyTorch regarding to your CUDA driver version
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
# or
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# install geopandas (DON'T use pip install)
conda install geopandas -y

# install other depandencies
pip install -r requirement_pypi.txt
```

:bulb: If you meet the `AttributeError: module ‘distutils‘ has no attribute ‘version‘` when using the `torch.tensorboard.__init__`, please check this [solution](https://zhuanlan.zhihu.com/p/556704117).  
:fire: If you upgrade to **PyTorch2.0**, there is an issue open talking about the `ImportError: cannot import name 'Backend' from 'torch._C._distributed_c10d' `. Please check [this](https://github.com/pytorch/pytorch/issues/94806) for a workaround solution.