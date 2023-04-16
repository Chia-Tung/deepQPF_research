# DeepQPF Codebase for Research
## Build Environment
1. Recommand to use a virtual environment like conda env
    ```bash
    # update conda in base env
    conda update conda

    # create a new env for this project
    conda create --name deepQPF_preparation python=3.8 -y

    # turn off the auto-activate
    conda config --set auto_activate_base false

    # install PyTorch regarding to your CUDA driver version
    pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

    # install geopandas (DON'T use pip install)
    conda install geopandas -y

    # install other depandencies
    pip install -r requirement_pypi.txt
    ```
