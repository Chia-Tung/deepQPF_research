import sys
import yaml
import torch
import torchvision
import torch.utils.cpp_extension

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src.data_manager import DataManager
from src.model_director import ModelDirector

def main():
    config_file = './exp_config/exp1.yml'
    with open(config_file, "r") as content:
        config = yaml.safe_load(content)

    dm = DataManager(
        data_meta_info=config['train_config']['data_meta_info'], 
        **config['train_config']['data_loader_params']
    )
    
    model = ModelDirector(
        **config['model'],
        data_info = dm.get_data_info()
    ).build()
    
    logger = TensorBoardLogger(save_dir='logs', name=config['model']['model_config']['name'])
    checkpoint_callback = model.get_checkpoint_callback()
    trainer = Trainer(
        benchmark=True,
        accelerator='gpu',
        devices=[0],
        max_epochs=50, 
        fast_dev_run=False, 
        logger=logger,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback, EarlyStopping(monitor="val_loss", patience=5)],
    )
    trainer.fit(model, dm)

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES']="0"
    print('Python version', sys.version)
    print('CUDA_HOME', torch.utils.cpp_extension.CUDA_HOME)
    print('CudaToolKit Version', torch.version.cuda)
    print('Cudnn Version', torch.backends.cudnn.version())
    print('torch Version', torch.__version__)
    print('torchvision Version', torchvision.__version__)

    main()
