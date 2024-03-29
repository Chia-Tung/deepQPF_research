import sys

import torch
import torch.utils.cpp_extension
import torchvision
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from src.const import CONFIG as config
from src.data_manager import DataManager
from src.model_director import ModelDirector


def main():
    dm = DataManager(
        data_meta_info=config["train_config"]["data_meta_info"],
        **config["train_config"]["data_loader_params"],
    )

    model = ModelDirector(
        data_info=dm.get_data_info(),
        **config["model"],
    ).build_model()

    logger = TensorBoardLogger(
        save_dir="logs", name=config["model"]["model_config"]["name"]
    )

    checkpoint_callback = model.get_checkpoint_callback()
    trainer = Trainer(
        num_sanity_val_steps=1,
        benchmark=True,
        fast_dev_run=False,  # use n batch(es) to fast run through train/valid
        logger=logger,
        check_val_every_n_epoch=1,
        max_epochs=50,
        limit_train_batches=None,
        limit_val_batches=None,
        accelerator="gpu",
        devices=[0, 1],
        strategy="ddp",
        callbacks=[
            LearningRateMonitor(),
            EarlyStopping(monitor="val_loss_epoch", patience=5),
            checkpoint_callback,
        ],
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    print("Python version", sys.version)
    print("CUDA_HOME", torch.utils.cpp_extension.CUDA_HOME)
    print("CudaToolKit Version", torch.version.cuda)
    print("Cudnn Version", torch.backends.cudnn.version())
    print("torch Version", torch.__version__)
    print("torchvision Version", torchvision.__version__)

    main()
