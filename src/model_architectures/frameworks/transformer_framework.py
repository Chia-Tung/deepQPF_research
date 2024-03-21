from typing import Any

import numpy as np
import pytorch_lightning.loggers as pl_loggers
import torch
from einops import rearrange
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from src.const import CONFIG
from visualization.plot_tb_viz import gen_plot


class TransformerFramework(LightningModule):
    def __init__(self, *, checkpoint_directory, **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["preprocess", "tf_encoder", "tf_decoder", "postprocess", "loss_fn"]
        )
        # model
        self.preprocess = kwargs["preprocess"]
        self.transformer = nn.Transformer(
            d_model=kwargs["d_model"],
            nhead=kwargs["n_head"],
            custom_encoder=kwargs["tf_encoder"],
            custom_decoder=kwargs["tf_decoder"],
            batch_first=True,
        )
        self.postprocess = kwargs["postprocess"]
        # save checkpoint
        self.loss_fn = kwargs["loss_fn"]
        self._ckp_dir = checkpoint_directory

    def forward(
        self, input_data: dict[str, np.ndarray], label: dict[str, np.ndarray]
    ) -> torch.Tensor:
        """
        Function: TransformerFramework.forward
        Args:
            input_data (dict[str, np.ndarray]): input data, keys are `radar` and `rain`
                and the array for each parameter is the shape of [B, S, C, H, W]
            label (dict[str, np.ndarray]): label, keys are `rain` and the array for each
                parameter is the shape of [B, C, H, W]
        Returns:
            torch.Tensor: output of the network
        """
        input_data = torch.concat(list(input_data.values()), dim=2)  # [B, 6, 2, H, W]
        label = rearrange(
            label["rain"], "b (s c) h w -> b s c h w", c=1
        )  # [B, 3, 1, H, W]
        prep_inp = self.preprocess["prep_src"](input_data)  # [B, 6, 512]
        prep_oup = self.preprocess["prep_tgt"](label)  # [B, 3, 512]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(prep_oup.size(-2))
        attn_data = self.transformer(prep_inp, prep_oup, tgt_mask=tgt_mask)
        oup = self.postprocess(attn_data)  # [B, 3, H, W]
        return oup

    def configure_optimizers(self):
        """
        optimizer <-> lr_scheduler (lr_scheduler_config)
        param groups in each optimizer <-> lambda in each lr_scheduler
        """
        # set optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

        # set lr scheduler
        def lr_lambda(epoch):
            if epoch < self.hparams.warmup_epochs:
                lr_scale = 1e1
            else:
                lr_scale = 0.95**epoch
            return lr_scale

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "name": "customized_lr",
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def training_step(self, batch, batch_idx):
        if batch_idx == 0 and self.global_step == 0 and self.global_rank == 0:
            self.log_tb_graph()

        inp_data, label = batch
        outputs = self(inp_data, label)
        loss = self.loss_fn(outputs, label["rain"])

        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.val_output_list = []
        return

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        inp_data, label = batch
        outputs = self(inp_data, label)
        label = label["rain"]
        loss = self.loss_fn(outputs, label)

        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )

        # self.val_output_list.append(loss.cpu())

        # log images on main GPU
        if batch_idx in range(1, 11, 2) and self.global_rank == 0:
            self.log_tb_images(inp_data, label, outputs, bid=batch_idx, max_img_num=3)

    # No need any more since `self.log(..., on_epoch=True)` automatically accumulates
    # variables using mean-reduction. The key name is the variable name with suffix
    # `_epoch`
    # def on_validation_epoch_end(self):
    #     total_loss = np.mean(self.val_output_list)
    #     self.log(
    #         "total_val_loss", total_loss, on_epoch=True, prog_bar=True, sync_dist=True
    #     )

    def get_checkpoint_callback(self):
        return ModelCheckpoint(
            dirpath=self._ckp_dir,
            filename="Transformer-{epoch:02d}-{val_loss_epoch:.6f}",
            save_top_k=1,
            verbose=True,
            monitor="val_loss_epoch",
            mode="min",
        )

    def get_tb_logger(self):
        """
        Retrieves the TensorBoard logger from the list of loggers in the trainer.

        Returns:
            tb_logger (tensorboardX.SummaryWriter): The TensorBoard logger object.

        Raises:
            ValueError: If the TensorBoard logger is not found in the trainer.
        """
        for logger in self.trainer.loggers:
            if isinstance(logger, pl_loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                return tb_logger

        raise ValueError("TensorboardLogger not found in trainer")

    def log_tb_images(
        self, *viz_batch: any, bid: int, max_img_num: int | None = None
    ) -> None:
        """
        Plot figures for DeepQPF project showing on Tensorboard.

        Args:
            viz_batch (tuple[any]):
                id 0 => input_data including Rain, radar, etc.
                    w/ format {"var_name", [B, input_len, num_channel, H, W]}
                id 1 => ground truth, accumulated rainfall.
                    w/ shape of [B, output_len, H, W]
                id 2 => prediction, model output.
                    w/ shape of [output_len, B, H, W]
            bid (int): batch index
            max_img_num: how many images to show on Tensorboard
        """
        tb_logger = self.get_tb_logger()

        # data preprocess
        input_data, label, pred = viz_batch
        input_var_config = CONFIG["train_config"]["data_meta_info"]

        tmp = []
        for key, value in input_data.items():
            try:
                nfactor = input_var_config[key]["normalize_factor"]
                orig_value = value * nfactor
                tmp.append(orig_value)
            except:
                raise KeyError(f"no such key: {key} in config")
        concat_input = torch.concat(tmp, dim=2)  # [B, S, C, H, W]
        img_num = max_img_num if max_img_num is not None else pred.size(0)

        # draw different type of data
        for img_idx in range(img_num):
            if self.current_epoch == 0:
                # input
                tb_logger.add_figure(
                    f"case_batch{bid}_{img_idx}/input_data",
                    gen_plot(concat_input[img_idx]),
                    global_step=0,
                )
                # ground truth
                tb_logger.add_figure(
                    f"case_batch{bid}_{img_idx}/ground_truth",
                    gen_plot(label[img_idx]),
                    global_step=0,
                )
            # prediction
            tb_logger.add_figure(
                f"case_batch{bid}_{img_idx}/prediction",
                gen_plot(pred[img_idx]),
                global_step=self.global_step,
            )

    def log_tb_graph(self):
        tb_logger = self.get_tb_logger()
        prototype_inp = {}
        for k, v in self.hparams.channel.items():
            prototype_inp[k] = torch.Tensor(
                self.hparams.batch_size,
                self.hparams.ilen,
                v,
                self.hparams.shape[0],
                self.hparams.shape[1],
            ).cuda()

        prototype_oup = dict(
            rain=torch.Tensor(
                self.hparams.batch_size,
                self.hparams.olen,
                self.hparams.shape[0],
                self.hparams.shape[1],
            )
        ).cuda()

        # this funcation will call self.forward
        tb_logger.add_graph(self, [prototype_inp, prototype_oup], verbose=False)
