import pytorch_lightning.loggers as pl_loggers
import torch
from einops import rearrange
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import LambdaLR

from src.const import CONFIG
from visualization.plot_tb_viz import gen_plot


class VitFramework(LightningModule):
    def __init__(
        self,
        *,
        checkpoint_dir,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["img_processor", "vit_model", "postprocess", "loss_fn"]
        )
        # model
        self.img_processor = kwargs["img_processor"]
        self.vit_model = kwargs["vit_model"]
        self.postprocess = kwargs["postprocess"]
        self.loss_fn = kwargs["loss_fn"]
        # checkpoint
        self._ckp_dir = checkpoint_dir

    def forward(self, x):
        """
        Function: VitFramework.forward
        Args:
            x (dict[str, torch.Tensor]): input dict, keys are `radar` and `rain`
                and the array for each parameter is the shape of [B, S, C, H, W]
        """
        inp = torch.concat([i for i in x.values()], dim=2)  # [B, S, C, H, W]
        inp = rearrange(inp, "b s c h w -> b (c s) h w")  # [B, S, H, W]
        inp = self.img_processor(inp)  # [B, S, 224, 224]
        oup = self.vit_model(inp)["last_hidden_state"]  # [B, 197, 768]
        patch_oup = oup[:, 1:]  # [B, 196, 768]
        ret = self.postprocess(patch_oup)  # [B, 1, 540, 420]
        return ret

    def common_step(self, batch, batch_idx):
        inp_data, label = batch
        label = label["rain"]

        model_oup = self(inp_data)
        loss = self.loss_fn(model_oup, label)
        return loss, model_oup

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
        loss, _ = self.common_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, pred = self.common_step(batch, batch_idx)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )

        # log images on main GPU
        if batch_idx in range(1, 11, 2) and self.global_rank == 0:
            self.log_tb_images(
                batch[0], batch[1]["rain"], pred, bid=batch_idx, max_img_num=3
            )

    def get_checkpoint_callback(self):
        return ModelCheckpoint(
            dirpath=self._ckp_dir,
            filename="ViT-{epoch:02d}-{val_loss_epoch:.6f}",
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
                    w/ shape of [B, output_len, H, W]
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
                    gen_plot(concat_input[img_idx].squeeze()),
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
