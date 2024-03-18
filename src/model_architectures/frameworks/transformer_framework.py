from typing import Any

import numpy as np
import pytorch_lightning.loggers as pl_loggers
import torch
from einops import rearrange
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from transformers import get_linear_schedule_with_warmup

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
        self.model = nn.Sequential(self.preprocess, self.transformer, self.postprocess)
        # save checkpoint
        self.loss_fn = kwargs["loss_fn"]
        self._ckp_dir = checkpoint_directory

    def forward(
        self, input_data: dict[str, np.ndarray], label: dict[str, np.ndarray]
    ) -> torch.Tensor:
        # inp = [B, S, 2, 54, 42], label = [B, 3, H, W]
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

    # def configure_optimizers(self) -> Any:
    #     """Prepare optimizer and schedule (linear warmup and decay)"""
    #     model = self.model
    #     no_decay = ["bias", "norm"]
    #     optimizer_grouped_parameters = [
    #         {
    #             "params": [
    #                 p
    #                 for n, p in model.named_parameters()
    #                 if not any(nd in n for nd in no_decay)
    #             ],
    #             "weight_decay": self.hparams.weight_decay,
    #         },
    #         {
    #             "params": [
    #                 p
    #                 for n, p in model.named_parameters()
    #                 if any(nd in n for nd in no_decay)
    #             ],
    #             "weight_decay": 0.0,
    #         },
    #     ]
    #     optimizer = torch.optim.AdamW(
    #         optimizer_grouped_parameters,
    #         lr=self.hparams.learning_rate,
    #         eps=self.hparams.adam_epsilon,
    #     )

    #     scheduler = get_linear_schedule_with_warmup(
    #         optimizer,
    #         num_warmup_steps=self.hparams.warmup_steps,
    #         num_training_steps=self.trainer.estimated_stepping_batches,
    #     )
    #     scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
    #     return [optimizer], [scheduler]

    def configure_optimizers(self, learning_rate=5e-4):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        inp_data, label = batch
        outputs = self(inp_data, label)
        loss = self.loss_fn(outputs, label["rain"])

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
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

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        self.val_output_list.append({"loss": loss, "preds": outputs, "labels": label})

        # log images
        if batch_idx in range(1, 11, 2):  # always log the first batch 64 images
            self.log_tb_images(inp_data, label, outputs, bid=batch_idx, max_img_num=3)

    def on_validation_epoch_end(self):
        total_loss = np.mean([v["loss"].item() for v in self.val_output_list])
        self.log(
            "total_val_loss", total_loss, on_epoch=True, prog_bar=True, logger=True
        )

    def get_checkpoint_callback(self):
        return ModelCheckpoint(
            dirpath=self._ckp_dir,
            filename="Transformer-{epoch:02d}-{total_val_loss:.6f}",
            save_top_k=1,
            verbose=True,
            monitor="total_val_loss",
            mode="min",
        )

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
        # Get tensorboard logger
        for logger in self.trainer.loggers:
            if isinstance(logger, pl_loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                break

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
