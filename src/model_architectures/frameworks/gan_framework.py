from typing import Dict

import numpy as np
import pytorch_lightning.loggers as pl_loggers
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint

from src.const import CONFIG
from src.utils.discriminator_statistic import DiscriminatorStats
from src.utils.performance_diagram import PerformanceDiagramStable
from src.utils.running_average import RunningAverage
from visualization.plot_tb_viz import gen_plot


class GANFramework(LightningModule):
    def __init__(
        self,
        encoder,
        forecaster,
        discriminator,
        loss_fn,
        dis_loss_fn,
        add_hetr_from_poni,
        learning_rate,
        target_len,
        adv_weight,
        checkpoint_directory,
    ):
        super().__init__()
        # models
        self.encoder = encoder
        self.forecaster = forecaster
        self.discriminator = discriminator
        self.loss_fn = loss_fn
        self.dis_loss_fn = dis_loss_fn
        # scalars
        self._add_from_poni = add_hetr_from_poni
        self._lr = learning_rate
        self._target_len = target_len
        self._adv_w = adv_weight
        # built-in functions
        self.loss_P = RunningAverage()
        self.loss_GD = RunningAverage()
        self.loss_G = RunningAverage()
        self.loss_D = RunningAverage()
        self.D_stats = DiscriminatorStats()
        self.eval_critirion = PerformanceDiagramStable()
        # GAN setting
        self.automatic_optimization = False
        self.validation_step_outputs = []
        # checkpint
        self._ckp_dir = checkpoint_directory
        # save hyperparameter
        self.save_hyperparameters(
            "add_hetr_from_poni", "learning_rate", "target_len", "adv_weight"
        )

    def forward(self, input_data: Dict[str, np.ndarray], label: Dict[str, np.ndarray]):
        """
        Args:
            input_data (Dict[str, np.ndarray]): All of the data including axiliary information,
                and the array for each parameter is the shape of [B, input_len, num_channel,
                H, W]
            label (Dict[str, np.ndarray]): Rainfall targets whose array is be the shape
                of [B, output_len, H, W]
        Return:
            output: rainfall predictions of shape [Seq, Batch, Height, Width]
        """
        # prepare auxiliary data
        rainfall_previous_hr = torch.mean(input_data["rain"], dim=1, keepdim=False)
        aux_rainmap = torch.cat(
            [rainfall_previous_hr, label[:, : self._target_len - 1]], dim=1
        )
        aux_rainmap = torch.unsqueeze(aux_rainmap, dim=2)  # [B, output_len, 1, H, W]

        if not self._add_from_poni:
            input_copy = torch.cat([value for value in input_data.values()], dim=2)
            aux_hetr_data = None
        else:
            input_copy = torch.cat([input_data["rain"], input_data["radar"]], dim=2)
            # only take the last frame info, ouput shape = [B, C', H, W]
            aux_hetr_data = torch.cat(
                [
                    v[:, -1, ...]
                    for k, v in input_data.items()
                    if k not in ["rain", "radar"]
                ],
                dim=1,
            )

        encoder_oup = self.encoder(input_copy)  # output: [N_layers][B, C, H', W']
        return self.forecaster(encoder_oup, aux_rainmap, aux_hetr_data)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.forecaster.parameters()),
            lr=self._lr,
        )
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self._lr)
        return opt_g, opt_d

    def training_step(self, batch, batch_idx):
        # data from torch.data.Dataloader will be automatically turned into tensor in batch
        inp_data, label = batch
        label = label["rain"]
        batch_size = label.shape[0]

        # get optimizers from `configure_optimizers`
        optimizer_g, optimizaer_d = self.optimizers()

        # train generator
        self.toggle_optimizer(optimizer_g)  # requires_grad=True
        loss_dict = self.generator_loss(inp_data, label, batch_size)
        self.loss_P.add(
            loss_dict["progress_bar"]["loss_pred"].item() * batch_size, batch_size
        )
        self.loss_GD.add(
            loss_dict["progress_bar"]["loss_gd"].item() * batch_size, batch_size
        )
        self.loss_G.add(loss_dict["total_loss"].item() * batch_size, batch_size)
        self.log(
            "loss_P", self.loss_P.get(), on_step=True, on_epoch=True, prog_bar=True
        )
        self.log(
            "loss_GD", self.loss_GD.get(), on_step=True, on_epoch=True, prog_bar=True
        )
        self.log(
            "loss_G", self.loss_G.get(), on_step=True, on_epoch=True, prog_bar=True
        )
        self.manual_backward(loss_dict["total_loss"])
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # train discriminator
        self.toggle_optimizer(optimizaer_d)
        loss_dict = self.discriminator_loss(loss_dict["prediction"], label, batch_size)
        self.loss_D.add(loss_dict["loss_d"].item() * batch_size, batch_size)
        self.log(
            "loss_D", self.loss_D.get(), on_step=True, on_epoch=True, prog_bar=True
        )
        self.manual_backward(loss_dict["loss_d"])
        optimizaer_d.step()
        optimizaer_d.zero_grad()
        self.untoggle_optimizer(optimizaer_d)

    def on_train_epoch_end(self):
        self.loss_P.reset()
        self.loss_GD.reset()
        self.loss_G.reset()
        self.loss_D.reset()

    def validation_step(self, batch, batch_idx):
        inp_data, label = batch
        label = label["rain"]
        batch_size = label.shape[0]

        # generator prediction loss
        loss_dict = self.generator_loss(inp_data, label, batch_size)
        log_data = loss_dict.pop("progress_bar")
        ret = {"val_loss": log_data["loss_pred"], "N": batch_size}
        self.validation_step_outputs.append(ret)

        # performance diagram
        aligned_prediction = loss_dict["prediction"]
        self.eval_critirion.compute(aligned_prediction, label)

        # Discriminator stats
        pos = self.discriminator(label)
        neg = self.discriminator(aligned_prediction)
        self.D_stats.update(neg, pos)

        # log images
        if batch_idx in range(1, 11, 2):  # always log the first batch 64 images
            self.log_tb_images(
                inp_data, label, loss_dict["prediction"], bid=batch_idx, max_img_num=3
            )

    def on_validation_epoch_end(self):
        # total validation loss
        val_loss_sum = 0
        N = 0
        for output in self.validation_step_outputs:
            val_loss_sum += output["val_loss"] * output["N"]
            N += output["N"]
        val_loss_mean = val_loss_sum / N
        self.log("val_loss", val_loss_mean)

        # performance diagram
        pdsr = self.eval_critirion.get()["Dotmetric"]
        self.log("pdsr", pdsr)

        # discriminator statistics
        d_stats = self.D_stats.get()
        # self.log('D_auc', d_stats['auc'])
        self.log("D_pos_acc", d_stats["pos_accuracy"])
        self.log("D_neg_acc", d_stats["neg_accuracy"])

        # reset all
        self.D_stats.reset()
        self.eval_critirion.reset()
        self.validation_step_outputs.clear()

    def generator_loss(self, input_data, label, batch_size):
        # Loss_prediction
        prediction = self(input_data, label)
        loss_pred = self.loss_fn(prediction, label)

        # Loss_generator_discriminator
        # ReLU is used since generator fools discriminator with -ve values
        disc_guess = self.discriminator(nn.ReLU()(prediction))
        disc_loss = self.dis_loss_fn(
            disc_guess, torch.ones([self._target_len * batch_size, 1]).type_as(label)
        )

        # Loss_generator = Loss_prediction + Loss_generator_discriminator
        total_loss = disc_loss * self._adv_w + loss_pred * (1 - self._adv_w)
        tqdm_dict = {"loss_pred": loss_pred, "loss_gd": disc_loss}
        ret = {
            "total_loss": total_loss,
            "progress_bar": tqdm_dict,
            "prediction": prediction,
        }
        return ret

    def discriminator_loss(self, prediction, label, batch_size):
        # how well can it recognize reality?
        disc_guess = self.discriminator(label)
        real_loss = self.dis_loss_fn(
            disc_guess, torch.ones([self._target_len * batch_size, 1]).type_as(label)
        )

        # how well can it label as fake?
        # ReLU is used since generator fools discriminator with -ve values
        disc_guess = self.discriminator(nn.ReLU()(prediction.detach()))
        fake_loss = self.dis_loss_fn(
            disc_guess, torch.zeros([self._target_len * batch_size, 1]).type_as(label)
        )

        # discriminator loss is the average of these
        loss_d = self._adv_w * (real_loss + fake_loss) / 2
        ret = {"loss_d": loss_d}
        return ret

    def get_checkpoint_callback(self):
        return ModelCheckpoint(
            dirpath=self._ckp_dir,
            filename="GAN-{epoch:02d}-{val_loss:.6f}",
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
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
