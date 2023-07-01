import torch
import torch.nn as nn
import numpy as np

from typing import Dict
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule

from src.utils.running_average import RunningAverage

from legacy.analysis_utils import DiscriminatorStats
from legacy.performance_diagram import PerformanceDiagramStable


class GANFramework(LightningModule):
    def __init__(
        self, 
        adv_weight, 
        disc_d, 
        encoder, 
        forecaster, 
        ipshape, 
        target_len,
        loss_kwargs, 
        checkpoint_directory,
        add_hetr_from_poni
    ):
        super().__init__()
        self.encoder = encoder
        self.forecaster = forecaster
        self.discriminator = discriminator
        self.loss_fn = loss_fn
        self.dis_loss_fn = dis_loss_fn
        #
        self._add_from_poni = add_hetr_from_poni
        self._lr = learning_rate
        self._target_len = target_len
        self._adv_w = adv_weight
        #
        self.loss_P = RunningAverage()
        self.loss_GD = RunningAverage()
        self.loss_G = RunningAverage()
        self.loss_D = RunningAverage()
        #
        self.automatic_optimization = False
        self.validation_step_outputs = []


        self._ckp_dir = checkpoint_directory

        self._val_criterion = PerformanceDiagramStable()
        self._D_stats = DiscriminatorStats()

        # save hyperparameter

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
        rainfall_previous_hr = torch.mean(input_data['rain'], dim=1, keepdim=False)
        aux_rainmap = torch.cat([rainfall_previous_hr, label[:, :self._target_len-1]], dim=1)
        aux_rainmap = torch.unsqueeze(aux_rainmap, dim=2) # [B, output_len, 1, H, W]

        if not self._add_from_poni:
            input_copy = torch.cat([value for value in input_data.values], dim=2)
            aux_hetr_data = None
        else:
            input_copy = torch.cat([input_data['rain'], input_data['radar']], dim=2)
            # only take the last frame info, ouput shape = [B, C', H, W]
            aux_hetr_data = torch.cat(
                [v[:, -1, ...] for k, v in input_data.items() if k not in ['rain', 'radar']], dim=1)

        encoder_oup = self.encoder(input_copy) # output: [N_layers][B, C, H', W']
        return self.forecaster(encoder_oup, aux_rainmap, aux_hetr_data)
    
    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.forecaster.parameters()),
            lr=self._lr
        )
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self._lr)
        return opt_g, opt_d
    
    def training_step(self, batch):
        # data from torch.data.Dataloader will be automatically turned into tensor in batch
        inp_data, label = batch
        label = label['rain']
        batch_size = label.shape[0]

        # get optimizers from `configure_optimizers`
        optimizer_g, optimizaer_d = self.optimizers()

        # train generator
        self.toggle_optimizer(optimizer_g) # requires_grad=True
        loss_dict = self.generator_loss(inp_data, label, batch_size)
        self.loss_P.add(loss_dict['progress_bar']['loss_pred'].item() * batch_size, batch_size)
        self.loss_GD.add(loss_dict['progress_bar']['loss_gd'].item() * batch_size, batch_size)
        self.loss_G.add(loss_dict['total_loss'].item() * batch_size, batch_size)
        self.log('loss_G', self.loss_G.get(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('loss_P', self.loss_P.get(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('loss_GD', self.loss_GD.get(), on_step=True, on_epoch=True, prog_bar=True)
        self.manual_backward(loss_dict['total_loss'])
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # train discriminator
        self.toggle_optimizer(optimizaer_d)
        loss_dict = self.discriminator_loss(loss_dict['prediction'], label, batch_size)
        self.loss_D.add(loss_dict['loss_d'].item() * batch_size, batch_size)
        self.log('loss_D', self.loss_D.get(), on_step=True, on_epoch=True, prog_bar=True)
        self.manual_backward(loss_dict['loss_d'] * self._adv_w)
        optimizaer_d.step()
        optimizaer_d.zero_grad()
        self.untoggle_optimizer(optimizaer_d)

    def training_epoch_end(self):
        self.loss_G.reset()
        self.loss_GD.reset()
        self.loss_P.reset()
        self.loss_D.reset()
        self._D_stats.reset()

    def validation_step(self, batch):
        inp_data, label = batch
        label = label['rain']
        batch_size = label.shape[0]

        # generator
        loss_dict = self.generator_loss(inp_data, label, batch_size)
        aligned_prediction = loss_dict['prediction'].permute(1, 0, 2, 3)
        self._val_criterion.compute(aligned_prediction, label)

        # Discriminator stats
        pos = self.D(val_label)
        neg = self.D(aligned_prediction)
        self._D_stats.update(neg, pos)

        log_data = loss_dict.pop('progress_bar')
        output = {'val_loss': log_data['recon_loss'], 'N': val_label.shape[0]}
        self.validation_step_outputs.append(output)
        return output

    def validation_epoch_end(self):
        val_loss_sum = 0
        N = 0
        for output in self.validation_step_outputs:
            val_loss_sum += output['val_loss'] * output['N']
            # this may not have the entire batch. but we are still multiplying it by N
            N += output['N']

        val_loss_mean = val_loss_sum / N
        self.logger.experiment.add_scalar('Loss/val', val_loss_mean.item(), self.current_epoch)
        self.log('val_loss', val_loss_mean)
        d_stats = self._D_stats.get()
        self.log('D_auc', d_stats['auc'])
        self.log('D_pos_acc', d_stats['pos_accuracy'])
        self.log('D_neg_acc', d_stats['neg_accuracy'])

        pdsr = self._val_criterion.get()['Dotmetric']
        self._val_criterion.reset()
        self.log('pdsr', pdsr)
        self.validation_step_outputs.clear()

        self.loss_G.reset()
        self.loss_GD.reset()
        self.loss_P.reset()
        self.loss_D.reset()
        self._D_stats.reset()
    
    def generator_loss(self, input_data, label, batch_size):
        # Loss_prediction
        prediction = self(input_data, label)
        loss_pred = self.loss_fn(prediction, label)

        # Loss_generator_discriminator
        # ReLU is used since generator fools discriminator with -ve values
        disc_guess = self.discriminator(nn.ReLU()(prediction))
        disc_loss = self.dis_loss_fn(
            disc_guess, torch.ones([self._target_len * batch_size, 1]).type_as(label))

        # Loss_generator = Loss_prediction + Loss_generator_discriminator
        total_loss = disc_loss * self._adv_w + loss_pred * (1 - self._adv_w)
        tqdm_dict = {'loss_pred': loss_pred, 'loss_gd': disc_loss}
        ret = {'total_loss': total_loss, 'progress_bar': tqdm_dict, 'prediction': prediction}
        return ret
    
    def discriminator_loss(self, prediction, label, batch_size):
        # how well can it recognize reality?
        disc_guess = self.discriminator(label)
        real_loss = self.adversarial_loss_fn(
            disc_guess, torch.ones([self._target_len * batch_size, 1]).type_as(label))

        # how well can it label as fake?
        # ReLU is used since generator fools discriminator with -ve values
        disc_guess = self.discriminator(nn.ReLU()(prediction.detach()))
        fake_loss = self.adversarial_loss_fn(
            disc_guess, torch.zeros([self._target_len * batch_size, 1]).type_as(label))

        # discriminator loss is the average of these
        loss_d = (real_loss + fake_loss) / 2
        ret = {'loss_d': loss_d}
        return ret
    
    def get_checkpoint_callback(self):
        return ModelCheckpoint(
            dirpath=self._ckp_dir,
            filename='{epoch}_{val_loss:.6f}_{pdsr:.2f}_{D_auc:.2f}_{D_pos_acc:.2f}_{D_neg_acc:.2f}',
            save_top_k=1,
            verbose=True,
            monitor='val_loss',
            mode='min'
        )