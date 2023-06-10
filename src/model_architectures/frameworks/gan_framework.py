import torch
import torch.nn as nn
import numpy as np

from typing import Dict
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule

from legacy.running_average import RunningAverage
from legacy.discriminator import Discriminator
from legacy.loss import get_criterion
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
        self.discriminator = Discriminator(ipshape, downsample=disc_d)
        self._disc_weight = adv_weight
        self._target_len = target_len
        self._loss_type = loss_kwargs['type']
        self._ckp_dir = checkpoint_directory
        self._loss_fn = get_criterion(loss_kwargs)
        self._add_from_poni = add_hetr_from_poni

        self.lr = 1e-04
        self._recon_loss = RunningAverage()
        self._GD_loss = RunningAverage()
        self._G_loss = RunningAverage()
        self._D_loss = RunningAverage()
        self._label_smoothing_alpha = 0.001
        self._val_criterion = PerformanceDiagramStable()
        self._D_stats = DiscriminatorStats()
        self.automatic_optimization = False
        self.validation_step_outputs = []

        # save hyperparameter
        print(f'[{self.__class__.__name__} Disc_Weight:{self._disc_weight}] Ckp:{self._ckp_dir}')

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
        label = label['rain']

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
            list(self.encoder.parameters())+list(self.forecaster.parameters()),lr=self.lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
        return opt_g, opt_d
    
    def training_step(self, batch, batch_idx):
        inp_data, label, mask = batch
        optimizer_g, optimizaer_d = self.optimizers() # return optimizers from `configure_optimizers`
        batch_size = inp_data['rain'].shape[0]

        # train generator
        self.toggle_optimizer(optimizer_g)
        loss_dict = self.generator_loss(inp_data, label, mask)
        self._recon_loss.add(loss_dict['progress_bar']['recon_loss'].item() * batch_size, batch_size)
        self._GD_loss.add(loss_dict['progress_bar']['adv_loss'].item() * batch_size, batch_size)
        self._G_loss.add(loss_dict['loss'].item() * batch_size, batch_size)
        self.log('G', self._G_loss.get(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('GRecon', self._recon_loss.get(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('GD', self._GD_loss.get(), on_step=True, on_epoch=True, prog_bar=True)
        self.manual_backward(loss_dict['loss'])
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # train discriminator
        self.toggle_optimizer(optimizaer_d)
        loss_dict = self.discriminator_loss(inp_data, label)
        self._D_loss.add(loss_dict['loss'].item() * batch_size, batch_size)
        self.log('D', self._D_loss.get(), on_step=True, on_epoch=True, prog_bar=True)
        self.manual_backward(loss_dict['loss'] * self._adv_w)
        optimizaer_d.step()
        optimizaer_d.zero_grad()
        self.untoggle_optimizer(optimizaer_d)

    def validation_step(self, batch, batch_idx):
        val_data, val_label, val_mask = batch
        # generator
        loss_dict = self.generator_loss(val_data, val_label, val_mask)
        aligned_prediction = loss_dict['prediction'].permute(1, 0, 2, 3)
        val_label = val_label['rain']
        self._val_criterion.compute(aligned_prediction, val_label)

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

        self._G_loss.reset()
        self._GD_loss.reset()
        self._recon_loss.reset()
        self._D_loss.reset()
        self._D_stats.reset()

    def training_epoch_end(self):
        self._G_loss.reset()
        self._GD_loss.reset()
        self._recon_loss.reset()
        self._D_loss.reset()
        self._D_stats.reset()
    
    def generator_loss(self, input_data, label, mask):
        # Loss_prediction
        reconstruction = self(input_data, label)
        recons_loss = self._loss_fn(reconstruction, label['rain'], mask)

        # Loss_generator_discriminator
        N = self._target_len * label['rain'].size(0)
        valid = torch.ones(N, 1)
        valid = valid.type_as(label['rain'])
        # ReLU is used since generator fools discriminator with -ve values
        disc_guess = self.discriminator(nn.ReLU()(reconstruction))
        adv_loss = self.adversarial_loss_fn(disc_guess.view(N, 1), valid, smoothing=False)

        # Loss_generator
        loss = adv_loss * self._adv_w + recons_loss * (1 - self._adv_w)
        tqdm_dict = {'recon_loss': recons_loss, 'adv_loss': adv_loss}
        output = {'loss': loss, 'progress_bar': tqdm_dict, 'prediction': reconstruction}
        return output
    
    def discriminator_loss(self, input_data, label):
        N = self._target_len * label['rain'].size(0)

        # how well can it recognize reality?
        valid = torch.ones(N, 1)
        valid = valid.type_as(label['rain'])
        disc_guess = self.discriminator(label['rain'])
        real_loss = self.adversarial_loss_fn(disc_guess.view(N, 1), valid)

        # how well can it label as fake?
        predicted_reconstruction = self(input_data, label)
        fake = torch.zeros(N, 1)
        fake = fake.type_as(label['rain'])
        # ReLU is used since generator fools discriminator with -ve values
        disc_guess = self.discriminator(nn.ReLU()(predicted_reconstruction.detach()))
        fake_loss = self.adversarial_loss_fn(disc_guess.view(N, 1), fake)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        tqdm_dict = {'d_loss': d_loss}
        output = {'loss': d_loss, 'progress_bar': tqdm_dict}
        return output
    
    def adversarial_loss_fn(self, y_hat, y, smoothing=True):
        uniq_y = torch.unique(y)
        assert len(uniq_y) == 1
        if smoothing:
            # one sided smoothing.
            y = y * (1 - self._label_smoothing_alpha)
        return nn.BCELoss()(y_hat, y)
    
    def get_checkpoint_callback(self):
        return ModelCheckpoint(
            dirpath=self._ckp_dir,
            filename='{epoch}_{val_loss:.6f}_{pdsr:.2f}_{D_auc:.2f}_{D_pos_acc:.2f}_{D_neg_acc:.2f}',
            save_top_k=1,
            verbose=True,
            monitor='val_loss',
            mode='min'
        )