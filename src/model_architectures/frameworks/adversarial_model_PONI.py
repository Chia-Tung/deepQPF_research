import torch, os
import torch.nn as nn
from collections import OrderedDict
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule

from legacy.running_average import RunningAverage
from legacy.discriminator import Discriminator
from legacy.loss import get_criterion
from legacy.analysis_utils import DiscriminatorStats
from legacy.performance_diagram import PerformanceDiagramStable

    
class BalAdvPoniModel(LightningModule):
    def __init__(
        self, adv_weight, discriminator_downsample, encoder, forecaster, ipshape, target_len,
        loss_kwargs, checkpoint_directory
    ):
        super().__init__()
        self.encoder = encoder
        self.forecaster = forecaster
        ''' jeffrey add this'''
        # for y_input use
        self.aux = torch.nn.Sequential(
            nn.Conv2d(1, 8, 7, 5, 1), # input C=1, only rainmap
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(8, 32, 5, 3, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 128, 3, 2, 1), 
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self._img_shape = ipshape
        self.D = Discriminator(self._img_shape, downsample=discriminator_downsample)
        self._adv_w = adv_weight
        self.lr = 1e-04
        self._loss_type = loss_kwargs['type']
        self._ckp_dir = checkpoint_directory
        self._target_len = target_len
        self._criterion = get_criterion(loss_kwargs)
        self._recon_loss = RunningAverage()
        self._GD_loss = RunningAverage()
        self._G_loss = RunningAverage()
        self._D_loss = RunningAverage()
        self._label_smoothing_alpha = 0.001
        # self._train_D_stats = DiscriminatorStats()
        self._val_criterion = PerformanceDiagramStable()
        self._D_stats = DiscriminatorStats()
        self.automatic_optimization = False
        self.validation_step_outputs = []
        print(f'[{self.__class__.__name__} W:{self._adv_w}] Ckp:{os.path.join(self._ckp_dir)} ')
    
    def training_step(self, batch, batch_idx):
        train_data, train_label, train_mask = batch
        g_opt, d_opt = self.optimizers()
        N = train_data['rain'].shape[0]

        # train generator
        self.toggle_optimizer(g_opt)
        loss_dict = self.generator_loss(train_data, train_label, train_mask)
        self._recon_loss.add(loss_dict['progress_bar']['recon_loss'].item() * N, N)
        self._GD_loss.add(loss_dict['progress_bar']['adv_loss'].item() * N, N)
        self._G_loss.add(loss_dict['loss'].item() * N, N)
        self.log('G', self._G_loss.get(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('GRecon', self._recon_loss.get(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('GD', self._GD_loss.get(), on_step=True, on_epoch=True, prog_bar=True)
        self.manual_backward(loss_dict['loss'])
        g_opt.step()
        g_opt.zero_grad()
        self.untoggle_optimizer(g_opt)

        # train discriminator
        self.toggle_optimizer(d_opt)
        loss_dict = self.discriminator_loss(train_data, train_label)
        self._D_loss.add(loss_dict['loss'].item() * N, N)
        self.log('D', self._D_loss.get(), on_step=True, on_epoch=True, prog_bar=True)
        # balance discriminator
        self.manual_backward(loss_dict['loss'] * self._adv_w)
        d_opt.step()
        d_opt.zero_grad()
        self.untoggle_optimizer(d_opt)

    def generator_loss(self, input_data, target_label, target_mask):
        predicted_reconstruction = self(input_data, target_label)
        recons_loss = self._criterion(predicted_reconstruction, target_label['rain'], target_mask)
        # train generator
        input_data = input_data['rain']
        N = self._target_len * input_data.size(0)
        valid = torch.ones(N, 1)
        valid = valid.type_as(input_data)

        # adversarial loss is binary cross-entropy
        # ReLU is used since generator fools discriminator with -ve values
        adv_loss = self.adversarial_loss_fn(
            self.D(nn.ReLU()(predicted_reconstruction)).view(N, 1), valid, smoothing=False)
        tqdm_dict = {'recon_loss': recons_loss, 'adv_loss': adv_loss}
        loss = adv_loss * self._adv_w + recons_loss * (1 - self._adv_w)
        output = OrderedDict({'loss': loss, 'progress_bar': tqdm_dict, 'prediction': predicted_reconstruction})
        return output

    def discriminator_loss(self, input_data, target_label):
        predicted_reconstruction = self(input_data, target_label)
        input_data = input_data['rain']
        target_label = target_label['rain']
        N = self._target_len * input_data.size(0)
        valid = torch.ones(N, 1)
        valid = valid.type_as(input_data)
        d_out = self.D(target_label)
        real_loss = self.adversarial_loss_fn(d_out.view(N, 1), valid)

        # how well can it label as fake?
        fake = torch.zeros(N, 1)
        fake = fake.type_as(input_data)
        # ReLU is used since generator fools discriminator with -ve values
        fake_loss = self.adversarial_loss_fn(self.D(nn.ReLU()(predicted_reconstruction.detach())).view(N, 1), fake)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        tqdm_dict = {'d_loss': d_loss}
        output = OrderedDict({'loss': d_loss, 'progress_bar': tqdm_dict})
        return output

    def forward(self, input, label):
        # input: [B, 6, ch, H, W]; label: [B, 3, H, W]
        input_copy = torch.cat((input['rain'], input['radar']), dim=2)
        label = label['rain']
        output_from_ecd = self.encoder(input_copy) # output: [3][B, n_out, H', W']
        y_0 = torch.mean(input['rain'], dim=1, keepdim=False) # y_0: [B, 1, H, W]
        tmp = [label[:, x:x+1] for x in range(0, label.size(1) - 1)]
        y = torch.cat((y_0, *tmp), dim=1) # [B, 3, H, W]
        del tmp, y_0
        return self.forecaster(list(output_from_ecd), y, self.aux)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.forecaster.parameters()) +
            list(self.aux.parameters()),
            lr=self.lr
        )
        opt_d = torch.optim.Adam(self.D.parameters(), lr=self.lr)
        return opt_g, opt_d

    def adversarial_loss_fn(self, y_hat, y, smoothing=True):
        uniq_y = torch.unique(y)
        assert len(uniq_y) == 1
        if smoothing:
            # one sided smoothing.
            y = y * (1 - self._label_smoothing_alpha)
        return nn.BCELoss()(y_hat, y)

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

    def on_validation_epoch_end(self):
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

    def on__epoch_end(self, *args):
        self._G_loss.reset()
        self._GD_loss.reset()
        self._recon_loss.reset()
        self._D_loss.reset()
        self._D_stats.reset()

    def get_checkpoint_callback(self):
        return ModelCheckpoint(
            dirpath=self._ckp_dir,
            filename='{epoch}_{val_loss:.6f}_{pdsr:.2f}_{D_auc:.2f}_{D_pos_acc:.2f}_{D_neg_acc:.2f}',
            save_top_k=1,
            verbose=True,
            monitor='val_loss',
            mode='min'
        )