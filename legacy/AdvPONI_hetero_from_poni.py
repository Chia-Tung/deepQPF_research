from pytorch_lightning import LightningModule
from collections import OrderedDict
import torch, os
import torch.nn as nn

from legacy.running_average import RunningAverage
from legacy.discriminator import Discriminator
from legacy.loss import get_criterion
from legacy.adversarial_model_PONI import BalAdvPoniModel
from legacy.adversarial_model_PONI_persist import TargetAttentionLean_1
from legacy.analysis_utils import DiscriminatorStats
from legacy.performance_diagram import PerformanceDiagramStable
from legacy.adversarial_model_PONI import BalAdvPoniModel

class PoniModel_addponi(BalAdvPoniModel):
    def __init__(self, adv_weight, discriminator_downsample, encoder, forecaster, ipshape, target_len,
                 loss_kwargs, checkpoint_prefix, checkpoint_directory, nc_minus):
        super(BalAdvPoniModel, self).__init__()
        self.encoder = encoder
        self.forecaster = forecaster
        ''' jeffrey add this'''
        # for y_input use
        self.aux = torch.nn.Sequential(nn.Conv2d(1+nc_minus, 8, 7, 5, 1), # input C=1, only rainmap
                                       nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                       nn.Conv2d(8, 32, 5, 3, 1),
                                       nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                       nn.Conv2d(32, 128, 3, 2, 1), 
                                       nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      )

        self._img_shape = ipshape
        ''' '''
        self.D = Discriminator(self._img_shape, downsample=discriminator_downsample)
        self._adv_w = adv_weight
        self.lr = 1e-04
        self._loss_type = loss_kwargs['type']
        self._ckp_prefix = checkpoint_prefix
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
        print(f'[{self.__class__.__name__} W:{self._adv_w}] Ckp:{os.path.join(self._ckp_dir,self._ckp_prefix)} ')
    
    def forward(self, input, label, addition):
        # input: [B, 6, 2, H, W]; label: [B, 3, H, W]; addition: [B, 6, N_type, H, W]
        output_from_ecd = self.encoder(input) # output: [3][B, n_out, H', W']
        poni_rain = self.make_poni_input(input, label) # [B, 3, H, W]
        pred = self.forecaster(list(output_from_ecd), poni_rain, self.aux, addition) # pred shape: [Seq, batch, H, W]
        return pred
    
    def make_poni_input(self, input, label):
        # rainmap
        y_0 = torch.mean(input[:,:,-1], dim=1, keepdim=True) # y_0: [B, 1, H, W]
        tmp = [label[:, x:x+1] for x in range(0, label.size(1) - 1)]
        
        return torch.cat((y_0, *tmp), dim=1)
            
    def training_step(self, batch, batch_idx, optimizer_idx):
        train_data, train_label, train_mask = batch
        assert train_data.size(2) > 2, f'only radar and rainmap as input! Wrong!'
        train_addition = train_data[:, :, 1:-1]
        train_data = torch.cat([train_data[:, :, 0:1], train_data[:, :, -1:]], dim=2)
        N = train_data.shape[0]
        if optimizer_idx == 0:
            loss_dict = self.generator_loss(train_data, train_label, train_mask, train_addition)
            self._recon_loss.add(loss_dict['progress_bar']['recon_loss'].item() * N, N)
            self._GD_loss.add(loss_dict['progress_bar']['adv_loss'].item() * N, N)
            self._G_loss.add(loss_dict['loss'].item() * N, N)
            self.log('G', self._G_loss.get(), on_step=True, on_epoch=True, prog_bar=True)
            self.log('GRecon', self._recon_loss.get(), on_step=True, on_epoch=True, prog_bar=True)
            self.log('GD', self._GD_loss.get(), on_step=True, on_epoch=True, prog_bar=True)

        # train discriminator
        if optimizer_idx == 1:
            loss_dict = self.discriminator_loss(train_data, train_label, train_addition)
            self._D_loss.add(loss_dict['loss'].item() * N, N)
            self.log('D', self._D_loss.get(), on_step=True, on_epoch=True, prog_bar=True)
            # balance discriminator
            return loss_dict['loss'] * self._adv_w
        return loss_dict['loss']

    def generator_loss(self, input_data, target_label, target_mask, target_addition):
        predicted_reconstruction = self(input_data, target_label, target_addition)
        recons_loss = self._criterion(predicted_reconstruction, target_label, target_mask)
        # train generator
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

    def discriminator_loss(self, input_data, target_label, target_addition):
        predicted_reconstruction = self(input_data, target_label, target_addition)
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
    
    def validation_step(self, batch, batch_idx):
        val_data, val_label, val_mask = batch
        val_addition = val_data[:, :, 1:-1]
        val_data = torch.cat([val_data[:, :, 0:1], val_data[:, :, -1:]], dim=2)
        # generator
        loss_dict = self.generator_loss(val_data, val_label, val_mask, val_addition)
        aligned_prediction = loss_dict['prediction'].permute(1, 0, 2, 3)
        self._val_criterion.compute(aligned_prediction, val_label)

        # Discriminator stats
        pos = self.D(val_label)
        neg = self.D(aligned_prediction)
        self._D_stats.update(neg, pos)

        log_data = loss_dict.pop('progress_bar')
        return {'val_loss': log_data['recon_loss'], 'N': val_label.shape[0]}

class PoniAttenModel_addponi(PoniModel_addponi):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention = TargetAttentionLean_1()
        
    def forward(self, input, label, addition):
        pred = super().forward(input, label, addition) # pred shape: [Seq, batch, H, W]
        # NOTE most recent rain frame
        output = torch.clone(pred)
        output[0] = self.attention(input[:, -1:, -1], pred[0]) # input shape: [batch, Seq, C, H, W]
        return output
        
    def configure_optimizers(self):
        opt_g = torch.optim.Adam(list(self.encoder.parameters()) +
                             list(self.forecaster.parameters()) +
                             list(self.aux.parameters()) + 
                             list(self.attention.parameters()),
                             lr=self.lr
                             )
        opt_d = torch.optim.Adam(self.D.parameters(), lr=self.lr)
        return opt_g, opt_d
