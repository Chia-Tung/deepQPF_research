from __future__ import annotations
import src.model_architectures as ma
from src.model_architectures.loss_fn.loss_type import LossType
from src.model_architectures.utils import *

class BalancedGRUAdvPoniBuilder:
    def __init__(
        self,
        model_config,
        loss_config,
        data_info,
        checkpoint_dir,
    ):
        """
        Balanced: Ignore the loss from pixels below a certain threshold (0.5)
        GRU: the encoder/decoder type
        Adv: GAN framework
        Poni: Accessory adopted from Seq2Seq model
        """
        # model config
        self._from_poni = model_config['add_hetr_from_poni']
        self._tf = model_config['teach_forcing_ratio']
        self._dis_d = model_config['discriminator_downsample']
        # loss config
        self._loss_type = loss_config['type']
        self._bal_w = loss_config['balance_weight']
        self._bal_thsh = loss_config['balance_threshold']
        self._adv_w = loss_config['adversarial_weight']
        self._lr = loss_config['learning_rate']
        # others
        self._data_info = data_info
        self._checkpoint_dir = checkpoint_dir

        # framework
        self._framework = ma.GANFramework
        self._encoder = None
        self._forecaster = None
        self._discriminator = None
        self._loss_fn = None
        self._dis_loss_fn = None

        # prepare the model components
        self.prepare_components()

    def prepare_components(self):
        num_channel = sum(self._data_info['channel'].values())
        num_channel_add = 1 # at least rainmap

        if self._from_poni:
            for k, v in self._data_info['channel'].items():
                if k not in ['rain', 'radar']:
                    num_channel_add += v
            num_channel -= num_channel_add - 1 # add rain back

        self.prepare_encoder(num_channel, self._data_info['shape'])\
            .prepare_decoder(num_channel_add)\
            .prepare_discriminator(self._data_info['shape'])\
            .prepare_loss_fn(self._loss_type)\
            .prepare_dis_loss_fn()
    
    def prepare_encoder(
        self, 
        n_channel: int, 
        shape: tuple[int]
    ) -> BalancedGRUAdvPoniBuilder:
        encoder_params = get_encoder_params_GRU(n_channel, shape)
        self._encoder = ma.Encoder(encoder_params[0], encoder_params[1])
        return self
    
    def prepare_decoder(
        self,
        n_channel: int
    ) -> BalancedGRUAdvPoniBuilder:
        aux_encoder_params = get_aux_encoder_params(n_channel)
        decoder_params = get_forecaster_params_GRU()
        self._forecaster = ma.ForecasterPONI(
            decoder_params[0],
            decoder_params[1], 
            self._data_info['olen'], 
            self._tf, 
            aux_encoder_params
        )
        return self
    
    def prepare_discriminator(
        self, 
        shape: tuple[int]
    ) -> BalancedGRUAdvPoniBuilder:
        self._discriminator = ma.Discriminator(shape, downsample=self._dis_d)
        return self
    
    def prepare_loss_fn(
        self, 
        loss_type: str
    ) -> BalancedGRUAdvPoniBuilder:
        self._loss_fn = LossType[loss_type].value(self._bal_w, threshold=self._bal_thsh)
        return self
    
    def prepare_dis_loss_fn(self) -> BalancedGRUAdvPoniBuilder:
        self._dis_loss_fn = nn.BCELoss()
        return self
    
    def build(self):
        return self._framework(
            # models
            self._encoder,
            self._forecaster,
            self._discriminator,
            self._loss_fn,
            self._dis_loss_fn,
            # scalars
            self._from_poni,
            self._lr,
            self._data_info['olen'],
            self._adv_w,
            # checkpoint
            self._checkpoint_dir
        )