import src.model_architectures as ma
from src.model_architectures.loss_fn.loss_type import LossType, BlockAggregationMode
from src.model_architectures.utils import *

class BalancedGRUAdvPoniBuilder:
    def __init__(
        self,
        model_config,
        loss_config,
        data_info,
        checkpoint_dir,
    ):
        self._teach_force = model_config['teach_force']
        self._adv_w = model_config['adv_w']
        self._dis_d = model_config['dis_d']
        self._loss_config = loss_config
        self._data_info = data_info
        self._checkpoint_dir = checkpoint_dir

        # @deprecated
        self.loss_kwarg = {
            'type': int(LossType.WeightedMAE),
            'aggregation_mode': int(BlockAggregationMode.MAX),
            'kernel_size': 5,
            'residual_loss': False,
            'w': float(1)
        }

        # heterogeneous data import
        self._from_poni = model_config['add_from_poni']

        # framework
        self._framework = ma.GANFramework

        # prepare the model components
        self.prepare_components()

    def prepare_components(self):
        """ prepare encoder and forecaster"""
        num_channel = sum(self._data_info['channel'].values())

        if self._from_poni:
            num_channel_add = sum(
                [v for k, v in self._data_info['channel'].items() \
                if k not in ['rain', 'radar']]
            )
            num_channel -= num_channel_add
            num_channel_add += 1 # at least rainmap
        else:
            num_channel_add = 1 # at least rainmap

        encoder_params = get_encoder_params_GRU(num_channel, self._data_info['shape'])
        decoder_params = get_forecaster_params_GRU()
        aux_encoder_params = get_aux_encoder_params(num_channel_add)
        self.encoder = ma.Encoder(encoder_params[0], encoder_params[1])
        self.forecaster = ma.ForecasterPONI(
            decoder_params[0],
            decoder_params[1], 
            self._data_info['olen'], 
            self._teach_force, 
            aux_encoder_params
        )

    def build(self):
        return self._framework(
            self._adv_w,
            self._dis_d,
            self.encoder,
            self.forecaster,
            self._data_info['shape'],
            self._data_info['olen'],
            self.loss_kwarg,
            self._checkpoint_dir,
            self._from_poni
        )