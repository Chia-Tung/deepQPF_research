from legacy.loss_type import LossType, BlockAggregationMode
from legacy.adversarial_model_PONI_persist import BalAdvPoniAttenModel
from legacy.AdvPONI_hetero_from_poni import PoniModel_addponi, PoniAttenModel_addponi
from legacy.AdvPONI_hetero_from_atten import PoniModel_addatten

import src.model_architectures as ma
from src.model_type import ModelType
from src.model_architectures.utils import *

class ModelBuilder:
    def __init__(self, data_info, **kwarg) -> None:
        model_config = kwarg['model_config']
        self._model_type = ModelType(model_config['name'])
        self._from_input = model_config['add_from_input']
        self._from_poni = model_config['add_from_poni']
        self._from_atten = model_config['add_from_atten']
        self._teach_force = model_config['teach_force']
        self._adv_w = model_config['adv_w']
        self._dis_d = model_config['dis_d']

        self._data_info = data_info
        self._loss_config = kwarg['loss_config']
        self.checkpoint_dir = kwarg['checkpoint_dir']
        # @deprecated
        self.loss_kwarg = {
            'type': int(LossType.WeightedMAE),
            'aggregation_mode': int(BlockAggregationMode.MAX),
            'kernel_size': 5,
            'residual_loss': False,
            'w': float(1)
        }

        # PROPERTY
        self.encoder = None
        self.forecaster = None
        self.loss_fn = None
        self._setup()

    def _setup(self):
        nc = sum(self._data_info['channel'].values())

        if self._model_type == ModelType.CPN:
            encoder_params = get_encoder_params_GRU(nc, self._data_info['shape'])
            decoder_params = get_forecaster_params_GRU()
            self.encoder = ma.Encoder(encoder_params[0], encoder_params[1])
            self.forecaster = ma.Forecaster(
                decoder_params[0],
                decoder_params[1], 
                self._data_info['olen']
            )
        elif self._model_type == ModelType.CPN_PONI:
            if self._from_poni:
                nc_add = sum(
                    [v for k, v in self._data_info['channel'].items() \
                    if k not in ['rain', 'radar']])
                nc -= nc_add
                nc_add += 1 # at least rainmap
            else:
                nc_add = 1 # at least rainmap
            
            encoder_params = get_encoder_params_GRU(nc, self._data_info['shape'])
            decoder_params = get_forecaster_params_GRU()
            self.encoder = ma.Encoder(encoder_params[0], encoder_params[1])
            self.forecaster = ma.ForecasterPONI(
                decoder_params[0],
                decoder_params[1], 
                self._data_info['olen'], 
                self._teach_force, 
                nc_add
            )

        print(f'Using {self._model_type.name} model')
    
    def build(self):
        if self._model_type == ModelType.CPN_PONI:
            model = ma.BalAdvPoniModel(
                self._adv_w,
                self._dis_d,
                self.encoder,
                self.forecaster,
                self._data_info['shape'],
                self._data_info['olen'],
                self.loss_kwarg,
                self.checkpoint_dir,
            )
        return model

    #     if self._model_type == ModelType.BalancedGRUAdverserial:
    #         model = BalAdvModel(
    #             self._adv_w,
    #             self._dis_d,
    #             encoder,
    #             forecaster,
    #             self._data_info['shape'],
    #             self._data_info['olen'],
    #             self.loss_kwarg,
    #             self.checkpoint_dir,
    #         )
    #     elif self._model_type == ModelType.BalancedGRUAdverserialAttention:
    #         model = BalAdvAttentionModel(
    #             self._adv_w,
    #             self._dis_d,
    #             encoder,
    #             forecaster,
    #             self._data_info['shape'],
    #             self._data_info['olen'],
    #             self.loss_kwarg,
    #             self.checkpoint_dir,
    #         )
    #     elif self._model_type == ModelType.BalancedGRUAdvPONI:
    #         model = BalAdvPoniModel(
    #             self._adv_w,
    #             self._dis_d,
    #             encoder,
    #             forecaster,
    #             self._data_info['shape'],
    #             self._data_info['olen'],
    #             self.loss_kwarg,
    #             self.checkpoint_dir,
    #         )
    #     elif self._model_type == ModelType.BalancedGRUAdvPONIAtten:
    #         # assert data_kwargs['data_type'] == DataType.Radar + DataType.Rain
    #         model = BalAdvPoniAttenModel(
    #             self._adv_w,
    #             self._dis_d,
    #             encoder,
    #             forecaster,
    #             self._data_info['shape'],
    #             self._data_info['olen'],
    #             self.loss_kwarg,
    #             self.checkpoint_dir,
    #         )
    #     elif self._model_type == ModelType.BalGRUAdvPONI_addponi:
    #         model = PoniModel_addponi(
    #             self._adv_w,
    #             self._dis_d,
    #             encoder,
    #             forecaster,
    #             self._data_info['shape'],
    #             self._data_info['olen'],
    #             self.loss_kwarg,
    #             self.checkpoint_dir,
    #             sum(self._data_info['channel'].values()) - nc,
    #         )
    #     elif self._model_type == ModelType.BalGRUAdvPONIAtten_addponi:
    #         model = PoniAttenModel_addponi(
    #             self._adv_w,
    #             self._dis_d,
    #             encoder,
    #             forecaster,
    #             self._data_info['shape'],
    #             self._data_info['olen'],
    #             self.loss_kwarg,
    #             self.checkpoint_dir,
    #             sum(self._data_info['channel'].values()) - nc,
    #         )
    #     elif self._model_type == ModelType.BalGRUAdvPONI_addatten:
    #         model = PoniModel_addatten(
    #             self._adv_w,
    #             self._dis_d,
    #             encoder,
    #             forecaster,
    #             self._data_info['shape'],
    #             self._data_info['olen'],
    #             self.loss_kwarg,
    #             self.checkpoint_dir,
    #             sum(self._data_info['channel'].values()) - nc,
    #         )

    #     return model