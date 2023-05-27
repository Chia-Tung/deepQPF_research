from legacy.loss_type import LossType, BlockAggregationMode
from legacy.adversarial_model_PONI import BalAdvPoniModel
from legacy.adversarial_model_PONI_persist import BalAdvPoniAttenModel
from legacy.AdvPONI_hetero_from_poni import PoniModel_addponi, PoniAttenModel_addponi
from legacy.AdvPONI_hetero_from_atten import PoniModel_addatten

import src.model_architectures as ma
from src.model_type import ModelType
from src.model_architectures.utils import *

class ModelBuilder:
    def __init__(self, data_info, **kwarg) -> None:
        # INPUT
        self._data_info = data_info
        self._model_type = ModelType(kwarg['model_name'])
        self._loss_config = kwarg['loss_config']
        self._teach_force = kwarg['teach_force']
        self._adv_w = kwarg['adv_w']
        self._dis_d = kwarg['dis_d']
        self.checkpoint_dir = kwarg['checkpoint_dir']
        # deprecated
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

        # setup
        self._setup()

        print(f'Using {self._model_type.name} model')

    def _setup(self):
        # encoder decoder loss_fn
        if self._model_type == ModelType.CPN:
            nc = sum([v for k, v in self._data_info['channel'].items() if k in ['rain', 'radar']])
            encoder_params = get_encoder_params_GRU(nc, self._data_info['shape'])
            decoder_params = get_forecaster_params_GRU()
            self.encoder = ma.Encoder(encoder_params[0], encoder_params[1])
            self.forecaster = ma.Forecaster(decoder_params[0],decoder_params[1], 
                self._data_info['olen'])
        elif self._model_type == ModelType.CPN_PONI:
            nc = sum([v for k, v in self._data_info['channel'].items() if k in ['rain', 'radar']])
            encoder_params = get_encoder_params_GRU(nc, self._data_info['shape'])
            decoder_params = get_forecaster_params_GRU()
            self.encoder = ma.Encoder(encoder_params[0], encoder_params[1])
            self.forecaster = ma.ForecasterPONI(decoder_params[0],decoder_params[1], 
                self._data_info['olen'], self._teach_force)
        # count nc has not been implemented
        
        # if self._model_type in [ModelType.BalancedGRUAdvPONI, ModelType.BalancedGRUAdvPONIAtten,]:
        #     forecaster = Forecaster_PONI(forecaster_params_GRU[0], forecaster_params_GRU[1], 
        #         self._data_info['olen'], self._teach_force)
        # elif self._model_type in [ModelType.BalGRUAdvPONI_addponi, ModelType.BalGRUAdvPONIAtten_addponi,]:
        #     nc -= sum([v for k, v in self._data_info['channel'].items() if k not in ['rain', 'radar']])
        #     forecaster = Forecaster_addPONI(forecaster_params_GRU[0],forecaster_params_GRU[1],
        #         self._data_info['olen'], self._teach_force)
        # elif self._model_type in [ModelType.BalGRUAdvPONI_addatten,]:
        #     nc -= sum([v for k, v in self._data_info['channel'].items() if k not in ['rain', 'radar']])
        #     forecaster = Forecaster_PONI(forecaster_params_GRU[0],forecaster_params_GRU[1], 
        #         self._data_info['olen'], self._teach_force)
        # else:
        #     forecaster = Forecaster(forecaster_params_GRU[0],forecaster_params_GRU[1], self._data_info['olen'])
        encoder_params = get_encoder_params_GRU(nc, self._data_info['shape'])
        encoder = Encoder(encoder_params[0], encoder_params[1])

        if self._model_type == ModelType.BalancedGRUAdverserial:
            model = BalAdvModel(
                self._adv_w,
                self._dis_d,
                encoder,
                forecaster,
                self._data_info['shape'],
                self._data_info['olen'],
                self.loss_kwarg,
                self.checkpoint_dir,
            )
        elif self._model_type == ModelType.BalancedGRUAdverserialAttention:
            model = BalAdvAttentionModel(
                self._adv_w,
                self._dis_d,
                encoder,
                forecaster,
                self._data_info['shape'],
                self._data_info['olen'],
                self.loss_kwarg,
                self.checkpoint_dir,
            )
        elif self._model_type == ModelType.BalancedGRUAdvPONI:
            model = BalAdvPoniModel(
                self._adv_w,
                self._dis_d,
                encoder,
                forecaster,
                self._data_info['shape'],
                self._data_info['olen'],
                self.loss_kwarg,
                self.checkpoint_dir,
            )
        elif self._model_type == ModelType.BalancedGRUAdvPONIAtten:
            # assert data_kwargs['data_type'] == DataType.Radar + DataType.Rain
            model = BalAdvPoniAttenModel(
                self._adv_w,
                self._dis_d,
                encoder,
                forecaster,
                self._data_info['shape'],
                self._data_info['olen'],
                self.loss_kwarg,
                self.checkpoint_dir,
            )
        elif self._model_type == ModelType.BalGRUAdvPONI_addponi:
            model = PoniModel_addponi(
                self._adv_w,
                self._dis_d,
                encoder,
                forecaster,
                self._data_info['shape'],
                self._data_info['olen'],
                self.loss_kwarg,
                self.checkpoint_dir,
                sum(self._data_info['channel'].values()) - nc,
            )
        elif self._model_type == ModelType.BalGRUAdvPONIAtten_addponi:
            model = PoniAttenModel_addponi(
                self._adv_w,
                self._dis_d,
                encoder,
                forecaster,
                self._data_info['shape'],
                self._data_info['olen'],
                self.loss_kwarg,
                self.checkpoint_dir,
                sum(self._data_info['channel'].values()) - nc,
            )
        elif self._model_type == ModelType.BalGRUAdvPONI_addatten:
            model = PoniModel_addatten(
                self._adv_w,
                self._dis_d,
                encoder,
                forecaster,
                self._data_info['shape'],
                self._data_info['olen'],
                self.loss_kwarg,
                self.checkpoint_dir,
                sum(self._data_info['channel'].values()) - nc,
            )

        return model