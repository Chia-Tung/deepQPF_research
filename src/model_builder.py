from legacy.loss_type import LossType, BlockAggregationMode
from legacy.model_type import ModelType
from legacy.encoder import Encoder
from legacy.forecaster import Forecaster, Forecaster_PONI, Forecaster_addPONI
from legacy.simpleGRU_model_params import forecaster_params_GRU, get_encoder_params_GRU
from legacy.adversarial_model_PONI import BalAdvPoniModel
from legacy.adversarial_model_PONI_persist import BalAdvPoniAttenModel
from legacy.AdvPONI_hetero_from_poni import PoniModel_addponi, PoniAttenModel_addponi
from legacy.AdvPONI_hetero_from_atten import PoniModel_addatten

class ModelBuilder:
    def __init__(self, **kwarg) -> None:
        self._model_type = ModelType.from_name(kwarg['name'])
        self._data_info = kwarg['data_info']
        self._loss_config = kwarg['loss_config']
        self._teach_force = kwarg['teach_force']
        self._adv_w = kwarg['adv_w']
        self._dis_d = kwarg['dis_d']
        self.checkpoint_dir = kwarg['checkpoint_dir']
        self.loss_kwarg = {
            'type': int(LossType.WeightedMAE),
            'aggregation_mode': int(BlockAggregationMode.MAX),
            'kernel_size': 5,
            'residual_loss': False,
            'w': float(1)
        }

        print(f'Using {ModelType.name(self._model_type)} model')

    def build(self):
        # count nc has not been implemented
        nc = sum(self._data_info['channel'].values())
        if self._model_type in [ModelType.BalancedGRUAdvPONI, ModelType.BalancedGRUAdvPONIAtten,]:
            forecaster = Forecaster_PONI(forecaster_params_GRU[0], forecaster_params_GRU[1], 
                self._data_info['olen'], self._teach_force)
        elif self._model_type in [ModelType.BalGRUAdvPONI_addponi, ModelType.BalGRUAdvPONIAtten_addponi,]:
            nc -= sum([v for k, v in self._data_info['channel'].items() if k not in ['rain', 'radar']])
            forecaster = Forecaster_addPONI(forecaster_params_GRU[0],forecaster_params_GRU[1],
                self._data_info['olen'], self._teach_force)
        elif self._model_type in [ModelType.BalGRUAdvPONI_addatten,]:
            nc -= sum([v for k, v in self._data_info['channel'].items() if k not in ['rain', 'radar']])
            forecaster = Forecaster_PONI(forecaster_params_GRU[0],forecaster_params_GRU[1], 
                self._data_info['olen'], self._teach_force)
        else:
            forecaster = Forecaster(forecaster_params_GRU[0],forecaster_params_GRU[1], self._data_info['olen'])
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