from src.model_type import ModelType
from src.model_architectures.builders.balanced_gru_adv_poni_builder import BalancedGRUAdvPoniBuilder

class ModelDirector:
    def __init__(self, data_info, **kwarg) -> None:
        # ARGUMENT
        self._model_config = kwarg['model_config']
        self._loss_config = kwarg['loss_config']
        self._checkpoint_dir = kwarg['checkpoint_dir']
        self._data_info = data_info
        self._model_type = ModelType(self._model_config['name'])

        # PROPERTY
        self.model_builder = None
        self._setup()

    def _setup(self):
        if self._model_type == ModelType.CPN:
            self.model_builder = None
        elif self._model_type == ModelType.CPN_PONI:
            self.model_builder = BalancedGRUAdvPoniBuilder(
                self._model_config,
                self._loss_config,
                self._data_info,
                self._checkpoint_dir
            )

        print(f'Using {self._model_type.name} model')
    
    def build_model(self):
        if self.model_builder is not None:
            return self.model_builder.build()
        else:
            raise RuntimeError ("model builder is NONE.")
