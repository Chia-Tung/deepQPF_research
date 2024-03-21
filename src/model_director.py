from src.model_architectures import BalancedGRUAdvPoniBuilder, TransformerBuilder
from src.model_architectures.builders.base_builder import BaseBuilder
from src.model_architectures.model_type import ModelType


class ModelDirector:
    def __init__(self, data_info, **kwarg) -> None:
        # from argument
        self._model_config = kwarg["model_config"]
        self._loss_config = kwarg["loss_config"]
        self._checkpoint_dir = kwarg["checkpoint_dir"]
        self._data_info = data_info
        self._model_type = ModelType(self._model_config["name"])

        # from late init
        self.model_builder: BaseBuilder = None
        self._setup()

    def _setup(self):
        match self._model_type:
            case ModelType.CAPN:
                pass
            case ModelType.CAPN_PONI:
                self.model_builder = BalancedGRUAdvPoniBuilder(
                    self._model_config,
                    self._loss_config,
                    self._data_info,
                    self._checkpoint_dir,
                )
            case ModelType.CAPN_PONI_PERSIST:
                pass
            case ModelType.TRANSFORMER:
                self.model_builder = TransformerBuilder(
                    self._model_config,
                    self._loss_config,
                    self._data_info,
                    self._checkpoint_dir,
                )
            case _:
                print(f"{self._model_type} not supported")

        print(f"[{self.__class__.__name__}] Using {self._model_type.name} model")

    def build_model(self):
        if self.model_builder is not None:
            return self.model_builder.build()
        else:
            raise RuntimeError("model builder is NONE.")
