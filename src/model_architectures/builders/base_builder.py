import abc

from pytorch_lightning import LightningModule


class BaseBuilder(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def build() -> LightningModule:
        return NotImplemented
    
    def handle_model_config(self, model_config):
        """
        Handles the model configuration by converting string of values to floats.

        Parameters:
            model_config (dict): A dictionary containing the model configuration.

        Returns:
            dict: The updated model configuration dictionary.
        """
        for k, v in model_config.items():
            if k in ["learning_rate", "adam_epsilon"]:
                model_config[k] = float(v)
        return model_config
