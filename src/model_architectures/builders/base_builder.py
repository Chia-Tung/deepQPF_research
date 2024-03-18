import abc

from pytorch_lightning import LightningModule


class BaseBuilder(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def build() -> LightningModule:
        return NotImplemented
