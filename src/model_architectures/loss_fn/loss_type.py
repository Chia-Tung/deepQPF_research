from enum import Enum

from src.model_architectures.loss_fn.weighted_mae_loss import WeightedMaeLoss


class LossType(Enum):
    WeightedMAE = WeightedMaeLoss
