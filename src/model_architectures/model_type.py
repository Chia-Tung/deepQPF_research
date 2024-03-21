from enum import Enum


class StrEnum(str, Enum):
    pass


class ModelType(StrEnum):
    CAPN = "BalancedGRUAdv"
    CAPN_PONI = "BalancedGRUAdvPONI"
    CAPN_PONI_PERSIST = "BalancedGRUAdvPONIAtten"
    TRANSFORMER = "Transformer"
