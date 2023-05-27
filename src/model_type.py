from enum import Enum

class StrEnum(str, Enum):
    pass

class ModelType(StrEnum):
    CPN = 'BalancedGRUAdv'
    CPN_PONI = 'BalancedGRUAdvPONI'