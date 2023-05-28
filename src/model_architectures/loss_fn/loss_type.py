class LossType:
    WeightedMAE = 0
    BlockWeightedMAE = 1
    BlockWeightedAvgMAE = 2
    BlockWeightedMAEDiversify = 3
    WeightedAbsoluteMAE = 4
    MAE = 5
    ReluWeightedMaeLoss = 6
    BalancedWeightedMaeLoss = 7
    WeightedMaeLossDiversify = 8
    WeightedMAEWithBuffer = 9
    KernelWeightedMAE = 10
    WeightedMAEandMSE = 11
    ClassificationBCE = 12
    SSIMBasedLoss = 13
    NormalizedSSIMBasedLoss = 14

    @classmethod
    def name(cls, enum_type):
        for key, value in cls.__dict__.items():
            if enum_type == value:
                return key

    @classmethod
    def from_name(cls, enum_type_str):
        for key, value in cls.__dict__.items():
            if key == enum_type_str:
                return value
        assert f'{cls.__name__}:{enum_type_str} doesnot exist.'


class BlockAggregationMode:
    MAX = 0
    MEAN = 1

    @classmethod
    def name(cls, enum_type):
        for key, value in cls.__dict__.items():
            if enum_type == value:
                return key

    @classmethod
    def from_name(cls, enum_type_str):
        for key, value in cls.__dict__.items():
            if key == enum_type_str:
                return value
        assert f'{cls.__name__}:{enum_type_str} doesnot exist.'
