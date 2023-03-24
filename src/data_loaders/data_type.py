from enum import Enum, auto
from typing import List

class DataType(Enum):
    RAIN = auto()
    RADAR = auto()
    WIND = auto()
    MONTH = auto()
    ELEVATION = auto()
    THETAE = auto()
    LIFTING = auto()

    @classmethod
    def data_type_mapping(cls, dtypes: List[str]):
        container = []
        for dtype in dtypes:
            if dtype == 'rain':
                container.append(cls.RAIN)
            elif  dtype == 'radar':
                container.append(cls.RADAR)
        return container


    """"
    @classmethod
    def all_data(cls):
        output = 0
        for key, value in cls.__dict__.items():
            if not isinstance(value, int):
                continue
            output += value
        return output

    @classmethod
    def count(cls, dtype):
        return cls.count2D(dtype) + cls.count1D(dtype)

    @classmethod
    def count1D(cls, dtype):
        dt = int(dtype & cls.MONTH == cls.MONTH) * 2 # (sin, cos)
        wind = int(dtype & cls.WIND == cls.WIND) * 2
        theta_e = int(dtype & cls.THETAE == cls.THETAE)
        return dt + wind + theta_e

    @classmethod
    def count2D(cls, dtype):
        rain = int(dtype & cls.RAIN == cls.RAIN)
        radar = int(dtype & cls.RADAR == cls.RADAR)
        elevation = int(dtype & cls.ELEVATION == cls.ELEVATION)
        wt = int(dtype & cls.WTDOT == cls.WTDOT)
        return rain + radar + elevation + wt
    
    @classmethod
    def countMinus(cls, dtype):
        # this function is for add_from_poni
        dt = int(dtype & cls.MONTH == cls.MONTH) * 2 #(sin, cos)
        wt = int(dtype & cls.WTDOT == cls.WTDOT)
        elevation = int(dtype & cls.ELEVATION == cls.ELEVATION)
        wind = int(dtype & cls.WIND == cls.WIND) * 2
        theta_e = int(dtype & cls.THETAE == cls.THETAE)
        return dt + wt + elevation + wind + theta_e
    
    @classmethod
    def print(cls, dtype, prefix=''):
        for key, value in cls.__dict__.items():
            if value == cls.NONEATALL or not isinstance(value, int):
                continue
            if (dtype & value) == value:
                print(f'[{prefix} Dtype]', key)

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
    """