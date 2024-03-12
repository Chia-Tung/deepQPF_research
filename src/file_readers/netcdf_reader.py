import netCDF4 as nc
import numpy as np

from src.file_readers.basic_reader import BasicReader


class NetcdfReader(BasicReader):
    INVALID_VALUE = -999.0

    def __init__(self):
        pass

    def read(self, filename: str, variable_name: str) -> np.ndarray:
        self.check_file_exist(filename)
        # load data
        mask_data = nc.Dataset(filename)[variable_name][:]
        # handle ma.MaskArray
        mask_data[mask_data.mask != 0] = self.INVALID_VALUE
        # convert to np.ndarray
        array_data = np.array(mask_data)
        return array_data

    def show_keys(self, filename: str):
        self.check_file_exist(filename)
        print(nc.Dataset(filename).variables.keys())
