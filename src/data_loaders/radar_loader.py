import numpy as np
from typing import List
from datetime import datetime, timedelta

from src.data_loaders.basic_loader import BasicLoader
from src.file_readers.netcdf_reader import NetcdfReader
from src.utils.crop_util import CropUtil

class RadarLoader(BasicLoader):
    GRANULARITY = 10 # 10-min

    def __init__(self, **kwarg):
        super().__init__(**kwarg)
        if self._lat_range is None:
            self.set_lat_range()
        if self._lon_range is None:
            self.set_lon_range()
        if self._reader == None:
            self.set_reader()

    def cross_check_start_time(
        self, 
        original_time_list: List[datetime], 
        ilen: int
    ) -> List[datetime]:
        output_time_list = []
        input_offset = ilen - 1
        for idx, dt in reversed(list(enumerate(self.time_list))):
            if idx - input_offset < 0:
                break

            if (dt - self.time_list[idx - input_offset] == 
                timedelta(minutes=input_offset * self.GRANULARITY)):
                output_time_list.append(dt)
        
        return list(set(output_time_list).intersection(set(original_time_list)))

    def load_input_data(
        self, 
        target_time: datetime, 
        ilen: int,
        target_lat: List[float],
        target_lon: List[float]
    ) -> np.ndarray:
        """
        Args:
            target_time (datetime): The start time of a predicted event.
            ilen (int): Input length.
            target_lat (List[float]): Target latitude to crop.
            target_lon (List[float]): Target longitude to crop.

        Returns: 
            data (np.ndarray): shape is [ilen, CH, H, W]
        """
        data = []
        for time_offset in (timedelta(minutes=self.GRANULARITY * i) for i in range(ilen-1, -1, -1)):
            past_time = target_time - time_offset
            array_data = self.load_data_from_datetime(past_time)
            # handle negative value
            array_data[array_data < 0] = 0
            # handle shape
            array_data = CropUtil.crop_by_coor(
                array_data, self._lat_range, self._lon_range, target_lat, target_lon
            )
            # expand dimension
            array_data = array_data[None]
            # normalizatoin
            array_data /= self._BasicLoader__FACTOR

            data.append(array_data)
        return np.concatenate(data, axis=0)[:, None]
    
    def load_output_data(self, target_time, olen, oint, target_lat, target_lon) -> np.ndarray:
        pass
    
    # NOTE: There is no validation check for this function.
    def load_data_from_datetime(self, dt: datetime) -> np.ndarray:
        file_path = self._BasicLoader__all_files[self.time_list.index(dt)]
        return self._reader.read(file_path, 'cv')
    
    def set_lat_range(self):
        file_path = self._BasicLoader__all_files[0]
        self._lat_range = self._reader.read(file_path, 'lat')
        assert np.sum(np.isin(self._lat_range, self._reader.INVALID_VALUE)) == 0, \
            f"Latitude range has invalid value in {file_path}"

    def set_lon_range(self):
        file_path = self._BasicLoader__all_files[0]
        self._lon_range = self._reader.read(file_path, 'lon')
        assert np.sum(np.isin(self._lon_range, self._reader.INVALID_VALUE)) == 0, \
            f"Longitude range has invalid value in {file_path}"

    def set_reader(self):
        self._reader = NetcdfReader()