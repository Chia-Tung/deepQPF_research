import numpy as np
from typing import List
from pathlib import Path
from datetime import datetime, timedelta

from src.data_loaders.basic_loader import BasicLoader
from src.file_readers.netcdf_reader import NetcdfReader
from src.utils.crop_util import CropUtil

class RainLoaderNc(BasicLoader):
    GRANULARITY = 10 # 10-min
    _MAX_RAIN_ACCU = 250 # mm

    def __init__(self, **kwarg):
        super().__init__(**kwarg)
        if self._reader == None:
            self.set_reader()
        if self._lat_range is None:
            self.set_lat_range()
        if self._lon_range is None:
            self.set_lon_range()

    def inp_initial_time_fn(self, dt_list: list[datetime], ilen: int) -> list[datetime]:
        output_time_list = []
        input_offset = ilen - 1
        for idx, dt in reversed(list(enumerate(dt_list))):
            if idx - input_offset < 0:
                break

            if (dt - dt_list[idx - input_offset] == 
                timedelta(minutes=input_offset * self.GRANULARITY)):
                output_time_list.append(dt)

        return output_time_list
    
    def oup_initial_time_fn(self, dt_list: list[datetime], olen: int, oint: int) -> list[datetime]:
        time_map = []
        raw_idx = 0
        target_offset = olen * oint
        while raw_idx < len(dt_list):
            if raw_idx + target_offset >= len(dt_list):
                break

            if (dt_list[raw_idx + target_offset] - dt_list[raw_idx]
                == timedelta(minutes = target_offset * self.GRANULARITY)):
                time_map.append(dt_list[raw_idx])

            raw_idx += 1
        return time_map 
    
    def load_input_data(
        self, 
        target_time: datetime, 
        ilen: int,
        target_lat: List[float],
        target_lon: List[float]
    ) -> np.ndarray:
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
    
    def load_output_data(
        self, 
        target_time: datetime, 
        olen: int, 
        oint: int, 
        target_lat: List[float], 
        target_lon: List[float]
    ) -> np.ndarray:
        data = []
        for time_offset in (
            timedelta(minutes=self.GRANULARITY * i) for i in range(1, olen * oint + 1)):
            future_time = target_time + time_offset
            array_data = self.load_data_from_datetime(future_time)
            # handle negative value
            array_data[array_data < 0] = 0
            # handle shape
            array_data = CropUtil.crop_by_coor(
                array_data, self._lat_range, self._lon_range, target_lat, target_lon
            )
            # expand dimension
            array_data = array_data[None]
            data.append(array_data)

            # convert rain rate to accumulated rainfall
            if (time_offset / timedelta(hours=1)) % 1 == 0:
                new_data = np.concatenate(data[-oint:], axis=0).mean(axis=0)
                del data[-oint:]
                data.append(new_data)
        data = np.stack(data, axis=0)

        ### Why don't we have a shape check?
        # Becasue the shape of rain rate date is [1, H', W'] while the shape 
        # of accumulated rain fall is [H', W']. If both data exist in `data`
        # the `stack` function will throw error messages.

        # invalid check
        if np.max(data) >= self._MAX_RAIN_ACCU:
            raise RuntimeError (
                f"[{self.__class__.__name__}] Invalid quantity when loading {target_time}")
        
        return data
    
    def load_data_from_datetime(self, dt: datetime) -> np.ndarray:
        file_path = self.get_filename_from_dt(dt)
        return self._reader.read(file_path, 'qperr')
    
    # TODO: more elegant way to get lat range
    def set_lat_range(self):
        file_path = self.get_filename_from_dt(datetime(2016, 1, 1, 1, 0))
        self._lat_range = self._reader.read(file_path, 'lat')
        assert np.sum(np.isin(self._lat_range, self._reader.INVALID_VALUE)) == 0, \
            f"Latitude range has invalid value in {file_path}"

    # TODO: more elegant way to get lon range
    def set_lon_range(self):
        file_path = self.get_filename_from_dt(datetime(2016, 1, 1, 1, 0))
        self._lon_range = self._reader.read(file_path, 'lon')
        assert np.sum(np.isin(self._lon_range, self._reader.INVALID_VALUE)) == 0, \
            f"Longitude range has invalid value in {file_path}"

    def set_reader(self):
        self._reader = NetcdfReader()

    def get_filename_from_dt(self, dt: datetime) -> Path:
        basename = dt.strftime(self.formatter)
        return Path(*[
            self.path, 
            f"{dt.year:04d}", 
            f"{dt.year:04d}{dt.month:02d}", 
            f"{dt.year:04d}{dt.month:02d}{dt.day:02d}",
            basename
        ])
