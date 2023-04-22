import numpy as np
from datetime import datetime, timedelta
from typing import List, Union

from src.file_readers.jay_reader import JayReader
from src.file_readers.netcdf_reader import NetcdfReader
from src.data_loaders.rain_loader import RainLoader
from src.utils.crop_util import CropUtil


class RainLoaderJay(RainLoader):
    def __init__(self, **kwarg):
        super().__init__(**kwarg)

    # TODO: the specific path should be removed
    def set_lat_range(self):
        file_path = self._BasicLoader__all_files[0]
        nc_file_path = file_path.replace('jay', 'nc')
        self._lat_range = NetcdfReader().read(nc_file_path, 'lat')

    def set_lon_range(self):
        file_path = self._BasicLoader__all_files[0]
        nc_file_path = file_path.replace('jay', 'nc')
        self._lon_range = NetcdfReader().read(nc_file_path, 'lon')

    def set_reader(self):
        self._reader = JayReader(
            self._lon_range.size, self._lat_range.size
        )

    def load_data_from_datetime(self, dt: Union[List[datetime], datetime]) -> np.ndarray:
        """
        Returns:
            3D array with shape of [B, H, W]
        """
        if isinstance(dt, list):
            idx_first = self.time_list.index(dt[0])
            idx_last = self.time_list.index(dt[-1])
            file_paths = self._BasicLoader__all_files[idx_first:idx_last+1]
            return self._reader.read(file_paths)
        else:
            raise RuntimeError("datetime type is not supported.")

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
            data (np.ndarray): Rain rate which has a shape of [ilen, CH, H, W].
        """
        array_data = self.load_data_from_datetime(
            [target_time - timedelta(minutes=self.GRANULARITY * 5), target_time]
        )
        # handle negative value
        array_data[array_data < 0] = 0
        # handle shape
        array_data = CropUtil.crop_by_coor(
            array_data, self._lat_range, self._lon_range, target_lat, target_lon
        )
        # normalizatoin
        array_data /= self._BasicLoader__FACTOR
        return array_data[:, None]
    
    def load_output_data(
        self, 
        target_time: datetime, 
        olen: int, 
        oint: int, 
        target_lat: List[float], 
        target_lon: List[float]
    ) -> np.ndarray:
        """
        Args:
            target_time (datetime): The start time of a predicted event.
            olen (int): Output length.
            oint (int): Output interval. Granularity * interval = 1 hour.
            target_lat (List[float]): Target latitude to crop.
            target_lon (List[float]): Target longitude to crop.

        Returns: 
            output_data (np.ndarray): Hourly accumulated rainfall. The shape 
                is [olen, H, W].
        """
        array_data = self.load_data_from_datetime([
            target_time + timedelta(minutes=self.GRANULARITY * 1), 
            target_time + timedelta(minutes=self.GRANULARITY * olen * oint)
            ])
        # handle negative value
        array_data[array_data < 0] = 0
        # handle shape
        array_data = CropUtil.crop_by_coor(
            array_data, self._lat_range, self._lon_range, target_lat, target_lon
        )
        # convert rain rate to accumulated rainfall
        list_array_data = np.split(array_data, olen, axis=0)
        output_data = [data.mean(axis=0, keepdims=True) for data in list_array_data]
        output_data = np.concatenate(output_data, axis=0)

        # invalid check
        if np.max(output_data) >= self._MAX_RAIN_ACCU:
            raise RuntimeError (
                f"[{self.__class__.__name__}] Invalid quantity when loading {target_time}")
        
        return output_data