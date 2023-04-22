import numpy as np
from datetime import datetime, timedelta
from typing import List, Union

from src.data_loaders.radar_loader import RadarLoader
from src.file_readers.netcdf_reader import NetcdfReader
from src.file_readers.jay_reader import JayReader
from src.utils.crop_util import CropUtil

class RadarLoaderJay(RadarLoader):
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