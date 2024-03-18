from datetime import datetime, timedelta
from typing import List, Union

import numpy as np

from src.data_loaders.rain_loader_nc import RainLoaderNc
from src.file_readers.jay_reader import JayReader
from src.file_readers.netcdf_reader import NetcdfReader
from src.utils.crop_util import CropUtil


class RainLoaderJay(RainLoaderNc):
    def __init__(self, **kwarg):
        super(RainLoaderNc, self).__init__(**kwarg)
        if self._lat_range is None:
            self.set_lat_range()
        if self._lon_range is None:
            self.set_lon_range()
        if self._reader == None:
            self.set_reader()

    # TODO: remove the specific path
    def set_lat_range(self):
        file_path = self.get_filename_from_dt(datetime(2016, 1, 1, 1, 0))
        nc_file_path = str(file_path).replace("jay", "nc")
        self._lat_range = NetcdfReader().read(nc_file_path, "lat")

    # TODO: remove the specific path
    def set_lon_range(self):
        file_path = self.get_filename_from_dt(datetime(2016, 1, 1, 1, 0))
        nc_file_path = str(file_path).replace("jay", "nc")
        self._lon_range = NetcdfReader().read(nc_file_path, "lon")

    def set_reader(self):
        self._reader = JayReader(self._lon_range.size, self._lat_range.size)

    def load_data_from_datetime(
        self, dt: Union[List[datetime], datetime]
    ) -> np.ndarray:
        """
        Returns:
            3D array with shape of [ilen, H, W]
        """
        if isinstance(dt, list):
            file_paths = []
            dt_start = dt[0]
            while dt_start <= dt[-1]:
                file_paths.append(self.get_filename_from_dt(dt_start))
                dt_start += timedelta(minutes=self.GRANULARITY)
            return self._reader.read(file_paths)
        else:
            raise RuntimeError("datetime type is not supported.")

    def load_input_data(
        self,
        target_time: datetime,
        ilen: int,
        target_lat: List[float],
        target_lon: List[float],
        target_shape: tuple[int],
    ) -> np.ndarray:
        array_data = self.load_data_from_datetime(
            [target_time - timedelta(minutes=self.GRANULARITY * 5), target_time]
        )
        # handle negative value
        array_data[array_data < 0] = 0
        # handle shape
        array_data = CropUtil.crop_by_coor(
            array_data, self._lat_range, self._lon_range, target_lat, target_lon
        )
        if array_data.shape[-2:] != target_shape:
            const = [int(array_data.shape[-2:][i] / target_shape[i]) for i in range(2)]
            array_data = array_data[..., :: const[0], :: const[1]]
        # normalizatoin
        array_data /= self._BasicLoader__FACTOR

        return array_data[:, None]

    def load_output_data(
        self,
        target_time: datetime,
        olen: int,
        oint: int,
        target_lat: List[float],
        target_lon: List[float],
        target_shape: tuple[int],
    ) -> np.ndarray:
        array_data = self.load_data_from_datetime(
            [
                target_time + timedelta(minutes=self.GRANULARITY * 1),
                target_time + timedelta(minutes=self.GRANULARITY * olen * oint),
            ]
        )
        # handle negative value
        array_data[array_data < 0] = 0
        # handle shape
        array_data = CropUtil.crop_by_coor(
            array_data, self._lat_range, self._lon_range, target_lat, target_lon
        )
        if array_data.shape[-2:] != target_shape:
            const = [int(array_data.shape[-2:][i] / target_shape[i]) for i in range(2)]
            array_data = array_data[..., :: const[0], :: const[1]]
        # convert rain rate to accumulated rainfall
        list_array_data = np.split(array_data, olen, axis=0)
        output_data = [data.mean(axis=0, keepdims=True) for data in list_array_data]
        output_data = np.concatenate(output_data, axis=0)

        # invalid check
        if np.max(output_data) >= self._MAX_RAIN_ACCU:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Invalid quantity when loading {target_time}"
            )

        return output_data
