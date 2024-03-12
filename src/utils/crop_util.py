from typing import List

import numpy as np


class CropUtil:
    @staticmethod
    def crop_by_coor(
        data: np.ndarray,
        data_lat: np.ndarray,
        data_lon: np.ndarray,
        target_lat: List[float],
        target_lon: List[float],
    ) -> np.ndarray:
        """
        Args:
            data (np.ndarray): The input data which is a 2d/3d np.ndarray with
                corresponding coordiante `data_lat` and `data_lon`.
            data_lat (np.ndarray): Latitude array for `data`.
            data_lon (np.ndarray): Longitude array for `data`.
            target_lat (List[float]): Target latitude to crop.
            target_lon (List[float]): Target longitude to crop.

        Returns:
            data (np.ndarray): Shape is [H', W']
        """
        if (
            (target_lat[0] not in data_lat)
            | (target_lat[1] not in data_lat)
            | (target_lon[0] not in data_lon)
            | (target_lon[1] not in data_lon)
        ):
            raise RuntimeError(f"Invalid target shape.")

        iloc = []
        iloc.extend(list(map(lambda x: np.where(data_lat == x)[0][0], target_lat)))
        iloc.extend(list(map(lambda x: np.where(data_lon == x)[0][0], target_lon)))
        if len(data.shape) == 2:
            return data[iloc[0] : iloc[1] + 1, iloc[2] : iloc[3] + 1]
        elif len(data.shape) == 3:
            return data[:, iloc[0] : iloc[1] + 1, iloc[2] : iloc[3] + 1]
