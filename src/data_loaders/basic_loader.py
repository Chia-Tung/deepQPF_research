import abc
from datetime import datetime

import numpy as np


class BasicLoader(metaclass=abc.ABCMeta):

    def __init__(
        self,
        path: str,
        formatter: str,
        normalize_factor: int = 1,
        is_inp: bool = False,
        is_oup: bool = False,
    ):
        self.__PATH = path
        self.__FORMAT = formatter
        self.__FACTOR = normalize_factor
        self.__IS_INP = is_inp
        self.__IS_OUP = is_oup
        self._lat_range = None  # late init
        self._lon_range = None  # late init
        self._reader = None  # strategy pattern

    @abc.abstractmethod
    def load_input_data(
        self,
        target_time: datetime,
        ilen: int,
        target_lat: list[float],
        target_lon: list[float],
        target_shape: tuple[int],
    ) -> np.ndarray:
        """
        Args:
            target_time (datetime): The start time of a predicted event.
            ilen (int): Input length.
            target_lat (List[float]): Target latitude to crop.
            target_lon (List[float]): Target longitude to crop.
            target_shape (Tuple[int]): In case one needs different resolution from orig

        Returns:
            data (np.ndarray): Rain rate which has a shape of [ilen, CH, H, W].
        """
        return NotImplemented

    @abc.abstractmethod
    def load_output_data(
        self,
        target_time: datetime,
        olen: int,
        oint: int,
        target_lat: list[float],
        target_lon: list[float],
        target_shape: tuple[int],
    ) -> np.ndarray:
        """
        Args:
            target_time (datetime): The start time of a predicted event.
            olen (int): Output length.
            oint (int): Output interval. Granularity * interval = 1 hour.
            target_lat (List[float]): Target latitude to crop.
            target_lon (List[float]): Target longitude to crop.
            target_shape (Tuple[int]): In case one needs different resolution from orig

        Returns:
            data (np.ndarray): Hourly accumulated rainfall. The shape
                is [olen, H, W].
        """
        return NotImplemented

    @abc.abstractmethod
    def load_data_from_datetime(self, dt: datetime) -> np.ndarray:
        return NotImplemented

    @abc.abstractmethod
    def inp_initial_time_fn(self, dt_list: list[datetime], ilen: int) -> list[datetime]:
        """
        This function stores the "initial time" defined in Meteorology.
        Note that most of the parameters are recording "data in the past".
        """
        return NotImplemented

    @abc.abstractmethod
    def oup_initial_time_fn(
        self, dt_list: list[datetime], olen: int, oint: int
    ) -> list[datetime]:
        """
        This function stores the "initial time" defined in Meteorology.
        Note that most of the parameters are recording "data in the past".
        """
        return NotImplemented

    @property
    def path(self):
        return self.__PATH

    @property
    def formatter(self):
        return self.__FORMAT

    @property
    def is_inp(self):
        return self.__IS_INP

    @property
    def is_oup(self):
        return self.__IS_OUP
