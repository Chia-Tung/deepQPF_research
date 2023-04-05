import abc
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime, timedelta
from typing import List

from src.utils.time_util import TimeUtil

class BasicLoader(metaclass=abc.ABCMeta):
    # static variable
    _instance = None
    init_flag = False
    
    def __new__(cls, *args, **kwargs): 
        if cls._instance is None: 
            # assign memory address
            cls._instance = super().__new__(cls) 
        return cls._instance 
    
    def __init__(
        self, 
        path: str, 
        formatter: str, 
        normalize_factor: int = 1, 
        is_inp: bool = False, 
        is_oup: bool = False
    ):
        if self.__class__.init_flag: # singleton pattern
            print(f"Call {self.__class__.__name__} Singleton Object.")
            return
        
        if path == None  or len(path) == 0:
            raise RuntimeError(self.__class__.__name__, " data path must be given.")
        
        if formatter == None  or len(formatter) == 0:
            raise RuntimeError(self.__class__.__name__, " a file format must be given.")
        
        self.__PATH = Path(path)
        self.__FORMAT = formatter
        self.__FACTOR = normalize_factor
        self.__IS_INP = is_inp
        self.__IS_OUP = is_oup
        self._lat_range = None # late init or non-init
        self._lon_range = None # late init or non-init
        self._reader = None # strategy pattern

        self.__all_files, self.__time_list = self.list_all_time()
        self.__class__.init_flag = True
        print(self.__class__.__name__, " instantiate.")
    
    @abc.abstractmethod
    def load_input_data(self, target_time: datetime, ilen: int,
        target_lat: List[float], target_lon: List[float]) -> np.ndarray:
        return NotImplemented
    
    @abc.abstractmethod
    def load_output_data(self, target_time: datetime, olen: int,
        oint: int, target_lat: List[float], target_lon: List[float]) -> np.ndarray:
        return NotImplemented
    
    @abc.abstractmethod
    def load_data_from_datetime(self, dt: datetime) -> np.ndarray:
        return NotImplemented
    
    @abc.abstractmethod
    def cross_check_start_time(self, original_time_list: List[datetime], 
        ilen: int) -> List[datetime]:
        return NotImplemented
    
    @property
    def is_inp(self):
        return self.__IS_INP
    
    @property
    def is_oup(self):
        return self.__IS_OUP
    
    @property
    def time_list(self):
        return self.__time_list
    
    def list_all_time(self):
        # set path -> type:pathlib.Path
        all_paths = sorted(self.__PATH.rglob(f"**/*.{self.__FORMAT.split('.')[-1]}"))

        # set datetime -> type:datetime.datetime
        all_time = []
        for path in tqdm(all_paths):
            all_time.append(TimeUtil.parse_filename_to_time(path, self.__FORMAT))

        return all_paths, all_time
    
    def set_start_time_list(self, output_len: int, output_interval: int) -> List[datetime]:
        """
        This function stores the "initial time" defined in Meteorology. 
        Note that most of the parameters are recording "data in the past".
        """
        time_map = []
        raw_idx = 0
        target_offset = output_len * output_interval
        granularity = 60 / output_interval
        while raw_idx < len(self.__time_list):
            if raw_idx + target_offset >= len(self.__time_list):
                break

            if (self.__time_list[raw_idx + target_offset] - self.__time_list[raw_idx]
                == timedelta(minutes = target_offset * granularity)):
                time_map.append(self.__time_list[raw_idx])

            raw_idx += 1
        return time_map
