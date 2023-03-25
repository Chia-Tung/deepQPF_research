import abc
from tqdm import tqdm
from pathlib import Path
from datetime import datetime, timedelta
from typing import List

from src.time_util import TimeUtil

class BasicLoader(metaclass=abc.ABCMeta):
    def __init__(
        self, 
        path: str, 
        formatter: str, 
        normalize_factor: int = 1, 
        is_inp: bool = False, 
        is_oup: bool = False
    ):
        if path == None  or len(path) == 0:
            raise RuntimeError(self.__class__.__name__, " data path must be given.")
        
        if formatter == None  or len(formatter) == 0:
            raise RuntimeError(self.__class__.__name__, " a file format must be given.")
        
        self.__PATH = Path(path)
        self.__FORMAT = formatter
        self.__FACTOR = normalize_factor
        self.__IS_INP = is_inp
        self.__IS_OUP = is_oup

        self.__all_files, self.__time_list = self.list_all_time()
        print(self.__class__.__name__, " instantiate.")
    
    @abc.abstractmethod
    def load_data_from_datetime(self, start_t: datetime, end_t: datetime):
        return NotImplemented
    
    @abc.abstractmethod
    def cross_check_input_time(self, target_time_list: List[datetime]):
        return NotImplemented
    
    # @property
    # def is_inp(self):
    #     return self._is_inp
    
    @property
    def is_oup(self):
        return self.__IS_OUP
 
    # @is_oup.setter
    # def is_oup(self, value: bool):
    #     self._is_oup = value
    
    def list_all_time(self):
        # set path -> type:pathlib.Path
        all_paths = sorted(self.__PATH.rglob(f"**/*.{self.__FORMAT.split('.')[-1]}"))

        # set datetime -> type:datetime.datetime
        all_time = []
        for path in tqdm(all_paths):
            all_time.append(TimeUtil.parse_filename_to_time(path, self.__FORMAT))

        return all_paths, all_time
    
    def set_time_list_as_target(self, output_len: int, output_interval: int) -> List[datetime]:
        time_map = []
        raw_idx = 0
        target_offset = output_len * output_interval
        while raw_idx < len(self.__time_list):
            if raw_idx + target_offset >= len(self.__time_list):
                break
            
            # 60/output_interval: 
            #   Number of minutes between each interval. 
            #   Also representing the granularity of output data
            if not (self.__time_list[raw_idx + target_offset] - self.__time_list[raw_idx] 
                != timedelta(seconds = target_offset * (60/output_interval) * 60)):
                time_map.append(self.__time_list[raw_idx])

            raw_idx += 1
        return time_map