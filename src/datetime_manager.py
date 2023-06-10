import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from collections.abc import Callable

from src.utils.time_util import TimeUtil
from src.data_loaders import BasicLoader
from src.loader_mapping import LoaderMapping
from src.blacklist import Blacklist

class DatetimeManager:
    """
    This class handles all process about TIME and save the target
    time for training, validating and evaluation respectively.
    """
    def __init__(self):
        self.__initial_time_list = []
        self.__blacklist = []
        self.__greylist = []
        self._train_time_list = None
        self._valid_time_list = None
        self._evalu_time_list = None
        
        self._load_blacklist()

    def import_time_from_loader(
        self, 
        loader:BasicLoader,
        ilen: int,
        olen:int,
        oint:int
    ) -> None:
        data_type = LoaderMapping.get_loader_nickname(loader)
        _, datetime_list = self.list_all_time(data_type, loader.path, loader.formatter)
        
        if loader.is_inp:
            self._load_time(datetime_list, lambda x: loader.inp_initial_time_fn(x, ilen))
        if loader.is_oup:
            self._load_time(datetime_list, lambda x: loader.oup_initial_time_fn(x, olen, oint))

        print(f"{loader.__class__.__name__}: time list is loaded.")

    def _load_time(
        self, 
        datetime_candidates: list[datetime], 
        datetime_filter: Callable[..., list[datetime]]
    ) -> None:
        datetime_survivor = datetime_filter(datetime_candidates)
        if not self.__initial_time_list:
            self.__initial_time_list = sorted(datetime_survivor)
        else:
            self.__initial_time_list = sorted(
                set(datetime_survivor).intersection(set(self.__initial_time_list)))
    
    def load_blacklist(self) -> None:
        if Blacklist.BLACKLIST_PATH:
            pass

        if Blacklist.GREYLIST:
            self.__greylist = Blacklist.GREYLIST
            self.__blacklist.extend(self.__greylist)

    def list_all_time(
        self,
        data_type: str,
        path:str, 
        formatter:str
    ) -> tuple[list[Path], list[datetime]]:
        if path == None  or len(path) == 0:
            raise RuntimeError(f"{data_type}: data path must be given.")
        
        if formatter == None  or len(formatter) == 0:
            raise RuntimeError(f"{data_type}: a file format must be given.")
        
        all_paths = sorted(Path(path).rglob(f"**/*.{formatter.split('.')[-1]}"))

        all_time = []
        for path in tqdm(all_paths):
            all_time.append(TimeUtil.parse_filename_to_time(path, formatter))
        
        return all_paths, all_time

    def remove_illegal_time(self, start_dt: datetime, end_dt: datetime) -> None:
        """ delete datetime not in interested range or in the black list"""
        start_id, end_id = TimeUtil.find_start_end_index(self.__initial_time_list, start_dt, end_dt)
        self.__initial_time_list = self.__initial_time_list[start_id:end_id+1]

        if self.__blacklist:
            self.__initial_time_list = [
                x for x in self.__initial_time_list if x not in self.__blacklist]
            
    def random_split(self, order_by_time: bool, ratios: list[float]) -> tuple[list[datetime]]:
        """
        Two split strategies:
        1. random shuffle (not order by time)
        2. sequentially sample (order by time) 
        """
        # summation = 1
        ratios = np.array(ratios) / np.array(ratios).sum()
        if order_by_time:
            ratios = np.round(ratios * 10).astype(int)

            list(self.split_list(self.__initial_time_list, ratios.sum()))
            ##### Draft #####

        else:
            random.seed(1000)
            random.shuffle(self.__initial_time_list)
            num_train = int(len(self.__initial_time_list) * ratios[0])
            num_valid = int(len(self.__initial_time_list) * ratios[1])

            train_time = self.__initial_time_list[:num_train]
            valid_time = self.__initial_time_list[num_train:num_train+num_valid]
            test_time = self.__initial_time_list[num_train+num_valid:]

        return train_time, valid_time, test_time
    
    def split_list(data: list[datetime], chunk_size: int):
        for idx in range(0, len(data), chunk_size):
            yield data[idx:idx+chunk_size]
