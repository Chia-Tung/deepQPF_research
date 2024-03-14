import random
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.blacklist import Blacklist
from src.const import BLACKLIST_PATH, GREYLIST
from src.data_loaders import BasicLoader
from src.loader_mapping import LoaderMapping
from src.utils.time_util import TimeUtil


class DatetimeManager:
    """
    This class handles all process about TIME and save the target
    time for training, validating and evaluation respectively.
    """

    def __init__(self):
        self.__initial_time_list = list()
        self.__blacklist = set()
        self.__greylist = set()
        self.train_time = list()
        self.valid_time = list()
        self.test_time = list()

        self._load_blacklist()

    def import_time_from_loader(
        self, loader: BasicLoader, ilen: int, olen: int, oint: int
    ) -> None:
        data_type = LoaderMapping.get_loader_nickname(loader)
        _, datetime_list = self.list_all_time(data_type, loader.path, loader.formatter)

        if loader.is_inp:
            self._load_time(
                datetime_list, lambda x: loader.inp_initial_time_fn(x, ilen)
            )
        if loader.is_oup:
            self._load_time(
                datetime_list, lambda x: loader.oup_initial_time_fn(x, olen, oint)
            )

        print(f"{loader.__class__.__name__}: time list is loaded.")

    def _load_time(
        self,
        datetime_candidates: list[datetime],
        datetime_filter: Callable[..., list[datetime]],
    ) -> None:
        datetime_survivor = datetime_filter(datetime_candidates)
        if not self.__initial_time_list:
            self.__initial_time_list = sorted(datetime_survivor)
        else:
            self.__initial_time_list = sorted(
                set(datetime_survivor).intersection(set(self.__initial_time_list))
            )

    def _load_blacklist(self) -> None:
        if BLACKLIST_PATH:
            Blacklist.read_blacklist()
            self.__blacklist = Blacklist.BLACKLIST

        if GREYLIST:
            self.__greylist = Blacklist.GREYLIST
            self.__blacklist = self.__blacklist.union(self.__greylist)

    def list_all_time(
        self, data_type: str, path: str, formatter: str
    ) -> tuple[list[Path], list[datetime]]:
        if path == None or len(path) == 0:
            raise RuntimeError(f"{data_type}: data path must be given.")

        if formatter == None or len(formatter) == 0:
            raise RuntimeError(f"{data_type}: a file format must be given.")

        all_paths = sorted(Path(path).rglob(f"**/*.{formatter.split('.')[-1]}"))

        all_time = []
        for path in tqdm(all_paths):
            all_time.append(TimeUtil.parse_filename_to_time(path, formatter))

        return all_paths, all_time

    def remove_illegal_time(self, start_dt: datetime, end_dt: datetime) -> None:
        """delete datetime not in interested range or in the black list"""
        start_id, end_id = TimeUtil.find_start_end_index(
            self.__initial_time_list, start_dt, end_dt
        )
        self.__initial_time_list = self.__initial_time_list[start_id : end_id + 1]

        if self.__blacklist:
            self.__initial_time_list = [
                x for x in self.__initial_time_list if x not in self.__blacklist
            ]

    def random_split(self, order_by_time: bool, ratios: list[float]) -> None:
        """
        Two split strategies:
        1. random shuffle (not order by time)
        2. sequentially sample (order by time)
        """
        # summation = 1
        ratios = np.array(ratios) / np.array(ratios).sum()

        if order_by_time:
            ratios = np.round(ratios * 10).astype(int)
            chunk_size = ratios.sum()
            time_list_array = np.array(self.__initial_time_list)

            for i in range(chunk_size):
                tmp = time_list_array[i::chunk_size]
                if i < ratios[0]:
                    self.train_time.extend(list(tmp))
                elif i >= chunk_size - ratios[-1]:
                    self.test_time.extend(list(tmp))
                else:
                    self.valid_time.extend(list(tmp))

            # ==================================================
            # DON'T sort the time list, or the `AdoptedDataset`
            # will sample those data in the front forever (bias).
            #
            # self.train_time.sort()
            # self.valie_time.sort()
            # self.test_time.sort()
            # ==================================================
        else:
            random.seed(1000)
            random.shuffle(self.__initial_time_list)
            num_train = int(len(self.__initial_time_list) * ratios[0])
            num_valid = int(len(self.__initial_time_list) * ratios[1])

            self.train_time = self.__initial_time_list[:num_train]
            self.valid_time = self.__initial_time_list[
                num_train : num_train + num_valid
            ]
            self.test_time = self.__initial_time_list[num_train + num_valid :]
