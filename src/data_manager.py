import ast
import random
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import Dict, List

from src.loader_mapping import LoaderMapping
from src.adopted_dataset import AdoptedDataset

class DataManager(LightningDataModule):
    def __init__(
        self,
        start_date: str,
        end_date: str,
        ratios: List[float],
        data_meta_info: Dict[str, Dict],
        input_len: int,
        output_len: int,
        output_interval: int,
        threshold: float = None,
        hourly_data: bool = False,
        target_shape: str = None,
        target_lat: List[float] = None,
        target_lon: List[float] = None,
        sampling_rate: int = None,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self._start_date = start_date
        self._end_date = end_date
        self._ratios = ratios
        self._data_meta_info = data_meta_info
        self._ilen = input_len
        self._olen = output_len
        self._oint = output_interval
        self._threshold = threshold
        self._hourly_data = hourly_data
        self._target_shape = ast.literal_eval(target_shape)
        self._target_lat = target_lat
        self._target_lon = target_lon
        self._sampling_rate = sampling_rate
        self._batch_size = batch_size
        self._workers = num_workers
        self._train_dataset = None
        self._valid_dataset = None
        self._eval_dataset = None
        self._setup()

    def _setup(self):
        # TODO: Load data in sparse metrix.
        self._all_loaders = LoaderMapping.get_all_loaders(self._data_meta_info)

        # handle output loader
        # NOTE: Only one output parameter is allowed.
        # TODO: Try-catch more than one output.
        initial_time_list = next(
            filter(lambda x: x.is_oup, self._all_loaders)
        ).set_start_time_list(self._olen, self._oint)

        # handle input loaders
        for single_loader in (loader for loader in self._all_loaders if loader.is_inp):
            initial_time_list = single_loader.cross_check_start_time(initial_time_list, self._ilen)

        # random split
        # TODO: Make the dispatch more solid. Namely, seperate testing time in 
        #       a `Constant.py` or use a rule-based dispatch algorithm.
        random.seed(1000)
        random.shuffle(sorted(initial_time_list))
        self._ratios = np.array(self._ratios) / np.array(self._ratios).sum()
        num_train = int(len(initial_time_list) * self._ratios[0])
        num_valid = int(len(initial_time_list) * self._ratios[1])

        train_time = initial_time_list[:num_train]
        valid_time = initial_time_list[num_train:num_train+num_valid]
        test_time = initial_time_list[num_train+num_valid:]
        
        print(f"[{self.__class__.__name__}] Training Data Size: {len(train_time)}; " + 
            f"Developing Data Size: {len(valid_time)}; " + 
            f"Testing Data Size: {len(test_time)}")

        self._train_dataset = AdoptedDataset(
            self._ilen,
            self._olen,
            self._oint,
            self._target_shape,
            self._target_lat,
            self._target_lon,
            initial_time_list = train_time,
            data_loader_list = self._all_loaders,
            sampling_rate = self._sampling_rate,
            threshold = self._threshold,
            is_train = True
        )

        self._valid_dataset = AdoptedDataset(
            self._ilen,
            self._olen,
            self._oint,
            self._target_shape,
            self._target_lat,
            self._target_lon,
            initial_time_list = valid_time,
            data_loader_list = self._all_loaders,
            sampling_rate = self._sampling_rate,
            threshold = self._threshold,
            is_valid = True
        )

        self._eval_dataset = AdoptedDataset(
            self._ilen,
            self._olen,
            self._oint,
            self._target_shape,
            self._target_lat,
            self._target_lon,
            initial_time_list = test_time,
            data_loader_list = self._all_loaders,
            sampling_rate = self._sampling_rate,
            threshold = self._threshold,
            is_test = True
        )

    def train_dataloader(self):
        return DataLoader(self._train_dataset, batch_size=self._batch_size, 
            num_workers=self._workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self._valid_dataset, batch_size=self._batch_size, 
            num_workers=self._workers, shuffle=True)
