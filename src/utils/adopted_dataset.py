import numpy as np
from torch.utils.data import Dataset
from datetime import datetime
from typing import List, Tuple, Dict

from src.data_loaders.basic_loader import BasicLoader
from src.loader_mapping import LoaderMapping

class AdoptedDataset(Dataset):
    def __init__(
        self,
        ilen: int, 
        olen: int,
        oint: int,
        target_shape: Tuple[int],
        target_lat: List[float],
        target_lon: List[float],
        initial_time_list: List[datetime],
        data_meta_info: List[BasicLoader],
        sampling_rate: int,
        threshold: float, 
        is_train: bool = False,
        is_valid: bool = False,
        is_test: bool = False,
    ):
        super().__init__()
        self._initial_time_list = initial_time_list
        self._sampling_rate = sampling_rate
        self._ilen = ilen
        self._olen = olen
        self._oint = oint
        self._target_shape = target_shape
        self._target_lat = target_lat
        self._target_lon = target_lon
        self._thsh = threshold

        self._data_loader_list = LoaderMapping.get_all_loaders(data_meta_info)

    def __len__(self):
        return len(self._initial_time_list) // self._sampling_rate
    
    def __getitem__(self, index):
        target_time = self._initial_time_list[index]

        input_data_map = {}
        output_data_map = {}
        for data_loader in self._data_loader_list:
            nickname = LoaderMapping.get_loader_nickname(data_loader)
            if data_loader.is_inp:
                input_data_map[nickname] = data_loader.load_input_data(
                    target_time, self._ilen, self._target_lat, self._target_lon)
            if data_loader.is_oup:
                output_data_map[nickname] = data_loader.load_output_data(
                    target_time, self._olen, self._oint, self._target_lat, self._target_lon)

        self.shape_check(input_data_map)
        self.shape_check(output_data_map)
        
        # build mask
        mask = np.zeros_like(output_data_map['rain'])
        mask[output_data_map['rain'] > self._thsh] = 1

        return input_data_map, output_data_map, mask
    
    def shape_check(self, data_map: Dict[str, np.ndarray]):
        if not list(data_map.values())[0].shape[-2:] == self._target_shape:
            raise RuntimeError ("Data shape is not compatible to the target shape.")
