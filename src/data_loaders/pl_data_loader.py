from datetime import datetime
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import Tuple, Dict

from src.data_loaders.data_loader_integration import DataLoaderIntegration


class PLDataLoader(LightningDataModule):
    def __init__(
        self,
        train_start: datetime,
        train_end: datetime,
        val_start: datetime,
        val_end: datetime,
        data_type_info: Dict[str, Dict],
        input_len: int,
        output_len: int,
        output_interval: int,
        threshold: float = None,
        hourly_data: bool = False,
        img_size: Tuple[int] = None,
        sampling_rate: int = None,
        batch_size: int = 32,
        num_workers: int = 4,
    ):

        super().__init__()
        self._train_start = train_start
        self._train_end = train_end
        self._val_start = val_start
        self._val_end = val_end
        self._dtype_info = data_type_info
        self._ilen = input_len
        self._olen = output_len
        self._output_interval = output_interval
        self._threshold = threshold
        self._hourly_data = hourly_data
        self._img_size = img_size
        self._sampling_rate = sampling_rate
        self._batch_size = batch_size
        self._workers = num_workers
        self._train_dataset = None
        self._val_dataset = None
        self._setup()

    def _setup(self):
        self._train_dataset = DataLoaderIntegration(
            self._train_start,
            self._train_end,
            self._ilen,
            self._olen,
            output_interval = self._output_interval,
            threshold = self._threshold,
            data_type_info = self._dtype_info,
            hourly_data = self._hourly_data,
            img_size = self._img_size,
            sampling_rate = self._sampling_rate,
            is_train = True,
        )

        self._val_dataset = DataLoaderIntegration(
            self._val_start,
            self._val_end,
            self._ilen,
            self._olen,
            output_interval = self._output_interval,
            threshold = self._threshold,
            data_type_info = self._dtype_info,
            hourly_data = self._hourly_data,
            img_size = self._img_size,
            sampling_rate = self._sampling_rate,
            is_valid = True,
        )

    def train_dataloader(self):
        return DataLoader(self._train_dataset, batch_size=self._batch_size, 
                          num_workers=self._workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self._val_dataset, batch_size=self._batch_size, 
                          num_workers=self._workers, shuffle=False)
