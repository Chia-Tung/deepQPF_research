import ast

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.datetime_manager import DatetimeManager
from src.loader_mapping import LoaderMapping
from src.utils.adopted_dataset import AdoptedDataset
from src.utils.time_util import TimeUtil


class DataManager(LightningDataModule):
    def __init__(
        self,
        start_date: int,
        end_date: int,
        order_by_time: bool,
        ratios: list[float],
        input_len: int,
        output_len: int,
        output_interval: int,
        target_lat: list[float],
        target_lon: list[float],
        target_shape: str,
        sampling_rate: int,
        batch_size: int,
        num_workers: int,
        data_meta_info: dict[str, dict],
    ):
        super().__init__()
        # hyperparameters from config
        self._start_date = TimeUtil.parse_string_to_time(str(start_date), "%Y%m%d%H%M")
        self._end_date = TimeUtil.parse_string_to_time(str(end_date), "%Y%m%d%H%M")
        self._order_by_time = order_by_time
        self._ratios = ratios
        self._ilen = input_len
        self._olen = output_len
        self._oint = output_interval
        self._target_lat = target_lat
        self._target_lon = target_lon
        self._target_shape = ast.literal_eval(target_shape)
        self._sampling_rate = sampling_rate
        self._batch_size = batch_size
        self._workers = num_workers
        self._data_meta_info = data_meta_info

        # internal property
        self._hourly_data = None
        self._train_dataset = None
        self._valid_dataset = None
        self._evalu_dataset = None
        self._datetime_maneger = DatetimeManager()
        self._setup()

    def _setup(self):
        # data loaders instantiate
        self._all_loaders = LoaderMapping.get_all_loaders(self._data_meta_info)

        # set initial time list
        for loader in self._all_loaders:
            self._datetime_maneger.import_time_from_loader(
                loader, self._ilen, self._olen, self._oint
            )

        # remove illegal datetime
        self._datetime_maneger.remove_illegal_time(self._start_date, self._end_date)

        # random split
        self._datetime_maneger.random_split(self._order_by_time, self._ratios)
        train_time, valid_time, test_time = (
            self._datetime_maneger.train_time,
            self._datetime_maneger.valid_time,
            self._datetime_maneger.test_time,
        )

        print(
            f"[{self.__class__.__name__}] "
            f"Total data collected: {len(train_time)+len(valid_time)+len(test_time)}, "
            f"Sampling Rate: {self._sampling_rate}\n"
            f"[{self.__class__.__name__}] "
            f"Training Data Size: {len(train_time) // self._sampling_rate}, "
            f"Validating Data Size: {len(valid_time) // self._sampling_rate}, "
            f"Testing Data Size: {len(test_time) // self._sampling_rate} \n"
            f"[{self.__class__.__name__}] "
            f"Image Shape: {self._target_shape}, Batch Size: {self._batch_size}"
        )

        self._train_dataset = AdoptedDataset(
            self._ilen,
            self._olen,
            self._oint,
            self._target_shape,
            self._target_lat,
            self._target_lon,
            initial_time_list=train_time,
            data_meta_info=self._data_meta_info,
            sampling_rate=self._sampling_rate,
            is_train=True,
        )

        self._valid_dataset = AdoptedDataset(
            self._ilen,
            self._olen,
            self._oint,
            self._target_shape,
            self._target_lat,
            self._target_lon,
            initial_time_list=valid_time,
            data_meta_info=self._data_meta_info,
            sampling_rate=self._sampling_rate,
            is_valid=True,
        )

        self._evalu_dataset = AdoptedDataset(
            self._ilen,
            self._olen,
            self._oint,
            self._target_shape,
            self._target_lat,
            self._target_lon,
            initial_time_list=test_time,
            data_meta_info=self._data_meta_info,
            sampling_rate=self._sampling_rate,
            is_test=True,
        )

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            num_workers=self._workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self._valid_dataset,
            batch_size=self._batch_size,
            num_workers=self._workers,
            shuffle=False,
        )

    def get_data_info(self):
        inp_data_map = self._train_dataset[0][0]
        return {
            "shape": self._target_shape,
            "channel": {k: v.shape[1] for k, v in inp_data_map.items()},
            "olen": self._olen,
        }
