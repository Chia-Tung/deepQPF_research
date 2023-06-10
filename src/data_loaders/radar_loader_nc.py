import numpy as np
from datetime import datetime

from src.data_loaders.rain_loader_nc import RainLoaderNc

class RadarLoaderNc(RainLoaderNc):

    def __init__(self, **kwarg):
        super().__init__(**kwarg)
    
    def load_output_data(self, target_time, olen, oint, target_lat, target_lon) -> np.ndarray:
        pass
    
    def load_data_from_datetime(self, dt: datetime) -> np.ndarray:
        file_path = self.get_filename_from_dt(dt)
        return self._reader.read(file_path, 'cv')