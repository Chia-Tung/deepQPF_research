import numpy as np

from src.data_loaders.rain_loader_jay import RainLoaderJay


class RadarLoaderJay(RainLoaderJay):
    def __init__(self, **kwarg):
        super().__init__(**kwarg)

    def load_output_data(
        self, target_time, olen, oint, target_lat, target_lon
    ) -> np.ndarray:
        pass
