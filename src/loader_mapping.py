from enum import Enum
from typing import Dict

from src.data_loaders.basic_loader import BasicLoader
from src.data_loaders.rain_loader import RainLoader
from src.data_loaders.rain_loader_jay import RainLoaderJay
from src.data_loaders.radar_loader import RadarLoader
from src.data_loaders.radar_loader_jay import RadarLoaderJay

class LoaderMapping(Enum):
    """ Genrate Multiple Customized Data Loaders in Singleton Pattern """
    rain = RainLoaderJay
    radar = RadarLoaderJay

    @classmethod
    def get_all_loaders(cls, data_infos: Dict[str, Dict]):
        all_loaders = []
        for key, value in data_infos.items():
            all_loaders.append(cls.get_single_loader(key, value))
        return all_loaders

    def get_single_loader(key, value):
        return LoaderMapping.get_loader_type(key)(**value)
    
    @classmethod
    def get_loader_type(cls, key):
        return cls[key].value
    
    @classmethod
    def get_loader_nickname(cls, data_loader: BasicLoader) -> str:
        for member in cls:
            if isinstance(data_loader, member.value):
                return member.name