from enum import Enum

from src.data_loaders import BasicLoader, RainLoaderJay, RadarLoaderJay

class LoaderMapping(Enum):
    """ Genrate Multiple Customized Data Loaders """
    rain = RainLoaderJay
    radar = RadarLoaderJay

    @classmethod
    def get_all_loaders(cls, data_infos: dict[str, dict]) -> list[BasicLoader]:
        all_loaders = []
        for key, value in data_infos.items():
            all_loaders.append(cls.get_single_loader(key, value))
        return all_loaders

    def get_single_loader(self, key, value):
        return self.get_loader_type(key)(**value)
    
    @classmethod
    def get_loader_type(cls, key):
        return cls[key].value
    
    @classmethod
    def get_loader_nickname(cls, data_loader: BasicLoader) -> str:
        for member in cls:
            if isinstance(data_loader, member.value):
                return member.name