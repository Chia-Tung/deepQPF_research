import abc
from datetime import datetime

class BasicLoader(metaclass=abc.ABCMeta):
    def __init__(self, path: str, filename: str, normalize_factor: int):
        self.__PATH = path
        self.__FILENAME = filename
        self.__factor = normalize_factor

        print(self.__class__.__name__, " instantiated.")

    @abc.abstractmethod
    def get_time_from_filename(self):
        return NotImplemented
    
    @abc.abstractmethod
    def load_data(self, start_t: datetime, end_t: datetime):
        return NotImplemented