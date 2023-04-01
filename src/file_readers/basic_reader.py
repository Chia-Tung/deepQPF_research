import os
import abc

class BasicReader(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __init__(self) -> None:
        return NotImplemented
    
    @abc.abstractmethod
    def read(self) -> None:
        return NotImplemented
    
    def check_file_exist(self, filename: str):
        assert os.path.exists(filename), f'{filename} does not exist!'
    