import netCDF4 as nc

from src.file_readers.basic_reader import BasicReader

class NetcdfReader(BasicReader):

    def __init__(self):
        pass
    
    def read(self, filename: str, variable_name: str):
        self.check_file_exist(filename)
        return nc.Dataset(filename)[variable_name][:]
    
    def show_keys(self, filename: str):
        self.check_file_exist(filename)
        print(nc.Dataset(filename).variables.keys())