from typing import List
from datetime import datetime, timedelta

from src.data_loaders.basic_loader import BasicLoader
from src.file_readers.netcdf_reader import NetcdfReader

class RadarLoader(BasicLoader):
    GRANULARITY = 10 # 10-min

    def __init__(self, **kwarg):
        super().__init__(**kwarg)

    def cross_check_start_time(
        self, 
        original_time_list: List[datetime], 
        ilen: int
    ) -> List[datetime]:
        output_time_list = []
        input_offset = ilen - 1
        for idx, dt in reversed(list(enumerate(self.time_list))):
            if idx - input_offset < 0:
                break

            if (dt - self.time_list[idx - input_offset] == 
                timedelta(minutes=input_offset * self.GRANULARITY)):
                output_time_list.append(dt)
        
        return list(set(output_time_list).intersection(set(original_time_list)))
    
    def load_data_from_datetime(self, dt: datetime):
        if self._reader == None:
            self._reader = NetcdfReader()
        file_path = self._BasicLoader__all_files[self.time_list.index(dt)]
        return self._reader.read(file_path, 'cv')