from datetime import datetime
from pathlib import Path
from typing import List
from bisect import bisect_left

class TimeUtil:
    @staticmethod
    def parse_filename_to_time(filename: Path, format: str):
        filename = filename.name
        return TimeUtil.parse_string_to_time(filename, format)
    
    @staticmethod
    def parse_string_to_time(string: str, format: str):
        return datetime.strptime(string, format)
    
    @staticmethod
    def find_start_end_index(dt_list: List[datetime], start: datetime, end: datetime):
        """
        Args:
            dt_list (List[datetime]): A sorted datetime list with ascending order.
            start (datetime): Start datetime.
            end (datetime): End datetime.
        """
        start_index = bisect_left(dt_list, start)
        end_index = bisect_left(dt_list, end)

        return start_index, end_index
        