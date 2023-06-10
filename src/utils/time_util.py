from datetime import datetime, timedelta
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

@staticmethod
def whole_day(year:int, month:int, day:int) -> list[datetime]:
    ts = []
    dt = datetime(year, month, day)
    while dt.day == day:
        ts.append(dt)
        dt += timedelta(minutes=10)
    return ts

@staticmethod
def whole_hour(year:int, month:int, day:int, hour:int) -> list[datetime]:
    ts = []
    dt = datetime(year, month, day, hour)
    while dt.hour == hour:
        ts.append(dt)
        dt += timedelta(minutes=10)
    return ts

@staticmethod
def three_days(year:int, month:int, day:int) -> list[datetime]:
    target_t = [datetime(year, month, day) + i * timedelta(days=1) for i in range(-1, 2)]
    ts = []
    for calendar in target_t:
        ts.extend(whole_day(calendar.year, calendar.month, calendar.day))
    return ts
        