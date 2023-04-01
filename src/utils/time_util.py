from datetime import datetime
from pathlib import Path

class TimeUtil():
    @staticmethod
    def parse_filename_to_time(filename: Path, format: str):
        filename = filename.name
        return datetime.strptime(filename, format)