from src.data_loaders.basic_loader import BasicLoader

class RadarLoader(BasicLoader):
    def __init__(self, **kwarg):
        super().__init__(**kwarg)

    def parse_filename_to_datetime(self):
        return
    
    def load_data(self):
        pass