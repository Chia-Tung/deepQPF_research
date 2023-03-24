from src.data_loaders.basic_loader import BasicLoader

class RadarLoader(BasicLoader):
    def __init__(self, **kwarg):
        super().__init__(**kwarg)

    def get_time_from_filename(self):
        return
    
    def load_data(self):
        pass