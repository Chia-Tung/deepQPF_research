from src.data_loaders.basic_loader import BasicLoader

class RadarLoader(BasicLoader):
    def __init__(self, **kwarg):
        super().__init__(**kwarg)

    def cross_check_input_time(self):
        return
    
    def load_data_from_datetime(self):
        pass