class ModelManager:
    def __init__(self, **kwarg) -> None:
        self.__model_type = kwarg['name']
        self.__img_shape = kwarg['data_info']['shape']
        self.__variable_names = kwarg['data_info']['vname']