import logging
from typing import Union


class Logger:
    """
    This class wraps an original Python logger and make it easy
    to use universally in SINGLETON pattern.

    TODO: FileHandler
    """

    # static variable
    _instance = None
    init_flag = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # assign memory address
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, name: str = "dev", level: int = logging.warning):
        # singleton pattern
        if self.__class__.init_flag:
            print(f"Call {self.__class__.__name__} Singleton Object.")
            return

        self._logger = logging.getLogger(name=name)  # alias name for logger
        self._logger.setLevel(level)  # 1st filter
        self.set_handler()

        self.__class__.init_flag = True
        print(self.__class__.__name__, " instantiated.")

    def set_handler(self):
        # set the output format on screen
        formatter = logging.Formatter(
            "%(asctime)s - [line:%(lineno)d] - %(levelname)s: %(message)s"
        )
        self._handler = logging.StreamHandler()  # 2nd filter
        self._handler.setFormatter(formatter)
        self._logger.addHandler(self._handler)

    def debug(self, msg: Union[str, int, float]):
        self._logger.debug(msg)

    def info(self, msg: Union[str, int, float]):
        self._logger.info(msg)

    def warning(self, msg: Union[str, int, float]):
        self._logger.warning(msg)

    def error(self, msg: Union[str, int, float]):
        self._logger.error(msg)

    def caritical(self, msg: Union[str, int, float]):
        self._logger.critical(msg)
