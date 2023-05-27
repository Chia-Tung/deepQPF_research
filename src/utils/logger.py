import logging
from typing import Union

class Logger:
    """
    This class wraps an original Python logger and make it easy 
    to use universally.

    TODO: FileHandler
    """
    def __init__(self, name: str = 'dev', level: int = logging.warning):
        self._logger = logging.getLogger(name=name) # alias name for logger
        self._logger.setLevel(level) # log level to show
        self.set_handler()

    def set_handler(self):
        # set the output format on screen
        formatter = logging.Formatter("%(asctime)s - [line:%(lineno)d] - %(levelname)s: %(message)s")
        self._handler = logging.StreamHandler()
        self._handler.setFormatter(formatter)
        self._logger.addHandler(self._handler)

    def debug(self, msg: Union[str,int,float]):
        self._logger.debug(msg)

    def info(self, msg: Union[str,int,float]):
        self._logger.info(msg)

    def warning(self, msg: Union[str,int,float]):
        self._logger.warning(msg)

    def error(self, msg: Union[str,int,float]):
        self._logger.error(msg)

    def caritical(self, msg: Union[str,int,float]):
        self._logger.critical(msg)