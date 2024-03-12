from typing import List

import datatable as dtb
import numpy as np

from src.file_readers.basic_reader import BasicReader
from src.utils.sparse_matrix_util import SparseMatrixUtil


class JayReader(BasicReader):
    def __init__(self, xdim: int, ydim: int):
        """
        Args:
            xdim (int): The grid number of latitudinal axis.
            ydim (int): The grid number of longitudinal axis.
        """
        self._xdim = xdim
        self._ydim = ydim

    def read(self, filename_list: List[str]) -> np.ndarray:
        """
        Args:
            filename_list (List[str]): List of .jay file paths.

        Returns:
            array_data (np.ndarray): Array data with shape of [B, ydim, xdim]
        """
        # load data
        df_generator = dtb.iread(filename_list)
        # revert sparse matrix to 2D array
        array_data = []
        for df in df_generator:
            array_data.append(
                SparseMatrixUtil.revert_sparse_to_array(df, self._xdim, self._ydim)
            )
        return np.stack(array_data, axis=0)
