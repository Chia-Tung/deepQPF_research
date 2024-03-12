import datatable as dtb
import numpy as np


class SparseMatrixUtil:

    @staticmethod
    def revert_sparse_to_array(
        sparse_matrix: dtb.Frame,
        xdim: int,
        ydim: int,
        default: float = 0.0,
        dtype: np.dtype = np.float32,
    ) -> np.ndarray:
        """
        Args:
            sparse_matrix (dtb.Frame): A Nx3 table containing the x, y and value.
            xdim (int): The grid number of latitudinal axis.
            ydim (int): The grid number of longitudinal axis.
            default (float): Default value to fill in the matrix.
            dtype (np.dtype): The data type chosen from numpy.
        """
        output = np.full([ydim, xdim], default, dtype=dtype)
        output[sparse_matrix["lat_id"], sparse_matrix["lon_id"]] = sparse_matrix[
            "value"
        ]
        return output
