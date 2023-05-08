import numpy as np
import datatable as dtb

class SparseMatrixUtil:
    
    @staticmethod
    def revert_sparse_to_array(
        sparse_matrix: dtb.Frame,
        xdim: int, 
        ydim: int, 
        default: float = 0.,
        dtype: np.dtype = np.float32
    ) -> np.ndarray:
        """
        Args:
            sparse_matrix (dtb.Frame): A Nx3 table containing the x, y and value.
            xdim (int): The grid number of latitudinal axis.
            ydim (int): The grid number of longitudinal axis.
            default (float): Default value to fill in the matrix.
        """
        output = np.full([ydim, xdim], default, dtype=dtype)
        output[sparse_matrix['x'], sparse_matrix['y']] = sparse_matrix['value']
        return output