import logging

import numpy as np

logger = logging.getLogger(__name__)


class Cell2D:
    """
    Define a base class of Cell to perform similar cell functions in MATLAB.

    """

    def __init__(self, rows, cols, dtype=np.float32):

        self.dtype = np.dtype(dtype)
        self.dtype = dtype
        self.rows = rows
        self.cols = cols
        self.nrow = np.size(rows)
        self.ncol = np.size(cols)

        size = 0
        for i in range(0, np.size(rows)):
            for j in range(0, np.size(cols)):
                size += rows[i] * cols[j]
        self.size = size
        self.cell_list = []
        for i in range(0, np.size(rows)):
            for j in range(0, np.size(cols)):
                self.cell_list.append(np.zeros((rows[i], cols[j]), dtype=self.dtype))

    def mat2cell(self, mat, rows, cols):
        if np.sum(rows) != np.size(mat, 0) or np.sum(cols) != np.size(mat, 1):
            raise RuntimeError(
                "Matrix is not compatible wth the cell array in rows or cols size!"
            )
        offset = 0
        offsetr = 0
        for i in range(0, self.nrow):
            offsetc = 0
            for j in range(0, self.ncol):
                self.cell_list[offset][:] = mat[
                    offsetr : offsetr + rows[i], offsetc : offsetc + cols[j]
                ]
                offset += 1
                offsetc += cols[j]
            offsetr += rows[i]
        return self.cell_list
