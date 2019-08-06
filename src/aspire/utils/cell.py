import numpy as np
import logging

logger = logging.getLogger(__name__)


class Cell2D:
    """
    Define a base class of Cell to perform similar cell functions in MATLAB.

    """

    def __init__(self, rows, cols, dtype=None):

        if dtype is None:
            dtype = 'double'
        self.dtype = dtype
        self.rows = rows
        self.cols = cols
        self.nrow = np.size(rows)
        self.ncol = np.size(cols)

        size = 0
        for i in range(0, np.size(rows)):
            for j in range(0, np.size(cols)):
                size += rows[i]*cols[j]
        self.size = size
        self.cell_list = []
        for i in range(0, np.size(rows)):
            for j in range(0, np.size(cols)):
                self.cell_list.append(np.zeros((rows[i], cols[j]), dtype=dtype))

    def mat2cell(self, mat, rows, cols):
        if np.sum(rows) != np.size(mat, 0) or np.sum(cols) != np.size(mat, 1):
            raise RuntimeError('Matrix is not compatible wth the cell array in rows or cols size!')
        offset = 0
        offsetr = 0
        for i in range(0, self.nrow):
            offsetc = 0
            for j in range(0, self.ncol):
                for k in range(0, rows[i]):
                    for m in range(0, cols[j]):
                        self.cell_list[offset][k, m] = mat[offsetr + k, offsetc+m]
                offset += 1
                offsetc += cols[j]
            offsetr += rows[i]
        return self.cell_list

    def mat2blk_diag(self, mat, rows, cols):
        self.mat2cell(mat, rows, cols)
        blk_diag=[]
        offset = 0
        for i in range(0, self.nrow):
            for j in range(0, self.ncol):
                offset += 1
                if i == j:
                    blk_diag.append(self.cell_list[offset])
        return blk_diag
