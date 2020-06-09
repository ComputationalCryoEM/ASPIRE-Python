import logging

from aspire.utils.numeric import xp

logger = logging.getLogger(__name__)


class Cell2D:
    """
    Define a base class of Cell to perform similar cell functions in MATLAB.

    """
    def __init__(self, rows, cols, dtype=None):

        if dtype is None:
            dtype = 'double'
        self.dtype = dtype
        self.rows = xp.asarray(rows)
        self.cols = xp.asarray(cols)
        self.nrow = xp.size(rows)
        self.ncol = xp.size(cols)

        size = 0
        for i in range(0, xp.size(rows)):
            for j in range(0, xp.size(cols)):
                size += rows[i]*cols[j]
        self.size = size
        self.cell_list = []
        for i in range(0, xp.size(rows)):
            for j in range(0, xp.size(cols)):
                self.cell_list.append(xp.zeros((int(rows[i]), int(cols[j])), dtype=dtype))

    def mat2cell(self, mat, rows, cols):
        if xp.sum(rows) != xp.size(mat, 0) or xp.sum(cols) != xp.size(mat, 1):
            raise RuntimeError('Matrix is not compatible wth the cell array in rows or cols size!')
        offset = 0
        offsetr = 0
        for i in range(0, self.nrow):
            offsetc = 0
            for j in range(0, self.ncol):
                self.cell_list[offset][:] = mat[offsetr:offsetr + rows[i], offsetc:offsetc + cols[j]]
                offset += 1
                offsetc += cols[j]
            offsetr += rows[i]
        return self.cell_list
