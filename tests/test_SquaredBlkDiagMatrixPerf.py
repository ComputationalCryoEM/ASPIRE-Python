from aspire.utils.numeric import xp
from aspire import config

from aspire.utils.squared_blk_diag_matrix import SquaredBlkDiagMatrix
from aspire.utils.blk_diag_matrix import BlkDiagMatrix
from matrices_testCase import matBlockShapeList
#from tests.matrices_testCase import matBlockShapeList
from aspire import config
import time

## [(blockSize, nbBlocks)]
#blockShapeList = [(4,1),(2,1)]
#blockShapeList = [(128,256)]
#blockShapeList = [(128,1), (64,2), (32,4), (16,8), (8,16), (4,16), (1,16)]
#blockShapeList = [(128,1), (64,2), (32,4), (16,8), (8,16), (4,32), (2,64), (1,129)]

iterations = 1
for  blockShapeList in matBlockShapeList:
    totalBlocks = sum([x[1] for x in blockShapeList])
    print("total blocks in matrix",totalBlocks)
    blocklistA = []
    blocklistB = []
    for i,bc in enumerate(blockShapeList):
        blocklistA.extend(xp.random.rand(bc[1],bc[0],bc[0]))
        blocklistB.extend(xp.random.rand(bc[1],bc[0],bc[0]))

    A = BlkDiagMatrix.from_list(blocklistA)
    B = BlkDiagMatrix.from_list(blocklistB)
    squaredA = SquaredBlkDiagMatrix.from_list(blocklistA)
    squaredB = SquaredBlkDiagMatrix.from_list(blocklistB)

    t1 = time.time()
    for i in range(iterations):
        A @= B
    if config.common.cupy ==1:
        xp.cuda.Device(0).synchronize()
    t2 = time.time()
    total = t2-t1
    print("ref {:.3E}".format(total))

    t1 = time.time()
    for i in range(iterations):
        squaredA @= squaredB
    if config.common.cupy ==1:
        xp.cuda.Device(0).synchronize()
    t2 = time.time()
    total = t2-t1
    print("batched {:.3E}".format(total))
