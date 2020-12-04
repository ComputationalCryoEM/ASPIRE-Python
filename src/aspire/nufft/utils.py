import numpy as np


def nextpow2(x):
    return np.ceil(np.log2(np.array(x))).astype("int")
