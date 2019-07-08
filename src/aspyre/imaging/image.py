import numpy as np


class Image(np.ndarray):
    def shift(self):
        raise NotImplementedError

    def rotate(self):
        raise NotImplementedError


class CartesianImage(Image):
    def expand(self, basis):
        return BasisImage(basis)


class BasisImage(Image):
    def __init__(self, basis):
        self.basis = basis

    def evaluate(self):
        return CartesianImage()


class FBBasisImage(BasisImage):
    pass
