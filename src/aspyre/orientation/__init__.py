import logging

logger = logging.getLogger(__name__)


class Orient3D:
    """
    Define a base class for estimating 3D orientations
    """
    def __init__(self, src):
        """
        constructor of an object for estimating 3D orientations
        """
        pass

    def get_commonlines(self):
        """
        find the common lines by voting method
        """
        pass

    def build_clmatrix(self):
        """
        build the common lines matrix
        """
        pass

    def estimate(self):
        """
        perform estimation of orientations
        """
        pass

    def output(self):
        """
        Output the 3D orientations
        """
        pass




