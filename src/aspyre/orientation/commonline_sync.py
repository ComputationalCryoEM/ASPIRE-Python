from aspyre.orientation import Orient3D
import logging

logger = logging.getLogger(__name__)


class CommLineSync(Orient3D):
    """
    Define a class to estimate 3D orientations using Synchronization described as below:
    Y. Shkolnisky, and A. Singer, Viewing Direction Estimation in Cryo-EM Using Synchronization,
    SIAM J. Imaging Sciences, 5, 1088-1110 (2012).

    """

    def __init__(self, src):
        """
        constructor of an object for estimating 3D orientations
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




