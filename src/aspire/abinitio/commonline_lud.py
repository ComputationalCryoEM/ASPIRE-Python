import logging

from aspire.abinitio import CLOrient3D

logger = logging.getLogger(__name__)


class CommLineLUD(CLOrient3D):
    """
    Define a derived class to estimate 3D orientations using Least Unsquared Deviations described as below:
    L. Wang, A. Singer, and  Z. Wen, Orientation Determination of Cryo-EM Images Using Least Unsquared Deviations,
    SIAM J. Imaging Sciences, 6, 2450-2483 (2013).

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
