import logging

from aspire.abinitio import CLOrient3D

logger = logging.getLogger(__name__)


class CommLineEV(CLOrient3D):
    """
    Class to estimate 3D orientations using Eigenvector method
    :cite:`DBLP:journals/siamis/SingerS11`
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
