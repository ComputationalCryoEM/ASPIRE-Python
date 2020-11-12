import logging

from aspire.abinitio import CLOrient3D

logger = logging.getLogger(__name__)


class CommLineGCAR(CLOrient3D):
    """
    Define a derived class to estimate 3D orientations using Globally Consistent Angular Reconstitution described
    as below:
    R. Coifman, Y. Shkolnisky, F. J. Sigworth, and A. Singer, Reference Free Structure Determination through
    Eigenvestors of Center of Mass Operators, Applied and Computational Harmonic Analysis, 28, 296-312 (2010).

    """

    def __init__(self, src):
        """
        constructor of an object for estimating 3D oreintations
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
