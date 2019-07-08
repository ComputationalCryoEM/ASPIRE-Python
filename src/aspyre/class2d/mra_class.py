from aspyre.class2d import Class2D
import logging

logger = logging.getLogger(__name__)


class MRAClass2D(Class2D):

    def __init__(self, src):
        """
        constructor of an object for classifying 2D images using
        Multi-Reference Alignment(MRA) algorithm
        """
        pass

    def calculate_invariant(self):
        """
        calculate invariant quantities such as ACF and DACF
        """
        pass

    def classify(self, src):
        """
        perform classifying 2D images
        """
        pass

    def output(self):
        """
        Output the clean images
        """
        pass




