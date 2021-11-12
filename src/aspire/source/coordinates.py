import logging

from abc import ABC, abstractmethod
from collections import OrderedDict
from aspire.source.image import ImageSource

logger = logging.getLogger(__name__)

class CoordinateSourceBase(ImageSource, ABC):
    """ Base Class defining CoordinateSource interface. """
    # This is the point where we get down to MRC files and coordinates.
    #   There appears to be a variety of ways to get there,
    #   which can be implemented in subclasses.
    # Grabbed some of the main ideas from the current PR.

    def __init__(self):
        self.mrc2coords = OrderedDict()

    @abstractmethod
    def populate_mrc2coords(self):
        """ current"""

    def populate_particles(self):
        """ current pr's all_particles..."""

    def exclude_boundary_particles(self):
        """ Remove particles on the boundary. """

    def _images(self):
        """ Our image chunker/getter """
        
# example subclasses.
class EmanCoordinateSource(CoordinateSourceBase):
    """ Eman specific implementations. """

class GuatoMatchCoordinateSource(CoordinateSourceBase):
    """ Guato specific implementations. """

class RelionCoordinateSource(CoordinateSourceBase):
    """ Relion specific implementations. """

# potentially one or two of these are potentially different 
class XYZProjectDirSource(RelionCoordinateSource):
    """ just an example... """

    
class CoordinateSource:
    # Factory for selecting and implementing a concrete subclass of CoordinateSourceBase
    # Pretty much it's only purpose is to select and return the right subclass of CoordinateSourceBase 
    """ Our User Facing Class ... """

    def __init__(self, *args, **kwargs):
        """ logic to pick the right subclass """
        # returns ______CoordinateSourceBase(*args, **kwargs)



### When explaining usually people use different naming scheme
# my CoordinateSourceBase  would be  CoordinateSource
# my CoordinateSource      would be  CoordinateSourceFactory

## (because CoordinateSourceFactory is a factory that stamps out different CoordinateSource)
##   However, our users probably don't need to know about any of this... so we change the names
##   to protect the innocent.
