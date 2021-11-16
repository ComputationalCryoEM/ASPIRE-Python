import logging

from abc import ABC, abstractmethod
from collections import OrderedDict
from aspire.source.image import ImageSource

logger = logging.getLogger(__name__)


class CoordinateSourceBase(ImageSource, ABC):
    """Base Class defining CoordinateSource interface."""

    # This is the point where we get down to MRC files and coordinates.
    # There appears to be a variety of ways to get there,
    # which can be implemented in subclasses.
    # Grabbed some of the main ideas from the current PR.

    def __init__(self):
        self.mrc2coords = OrderedDict()

    @abstractmethod
    def populate_mrc2coords(self):
        """
        Subclasses use this method to read coordinate files and convert them into the canonical form
        in self.mrc2coords. Different sources store coordinate information differently, so the
        arguments and details of this method may vary.
        """
        raise NotImplementedError(
            "Subclasses should implement this method to populate mrc2coords"
        )

    def populate_particles(self):
        pass

    def exclude_boundary_particles(self):
        pass
        """Remove particles on the boundary."""

    def _images(self):
        pass
        """Our image chunker/getter"""


class FromFilesCoordinateSource(CoordinateSourceBase):
    """
    This class represents data sources that will be read in via the `files` parameter
    That is, the user will provide micrograph paths and their corresponding coords
    """

    def __init__(self, files, data_folder, particle_size):
        # call method common to all subclasses
        mrc_paths, coord_paths = self._check_and_get_paths(files, data_folder)
        # call subclass's implementation of populate_mrc2coords
        self.populate_mrc2coords(mrc_paths, coord_paths, particle_size)

    def populate_mrc2coords(self):
        """
        Not implemented here since this class just stores common code of its subclasses.
        """
        raise NotImplementedError(
            "FromFilesCoordinateSource subclasses should implement this method based on the details of the coordinate files they read."
        )

    def _check_and_get_paths(self, files, data_folder):
        """
        Turns all of our paths into absolute paths, checks the data folder provided
        against the paths of the mrc and coord files.
        Returns lists mrc_paths, coord_paths
        """
        mrc_absolute_paths = False
        coord_absolute_paths = False

        if data_folder is not None:
            # get absolute path of data folder
            if not os.path.isabs(data_folder):
                data_folder = os.path.join(os.getcwd(), data_folder)
            # get first pair of files
            first_mrc, first_coord = files[0]
            if os.path.isabs(first_mrc):
                # check that abs paths to mrcs matches data folder
                if os.path.dirname(first_mrc) != data_folder:
                    raise ValueError(
                        f"data_folder provided ({data_folder}) does not match dirname of mrc files ({os.path.dirname(first_mrc)})"
                    )
                mrc_absolute_paths = True
            if os.path.isabs(first_coord):
                # check that abs paths to coords matches data folder
                if os.path.dirname(first_coord) != data_folder:
                    raise ValueError(
                        f"data_folder provided ({data_folder}) does not match dirname of coordinate files ({os.path.dirname(first_coord)})"
                    )
                coord_absolute_paths = True
        else:
            data_folder = os.getcwd()

        # split up the mrc paths from the coordinate file paths
        mrc_paths = [f[0] for f in files]
        coord_paths = [f[1] for f in files]
        # if we weren't given absolute paths, fill in the full paths
        if not mrc_absolute_paths:
            mrc_paths = [os.path.join(data_folder, m) for m in mrc_paths]
        if not coord_absolute_paths:
            coord_paths = [os.path.join(data_folder, c) for c in coord_paths]

        return mrc_paths, coord_paths


class EmanCoordinateSource(FromFilesCoordinateSource):
    """Eman specific implementations."""

    def __init__(self, files, data_folder, particle_size):
        super().__init__(self, files, data_folder, particle_size)

    def populate_mrc2coords(self, mrc_paths, coord_paths, particle_size):
        """
        Extract coordinates from .box format particles, which specify particles
        as 'lower_left_x lower_left_y size_x size_y'
        """
        # first do some input validation
        with open(coord_paths[0], "r") as first_coord_file:
            first_line = first_coord_file.readlines()[0]
            # box format requires 4 numbers per coordinate
            if len(first_line.split()) < 4:
                raise ValueError(
                    "Coordinate file contains less than 4 numbers per coordinate. If these are particle centers, run with centers=True"
                )
            # we can only accept square particles
            size_x, size_y = int(first_line.split()[2]), int(first_line.split()[3])
            if size_x != size_y:
                raise ValueError(
                    f"Coordinate file gives non-square particle size {size_x}x{size_y}, but only square particles are supported"
                )

        # for each mrc, read its corresponding coordinates file
        _mrc2coords = OrderedDict()
        for i, coord_path in enumerate(coord_paths):
            coord_list = []
            # We are reading particle centers from the coordinate file
            # We open the corresponding coordinate file
            with open(coord_path, "r") as coord_file:
                # each coordinate is a whitespace separated line in the file
                lines = coord_file.readlines()
            for line in lines:
                lower_left_x, lower_left_y, size_x, size_y = [
                    int(x) for x in line.split()
                ]
                coord = [lower_left_x, lower_left_y, size_x, size_y]
                coord_list.append(coord)
            _mrc2coords[mrc_paths[i]] = coord_list

        # if particle size set by user, we have to re-do the coordinates
        if particle_size > 0:
            # get original size
            old_size = size_x
            # user-specified size
            new_size = particle_size
            trim_length = (old_size - new_size) // 2
            _resized_mrc2coords = OrderedDict()
            for mrc, coordsList in _mrc2coords.items():
                _resized_mrc2coords[mrc] = []
                for coords in coordsList:
                    temp_coord = [-1, -1, new_size, new_size]
                    temp_coord[0] = coords[0] + trim_length
                    temp_coord[1] = coords[1] + trim_length
                    _resized_mrc2coords[mrc].append(temp_coord)
            self.mrc2coords = _resized_mrc2coords
        else:
            self.mrc2coords = _mrc2coords


class GuatoMatchCoordinateSource(FromFilesCoordinateSource):
    """Guato specific implementations."""

    def __init__(self, files, data_folder, particle_size, pixel_size, B, max_rows):
        super().__init__(self, files, data_folder, particle_size)

    def populate_mrc2coords(self, mrc_paths, coord_paths):
        """
        Extract coordinates from .coord files, which specify particles by the
        coordinates of their centers.
        """
        # for each mrc, read its corresponding coordinates file
        _mrc2coords = OrderedDict()
        for i, coord_path in enumerate(coord_paths):
            coord_list = []
            # We are reading particle centers from the coordinate file
            # We open the corresponding coordinate file
            with open(coord_path, "r") as coord_file:
                # each coordinate is a whitespace separated line in the file
                lines = coord_file.readlines()
            for line in lines:
                center_x, center_y = [int(x) for x in line.split()]
                # subtract off half the particle size to get the lower left of box
                coord = [
                    center_x - particle_size // 2,
                    center_y - particle_size // 2,
                    particle_size,
                    particle_size,
                ]
                coord_list.append(coord)
            _mrc2coords[mrc_paths[i]] = coord_list
        self.mrc2coords = _mrc2coords


class RelionCoordinateSource(CoordinateSourceBase):
    """Relion specific implementations."""

    def __init__(
        self, relion_autopick_star, data_folder, particle_size, pixel_size, B, max_rows
    ):
        self.populate_mrc2coords(relion_autopick_star, data_folder, particle_size)

    def populate_mrc2coords(self, relion_autopick_star, data_folder, particle_size):
        """
        Extract coordinates from a Relion autopick.star file. This STAR file contains
        a list of micrographs and corresponding coordinate files (also STAR files)
        """
        if data_folder is None:
            raise ValueError(
                "Provide Relion project direcetory when loading from Relion picked coordinates STAR file"
            )

        if not os.path.isabs(relion_autopick_star):
            relion_autopick_star = os.path.join(data_folder, relion_autopick_star)
        df = StarFile(relion_autopick_star)["coordinate_files"]
        files = list(zip(df["_rlnMicrographName"], df["_rlnMicrographCoordinates"]))
        mrc_paths = [os.path.join(data_folder, f[0]) for f in files]
        coord_paths = [os.path.join(data_folder, f[1]) for f in files]

        _mrc2coords = OrderedDict()
        for i in range(num_files):
            coordList = []
            df = StarFile(coord_paths[i]).get_block_by_index(0)
            x_coords = list(df["_rlnCoordinateX"])
            y_coords = list(df["_rlnCoordinateY"])
            # subtract off half of the particle size from center to get lower left
            particles = [
                [
                    int(float(x_coords[i])) - particle_size // 2,
                    int(float(y_coords[i])) - particle_size // 2,
                    particle_size,
                    particle_size,
                ]
                for i in range(len(df))
            ]
            for particle_coord in particles:
                coordList.append(particle_coord)
            _mrc2coords[mrc_paths[i]] = coordList

            self.mrc2coords = _mrc2coords


# potentially one or two of these are potentially different
class XYZProjectDirSource(RelionCoordinateSource):
    """just an example..."""


class CoordinateSource:
    # Factory for selecting and implementing a concrete subclass of CoordinateSourceBase
    # Pretty much it's only purpose is to select and return the right subclass of CoordinateSourceBase
    """Our User Facing Class ..."""

    def __init__(
        self,
        files=None,
        data_folder=None,
        particle_size=0,
        centers=False,
        pixel_size=1,
        B=0,
        max_rows=None,
        relion_autopick_star=None,
    ):
        """
        Based on arguments provided to __init__, returns an instance of a
        CoordinateSourceBase subclass
        :param files: A list of tuples (mrc_path, coord_path). Relative paths allowed.
        :param data_folder: Path to which filepaths provided are relative.
        :param particle_size: Desired size of cropped particles
        :param centers: Set to true if the coordinates represent particle centers
        :param pixel_size: Pixel size of  micrographs in Angstroms (default: 1)
        :param B: Envelope decay of the CTF in inverse Angstroms (default: 0)
        :param max_rows: Maximum bumber of particles to read. (If None, will attempt to load all particles)
        :param relion_autopick_star: Relion star file from AutoPick or ManualPick jobs (e.g. AutoPick/job006/autopick.star)
        """

        # If a relion_autopick starfile is specified, we are loading from a
        # Relion Autopick or ManualPick project directory, and the starfile gives us
        # the paths to the micrographs and coordinates relative to the project dir
        if relion_autopick_star:
            return RelionCoordinateSource(
                relion_autopick_star,
                data_folder,
                particle_size,
                pixel_size,
                B,
                max_rows,
            )

        # Otherwise, we are reading from .box or .coord files and the user must
        # must provide a list of tuples, files, matching micrographs to coords
        else:
            # if particle centers, we are generally reading a gautomatch .coord file
            if centers:
                if particle_size == 0:
                    raise ValueError(
                        "If reading particle centers, a particle_size must be specified"
                    )
                return GuatoMatchCoordinateSource(
                    files, data_folder, particle_size, pixel_size, B, max_rows
                )
            # otherwise it is in the EMAN-specified .box format
            else:
                return EmanCoordinateSource(
                    files, data_folder, particle_size, pixel_size, B, max_rows
                )


### When explaining usually people use different naming scheme
# my CoordinateSourceBase  would be  CoordinateSource
# my CoordinateSource      would be  CoordinateSourceFactory

## (because CoordinateSourceFactory is a factory that stamps out different CoordinateSource)
##   However, our users probably don't need to know about any of this... so we change the names
##   to protect the innocent.
