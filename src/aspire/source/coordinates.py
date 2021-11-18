import logging
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
import mrcfile
import numpy as np
import itertools

from aspire.operators import IdentityFilter
from aspire.source.image import ImageSource
from aspire.storage import StarFile
from aspire.image import Image

logger = logging.getLogger(__name__)


class CoordinateSourceBase(ImageSource, ABC):
    """Base Class defining CoordinateSource interface."""

    def __init__(
        self,
        mrc_paths,
        coord_paths,
        data_folder,
        particle_size,
        pixel_size,
        B,
        max_rows,
        dtype,
    ):

        self.dtype = np.dtype(dtype)

        # get first micrograph and coordinate list to report some data the user
        first_micrograph = mrc_paths[0]
        with mrcfile.open(first_micrograph) as mrc_file:
            mrc_dtype = np.dtype(mrc_file.data.dtype)
            shape = mrc_file.data.shape
        if len(shape) != 2:
            raise ValueError(
                "Shape of mrc file is {shape} but expected shape of size 2. Are these unaligned\
 micrographs?"
            )
        if self.dtype != mrc_dtype:
            logger.warn(
                f"dtype of micrograph is {mrc_dtype}. Will attempt to cast to {self.dtype}"
            )

        self.mrc_shape = shape

        logger.info(f"Micrograph size = {self.mrc_shape[1]}x{self.mrc_shape[0]}")
        # The internal representation of micrographs and their picked coords
        # is mrc2coords, an OrderedDict with micrograph filepaths as keys, and
        # lists of coordinates as values. Coordinates are lists of integers:
        # [lower left X, lower left Y, size X, size Y]. Different coordinate
        # files store this information differently, but all coordinates are
        # converted to this canonical format
        # this structure is populated by subclasses via the abstract method
        # CoordinateSourceBase.populate_mrc2coords()
        self.mrc2coords = OrderedDict()
        self.particle_size = particle_size
        self.populate_mrc2coords(mrc_paths, coord_paths)
        # total number of particles given in coord files
        # before removing those that do not fit
        self.original_n = sum(
            [len(coord_list) for mrc, coord_list in self.mrc2coords.items()]
        )

        # look at first coord to get particle size
        # here we're checking the final coordinate of the first particle
        # which is the Y-size of the box (the same as the X-size)
        first_coords = list(self.mrc2coords.items())[0][1]
        L = first_coords[0][3]
        logger.info(f"Particle size = {L}x{L}")
        self._original_resolution = L

        # remove particles whose boxes do not fit at given particle_size
        self.exclude_boundary_particles()
        # if max_rows is specified, mrc2coords will be cut down to contain
        # exactly max_rows particles
        self.get_n_particles(max_rows)

        # final number of particles in *this* source
        n = sum([len(self.mrc2coords[x]) for x in self.mrc2coords])
        logger.info(
            f"ParticleCoordinateSource from {data_folder} contains {len(mrc_paths)} micrographs, {self.original_n} picked particles."
        )
        if self.removed > 0:
            logger.info(
                f"{self.removed} particles did not fit into micrograph dimensions at particle size {L}, so were excluded. Maximum number of particles at this resolution is {self.original_n - self.removed}."
            )
        logger.info(f"ParticleCoordinateSource object contains {n} particles.")

        # create a flattened representation of the particles
        self.particles_flat = self.populate_particles()

        ImageSource.__init__(self, L=L, n=n, dtype=dtype)

        # Create filter indices for the source. These are required in order to
        # pass through the filter eval code.
        # Bypassing the filter_indices setter in ImageSource allows us
        # create this source with absolutely *no* metadata.
        # Otherwise, six default Relion columns are created w/defualt values
        self.set_metadata("__filter_indices", np.zeros(self.n, dtype=int))
        self.unique_filters = [IdentityFilter()]

    @abstractmethod
    def populate_mrc2coords(self):
        """
        Subclasses use this method to read coordinate files, validate them, and
        convert them into the canonical form in self.mrc2coords. Different
        sources store coordinate information differently, so the
        arguments and details of this method may vary.
        """

    def _populate_mrc2coords(self, mrc_paths, coord_paths):
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
                coord = self.convert_coords_to_box_format(line)
                coord_list.append(coord)
            _mrc2coords[mrc_paths[i]] = coord_list
        self.mrc2coords = _mrc2coords

    @abstractmethod
    def convert_coords_to_box_format(self, line):
        """
        Given a line from a coordinate file, convert the coordinates into the canonical format
        That is, a list of [lower left x, lower left y, x size, y size]
        Subclasses implement according to the details of the files they read
        """

    def _check_and_get_paths(self, files, data_folder):
        """
        Used in subclasses accepting the `files` kwarg
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

        return mrc_paths, coord_paths, data_folder

    def populate_particles(self):
        """
        Creates a flattened representation of all the particles in the source
        """
        particles_flat = []
        for mrc in self.mrc2coords.keys():
            for i, coord_list in enumerate(self.mrc2coords[mrc]):
                particles_flat.append((i, mrc))
        return particles_flat

    def exclude_boundary_particles(self):
        """
        Remove particles boxes which do not fit in the micrograph
        with the given particle_size
        """
        self.removed = 0
        for _mrc, coord_list in self.mrc2coords.items():
            out_of_range = []
            for i, coord in enumerate(coord_list):
                coord = coord_list[i]
                start_x, start_y, size_x, size_y = coord
                if (
                    start_x < 0
                    or start_y < 0
                    or (start_x + size_x >= self.mrc_shape[1])
                    or (start_y + size_y >= self.mrc_shape[0])
                ):
                    out_of_range.append(i)

            self.removed += len(out_of_range)

            # out_of_range stores the indices of the particles in the
            # unmodified coord_list that we must remove.
            # If we pop these indices of _coord list going forward, the
            # following indices will be shifted. Thus we pop in reverse, since
            # the indices prior to each removed index are unchanged
            for j in reversed(out_of_range):
                coordsList.pop(j)

    def get_n_particles(self, max_rows):
        """
        If the `max_rows` argument is given, this method will remove all
        but the first `max_rows` particles from the source (after boundary
        particles are removed). This may result in only some particles from
        one micrograph being included
        """
        # max_rows means max number of particles, but each micrograph has a
        # differing number of particles
        if max_rows:
            # we cannot get more particles than we actually have
            max_rows = min(max_rows, self.original_n - self.removed)
            # cumulative number of particles in each micrograph
            accum_lengths = list(
                itertools.accumulate([len(self.mrc2coords[d]) for d in self.mrc2coords])
            )
            # get the index of the micrograph that brings us over max_rows
            i_gt_max_rows = next(
                elem[0] for elem in enumerate(accum_lengths) if elem[1] > max_rows
            )
            # subtract off the difference
            remainder = max_rows - accum_lengths[i_gt_max_rows - 1]
            # get items of mrc2coords
            itms = list(self.mrc2coords.items())
            # include all the micrographs with coordinates that we don't
            # need to trim down to get exactly max_rows
            _tempdict = OrderedDict(
                {itms[i][0]: itms[i][1] for i in range(i_gt_max_rows)}
            )
            # add in the last micrograph, only up to 'remainder' particles
            _tempdict[itms[i_gt_max_rows][0]] = itms[i_gt_max_rows][1][:remainder]
            self.mrc2coords = _tempdict

    @staticmethod
    def crop_micrograph(data, coord):
        """
        Crops a particle box defined by `coord` out of `data`
        :param data: A 2D numpy array representing a micrograph
        :param coord: A list of integers: (lower left X, lower left Y, X, Y)
        """
        start_x, start_y, size_x, size_y = coord
        # according to MRC 2014 convention, origin represents
        # bottom-left corner of image
        return data[start_y : start_y + size_y, start_x : start_x + size_x]

    def _images(self, start=0, num=np.inf, indices=None):
        # the indices passed to this method refer to the index
        # of the *particle*, not the micrograph
        if indices is None:
            indices = np.arange(start, min(start + num, self.n))
        else:
            start = indices.min()
        logger.info(f"Loading {len(indices)} images from micrographs")

        # select the desired particles from this list
        _particles = [self.particles_flat[i] for i in indices]
        # initialize empty array to hold particle stack
        im = np.empty(
            (len(indices), self._original_resolution, self._original_resolution),
            dtype=self.dtype,
        )

        for i, particle in enumerate(_particles):
            # get the particle number and the micrograph
            num, fp = particle
            # load the image data for this micrograph
            arr = mrcfile.open(fp).data.astype(self.dtype)
            if arr.shape != self.mrc_shape:
                raise ValueError(
                    f"Shape of {fp} is {arr.shape}, but expected {self.mrc_shape}"
                )
            # get the specified particle coordinates
            coord = self.mrc2coords[fp][num]
            cropped = self.crop_micrograph(arr, coord)
            im[i] = cropped

        return Image(im)


class EmanCoordinateSource(CoordinateSourceBase):
    """Eman .box-format specific implementations."""

    def __init__(
        self, files, data_folder, particle_size, pixel_size, B, max_rows, dtype
    ):

        mrc_paths, coord_paths, data_folder = self._check_and_get_paths(
            files, data_folder
        )
        CoordinateSourceBase.__init__(
            self,
            mrc_paths,
            coord_paths,
            data_folder,
            particle_size,
            pixel_size,
            B,
            max_rows,
            dtype,
        )

    def populate_mrc2coords(self, mrc_paths, coord_paths):
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

        self._populate_mrc2coords(mrc_paths, coord_paths)

        # if particle size set by user, we have to re-do the coordinates
        if self.particle_size > 0:
            # original size from coordinate file
            old_size = size_x
            self.force_new_particle_size(self.particle_size, old_size)

    def convert_coords_to_box_format(self, line):
        # box format is the same as our canonical format
        # so we just read in the line as is
        return [int(x) for x in line.split()]

    def force_new_particle_size(self, new_size, old_size):
        """
        Given a new particle size, rewrite the coordinates so that the box shape
        is changed, but still centered around the particle
        """
        trim_length = (old_size - new_size) // 2
        _resized_mrc2coords = OrderedDict()
        for mrc, coordsList in self.mrc2coords.items():
            _resized_mrc2coords[mrc] = []
            for coords in coordsList:
                temp_coord = [-1, -1, new_size, new_size]
                temp_coord[0] = coords[0] + trim_length
                temp_coord[1] = coords[1] + trim_length
                _resized_mrc2coords[mrc].append(temp_coord)
        self.mrc2coords = _resized_mrc2coords


class CentersCoordinateSource(CoordinateSourceBase):
    """
    Code specifically handling data sources with coordinate files containing just particle centers
    """

    def __init__(
        self, files, data_folder, particle_size, pixel_size, B, max_rows, dtype
    ):
        mrc_paths, coord_paths, data_folder = self._check_and_get_paths(
            files, data_folder
        )
        CoordinateSourceBase.__init__(
            self,
            mrc_paths,
            coord_paths,
            data_folder,
            particle_size,
            pixel_size,
            B,
            max_rows,
            dtype,
        )

    def populate_mrc2coords(self, mrc_paths, coord_paths):
        """
        Extract coordinates from .coord files, which specify particles by the
        coordinates of their centers.
        """
        self._populate_mrc2coords(mrc_paths, coord_paths)

    def convert_coords_to_box_format(self, line):
        center_x, center_y = [int(x) for x in line.split()[:2]]
        # subtract off half the particle size to get the lower left of box
        return [
            center_x - self.particle_size // 2,
            center_y - self.particle_size // 2,
            self.particle_size,
            self.particle_size,
        ]


class RelionCoordinateSource(CoordinateSourceBase):
    """Relion specific implementations."""

    def __init__(
        self,
        relion_autopick_star,
        data_folder,
        particle_size,
        pixel_size,
        B,
        max_rows,
        dtype,
    ):
        if data_folder is None:
            raise ValueError(
                "Provide Relion project directory when loading from Relion picked coordinates STAR file"
            )
        if not os.path.isabs(relion_autopick_star):
            relion_autopick_star = os.path.join(data_folder, relion_autopick_star)
        df = StarFile(relion_autopick_star)["coordinate_files"]
        files = list(zip(df["_rlnMicrographName"], df["_rlnMicrographCoordinates"]))
        mrc_paths = [os.path.join(data_folder, f[0]) for f in files]
        coord_paths = [os.path.join(data_folder, f[1]) for f in files]
        CoordinateSourceBase.__init__(
            self,
            mrc_paths,
            coord_paths,
            data_folder,
            particle_size,
            pixel_size,
            B,
            max_rows,
            dtype,
        )

    def populate_mrc2coords(self, mrc_paths, coord_paths):
        """
        Extract coordinates from a Relion autopick.star file.
        This STAR file contains a list of micrographs and corresponding
        coordinate files (also STAR files)
        """
        _mrc2coords = OrderedDict()
        for i, coord_path in enumerate(coord_paths):
            coord_list = []
            df = StarFile(coord_path).get_block_by_index(0)
            x_coords = list(df["_rlnCoordinateX"])
            y_coords = list(df["_rlnCoordinateY"])
            # subtract off half of the particle size from center to get lower left
            particles = [
                [
                    int(float(x_coords[i])) - self.particle_size // 2,
                    int(float(y_coords[i])) - self.particle_size // 2,
                    self.particle_size,
                    self.particle_size,
                ]
                for i in range(len(df))
            ]
            for particle_coord in particles:
                coord_list.append(particle_coord)
            _mrc2coords[mrc_paths[i]] = coord_list

            self.mrc2coords = _mrc2coords

    def convert_coords_to_box_format(self):
        pass


# potentially one or two of these are potentially different
class XYZProjectDirSource(RelionCoordinateSource):
    """just an example..."""


class CoordinateSource:
    """
    User-facing interface for constructing a CoordinateSource. This class selects and returns
    an appropriate subclass of `CoordinateSourceBase` based on the arguments provided
    """

    # Factory for selecting and implementing a concrete subclass of CoordinateSourceBase
    # Pretty much it's only purpose is to select and return the right subclass of CoordinateSourceBase

    def __new__(
        self,
        files=None,
        data_folder=None,
        particle_size=0,
        centers=False,
        pixel_size=1,
        B=0,
        max_rows=None,
        relion_autopick_star=None,
        dtype="double",
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
        :param max_rows: Maximum number of particles to read. (If None, will attempt to load all particles)
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
                dtype,
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

                return CentersCoordinateSource(
                    files,
                    data_folder,
                    particle_size,
                    pixel_size,
                    B,
                    max_rows,
                    dtype,
                )

            # otherwise it is in the EMAN-specified .box format
            else:
                return EmanCoordinateSource(
                    files,
                    data_folder,
                    particle_size,
                    pixel_size,
                    B,
                    max_rows,
                    dtype,
                )

    def return_subclass(self, obj):
        """
        Takes the subclass passed by __init__ and simply returns it to the user
        __init__ can only return None
        """
        return obj


### When explaining usually people use different naming scheme
# my CoordinateSourceBase  would be  CoordinateSource
# my CoordinateSource      would be  CoordinateSourceFactory

## (because CoordinateSourceFactory is a factory that stamps out different CoordinateSource)
##   However, our users probably don't need to know about any of this... so we change the names
##   to protect the innocent.
