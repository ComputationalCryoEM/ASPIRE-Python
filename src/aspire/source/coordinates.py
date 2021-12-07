import logging
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

import mrcfile
import numpy as np

from aspire.image import Image
from aspire.operators import IdentityFilter
from aspire.source.image import ImageSource
from aspire.storage import StarFile

logger = logging.getLogger(__name__)


class CoordinateSource(ImageSource, ABC):
    """
    Base class defining common methods for data sources consisting of full
    micrographs coupled with files specifying the locations of picked
    particles in each micrograph.

    Broadly, there are two ways this information is represented. Sometimes each
    coordinate is simply the (X,Y) center location of the picked particle. This
    is sometimes stored in a `.coord` text file, and sometimes in a STAR file
    These sources may be loaded via the `CentersCoordinateSource` class for both
    filetypes.

    Other formats adhere to the EMAN1 .box file specification, which
    specifies a coordinate via four numbers:
    (lower left X coordinate, lower left Y coordinate, X size, Y size)
    These can be loaded via the `EmanCoordinateSource` class.

    Regardless of source, the coordinates of each particle are represented
    internally in the EMAN1 .box format.

    An addtional subclass exists for points in the Relion pipeline where
    particles in a micrograph are represented by coordinates, but not yet
    cropped out: `RelionCoordinateSource`. This class allows the output of
    AutoPick and ManualPick jobs to be loaded into an ASPIRE source from a
    single index STAR file (usually autopick.star)

    Particle information is extracted from the micrographs and coordinate files
    and put into a common data structure (self.particles)

    The `_images()` method, called via `ImageSource.images()` crops
    the particle images out of the micrograph and returns them as a stack.
    This also allows the CoordinateSource to be saved to an `.mrcs` stack.
    """

    def __init__(self, mrc_paths, coord_paths, particle_size, max_rows, dtype):
        self.dtype = np.dtype(dtype)
        self.particle_size = particle_size

        # keep this list to identify micrograph paths by index rather than
        # storing many copies of the same string
        self.mrc_paths = mrc_paths

        # The internal representation of micrographs and their picked coords
        # is a list of tuples (index of micrograph, coordinate), where
        # the coordinate is a list of the form:
        # [lower left X, lower left Y, size X, size Y].
        # The micrograph's filepath can be recovered from self.mrc_paths
        self.particles = []
        self.populate_particles(mrc_paths, coord_paths)

        # get first micrograph and first coordinate to report some data
        first_mrc_index, first_coord = self.particles[0]
        first_mrc = self.mrc_paths[first_mrc_index]
        with mrcfile.open(first_mrc) as mrc_file:
            mrc_dtype = np.dtype(mrc_file.data.dtype)
            shape = mrc_file.data.shape
        if len(shape) != 2:
            raise ValueError(
                "Shape of mrc file is {shape} but expected shape of size 2. Are these unaligned micrographs?"
            )
        if self.dtype != mrc_dtype:
            logger.warning(
                f"dtype of micrograph is {mrc_dtype}. Will attempt to cast to {self.dtype}"
            )

        # save the shape to compare the rest of the mrcs against
        self.mrc_shape = shape

        # look at first coord to get particle size
        # here we're checking the final coordinate of the first particle
        # which is the Y-size of the box (the same as the X-size)
        L = first_coord[3]

        # total number of particles given in coord files
        # before removing boundary particles and satisfying max_rows
        original_n = len(self.particles)

        logger.info(f"Micrograph size = {self.mrc_shape[1]}x{self.mrc_shape[0]}")
        logger.info(f"Particle size = {L}x{L}")
        self._original_resolution = L

        # remove particles whose boxes do not fit at given particle_size
        # and get number removed
        boundary_removed = self.exclude_boundary_particles()
        # if max_rows is specified, only return up to max_rows many
        # (after excluding boundary particles)
        if max_rows:
            max_rows = min(max_rows, original_n - boundary_removed)
            self.particles = self.particles[:max_rows]

        # final number of particles in *this* source
        n = len(self.particles)
        # total micrographs and particles represented by source (info)
        logger.info(
            f"{self.__class__.__name__} from {os.path.dirname(first_mrc)} contains {len(mrc_paths)} micrographs, {original_n} picked particles."
        )
        # total particles we can load given particle_size (info)
        if boundary_removed > 0:
            logger.info(
                f"{boundary_removed} particles did not fit into micrograph dimensions at particle size {L}, so were excluded"
            )
            logger.info(
                f"Maximum number of particles at this particle size is {original_n - boundary_removed}."
            )
        # total particles loaded (specific to this instance)
        logger.info(f"CoordinateSource object contains {n} particles.")

        ImageSource.__init__(self, L=L, n=n, dtype=dtype)

        # Create filter indices for the source. These are required in order to
        # pass through the filter eval code.
        # Bypassing the filter_indices setter in ImageSource allows us
        # create this source with absolutely *no* metadata.
        # Otherwise, six default Relion columns are created w/defualt values
        self.set_metadata("__filter_indices", np.zeros(self.n, dtype=int))
        self.unique_filters = [IdentityFilter()]

    @abstractmethod
    def populate_particles(self):
        """
        Subclasses use this method to read coordinate files, validate them, and
        convert them into the canonical form in self.particles. Different
        sources store coordinate information differently, so the
        arguments and details of this method may vary.
        """

    def _populate_particles(self, mrc_paths, coord_paths):
        """
        All subclasses create mrc_paths and coord_paths lists and pass them to
        this method.
        """
        for i, _mrc in enumerate(mrc_paths):
            # read in all coordinates for the given mrc using subclass's
            # method of reading the corresponding coord file
            self.particles += [
                (i, coord) for coord in self.coords_list_from_file(coord_paths[i])
            ]

    @abstractmethod
    def coords_list_from_file(self, coord_file):
        """
        Given a coordinate file, convert the coordinates into the canonical format, that is, a
        list of the form [lower left x, lower left y, x size, y size].
        Subclasses implement according to the details of the files they read.
        """

    def coords_list_from_star(self, star_file):
        """
        Given a Relion STAR coordinate file (generally containing particle centers)
        return a list of coordinates in the canonical (.box) format
        """
        df = StarFile(star_file).get_block_by_index(0)
        coords = list(zip(df["_rlnCoordinateX"], df["_rlnCoordinateY"]))
        # subtract off half of particle size from center coord
        # populate final two coordinates with the particle_size
        # Relion coordinates are represented as floats. Here we are reading
        # the value as a float and then intentionally taking the floor
        # of the result 
        return [
            list(map(lambda x: int(float(x)) - self.particle_size // 2, coord[:2]))
            + [self.particle_size] * 2
            for coord in coords
        ]

    def _check_and_get_paths(self, files):
        """
        Used in subclasses accepting the `files` kwarg.
        Turns all of our paths into absolute paths.
        Returns lists mrc_paths, coord_paths
        """
        # split up the mrc paths from the coordinate file paths
        mrc_paths = [f[0] for f in files]
        coord_paths = [f[1] for f in files]
        # check whether we were given absolute paths
        mrc_absolute_paths = os.path.isabs(mrc_paths[0])
        coord_absolute_paths = os.path.isabs(coord_paths[0])
        # if we weren't given absolute paths, fill in the full paths
        if not mrc_absolute_paths:
            mrc_paths = [os.path.join(os.getcwd(), m) for m in mrc_paths]
        if not coord_absolute_paths:
            coord_paths = [os.path.join(os.getcwd(), c) for c in coord_paths]

        return mrc_paths, coord_paths

    def exclude_boundary_particles(self):
        """
        Remove particles boxes which do not fit in the micrograph
        with the given particle_size
        """
        out_of_range = []
        for i, particle in enumerate(self.particles):
            start_x, start_y, size_x, size_y = particle[1]
            if (
                start_x < 0
                or start_y < 0
                or (start_x + size_x >= self.mrc_shape[1])
                or (start_y + size_y >= self.mrc_shape[0])
            ):
                out_of_range.append(i)

        # out_of_range stores the indices of the particles in the
        # unmodified coord_list that we must remove.
        # If we pop these indices of _coord list going forward, the
        # following indices will be shifted. Thus we pop in reverse, since
        # the indices prior to each removed index are unchanged
        for j in reversed(out_of_range):
            self.particles.pop(j)

        return len(out_of_range)

    @staticmethod
    def crop_micrograph(data, coord):
        """
        Crops a particle box defined by `coord` out of `data`
        According to MRC 2014 convention, origin represents the bottom-left
        corner of the image
        :param data: A 2D numpy array representing a micrograph
        :param coord: A list of integers: (lower left X, lower left Y, X, Y)
        """
        start_x, start_y, size_x, size_y = coord
        return data[start_y : start_y + size_y, start_x : start_x + size_x]

    def _images(self, start=0, num=np.inf, indices=None):
        """
        Given a range or selection of indices, returns an Image stack
        of the particles specified. Note that the indices refer to the order
        of the particles loaded in this *specific* CoordinateSource. This may
        not correspond to the particles in the original source on disk, if some
        particles were excluded due to their box not fitting into the mrc
        dimensions. Thus, the exact particles returned are a function of the
        particle_size.
        :param start: Starting index (default: 0)
        :param num: number of images to return starting from `start` (default: numpy.inf)
        :param indices: A numpy array of integer indices
        """
        if indices is None:
            indices = np.arange(start, min(start + num, self.n))

        logger.info(f"Loading {len(indices)} images from micrographs")

        selected_particles = [self.particles[i] for i in indices]
        # initialize empty array to hold particle stack
        im = np.empty(
            (len(indices), self._original_resolution, self._original_resolution),
            dtype=self.dtype,
        )

        # group particles by micrograph in order to
        # only open each one once
        grouped = defaultdict(list)
        # this creates a dict of the form
        # { mrc_index : list of coords in that mrc, with order preserved }
        for mrc_index, coord in selected_particles:
            grouped[mrc_index].append(coord)

        for mrc_index, coord_list in grouped.items():
            # get explicit filepath from cached list
            fp = self.mrc_paths[mrc_index]
            with mrcfile.open(fp) as mrc_in:
                arr = mrc_in.data.astype(self.dtype)
            if arr.shape != self.mrc_shape:
                raise ValueError(
                    f"Shape of {fp} is {arr.shape}, but expected {self.mrc_shape}"
                )
            # create iterable of the coordinates in this mrc
            # we don't need to worry about exhausting this iter
            # because we know it contains the exact number of particles
            # selected from this micrograph
            coord = iter(coord_list)
            # iterate through selected particles
            for i, particle in enumerate(selected_particles):
                idx = particle[0]
                # we stop and populate the image stack every time
                # we hit a particle whose location is this micrograph
                if idx == mrc_index:
                    cropped = self.crop_micrograph(arr, next(coord))
                    im[i] = cropped

        return Image(im)


class EmanCoordinateSource(CoordinateSource):
    """
    Represents a data source consisting of micrographs and coordinate files
    in EMAN1 .box format.
    """

    def __init__(
        self,
        files,
        particle_size=0,
        max_rows=None,
        dtype="double",
    ):
        """
        :param files: A list of tuples of the form (path_to_mrc, path_to_coord)
        :particle_size: Desired size of cropped particles (will override the size specified in coordinate file)
        :param max_rows: Maximum number of particles to read. (If `None`, will attempt to load all particles)
        :param dtype: dtype with which to load images (default: double)
        """

        # get full filepaths and data folder
        mrc_paths, coord_paths = self._check_and_get_paths(files)
        # instantiate super
        CoordinateSource.__init__(
            self,
            mrc_paths,
            coord_paths,
            particle_size,
            max_rows,
            dtype,
        )

    def populate_particles(self, mrc_paths, coord_paths):
        """
        Extract coordinates from .box format particles, which specify particles
        and populate self.particles
        """
        # Look into the first coordinate path given and validate format
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

        self._populate_particles(mrc_paths, coord_paths)

        # if particle size set by user, we have to re-do the coordinates
        if self.particle_size > 0:
            # original size from coordinate file
            old_size = size_x
            self.force_new_particle_size(self.particle_size, old_size)

    def coords_list_from_file(self, coord_file):
        """
        Given a coordinate file in .box format, returns a list of coordinates
        """
        with open(coord_file, "r") as infile:
            lines = [line.split() for line in infile.readlines()]
        # coords are already in canonical .box format, so simply cast to int
        return [[int(x) for x in line] for line in lines]

    def force_new_particle_size(self, new_size, old_size):
        """
        Given a new particle size, rewrite the coordinates so that the box size
        is changed, but still centered around the particle
        """
        _resized_particles = []
        for particle in self.particles:
            mrc_index, coord = particle
            _resized_particles.append(
                (
                    mrc_index,
                    [
                        coord[0] + (old_size - new_size) // 2,
                        coord[1] + (old_size - new_size) // 2,
                        new_size,
                        new_size,
                    ],
                )
            )
        self.particles = _resized_particles


class CentersCoordinateSource(CoordinateSource):
    """
    Represents a data source consisting of micrographs and coordinate files specifying particle centers only. Files can be text (.coord) or STAR files.
    """

    def __init__(
        self,
        files,
        particle_size,
        max_rows=None,
        dtype="double",
    ):
        """
        :param files: A list of tuples of the form (path_to_mrc, path_to_coord)
        :particle_size: Desired size of cropped particles (will override the size specified in coordinate file).
        :param max_rows: Maximum number of particles to read. (If `None`, will
        attempt to load all particles)
        :param dtype: dtype with which to load images (default: double)
        """
        # get full filepaths and data folder
        mrc_paths, coord_paths = self._check_and_get_paths(files)
        # instantiate super
        CoordinateSource.__init__(
            self, mrc_paths, coord_paths, particle_size, max_rows, dtype
        )

    def populate_particles(self, mrc_paths, coord_paths):
        """
        Extract coordinates from .coord files, which specify particles by the
        coordinates of their centers.
        """
        self._populate_particles(mrc_paths, coord_paths)

    def coords_list_from_file(self, coord_file):
        """
        Given a coordinate file with (x,y) particle centers,
        return a list of coordinates in our canonical (.box) format
        """
        # check if it's a STAR file list of centers
        if os.path.splitext(coord_file)[1] == ".star":
            return self.coords_list_from_star(coord_file)
        # otherwise we assume text file format with one coord per line:
        with open(coord_file, "r") as infile:
            lines = [line.split() for line in infile.readlines()]
        # subtract off half of particle size from center coord
        # populate final two coordinates with the particle_size
        return [
            list(map(lambda x: int(x) - self.particle_size // 2, line[:2]))
            + [self.particle_size] * 2
            for line in lines
        ]


class RelionCoordinateSource(CoordinateSource):
    """
    Represents a data source derived from an autopick.star file within a Relion
    project directory.
    """

    def __init__(
        self, relion_autopick_star, particle_size, max_rows=None, dtype="double"
    ):
        """
                :param files: Relion STAR file e.g. autopick.star mapping micrographs to coordinate STAR files.
                :particle_size: Desired size of cropped particles
                :param max_rows: Maximum number of particles to read. (If `None`, will
        attempt to load all particles)
                :param dtype: dtype with which to load images (default: double)
        """

        # if not absolute path to star file, assume relative to working dir
        if not os.path.isabs(relion_autopick_star):
            relion_autopick_star = os.path.join(os.getcwd(), relion_autopick_star)

        # the 'coordinate_files' block of the starfile specifies
        # paths to micrographs and coordinate files relative to the
        # Relion project dir
        df = StarFile(relion_autopick_star)["coordinate_files"]
        files = list(zip(df["_rlnMicrographName"], df["_rlnMicrographCoordinates"]))

        # infer relion project dir since autopick.star will be at e.g.
        # /relion/project/dir/Autopick/job00X/autopick.star
        # get path 3 directories up
        data_folder = Path(relion_autopick_star).parents[2]

        # get absolute paths based on project dir
        mrc_paths = [os.path.join(data_folder, f[0]) for f in files]
        coord_paths = [os.path.join(data_folder, f[1]) for f in files]

        # instantiate super
        CoordinateSource.__init__(
            self, mrc_paths, coord_paths, particle_size, max_rows, dtype
        )

    def populate_particles(self, mrc_paths, coord_paths):
        self._populate_particles(mrc_paths, coord_paths)

    def coords_list_from_file(self, coord_file):
        return self.coords_list_from_star(coord_file)
