import logging
import os
from abc import ABC, abstractmethod

import mrcfile
import numpy as np

from aspire.image import Image
from aspire.operators import IdentityFilter
from aspire.source.image import ImageSource
from aspire.storage import StarFile

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
        self.particle_size = particle_size
        # The internal representation of micrographs and their picked coords
        # is a list of tuples (micrograph_filepath, coordinate), where the coordinate
        # is a list of the form [lower left X, lower left Y, size X, size Y].
        self.particles = []
        self.populate_particles(mrc_paths, coord_paths)

        # get first micrograph and first coordinate to report some data the user
        first_mrc, first_coord = self.particles[0]
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
        boundary_removed = self.exclude_boundary_particles()
        # if max_rows is specified, only return up to max_rows many
        # (after excluding boundary particles)
        if max_rows:
            max_rows = min(max_rows, original_n - boundary_removed)
            self.particles = self.particles[:max_rows]

        # final number of particles in *this* source
        n = len(self.particles)
        logger.info(
            f"ParticleCoordinateSource from {data_folder} contains {len(mrc_paths)} micrographs, {original_n} picked particles."
        )

        if boundary_removed > 0:
            logger.info(
                f"{boundary_removed} particles did not fit into micrograph dimensions at particle size {L}, so were excluded"
            )
            logger.info(
                f"Maximum number of particles at this particle size is {original_n - boundary_removed}."
            )

        logger.info(f"ParticleCoordinateSource object contains {n} particles.")

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
        # for each mrc, read its corresponding coordinates file
        for i, mrc_path in enumerate(mrc_paths):
            self.particles += [
                (mrc_path, coord)
                for coord in self.coords_list_from_file(coord_paths[i])
            ]

    @abstractmethod
    def coords_list_from_file(self, coord_file):
        """
        Given a coordinate file, convert the coordinates into the canonical format, that is, a
        list of [lower left x, lower left y, x size, y size
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
        # the indices passed to this method refer to the index
        # of the *particle*, not the micrograph
        if indices is None:
            indices = np.arange(start, min(start + num, self.n))
        else:
            start = indices.min()
        logger.info(f"Loading {len(indices)} images from micrographs")

        selected_particles = [self.particles[i] for i in indices]
        # initialize empty array to hold particle stack
        im = np.empty(
            (len(indices), self._original_resolution, self._original_resolution),
            dtype=self.dtype,
        )

        for i, particle in enumerate(selected_particles):
            # get the particle number and the micrograph
            fp, coord = particle
            # load the image data for this micrograph
            arr = mrcfile.open(fp).data.astype(self.dtype)
            if arr.shape != self.mrc_shape:
                raise ValueError(
                    f"Shape of {fp} is {arr.shape}, but expected {self.mrc_shape}"
                )
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

    def populate_particles(self, mrc_paths, coord_paths):
        """
        Extract coordinates from .box format particles, which specify particles
        as 'lower_left_x lower_left_y size_x size_y'
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
        with open(coord_file, "r") as infile:
            lines = [line.split() for line in infile.readlines()]
        return [[int(x) for x in line] for line in lines]

    def force_new_particle_size(self, new_size, old_size):
        """
        Given a new particle size, rewrite the coordinates so that the box shape
        is changed, but still centered around the particle
        """
        trim_length = (old_size - new_size) // 2
        _resized_particles = []
        for particle in self.particles:
            fp, coord = particle
            _resized_particles.append(
                (
                    fp,
                    [
                        coord[0] + trim_length,
                        coord[1] + trim_length,
                        new_size,
                        new_size,
                    ],
                )
            )
        self.particles = _resized_particles


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

    def populate_particles(self, mrc_paths, coord_paths):
        """
        Extract coordinates from .coord files, which specify particles by the
        coordinates of their centers.
        """
        self._populate_particles(mrc_paths, coord_paths)

    def coords_list_from_file(self, coord_file):
        # subtract off half of particle size from center coord
        # populate final two coordinates with the particle_size
        with open(coord_file, "r") as infile:
            lines = [line.split() for line in infile.readlines()]
        return [
            list(map(lambda x: int(x) - self.particle_size // 2, line[:2]))
            + [self.particle_size] * 2
            for line in lines
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

    def populate_particles(self, mrc_paths, coord_paths):
        self._populate_particles(mrc_paths, coord_paths)

    def coords_list_from_file(self, coord_file):
        df = StarFile(coord_file).get_block_by_index(0)
        coords = list(zip(df["_rlnCoordinateX"], df["_rlnCoordinateY"]))
        return [
            list(map(lambda x: int(x) - self.particle_size // 2, coord[:2]))
            + [self.particle_size] * 2
            for coord in coords
        ]


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
