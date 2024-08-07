import logging
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from math import floor

import mrcfile
import numpy as np

from aspire.image import Image
from aspire.operators import CTFFilter, IdentityFilter
from aspire.source.image import ImageSource
from aspire.storage import StarFile
from aspire.utils import RelionStarFile

logger = logging.getLogger(__name__)


class CoordinateSource(ImageSource, ABC):
    """
    Base class defining common methods for data sources consisting of full
    micrographs coupled with files specifying the locations of picked
    particles in each micrograph.

    Broadly, there are two ways this information is represented. Sometimes each
    coordinate is simply the (X,Y) center location of the picked particle. This
    is sometimes stored in a `.coord` text file, and sometimes in a STAR file.
    The particle box is then computed using an externally supplied box size.
    These sources may be loaded via the `CentersCoordinateSource` class for
    both filetypes.

    Other formats adhere to the box file specification, which
    specifies a particle via four numbers:
    (lower left X coordinate, lower left Y coordinate, X size, Y size).
    These can be loaded via the `BoxesCoordinateSource` class.

    Regardless of source, the coordinates of each particle are represented
    internally in the box format.

    Particle information is extracted from the micrographs and coordinate files
    and put into a common data structure (self.particles).

    The `_images()` method, called via `ImageSource.images()` crops
    the particle images out of the micrograph and returns them as a stack.
    This also allows the CoordinateSource to be saved to an `.mrcs` stack.
    """

    def __init__(self, files, particle_size, max_rows, B, symmetry_group):
        mrc_paths, coord_paths = [f[0] for f in files], [f[1] for f in files]
        # the particle_size parameter is the *user-specified* argument
        # and is used in self._populate_particles
        # it may be None in the case of an BoxesCoordinateSource
        self.particle_size = particle_size

        # keep this list to identify micrograph paths by index rather than
        # storing many copies of the same string
        self.mrc_paths = mrc_paths
        self.num_micrographs = len(self.mrc_paths)

        # The internal representation of micrographs and their picked coords
        # is a list of tuples (index of micrograph, coordinate), where
        # the coordinate is a list of the form:
        # [lower left X, lower left Y, size X, size Y].
        # The micrograph's filepath can be recovered from self.mrc_paths
        self.particles = []
        self._populate_particles(coord_paths)

        # get first mrc and coordinate file to report some data
        first_mrc_index, first_coord = self.particles[0]
        first_mrc = self.mrc_paths[first_mrc_index]
        self._ext = os.path.splitext(first_mrc)[1]
        if self._ext not in Image.extensions:
            raise RuntimeError(f"Unsupported file type {self.ext}.")

        # Default non MRC files (eg, TIFF) to single precision.
        dtype = "float32"
        # If file is MRC, we can get dtype from header.
        if self._ext == ".mrc":
            with mrcfile.open(first_mrc) as mrc:
                # get dtype from first micrograph
                mode = int(mrc.header.mode)
                dtypes = {0: "int8", 1: "int16", 2: "float32", 6: "uint16"}
                assert (
                    mode in dtypes
                ), f"Only modes={list(dtypes.keys())} in MRC files are supported for now."
                dtype = dtypes[mode]

        # Read shapes of all micrographs
        self.mrc_shapes = self._get_mrc_shapes()

        # look at first coord to get the particle size
        # this was either provided by the user or read from a .box file
        # here we're checking the final coordinate of the first particle
        # which is the Y-size of the box (the same as the X-size)
        # this is not the same as the argument particle_size
        # which can be None
        L = first_coord[3]

        logger.info(f"Particle size = {L}x{L}")
        self._original_resolution = L

        # total micrographs and particles represented by source (info)
        logger.info(
            f"{self.__class__.__name__} from {os.path.dirname(self.mrc_paths[0])} contains {self.num_micrographs} micrographs, {len(self.particles)} picked particles."
        )
        # report different mrc shapes
        logger.info(
            f"Micrographs have the following shapes: {np.unique(self.mrc_shapes, axis=0)}"
        )

        # remove particles whose boxes do not fit at given particle_size
        # and get number removed
        boundary_removed = self._exclude_boundary_particles()

        # total particles we can load given particle_size (info)
        if boundary_removed > 0:
            logger.info(
                f"{boundary_removed} particles did not fit into micrograph dimensions at particle size {L}, so were excluded"
            )
            logger.info(
                f"Maximum number of particles at this particle size is {len(self.particles)}."
            )

        # if max_rows is specified, only return up to max_rows many
        # (after excluding boundary particles)
        if max_rows:
            max_rows = min(max_rows, len(self.particles))
            self.particles = self.particles[:max_rows]

        # final number of particles in *this* source
        n = len(self.particles)

        # total particles loaded (specific to this instance)
        logger.info(f"CoordinateSource object contains {n} particles.")

        ImageSource.__init__(self, L=L, n=n, dtype=dtype, symmetry_group=symmetry_group)

        # map mrc indices to particle indices
        # i'th element contains a list of particle indices corresponding to i'th mrc
        self.mrc_index_to_particles = []
        for mrc_idx in range(self.num_micrographs):
            self.mrc_index_to_particles.append(
                [
                    particle_idx
                    for particle_idx, particle in enumerate(self.particles)
                    if particle[0] == mrc_idx
                ]
            )

        # CTF envelope decay factor
        self.B = B
        # set CTF metadata to defaults
        # this can be updated with import_ctf()
        self.filter_indices = np.zeros(self.n, dtype=int)
        self.unique_filters = [IdentityFilter()]
        self.set_metadata("__filter_indices", np.zeros(self.n, dtype=int))

        # populate __mrc_filename and __mrc_index
        for mrc_index, particle_indices in enumerate(self.mrc_index_to_particles):
            self.set_metadata(
                "__mrc_index",
                mrc_index,
                particle_indices,
            )
            self.set_metadata(
                "__mrc_filepath", self.mrc_paths[mrc_index], particle_indices
            )

        # Any further operations should not mutate this instance.
        self._mutable = False

    def _populate_particles(self, coord_paths):
        """
        All subclasses create mrc_paths and coord_paths lists and pass them to
        this method.
        """
        for i in range(self.num_micrographs):
            # read in all coordinates for the given mrc using subclass's
            # method of reading the corresponding coord file
            self.particles += [
                (i, coord) for coord in self._coords_list_from_file(coord_paths[i])
            ]

    @abstractmethod
    def _coords_list_from_file(self, coord_file):
        """
        Given a coordinate file, convert the coordinates into box format, i.e. a
        list of the form [lower left x, lower left y, x size, y size].
        Subclasses implement according to the details of the files they read.
        """

    @staticmethod
    def _box_coord_from_center(center, particle_size):
        """
        Convert a list `[x,y]` representing a particle center
        to a list
        `[lower left x, lower left y, particle_size, particle_size]`
        representing the box around the particle in box format.

        :param center: a list of length two representing a center
        :param particle_size: the size of the box around the particle
        """
        # subtract off floor(particle size/2) from center coords
        r = floor(particle_size / 2)
        x, y = center[:2]
        return [
            # centers may be represented as floats in STAR files
            # chop off the non integer part to account for this
            int(x) - r,
            int(y) - r,
            particle_size,
            particle_size,
        ]

    @staticmethod
    def _center_from_box_coord(box_coord):
        """
        Convert a list
        `[lower left x, lower left y, particle_size, particle_size]`
        representing a particle in the box format to a list
        `[x, y]` representing the particle center.

        :param box_coord: a list of length 4 representing the particle box
        """
        # Get lower left corner x and y coordinates
        # and particle size from the box coordinate
        llx, lly, particle_size = box_coord[:3]
        # turn lower left corner coordinate into center coordinate
        r = floor(particle_size / 2)
        return [llx + r, lly + r]

    def _coords_list_from_star(self, star_file):
        """
        Given a Relion STAR coordinate file (generally containing particle centers)
        return a list of coordinates in box format.

        :param star_file: A path to a STAR file containing particle centers
        """
        data_block = StarFile(star_file).get_block_by_index(0)
        coords = list(zip(data_block["_rlnCoordinateX"], data_block["_rlnCoordinateY"]))
        coords = [(float(x), float(y)) for x, y in coords]
        return [
            self._box_coord_from_center(coord, self.particle_size) for coord in coords
        ]

    def _populate_local_metadata(self):
        """
        Called during ImageSource.save(), populates metadata columns specific to
            `CoordinateSource` when saving to STAR file.

        :return: A list of the names of the columns added.
        """
        # Insert stored particle coordinates (centers) into metadata
        self.set_metadata(
            "_rlnCoordinateX",
            [
                self._center_from_box_coord(particle[1])[0]
                for particle in self.particles
            ],
        )
        self.set_metadata(
            "_rlnCoordinateY",
            [
                self._center_from_box_coord(particle[1])[1]
                for particle in self.particles
            ],
        )
        return ["_rlnCoordinateX", "_rlnCoordinateY"]

    def _exclude_boundary_particles(self):
        """
        Remove particles boxes which do not fit in the micrograph
        with the given `particle_size`.

        :return: Number of particles removed
        """
        out_of_range = []
        for i, particle in enumerate(self.particles):
            start_x, start_y, size_x, size_y = particle[1]
            # get shape of corresponding micrograph
            mrc_index = particle[0]
            mrc_shape = self.mrc_shapes[mrc_index]
            if (
                start_x < 0
                or start_y < 0
                or (start_x + size_x >= mrc_shape[1])
                or (start_y + size_y >= mrc_shape[0])
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

    def _get_mrc_shapes(self):
        """
        Iterate through self.mrc_paths and read the dimensions of each micrograph.

        :return mrc_shapes: A list of tuples representing the corresponding shapes
        """

        mrc_shapes = np.zeros((self.num_micrographs, 2), dtype=int)
        for i, mrc in enumerate(self.mrc_paths):
            mrc_shapes[i, :] = Image.load(mrc).resolution

        return mrc_shapes

    def import_aspire_ctf(self, ctf):
        """
        Import CTF information from STAR file(s) generated by ASPIRE's CTF Estimator.

        :param ctf: A path or iterable of paths to STAR files from ASPIRE's CTF Estimator.
            Note that number of files provided must match number of micrographs in this
            `CoordinateSource`.
        """
        if not isinstance(ctf, Iterable):
            ctf = [ctf]
        if not len(ctf) == self.num_micrographs:
            raise ValueError(
                "Number of CTF STAR files must match number of micrographs."
            )

        # merge dicts from CTF files
        data_blocks = defaultdict(list)
        for f in ctf:
            # ASPIRE's CTF Estimator produces legacy (=< 3.0) STAR files containing one row
            star = RelionStarFile(f)
            data_block = star.data_block
            for k, v in data_block.items():
                data_blocks[k].append(v)

        self._extract_ctf(data_blocks)

    def import_relion_ctf(self, ctf):
        """
        Import CTF information Relion micrograph STAR files containing CTF information.

        :param ctf: Path to a Relion micrograph STAR file containing CTF information.
            Note that number of files provided must match number of micrographs in this
            `CoordinateSource`.
        """
        data_block = RelionStarFile(ctf).get_merged_data_block()

        # data_block is a dict containing the micrographs
        if not len(list(data_block.values())[0]) == self.num_micrographs:
            raise ValueError(
                f"{ctf} has CTF information for {len(data_block)}",
                f" micrographs but this source has {self.num_micrographs} micrographs.",
            )

        self._extract_ctf(data_block)

    def _extract_ctf(self, data_block):
        """
        Receives a dict containing micrograph CTF information, and populates
            the Source's CTF Filters, filter indices, and metadata.
        """
        # required CTF params excluding pixel size
        CTF_params = [
            "_rlnVoltage",
            "_rlnDefocusU",
            "_rlnDefocusV",
            "_rlnDefocusAngle",
            "_rlnSphericalAberration",
            "_rlnAmplitudeContrast",
            "_rlnMicrographPixelSize",
        ]

        # get unique ctfs from the data block
        # i'th entry of `indices` contains the index of `filter_params` with corresponding CTF params
        ctf_data = np.stack([data_block[c] for c in CTF_params]).astype(self.dtype).T
        filter_params, indices = np.unique(
            ctf_data,
            return_inverse=True,
            axis=0,
        )

        # convert defocus_ang from degrees to radians
        filter_params[:, 3] *= np.pi / 180.0

        # construct filters
        self.unique_filters = [
            CTFFilter(
                pixel_size=filter_params[i, 6],
                voltage=filter_params[i, 0],
                defocus_u=filter_params[i, 1],
                defocus_v=filter_params[i, 2],
                defocus_ang=filter_params[i, 3],
                Cs=filter_params[i, 4],
                alpha=filter_params[i, 5],
                B=self.B,
            )
            for i in range(len(filter_params))
        ]

        # set metadata
        for mrc_idx, filter_index in enumerate(indices):
            particle_indices_this_mrc = self.mrc_index_to_particles[mrc_idx]
            num_particles_this_mrc = len(particle_indices_this_mrc)
            self.set_metadata(
                CTF_params,
                np.vstack((filter_params[filter_index],) * num_particles_this_mrc),
                particle_indices_this_mrc,
            )
            self.set_metadata(
                "__filter_indices", filter_index, particle_indices_this_mrc
            )

    @staticmethod
    def _crop_micrograph(data, coord):
        """
        Crops a particle box defined by `coord` out of `data`.
        According to MRC 2014 convention, the origin represents the bottom-left
        corner of the image.

        :param data: A 2D numpy array representing a micrograph
        :param coord: A list of integers: (lower left X, lower left Y, X, Y)
        """
        start_x, start_y, size_x, size_y = coord
        return data[start_y : start_y + size_y, start_x : start_x + size_x]

    def _images(self, indices):
        """
        Given a range or selection of indices, returns an Image stack
        when accessed via the `ImageSource.images` property.
        Note that the indices refer to the order
        of the particles loaded in this *specific* CoordinateSource. This may
        not correspond to the particles in the original source on disk, if some
        particles were excluded due to their box not fitting into the mrc
        dimensions. Thus, the exact particles returned are a function of the
        `particle_size`.

        :param indices: A 1-D NumPy array of integer indices.
        :return: An `Image` object.
        """
        # check for cached images first
        if self._cached_im is not None:
            logger.info("Loading images from cache")
            return self.generation_pipeline.forward(
                self._cached_im[indices, :, :], indices
            )

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
        # { mrc_index : list of coords in that mrc, with relative order preserved }
        for mrc_index, coord in selected_particles:
            grouped[mrc_index].append(coord)

        # this loops over the micrographs
        # now that the particles have been grouped by
        # their origin micrograph
        for mrc_index, coord_list in grouped.items():
            # Load file as 2D numpy array.
            arr = Image.load(self.mrc_paths[mrc_index]).asnumpy()[0]

            # create iterable of the coordinates in this mrc
            # we don't need to worry about exhausting this iter
            # because we know it contains the exact number of particles
            # selected from this micrograph.
            # the next()  method on this iterable will give us the next
            # particle from this micrograph to appear in selected_particles
            coord = iter(coord_list)
            # iterate through selected particles
            # we are potentially populating the particles
            # out of order, to optimize the slower operation
            # of opening the micrograph file
            for i, particle in enumerate(selected_particles):
                idx = particle[0]
                # we stop and populate the image stack every time
                # we hit a particle whose location is this micrograph
                if idx == mrc_index:
                    cropped = self._crop_micrograph(arr, next(coord))
                    im[i] = cropped
        # Finally, apply transforms to resulting Image
        return self.generation_pipeline.forward(
            Image(im, pixel_size=self.pixel_size), indices
        )

    @staticmethod
    def _is_number(text):
        """
        Used in validation of coordinate files. We allow strings containing
        - or . to account for negative values and floats.
        """
        return text.replace("-", "1").replace(".", "1").isdigit()


class BoxesCoordinateSource(CoordinateSource):
    """
    Represents a data source consisting of micrographs and coordinate files
    in box format.
    """

    def __init__(
        self,
        files,
        particle_size=None,
        max_rows=None,
        B=0,
        symmetry_group=None,
    ):
        """
        :param files: A list of tuples of the form (path_to_mrc, path_to_coord)
        :particle_size: Desired size of cropped particles (will override the size specified in coordinate file)
        :param max_rows: Maximum number of particles to read. (If `None`, will attempt to load all particles)
        :param symmetry_group: A `SymmetryGroup` object or string corresponding to the symmetry of the molecule.
        """
        # instantiate super
        CoordinateSource.__init__(
            self, files, particle_size, max_rows, B, symmetry_group
        )

    def _extract_box_size(self, box_file):
        with open(box_file, "r") as box:
            first_line = box.readlines()[0].split()
            if len(first_line) >= 4:
                box_size = int(float(first_line[2]))  # x size or y size works
                return box_size
            else:
                logger.error(f"Problem with coordinate file: {box_file}")
                raise ValueError(
                    "Coordinate file contains less than 4 numbers "
                    "per line. If these are particle centers, "
                    "use CentersCoordinateSource or  use the --centers "
                    "flag in aspire extract-particles."
                )

    def _validate_box_file(self, box_file, global_particle_size):
        with open(box_file, "r") as box:
            # validate each line, i.e. each particle
            for line in box.readlines():
                # box format requires 4 numbers per particle
                if len(line.split()) < 4:
                    logger.error(f"Problem with coordinate file: {box_file}")
                    raise ValueError(
                        "Coordinate file contains less than 4 numbers "
                        "per line. If these are particle centers, "
                        "use CentersCoordinateSource or  use the --centers "
                        "flag in aspire extract-particles."
                    )

                if not all(self._is_number(p) for p in line.split()):
                    logger.error(f"Problem with coordinate file: {box_file}")
                    raise ValueError(
                        "Coordinate file contains non-numeric coordinate values."
                    )

                # we can only accept square particles
                size_x, size_y = float(line.split()[2]), float(line.split()[3])
                if size_x != size_y:
                    logger.error(f"Problem with coordinate file: {box_file}")
                    raise ValueError(
                        "Coordinate file specifies non-square particle size "
                        f"{size_x}x{size_y}, but only square particles are supported."
                    )
                # check that this particle size is the *right* particle size
                if size_x != global_particle_size:
                    logger.error(f"Problem with coordinate file: {box_file}")
                    raise ValueError(
                        f"Coordinate file specifies a box size {size_x}x{size_x} "
                        "different from the box size found in the first "
                        f"coordinate file ({global_particle_size}x{global_particle_size}). "
                        "Particle size must be consistent."
                    )

    def _populate_particles(self, coord_paths):
        # overrides CoordinateSource._populate_particles because of the
        # possibility that force_new_particle_size will be called,
        # which requires self.particles to be populated already.
        # Also allows for validation of .box files prior to parsing them

        global_particle_size = self._extract_box_size(coord_paths[0])
        # validate the rest of the box files
        for coord_path in coord_paths:
            self._validate_box_file(coord_path, global_particle_size)

        # populate self.particles
        super()._populate_particles(coord_paths)

        # if particle size set by user, we have to re-do the coordinates
        if self.particle_size:
            self._force_new_particle_size(self.particle_size)

    def _coords_list_from_file(self, coord_file):
        """
        Given a coordinate file in box format, returns a list of coordinates.
        """
        with open(coord_file, "r") as infile:
            lines = [line.split() for line in infile.readlines()]
        # coords are already in box format, so simply cast to int
        return [[int(float(x)) for x in line] for line in lines]

    def _force_new_particle_size(self, new_size):
        """
        Given a new particle size, rewrite the coordinates so that the box size
        is changed, but still centered around the particle.
        """
        _resized_particles = []
        for particle in self.particles:
            mrc_index, box_coord = particle
            # get the coordinates of the center
            center = self._center_from_box_coord(box_coord)
            # rewrite to a box coordinate with new size
            new_coord = self._box_coord_from_center(center, new_size)
            _resized_particles.append((mrc_index, new_coord))
        self.particles = _resized_particles


class CentersCoordinateSource(CoordinateSource):
    """
    Represents a data source consisting of micrographs and coordinate files specifying particle centers only. Files can be text (.coord) or STAR files.
    """

    def __init__(self, files, particle_size, max_rows=None, B=0, symmetry_group=None):
        """
        :param files: A list of tuples of the form (path_to_mrc, path_to_coord)
        :particle_size: Desired size of cropped particles (mandatory)
        :param max_rows: Maximum number of particles to read. (If `None`, will
        attempt to load all particles)
        :param symmetry_group: A `SymmetryGroup` object or string corresponding to the symmetry of the molecule.
        """
        # instantiate super
        CoordinateSource.__init__(
            self, files, particle_size, max_rows, B, symmetry_group
        )

    def _validate_centers_file(self, coord_file):
        """
        Ensures that a text file contains numeric particle centers.
        """
        with open(coord_file, "r") as infile:
            for line in infile.readlines():
                # need at least two numbers
                if len(line.split()) < 2:
                    logger.error(f"Problem with coordinate file: {coord_file}")
                    raise ValueError(
                        "Coordinate file contains a line with less than 2 numbers."
                    )
                # check that the coordinate has numeric values
                if not all(self._is_number(c) for c in line.split()):
                    logger.error(f"Problem with coordinate file: {coord_file}")
                    raise ValueError(
                        "Coordinate file contains non-numeric coordinate values."
                    )

    def _validate_starfile(self, coord_file):
        """
        Ensures that a STAR file contains numeric particle centers.
        """
        data_block = StarFile(coord_file).get_block_by_index(0)
        # We're looking for specific columns for the X and Y coordinates
        if not all(col in data_block for col in ["_rlnCoordinateX", "_rlnCoordinateY"]):
            logger.error(f"Problem with coordinate file: {coord_file}")
            raise ValueError(
                "STAR file does not contain _rlnCoordinateX, _rlnCoordinateY columns."
            )
        # check that all values in each column are numeric
        if not all(
            all(map(self._is_number, data_block[col]))
            for col in ["_rlnCoordinateX", "_rlnCoordinateY"]
        ):
            logger.error(f"Problem with coordinate file: {coord_file}")
            raise ValueError("STAR file contains non-numeric coordinate values.")

    def _populate_particles(self, coord_paths):
        # overrides CoordinateSource._populate_particles() in order
        # to validate coordinate files
        for coord_file in coord_paths:
            if os.path.splitext(coord_file)[1] == ".star":
                self._validate_starfile(coord_file)
            else:
                # assume text/.coord format
                self._validate_centers_file(coord_file)
        super()._populate_particles(coord_paths)

    def _coords_list_from_file(self, coord_file):
        """
        Given a coordinate file with (x,y) particle centers,
        return a list of coordinates in box format.
        """
        # check if it's a STAR file list of centers
        if os.path.splitext(coord_file)[1] == ".star":
            return self._coords_list_from_star(coord_file)
        # otherwise we assume text file format with one coord per line:
        with open(coord_file, "r") as infile:
            lines = [[float(c) for c in line.split()] for line in infile.readlines()]
        return [self._box_coord_from_center(line, self.particle_size) for line in lines]
