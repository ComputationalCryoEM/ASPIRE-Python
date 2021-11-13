import itertools
import logging
import os
from collections import OrderedDict

import mrcfile
import numpy as np

from aspire.image import Image
from aspire.operators import IdentityFilter

# need to import explicitly, since from_particles is alphabetically
# ahead of image in __init__.py
from aspire.source.image import ImageSource
from aspire.storage import StarFile

logger = logging.getLogger(__name__)


class ParticleCoordinateSource(ImageSource):
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
        :param files: a list of tuples (micrograph path, coordinate file path)
        :param data_folder: Path to which filepaths are relative do. Default: current directory.
        :param particle_size: Desired size of cropped particles
        :param centers: Set to true if the coordinates provided represent the centers of picked particles.
        :param pixel_size: Pixel size of micrograph in Angstroms (default: 1)
        :param B: Envelope decay of the CTF in inverse Angstroms (default: 0)
        :param max_rows: Maximum number of particles to read. (If None, all particles will be loaded)
        :param relion_autopick_star: Relion star file from AutoPick or ManualPick jobs (e.g. AutoPick/job006/autopick.star)
        """

        # The internal representation of micrographs and their picked coordinates
        # is mrc2coords, an OrderedDict with micrograph filepaths as keys, and
        # lists of coordinates as values. Coordinates are lists of integers:
        # [lower left X, lower left Y, size X, size Y]. Different coordinate
        # files store this information differently, but all coordinates are converted
        # to this canonical format
        self.mrc2coords = self._extract_coordinates(
            files, data_folder, particle_size, centers, relion_autopick_star
        )
        num_micrographs = len(self.mrc2coords)

        # get first micrograph and coordinate list to report some data the user
        first_micrograph, first_coords = list(self.mrc2coords.items())[0]

        with mrcfile.open(first_micrograph) as mrc_file:
            dtype = np.dtype(mrc_file.data.dtype)
            shape = mrc_file.data.shape
        if len(shape) != 2:
            raise ValueError(
                "Shape of mrc file is {shape} but expected shape of size 2. Are these unaligned micrographs?"
            )

        self.Y = shape[0]
        self.X = shape[1]
        logger.info(f"Micrograph size = {self.X}x{self.Y}")

        # look at first coord to get particle size (not necessarily specified by user)
        L = first_coords[0][3]
        logger.info(f"Particle size = {L}x{L}")
        self._original_resolution = L

        # total number of particles given in coord files
        # before removing those that do not fit
        original_n = sum([len(self.mrc2coords[x]) for x in self.mrc2coords])
        removed = 0

        # Exclude particle coordinate boxes that do not fit into the micrograph
        for _mrc, coordsList in self.mrc2coords.items():
            out_of_range = []
            for i, coord in enumerate(coordsList):
                coord = coordsList[i]
                start_x, start_y, size_x, size_y = coord
                if (
                    start_x < 0
                    or start_y < 0
                    or (start_x + size_x >= self.X)
                    or (start_y + size_y >= self.Y)
                ):
                    out_of_range.append(i)
                    removed += 1
            # pop in reverse order to avoid messing up indices
            for j in reversed(out_of_range):
                coordsList.pop(j)

        # max_rows means max number of particles, but each micrograph has a differing
        # number of particles
        if max_rows:
            # cumulative number of particles in each micrograph
            accum_lengths = list(
                itertools.accumulate([len(self.mrc2coords[d]) for d in self.mrc2coords])
            )
            # the index of the micrograph that brings us over max_rows
            i_gt_max_rows = next(
                elem[0] for elem in enumerate(accum_lengths) if elem[1] > max_rows
            )
            # subtract off the difference
            remainder = max_rows - accum_lengths[i_gt_max_rows - 1]
            itms = list(self.mrc2coords.items())
            # include all the micrographs and coordinates that we don't need to trim
            tempdict = OrderedDict(
                {itms[i][0]: itms[i][1] for i in range(i_gt_max_rows)}
            )
            # add in the last micrograph, only up to 'remainder' particles
            tempdict[itms[i_gt_max_rows][0]] = itms[i_gt_max_rows][1][:remainder]
            self.mrc2coords = tempdict

        # final number of particles in *this* source
        n = sum([len(self.mrc2coords[x]) for x in self.mrc2coords])

        logger.info(
            f"ParticleCoordinateSource from {data_folder} contains {num_micrographs} micrographs, {original_n} picked particles."
        )
        if removed > 0:
            logger.info(
                f"{removed} particles did not fit into micrograph dimensions at particle size {L}, so were excluded. Maximum number of particles at this resolution is {original_n - removed}."
            )
        logger.info(f"ParticleCoordinateSource object contains {n} particles.")

        ImageSource.__init__(self, L=L, n=n, dtype=dtype)

        # Create filter indices for the source. These are required in order to pass through filter eval code
        # bypassing the filter_indices setter in ImageSource allows us to create this source with
        # absolutely *no* metadata. otherwise, six default Relion columns are created w/defualt values
        self.set_metadata("__filter_indices", np.zeros(self.n, dtype=int))
        self.unique_filters = [IdentityFilter()]

    @staticmethod
    def crop_micrograph(data, coord):
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

        # flatten mrc2coords into a list
        all_particles = []
        # identify each particle as e.g. 000001@mrcfile, analogously to Relion
        for mrc in self.mrc2coords:
            for i in range(len(self.mrc2coords[mrc])):
                all_particles.append(f"{i:06d}@{mrc}")
        # select the desired particles from this list
        _particles = [all_particles[i] for i in indices]
        # initialize empty array to hold particle stack
        im = np.empty(
            (len(indices), self._original_resolution, self._original_resolution),
            dtype=self.dtype,
        )

        for i in range(len(_particles)):
            # get the particle number and the micrograph
            num, fp = int(_particles[i].split("@")[0]), _particles[i].split("@")[1]
            # load the image data for this micrograph
            arr = mrcfile.open(fp).data
            # get the specified particle coordinates
            coord = self.mrc2coords[fp][num]
            cropped = self.crop_micrograph(arr, coord)
            im[i] = cropped

        return Image(im)

    def _relion_star_parser(self, relion_autopick_star, data_folder, particle_size):
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
        micrographs = list(df["_rlnMicrographName"])
        coord_stars = list(df["_rlnMicrographCoordinates"])
        files = [(micrographs[i], coord_stars[i]) for i in range(len(df))]
        num_files = len(files)
        mrc_paths = [os.path.join(data_folder, files[i][0]) for i in range(num_files)]
        coord_paths = [os.path.join(data_folder, files[i][1]) for i in range(num_files)]
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

            return _mrc2coords

    def _centers_coord_parser(self, mrc_paths, coord_paths, particle_size):
        """
        Extract coordinates from .coord files, which specify particles by the
        coordinates of their centers.
        """
        # for each mrc, read its corresponding coordinates file
        _mrc2coords = OrderedDict()
        for i in range(len(mrc_paths)):
            coordList = []
            # We are reading particle centers from the coordinate file
            # We open the corresponding coordinate file
            with open(coord_paths[i], "r") as coord_file:
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
                coordList.append(coord)
            _mrc2coords[mrc_paths[i]] = coordList
        return _mrc2coords

    def _box_coord_parser(self, mrc_paths, coord_paths, particle_size):
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
                    "Coordinate file gives non-square particle size {size_x}x{size_y}, but only square particles are supported"
                )

        # for each mrc, read its corresponding coordinates file
        _mrc2coords = OrderedDict()
        for i in range(len(mrc_paths)):
            coordList = []
            # We are reading particle centers from the coordinate file
            # We open the corresponding coordinate file
            with open(coord_paths[i], "r") as coord_file:
                # each coordinate is a whitespace separated line in the file
                lines = coord_file.readlines()
            for line in lines:
                lower_left_x, lower_left_y, size_x, size_y = [
                    int(x) for x in line.split()
                ]
                coord = [lower_left_x, lower_left_y, size_x, size_y]
                coordList.append(coord)
            _mrc2coords[mrc_paths[i]] = coordList

        # if particle size is not zero, we have to re-do the coordinates
        # get the particle size of the first coordinate of the first micrographs
        if particle_size > 0:
            old_size = size_x
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
            return _resized_mrc2coords
        else:
            return _mrc2coords

    def _check_and_get_paths(self, files, data_folder):
        num_files = len(files)
        _mrc_absolute_paths = False
        _coord_absolute_paths = False
        if data_folder is not None:
            if not os.path.isabs(data_folder):
                data_folder = os.path.join(os.getcwd(), data_folder)
            if os.path.isabs(files[0][0]):
                # check that abs paths to mrcs matches data folder
                if os.path.dirname(files[0][0]) != data_folder:
                    raise ValueError(
                        "data_folder provided ({data_folder}) does not match dirname of mrc files ({os.path.dirname(files[0][0])})"
                    )
                _mrc_absolute_paths = True
            if os.path.isabs(files[0][1]):
                # check that abs paths to coords matches data folder
                if os.path.dirname(files[0][1]) != data_folder:
                    raise ValueError(
                        "data_folder provided ({data_folder}) does not match dirname of coordinate files ({os.path.dirname(files[0][1])})"
                    )
                _coord_absolute_paths = True
        else:
            data_folder = os.getcwd()

        mrc_paths = [
            os.path.join(data_folder, files[i][0])
            if not _mrc_absolute_paths
            else files[i][0]
            for i in range(num_files)
        ]
        coord_paths = [
            os.path.join(data_folder, files[i][1])
            if not _coord_absolute_paths
            else files[i][1]
            for i in range(num_files)
        ]

        return data_folder, mrc_paths, coord_paths

    def _extract_coordinates(
        self, files, data_folder, particle_size, centers, relion_autopick_star
    ):
        """
        Based on arguments passed to __init__, decide which type of coordinate file
        we are dealing with and call corresponding parser method. Regardless of the
        type of file, all coordinates are transformed to a canonical format:
        [lower_left_x, lower_left_y, x_size, y_size]
        """

        # if reading from a Relion STAR file
        if relion_autopick_star:
            return self._relion_star_parser(
                relion_autopick_star, data_folder, particle_size
            )
        # if reading from a .box or .coord
        else:
            # check the data folder against the paths provided in the 'files' kwarg
            # and return the full filepaths
            data_folder, mrc_paths, coord_paths = self._check_and_get_paths(
                files, data_folder
            )
            # if centers, we are reading (X,Y) center coordinates only
            if centers:
                if particle_size == 0:
                    raise ValueError(
                        "If reading particle centers, a particle_size must be specified"
                    )
                return self._centers_coord_parser(mrc_paths, coord_paths, particle_size)
            # otherwise we are reading (lower left X, lower left Y, X size, Y size)
            # this is the .box format specified by EMAN
            # this method also forces a new particle_size, if one was specified
            else:
                return self._box_coord_parser(mrc_paths, coord_paths, particle_size)
