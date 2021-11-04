import logging
import os
from collections import OrderedDict

import mrcfile
import numpy as np

from aspire.image import Image
from aspire.operators import IdentityFilter

# need to import explicitly, since EmanSource is alphabetically
# ahead of ImageSource in __init__.py
from aspire.source.image import ImageSource
from aspire.storage import StarFile

logger = logging.getLogger(__name__)


class EmanSource(ImageSource):
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

        self.centers = centers
        self.pixel_size = pixel_size
        self.B = B
        self.max_rows = max_rows

        # dictionary indexed by mrc file paths, leading to a list of coordinates
        # coordinates represented by a list of integers
        self.mrc2coords = OrderedDict()

        # if reading from a Relion STAR file
        self.relion = False
        if relion_autopick_star:
            if data_folder is None:
                raise ValueError(
                    "Provide Relion project directory when loading from Relion picked coordinates STAR file"
                )
            self.centers = True
            self.relion = True
            star_in = StarFile(relion_autopick_star)
            df = star_in["coordinate_files"]
            micrographs = list(df["_rlnMicrographName"])
            coord_stars = list(df["_rlnMicrographCoordinates"])
            files = [(micrographs[i], coord_stars[i]) for i in range(len(df))]

        mrc_absolute_paths = False
        coord_absolute_paths = False
        if data_folder is not None:
            if not os.path.isabs(data_folder):
                data_folder = os.path.join(os.getcwd(), data_folder)
            if os.path.isabs(files[0][0]):
                assert (
                    os.path.dirname(files[0][0]) == data_folder
                ), f"data_folder provided ({data_folder}) does not match dirname of mrc files ({os.path.dirname(files[0][0])})"
                mrc_absolute_paths = True
            if os.path.isabs(files[0][1]):
                assert (
                    os.path.dirname(files[0][1]) == data_folder
                ), f"data_folder provided ({data_folder}) does not match dirname of coordinate files ({os.path.dirname(files[0][1])})"
                coord_absolute_paths = True
        else:
            data_folder = os.getcwd()

        # fill in paths to micrographs and coordinate files
        self.num_micrographs = len(files)
        mrc_paths = [
            os.path.join(data_folder, files[i][0])
            if not mrc_absolute_paths
            else files[i][0]
            for i in range(self.num_micrographs)
        ]
        coord_paths = [
            os.path.join(data_folder, files[i][1])
            if not coord_absolute_paths
            else files[i][1]
            for i in range(self.num_micrographs)
        ]

        # populate mrc2coords
        # for each mrc, read its corresponding box file and load in the coordinates
        for i in range(len(mrc_paths)):
            coordList = []
            # two types of coordinate files, each containing coords of
            # multiple particles in one micrograph
            # for both, read coordinates into the form
            # [ [particle1_X, particle1_Y, ..], [particle2_X, particle2_Y, ..]]
            if self.relion:
                df = StarFile(coord_paths[i]).get_block_by_index(0)
                x_coords = list(df["_rlnCoordinateX"])
                y_coords = list(df["_rlnCoordinateY"])
                particles = [
                    [int(float(x_coords[i])), int(float(y_coords[i]))] for i in range(len(df))
                ]
            else:
                # open coordinate file and read in the coordinates
                with open(coord_paths[i], "r") as coord_file:
                    lines = [line.split() for line in coord_file.readlines()]
                    particles = [[int(x) for x in line] for line in lines]
            for particle_coord in particles:
                # if there are less than 4 numbers, we are most likely being given centers
                if len(particle_coord) < 4:
                    # pad list to length 4 so that it can be filled in with proper values later
                    particle_coord += [-1] * 2
                    if not self.centers:
                        logger.error(
                            f"{coord_paths[i]}: This coordinate file does not contain height and width information for particles. This may mean that the coordinates represent the center of the particle. Try setting centers=True and specifying a particle_size."
                        )
                        raise ValueError
                coordList.append(particle_coord)
            self.mrc2coords[mrc_paths[i]] = coordList

        # open first mrc file to populate micrograph dimensions and data type
        with mrcfile.open(mrc_paths[0]) as mrc_file:
            dtype = np.dtype(mrc_file.data.dtype)
            shape = mrc_file.data.shape
        if len(shape) != 2:
            logger.error(
                f"Shape of micrographs is {shape}, but expected shape of length 2. Hint: are these unaligned micrographs?"
            )
            raise ValueError

        self.Y = shape[0]
        self.X = shape[1]
        logger.info(f"Image size = {self.X}x{self.Y}")

        def force_new_size(new_size, old_size):
            trim_length = (old_size - new_size) // 2
            tempdir = OrderedDict()
            for mrc, coordsList in self.mrc2coords.items():
                tempdir[mrc] = []
                for coords in coordsList:
                    temp_coord = [-1, -1, new_size, new_size]
                    temp_coord[0] = coords[0] + trim_length
                    temp_coord[1] = coords[1] + trim_length
                    tempdir[mrc].append(temp_coord)
            self.mrc2coords = tempdir

        def write_coords_from_centers(size):
            trim_length = size // 2
            tempdir = OrderedDict()
            for mrc, coordsList in self.mrc2coords.items():
                tempdir[mrc] = []
                for coords in coordsList:
                    temp_coord = [-1, -1, size, size]
                    temp_coord[0] = coords[0] - trim_length
                    temp_coord[1] = coords[1] - trim_length
                    tempdir[mrc].append(temp_coord)
            self.mrc2coords = tempdir

        if self.centers:
            assert (
                particle_size > 0
            ), "When constructing an EmanSource with coordinates of par\
ticle centers, a particle size must be specified."
            # recompute coordinates to account for the fact that we were given centers
            write_coords_from_centers(particle_size)
            L = particle_size
        else:
            # open first coord file to get the particle size in the file
            first_coord_filepath = coord_paths[0]
            with open(first_coord_filepath, "r") as coord_file:
                first_line = coord_file.readlines()[0]
                L = int(first_line.split()[2])
                other_side = int(first_line.split()[3])

            # firstly, ensure square particles
            if L != other_side:
                logger.error(
                    "Particle size in coordinates file is {L}x{other_side}, but only square particle images are supported."
                )
                raise ValueError
            # if particle_size specified by user, we will recompute the coordinates around the center of the particle
            if particle_size != 0:
                logger.info(
                    f"Overriding particle size of {L}x{L} specified in coordinates file."
                )
                force_new_size(particle_size, L)
                L = particle_size

        logger.info(f"Particle size = {L}x{L}")
        self._original_resolution = L

        original_n = sum([len(self.mrc2coords[x]) for x in self.mrc2coords])
        removed = 0
        # Lastly, exclude particle coordinate boxes that do not fit into the micrograph dimensions
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

        if max_rows:
            count = 0
            tempdict = {}
            done = False
            for mrc, coordList in self.mrc2coords.items():
                if done:
                    break
                tempdict[mrc] = []
                for coord in coordList:
                    count += 1
                    if count <= max_rows:
                        tempdict[mrc].append(coord)
                    else:
                        done = True
                        break
            self.mrc2coords = tempdict

        n = sum([len(self.mrc2coords[x]) for x in self.mrc2coords])
        logger.info(
            f"Data source from {data_folder} contains {self.num_micrographs} micrographs, {original_n} picked particles."
        )
        logger.info(
            f"{removed} particles did not fit into micrograph dimensions at particle size {L}, so were excluded. Maximum number of particles at this resolution is {original_n - removed}."
        )
        logger.info(f"EmanSource object contains {n} particles.")
        ImageSource.__init__(self, L=L, n=n, dtype=dtype)

        # Create filter indices for the source. These are required in order to pass through filter eval code
        self.filter_indices = np.zeros(self.n, dtype=int)
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
