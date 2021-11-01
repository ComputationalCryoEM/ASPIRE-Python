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

logger = logging.getLogger(__name__)


class EmanSource(ImageSource):
    def __init__(
        self,
        mrc_list,
        coord_list,
        data_folder=None
        particle_size=0,
        centers=False,
        pixel_size=1,
        B=0,
        max_rows=None
    ):
        """
        :param data_folder: Path to folder w.r.t. which all relative paths to .mrc
        and .box files are resolved.
        :param mrc_list: Python list of micrographs that are part of this source. Note that the order of this list is assumed to correspond to the order of the coord_list parameter.
        :param coord_list: Python list of coordinate files (.box or .coord) containing coordinates of the particles in the micrographs. Note that the order of this list is assumed to correspond to the order of the mrc_list parameter.
        :param particle_size: Desired size of cropped particles (will override size in coordinate file). This parameter is mandatory when coordinates provided are centers (instead of lower-left corners)
        :param centers: Set to true if the coordinates provided represent the centers of picked particles. By default, they are taken to be the coordinates of the lower left corner of the particle's box. If this flag is set, `particle_size` must be specified.
        """

        self.centers = centers
        self.pixel_size = pixel_size
        self.B = B
        self.max_rows = max_rows

        # dictionary indexed by mrc file paths, leading to a list of coordinates
        # coordinates represented by a tuple of integers
        self.mrc2coords = OrderedDict()
               
        # must have one coordinate file for each micrograph
        assert len(mrc_list) == len(
            coord_list
        ), f"mrc_list contains {len(mrc_list)} micrographs, but coord_list contains {len(coord_list)} coordinate files."

        if data_folder is not None:
            if not os.path.isabs(data_folder):
                data_folder = os.path.join(os.getcwd(), data_folder)
        else:
            data_folder = os.getcwd()

        mrc_paths = [os.path.join(data_folder, mrc_list[i]) if not os.path.isabs(mrc_list[i]) else mrc_list[i] for i in range(len(mrc_paths))]
        coord_paths = [os.path.join(data_folder, coord_list[i]) if not os.path.isabs(coord_list[i) else coord_list[i] for i in range(len(mrc_paths))]

        # populate mrc2coords
        # for each mrc, read its corresponding box file and load in the coordinates
        for i in range(len(mrc_paths)):
            coordList = []
            # open box file and read in the coordinates (one particle per line)
            with open(coord_paths[i], "r") as coord_file:
                for line in coord_file.readlines():
                    particle_coord = [int(x) for x in line.split()]
                    if len(particle_coord) < 4:
                        particle_coord += [-1] * 2
                        if not self.centers:
                            logger.error(
                                f"{coord_paths[i]}: This coordinate file does not contain height and width information for particles. This may mean that the coordinates represent the center of the particle. Try setting centers=True and specifying a particle_size."
                            )
                            raise ValueError
                    coordList.append(particle_coord)
            self.mrc2coords[mrc_paths[i]] = coordList

        # discard the last N - max_rows particles
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
                    if count < max_rows:
                        tempdict[mrc].append(coord)
                    else:
                        done = True
                        break
            self.mrc2coords = tempdict

        # open first mrc file to populate micrograph dimensions and data type
        with mrcfile.open(mrc_paths[0]) as mrc_file:
            mode = int(mrc_file.header.mode)
            dtypes = {0: "int8", 1: "int16", 2: "float32", 6: "uint16"}
            dtype = dtypes[mode]
            shape = mrc_file.data.shape
        if not len(shape) == 2:
            logger.error(
                f"Shape of micrographs is {shape}, but expected shape of length 2. Hint: are these unaligned micrographs?"
            )
            raise ValueError

        self.Y = shape[0]
        self.X = shape[1]
        logger.info(f"Image size = {self.X}x{self.Y}")

        def size_particles(mrc2coords, new_size, old_size=0):
            # this method covers two scenarios:
            # 1. the coordinates represent the default, lower-left corner of the box (.box file standard), but the user wants to force a smaller particle size when loading
            # 2. the coordinates given in the file are *centers* of particles, not corners, and must be corrected
            for _mrc, coordsList in mrc2coords.items():
                for coords in coordsList:
                    # if self.centers, subtract off half of the particle size to get the lower-left corner position
                    # otherwise, we are reslicing from old_size to new_size, and we add half the difference of the two sizes to the lower left coordinates
                    trim_length = (
                        -(new_size // 2) if self.centers else (old_size - new_size) // 2
                    )
                    coords[0] += trim_length
                    coords[1] += trim_length
                    coords[2] = coords[3] = new_size

        if self.centers:
            assert (
                particle_size > 0
            ), "When constructing an EmanSource with coordinates of par\
ticle centers, a particle size must be specified."
            # recompute coordinates to account for the fact that we were given centers
            size_particles(self.mrc2coords, particle_size)
            L = particle_size
        else:
            # open first coord file to get the particle size in the file
            first_coord_filepath = coord_paths[0]
            with open(first_coord_filepath, "r") as coord_file:
                first_line = coord_file.readlines()[0]
                L = int(first_line.split()[2])
                other_side = int(first_line.split()[3])

            # firstly, ensure square particles
            if not L == other_side:
                logger.error(
                    "Particle size in coordinates file is {L}x{other_side}, but only square particle images are supported."
                )
                raise ValueError
            # if particle_size specified by user, we will recompute the coordinates around the center of the particle
            if not particle_size == 0:
                logger.info(
                    f"Overriding particle size of {L}x{L} specified in coordinates file."
                )
                size_particles(self.mrc2coords, particle_size, L)
                L = particle_size

        logger.info(f"Particle size = {L}x{L}")
        self._original_resolution = L

        original_n = sum([len(self.mrc2coords[x]) for x in self.mrc2coords])

        # Lastly, exclude particle coordinate boxes that do not fit into the micrograph dimensions
        for _mrc, coordsList in self.mrc2coords.items():
            out_of_range = []
            for i in range(len(coordsList)):
                coord = coordsList[i]
                start_x, start_y, size_x, size_y = (
                    coord[0],
                    coord[1],
                    coord[2],
                    coord[3],
                )
                if (
                    start_x < 0
                    or start_y < 0
                    or (start_x + size_x >= self.X)
                    or (start_y + size_y >= self.Y)
                ):
                    out_of_range.append(i)
            # pop in reverse order to avoid messing up indices
            for j in reversed(out_of_range):
                coordsList.pop(j)

        n = sum([len(self.mrc2coords[x]) for x in self.mrc2coords])
        removed = original_n - n
        logger.info(
            f"EmanSource from {data_folder} contains {len(self.mrc2coords)} micrographs, {n} picked particles."
        )
        logger.info(
            f"{removed} particles did not fit into micrograph dimensions at particle size {L}, so were excluded."
        )
        ImageSource.__init__(self, L=L, n=n, dtype=dtype)

        # Create filter indices for the source. These are required in order to pass through filter eval code
        self.filter_indices = np.zeros(self.n, dtype=int)
        self.unique_filters = [IdentityFilter()]

    def _images(self, start=0, num=np.inf, indices=None):
        """
        :param remove_out_of_bounds: If a set of coordinates creates a box that is outside the bounds of the micrograph, do not include the particle in the result. (If not set, the particle that could not be cropped will be an array of zeros)
        """
        # very important: the indices passed to this method will refer to the index
        # of the *particle*, not the micrograph
        if indices is None:
            indices = np.arange(start, min(start + num, self.n))
        else:
            start = indices.min()
        logger.info(f"Loading {len(indices)} images from micrographs")

        # explode mrc2coords into a flat list
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

        def crop_micrograph(data, coord):
            start_x, start_y, size_x, size_y = coord[0], coord[1], coord[2], coord[3]
            # according to MRC 2014 convention, origin represents
            # bottom-left corner of image
            return data[start_y : start_y + size_y, start_x : start_x + size_x]

        for i in range(len(_particles)):
            # get the particle number and the migrocraph
            num, fp = int(_particles[i].split("@")[0]), _particles[i].split("@")[1]
            # load the image data for this micrograph
            arr = mrcfile.open(fp).data
            # get the specified particle coordinates
            coord = self.mrc2coords[fp][num]
            cropped = crop_micrograph(arr, coord)
            im[i] = cropped

        return Image(im)
