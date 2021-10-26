import logging
import os.path
from collections import OrderedDict

import mrcfile
import numpy as np

from aspire.image import Image

# need to import explicitly, since EmanSource is alphabetically
# ahead of ImageSource in __init__.py
from aspire.source.image import ImageSource
from aspire.storage import StarFile

logger = logging.getLogger(__name__)


class EmanSource(ImageSource):
    def __init__(
        self,
        filepath,
        data_folder,
        particle_size=0,
        centers=False,
        pixel_size=1,
        B=0,
        max_rows=None,
    ):
        """
        Load a STAR file at a given filepath. This starfile must contains pairs of
        '_mrcFile' and '_boxFile', representing the mrc micrograph file and the
        EMAN format .box coordinates file respectively
        :param filepath: Absolute or relative path to STAR file
        :param data_folder: Path to folder w.r.t. which all relative paths to .mrc
        and .box files are resolved. If None the folder corresponding to the filepath
        is used
        :param particle_size: Desired size of cropped particles (will override size in coordinate file)
        :param centers: Set to true if the coordinates provided represent the centers of picked particles. By default, they are taken to be the coordinates of the lower left corner of the particle's box. If this flag is set, `particle_size` must be specified.
        """
        logger.debug(f"Creating ImageSource from STAR file at path {filepath}")

        self.centers = centers
        self.pixel_size = pixel_size
        self.max_rows = max_rows

        # dictionary indexed by mrc file paths, leading to a list of coordinates
        # coordinates represented by a tuple of integers
        self.mrc2coords = OrderedDict()

        # load in the STAR file as a data frame. this STAR file has one block
        df = StarFile(filepath).get_block_by_index(0)
        if data_folder is not None:
            if not os.path.isabs(data_folder):
                data_folder = os.path.join(os.path.dirname(filepath), data_folder)
        else:
            data_folder = os.path.dirname(filepath)
        mrc_paths = [os.path.join(data_folder, p) for p in list(df["_mrcFile"])]
        box_paths = [os.path.join(data_folder, p) for p in list(df["_boxFile"])]

        # populate mrc2coords
        # for each mrc, read its corresponding box file and load in the coordinates
        for i in range(len(mrc_paths)):
            coordList = []
            # open box file and read in the coordinates (one particle per line)
            with open(box_paths[i], "r") as boxfile:
                for line in boxfile.readlines():
                    coordList.append([int(x) for x in line.split()])
            self.mrc2coords[mrc_paths[i]] = coordList

        original_n = sum([len(self.mrc2coords[x]) for x in self.mrc2coords])

        # open first mrc file to populate micrograph dimensions and data type
        with mrcfile.open(mrc_paths[0]) as mrc:
            mode = int(mrc.header.mode)
            dtypes = {0: "int8", 1: "int16", 2: "float32", 6: "uint16"}
            dtype = dtypes[mode]
            shape = mrc.data.shape
        if not len(shape) == 2:
            logger.warn(
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
            # open first box file to get the particle size in the file
            first_box_filepath = box_paths[0]
            with open(first_box_filepath, "r") as boxfile:
                first_line = boxfile.readlines()[0]
                L = int(first_line.split()[2])
                other_side = int(first_line.split()[3])

            # firstly, ensure square particles
            if not L == other_side:
                logger.warn(
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
        
        # Lastly, exclude particle coordinate boxes that do not fit into the micrograph dimensions
        for _mrc, coordsList in self.mrc2coords:
            out_of_range = []
            for i in range(len(coordsList)):
                coord = coordsList[i]
                start_x, start_y, size_x, size_y = coord[0], coord[1], coord[2], coord[3] 
                if start_x < 0 or start_y < 0 or (start_x + size_x >= self.X) or (start_y + size_y >= self.Y):
                    out_of_range.append(i)
            # pop in reverse order to avoid messing up indices
            for j in reversed(out_of_range):
                coordsList.pop(j)    

        n = sum([len(self.mrc2coords[x]) for x in self.mrc2coords])
        removed = original_n - n
        logger.info(
            f"EmanSource from {filepath} contains {len(self.mrc2coords)} micrographs, {n} picked particles."
        )
        logger.info(f"{removed} particles did not fit into micrograph dimensions at particle size {L}, so were excluded.")
        ImageSource.__init__(self, L=L, n=n, dtype=dtype)

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

        out_of_bounds = []
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
