import os.path

import logging
from aspire.source import ImageSource
from aspire.storage import StarFile
from aspire.image import Image

import mrcfile
import numpy as np
from pandas import DataFrame

logger = logging.getLogger(__name__)

class EmanSource(ImageSource):
       
    def __init__(self, filepath, data_folder, pixel_size=1, B=0, max_rows=None):
        """
        Load a STAR file at a given filepath. This starfile must contains pairs of
        '_mrcFile' and '_boxFile', representing the mrc micrograph file and the 
        EMAN format .box coordinates file respectively
        :param filepath: Absolute or relative path to STAR file
        :param data_folder: Path to folder w.r.t. which all relative paths to .mrc
        and .box files are resolved. If None the folder corresponding to the filepath
        is used

        """
        logger.debug(f"Creating ImageSource from STAR file at path {filepath}")
        
        # dictionary indexed by mrc file paths, leading to a list of coordinates
        # coordinates represented by a tuple of integers
        self.mrc2coords = {}
        
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
            with open(box_paths[i], 'r') as boxfile:
                for line in boxfile.readlines():
                    coordList.append((int(x) for x in line.split()))
            self.mrc2coords[mrc_paths[i]] = coordList     

        self.num_particles = sum([len(self.mrc2coords[x]) for x in self.mrc2coords])
        logger.info(f"EmanSource from {filepath} contains {len(self.mrc2coords)} micrographs, {self.num_particles} picked particles.")        
        
        # open first mrc file to populate metadata
        first_mrc_filepath = mrc_paths[0]
        mrc = mrcfile.open(first_mrc_filepath)
        mode = int(mrc.header.mode)
        dtypes = {0: "int8", 1: "int16", 2: "float32", 6: "uint16"}
        dtype = dtypes[mode]
        shape = mrc.data.shape
        assert(len(shape)==2)
        assert(shape[0] == shape[1])
        L = shape[1]
        logger.info(f"Image size = {L}x{L}")
        self._original_resolution = L
        

       
    def _images(self, start=0, num=np.inf, indices=None):
        # very important: the indices passed to this method will refer to the index
        # of the *particle*, not the micrograph
        if indices is None:
            indices = np.arange(start, min(start + num, self.num_particles))
        else:
            start = indices.min()
        logger.info(f"Loading {len(indices)} images from micrographs")
        
        # explode mrc2coords into a flat list
        all_particles = []
        for mrc in self.mrc2coords:
            for i in range(len(self.mrc2coords[mrc])):
                all_particles.append(f"{i:05d}@{mrc}")
        _particles = [all_particles[i] for i in indices]
        im = np.empty((len(indices),self._original_resolution, self._original_resolution), dtype=self.dtype)
        
        for i in range(len(_particles)):
            num, fp = int(particle.split("@")[0]), particle.split("@")[1]
            # load the image data for this micrograph
            arr = mrcfile.open(fp).data
            # get the specified particle coordinates
            coord = self.mrc2coords[fp][num]
            im[i] = crop_micrograph(arr, coord)

        return Image(im)

    def crop_micrograph(data, coord):         
        bottom_left_corner_x = coord[0]
        bottom_left_corner_y = coord[1]
        size_x = coord[2]
        size_y = coord[3]
        
