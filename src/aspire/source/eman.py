import os.path

import logging
from aspire.source import ImageSource
from aspire.storage import StarFile

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
        
        # dictionary from 
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
        for i in range(len(mrc_paths)):
            coordList = []
            # open corresponding box file to see how many particles it contains
            with open(box_paths[i], 'r') as boxfile:
                for line in boxfile.readlines():
                    coordList.append([int(x) for x in line.split()])
            self.mrc2coords[mrc_paths[i]] = coordList     
        num_particles = sum([len(self.mrc2coords[x]) for x in self.mrc2coords])
        logger.info(f"EmanSource from {filepath} contains {len(self.mrc2coords)} micrographs, {num_particles} picked particles.")        
        
       
    def _images(self, start=0, num=np.inf, indices=None):
        pass   
    
    def load_micrograph(filepath):
        return mrcfile.open(filepath).data

    def crop_micrograph(im):         
        pass
