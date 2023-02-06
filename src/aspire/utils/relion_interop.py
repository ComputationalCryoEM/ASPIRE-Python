import logging
from collections import OrderedDict

import numpy as np

from aspire.storage import StarFile

logger = logging.getLogger(__name__)


# The metadata_fields dictionary below specifies default data types
# of certain key fields used in the codebase,
# which are originally read from Relion STAR files.
relion_metadata_fields = {
    "_rlnVoltage": float,
    "_rlnDefocusU": float,
    "_rlnDefocusV": float,
    "_rlnDefocusAngle": float,
    "_rlnSphericalAberration": float,
    "_rlnDetectorPixelSize": float,
    "_rlnCtfFigureOfMerit": float,
    "_rlnMagnification": float,
    "_rlnAmplitudeContrast": float,
    "_rlnImageName": str,
    "_rlnOriginalName": str,
    "_rlnCtfImage": str,
    "_rlnCoordinateX": float,
    "_rlnCoordinateY": float,
    "_rlnCoordinateZ": float,
    "_rlnNormCorrection": float,
    "_rlnMicrographName": str,
    "_rlnGroupName": str,
    "_rlnGroupNumber": str,
    "_rlnOriginX": float,
    "_rlnOriginY": float,
    "_rlnAngleRot": float,
    "_rlnAngleTilt": float,
    "_rlnAnglePsi": float,
    "_rlnClassNumber": int,
    "_rlnLogLikeliContribution": float,
    "_rlnRandomSubset": int,
    "_rlnParticleName": str,
    "_rlnOriginalParticleName": str,
    "_rlnNrOfSignificantSamples": float,
    "_rlnNrOfFrames": int,
    "_rlnMaxValueProbDistribution": float,
    "_rlnOpticsGroup": int,
    "_rlnOpticsGroupName": str,
}


def df_to_relion_types(df):
    # Convert STAR file strings to data type for each field.
    # Columns without a specified data type are read as dtype=object.
    column_types = {name: relion_metadata_fields.get(name, str) for name in df.columns}
    return df.astype(column_types)


class Relion30StarFile(StarFile):
    def __init__(self, filepath):
        super().__init__(filepath, blocks=None)

        # first convert types where possible
        _blocks = OrderedDict()
        for block_name, block in self.blocks.items():
            _blocks[block_name] = df_to_relion_types(block)
        self.blocks = _blocks


class Relion31StarFile(Relion30StarFile):
    def __init__(self, filepath):
        super().__init__(filepath)

        rln_data_block_names = ["particles", "micrographs", "movies"]
        # figure out whether particles, micrographs, or movies
        data_block_name = ""
        for name in star.blocks.keys():
            if name in rln_data_block_names:
                data_block_name = name
                break
        self.optics_block = self.blocks["optics"]
        self.data_block = self.blocks[data_block_name]

    def merged_data_block(self):
        """
        Applies the parameters in the optics block as new columns in the data block,
            based on the corresponding optics group number. Returns a new DataFrame.

        :return: A new DataFrame with the optics parameters added as columns.
        """
        data_block = self.data_block.copy()
        # get a NumPy array of optics indices for each row of data
        optics_indices = self.data_block["_rlnOpticsGroup"].astype(int).to_numpy()
        for optics_index, row in self.optics_block.iterrows():
            # find row indices with this optics index
            # Note optics group number is 1-indexed in Relion
            match = np.nonzero(optics_indices == optics_index + 1)[0]  # returns 1-tuple
            for param in self.optics_block.columns:
                data_block.loc[match, param] = getattr(row, param)
        return data_block
