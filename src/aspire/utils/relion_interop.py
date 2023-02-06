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


class RelionStarFile(StarFile):
    def __init__(self, filepath):
        super().__init__(filepath, blocks=None)

        # validate Star file and get Relion version (3.0 or >=3.1)
        self.validate_and_detect_version()

        # convert dtypes in dataframes where possible
        self.convert_dtypes()

    def validate_and_detect_version(self):
        self.data_block_name = ""
        self.relion_version = ""

        rln_data_block_names = ["particles", "micrographs", "movies"]

        # validate 3.0 STAR file
        if len(self.blocks) == 1:
            data_block = self.get_block_by_index(0)
            columns = data_block.columns.to_list()
            if not any(
                col in columns
                for col in ["_rlnImageName", "_rlnMicrographName", "_rlnMovieName"]
            ):
                raise ValueError(
                    f"{self.filepath} does not contain Relion data columns."
                )

            self.relion_version = "3.0"
            self.data_block = data_block

        # validate 3.1 STAR file
        if len(self.blocks) == 2:
            # must have an optics block
            if "optics" not in self.blocks.keys():
                raise ValueError(f"{self.filepath} does not contain an optics block.")

            # find type of data
            for name in self.blocks.keys():
                if name in ["particles", "micrographs", "movies"]:
                    self.data_block_name = name
                    break
            if not self.data_block_name:
                raise ValueError(
                    f"{self.filepath} does not contain a block identifying particle, ",
                    "micrograph, or movie data.",
                )

            data_block = self[self.data_block_name]
            # lastly, data block must contain a column identifying the type of data as well
            columns = data_block.columns.to_list()
            if not any(
                col in columns
                for col in ["_rlnImageName", "_rlnMicrographName", "_rlnMovieName"]
            ):
                raise ValueError(
                    f"{self.filepath} data block does not contain Relion data columns."
                )

            self.relion_version = "3.1"
            self.data_block = data_block
            self.optics_block = self["optics"]

    def convert_dtypes(self):
        _blocks = OrderedDict()
        for block_name, block in self.blocks.items():
            _blocks[block_name] = df_to_relion_types(block)
        self.blocks = _blocks

    def get_data_block(self):
        if self.relion_version == "3.0":
            return self.data_block

        if self.relion_version == "3.1":
            data_block = self.data_block.copy()
            # get a NumPy array of optics indices for each row of data
            optics_indices = self.data_block["_rlnOpticsGroup"].astype(int).to_numpy()
            for optics_index, row in self.optics_block.iterrows():
                # find row indices with this optics index
                # Note optics group number is 1-indexed in Relion
                match = np.nonzero(optics_indices == optics_index + 1)[
                    0
                ]  # returns 1-tuple
                for param in self.optics_block.columns:
                    data_block.loc[match, param] = getattr(row, param)
            return data_block
