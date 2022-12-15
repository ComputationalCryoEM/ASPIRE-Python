import logging
import os

import numpy as np
import pandas as pd

from aspire.storage import StarFile, getRelionStarFileVersion

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
    # convert STAR file strings to data type for each field
    # columns without a specified data type are read as dtype=object
    column_types = {name: relion_metadata_fields.get(name, str) for name in df.columns}
    return df.astype(column_types)


class RlnOpticsGroup:
    """
    A thin container to store RELION Optics Group parameters.
    """

    def __init__(self, df_row):
        """
        Creates a simple object to store optics group data given a pandas dataframe row.
        """
        self.name = df_row._rlnOpticsGroupName
        self.number = int(df_row._rlnOpticsGroup)
        self.voltage = float(df_row._rlnVoltage)
        self.cs = float(df_row._rlnSphericalAberration)
        self.amplitude_contrast = float(df_row._rlnAmplitudeContrast)


class RlnParticleOpticsGroup(RlnOpticsGroup):
    def __init__(self, df_row, version="3.1"):
        if version == "3.1":
            self.pixel_size = float(df_row._rlnImagePixelSize)
        super().__init__(df_row)


class RlnMicrographOpticsGroup(RlnOpticsGroup):
    def __init__(self, df_row, version="3.1"):
        if version == "3.1":
            self.pixel_size = float(df_row._rlnMicrographPixelSize)
        super().__init__(df_row)


class RelionLegacyParticlesStarFile(StarFile):
    """ """


class RelionLegacyMicrographsStarFile(StarFile):
    """ """


class RelionLegacyMoviesStarFile(StarFile):
    """ """


class RelionDataStarFile(StarFile):

    optics_params = [
        "_rlnImagePixelSize",
        "_rlnVoltage",
        "_rlnSphericalAberration",
        "_rlnAmplitudeContrast",
        "_rlnOpticsGroupName",
    ]

    def __init__(self, filepath):
        super().__init__(filepath, blocks=None)

        # update blocks with appropriate dtypes
        self.blocks["optics"] = df_to_relion_types(self.optics_block)
        self.blocks[self.data_block_name] = df_to_relion_types(self.data_block)

    @property
    def data_block(self):
        return self._data_block()

    @property
    def optics_block(self):
        return self.blocks["optics"]


class RelionParticlesStarFile(RelionDataStarFile):
    """ """

    def __init__(self, filepath):
        self.data_block_name = "particles"
        super().__init__(filepath)

    def _data_block(self):
        return self.blocks["particles"]

    def get_aspire_metadata(self, data_folder):

        # get processed data block (ASPIRE columnsadded  and type conversion applied)
        proc_data_block = self.process_data_block(data_folder)

        # apply optics parameters into data block and get one metadata DF out
        metadata = self.apply_optics_block(proc_data_block)

        return metadata

    def apply_optics_block(self, proc_data_block):
        for optics_index, row in self.optics_block.iterrows():
            match = np.argwhere(
                proc_data_block["_rlnOpticsGroup"].astype(int).to_numpy()
                == optics_index + 1
            )
            match = np.squeeze(match.T)
            for param in RelionDataStarFile.optics_params:
                proc_data_block.loc[match, param] = getattr(row, param)

        return proc_data_block

    def process_data_block(self, data_folder):
        # get block containing particles and cast to appropriate types
        df = self.data_block

        # particle locations are stored as e.g. '000001@first_micrograph.mrcs'
        # in the _rlnImageName column. here, we're splitting this information
        # so we can get the particle's index in the .mrcs stack as an int
        df[["__mrc_index", "__mrc_filename"]] = df["_rlnImageName"].str.split(
            "@", 1, expand=True
        )
        # __mrc_index corresponds to the integer index of the particle in the __mrc_filename stack
        # Note that this is 1-based indexing
        df["__mrc_index"] = pd.to_numeric(df["__mrc_index"])

        # Adding a full-filepath field to the Dataframe helps us save time later
        # Note that os.path.join works as expected when the second argument is an absolute path itself
        df["__mrc_filepath"] = df["__mrc_filename"].apply(
            lambda filename: os.path.join(data_folder, filename)
        )

        return df


class RelionMicrographsStarFile(StarFile):
    """ """

    def __init__(self, filepath):
        super().__init__(filepath, blocks=None)
        self.data_block_name = "micrographs"

    def _data_block(self):
        return self.blocks["micrographs"]


class RelionMoviesStarFile(StarFile):
    """ """

    def __init__(self, filepath):
        super().__init__(filepath, blocks=None)
        self.data_block_name = "movies"

    def _data_block(self):
        return self.blocks["movies"]
