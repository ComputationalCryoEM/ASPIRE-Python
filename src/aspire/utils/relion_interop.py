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

    def __init__(self, filepath):
        super().__init__(filepath, blocks=None)
        self.data_block = df_to_relion_types(self.data_block)

    def get_aspire_metadata(self, data_folder):
        self.process_particles_block(data_folder)
        return self.data_block

    def process_particles_block(self, data_folder):
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

    @property
    def data_block(self):
        return self.get_block_by_index(0)

    @data_block.setter
    def data_block(self, df):
        """
        Set the underlying StarFile block representing image/micrograph/movie data to a new DataFrame.

        :param df: A Pandas DataFrame.
        """
        # get name of sole block
        block_name = list(self.blocks.keys())[0]
        return self.set_starfile_block(block_name, df)

    def set_starfile_block(self, block_name, df):
        """
        Create a new name/DataFrame pair in this StarFile, or overwrite an existing block with a new DataFrame.

        :param block_name: The name of the block in this StarFile.
        :param df: A pandas DataFrame.
        """
        self.blocks[block_name] = df


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
        # convert types
        self.data_block = df_to_relion_types(self.data_block)
        self.optics_block = df_to_relion_types(self.optics_block)

    @property
    def data_block(self):
        """
        Returns the DataFrame containing the image/micrograph/movie data represented by this STAR file.

        :return: A Pandas DataFrame.
        """
        return self._data_block()

    @data_block.setter
    def data_block(self, df):
        """
        Set the underlying StarFile block representing image/micrograph/movie data to a new DataFrame.

        :param df: A Pandas DataFrame.
        """
        return self.set_starfile_block(self.data_type, df)

    @property
    def data_type(self):
        """
        Returns either "particles", "micrographs" or "movies" depending on the underlying data being represented.
        """
        return self._data_type()

    @property
    def optics_block(self):
        """
        Returns the DataFrame containing the RELION optics group information for this STAR file.

        :return: A Pandas DataFrame.
        """
        return self.blocks["optics"]

    @optics_block.setter
    def optics_block(self, df):
        """
        Set the underlying StarFile block representing the RELION optics group information to a new DataFrame.

        :param df: A Pandas DataFrame.
        """
        return self.set_starfile_block("optics", df)

    def set_starfile_block(self, block_name, df):
        """
        Create a new name/DataFrame pair in this StarFile, or overwrite an existing block with a new DataFrame.

        :param block_name: The name of the block in this StarFile.
        :param df: A pandas DataFrame.
        """
        self.blocks[block_name] = df


class RelionParticlesStarFile(RelionDataStarFile):
    """ """

    def __init__(self, filepath):
        super().__init__(filepath)

    def _data_block(self):
        return self.blocks[self.data_type]

    def _data_type(self):
        return "particles"

    def get_aspire_metadata(self, data_folder):

        # get processed data block (ASPIRE columnsadded  and type conversion applied)
        self.process_particles_block(data_folder)

        # apply optics parameters into data block and get one metadata DF out
        self.apply_optics_block()

        return self.data_block

    def apply_optics_block(self):
        for optics_index, row in self.optics_block.iterrows():
            match = np.argwhere(
                self.data_block["_rlnOpticsGroup"].astype(int).to_numpy()
                == optics_index + 1
            )
            match = np.squeeze(match.T)
            for param in RelionDataStarFile.optics_params:
                self.data_block.loc[match, param] = getattr(row, param)

    def process_particles_block(self, data_folder):
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


class RelionMicrographsStarFile(StarFile):
    """ """

    def __init__(self, filepath):
        super().__init__(filepath, blocks=None)

    def _data_block(self):
        return self.blocks[self.data_type]

    def _data_type(self):
        return "micrographs"


class RelionMoviesStarFile(StarFile):
    """ """

    def __init__(self, filepath):
        super().__init__(filepath, blocks=None)

    def _data_block(self):
        return self.blocks[self.data_type]

    def _data_type(self):
        return "movies"
