import numpy as np
from aspire.source import ImageSource
from aspire.source.starfile import StarfileStack
from aspire.utils.coor_trans import angles_to_rots
from aspire.utils.filters import CTFFilter
from aspire.source import SourceFilter


class RelionStarfileStack(StarfileStack):

    column_mappings = {
        'rlnVoltage': float,
        'rlnDefocusU': float,
        'rlnDefocusV': float,
        'rlnDefocusAngle': float,
        'rlnSphericalAberration': float,
        'rlnDetectorPixelSize': float,
        'rlnCtfFigureOfMerit': float,
        'rlnMagnification': float,
        'rlnAmplitudeContrast': float,
        'rlnImageName': str,
        'rlnOriginalName': str,
        'rlnCtfImage': str,
        'rlnCoordinateX': float,
        'rlnCoordinateY': float,
        'rlnCoordinateZ': float,
        'rlnNormCorrection': float,
        'rlnMicrographName': str,
        'rlnGroupName': str,
        'rlnGroupNumber': str,
        'rlnOriginX': float,
        'rlnOriginY': float,
        'rlnAngleRot': float,
        'rlnAngleTilt': float,
        'rlnAnglePsi': float,
        'rlnClassNumber': int,
        'rlnLogLikeliContribution': float,
        'rlnRandomSubset': int,
        'rlnParticleName': str,
        'rlnOriginalParticleName': str,
        'rlnNrOfSignificantSamples': float,
        'rlnNrOfFrames': int,
        'rlnMaxValueProbDistribution': float
    }

    def add_metadata(self):

        df = self.df
        if 'rlnDefocusU' not in df:
            df['rlnDefocusU'] = np.nan

        if 'rlnDefocusV' not in df:
            df['rlnDefocusV'] = df['rlnDefocusU']
            df['rlnDefocusAngle'] = 0.

        if 'rlnAngleRot' not in df:
            df['rlnAngleRot'] = np.nan
            df['rlnAngleTilt'] = np.nan
            df['rlnAnglePsi'] = np.nan

        if 'rlnOriginX' not in df:
            df['rlnOriginX'] = np.nan
            df['rlnOriginY'] = np.nan

        if 'rlnClassNumber' not in df:
            df['rlnClassNumber'] = np.nan

        # Columns representing angles in radians
        df['_rlnAngleRot_radians'] = (df['rlnAngleRot'] / 180) * np.pi
        df['_rlnAngleTilt_radians'] = (df['rlnAngleTilt'] / 180) * np.pi
        df['_rlnAnglePsi_radians'] = (df['rlnAnglePsi'] / 180) * np.pi
        df['_rlnDefocusAngle_radians'] = (df['rlnDefocusAngle'] / 180) * np.pi

        return df

    def __init__(self, filepath, pixel_size=1, B=0, n_workers=-1, ignore_missing_files=False, max_rows=None):
        """
        Load a Relion starfile at given filepath
        :param filepath: Absolute or relative path to .star file
        :param pixel_size: the pixel size of the images in angstroms (Default 1)
        :param B: the envelope decay of the CTF in inverse square angstrom (Default 0)
        :param ignore_missing_files: Whether to ignore missing MRC files or not (Default False)
        :param max_rows: Maximum no. of rows in .star file to read. If None (default), all rows are read.
            Note that this refers to the max no. of images to load, not the max. number of .mrcs files (which may be
            equal to or less than the no. of images).
            If ignore_missing_files is False, the first max_rows rows read from the .star file are considered.
            If ignore_missing_files is True, then the first max_rows *available* rows from the .star file are
            considered.
        """

        StarfileStack.__init__(self, filepath, n_workers=n_workers, ignore_missing_files=ignore_missing_files,
                               max_rows=max_rows)

        self.pixel_size = pixel_size
        self.B = B

        rots = angles_to_rots(
            self.df[['_rlnAngleRot_radians', '_rlnAngleTilt_radians', '_rlnAnglePsi_radians']].values.T
        )

        filter_params, filter_indices = np.unique(
            self.df[[
                'rlnVoltage',
                'rlnDefocusU',
                'rlnDefocusV',
                '_rlnDefocusAngle_radians',
                'rlnSphericalAberration',
                'rlnAmplitudeContrast'
            ]].values,
            return_inverse=True,
            axis=0
        )

        filters = []
        for row in filter_params:
            filters.append(
                CTFFilter(
                    pixel_size=self.pixel_size,
                    voltage=row[0],
                    defocus_u=row[1],
                    defocus_v=row[2],
                    defocus_ang=row[3],
                    Cs=row[4],
                    alpha=row[5],
                    B=self.B
                )
            )
        filters = SourceFilter(filters, indices=filter_indices)

        offsets = self.df[['rlnOriginX', 'rlnOriginY']].values.T
        amplitudes = np.ones(self.n)
        states = self.df['rlnClassNumber'].values

        ImageSource.__init__(
            self,
            L=self.L,
            n=self.n,
            states=states,
            filters=filters,
            offsets=offsets,
            amplitudes=amplitudes,
            rots=rots,
            dtype=self.dtype
        )
