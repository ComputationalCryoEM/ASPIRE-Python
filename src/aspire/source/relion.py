from aspire.source.starfile import StarfileStack


class RelionStarfileStack(StarfileStack):

    _metadata_aliases = {
        '_image_name':  '_rlnImageName',
        '_offset_x':    '_rlnOriginX',
        '_offset_y':    '_rlnOriginY',
        '_state':       '_rlnClassNumber',
        '_angle_0':     '_rlnAngleRot',
        '_angle_1':     '_rlnAngleTilt',
        '_angle_2':     '_rlnAnglePsi',
        '_amplitude':   '_amplitude',
        '_voltage':     '_rlnVoltage',
        '_defocus_u':   '_rlnDefocusU',
        '_defocus_v':   '_rlnDefocusV',
        '_defocus_ang': '_rlnDefocusAngle',
        '_Cs':          '_rlnSphericalAberration',
        '_alpha':       '_rlnAmplitudeContrast'
    }

    _metadata_types = {
        '_rlnVoltage': float,
        '_rlnDefocusU': float,
        '_rlnDefocusV': float,
        '_rlnDefocusAngle': float,
        '_rlnSphericalAberration': float,
        '_rlnDetectorPixelSize': float,
        '_rlnCtfFigureOfMerit': float,
        '_rlnMagnification': float,
        '_rlnAmplitudeContrast': float,
        '_rlnImageName': str,
        '_rlnOriginalName': str,
        '_rlnCtfImage': str,
        '_rlnCoordinateX': float,
        '_rlnCoordinateY': float,
        '_rlnCoordinateZ': float,
        '_rlnNormCorrection': float,
        '_rlnMicrographName': str,
        '_rlnGroupName': str,
        '_rlnGroupNumber': str,
        '_rlnOriginX': float,
        '_rlnOriginY': float,
        '_rlnAngleRot': float,
        '_rlnAngleTilt': float,
        '_rlnAnglePsi': float,
        '_rlnClassNumber': int,
        '_rlnLogLikeliContribution': float,
        '_rlnRandomSubset': int,
        '_rlnParticleName': str,
        '_rlnOriginalParticleName': str,
        '_rlnNrOfSignificantSamples': float,
        '_rlnNrOfFrames': int,
        '_rlnMaxValueProbDistribution': float
    }
