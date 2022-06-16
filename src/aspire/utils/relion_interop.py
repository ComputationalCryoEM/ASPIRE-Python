class RlnOpticsGroup:
    def __init__(self, df_row):
        """
        Creates a simple object to store optics group data given a pandas dataframe row.
        """
        self.name = df_row._rlnOpticsGroupName
        self.number = int(df_row._rlnOpticsGroup)
        self.voltage = float(df_row._rlnVoltage)
        self.cs = float(df_row._rlnSphericalAberration)
        self.amplitude_contrast = float(df_row._rlnAmplitudeContrast)
        self.pixel_size = float(df_row._rlnMicrographPixelSize)
