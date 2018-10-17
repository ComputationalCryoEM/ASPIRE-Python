
class AspireConfig:
    log_file_mode = 'w'  # change to 'a' to avoid trimming on each run
    verbosity = 0
    binaries_folder = 'binaries'


class ClassAveragesConfig(AspireConfig):
    pass


class AbinitioConfig(AspireConfig):
    pass


class PreProcessorConfig(AspireConfig):
    pass


class CropStackConfig(AspireConfig):
    fill_value = 0
