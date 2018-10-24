
class AspireConfig:
    log_file_mode = 'w'  # change to 'a' to avoid trimming on each run
    verbosity = 0
    binaries_folder = 'binaries'


class ClassAveragesConfig(AspireConfig):
    pass


class AbinitioConfig(AspireConfig):
    algo = 2  # currently only supports 2
    n_theta = 360
    n_r = 0.5
    max_shift = 0.15
    shift_step = 1
    fuzzy_mask_dims = 2
    rise_time = 2


class PreProcessorConfig(AspireConfig):
    pass


class CropStackConfig(AspireConfig):
    fill_value = 0
