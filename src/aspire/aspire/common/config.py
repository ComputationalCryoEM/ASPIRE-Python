
class AspireConfig:
    log_file_mode = 'w'  # change to 'a' to avoid trimming on each run
    verbosity = 0
    binaries_folder = 'binaries'


class ClassAveragesConfig(AspireConfig):
    bessel_file = "./binaries/bessel.npy"


class AbinitioConfig(AspireConfig):
    algo = 2  # currently only supports 2
    n_theta = 360
    n_r = 0.5
    max_shift = 0.15
    shift_step = 1
    fuzzy_mask_dims = 2
    rise_time = 2


class PreProcessorConfig(AspireConfig):
    crop_stack_fill_value = 0


necessary_workflow_fields = {'info': ['working_dir',
                                      'logfile',
                                      'rawdata'
                                      ],

                             'preprocess': ['phaseflip',
                                            'nprojs',
                                            'do_crop',
                                            'croppeddim',
                                            'do_downsample',
                                            'downsampleddim',
                                            'do_normalize',
                                            'do_prewhiten',
                                            'split',
                                            'numgroups'
                                            ]
                             }