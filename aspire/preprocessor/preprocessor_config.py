"""
This module holds all configurations needed for preprocessing.
"""

# these entires must exist in the workflow file
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
