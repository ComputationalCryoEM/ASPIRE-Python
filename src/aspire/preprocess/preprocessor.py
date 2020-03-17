import numpy as np
from aspire.preprocess.phaseflip import phaseflip_star_file
from aspire.utils.preprocess import cryo_crop, downsample_preprocess, normalize_background, global_phaseflip
from aspire.preprocess.prewhiten import prewhiten
import time


def preprocess(star_file, pixel_size=None, crop_size=-1, downsample_size=89):
    use_crop = crop_size > 0
    use_downsample = downsample_size > 0
    # flag to indicate not to transform back in phaseflip and to to transform in downsample
    flag = use_downsample and not use_crop
    print('Starting phaseflip')
    tic = time.time()
    stack = phaseflip_star_file(star_file, pixel_size, flag)
    toc = time.time()
    s = stack.shape
    print('Finished phaseflip in {} seconds, found {} images with resolution {}'.format(toc - tic, s[0], s[1]))
    if use_crop:
        print('Start cropping')
        tic = time.time()
        stack = cryo_crop(stack, (-1, crop_size, crop_size))
        toc = time.time()
        print('Finished cropping in {} seconds, from {} to {}'.format(toc - tic, s[1], crop_size))
    else:
        print('Skip cropping')
        crop_size = s[1]
    if use_downsample > 0:
        print('Start downsampling')
        tic = time.time()
        stack = downsample_preprocess(stack, downsample_size, stack_in_fourier=flag)
        toc = time.time()
        print('Finished downsampling in {} seconds, from {} to {}'.format(toc - tic, crop_size, downsample_size))
    else:
        print('Skip downsampling')

    # Up to this point, the stacks are C aligned, now aligning to matlab (in the future it will stay C aligned)
    print('Changing the stack to matlab align (temporary)')
    stack = np.ascontiguousarray(stack.T)

    print('Start normalizing background')
    stack, _, _ = normalize_background(stack, stack.shape[1] * 45 // 100)
    print('Start prewhitening')
    stack = prewhiten(stack)
    print('Start global phaseflip')
    stack = global_phaseflip(stack)
    return stack
