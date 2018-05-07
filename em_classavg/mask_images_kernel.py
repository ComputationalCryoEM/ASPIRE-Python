import time

import numpy as np
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.tools import context_dependent_memoize

@context_dependent_memoize
def get_mask_kernel():
    mod = SourceModule("""
        #include <pycuda-complex.hpp>
       __global__ void mask(pycuda::complex<float>* const arr_in, pycuda::complex<float>* const arr_mask, pycuda::complex<float>* arr_out, int width, int height)
        {
            int i =  blockIdx.y * blockDim.y + threadIdx.y;
            int j =  blockIdx.x * blockDim.x + threadIdx.x;
            int k =  blockIdx.z * blockDim.z + threadIdx.z;


            int stride_x = blockDim.x * gridDim.x;
            int stride_y = blockDim.y * gridDim.y;
          
            for( ; i < height; i += stride_y){
                for( ; j < width; j += stride_x){

                    int ind_msk      = i*width + j;
                    int ind_arr      = k*height*width + i*width + j;
                    
                    arr_out[ind_arr] = arr_in[ind_arr] * arr_mask[ind_msk];
                }
            }
        }
        """)

    return mod.get_function("mask")


def do_mask_gpu(images, mask):

    bdim = (32, 32, 1)

    if images.ndim == 2:
        height, width = images.shape
        depth = 1
    else:
        depth, height, width = images.shape

    mask_height, mask_width = mask.shape
    dx, mx = divmod(width, bdim[0])
    dy, my = divmod(height, bdim[1])
    gdim = ((dx + int(mx > 0)), (dy + int(my > 0)), depth)

    mask_kernel = get_mask_kernel()
    images_masked = gpuarray.empty_like(images)
    mask_kernel(images, mask, images_masked, np.int32(width), np.int32(height),
                block=bdim, grid=gdim)
    return images_masked


def main():
    # create a couple of random matrices with a given shape
    from pycuda.curandom import rand as curand
    depth = 50
    height = 506
    width = 506
    images_in_gpu = curand((depth, height, width)).astype('complex64')
    images_in = images_in_gpu.get()
    images_out_gpu = gpuarray.empty_like(images_in_gpu)

    mask = np.random.randint(2, size=(height,width))
    mask_gpu = gpuarray.to_gpu(mask).astype('complex64')

    n_iters = 100
    t = time.time()
    for i in np.arange(n_iters):
        images_out_cpu = images_in * mask
    print('CPU took %.4f secs' % (time.time() - t))

    import skcuda.linalg as linalg
    images_out_gpu_pycuda = gpuarray.empty_like(images_out_gpu)
    t = time.time()
    for i in np.arange(n_iters):
        for i, image in enumerate(images_in_gpu):
            images_out_gpu_pycuda[i] = linalg.misc.multiply(image, mask_gpu)
    print('GPU pycuda took %.4f secs' % (time.time() - t))
    assert np.allclose(images_out_gpu_pycuda.get(), images_out_cpu)

    t = time.time()
    for i in np.arange(n_iters):
        images_out_gpu = do_mask_gpu(images_in_gpu, mask_gpu)
    print('GPU kernel took %.4f secs' % (time.time() - t))
    assert np.allclose(images_out_gpu.get(), images_out_cpu)


if __name__ == "__main__":
    main()