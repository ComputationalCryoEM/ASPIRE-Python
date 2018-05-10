import time

import numpy as np
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.tools import context_dependent_memoize
from random import randint


@context_dependent_memoize
def get_slice_assign_2d_kernel():
    mod = SourceModule("""
        #include <pycuda-complex.hpp>
       __global__ void slice_assign_2d(pycuda::complex<float>* arr_dst, pycuda::complex<float>* const arr_src, int k, int width, int height)
        {
            int i =  blockIdx.y * blockDim.y + threadIdx.y;
            int j =  blockIdx.x * blockDim.x + threadIdx.x;


            int stride_x = blockDim.x * gridDim.x;
            int stride_y = blockDim.y * gridDim.y;

            for( ; i < height; i += stride_y){
                for( ; j < width; j += stride_x){

                    int ind_src      = i*width + j;
                    int ind_dst      = k*height*width + i*width + j;

                    arr_dst[ind_dst] = arr_src[ind_src];
                }
            }
        }
        """)

    return mod.get_function("slice_assign_2d")


@context_dependent_memoize
def get_slice_assign_1d_kernel():
    mod = SourceModule("""
        #include <pycuda-complex.hpp>
       __global__ void slice_assign_1d(pycuda::complex<float>* arr_dst, pycuda::complex<float>* const arr_src, int row_num, int width)
        {
            int j =  blockIdx.x * blockDim.x + threadIdx.x;

            int stride_x = blockDim.x * gridDim.x;

            for( ; j < width; j += stride_x){

                arr_dst[row_num*width + j] = arr_src[j];
            }
        }
        """)

    return mod.get_function("slice_assign_1d")

def slice_assign_1d(arr_2d, arr_1d, row_num):

    assert arr_2d.ndim == 2 and arr_1d.ndim == 2
    height, width = arr_2d.shape
    _height, _width = arr_1d.shape

    assert _height == 1
    assert width == _width
    assert row_num < width
    bdim = (1024, 1, 1)

    dx, mx = divmod(width, bdim[0])
    # dy, my = divmod(height, bdim[1])
    gdim = ((dx + int(mx > 0)), 1, 1)

    slice_assign_1d_kernel = get_slice_assign_1d_kernel()
    slice_assign_1d_kernel(arr_2d, arr_1d, np.int32(row_num), np.int32(width), block=bdim, grid=gdim)


def slice_assign_2d(arr_3d, arr_2d, slice_num):

    assert arr_3d.ndim == 3 and arr_2d.ndim == 2
    depth, height, width = arr_3d.shape
    _height, _width = arr_2d.shape

    assert height == _height and width == _width
    assert slice_num < depth
    bdim = (32, 32, 1)

    dx, mx = divmod(width, bdim[0])
    dy, my = divmod(height, bdim[1])
    gdim = ((dx + int(mx > 0)), (dy + int(my > 0)), depth)

    slice_assign_2d_kernel = get_slice_assign_2d_kernel()
    slice_assign_2d_kernel(arr_3d, arr_2d, np.int32(slice_num), np.int32(width), np.int32(height), block=bdim, grid=gdim)


def test_assign_1d():
    # create a couple of random matrices with a given shape
    from pycuda.curandom import rand as curand
    height = 3
    width = 4
    # arr_2d_gpu = curand((height, width)).astype('complex64')
    arr_2d_gpu = gpuarray.zeros((height, width), dtype='complex64')

    arr_1d_gpu = gpuarray.to_gpu(np.arange(width).astype('complex64')).reshape((1,-1))
    # arr_1d_gpu = curand(width).astype('complex64')
    # arr_1d = arr_1d_gpu.get()

    row_num = randint(0, height)
    # row_num = 2

    slice_assign_1d(arr_2d_gpu, arr_1d_gpu, row_num)


    arr_2d = arr_2d_gpu.get()
    arr_1d = arr_1d_gpu.get()
    arr_2d[row_num] = arr_1d

    print(arr_1d)
    print(arr_2d)

    assert np.allclose(arr_2d_gpu.get(), arr_2d)

    t = time.time()
    for i in np.arange(height):
        slice_assign_1d(arr_2d_gpu, arr_1d_gpu, i)
    print('took %.4f secs' % (time.time() - t))


def test_assign_2d():
    # create a couple of random matrices with a given shape
    from pycuda.curandom import rand as curand
    depth = 100
    height = 500
    width = 500
    arr_3d_gpu = curand((depth, height, width)).astype('complex64')
    # arr_3d_gpu = gpuarray.zeros((depth, height, width), dtype='complex64')

    # arr_2d_gpu = gpuarray.to_gpu(np.arange(height * width).reshape(height, width).astype('complex64'))
    arr_2d_gpu = curand((height, width)).astype('complex64')
    # arr_2d = arr_2d_gpu.get()

    slice_num = randint(0, depth)


    slice_assign_2d(arr_3d_gpu, arr_2d_gpu, slice_num)
    # print(arr_2d_gpu.get())
    # print(arr_3d_gpu.get())

    arr_3d = arr_3d_gpu.get()
    arr_2d = arr_2d_gpu.get()
    arr_3d[slice_num] = arr_2d
    assert np.allclose(arr_3d_gpu.get(), arr_3d)

    t = time.time()
    for i in np.arange(depth):
        slice_assign_2d(arr_3d_gpu, arr_2d_gpu, i)
    print('took %.4f secs' % (time.time() - t))


def main():
    test_assign_1d()
    # test_assign_2d()


if __name__ == "__main__":
    main()