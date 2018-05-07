import time

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray


def init_circ_shift():
    mod = SourceModule("""
        #include <pycuda-complex.hpp>
       __global__ void circ_shift(pycuda::complex<float>* const arr_in, pycuda::complex<float>* arr_out, int shift_x, int shift_y, int width, int height)
        {
            int i =  blockIdx.y * blockDim.y + threadIdx.y;
            int j =  blockIdx.x * blockDim.x + threadIdx.x;
            int k =  blockIdx.z * blockDim.z + threadIdx.z;


            int stride_x = blockDim.x * gridDim.x;
            int stride_y = blockDim.y * gridDim.y;

            int i_out, j_out, ind_in, ind_out;          
            for( ; i < height; i += stride_y){
                for( ; j < width; j += stride_x){

                    i_out = (i + shift_y) % height;
                    j_out = (j + shift_x) % width;

                    ind_in   = k*height*width + i*width + j;
                    ind_out  = k*height*width + i_out*width + j_out;

                    arr_out[ind_out] = arr_in[ind_in];
                }
            }
        }
        """)

    circ_shift_gpu_handle = mod.get_function("circ_shift")
    return circ_shift_gpu_handle

def circ_shift(circ_shift_gpu_handle,mat, mat_out, shift_x, shift_y):

    bdim = (32, 32, 1)
    depth, height, width = mat.shape
    dx, mx = divmod(width, bdim[0])
    dy, my = divmod(height, bdim[1])
    gdim = ((dx + int(mx > 0)), (dy + int(my > 0)), depth)

    # mat_out = gpuarray.empty_like(mat)
    circ_shift_gpu_handle(mat, mat_out, np.int32(shift_x), np.int32(shift_y), np.int32(width), np.int32(height),
                   block=bdim, grid=gdim)

    # return mat_out


width = 65
height = 65
depth = 260

bdim = (32, 32, 1)
dx, mx = divmod(width, bdim[0])
dy, my = divmod(height, bdim[1])
gdim = ((dx + int(mx>0)), (dy + int(my>0)), depth)

shift_x = 20
shift_y = 17

a = np.arange(depth * height * width).reshape(depth, height, width)
a = (a + 1j * (a + 5)).astype('complex64')

b = np.zeros_like(a)

a_gpu = gpuarray.to_gpu(a)
b_gpu = gpuarray.to_gpu(b)
circ_shift_handle = init_circ_shift()

circ_shift(circ_shift_handle,a_gpu, b_gpu, shift_x, shift_y)
b = b_gpu.get()
# print(a)
# print(b)
print(np.all(b == np.roll(np.roll(a, shift_x, axis=2), shift_y, axis=1)))


t = time.time()
b_gpu = gpuarray.empty_like(a_gpu)
for i in np.arange(100):
    circ_shift(circ_shift_handle,a_gpu, b_gpu, shift_x, shift_y)
print('GPU took %.4f secs' % (time.time() - t))

t = time.time()
for i in np.arange(100):
    np.roll(np.roll(a, shift_x, axis=2), shift_y, axis=1)
print('CPU took %.4f secs' % (time.time() - t))
