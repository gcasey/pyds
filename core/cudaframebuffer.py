"""lds.py
"""


__license__ = "Apache License, Version 2.0"
__author__  = "Casey Goodlett, Kitware Inc., 2013"
__email__   = "E-Mail: casey.goodlett@kitware.com"
__status__  = "Development"

import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
from pycuda.compiler import SourceModule
import scikits.cuda.linalg as cla
import numpy as np

cla.init()

mod = SourceModule('''
__global__ void cyclebuffer(float *src, float* newframe, float* dst,
                            int rows, int cols)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if(col == 0 && row < rows)
    {
    dst[row*cols] = newframe[row];
    }
  else if((row < rows) && (col < cols))
    {
    dst[row*cols + col] = src[row*cols + col - 1];
    }

}
''')

class cudabuffer:
    """Implements a linear dynamical system (LDS) of the form:
    
    x_{t+1} = A*x_t + w_t
    y_{t}   = C*x_t + v_t
    
    where x_t is the state at time t, y_t is the observations at time t. w_t
    and v_t are the state / observation noise, respectively.
    """
    
    cyclebuffer = mod.get_function('cyclebuffer')

    def __init__(self, Y, verbose=False):
        """Initialize instance.
        """
        self._verbose = verbose
        self._Y_gpu = gpuarray.to_gpu_async(Y)
        self._Y_gpu_scratch = gpuarray.to_gpu_async(Y)
        self._featuredim = Y.shape[0]
        self._framecount = Y.shape[1]

    def add_new_frame(self, x):
        x = x.flatten()
        assert len(x) == self._featuredim
        
        x_gpu = gpuarray.to_gpu_async(x)
        BLOCK_SIZE = (1,256,1)
        nblocks = int(np.ceil(float(self._featuredim) / BLOCK_SIZE[0]))
        GRID_SIZE = (self._framecount, nblocks, 1)

        cudabuffer.cyclebuffer(self._Y_gpu_scratch, x_gpu, self._Y_gpu,
                               np.int32(self._featuredim), np.int32(self._framecount),
                               block=BLOCK_SIZE, grid=GRID_SIZE)

        # Copy self._Y_gpu into self._Y_gpu_scratch
        cuda.memcpy_dtod_async(self._Y_gpu_scratch.gpudata, self._Y_gpu.gpudata, self._Y_gpu.nbytes)
        
    def current_frames(self):
        return self._Y_gpu.get()
        
    def decompose(self):
        gcov = cla.dot(self._Y_gpu, self._Y_gpu, transa='C')
        ge_g, gh_g = np.linalg.eigh(gcov.get())
        I = np.argsort(ge_g)[::-1]
        ge_g, gh_g = np.sqrt(ge_g[I]), gh_g[:,I]
        # push the matrix back out
        gpueigs = gpuarray.to_gpu(gh_g)
        W_g = cla.dot(self._Y_gpu, gpueigs)
        # Unitize W_g - could be done on gpu to allow async returning
        W_g = W_g.get()
        W_g = W_g / np.sqrt(np.sum(W_g**2, axis=0))[np.newaxis, :]

        return W_g, ge_g, gh_g.T # Not sure whether the last one should be transposed


if __name__ == '__main__':
    Y = np.random.random( (512, 4) ).astype(np.float32)
    mybuffer = cudabuffer(Y)
    print mybuffer.current_frames().astype(int)

    # mybuffer.add_new_frame(9*np.ones(Y.shape[0],).astype(np.float32))
    # print mybuffer.current_frames().astype(int)

    # mybuffer.add_new_frame(6*np.ones(Y.shape[0],).astype(np.float32))
    # print mybuffer.current_frames().astype(int)

    U_g, w, V = mybuffer.decompose()
    print w
    print V
    print U_g
    
    print '----'
    U, v, W = np.linalg.svd(mybuffer.current_frames())
    print v
    print W
    print U
