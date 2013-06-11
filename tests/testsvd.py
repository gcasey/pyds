import time

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print '[%s]' % self.name,
        print 'Elapsed: %s' % (time.time() - self.tstart)

import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import numpy as np
import scikits.cuda.linalg as cla
import numpy.linalg as la
import scikits.cuda.cula as cula

cla.init()

x = np.random.rand(256**2, 40).astype(np.float32)

def svdoverwrite(a_gpu, u_gpu, s_gpu, v_gpu, m, n, lda, ldu, ldvt):
    data_type = a_gpu.dtype.type
    real_type = np.float32
    cula_func = cula._libcula.culaDeviceSgesvd
    jobu = 'S'
    jobvt = 'S'

    status = cula_func(jobu, jobvt, m, n, int(a_gpu.gpudata),
                       lda, int(s_gpu.gpudata), int(u_gpu.gpudata),
                       ldu, int(v_gpu.gpudata), ldvt)

    cula.culaCheckStatus(status)

    # Free internal CULA memory:
    cula.culaFreeBuffers()

with Timer('Push results'):
    gpux = gpuarray.to_gpu(x)

with Timer('GPU SVD'):
    u_g, s_g, v_g = cla.svd(gpux, 'S', 'S')   

with Timer('CPU SVD'):
    u_c, s_c, v_c = la.svd(x, full_matrices=False)

with Timer('Get results'):
    u_g = u_g.get()
    s_g = s_g.get()
    v_g = v_g.get()
