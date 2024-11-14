# cython/processing.pyx
# This file uses OpenMP via Cython's cython.parallel module

import numpy as np
cimport numpy as np
from cython.parallel import prange

def parallel_resize(np.ndarray[np.float32_t, ndim=3] image, int new_width, int new_height):
    cdef int channels = image.shape[0]
    cdef int old_height = image.shape[1]
    cdef int old_width = image.shape[2]
    cdef np.ndarray[np.float32_t, ndim=3] resized = np.zeros((channels, new_height, new_width), dtype=np.float32)
    cdef int c, i, j
    cdef float scale_x = old_width / new_width
    cdef float scale_y = old_height / new_height

    for c in prange(channels, nogil=True, num_threads=4):  # Adjust num_threads as needed
        for i in range(new_height):
            for j in range(new_width):
                src_x = j * scale_x
                src_y = i * scale_y
                x = int(src_x)
                y = int(src_y)
                resized[c, i, j] = image[c, y, x]

    return resized
