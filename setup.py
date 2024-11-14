from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "cython.processing",
        ["cython/processing.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()],
    )
]

setup(
    name="processing",
    ext_modules=cythonize(extensions),
)
