from distutils.core import setup
from Cython.Build import cythonize
import numpy 

setup(
    name = "utils",
    ext_modules = cythonize("utils.pyx"),
    include_dirs=[numpy.get_include()]
)
