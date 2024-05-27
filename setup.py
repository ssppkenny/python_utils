from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = "utils",
    ext_modules = cythonize("utils.pyx")
)
