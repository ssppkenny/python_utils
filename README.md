# How to build
- cython utils.pyx
- LDFLAGS="-framework CoreFoundation -framework CoreGraphics" python setup.py build_ext --inplace

