try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import numpy


# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

# mise (efficient mesh extraction)
mise_module = Extension(
    'src.lib.libmise.mise',
    sources=[
        'src/lib/libmise/mise.pyx'
    ],
    include_dirs=[numpy_include_dir]
)


triangle_hash_module = Extension(
    'src.lib.libmesh.triangle_hash',
    sources=[
        'src/lib/libmesh/triangle_hash.pyx'
    ],
    libraries=['m'], # Unix-like specific
    include_dirs=[numpy_include_dir]
)

mcubes_module = Extension(
    'src.lib.libmcubes.mcubes',
    sources=[
        'src/lib/libmcubes/mcubes.pyx',
        'src/lib/libmcubes/pywrapper.cpp',
        'src/lib/libmcubes/marchingcubes.cpp'
    ],
    language='c++',
    extra_compile_args=['-std=c++11'],
    include_dirs=[numpy_include_dir]
)

# Gather all extension modules
ext_modules = [
    mise_module,
    triangle_hash_module,
    mcubes_module,
]

setup(
    ext_modules=cythonize(ext_modules),
    cmdclass={
        'build_ext': BuildExtension
    }
)
