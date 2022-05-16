from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='sdf_cuda',
    ext_modules=[
        CUDAExtension('sdf_cuda', [
            'sdf_cuda.cpp',
            'sdf_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
