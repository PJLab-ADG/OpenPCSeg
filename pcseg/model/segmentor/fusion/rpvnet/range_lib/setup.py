import torch
from setuptools import find_packages, setup

has_cuda = torch.cuda.is_available()
if has_cuda:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
else:
    from torch.utils.cpp_extension import BuildExtension, CppExtension

from range_utils import __version__

# Notice that CUDA files, header files should not share names with CPP files.
# Otherwise, there will be "ninja: warning: multiple rules generate xxx.o", which leads to
# multiple definitions error!

file_lis = [
    'range_utils/src/rangelib_bindings_gpu.cpp',
    'range_utils/src/map_count.cpp',
    'range_utils/src/map_count_gpu.cu',
    'range_utils/src/denselize_gpu.cu',
    'range_utils/src/denselize.cpp',
]

extra_compile_args = {
    'cxx': ['-g', '-O3', '-fopenmp', '-lgomp'],
    'nvcc': ['-O3']
} if has_cuda else {
    'cxx': ['-g', '-O3', '-fopenmp', '-lgomp']
}

extension_type = CUDAExtension if has_cuda else CppExtension
setup(
    name='rangelib',
    version=__version__,
    packages=find_packages(),
    ext_modules=[
        extension_type('rangelib_cuda',
                       file_lis,
                       extra_compile_args=extra_compile_args)
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
