#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from setuptools import setup, find_packages, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import sys


if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required for fairseq.')

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    reqs = f.read()


bleu = Extension(
    'fairseq.libbleu',
    sources=[
        'fairseq/clib/libbleu/libbleu.cpp',
        'fairseq/clib/libbleu/module.cpp',
    ],
    extra_compile_args=['-std=c++11'],
)

strided_batched_gemm = CUDAExtension(
                        name='strided_batched_gemm',
                        sources=['fairseq/modules/strided_batched_gemm/strided_batched_gemm.cpp', 'fairseq/modules/strided_batched_gemm/strided_batched_gemm_cuda.cu'],
                        extra_compile_args={
                                'cxx': ['-O2',],
                                'nvcc':['--gpu-architecture=compute_70','--gpu-code=sm_70','-O3','-I./cutlass/','-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__']
                        }
)

padded_softmax = CUDAExtension(
                        name='padded_softmax',
                        sources=['fairseq/modules/padded_softmax/softmax.cu', 'fairseq/modules/padded_softmax/padded_softmax.cpp'],
                        extra_compile_args={
                                'cxx': ['-O2',],
                                #'nvcc':['--gpu-architecture=compute_70','--gpu-code=sm_70','-O3','-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', "--expt-relaxed-constexpr", "--use_fast_math"]
                                'nvcc':['--gpu-architecture=compute_70','--gpu-code=sm_70','-O3','-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', "--expt-relaxed-constexpr"]
                        }
)

fused_dropout_add = CUDAExtension(
                        name='fused_dropout_add_cuda',
                        sources=['fairseq/models/fused_dropout_add/fused_dropout_add_cuda.cpp', 'fairseq/models/fused_dropout_add/fused_dropout_add_cuda_kernel.cu'],
                        extra_compile_args={
                                'cxx': ['-O2',],
                                'nvcc':['--gpu-architecture=sm_70','-O3','--use_fast_math', '--expt-extended-lambda'],
                        }
)
fused_relu_dropout = CUDAExtension(
                        name='fused_relu_dropout_cuda',
                        sources=['fairseq/models/fused_relu_dropout/fused_relu_dropout_cuda.cpp', 'fairseq/models/fused_relu_dropout/fused_relu_dropout_cuda_kernel.cu'],
                        extra_compile_args={
                                'cxx': ['-O2',],
                                'nvcc':['--gpu-architecture=sm_70','-O3','--use_fast_math', '--expt-extended-lambda'],
                        }
)
batch_utils_v0p5 = CppExtension(
                        name='fairseq.data.batch_C_v0p5',
                        sources=['fairseq/data/csrc/make_batches_v0p5.cpp'],
                        extra_compile_args={
                                'cxx': ['-O2',],
                        }
)
batch_utils_v0p5_better = CppExtension(
                        name='fairseq.data.batch_C_v0p5_better',
                        sources=['fairseq/data/csrc/make_batches_v0p5_better.cpp'],
                        extra_compile_args={
                                'cxx': ['-O2', '--std=c++14'],
                        }
)
batch_utils_v0p6 = CppExtension(
                        name='fairseq.data.batch_C_v0p6',
                        sources=['fairseq/data/csrc/make_batches_v0p6.cpp'],
                        extra_compile_args={
                                'cxx': ['-O2', '--std=c++14'],
                        }
)

setup(
    name='fairseq',
    version='0.5.0',
    description='Facebook AI Research Sequence-to-Sequence Toolkit',
    long_description=readme,
    license=license,
    install_requires=reqs.strip().split('\n'),
    packages=find_packages(),
    ext_modules=[bleu, strided_batched_gemm, padded_softmax, fused_dropout_add, fused_relu_dropout, batch_utils_v0p5, batch_utils_v0p5_better, batch_utils_v0p6],
    cmdclass={
                'build_ext': BuildExtension
    },
    test_suite='tests',
)
