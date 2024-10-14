#!/bin/bash
# Compile of the causal_product shared library.
# These lines were copy pasted from the build process of the full library:
# git clone https://github.com/idiap/fast-transformers
# cd fast-transformers
# python setup.py install
# 
dir=causal_product
/usr/local/cuda-12.6/bin/nvcc -I/home/anthon@ad.cmm.se/projects/dataintegration/pyenv/lib/python3.12/site-packages/torch/include -I/home/anthon@ad.cmm.se/projects/dataintegration/pyenv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -I/home/anthon@ad.cmm.se/projects/dataintegration/pyenv/lib/python3.12/site-packages/torch/include/TH -I/home/anthon@ad.cmm.se/projects/dataintegration/pyenv/lib/python3.12/site-packages/torch/include/THC -I/usr/local/cuda-12.6/include -I/home/anthon@ad.cmm.se/projects/dataintegration/pyenv/include -I/usr/include/python3.12 -c fast_transformers/causal_product/causal_product_cuda.cu -o build/temp.linux-x86_64-cpython-312/fast_transformers/causal_product/causal_product_cuda.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options '-fPIC' -arch=compute_60 -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -DTORCH_EXTENSION_NAME=causal_product_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++17

x86_64-linux-gnu-g++ -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -g -fwrapv -O2 build/temp.linux-x86_64-cpython-312/fast_transformers/causal_product/causal_product_cuda.o -L/home/anthon@ad.cmm.se/projects/dataintegration/pyenv/lib/python3.12/site-packages/torch/lib -L/usr/local/cuda-12.6/lib64 -L/usr/lib/x86_64-linux-gnu -lc10 -ltorch -ltorch_cpu -ltorch_python -lcudart -lc10_cuda -ltorch_cuda -o build/lib.linux-x86_64-cpython-312/fast_transformers/causal_product/causal_product_cuda.cpython-312-x86_64-linux-gnu.so


