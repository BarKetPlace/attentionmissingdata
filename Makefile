root_dir=.
ifdef remote
	root_dir=/mnt/berzelius/attentionmissingdata
endif
ifndef CUDA
	CUDA=12.2
endif

python_root=$(shell pwd)/pyenv

dir=$(root_dir)/src/causal_product

CLFLAGS=-shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -g -fwrapv -O2 -Wl,-Bsymbolic-functions -g -fwrapv -O2 -g -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2
LFLAGS=-L$(python_root)/lib/python3.12/site-packages/torch/lib -lc10 -ltorch -ltorch_cpu -ltorch_python 
CFLAGS=-Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O2 -Wall -g -fstack-protector-strong -Wformat -Werror=format-security -g -fwrapv -O2 -g -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 -fPIC
includes=-I$(python_root)/lib/python3.12/site-packages/torch/include -I$(python_root)/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -I$(python_root)/lib/python3.12/site-packages/torch/include/TH -I$(python_root)/lib/python3.12/site-packages/torch/include/THC -I$(python_root)/include -I/usr/include/python3.12 


all:  gpu cpu ref

gpu: $(dir)/causal_product_numerator_cuda.cpython-312-x86_64-linux-gnu.so
cpu: $(dir)/causal_product_numerator_cpu.cpython-312-x86_64-linux-gnu.so
ref: $(dir)/causal_product_cpu.cpython-312-x86_64-linux-gnu.so

$(dir)/causal_product_cpu.cpython-312-x86_64-linux-gnu.so:$(dir)/causal_product_cpu.o 
	x86_64-linux-gnu-g++ $(CLFLAGS) $^ $(LFLAGS) -o $@

$(dir)/causal_product_%_cuda.cpython-312-x86_64-linux-gnu.so: $(dir)/causal_product_%_cuda.o 
	x86_64-linux-gnu-g++ -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -g -fwrapv -O2 $^ -L$(python_root)/lib/python3.12/site-packages/torch/lib -L/usr/local/cuda-$(CUDA)/lib64 -L/usr/lib/x86_64-linux-gnu -lc10 -ltorch -ltorch_cpu -ltorch_python -lcudart -lc10_cuda -ltorch_cuda -o $@


$(dir)/causal_product_%_cpu.cpython-312-x86_64-linux-gnu.so: $(dir)/causal_product_%_cpu.o 
	x86_64-linux-gnu-g++ $(CLFLAGS) $^ $(LFLAGS) -o $@


$(dir)/%_cpu.o: $(dir)/%_cpu.cpp
	x86_64-linux-gnu-gcc $(CFLAGS) $(includes) -c $^ -o $@ -fopenmp -ffast-math -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -DTORCH_EXTENSION_NAME=$*_cpu -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++17


$(dir)/%_cuda.o: $(dir)/%_cuda.cu
	/usr/local/cuda-$(CUDA)/bin/nvcc -ccbin /usr/bin/g++-12 $(includes) -I/usr/local/cuda-$(CUDA)/include -c $^ -o $@ -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options '-fPIC' -arch=compute_60 -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -DTORCH_EXTENSION_NAME=$*_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++17

clean:
	rm -f $(dir)/causal_product_numerator_cpu.cpython-310-x86_64-linux-gnu.so $(dir)/causal_product_denominator_cpu.cpython-310-x86_64-linux-gnu.so $(dir)/*.o
