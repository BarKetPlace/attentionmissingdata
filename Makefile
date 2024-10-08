
dir=causal_product
CLFLAGS=-shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -g -fwrapv -O2 -Wl,-Bsymbolic-functions -g -fwrapv -O2 -g -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2
LFLAGS=-L/home/anthon@ad.cmm.se/projects/dataintegration/pyenv/lib/python3.10/site-packages/torch/lib -lc10 -ltorch -ltorch_cpu -ltorch_python 
CFLAGS=-Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O2 -Wall -g -fstack-protector-strong -Wformat -Werror=format-security -g -fwrapv -O2 -g -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 -fPIC
includes=-I/home/anthon@ad.cmm.se/projects/dataintegration/pyenv/lib/python3.10/site-packages/torch/include -I/home/anthon@ad.cmm.se/projects/dataintegration/pyenv/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/anthon@ad.cmm.se/projects/dataintegration/pyenv/lib/python3.10/site-packages/torch/include/TH -I/home/anthon@ad.cmm.se/projects/dataintegration/pyenv/lib/python3.10/site-packages/torch/include/THC -I/home/anthon@ad.cmm.se/projects/dataintegration/pyenv/include -I/usr/include/python3.10 


all: $(dir)/causal_product_numerator_cpu.cpython-310-x86_64-linux-gnu.so $(dir)/causal_product_denominator_cpu.cpython-310-x86_64-linux-gnu.so

$(dir)/causal_product_%_cpu.cpython-310-x86_64-linux-gnu.so: $(dir)/causal_product_%_cpu.o 
	x86_64-linux-gnu-g++ $(CLFLAGS) $^ $(LFLAGS) -o $@


$(dir)/%.o: $(dir)/%.cpp
	x86_64-linux-gnu-gcc $(CFLAGS) $(includes) -c $^ -o $@ -fopenmp -ffast-math -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -DTORCH_EXTENSION_NAME=$* -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++17


clean:
	rm -f $(dir)/$(dir)/causal_product_numerator_cpu.cpython-310-x86_64-linux-gnu.so $(dir)/causal_product_denominator_cpu.cpython-310-x86_64-linux-gnu.so $(dir)/*.o