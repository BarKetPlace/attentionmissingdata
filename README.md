# Causal attention with missing data
The goal is to modify the cpp and cuda kernels for causal dot product proposed in https://github.com/idiap/fast-transformers , to be able to use them with non triangular causal masks.

## The existing repo
- The dot product function is read from a local copy of the shared cpp library 
```bash
python main.py
```
- The source code for the library, as well as the python binding code is available in the folder `causal_product`. 

- I compiled the library locally and copied the two lines related to (1) compiling `causal_product_cpu.cpp` and (2) creating the shared `.so` library in `causal_product/compile.sh`
```bash
./compile.sh
```

- Problem the compiling does not work.