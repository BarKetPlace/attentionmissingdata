# Causal attention with missing data

## What I have and want to do
The goal is to modify the cpp and cuda kernels for causal dot product proposed in https://github.com/idiap/fast-transformers, to be able to use them with non-triangular causal masks.

- The source code for the library, as well as the python binding code is available in the folder `causal_product`. 

- You can run the pre-compiled dot product function from a local copy of the shared cpp library 
```bash
python main.py
```

- The two lines related to (1) compiling `causal_product/causal_product_cpu.cpp` and (2) creating the shared `.so` library are in `causal_product/compile.sh`

### Problems
- Compiling the cpp library does not work.

```bash
cd causal_product
./compile.sh
```

- I don't know where to start with the cuda kernel (source code in `causal_product/causal_product_cuda.cu`)