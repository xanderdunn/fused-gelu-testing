### Baseline
First, to make sure I have everything setup correctly on my system, I reproduced the vector element-wise addition and matrix multiplication benchmarks from the tutorials. All of the benchmarks in this doc were run on a single A100 machine on GCP with triton built from the commit `3fa8a5a864c48a490625648387a86be3eb7c2c06`.

Vector element-wise addition on my A100 machine:
![vector element-wise addition performance graph triton vs pytorch](vector-add-performnace.png "Vector element-wise performance")
This compares favorably to [the same benchmark in the tutorials](https://triton-lang.org/master/getting-started/tutorials/01-vector-add.html). My machine achieves 1400 GB/s at 10**8 sized vector, whereas the tutorial achieves 800 GB/s at that size.

Matrix-matrix product on my A100 machine:
![matrix product performance graph triton vs pytorch](matmul-performance.png "Matrix-matrix product performance")
This again compares favorably to the absolute values of [the same benchmark in the tutorials](https://triton-lang.org/master/getting-started/tutorials/03-matrix-multiplication.html). My machine achieves 250 TFLOPS at size 4000, whereas the tutorial achieves 90TFLOPS at that size. However, the superiority of the triton kernel over the cuBLAS is less clear in my results.

### Partial GELU Layer Performance
