#ifndef CHECK_
#define CHECK_

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdio.h>
#include "vector"
#include <iostream>
#define FULLMASK 0xffffffff
#define CEIL(x, y) (((x) + (y)-1) / (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

template <typename T>
__device__ __forceinline__ T __guard_load_default_one(const T *base,
                                                      int offset) {
  if (base != nullptr)
    return base[offset];
  else
    return static_cast<T>(1);
}

template <typename ValueType, typename IndexType>
void segment_coo_sequencial(const ValueType* src, const IndexType* index, const int nnz, const int N, const int dst_len, ValueType* dst){
    for (int i = 0; i < nnz; i++) {
        #pragma unroll
        for (int j = 0; j < N; j++) {
            dst[index[i] * N + j] += src[i * N + j];
        }
    }
}

void checkSegscan(float* dst, float* src, int* index, int nnz, int N, int dst_len){
    float* dst_cpu = (float*)malloc(sizeof(float) * N * dst_len);
    memset(dst_cpu, 0, sizeof(float) * N * dst_len);
    segment_coo_sequencial(src, index, nnz, N, dst_len, dst_cpu);
    for (int i = 0; i < N * dst_len; i++) {
        if (fabs(dst[i] - dst_cpu[i]) > 1e-2 * fabs(dst_cpu[i])) {
            printf("Error[%d][%d]: dst = %f, dst_cpu = %f\n", i / N, i % N, dst[i], dst_cpu[i]);
            return;
        }
    }
    printf("Check passed!\n");
}


#endif