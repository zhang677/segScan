#ifndef CHECK_
#define CHECK_

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "vector"

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

namespace util {
    template <typename ValueType, typename IndexType>
    void segment_coo_sequencial(const ValueType* src, const IndexType* index, const int nnz, const int N, const int dst_len, ValueType* dst){
        for (int i = 0; i < nnz; i++) {
            #pragma unroll
            for (int j = 0; j < N; j++) {
                dst[index[i] * N + j] += src[i * N + j];
            }
        }
    }
}

#endif