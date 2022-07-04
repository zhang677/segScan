#ifndef GROUP_CUH
#define GROUP_CUH

#include "check.cuh"
#include <cooperative_groups.h>
using namespace cooperative_groups;

template<typename DType>
__device__ __forceinline__ DType shfl_down_reduce(thread_block_tile<GROUP_SIZE,thread_block>& g, DType v){
    for (int i = g.size()>>1 ; i>0; i = i >> 1) {
        v += g.shfl_down(v, i);
    }
    return v;
}


#endif