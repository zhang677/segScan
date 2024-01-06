#include "../include/segscan/segscan.cuh"
#include "../include/utils/check.cuh"
#include "../include/dataloader/dataloader.h"

using namespace std;

int main(int argc, const char **argv) {
    
    int range, max_seg, nnz, N;
    // Random generate [nnz, N] dense vector
    for (int i = 1; i < argc; i++) {
        #define INT_ARG(argname, varname) do {      \
                  if (!strcmp(argv[i], (argname))) {  \
                    varname = atoi(argv[++i]);      \
                    continue;                       \
                  } } while(0);
            INT_ARG("-r", range);
            INT_ARG("-nnz", nnz);
            INT_ARG("-max", max_seg);
            INT_ARG("-N", N);
        #undef INT_ARG
    }

    std::vector<int> index;
    int dst_len = generateIndex(range, max_seg, nnz, index);
    std::vector<float> src;
    generateSrc(nnz, N, src);
    float dst[dst_len * N];
    // Copy the dst to GPU
    float *d_src, *d_dst;
    int *d_index;
    cudaMalloc((void **)&d_src, sizeof(float) * N * nnz);
    cudaMalloc((void **)&d_dst, sizeof(float) * N * dst_len);
    cudaMalloc((void **)&d_index, sizeof(int) * nnz);
    cudaMemcpy(d_src, src.data(), sizeof(float) * N * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_index, index.data(), sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, dst, sizeof(float) * N * dst_len, cudaMemcpyHostToDevice);
    // Call the kernel
    segment_coo<float,int,5,256,5,1,16>(d_src, d_index, nnz, N, dst_len, d_dst);
    // Copy the dst back to CPU
    cudaMemcpy(dst, d_dst, sizeof(float) * N * dst_len, cudaMemcpyDeviceToHost);
    // Check the result
    checkSegscan(dst, src.data(), index.data(), nnz, N, dst_len);
    return 0;
}