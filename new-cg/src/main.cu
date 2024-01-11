#include "../include/segscan/segscan.cuh"
#include "../include/utils/check.cuh"
#include "../include/dataloader/dataloader.h"

using namespace std;

int main(int argc, const char **argv) {
    
    int range, nnz, N, max_seg, min_seg;
    double cv; // CV (coefficient of variation) = std / mean
    
    // Random generate [nnz, N] dense vector
    for (int i = 1; i < argc; i++) {
        #define INT_ARG(argname, varname) do {      \
                  if (!strcmp(argv[i], (argname))) {  \
                    varname = atoi(argv[++i]);      \
                    continue;                       \
                  } } while(0);
        #define DOUBLE_ARG(argname, varname) do {      \
                  char* end;                           \
                  if (!strcmp(argv[i], (argname))) {  \
                    varname = strtod(argv[++i], &end);      \
                    continue;                       \
                  } } while(0);
            INT_ARG("-r", range);
            INT_ARG("-nnz", nnz);
            INT_ARG("-min", min_seg);
            INT_ARG("-max", max_seg);
            DOUBLE_ARG("-cv", cv);
            INT_ARG("-N", N);
        #undef INT_ARG
    }
    std::vector<int> index;
    int dst_len = generateIndex(range, min_seg, max_seg, nnz, cv, index);
    std::vector<float> src;
    generateSrc(nnz, N, src);
    std::vector<float> dst(range * N, 0);
    // Copy the dst to GPU
    float *d_src, *d_dst;
    int *d_index;
    checkCudaError(cudaMalloc((void **)&d_src, sizeof(float) * N * nnz));
    checkCudaError(cudaMalloc((void **)&d_dst, sizeof(float) * N * range));
    checkCudaError(cudaMalloc((void **)&d_index, sizeof(int) * nnz));
    checkCudaError(cudaMemcpy((void*)d_src, (void*)src.data(), sizeof(float) * N * nnz, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy((void*)d_index, (void*)index.data(), sizeof(int) * nnz, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy((void*)d_dst, (void*)dst.data(), sizeof(float) * N * range, cudaMemcpyHostToDevice));
    // Call the kernel
    // segment_coo<float,int,5,256,5,1,16>(d_src, d_index, nnz, N, dst_len, d_dst);
    // segment_coo_new<float,int,5,256,5,2>(d_src, d_index, nnz, N, dst_len, d_dst);
    segment_coo_sr<float,int,4,256,4,1,1,16>(d_src, d_index, nnz, N, d_dst);
    // Copy the dst back to CPU
    checkCudaError(cudaMemcpy((void*)dst.data(), (void*)d_dst, sizeof(float) * N * range, cudaMemcpyDeviceToHost));
    // Check the result
    checkSegscan(dst.data(), src.data(), index.data(), nnz, N, range);
    checkCudaError(cudaFree((void *)d_src));
    checkCudaError(cudaFree((void *)d_dst));
    checkCudaError(cudaFree((void *)d_index));
    return 0;
}