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
    std::cout << "range = " << range << ", nnz = " << nnz << ", max_seg = " << max_seg << ", N = " << N << ", dst_len = " << dst_len << std::endl;
    std::vector<float> src;
    generateSrc(nnz, N, src);
    std::vector<float> dst(dst_len * N, 0);
    // Copy the dst to GPU
    float *d_src, *d_dst;
    int *d_index;
    checkCudaError(cudaMalloc((void **)&d_src, sizeof(float) * N * nnz));
    checkCudaError(cudaMalloc((void **)&d_dst, sizeof(float) * N * dst_len));
    checkCudaError(cudaMalloc((void **)&d_index, sizeof(int) * nnz));
    checkCudaError(cudaMemcpy((void*)d_src, (void*)src.data(), sizeof(float) * N * nnz, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy((void*)d_index, (void*)index.data(), sizeof(int) * nnz, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy((void*)d_dst, (void*)dst.data(), sizeof(float) * N * dst_len, cudaMemcpyHostToDevice));
    // Call the kernel
    // segment_coo<float,int,5,256,5,1,16>(d_src, d_index, nnz, N, dst_len, d_dst);
    segment_coo_new<float,int,5,256,5,2>(d_src, d_index, nnz, N, dst_len, d_dst);
    // Copy the dst back to CPU
    checkCudaError(cudaMemcpy((void*)dst.data(), (void*)d_dst, sizeof(float) * N * dst_len, cudaMemcpyDeviceToHost));
    // Check the result
    checkSegscan(dst.data(), src.data(), index.data(), nnz, N, dst_len);
    checkCudaError(cudaFree((void *)d_src));
    checkCudaError(cudaFree((void *)d_dst));
    checkCudaError(cudaFree((void *)d_index));
    return 0;
}