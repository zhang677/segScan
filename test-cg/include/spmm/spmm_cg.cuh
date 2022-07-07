#ifndef SPMM_CG
#define SPMM_CG

#include "../dataloader/dataloader.hpp"
#include "../util/check.cuh"
#include "../util/ramArray.cuh"
#include "spmm.cuh"
#include <cuda.h>
#include <cusparse.h>
#include <cooperative_groups.h>
#include <iostream>
using namespace cooperative_groups;

template <typename access_t, int group_size, int tile_size>
__global__ void csrspmm_parreduce_nnzbalance_cg_kernel(
    const int M, const int N, const int K, const int nnz_,
    const int csr_indptr[], const int csr_indices[], const float csr_data[],
    const float B[], float C[]) {
    constexpr int CoarsenFactor = sizeof(access_t) / sizeof(float);
    int nnz = nnz_;
    if (nnz < 0)
        nnz = csr_indptr[M];
    
    int lane_id = (threadIdx.x & (tile_size - 1));
    int Nnzdim_warp_id = blockIdx.x * blockDim.y + threadIdx.y;
    int nz_start = Nnzdim_warp_id * tile_size;
    int stride = gridDim.x * (blockDim.y * tile_size);
    
    // get the dense column offset
    int col_offset = (blockIdx.y * tile_size) + (threadIdx.x / tile_size) * CoarsenFactor;
    const float *B_panel = B + col_offset;
    float *C_panel = C + col_offset;
    int ldB = N;
    int ldC = N;
    
    int k;
    float v;
    float c[CoarsenFactor] = {0};
    float buffer[CoarsenFactor] = {0};
    thread_block_tile<group_size,thread_block> group = tiled_partition<group_size>(this_thread_block());

    if (col_offset >= N)
        return;
    if (col_offset + CoarsenFactor >= N)
        goto Ndim_Residue;
    
    for (int nz_id = nz_start + lane_id;
            nz_id < nnz + lane_id; // make sure NO warp loop-divergence
            nz_id += stride) {
        int row = binary_search_segment_number<int>(csr_indptr, M, nnz, nz_id);
    
        if (nz_id < nnz) {
        k = csr_indices[nz_id];
        v = __guard_load_default_one<float>(csr_data, nz_id);
        } else {
        k = 0;
        v = 0.0f;
        }
    
        // load B-elements in vector-type
        *(access_t *)buffer = *(access_t *)(B_panel + k * ldB);
    #pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
        c[i] = buffer[i] * v;
        }
        


        // reduction
        int row_intv = group.shfl(row, group.size()-1) - group.shfl(row, 0);
        if (row_intv == 0) {
    // if all non-zeros in this warp belong to the same row, use a simple reduction
    #pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
            for (int k = group.size()>>1 ; k>0; k = k >> 1) {
                c[i] += group.shfl_down(c[i], k);
            }
        }
        if (group.thread_rank() == 0) {
    #pragma unroll
            for (int i = 0; i < CoarsenFactor; i++) {
            atomicAdd(C_panel + row * ldC + i, c[i]);
            }
        }
        } else {
        // if non-zeros belong to different rows, use a parallel-scan primitive
        // thread that holds the start of each segment are responsible for writing
        // results
        bool is_seg_start = ((group.shfl_up(row,1) != row)|| (group.thread_rank() == 0));
        float tmpv;
        int tmpr;
    #pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
            for (k = 1; k<group.size(); k = k<<1) {
                tmpv = group.shfl_down(c[i],k);
                tmpr = group.shfl_down(row,k);
                if (tmpr == row && group.thread_rank() < (group.size()-k)) {
                    c[i] += tmpv;
                }
            }
        }
        if (is_seg_start) {
    // atomic add has no vector-type form.
    #pragma unroll
            for (int i = 0; i < CoarsenFactor; i++) {
            atomicAdd(C_panel + row * ldC + i, c[i]);
            }
        }
        }
    }
    return;
    Ndim_Residue:
  int valid_lane_num = N - col_offset;

  for (int nz_id = nz_start + lane_id;
       nz_id < nnz + lane_id; // make sure NO warp loop-divergence
       nz_id += stride) {
    int row = binary_search_segment_number<int>(csr_indptr, M, nnz, nz_id);

    if (nz_id < nnz) {
      k = csr_indices[nz_id];
      v = __guard_load_default_one<float>(csr_data, nz_id);
    } else {
      k = 0;
      v = 0.0f;
    }

#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      if (i < valid_lane_num) {
        c[i] = B_panel[k * ldB + i] * v;
      }
    }

    // reduction
    int row_intv = group.shfl(row, group.size()-1) - group.shfl(row, 0);
    if (row_intv == 0) {
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        for (int k = group.size()>>1 ; k>0; k = k >> 1) {
            c[i] += group.shfl_down(c[i], k);
        }
      }
      if (group.thread_rank() == 0) {
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          if (i < valid_lane_num) {
            atomicAdd(C_panel + row * ldC + i, c[i]);
          }
        }
      }
    } else {
      bool is_seg_start = ((group.shfl_up(row,1) != row)|| (group.thread_rank() == 0));
      float tmpv;
      int tmpr;
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        for (k = 1; k<group.size(); k = k<<1) {
            tmpv = group.shfl_down(c[i],k);
            tmpr = group.shfl_down(row,k);
            if (tmpr == row && group.thread_rank() < (group.size()-k)) {
                c[i] += tmpv;
            }
        }
      }
      if (is_seg_start) {
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          if (i < valid_lane_num) {
            atomicAdd(C_panel + row * ldC + i, c[i]);
          }
        }
      }
    }
  }
  return;
    }
template <typename Index, typename DType, int group_factor, int thread_per_block, int tile_factor, int block_numer,int block_denom>
void csrspmm_parreduce_nnzbalance_cg(SpMatCsrDescr_t<Index, DType>& spmatA, 
    const int N, const DType *B, DType *C) {

    // factor of thread coarsening
    int coarsen_factor = (N % 4 == 0) ? 4 : (N % 2 == 0) ? 2 : 1;
    // number of parallel warps along M-dimension
    float block_factor = (float)block_numer / (float)block_denom;
    int Nnzdim_worker = (float)spmatA.nrow * block_factor;
    // partition large-N and map to blockdim.y to help cache performance
    int tile_size = 1<<tile_factor;
    int Ndim_threadblock = CEIL(N, tile_size); // 1
    int Ndim_warp_per_tb = min(N, tile_size) / coarsen_factor; // 32

    int ref_warp_per_tb = thread_per_block / tile_size; // 512/128 = 4 
    int Nnzdim_warp_per_tb = CEIL(ref_warp_per_tb, Ndim_warp_per_tb); // 1

    // total number of warps
    int gridDimX = CEIL(Nnzdim_worker, Nnzdim_warp_per_tb);
    int gridDimY = Ndim_threadblock;
    dim3 gridDim(gridDimX, gridDimY, 1);
    dim3 blockDim(Ndim_warp_per_tb * tile_size, Nnzdim_warp_per_tb, 1);

    if (coarsen_factor == 4) {
    csrspmm_parreduce_nnzbalance_cg_kernel<float4, 1<<group_factor, 1<<tile_factor ><<<gridDim, blockDim>>>(
    spmatA.nrow, N, spmatA.ncol, spmatA.nnz, spmatA.sp_csrptr.d_array.get(), spmatA.sp_csrind.d_array.get(),
    spmatA.sp_data.d_array.get(), B, C);
    } else if (coarsen_factor == 2) {
    csrspmm_parreduce_nnzbalance_cg_kernel<float2, 1<<group_factor, 1<<tile_factor ><<<gridDim, blockDim>>>(
    spmatA.nrow, N, spmatA.ncol, spmatA.nnz, spmatA.sp_csrptr.d_array.get(), spmatA.sp_csrind.d_array.get(),
    spmatA.sp_data.d_array.get(), B, C);
    } else {
    csrspmm_parreduce_nnzbalance_cg_kernel<float, 1<<group_factor, 1<<tile_factor ><<<gridDim, blockDim>>>(
    spmatA.nrow, N, spmatA.ncol, spmatA.nnz, spmatA.sp_csrptr.d_array.get(), spmatA.sp_csrind.d_array.get(),
    spmatA.sp_data.d_array.get(), B, C);
    }
    /*
    cudaDeviceSynchronize();
    std::cout<<"("<<gridDim.x<<","<<gridDim.y<<","<<gridDim.z<<")"<<std::endl;
    std::cout<<"("<<blockDim.x<<","<<blockDim.y<<","<<blockDim.z<<")"<<std::endl;  
    gpuErrchk(cudaGetLastError());
    */
}

#endif