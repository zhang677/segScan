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

template <typename access_t, int group_size, int tile_size>
__global__ void csrspmm_parreduce_rowbalance_cg_kernel(
    const int M, const int N, const int K, const int csr_indptr[],
    const int csr_indices[], const float csr_data[], const float B[],
    float C[]) {
  constexpr int CoarsenFactor = sizeof(access_t) / sizeof(float);

  int lane_id = (threadIdx.x & (tile_size - 1));
  int stride = gridDim.x * blockDim.y;
  int row = blockIdx.x * blockDim.y + threadIdx.y;

  // get the dense column offset
  int col_offset = blockIdx.y * tile_size + (threadIdx.x / tile_size) * CoarsenFactor;
  const float *B_panel = B + col_offset;
  float *C_panel = C + col_offset;
  int ldB = N;
  int ldC = N;
  // The largest group_size is 32
  thread_block_tile<group_size,thread_block> group = tiled_partition<group_size>(this_thread_block());
  
  if (col_offset >= N)
    return;
  if (col_offset + CoarsenFactor >= N)
    goto Ndim_Residue;

  for (; row < M; row += stride) {
    // declare accumulators
    float c[CoarsenFactor] = {0};
    float buffer[CoarsenFactor];

    int start = csr_indptr[row];
    int end = csr_indptr[row + 1];
    int k;
    float v;

    for (int jj = start + lane_id; jj < end; jj += tile_size) {
      k = csr_indices[jj];
      v = __guard_load_default_one<float>(csr_data, jj);

      // load B-elements in vector-type
      *(access_t *)buffer = *(access_t *)(B_panel + k * ldB);

#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        c[i] += v * buffer[i];
      }
    }

#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      for (int k = group.size()>>1 ; k>0; k = k >> 1) {
        c[i] += group.shfl_down(c[i], k);
      }
    }
    if (group.thread_rank() == 0) {
// atomic add has no vector-type form.
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        atomicAdd(C_panel + row * ldC + i, c[i]);
      }
    }
  }
  return;

Ndim_Residue:
  int valid_lane_num = N - col_offset;

  for (; row < M; row += stride) {
    // get row offsets
    float c[CoarsenFactor] = {0};
    float buffer[CoarsenFactor];
    // access_t res = init_zeros<access_t>();

    int start = csr_indptr[row];
    int end = csr_indptr[row + 1];
    int k;
    float v;

    for (int jj = start + lane_id; jj < end; jj += tile_size) {
      k = csr_indices[jj];
      v = __guard_load_default_one<float>(csr_data, jj);

#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        if (i < valid_lane_num) {
          buffer[i] = B_panel[k * ldB + i];
        }
      }

#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        c[i] += v * buffer[i];
      }
    }

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
  }
}

template <typename Index, typename DType, int group_factor, int thread_per_block, int tile_factor, int block_numer,int block_denom>
void csrspmm_parreduce_rowbalance_cg(const SpMatCsrDescr_t<Index, DType>& spmatA, 
  const int N, const DType *B, DType *C) {
  // factor of thread coarsening
  int coarsen_factor = (N % 4 == 0) ? 4 : (N % 2 == 0) ? 2 : 1;
  // number of parallel warps along M-dimension
  float block_factor = (float)block_numer / (float)block_denom;
  int Mdim_worker = (float)spmatA.nrow * block_factor;
  // partition large-N and map to blockdim.y to help cache performance
  int tile_size = 1<<tile_factor;
  int Ndim_threadblock = CEIL(N, tile_size);
  int Ndim_warp_per_tb = min(N, tile_size) / coarsen_factor;

  int ref_warp_per_tb = thread_per_block / tile_size;
  int Mdim_warp_per_tb = CEIL(ref_warp_per_tb, Ndim_warp_per_tb);

  // total number of warps
  int gridDimX = CEIL(Mdim_worker, Mdim_warp_per_tb);
  int gridDimY = Ndim_threadblock;
  dim3 gridDim(gridDimX, gridDimY, 1);
  dim3 blockDim(Ndim_warp_per_tb * tile_size, Mdim_warp_per_tb, 1);

  if (coarsen_factor == 4) {
  csrspmm_parreduce_rowbalance_cg_kernel<float4, 1<<group_factor, 1<<tile_factor>
  <<<gridDim, blockDim>>>(spmatA.nrow, N, spmatA.ncol, spmatA.sp_csrptr.d_array.get(),
  spmatA.sp_csrind.d_array.get(), spmatA.sp_data.d_array.get(), B, C);
  } else if (coarsen_factor == 2) {
  csrspmm_parreduce_rowbalance_cg_kernel<float2, 1<<group_factor, 1<<tile_factor>
  <<<gridDim, blockDim>>>(spmatA.nrow, N, spmatA.ncol, spmatA.sp_csrptr.d_array.get(),
  spmatA.sp_csrind.d_array.get(), spmatA.sp_data.d_array.get(), B, C);
  } else {
  csrspmm_parreduce_rowbalance_cg_kernel<float, 1<<group_factor, 1<<tile_factor>
  <<<gridDim, blockDim>>>(spmatA.nrow, N, spmatA.ncol, spmatA.sp_csrptr.d_array.get(),
  spmatA.sp_csrind.d_array.get(), spmatA.sp_data.d_array.get(), B, C);
  }
  /*
  cudaDeviceSynchronize();
  std::cout<<"("<<gridDim.x<<","<<gridDim.y<<","<<gridDim.z<<")"<<std::endl;
  std::cout<<"("<<blockDim.x<<","<<blockDim.y<<","<<blockDim.z<<")"<<std::endl;  
  gpuErrchk(cudaGetLastError());
  */
}

template <int CoarsenFactor, int ThreadNz, int group_size>
__global__ void csrspmm_rowcaching_nnzbalance_cg_kernel(
    const int M, const int N, const int K, const int nnz_,
    const int csr_indptr[], const int csr_indices[], const float csr_data[],
    const float B[], float C[]) {
  int nnz = nnz_;
  if (nnz < 0)
    nnz = csr_indptr[M];
  thread_block_tile<group_size,thread_block> group = tiled_partition<group_size>(this_thread_block());
  int warp_id = group.meta_group_rank();
  int lane_id = group.thread_rank();

  extern __shared__ int shared_mem[];
  int *workspace_rowid = &shared_mem[(warp_id * group_size)];
  int *workspace_colid = workspace_rowid + blockDim.x;
  float *workspace_data =
      (float *)(workspace_colid +
                blockDim.x); // float and int has the same size

  // get the sparse-value range of this row
  int global_warp_id = blockIdx.x * (blockDim.x / group_size) + warp_id;
  int nz_start = global_warp_id * (ThreadNz * group_size);

  // get the dense column offset
  int col_offset = blockIdx.y * group_size * CoarsenFactor;
  const float *B_lanes[CoarsenFactor];
  float *C_lanes[CoarsenFactor];
#pragma unroll
  for (int i = 0; i < CoarsenFactor; i++) {
    B_lanes[i] = B + col_offset + lane_id + i * group_size;
    C_lanes[i] = C + col_offset + lane_id + i * group_size;
  }
  int ldB = N;

  // declare accumulators
  float c[CoarsenFactor] = {0.0f};
  int ldC = N;

  int stride = gridDim.x * blockDim.x * ThreadNz;

  if (blockIdx.y == gridDim.y - 1)
    goto Ndim_Residue;

  for (; nz_start < nnz; nz_start += stride) {
  // iterate over the segment of this warp
  for (int tile_base = nz_start; 
    tile_base < min(nz_start + ThreadNz * group_size, nnz); tile_base += group_size) {

    int thread_nz_id = tile_base + lane_id;
    if (thread_nz_id < nnz) {
      workspace_colid[lane_id] = csr_indices[thread_nz_id];
      workspace_data[lane_id] =
          __guard_load_default_one<float>(csr_data, thread_nz_id);
    } else {
      workspace_colid[lane_id] = 0;
      workspace_data[lane_id] = 0.0f;
    }
    workspace_rowid[lane_id] =
        binary_search_segment_number<int>(csr_indptr, M, nnz, thread_nz_id);
    group.sync();

    // initialize with first value
    int k = workspace_colid[0];
    float v = workspace_data[0];
#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      c[i] = v * B_lanes[i][k * ldB];
    }
    int row_curr = workspace_rowid[0], next_row;

// scan
#pragma unroll
    for (int pp = 1; pp < group_size; pp++) {
      next_row = workspace_rowid[pp];
      if (next_row != row_curr) {
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          atomicAdd(C_lanes[i] + row_curr * ldC, c[i]);
        }
        row_curr = next_row;
        k = workspace_colid[pp];
        v = workspace_data[pp];
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          c[i] = v * B_lanes[i][k * ldB];
        }
      } else {
        k = workspace_colid[pp];
        v = workspace_data[pp];
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          c[i] = c[i] + v * B_lanes[i][k * ldB];
        }
      }
    }
#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      atomicAdd(C_lanes[i] + row_curr * ldC, c[i]);
    }
  }
  }
  return;

Ndim_Residue:

  int valid_lane_num = CEIL(N - col_offset - lane_id, group_size);
  
  for (; nz_start < nnz; nz_start += stride) {
  // iterate over the segment of this warp
  for (int tile_base = nz_start; 
    tile_base < min(nz_start + ThreadNz * group_size, nnz); tile_base += group_size) {

    int thread_nz_id = tile_base + lane_id;
    if (thread_nz_id < nnz) {
      workspace_colid[lane_id] = csr_indices[thread_nz_id];
      workspace_data[lane_id] =
          __guard_load_default_one<float>(csr_data, thread_nz_id);
    } else {
      workspace_colid[lane_id] = 0;
      workspace_data[lane_id] = 0.0f;
    }
    workspace_rowid[lane_id] =
        binary_search_segment_number<int>(csr_indptr, M, nnz, thread_nz_id);
    group.sync();

    // initialize with first value
    int k = workspace_colid[0];
    float v = workspace_data[0];
#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      if (i < valid_lane_num) {
        c[i] = v * B_lanes[i][k * ldB];
      }
    }
    int row_curr = workspace_rowid[0], next_row;

// scan
#pragma unroll
    for (int pp = 1; pp < group_size; pp++) {
      next_row = workspace_rowid[pp];
      if (next_row != row_curr) {
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          if (i < valid_lane_num) {
            atomicAdd(C_lanes[i] + row_curr * ldC, c[i]);
          }
        }
        row_curr = next_row;
        k = workspace_colid[pp];
        v = workspace_data[pp];
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          if (i < valid_lane_num) {
            c[i] = v * B_lanes[i][k * ldB];
          }
        }
      } else {
        k = workspace_colid[pp];
        v = workspace_data[pp];
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          if (i < valid_lane_num) {
            c[i] = c[i] + v * B_lanes[i][k * ldB];
          }
        }
      }
    }
#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      if (i < valid_lane_num) {
        atomicAdd(C_lanes[i] + row_curr * ldC, c[i]);
      }
    }
  }
  }
}

template <typename Index, typename DType, int CoarsenFactor, int group_factor, int thread_per_block, int ThreadNz, int block_numer,int block_denom>
void csrspmm_rowcaching_nnzbalance_cg(const SpMatCsrDescr_t<Index, DType>& spmatA, 
  const int N, const DType *B, DType *C) {
int group_size = 1<<group_factor;
int coarsen_factor = min(CEIL(N, group_size), CoarsenFactor);
int Ndim_threadblock = CEIL(N, (group_size * coarsen_factor));


// int Nnzdim_threadblock = CEIL(spmatA.nnz, Nnzdim_warp_per_tb * 32 *
// thread_nz );
float block_factor = (float)block_numer / (float)block_denom;
int Nnzdim_threadblock = (float)spmatA.nrow * block_factor;

dim3 gridDim(Nnzdim_threadblock, Ndim_threadblock, 1);
dim3 blockDim(RefThreadPerBlock, 1, 1);

size_t smem_size = (2 * sizeof(int) + sizeof(float)) * thread_per_block;

// simple heuristic

if (coarsen_factor == 4) {
csrspmm_rowcaching_nnzbalance_cg_kernel<4, ThreadNz, 1<<group_factor >
<<<gridDim, blockDim, smem_size>>>(spmatA.nrow, N, spmatA.ncol,
            spmatA.nnz, spmatA.sp_csrptr.d_array.get(), spmatA.sp_csrind.d_array.get(),
            spmatA.sp_data.d_array.get(), B, C);
} else if (coarsen_factor == 2) {
csrspmm_rowcaching_nnzbalance_cg_kernel<2, ThreadNz, 1<<group_factor >
<<<gridDim, blockDim, smem_size>>>(spmatA.nrow, N, spmatA.ncol,
            spmatA.nnz, spmatA.sp_csrptr.d_array.get(), spmatA.sp_csrind.d_array.get(),
            spmatA.sp_data.d_array.get(), B, C);
} else {
csrspmm_rowcaching_nnzbalance_cg_kernel<1, ThreadNz, 1<<group_factor >
<<<gridDim, blockDim, smem_size>>>(spmatA.nrow, N, spmatA.ncol,
            spmatA.nnz, spmatA.sp_csrptr.d_array.get(), spmatA.sp_csrind.d_array.get(),
            spmatA.sp_data.d_array.get(), B, C);
  }
}

#endif