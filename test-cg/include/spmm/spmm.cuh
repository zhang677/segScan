#include "../dataloader/dataloader.hpp"
#include "../util/check.cuh"
#include "../util/ramArray.cuh"
#include <cuda.h>
#include <cusparse.h>
#include <iostream>

enum spmm_kernel_met {
  cusparse,
  eb_pr,
};

template <typename access_t>
__global__ void csrspmm_parreduce_nnzbalance_kernel(
    const int M, const int N, const int K, const int nnz_,
    const int csr_indptr[], const int csr_indices[], const float csr_data[],
    const float B[], float C[]) {
  constexpr int CoarsenFactor = sizeof(access_t) / sizeof(float);
  int nnz = nnz_;
  if (nnz < 0)
    nnz = csr_indptr[M];

  int lane_id = (threadIdx.x & (32 - 1));
  int Nnzdim_warp_id = blockIdx.x * blockDim.y + threadIdx.y;
  int nz_start = Nnzdim_warp_id * 32;
  int stride = gridDim.x * (blockDim.y * 32);

  // get the dense column offset
  int col_offset = blockIdx.y * 32 + (threadIdx.x >> 5) * CoarsenFactor;
  const float *B_panel = B + col_offset;
  float *C_panel = C + col_offset;
  int ldB = N;
  int ldC = N;

  int k;
  float v;
  float c[CoarsenFactor] = {0};
  float buffer[CoarsenFactor] = {0};

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
    int row_intv =
        __shfl_sync(FULLMASK, row, (32 - 1)) - __shfl_sync(FULLMASK, row, 0);
    if (row_intv == 0) {
// if all non-zeros in this warp belong to the same row, use a simple reduction
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        SHFL_DOWN_REDUCE(c[i]);
      }
      if (lane_id == 0) {
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          atomicAdd(C_panel + row * ldC + i, c[i]);
        }
      }
    } else {
      // if non-zeros belong to different rows, use a parallel-scan primitive
      // thread that holds the start of each segment are responsible for writing
      // results
      bool is_seg_start =
          ((__shfl_up_sync(FULLMASK, row, 1) != row) || (lane_id == 0));
      float tmpv;
      int tmpr;
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        SEG_SHFL_SCAN(c[i], tmpv, row, tmpr);
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
    int row_intv =
        __shfl_sync(FULLMASK, row, (32 - 1)) - __shfl_sync(FULLMASK, row, 0);
    if (row_intv == 0) {
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        SHFL_DOWN_REDUCE(c[i]);
      }
      if (lane_id == 0) {
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          if (i < valid_lane_num) {
            atomicAdd(C_panel + row * ldC + i, c[i]);
          }
        }
      }
    } else {
      bool is_seg_start =
          ((__shfl_up_sync(FULLMASK, row, 1) != row) || (lane_id == 0));
      float tmpv;
      int tmpr;
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        SEG_SHFL_SCAN(c[i], tmpv, row, tmpr);
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

template <typename Index, typename DType>
void csrspmm_parreduce_nnzbalance(SpMatCsrDescr_t<Index, DType>& spmatA, 
  const int N, const DType *B, DType *C) {

  // factor of thread coarsening
  int coarsen_factor = (N % 4 == 0) ? 4 : (N % 2 == 0) ? 2 : 1;
  // number of parallel warps along M-dimension
  const int segreduce_size_per_warp = 32;
  int Nnzdim_worker = spmatA.nrow; // CEIL(spmatA.nnz, segreduce_size_per_warp);
  // partition large-N and map to blockdim.y to help cache performance
  int Ndim_threadblock = CEIL(N, 32);
  int Ndim_warp_per_tb = min(N, 32) / coarsen_factor;

  int ref_warp_per_tb = RefThreadPerBlock / 32;
  int Nnzdim_warp_per_tb = CEIL(ref_warp_per_tb, Ndim_warp_per_tb);

  // total number of warps
  int gridDimX = CEIL(Nnzdim_worker, Nnzdim_warp_per_tb);
  int gridDimY = Ndim_threadblock;
  dim3 gridDim(gridDimX, gridDimY, 1);
  dim3 blockDim(Ndim_warp_per_tb * 32, Nnzdim_warp_per_tb, 1);

  if (coarsen_factor == 4) {
  csrspmm_parreduce_nnzbalance_kernel<float4><<<gridDim, blockDim>>>(
  spmatA.nrow, N, spmatA.ncol, spmatA.nnz, spmatA.sp_csrptr.d_array.get(), spmatA.sp_csrind.d_array.get(),
  spmatA.sp_data.d_array.get(), B, C);
  } else if (coarsen_factor == 2) {
  csrspmm_parreduce_nnzbalance_kernel<float2><<<gridDim, blockDim>>>(
  spmatA.nrow, N, spmatA.ncol, spmatA.nnz, spmatA.sp_csrptr.d_array.get(), spmatA.sp_csrind.d_array.get(),
  spmatA.sp_data.d_array.get(), B, C);
  } else {
  csrspmm_parreduce_nnzbalance_kernel<float><<<gridDim, blockDim>>>(
  spmatA.nrow, N, spmatA.ncol, spmatA.nnz, spmatA.sp_csrptr.d_array.get(), spmatA.sp_csrind.d_array.get(),
  spmatA.sp_data.d_array.get(), B, C);
  }
}

template <typename Index, typename DType>
void csrspmm_cusparse(SpMatCsrDescr_t<Index, DType> &spmatA,
                            const int feature_size, DType *in_feature,
                            DType *out_feature) {
  //
  // Run Cusparse-SpMM and check result
  //
  cusparseHandle_t handle;
  cusparseSpMatDescr_t csrDescr;
  cusparseDnMatDescr_t dnMatInputDescr, dnMatOutputDescr;
  float alpha = 1.0f, beta = 0.0f;

  checkCuSparseError(cusparseCreate(&handle));

  // creating sparse csr matrix
  checkCuSparseError(cusparseCreateCsr(
      &csrDescr, spmatA.nrow, spmatA.ncol, spmatA.nnz,
      spmatA.sp_csrptr.d_array.get(), spmatA.sp_csrind.d_array.get(),
      spmatA.sp_data.d_array.get(),
      CUSPARSE_INDEX_32I, // index 32-integer for indptr
      CUSPARSE_INDEX_32I, // index 32-integer for indices
      CUSPARSE_INDEX_BASE_ZERO,
      CUDA_R_32F // datatype: 32-bit float real number
      ));

  // creating dense matrices
  checkCuSparseError(cusparseCreateDnMat(&dnMatInputDescr, spmatA.ncol,
                                         feature_size, feature_size, in_feature,
                                         CUDA_R_32F, CUSPARSE_ORDER_ROW));
  checkCuSparseError(cusparseCreateDnMat(
      &dnMatOutputDescr, spmatA.nrow, feature_size, feature_size, out_feature,
      CUDA_R_32F, CUSPARSE_ORDER_ROW));

  // allocate workspace buffer
  size_t workspace_size;
  checkCuSparseError(cusparseSpMM_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, csrDescr, dnMatInputDescr,
      &beta, dnMatOutputDescr, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
      &workspace_size));

  void *workspace = NULL;
  checkCudaError(cudaMalloc(&workspace, workspace_size));

  // run SpMM
  checkCuSparseError(cusparseSpMM(handle,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE, // opA
                                  CUSPARSE_OPERATION_NON_TRANSPOSE, // opB
                                  &alpha, csrDescr, dnMatInputDescr, &beta,
                                  dnMatOutputDescr, CUDA_R_32F,
                                  CUSPARSE_SPMM_ALG_DEFAULT, workspace));
}

template <class Index, class DType, spmm_kernel_met km>
void SpMM_check(SpMatCsrDescr_t<Index, DType>& H, const int feature_size,
util::RamArray<DType> &in_feature, util::RamArray<DType> &out_feature, util::RamArray<DType> &out_ref) {
  out_feature.reset();
  if (km == spmm_kernel_met::cusparse) {
    std::cout<<"cusparse: ";
    csrspmm_cusparse<Index, DType>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
  } else if (km == spmm_kernel_met::eb_pr) {
    std::cout<<"eb_pr: ";
    csrspmm_parreduce_nnzbalance<Index, DType>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
  } else {
    std::cout<<"Not implemented yet!"<<std::endl;
  }
  out_feature.download();
  bool pass = util::check_result(H.nrow, feature_size, out_feature.h_array.get(), out_ref.h_array.get());
  if (pass) {
    std::cout<<"Passed!"<<std::endl;
  } else {
    std::cout<<"Not Passed!"<<std::endl;
}
}