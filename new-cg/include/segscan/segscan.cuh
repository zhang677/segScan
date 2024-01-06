#include "../utils/check.cuh"
#include <cuda.h>
#include <cooperative_groups.h>

/// ge-spmm
template <typename ValueType, typename IndexType, typename AccessType, int group_size, int tile_size>
__global__ void segscan_kernel(const ValueType* src, const IndexType* index, const int nnz, const int N, ValueType* dst) {
    constexpr int CoarsenFactor = sizeof(AccessType) / sizeof(ValueType);
    
    int lane_id = (threadIdx.x & (tile_size - 1));
    int Nnzdim_warp_id = blockIdx.x * blockDim.y + threadIdx.y; 
    int nz_start = Nnzdim_warp_id * tile_size;
    int stride = gridDim.x * (blockDim.y * tile_size); 

    int col_offset = (blockIdx.y * tile_size) + (threadIdx.x / tile_size) * CoarsenFactor;
    const ValueType *src_panel = src + col_offset;
    ValueType *dst_panel = dst + col_offset;
    const int ldsrc = N;
    const int lddst = N;

    IndexType k;
    ValueType v;
    ValueType o[CoarsenFactor] = {0};
    thread_block_tile<group_size,thread_block> group = tiled_partition<group_size>(this_thread_block());

    if (col_offset >= N) return;    
    if (col_offset + CoarsenFactor >= N) goto Ndim_Residue;

    for (int nz_id = nz_start + lane_id; nz_id < nnz + lane_id; nz_id += stride) {
        IndexType row = index[nz_id];
        if (nz_id < nnz) {
            k = nz_id; // Feature is sorted
            v = (ValueType)1; // csr_data is set to 1
        } else {
            k = nnz - 1;
            v = (ValueType)0;
        }

        // load B-elements in vector-type
        *(AccessType *)o = *(AccessType *)(src_panel + k * ldsrc);
        #pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
            o[i] *= v;
        }

        int row_intv = group.shfl(row, group.size() - 1) - group.shfl(row, 0);
        if (row_intv == 0) {
            #pragma unroll
            for (int i = 0; i < CoarsenFactor; i++) {
                for (int k = group.size >> 1; k > 0; k >>= 1) {
                    o[i] += group.shfl_down(o[i], k);
                }
            } 
            if (group.thread_rank() == 0) {
                #pragma unroll
                for (int i = 0; i < CoarsenFactor; i++) {
                    atomicAdd(dst_panel + row * lddst + i, o[i]);
                }
            } 
        } else {
            bool is_seg_start = ((group.shfl_up(row,1) != row)|| (group.thread_rank() == 0));
            ValueType tmpv;
            IndexType tmpr;
            #pragma unroll
            for (int i = 0; i < CoarsenFactor; i++) {
                for (k = 1; k<group.size(); k = k<<1) {
                    tmpv = group.shfl_down(o[i],k);
                    tmpr = group.shfl_down(row,k);
                    if (tmpr == row && group.thread_rank() < (group.size()-k)) {
                        o[i] += tmpv;
                    }
                }
            }
            if (is_seg_start) {
                #pragma unroll
                for (int i = 0; i < CoarsenFactor; i++) {
                    atomicAdd(dst_panel + row * lddst + i, o[i]);
                }
            }
        }
     }
     return;

     Ndim_Residue:
     int valid_lane_num = N - col_offset;

     for (int nz_id = nz_start + lane_id; nz_id < nnz + lane_id; nz_id += stride) {
        IndexType row = index[nz_id];
        if (nz_id < nnz) {
            k = nz_id; // Feature is sorted
            v = (ValueType)1;
        } else {
            k = 0;
            v = (ValueType)0;
        }

        // load B-elements in vector-type
        #pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
            if (i < valid_lane_num) {
                o[i] = src_panel[k * ldsrc + i] * v;
            }
        }

        int row_intv = group.shfl(row, group.size() - 1) - group.shfl(row, 0);
        if (row_intv == 0) {
            #pragma unroll
            for (int i = 0; i < CoarsenFactor; i++) {
                for (int k = group.size >> 1; k > 0; k >>= 1) {
                    o[i] += group.shfl_down(o[i], k);
                }
            } 
            if (group.thread_rank() == 0) {
                #pragma unroll
                for (int i = 0; i < CoarsenFactor; i++) {
                    if (i < valid_lane_num) {
                        atomicAdd(dst_panel + row * lddst + i, o[i]);
                    }
                }
            } 
        } else {
            bool is_seg_start = ((group.shfl_up(row,1) != row)|| (group.thread_rank() == 0));
            ValueType tmpv;
            IndexType tmpr;
            #pragma unroll
            for (int i = 0; i < CoarsenFactor; i++) {
                for (k = 1; k<group.size(); k = k<<1) {
                    tmpv = group.shfl_down(o[i],k);
                    tmpr = group.shfl_down(row,k);
                    if (tmpr == row && group.thread_rank() < (group.size()-k)) {
                        o[i] += tmpv;
                    }
                }
            }
            if (is_seg_start) {
                #pragma unroll
                for (int i = 0; i < CoarsenFactor; i++) {
                    if (i < valid_lane_num) {
                        atomicAdd(dst_panel + row * lddst + i, o[i]);
                    }
                }
            }
        }
     }
     return;

}

template <typename ValueType, typename IndexType, int group_factor, int tile_factor, int block_numer,int block_denom>
void segment_coo(const ValueType* src, const IndexType* index, const int nnz, const int N, const int dst_len, ValueType* dst){
    int coarsen_factor = (N % 4 == 0) ? 4 : (N % 2 == 0) ? 2 : 1;
    float block_factor = (float)block_numer / (float)block_denom;
    int Nnzdim_worker = (float)dst_len * block_factor;
    int tile_size = 1<<tile_factor;
    int Ndim_threadblock = CEIL(N, tile_size); //
    int Ndim_warp_per_tb = min(N, tile_size) / coarsen_factor; // 32

    int ref_warp_per_tb = thread_per_block / tile_size; // 512/128 = 4 
    int Nnzdim_warp_per_tb = CEIL(ref_warp_per_tb, Ndim_warp_per_tb); // 1

    // total number of warps
    int gridDimX = CEIL(Nnzdim_worker, Nnzdim_warp_per_tb);
    int gridDimY = Ndim_threadblock;
    dim3 gridDim(gridDimX, gridDimY, 1);
    dim3 blockDim(Ndim_warp_per_tb * tile_size, Nnzdim_warp_per_tb, 1);

    if (coarsen_factor == 4) {
        segscan_kernel<ValueType, IndexType, float4, 1<<group_factor, 1<<tile_factor ><<<gridDim, blockDim>>>(
            src, index, nnz, N, dst);
    } else if (coarsen_factor == 2) {
        segscan_kernel<ValueType, IndexType, float2, 1<<group_factor, 1<<tile_factor ><<<gridDim, blockDim>>>(
            src, index, nnz, N, dst);
    } else {
        segscan_kernel<ValueType, IndexType, float, 1<<group_factor, 1<<tile_factor ><<<gridDim, blockDim>>>(
            src, index, nnz, N, dst);
    }

}