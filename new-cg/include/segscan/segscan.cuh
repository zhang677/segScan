#include "../utils/check.cuh"
#include <cuda.h>
#include <cooperative_groups.h>
using namespace cooperative_groups;

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
                for (int k = group.size() >> 1; k > 0; k >>= 1) {
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
                for (int k = group.size() >> 1; k > 0; k >>= 1) {
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

template <typename ValueType, typename IndexType, int group_factor, int thread_per_block, int tile_factor, int block_numer,int block_denom>
void segment_coo(const ValueType* src, const IndexType* index, const int nnz, const int N, ValueType* dst){
    int coarsen_factor = (N % 4 == 0) ? 4 : (N % 2 == 0) ? 2 : 1;
    float block_factor = (float)block_numer / (float)block_denom;
    int Nnzdim_worker = (float)nnz * block_factor;
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

    cudaDeviceSynchronize();
    std::cout<<"("<<gridDim.x<<","<<gridDim.y<<","<<gridDim.z<<")"<<std::endl;
    std::cout<<"("<<blockDim.x<<","<<blockDim.y<<","<<blockDim.z<<")"<<std::endl;  
    gpuErrchk(cudaGetLastError());

}

/// Update workload distribution
template <typename ValueType, typename IndexType, typename AccessType, int group_size>
__global__ void segscan_new_kernel(const ValueType* src, const IndexType* index, const int nnz, const int N, ValueType* dst) {
    constexpr int CoarsenFactor = sizeof(AccessType) / sizeof(ValueType);
    int lane_id = threadIdx.x;
    int nz_start = blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    int col_offset = (blockIdx.y * blockDim.y + threadIdx.y) * CoarsenFactor;
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
                for (int k = group.size() >> 1; k > 0; k >>= 1) {
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
                for (int k = group.size() >> 1; k > 0; k >>= 1) {
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

template <typename ValueType, typename IndexType, int group_factor, int thread_per_block, int tile_factor, int blockDimY>
void segment_coo_new(const ValueType* src, const IndexType* index, const int nnz, const int N, ValueType* dst){
    int coarsen_factor = (N % 4 == 0) ? 4 : (N % 2 == 0) ? 2 : 1;
    // group_size <= tile_size
    int tile_size = 1<<tile_factor;
    int max_blockDimY = thread_per_block / tile_size;
    int numTileY = CEIL(N, coarsen_factor);
    int real_blockDimY = blockDimY < max_blockDimY ? MIN(numTileY, blockDimY) : MIN(numTileY, max_blockDimY);
    int real_blockDimX = CEIL(thread_per_block, real_blockDimY);
    dim3 gridDim(CEIL(nnz, real_blockDimX), CEIL(numTileY, real_blockDimY), 1);
    dim3 blockDim(real_blockDimX, real_blockDimY, 1);

    if (coarsen_factor == 4) {
        segscan_new_kernel<ValueType, IndexType, float4, 1<<group_factor ><<<gridDim, blockDim>>>(
            src, index, nnz, N, dst);
    } else if (coarsen_factor == 2) {
        segscan_new_kernel<ValueType, IndexType, float2, 1<<group_factor ><<<gridDim, blockDim>>>(
            src, index, nnz, N, dst);
    } else {
        segscan_new_kernel<ValueType, IndexType, float, 1<<group_factor ><<<gridDim, blockDim>>>(
            src, index, nnz, N, dst);
    }

    cudaDeviceSynchronize();
    std::cout<<"("<<gridDim.x<<","<<gridDim.y<<","<<gridDim.z<<")"<<std::endl;
    std::cout<<"("<<blockDim.x<<","<<blockDim.y<<","<<blockDim.z<<")"<<std::endl;  
    gpuErrchk(cudaGetLastError());

}

template <typename ValueType, typename IndexType, int CoarsenFactor, int ThreadNz, int group_size>
__global__ void segscan_sr_kernel(const ValueType* src, const IndexType* index, const int nnz, const int N, ValueType* dst) {
    thread_block_tile<group_size,thread_block> group = tiled_partition<group_size>(this_thread_block());
    int group_id = group.meta_group_rank();
    int lane_id = group.thread_rank();

    extern __shared__ uint8_t shared_mem[];
    IndexType *workspace_rowid = (IndexType*)(shared_mem + group_id * group_size * sizeof(IndexType));// &shared_mem[(group_id * group_size)];
    IndexType *workspace_colid = (IndexType*)((uint8_t*)workspace_rowid + blockDim.x * sizeof(IndexType)); // workspace_rowid + blockDim.x;
    ValueType *workspace_data = (ValueType*)((uint8_t*)workspace_colid + blockDim.x * sizeof(ValueType));// (ValueType *)(workspace_colid + blockDim.x); // float and int has the same size

    // get the sparse-value range of this row
    int global_group_id = blockIdx.x * (blockDim.x / group_size) + group_id;
    int nz_start = global_group_id * (ThreadNz * group_size);

    // get the dense column offset
    int col_offset = blockIdx.y * group_size * CoarsenFactor;
    const ValueType *src_lanes[CoarsenFactor];
    ValueType *dst_lanes[CoarsenFactor];
    #pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
        src_lanes[i] = src + col_offset + lane_id + i * group_size;
        dst_lanes[i] = dst + col_offset + lane_id + i * group_size;
    }
    int ldsrc = N;
    int lddst = N;

    ValueType o[CoarsenFactor] = {0};
    int stride = gridDim.x * blockDim.x * ThreadNz;

    if (blockIdx.y == gridDim.y - 1)
        goto Ndim_Residue;

    for (; nz_start < nnz; nz_start += stride) {
        // iterate over the segment of this warp
        for (int tile_base = nz_start; tile_base < min(nz_start + ThreadNz * group_size, nnz); tile_base += group_size) {

            int thread_nz_id = tile_base + lane_id;
            if (thread_nz_id < nnz) {
                workspace_colid[lane_id] = thread_nz_id;
                workspace_data[lane_id] = (ValueType)1;
            } else {
                workspace_colid[lane_id] = 0;
                workspace_data[lane_id] = (ValueType)0;
            }
            workspace_rowid[lane_id] = index[thread_nz_id];
            group.sync();

            // initialize with first value
            IndexType k = workspace_colid[0];
            ValueType v = workspace_data[0];
            #pragma unroll
            for (int i = 0; i < CoarsenFactor; i++) {
                o[i] = src_lanes[i][k * ldsrc] * v;
            }
            IndexType row_curr = workspace_rowid[0], next_row;

            #pragma unroll
            for (int pp = 1; pp < group_size; pp++) {
                next_row = workspace_rowid[pp];
                if (next_row != row_curr) {
                    #pragma unroll
                    for (int i = 0; i < CoarsenFactor; i++) {
                        atomicAdd(dst_lanes[i] + row_curr * lddst, o[i]);
                    }
                    row_curr = next_row;
                    k = workspace_colid[pp];
                    v = workspace_data[pp];
                    #pragma unroll
                    for (int i = 0; i < CoarsenFactor; i++) {
                        o[i] = v * src_lanes[i][k * ldsrc];
                    }
                } else {
                    k = workspace_colid[pp];
                    v = workspace_data[pp];
                    #pragma unroll
                    for (int i = 0; i < CoarsenFactor; i++) {
                        o[i] += v * src_lanes[i][k * ldsrc];
                    }
                }
            }
            #pragma unroll
            for (int i = 0; i < CoarsenFactor; i++) {
                atomicAdd(dst_lanes[i] + row_curr * lddst, o[i]);
            }
        }
    }
    return;

    Ndim_Residue:
    int valid_lane_num = CEIL(N - col_offset - lane_id, group_size);

    for (; nz_start < nnz; nz_start += stride) {
        for (int tile_base = nz_start; tile_base < min(nz_start + ThreadNz * group_size, nnz); tile_base += group_size) {
            int thread_nz_id = tile_base + lane_id;
            if (thread_nz_id < nnz) {
                workspace_colid[lane_id] = thread_nz_id;
                workspace_data[lane_id] = (ValueType)1;
            } else {
                workspace_colid[lane_id] = 0;
                workspace_data[lane_id] = (ValueType)0;
            }
            workspace_rowid[lane_id] = index[thread_nz_id];
            group.sync();

            // initialize with first value
            IndexType k = workspace_colid[0];
            ValueType v = workspace_data[0];
            #pragma unroll
            for (int i = 0; i < CoarsenFactor; i++) {
                if (i < valid_lane_num) {
                    o[i] = src_lanes[i][k * ldsrc] * v;
                }
            }
            IndexType row_curr = workspace_rowid[0], next_row;

            #pragma unroll
            for (int pp = 1; pp < group_size; pp++) {
                next_row = workspace_rowid[pp];
                if (next_row != row_curr) {
                    #pragma unroll
                    for (int i = 0; i < CoarsenFactor; i++) {
                        if (i < valid_lane_num) {
                            atomicAdd(dst_lanes[i] + row_curr * lddst, o[i]);
                        }
                    }
                    row_curr = next_row;
                    k = workspace_colid[pp];
                    v = workspace_data[pp];
                    #pragma unroll
                    for (int i = 0; i < CoarsenFactor; i++) {
                        if (i < valid_lane_num) {
                            o[i] = v * src_lanes[i][k * ldsrc];
                        }
                    }
                } else {
                    k = workspace_colid[pp];
                    v = workspace_data[pp];
                    #pragma unroll
                    for (int i = 0; i < CoarsenFactor; i++) {
                        if (i < valid_lane_num) {
                            o[i] += v * src_lanes[i][k * ldsrc];
                        }
                    }
                }
            }
            #pragma unroll
            for (int i = 0; i < CoarsenFactor; i++) {
                if (i < valid_lane_num) {
                    atomicAdd(dst_lanes[i] + row_curr * lddst, o[i]);
                }
            }
        }
    }
}

template <typename ValueType, typename IndexType, int CoarsenFactor, int ThreadNz, int group_size>
__global__ void segscan_sr_noshmem_kernel(const ValueType* src, const IndexType* index, const int nnz, const int N, ValueType* dst) {
    thread_block_tile<group_size,thread_block> group = tiled_partition<group_size>(this_thread_block());
    int group_id = group.meta_group_rank();
    int lane_id = group.thread_rank();

    // get the sparse-value range of this row
    int global_group_id = blockIdx.x * (blockDim.x / group_size) + group_id;
    int nz_start = global_group_id * (ThreadNz * group_size);

    IndexType rowids[group_size];
    IndexType colids[group_size];
    ValueType data[group_size];

    // get the dense column offset
    int col_offset = blockIdx.y * group_size * CoarsenFactor;
    const ValueType *src_lanes[CoarsenFactor];
    ValueType *dst_lanes[CoarsenFactor];

    int ldsrc = N;
    int lddst = N;

    ValueType o[CoarsenFactor] = {0};
    int stride = gridDim.x * blockDim.x * ThreadNz;

    // if (blockIdx.y == gridDim.y - 1)
    //     goto Ndim_Residue;
    int valid_lane_num = min(CEIL(N - col_offset - lane_id, group_size), CoarsenFactor);
    if (valid_lane_num == 0) return;

    #pragma unroll
    for (int i = 0; i < valid_lane_num; i++) {
        src_lanes[i] = src + col_offset + lane_id + i * group_size;
        dst_lanes[i] = dst + col_offset + lane_id + i * group_size;
    }

    int thread_nz_id;
    IndexType k, curr_row, next_row;
    ValueType v;
    for (; nz_start < nnz; nz_start += stride) {
        for (int tile_base = nz_start; tile_base < min(nz_start + ThreadNz * group_size, nnz); tile_base += group_size) {
            for (int g = 0; g < group_size; g++) {
                thread_nz_id = tile_base + g;
                if (thread_nz_id < nnz) {
                    rowids[g] = index[thread_nz_id];
                    colids[g] = thread_nz_id;
                    data[g] = (ValueType)1;
                } else {
                    rowids[g] = nnz - thread_nz_id - 1;
                    colids[g] = 0;
                    data[g] = (ValueType)0;
                }
            }
            curr_row = rowids[0];
            k = colids[0];
            v = data[0];
            // initialize with first value
            #pragma unroll
            for (int i = 0; i < valid_lane_num; i++) {
                o[i] = src_lanes[i][k * ldsrc] * v;
            }
            
            #pragma unroll
            for (int pp = 1; pp < group_size; pp++) {
                next_row = rowids[pp];
                if (next_row < 0) {
                    break;
                }
                if (next_row != curr_row) {
                    #pragma unroll
                    for (int i = 0; i < valid_lane_num; i++) {
                        atomicAdd(dst_lanes[i] + curr_row * lddst, o[i]);
                    }
                    curr_row = next_row;
                    k = colids[pp];
                    v = data[pp];
                    #pragma unroll
                    for (int i = 0; i < valid_lane_num; i++) {
                        o[i] = v * src_lanes[i][k * ldsrc];
                    }
                } else {
                    k = colids[pp];
                    v = data[pp];
                    #pragma unroll
                    for (int i = 0; i < valid_lane_num; i++) {
                        o[i] += v * src_lanes[i][k * ldsrc];
                    }
                }
            }
            #pragma unroll
            for (int i = 0; i < valid_lane_num; i++) {
                atomicAdd(dst_lanes[i] + curr_row * lddst, o[i]);
            }
        }
    }
    return;   
}

template <typename ValueType, typename IndexType, int CoarsenFactor, int ThreadNz, int group_size>
__global__ void segscan_sr_noshmem_lessatom_kernel(const ValueType* src, const IndexType* index, const int nnz, const int N, ValueType* dst) {
    thread_block_tile<group_size,thread_block> group = tiled_partition<group_size>(this_thread_block());
    int group_id = group.meta_group_rank();
    int lane_id = group.thread_rank();

    // get the sparse-value range of this row
    int global_group_id = blockIdx.x * (blockDim.x / group_size) + group_id;
    int nz_start = global_group_id * (ThreadNz * group_size);

    IndexType rowids[group_size];
    IndexType colids[group_size];
    ValueType data[group_size];

    // get the dense column offset
    int col_offset = blockIdx.y * group_size * CoarsenFactor;
    const ValueType *src_lanes[CoarsenFactor];
    ValueType *dst_lanes[CoarsenFactor];

    int ldsrc = N;
    int lddst = N;

    ValueType o[CoarsenFactor] = {0};
    int stride = gridDim.x * blockDim.x * ThreadNz;

    // if (blockIdx.y == gridDim.y - 1)
    //     goto Ndim_Residue;
    int valid_lane_num = min(CEIL(N - col_offset - lane_id, group_size), CoarsenFactor);
    if (valid_lane_num == 0) return;

    #pragma unroll
    for (int i = 0; i < valid_lane_num; i++) {
        src_lanes[i] = src + col_offset + lane_id + i * group_size;
        dst_lanes[i] = dst + col_offset + lane_id + i * group_size;
    }

    int thread_nz_id;
    IndexType k, curr_row, next_row, start_row, end_row;
    ValueType v;
    for (; nz_start < nnz; nz_start += stride) {
        for (int tile_base = nz_start; tile_base < min(nz_start + ThreadNz * group_size, nnz); tile_base += group_size) {
            for (int g = 0; g < group_size; g++) {
                thread_nz_id = tile_base + g;
                if (thread_nz_id < nnz) {
                    rowids[g] = index[thread_nz_id];
                    colids[g] = thread_nz_id;
                    data[g] = (ValueType)1;
                } else {
                    rowids[g] = nnz - thread_nz_id - 1;
                    colids[g] = 0;
                    data[g] = (ValueType)0;
                }
            }
            start_row = rowids[0];
            end_row = rowids[group_size - 1];
            curr_row = rowids[0];
            k = colids[0];
            v = data[0];
            // initialize with first value
            #pragma unroll
            for (int i = 0; i < valid_lane_num; i++) {
                o[i] = src_lanes[i][k * ldsrc] * v;
            }
            
            #pragma unroll
            for (int pp = 1; pp < group_size; pp++) {
                next_row = rowids[pp];
                if (next_row < 0) {
                    break;
                }
                if (next_row != curr_row) {
                    if (curr_row == start_row || curr_row == end_row) {
                        #pragma unroll
                        for (int i = 0; i < valid_lane_num; i++) {
                            atomicAdd(dst_lanes[i] + curr_row * lddst, o[i]);
                        }
                    } else {
                        #pragma unroll
                        for (int i = 0; i < valid_lane_num; i++) {
                            dst_lanes[i][curr_row * lddst] += o[i];
                        }
                    }
                    curr_row = next_row;
                    k = colids[pp];
                    v = data[pp];
                    #pragma unroll
                    for (int i = 0; i < valid_lane_num; i++) {
                        o[i] = v * src_lanes[i][k * ldsrc];
                    }
                } else {
                    k = colids[pp];
                    v = data[pp];
                    #pragma unroll
                    for (int i = 0; i < valid_lane_num; i++) {
                        o[i] += v * src_lanes[i][k * ldsrc];
                    }
                }
            }
            #pragma unroll
            for (int i = 0; i < valid_lane_num; i++) {
                atomicAdd(dst_lanes[i] + curr_row * lddst, o[i]);
            }
        }
    }
    return;   
}

template<typename ValueType, typename IndexType, int CoarsenFactor, int ThreadNz, int DtileSize>
__global__ void segscan_sr_sorted_kernel(const ValueType* src, const IndexType* index, const int nnz, const int N, ValueType* dst) {
    int Dtile_id = threadIdx.x / DtileSize;
    int lane_id = threadIdx.x % DtileSize;

    int nz_start = (blockIdx.y * blockDim.y + threadIdx.y) * ThreadNz;
    
    IndexType rowids[ThreadNz];
    IndexType colids[ThreadNz];
    ValueType data[ThreadNz];

    int col_offset = blockIdx.x * blockDim.x * CoarsenFactor + Dtile_id * DtileSize * CoarsenFactor + lane_id;
    const ValueType *src_lanes[CoarsenFactor];
    ValueType *dst_lanes[CoarsenFactor];

    int ldsrc = N;
    int lddst = N;

    ValueType o[CoarsenFactor] = {0};
    int stride = gridDim.y * blockDim.y * ThreadNz;

    int valid_lane_num = min(CEIL(N - col_offset, DtileSize), CoarsenFactor);
    if (valid_lane_num == 0) return;

    #pragma unroll
    for (int i = 0; i < valid_lane_num; i++) {
        src_lanes[i] = src + col_offset + i * DtileSize;
        dst_lanes[i] = dst + col_offset + i * DtileSize;
    }

    int thread_nz_id;
    IndexType k, curr_row, next_row, start_row, end_row;
    ValueType v;
    for(; nz_start < nnz; nz_start += stride) {
        for (int g = 0; g < ThreadNz; g++) {
            thread_nz_id = nz_start + g;
            if (thread_nz_id < nnz) {
                rowids[g] = index[thread_nz_id];
                colids[g] = thread_nz_id;
                data[g] = (ValueType)1;
            } else {
                rowids[g] = nnz - thread_nz_id - 1;
                colids[g] = 0;
                data[g] = (ValueType)0;
            }
        }
        start_row = rowids[0];
        end_row = rowids[ThreadNz - 1];
        curr_row = rowids[0];
        k = colids[0];
        v = data[0];
        // initialize with first value
        #pragma unroll
        for (int i = 0; i < valid_lane_num; i++) {
            o[i] = src_lanes[i][k * ldsrc] * v;
        }
        
        #pragma unroll
        for (int pp = 1; pp < ThreadNz; pp++) {
            next_row = rowids[pp];
            if (next_row < 0) {
                break;
            }
            if (next_row != curr_row) {
                if (curr_row == start_row || curr_row == end_row) {
                    #pragma unroll
                    for (int i = 0; i < valid_lane_num; i++) {
                        atomicAdd(dst_lanes[i] + curr_row * lddst, o[i]);
                    }
                } else {
                    #pragma unroll
                    for (int i = 0; i < valid_lane_num; i++) {
                        dst_lanes[i][curr_row * lddst] += o[i];
                    }
                }
                curr_row = next_row;
                k = colids[pp];
                v = data[pp];
                #pragma unroll
                for (int i = 0; i < valid_lane_num; i++) {
                    o[i] = v * src_lanes[i][k * ldsrc];
                }
            } else {
                k = colids[pp];
                v = data[pp];
                #pragma unroll
                for (int i = 0; i < valid_lane_num; i++) {
                    o[i] += v * src_lanes[i][k * ldsrc];
                }
            }
        }
        #pragma unroll
        for (int i = 0; i < valid_lane_num; i++) {
            atomicAdd(dst_lanes[i] + curr_row * lddst, o[i]);
        }
    }
    return;
}

template <typename ValueType, typename IndexType, int group_factor, int thread_per_block, int CoarsenFactor, int ThreadNz, int block_numer,int block_denom>
void segment_coo_sr(const ValueType* src, const IndexType* index, const int nnz, const int N, ValueType* dst){
    int group_size = 1<<group_factor;
    int coarsen_factor = min(CEIL(N, group_size), CoarsenFactor);
    int Ndim_threadblock = CEIL(N, (group_size * coarsen_factor));

    float block_factor = (float)block_numer / (float)block_denom;
    int Nnzdim_threadblock = (float)nnz * block_factor;

    dim3 gridDim(Nnzdim_threadblock, Ndim_threadblock, 1);
    dim3 blockDim(thread_per_block, 1, 1);

    size_t smem_size = (2 * sizeof(int) + sizeof(float)) * thread_per_block;
    switch (coarsen_factor) {
        case 8:
            segscan_sr_kernel<ValueType, IndexType, 8, ThreadNz, 1 << group_factor ><<<gridDim, blockDim, smem_size>>>(
                src, index, nnz, N, dst);
            break;
        case 7:
            segscan_sr_kernel<ValueType, IndexType, 7, ThreadNz, 1 << group_factor ><<<gridDim, blockDim, smem_size>>>(
                src, index, nnz, N, dst);
            break;
        case 6:
            segscan_sr_kernel<ValueType, IndexType, 6, ThreadNz, 1 << group_factor ><<<gridDim, blockDim, smem_size>>>(
                src, index, nnz, N, dst);
            break;
        case 5:
            segscan_sr_kernel<ValueType, IndexType, 5, ThreadNz, 1 << group_factor ><<<gridDim, blockDim, smem_size>>>(
                src, index, nnz, N, dst);
            break;
        case 4:
            segscan_sr_kernel<ValueType, IndexType, 4, ThreadNz, 1 << group_factor ><<<gridDim, blockDim, smem_size>>>(
                src, index, nnz, N, dst);
            break;
        case 3:
            segscan_sr_kernel<ValueType, IndexType, 3, ThreadNz, 1 << group_factor ><<<gridDim, blockDim, smem_size>>>(
                src, index, nnz, N, dst);
            break;
        case 2:
            segscan_sr_kernel<ValueType, IndexType, 2, ThreadNz, 1 << group_factor ><<<gridDim, blockDim, smem_size>>>(
                src, index, nnz, N, dst);
            break;
        case 1:
            segscan_sr_kernel<ValueType, IndexType, 1, ThreadNz, 1 << group_factor ><<<gridDim, blockDim, smem_size>>>(
                src, index, nnz, N, dst);
            break;
        default:
            std::cout<<"CoarsenFactor = "<<coarsen_factor << " is not supported."<<std::endl;   
    }
}

template <typename ValueType, typename IndexType, int group_factor, int thread_per_block, int CoarsenFactor, int ThreadNz, int block_numer,int block_denom>
void segment_coo_noshmem_sr(const ValueType* src, const IndexType* index, const int nnz, const int N, ValueType* dst) {
    int group_size = 1<<group_factor;
    int coarsen_factor = min(CEIL(N, group_size), CoarsenFactor);
    int Ndim_threadblock = CEIL(N, (group_size * coarsen_factor));

    float block_factor = (float)block_numer / (float)block_denom;
    int Nnzdim_threadblock = (float)nnz * block_factor;

    dim3 gridDim(Nnzdim_threadblock, Ndim_threadblock, 1);
    dim3 blockDim(thread_per_block, 1, 1);

    switch (coarsen_factor) {
        case 8:
            segscan_sr_noshmem_kernel<ValueType, IndexType, 8, ThreadNz, 1 << group_factor ><<<gridDim, blockDim>>>(
                src, index, nnz, N, dst);
            break;
        case 7:
            segscan_sr_noshmem_kernel<ValueType, IndexType, 7, ThreadNz, 1 << group_factor ><<<gridDim, blockDim>>>(
                src, index, nnz, N, dst);
            break;
        case 6:
            segscan_sr_noshmem_kernel<ValueType, IndexType, 6, ThreadNz, 1 << group_factor ><<<gridDim, blockDim>>>(
                src, index, nnz, N, dst);
            break;
        case 5:
            segscan_sr_noshmem_kernel<ValueType, IndexType, 5, ThreadNz, 1 << group_factor ><<<gridDim, blockDim>>>(
                src, index, nnz, N, dst);
            break;
        case 4:
            segscan_sr_noshmem_kernel<ValueType, IndexType, 4, ThreadNz, 1 << group_factor ><<<gridDim, blockDim>>>(
                src, index, nnz, N, dst);
            break;
        case 3:
            segscan_sr_noshmem_kernel<ValueType, IndexType, 3, ThreadNz, 1 << group_factor ><<<gridDim, blockDim>>>(
                src, index, nnz, N, dst);
            break;
        case 2:
            segscan_sr_noshmem_kernel<ValueType, IndexType, 2, ThreadNz, 1 << group_factor ><<<gridDim, blockDim>>>(
                src, index, nnz, N, dst);
            break;
        case 1:
            segscan_sr_noshmem_kernel<ValueType, IndexType, 1, ThreadNz, 1 << group_factor ><<<gridDim, blockDim>>>(
                src, index, nnz, N, dst);
            break;
        default:
            std::cout<<"CoarsenFactor = "<<coarsen_factor << " is not supported."<<std::endl;   
    }
}

template <typename ValueType, typename IndexType, int group_factor, int thread_per_block, int CoarsenFactor, int ThreadNz, int block_numer,int block_denom>
void segment_coo_noshmem_lessatom_sr(const ValueType* src, const IndexType* index, const int nnz, const int N, ValueType* dst) {
    int group_size = 1<<group_factor;
    int coarsen_factor = min(CEIL(N, group_size), CoarsenFactor);
    int Ndim_threadblock = CEIL(N, (group_size * coarsen_factor));

    float block_factor = (float)block_numer / (float)block_denom;
    int Nnzdim_threadblock = (float)nnz * block_factor;

    dim3 gridDim(Nnzdim_threadblock, Ndim_threadblock, 1);
    dim3 blockDim(thread_per_block, 1, 1);

    switch (coarsen_factor) {
        case 8:
            segscan_sr_noshmem_lessatom_kernel<ValueType, IndexType, 8, ThreadNz, 1 << group_factor ><<<gridDim, blockDim>>>(
                src, index, nnz, N, dst);
            break;
        case 7:
            segscan_sr_noshmem_lessatom_kernel<ValueType, IndexType, 7, ThreadNz, 1 << group_factor ><<<gridDim, blockDim>>>(
                src, index, nnz, N, dst);
            break;
        case 6:
            segscan_sr_noshmem_lessatom_kernel<ValueType, IndexType, 6, ThreadNz, 1 << group_factor ><<<gridDim, blockDim>>>(
                src, index, nnz, N, dst);
            break;
        case 5:
            segscan_sr_noshmem_lessatom_kernel<ValueType, IndexType, 5, ThreadNz, 1 << group_factor ><<<gridDim, blockDim>>>(
                src, index, nnz, N, dst);
            break;
        case 4:
            segscan_sr_noshmem_lessatom_kernel<ValueType, IndexType, 4, ThreadNz, 1 << group_factor ><<<gridDim, blockDim>>>(
                src, index, nnz, N, dst);
            break;
        case 3:
            segscan_sr_noshmem_lessatom_kernel<ValueType, IndexType, 3, ThreadNz, 1 << group_factor ><<<gridDim, blockDim>>>(
                src, index, nnz, N, dst);
            break;
        case 2:
            segscan_sr_noshmem_lessatom_kernel<ValueType, IndexType, 2, ThreadNz, 1 << group_factor ><<<gridDim, blockDim>>>(
                src, index, nnz, N, dst);
            break;
        case 1:
            segscan_sr_noshmem_lessatom_kernel<ValueType, IndexType, 1, ThreadNz, 1 << group_factor ><<<gridDim, blockDim>>>(
                src, index, nnz, N, dst);
            break;
        default:
            std::cout<<"CoarsenFactor = "<<coarsen_factor << " is not supported."<<std::endl;   
    }

    cudaDeviceSynchronize();
    std::cout<<"CoarsenFactor = "<<coarsen_factor<<std::endl;
    std::cout<<"("<<gridDim.x<<","<<gridDim.y<<","<<gridDim.z<<")"<<std::endl;
    std::cout<<"("<<blockDim.x<<","<<blockDim.y<<","<<blockDim.z<<")"<<std::endl;  
    gpuErrchk(cudaGetLastError());
}

template <typename ValueType, typename IndexType, int CoarsenFactor, int ThreadNz, int DtileSize, int DtileNum, int BlockDimY>
void segment_coo_sorted_sr(const ValueType* src, const IndexType* index, const int nnz, const int N, ValueType* dst) {
    int coarsen_factor = min(CoarsenFactor, 4);
    int real_blockDimX = min(DtileNum, CEIL(N, DtileSize * CoarsenFactor)) * DtileSize;
    int real_blockDimY = min(BlockDimY, CEIL(nnz, ThreadNz));

    dim3 gridDim(CEIL(N, real_blockDimX * CoarsenFactor), CEIL(nnz, real_blockDimY * ThreadNz), 1);
    dim3 blockDim(real_blockDimX, real_blockDimY, 1);

    switch (coarsen_factor) {
        case 4:
            segscan_sr_sorted_kernel<ValueType, IndexType, 4, ThreadNz, DtileSize ><<<gridDim, blockDim>>>(
                src, index, nnz, N, dst);
            break;
        case 3:
            segscan_sr_sorted_kernel<ValueType, IndexType, 3, ThreadNz, DtileSize ><<<gridDim, blockDim>>>(
                src, index, nnz, N, dst);
            break;
        case 2:
            segscan_sr_sorted_kernel<ValueType, IndexType, 2, ThreadNz, DtileSize ><<<gridDim, blockDim>>>(
                src, index, nnz, N, dst);
            break;
        case 1:
            segscan_sr_sorted_kernel<ValueType, IndexType, 1, ThreadNz, DtileSize ><<<gridDim, blockDim>>>(
                src, index, nnz, N, dst);
            break;
        default:
            std::cout<<"CoarsenFactor = "<<coarsen_factor << " is not supported."<<std::endl;
    }
    cudaDeviceSynchronize();
    std::cout<<"CoarsenFactor = "<<coarsen_factor<<std::endl;
    std::cout<<"("<<gridDim.x<<","<<gridDim.y<<","<<gridDim.z<<")"<<std::endl;
    std::cout<<"("<<blockDim.x<<","<<blockDim.y<<","<<blockDim.z<<")"<<std::endl;  
    gpuErrchk(cudaGetLastError());
}

