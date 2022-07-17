# rb-pr
`coarsen_factor = (N % 4 == 0) ? 4 : (N % 2 == 0) ? 2 : 1;` 

N = 1,2,4,16,64
Test the block_factor; (tile,group)

```
// 256
//csrspmm_parreduce_rowbalance_cg<Index,DType,5,256,5,1,64>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,5,256,5,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,5,256,5,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,5,256,5,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,5,256,5,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,5,256,5,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,5,256,5,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,5,256,5,2,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());

//csrspmm_parreduce_rowbalance_cg<Index,DType,4,256,4,1,64>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,4,256,4,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,4,256,4,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,4,256,4,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,4,256,4,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,4,256,4,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,4,256,4,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,4,256,4,2,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
    
//csrspmm_parreduce_rowbalance_cg<Index,DType,4,256,5,1,64>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,4,256,5,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,4,256,5,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,4,256,5,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,4,256,5,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,4,256,5,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,4,256,5,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());         
csrspmm_parreduce_rowbalance_cg<Index,DType,4,256,5,2,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());         

//csrspmm_parreduce_rowbalance_cg<Index,DType,3,256,3,1,64>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,3,256,3,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,3,256,3,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,3,256,3,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,3,256,3,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,3,256,3,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,3,256,3,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,3,256,3,2,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());

//csrspmm_parreduce_rowbalance_cg<Index,DType,3,256,4,1,64>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,3,256,4,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,3,256,4,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,3,256,4,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,3,256,4,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,3,256,4,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,3,256,4,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,3,256,4,2,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
    
csrspmm_parreduce_rowbalance_cg<Index,DType,2,256,2,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,2,256,2,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,2,256,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,2,256,2,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,2,256,2,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,2,256,2,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,2,256,2,2,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
    
csrspmm_parreduce_rowbalance_cg<Index,DType,2,256,3,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,2,256,3,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,2,256,3,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,2,256,3,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,2,256,3,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,2,256,3,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());         
csrspmm_parreduce_rowbalance_cg<Index,DType,2,256,3,2,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());         

//csrspmm_parreduce_rowbalance_cg<Index,DType,2,256,4,1,64>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,2,256,4,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,2,256,4,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,2,256,4,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,2,256,4,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,2,256,4,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_parreduce_rowbalance_cg<Index,DType,2,256,4,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());         
csrspmm_parreduce_rowbalance_cg<Index,DType,2,256,4,2,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());         


// 512
csrspmm_parreduce_rowbalance_cg<Index,DType,5,512,5,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());         

csrspmm_parreduce_rowbalance_cg<Index,DType,4,512,4,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
    
csrspmm_parreduce_rowbalance_cg<Index,DType,4,512,5,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());         

csrspmm_parreduce_rowbalance_cg<Index,DType,3,512,3,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
    
csrspmm_parreduce_rowbalance_cg<Index,DType,3,512,4,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
    
csrspmm_parreduce_rowbalance_cg<Index,DType,3,512,5,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());         

csrspmm_parreduce_rowbalance_cg<Index,DType,2,512,2,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
    
csrspmm_parreduce_rowbalance_cg<Index,DType,2,512,3,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
    
csrspmm_parreduce_rowbalance_cg<Index,DType,2,512,4,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
    
csrspmm_parreduce_rowbalance_cg<Index,DType,2,512,5,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
```

# eb_sr
N = 64

```
// CEIL(64,32) = 2
csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,2,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,2,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,1,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,1,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
// CEIL(64,16) = 4
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,4,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,4,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,2,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,2,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,1,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,1,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
// CEIL(64,8) = 8
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,8,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,8,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,4,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,4,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,2,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,2,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,1,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,1,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
// CEIL(64,4) = 16
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,8,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,8,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,4,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,4,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,2,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,2,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,1,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,1,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
```

N = 16

```
// CEIL(16,32) = 1
csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,1,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,1,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
// CEIL(16,16) = 1
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,1,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,1,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
// CEIL(16,8) = 2
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,2,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,2,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,1,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,1,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
// CEIL(16,4) = 4
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,4,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,4,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,2,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,2,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,1,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,1,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
```

N = 4

```
// CEIL(4,32) = 1
csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,1,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,1,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
// CEIL(4,16) = 1
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,1,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,1,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
// CEIL(4,8) = 1
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,1,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,1,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
// CEIL(4,4) = 1
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,1,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,1,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
```

# rb_sr

N = 64

```
// Ndim_threadblock = CEIL(64, tilesize) = 1
// Ndim_thread_per_tb = min(64, tile_size) / coarsen_factor = 64
csrspmm_seqreduce_rowbalance_cg<Index,DType,1,256,6,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_seqreduce_rowbalance_cg<Index,DType,1,256,6,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_seqreduce_rowbalance_cg<Index,DType,1,256,6,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_seqreduce_rowbalance_cg<Index,DType,2,256,6,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_seqreduce_rowbalance_cg<Index,DType,2,256,6,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_seqreduce_rowbalance_cg<Index,DType,2,256,6,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_seqreduce_rowbalance_cg<Index,DType,4,256,6,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_seqreduce_rowbalance_cg<Index,DType,4,256,6,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_seqreduce_rowbalance_cg<Index,DType,4,256,6,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_seqreduce_rowbalance_cg<Index,DType,1,256,5,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_seqreduce_rowbalance_cg<Index,DType,2,256,5,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_seqreduce_rowbalance_cg<Index,DType,4,256,5,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_seqreduce_rowbalance_cg<Index,DType,1,256,4,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_seqreduce_rowbalance_cg<Index,DType,2,256,4,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_seqreduce_rowbalance_cg<Index,DType,4,256,4,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
```

N = 16

```
// Ndim_threadblock = CEIL(32, tilesize) = 1
// Ndim_thread_per_tb = min(32, tile_size) / coarsen_factor = 32
csrspmm_seqreduce_rowbalance_cg<Index,DType,1,256,5,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_seqreduce_rowbalance_cg<Index,DType,1,256,5,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_seqreduce_rowbalance_cg<Index,DType,1,256,5,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_seqreduce_rowbalance_cg<Index,DType,2,256,5,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_seqreduce_rowbalance_cg<Index,DType,2,256,5,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_seqreduce_rowbalance_cg<Index,DType,2,256,5,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_seqreduce_rowbalance_cg<Index,DType,4,256,5,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_seqreduce_rowbalance_cg<Index,DType,4,256,5,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_seqreduce_rowbalance_cg<Index,DType,4,256,5,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_seqreduce_rowbalance_cg<Index,DType,1,256,4,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_seqreduce_rowbalance_cg<Index,DType,2,256,4,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_seqreduce_rowbalance_cg<Index,DType,4,256,4,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_seqreduce_rowbalance_cg<Index,DType,1,256,3,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_seqreduce_rowbalance_cg<Index,DType,2,256,3,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_seqreduce_rowbalance_cg<Index,DType,4,256,3,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
```

N = 4
```
// Ndim_threadblock = CEIL(4, tilesize) = 1
// Ndim_thread_per_tb = min(4, tile_size) / coarsen_factor = 4
csrspmm_seqreduce_rowbalance_cg<Index,DType,1,256,2,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_seqreduce_rowbalance_cg<Index,DType,1,256,2,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_seqreduce_rowbalance_cg<Index,DType,1,256,2,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_seqreduce_rowbalance_cg<Index,DType,2,256,2,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_seqreduce_rowbalance_cg<Index,DType,2,256,2,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_seqreduce_rowbalance_cg<Index,DType,2,256,2,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
```
