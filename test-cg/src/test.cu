#include "../include/spmm/spmm.cuh"
#include "../include/spmm/spmm_cg.cuh"
#include "../include/dataloader/dataloader.hpp"
#include "../include/util/ramArray.cuh"
#include "../include/util/check.cuh"

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpGEMM
#include <fstream>
#include <iostream>
#include <stdio.h>  // printf
#include <stdlib.h> // EXIT_FAILURE

using namespace std;

// None-template, No timer version
enum Mode {
    check,
    test,
    tune,
};

int main(int argc, const char **argv) {

    // Check command-line argument
    if (argc < 3) {
        printf("Require command-line argument: path of the sparse matrix file in "
                ".mtx format. Feature size. Mode.\n");
        return EXIT_FAILURE;
    }
    const char *filename = argv[1];
    int feature_size = atoi(argv[2]);
    Mode mode = static_cast<Mode>(atoi(argv[3]));

    // Load Sparse Matrix
    SpMatCsrDescr_t<Index, DType> H = SingleDataLoader<Index, DType>(filename);
    int nrow = H.nrow;
    int ncol = H.ncol;

    // Prepare Dense Matrix
    util::RamArray<DType> in_feature(ncol * feature_size), out_feature(nrow * feature_size), out_ref(nrow * feature_size);
    in_feature.fill_default_one();
    out_feature.fill_zero_h();
    out_ref.fill_zero_h();

    // CopyToDevice
    H.upload();
    in_feature.upload();
    out_feature.upload();

    // Call the kernels
    if (mode == Mode::check) {
        util::spmm_reference_host<Index, DType>(
            H.nrow, feature_size, H.sp_csrptr.h_array.get(),
            H.sp_csrind.h_array.get(), H.sp_data.h_array.get(),
            in_feature.h_array.get(), out_ref.h_array.get());
        out_feature.reset();
        csrspmm_seqreduce_rowbalance<Index,DType>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_seqreduce_rowbalance_cg<Index,DType,4,256,5,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_seqreduce_rowbalance_cg<Index,DType,2,256,5,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_seqreduce_rowbalance_cg<Index,DType,1,256,5,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_seqreduce_rowbalance_cg<Index,DType,4,256,6,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_seqreduce_rowbalance_cg<Index,DType,2,256,6,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_seqreduce_rowbalance_cg<Index,DType,1,256,6,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_seqreduce_rowbalance_cg<Index,DType,4,256,4,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_seqreduce_rowbalance_cg<Index,DType,2,256,4,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_seqreduce_rowbalance_cg<Index,DType,1,256,4,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_seqreduce_rowbalance_cg<Index,DType,4,256,2,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_seqreduce_rowbalance_cg<Index,DType,2,256,2,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_seqreduce_rowbalance_cg<Index,DType,1,256,2,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        // 256
        /*
        csrspmm_parreduce_rowbalance<Index,DType>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix; 
        csrspmm_parreduce_rowbalance_cg<Index,DType,5,256,5,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix; 
        csrspmm_parreduce_rowbalance_cg<Index,DType,5,256,6,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix; 
    //    csrspmm_parreduce_rowbalance_cg<Index,DType,5,256,7>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
    //    checkSpMMsuffix; // Wrong
        csrspmm_parreduce_rowbalance_cg<Index,DType,4,256,4,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix; 
        csrspmm_parreduce_rowbalance_cg<Index,DType,4,256,5,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix; 
        csrspmm_parreduce_rowbalance_cg<Index,DType,4,256,6,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix; 
    //    csrspmm_parreduce_rowbalance_cg<Index,DType,4,256,7>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
    //    checkSpMMsuffix;// Wrong
        csrspmm_parreduce_rowbalance_cg<Index,DType,3,256,3,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix; 
        csrspmm_parreduce_rowbalance_cg<Index,DType,3,256,4,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix; 
        csrspmm_parreduce_rowbalance_cg<Index,DType,3,256,5,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix; 
        csrspmm_parreduce_rowbalance_cg<Index,DType,3,256,6,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix; 
    //    csrspmm_parreduce_rowbalance_cg<Index,DType,3,256,7>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
    //    checkSpMMsuffix;// Wrong
        csrspmm_parreduce_rowbalance_cg<Index,DType,2,256,2,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_parreduce_rowbalance_cg<Index,DType,2,256,3,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_parreduce_rowbalance_cg<Index,DType,2,256,4,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_parreduce_rowbalance_cg<Index,DType,2,256,5,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_parreduce_rowbalance_cg<Index,DType,2,256,6,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
    //    csrspmm_parreduce_rowbalance_cg<Index,DType,2,256,7>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
    //    checkSpMMsuffix;// Wrong
        */
        // 512
        /*
        csrspmm_parreduce_rowbalance_cg<Index,DType,5,512,5,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_parreduce_rowbalance_cg<Index,DType,5,512,6,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
    //    csrspmm_parreduce_rowbalance_cg<Index,DType,5,512,7>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
    //    checkSpMMsuffix; // Wrong
        csrspmm_parreduce_rowbalance_cg<Index,DType,4,512,4,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_parreduce_rowbalance_cg<Index,DType,4,512,5,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_parreduce_rowbalance_cg<Index,DType,4,512,6,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
    //    csrspmm_parreduce_rowbalance_cg<Index,DType,4,512,7>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
    //    checkSpMMsuffix;// Wrong
        csrspmm_parreduce_rowbalance_cg<Index,DType,3,512,3,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_parreduce_rowbalance_cg<Index,DType,3,512,4,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_parreduce_rowbalance_cg<Index,DType,3,512,5,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_parreduce_rowbalance_cg<Index,DType,3,512,6,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
    //    csrspmm_parreduce_rowbalance_cg<Index,DType,3,512,7>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
    //    checkSpMMsuffix;// Wrong
        csrspmm_parreduce_rowbalance_cg<Index,DType,2,512,2,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_parreduce_rowbalance_cg<Index,DType,2,512,3,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_parreduce_rowbalance_cg<Index,DType,2,512,4,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_parreduce_rowbalance_cg<Index,DType,2,512,5,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_parreduce_rowbalance_cg<Index,DType,2,512,6,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
    //    csrspmm_parreduce_rowbalance_cg<Index,DType,2,512,7>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
    //    checkSpMMsuffix;// Wrong
        
        
        csrspmm_parreduce_rowbalance_cg<Index,DType,5,256,5,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_parreduce_rowbalance_cg<Index,DType,5,256,5,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_parreduce_rowbalance_cg<Index,DType,5,256,5,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_parreduce_rowbalance_cg<Index,DType,5,256,5,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_parreduce_rowbalance_cg<Index,DType,5,256,5,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_parreduce_rowbalance_cg<Index,DType,5,256,5,2,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_parreduce_rowbalance_cg<Index,DType,5,256,5,3,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_parreduce_rowbalance_cg<Index,DType,5,256,5,4,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        */
        
        
    } else if (mode == Mode::test) {
        csrspmm_cusparse<Index,DType>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance<Index,DType>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        /*
        csrspmm_parreduce_rowbalance_cg<Index,DType,5,128,5,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());

        csrspmm_parreduce_rowbalance_cg<Index,DType,4,128,4,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
         
        csrspmm_parreduce_rowbalance_cg<Index,DType,4,128,5,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());         

        csrspmm_parreduce_rowbalance_cg<Index,DType,3,128,3,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
         
        csrspmm_parreduce_rowbalance_cg<Index,DType,3,128,4,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
         
        csrspmm_parreduce_rowbalance_cg<Index,DType,3,128,5,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());         

        csrspmm_parreduce_rowbalance_cg<Index,DType,2,128,2,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
         
        csrspmm_parreduce_rowbalance_cg<Index,DType,2,128,3,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
         
        csrspmm_parreduce_rowbalance_cg<Index,DType,2,128,4,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
         
        csrspmm_parreduce_rowbalance_cg<Index,DType,2,128,5,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        */
        //128
        /*
        //csrspmm_parreduce_rowbalance_cg<Index,DType,5,128,5,1,64>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,5,128,5,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,5,128,5,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,5,128,5,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,5,128,5,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,5,128,5,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,5,128,5,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,5,128,5,2,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());

        //csrspmm_parreduce_rowbalance_cg<Index,DType,4,128,4,1,64>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,4,128,4,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,4,128,4,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,4,128,4,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,4,128,4,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,4,128,4,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,4,128,4,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
            
        //csrspmm_parreduce_rowbalance_cg<Index,DType,4,128,5,1,64>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,4,128,5,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,4,128,5,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,4,128,5,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,4,128,5,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,4,128,5,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,4,128,5,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());         

        //csrspmm_parreduce_rowbalance_cg<Index,DType,3,128,3,1,64>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,3,128,3,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,3,128,3,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,3,128,3,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,3,128,3,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,3,128,3,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,3,128,3,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        
        //csrspmm_parreduce_rowbalance_cg<Index,DType,3,128,4,1,64>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,3,128,4,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,3,128,4,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,3,128,4,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,3,128,4,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,3,128,4,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,3,128,4,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        */
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
        
        /*
        csrspmm_parreduce_rowbalance_cg<Index,DType,4,128,5,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,4,128,5,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,4,128,5,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,4,128,5,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,4,128,5,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,4,128,5,2,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,4,128,5,3,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,4,128,5,4,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        */
        /*
        csrspmm_parreduce_rowbalance_cg<Index,DType,4,256,5,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,4,256,5,5,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,4,256,5,7,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,4,256,5,9,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,4,256,5,11,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,4,256,5,13,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_parreduce_rowbalance_cg<Index,DType,4,256,5,15,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        */        
    }
    else {
        std::cout<<"Not implemented yet!"<<std::endl;
    }


    return 0;
}