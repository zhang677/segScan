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
        csrspmm_parreduce_nnzbalance<Index,DType>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,5,256,5,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,5,256,5,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,5,256,5,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,5,256,5,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,5,256,5,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,5,256,5,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,4,256,5,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,4,256,5,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,4,256,5,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,4,256,5,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,4,256,5,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,4,256,5,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,3,256,5,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,3,256,5,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,3,256,5,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,3,256,5,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,3,256,5,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,3,256,5,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,256,5,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,256,5,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,256,5,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,256,5,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,256,5,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,256,5,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,4,256,4,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,4,256,4,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,4,256,4,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,4,256,4,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,4,256,4,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,4,256,4,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,3,256,4,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,3,256,4,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,3,256,4,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,3,256,4,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,3,256,4,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,3,256,4,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,256,4,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,256,4,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,256,4,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,256,4,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,256,4,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,256,4,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,3,256,3,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,3,256,3,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,3,256,3,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,3,256,3,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,3,256,3,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,3,256,3,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,256,3,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,256,3,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,256,3,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,256,3,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,256,3,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,256,3,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,256,2,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,256,2,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,256,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,256,2,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,256,2,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,256,2,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,4,512,4,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,4,512,4,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,4,512,4,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,4,512,4,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,4,512,4,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,4,512,4,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,3,512,4,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,3,512,4,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,3,512,4,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,3,512,4,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,3,512,4,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,3,512,4,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,512,4,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,512,4,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,512,4,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,512,4,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,512,4,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,512,4,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,3,512,3,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,3,512,3,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,3,512,3,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,3,512,3,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,3,512,3,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,3,512,3,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,512,3,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,512,3,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,512,3,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,512,3,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,512,3,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,512,3,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,512,2,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,512,2,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,512,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,512,2,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,512,2,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_parreduce_nnzbalance_cg<Index,DType,2,512,2,1,1>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        /*
        csrspmm_rowcaching_nnzbalance<Index,DType>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,2,1,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,2,1,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,2,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,2,1,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,2,1,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,2,2,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,2,2,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,2,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,2,2,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,2,2,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,1,1,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,1,1,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,1,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,1,1,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,1,1,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,1,2,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,1,2,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,1,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,1,2,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,1,2,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,4,1,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,4,1,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,4,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,4,1,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,4,1,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,4,2,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,4,2,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,4,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,4,2,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,4,2,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,2,1,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,2,1,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,2,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,2,1,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,2,1,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,2,2,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,2,2,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,2,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,2,2,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,2,2,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,1,1,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,1,1,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,1,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,1,1,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,1,1,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,1,2,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,1,2,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,1,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,1,2,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,1,2,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,8,1,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,8,1,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,8,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,8,1,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,8,1,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,8,2,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,8,2,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,8,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,8,2,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,8,2,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,4,1,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,4,1,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,4,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,4,1,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,4,1,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,4,2,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,4,2,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,4,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,4,2,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,4,2,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,2,1,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,2,1,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,2,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,2,1,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,2,1,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,2,2,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,2,2,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,2,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,2,2,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,2,2,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,1,1,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,1,1,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,1,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,1,1,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,1,1,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,1,2,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,1,2,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,1,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,1,2,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,1,2,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,8,1,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,8,1,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,8,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,8,1,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,8,1,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,8,2,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,8,2,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,8,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,8,2,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,8,2,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,4,1,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,4,1,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,4,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,4,1,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,4,1,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,4,2,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,4,2,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,4,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,4,2,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,4,2,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,2,1,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,2,1,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,2,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,2,1,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,2,1,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,2,2,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,2,2,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,2,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,2,2,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,2,2,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,1,1,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,1,1,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,1,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,1,1,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,1,1,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,1,2,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,1,2,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,1,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,1,2,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,1,2,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());checkSpMMsuffix;
        */
        
    } else if (mode == Mode::test) {
        csrspmm_cusparse<Index,DType>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        csrspmm_rowcaching_nnzbalance<Index,DType>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,1,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,1,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,1,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,1,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,2,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,2,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,1,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,1,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,4,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,4,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,2,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,2,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,1,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,1,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
        /*
csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,2,1,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,2,1,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,2,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,2,1,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,2,1,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,2,2,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,2,2,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,2,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,2,2,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,2,2,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,1,1,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,1,1,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,1,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,1,1,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,1,1,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,1,2,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,1,2,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,1,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,1,2,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,5,256,1,2,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,4,1,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,4,1,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,4,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,4,1,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,4,1,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,4,2,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,4,2,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,4,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,4,2,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,4,2,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,2,1,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,2,1,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,2,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,2,1,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,2,1,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,2,2,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,2,2,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,2,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,2,2,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,2,2,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,1,1,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,1,1,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,1,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,1,1,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,1,1,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,1,2,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,1,2,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,1,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,1,2,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,4,256,1,2,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,8,1,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,8,1,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,8,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,8,1,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,8,1,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,8,2,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,8,2,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,8,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,8,2,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,8,2,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,4,1,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,4,1,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,4,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,4,1,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,4,1,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,4,2,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,4,2,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,4,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,4,2,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,4,2,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,2,1,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,2,1,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,2,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,2,1,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,2,1,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,2,2,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,2,2,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,2,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,2,2,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,2,2,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,1,1,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,1,1,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,1,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,1,1,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,1,1,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,1,2,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,1,2,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,1,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,1,2,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,3,256,1,2,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,8,1,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,8,1,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,8,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,8,1,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,8,1,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,8,2,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,8,2,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,8,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,8,2,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,8,2,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,4,1,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,4,1,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,4,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,4,1,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,4,1,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,4,2,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,4,2,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,4,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,4,2,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,4,2,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,2,1,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,2,1,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,2,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,2,1,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,2,1,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,2,2,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,2,2,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,2,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,2,2,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,2,2,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,1,1,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,1,1,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,1,1,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,1,1,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,1,1,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,1,2,1,32>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,1,2,1,16>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,1,2,1,8>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,1,2,1,4>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());
csrspmm_rowcaching_nnzbalance_cg<Index,DType,2,256,1,2,1,2>(H, feature_size, in_feature.d_array.get(),out_feature.d_array.get());   
*/
    }
    else {
        std::cout<<"Not implemented yet!"<<std::endl;
    }


    return 0;
}