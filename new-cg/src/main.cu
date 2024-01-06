#include "../include/segscan/segscan.cuh"
#include "../include/utils/check.cuh"
#include "../include/dataloader/dataloader.h"

using namespace std;

int main(int argc, const char **argv) {
    // Read out the first column of the mtx file
    // Random generate [nnz, N] dense vector
    const char *filename = argv[1];
    int feature_size = atoi(argv[2]);

    return 0;
}