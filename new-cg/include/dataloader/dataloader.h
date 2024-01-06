#ifndef DATA_LOADER
#define DATA_LOADER
#include "mmio.h"
#include <random>
#include <cmath>
#include <vector>
#include <iostream>
#include <boost/algorithm/string.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/algorithm_ext.hpp>
#include <boost/range/irange.hpp>

void generateSrc(int nnz, int N, std::vector<float>& src) {
    std::default_random_engine generator;
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    src.resize(nnz * N);
    for (int i = 0; i < nnz * N; ++i) {
        src[i] = dis(generator);
    }
}

int generateIndex(int range, int max_seg, int total_count, std::vector<int>& result) {
    // range: Elements of index `i` in [0, range)
    // max_seg: Maximum repetition `mi` of each element of index `i`
    // avg: Desired average of `mi`
    double avg = static_cast<double>(total_count) / range;
    result.resize(total_count);

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(avg, avg/2); // mean max_seg/2, std dev max_seg/4

    int current_sum = 0;
    int dst_len = 0;
    for (int i = 0; i < range; ++i) {
        int mi = static_cast<int>(std::round(distribution(generator)));

        // Ensure mi is within bounds
        if (mi < 0) mi = 0;
        if (mi >= max_seg) mi = max_seg - 1;

        // Adjust the last element to match the desired total count
        if (i == range - 1) {
            mi = total_count - current_sum;
            if (mi < 0) mi = 0; // Ensure mi is not negative
            if (mi >= max_seg) mi = max_seg - 1; // Ensure mi is within bounds
        }

        for (int j = 0; j < mi; ++j) {
            result.push_back(i);
        }
        current_sum += mi;
        if (mi > 0) dst_len += 1;

        // Early exit if we reached the total count
        if (current_sum >= total_count) break;
    }
    return dst_len;
}

// Read out the first column of the mtx file and values are dropped.
void read_mtx_rowid(const char *filename, int &nrow, int &nnz,
                   std::vector<int> &coo_rowind_buffer,
                   int &col, 
                   bool one_based = true) {
  FILE *f;
  int ncol;

  if ((f = fopen(filename, "r")) == NULL) {
    printf("File %s not found", filename);
    exit(EXIT_FAILURE);
  }

  MM_typecode matcode;
  // Read MTX banner
  if (mm_read_banner(f, &matcode) != 0) {
    printf("Could not process this file.\n");
    exit(EXIT_FAILURE);
  }
  if (mm_read_mtx_crd_size(f, &nrow, &ncol, &nnz) != 0) {
    printf("Could not process this file.\n");
    exit(EXIT_FAILURE);
  }
  if (!mm_is_matrix(matcode) || !mm_is_sparse(matcode) ||
      !mm_is_coordinate(matcode)) {
    printf("Could not process this file.\n");
    exit(EXIT_FAILURE);
  }
  // printf("Reading matrix %d rows, %d columns, %d nnz.\n", nrow, ncol, nnz);

  /// read tuples
  std::vector<float> value_temp;
  std::vector<std::tuple<int, int>> coords;
  int row_id, col_id;
  float dummy;
  for (int64_t i = 0; i < nnz; i++) {
    if (fscanf(f, "%d", &row_id) == EOF) {
      std::cout << "Error: not enough rows in mtx file.\n";
      exit(EXIT_FAILURE);
    } else {
      fscanf(f, "%d", &col_id);
      if (mm_is_pattern(matcode)) {
        // random value between 0 and 1
        dummy = 1.0;
      } else {
        fscanf(f, "%f", &dummy);
      }
      // mtx format is 1-based
      if (one_based) {
        coords.push_back(std::make_tuple(row_id - 1, col_id - 1));
      } else {
        coords.push_back(std::make_tuple(row_id, col_id));
      }
      value_temp.push_back(dummy);
    }
  }

  /// make symmetric
  std::vector<int> index;
  std::vector<std::tuple<int, int>> new_coords;
  if (mm_is_symmetric(matcode)) {
    int cur_nz = 0;
    int cur_ptr = 0;
    for (auto iter = coords.begin(); iter != coords.end(); iter++) {
      int i = std::get<0>(*iter);
      int j = std::get<1>(*iter);
      if (i != j) {
        new_coords.push_back(std::make_tuple(i, j));
        index.push_back(cur_nz);
        cur_nz++;
        new_coords.push_back(std::make_tuple(j, i));
        index.push_back(cur_nz);
        cur_nz++;
      } else {
        new_coords.push_back(std::make_tuple(i, j));
        index.push_back(cur_nz);
        cur_nz++;
      }
      cur_ptr++;
    }
    //std::sort(new_coords.begin(), new_coords.end());
    std::sort(index.begin(), index.end(),
              [&new_coords](int i1, int i2) {
                return std::get<0>(new_coords[i1]) == std::get<0>(new_coords[i2]) ? std::get<1>(new_coords[i1]) < std::get<1>(new_coords[i2]) : std::get<0>(new_coords[i1]) < std::get<0>(new_coords[i2]);
              });
    nnz = cur_nz;
  } else {
    boost::range::push_back(index, boost::irange(0, nnz));
    std::sort(index.begin(), index.end(),
          [&coords](int i1, int i2) {
            return std::get<0>(coords[i1]) == std::get<0>(coords[i2]) ? std::get<1>(coords[i1]) < std::get<1>(coords[i2]) : std::get<0>(coords[i1]) < std::get<0>(coords[i2]);
          });
  }
  /// generate csr from coo

  coo_rowind_buffer.clear();

  int curr_pos = 0;
  if (mm_is_symmetric(matcode)) {
    for (int64_t row = 0; row < nrow; row++) {
      while ((curr_pos < nnz) && (std::get<0>(new_coords[index[curr_pos]]) == row)) {
        coo_rowind_buffer.push_back(std::get<0>(new_coords[index[curr_pos]]));
        curr_pos++;
      }
    }
  } else {
    for (int64_t row = 0; row < nrow; row++) {
      while ((curr_pos < nnz) && (std::get<0>(coords[index[curr_pos]]) == row)) {
        coo_rowind_buffer.push_back(std::get<0>(coords[index[curr_pos]]));
        curr_pos++;
      }
    }
  }
  nnz = coo_rowind_buffer.size();
  fclose(f);
}

#endif