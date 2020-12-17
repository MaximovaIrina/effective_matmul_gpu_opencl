#include <omp.h>
#include "Utils.h"


double matmul_OMP(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& c) {
  std::cout << "OMP : num threads = " << NUM_THR << "\n";
  double start = omp_get_wtime();

#pragma omp parallel for num_threads(NUM_THR)
  for (int i = 0; i < SIZE; ++i)
    for (int j = 0; j < SIZE; ++j)
      for (int k = 0; k < SIZE; ++k)
        c[i * SIZE + j] += a[i * SIZE + k] * b[k * SIZE + j];

  double end = omp_get_wtime();
  return end - start;
}


double matmul_OMP_ind(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& c) {
  double start = omp_get_wtime();

#pragma omp parallel for num_threads(NUM_THR)
  for (int i = 0; i < SIZE; ++i)
    for (int k = 0; k < SIZE; ++k)
      for (int j = 0; j < SIZE; ++j)
        c[i * SIZE + j] += a[i * SIZE + k] * b[k * SIZE + j];


  double end = omp_get_wtime();
  return end - start;
}


double matmul_OMP_opt(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& c) {
  int numBlocks = SIZE / BLOCK_SIZE;

  double start = omp_get_wtime();

#pragma omp parallel for num_threads(NUM_THR) schedule(static)
  for (int ii = 0; ii < SIZE; ii += BLOCK_SIZE)
    for (int kk = 0; kk < SIZE; kk += BLOCK_SIZE)
     for (int jj = 0; jj < SIZE; jj += BLOCK_SIZE)
        for (int i = ii; i < ii + BLOCK_SIZE; ++i)
          for (int k = kk; k < kk + BLOCK_SIZE; ++k)
            for (int j = jj; j < jj + BLOCK_SIZE; ++j)
              c[i * SIZE + j] += a[i * SIZE + k] * b[k * SIZE + j];

  double end = omp_get_wtime();
  return end - start;
}