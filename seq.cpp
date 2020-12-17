#include "Utils.h"


double matmul(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& c) {
  auto start = std::chrono::system_clock::now();

  for (int i = 0; i < SIZE; ++i)
    for (int j = 0; j < SIZE; ++j)
      for (int k = 0; k < SIZE; ++k)
        c[i * SIZE + j] += a[i * SIZE + k] * b[k * SIZE + j];

  auto end = std::chrono::system_clock::now();
  auto time = std::chrono::duration<double>(end - start);
  return time.count();
}


double matmul_ind(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& c) {
  auto start = std::chrono::system_clock::now();

  for (int i = 0; i < SIZE; ++i)
    for (int k = 0; k < SIZE; ++k)
      for (int j = 0; j < SIZE; ++j)
        c[i * SIZE + j] += a[i * SIZE + k] * b[k * SIZE + j];

  auto end = std::chrono::system_clock::now();
  auto time = std::chrono::duration<double>(end - start);
  return time.count();
}


double matmul_opt(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& c) {
  auto start = std::chrono::system_clock::now();
  int numBlocks = SIZE / BLOCK_SIZE;

  for (int ii = 0; ii < SIZE; ii += BLOCK_SIZE)
    for (int kk = 0; kk < SIZE; kk += BLOCK_SIZE)
      for (int jj = 0; jj < SIZE; jj += BLOCK_SIZE)
        for (int i = ii; i < ii + BLOCK_SIZE; i++)
          for (int k = kk; k < kk + BLOCK_SIZE; k++)
            for (int j = jj; j < jj + BLOCK_SIZE; j++)
              c[i * SIZE + j] += a[i * SIZE + k] * b[k * SIZE + j];

  auto end = std::chrono::system_clock::now();
  auto time = std::chrono::duration<double>(end - start);
  return time.count();
}