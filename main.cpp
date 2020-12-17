#include <iomanip>
#include "Utils.h"

bool check_result(std::vector<std::vector<int>> c) {
  bool flag = true;
  float ans = alpha * betta * SIZE;

  for (int i = 0; i < SIZE * SIZE; ++i) {
    for (int test = 0; test < N_TEST; ++test)
      if (c[test][i] != ans) {
        flag = false;
        std::cout << "[" << test << "] [" << (int)(i / SIZE) << "," << i % SIZE << "] " << c[test][i] << "\t";
      }
    if (!flag) std::cout << "\n";
  }
      
  if (flag) std::cout << "\nresult - CORRECT\n";
  return flag;
}


int main() {
  std::cout << "=============================================== START ==================================================\n";
  std::cout << "Size: " << SIZE << "\n";

  std::vector<int> a(SIZE * SIZE, alpha);
  std::vector<int> b(SIZE * SIZE, betta);
  std::vector<int> v(SIZE * SIZE, 0.0f);
  std::vector<std::vector<int>> c(N_TEST, v);

  // PC
  // 0 - Intel(R) OpenCL HD Graphics
  // 1 - Intel(R) CPU Runtime for OpenCL(TM) Applications

  // LAPTOP
  // 0 - NVIDIA CUDA
  // 2 - Intel(R) CPU Runtime for OpenCL(TM) Applications

  double gpu_t = matmul_CL(GPU, a, b, c[GPU]);
  double cpu_t = matmul_CL(CPU+1, a, b, c[CPU]); // EDIT
  double omp_t = matmul_OMP(a, b, c[OMP]);
  double seq_t = matmul(a, b, c[SEQ]);

  double gpu_ind_t = matmul_CL_ind(GPU, a, b, c[GPU_IND]);
  double cpu_ind_t = matmul_CL_ind(CPU+1, a, b,  c[CPU_IND]); // EDIT
  double omp_ind_t = matmul_OMP_ind(a, b, c[OMP_IND]);
  double seq_ind_t = matmul_ind(a, b, c[SEQ_IND]);

  double gpu_opt_t = matmul_CL_opt(GPU, a, b, c[GPU_OPT]);
  double cpu_opt_t = matmul_CL_opt(CPU+1, a, b, c[CPU_OPT]); // EDIT
  double omp_opt_t = matmul_OMP_opt(a, b, c[OMP_OPT]);
  //double seq_opt_t = matmul_ind(a, b, c[SEQ_OPT]);

  double gpu_img_t = matmul_CL_img(GPU, a, b, c[GPU_IMG]);
  double cpu_img_t = matmul_CL_img(CPU+1, a, b, c[CPU_IMG]); // EDIT

  //check_result(c);

  std::cout << "\n";
  std::cout << std::setprecision(6) << "GPU : " << gpu_t << "\n";
  std::cout << std::setprecision(6) << "CPU : " << cpu_t << "\n";
  std::cout << std::setprecision(6) << "OMP : " << omp_t << "\n";
  std::cout << std::setprecision(6) << "SEQ : " << seq_t << "\n\n";

  std::cout << std::setprecision(6) << "GPU_ind : " << gpu_ind_t << "\n";
  std::cout << std::setprecision(6) << "CPU_ind : " << cpu_ind_t << "\n";
  std::cout << std::setprecision(6) << "OMP_ind : " << omp_ind_t << "\n";
  std::cout << std::setprecision(6) << "SEQ_ind : " << seq_ind_t << "\n\n";

  std::cout << std::setprecision(6) << "GPU_block : " << gpu_opt_t << "\n";
  std::cout << std::setprecision(6) << "CPU_block : " << cpu_opt_t << "\n";
  std::cout << std::setprecision(6) << "OMP_block : " << omp_opt_t << "\n\n";
  //std::cout << std::setprecision(6) << "SEQ_OPT : " << seq_opt_t << "\n\n";

  std::cout << std::setprecision(6) << "GPU_img : " << gpu_img_t << "\n";
  std::cout << std::setprecision(6) << "CPU_img : " << cpu_img_t << "\n";

  return 0;
}