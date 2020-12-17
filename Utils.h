#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <iostream>
#include <chrono>
#include <ctime>
#include <time.h>
#include <CL/cl.h>
#include <vector>

#define alpha 1.0f
#define betta 2.0f
#define SIZE 1024
#define BLOCK_SIZE 16
#define N_TEST 14
#define NUM_THR 4
#define NUM_PLATFORMS 3

enum Test {
	GPU,
	CPU, 
	OMP,
	SEQ,
	GPU_IND, //
	CPU_IND, //
	OMP_IND, 
	SEQ_IND,
	GPU_OPT, //
	CPU_OPT, //
	OMP_OPT, 
	SEQ_OPT,
	GPU_IMG, //
	CPU_IMG  //
};

void CHK(int status);

double matmul(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& c);
double matmul_CL(const size_t plat_id, const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& c);
double matmul_OMP(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& c);

double matmul_ind(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& c);
double matmul_CL_ind(const size_t plat_id, const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& c);
double matmul_OMP_ind(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& c);

double matmul_opt(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& c);
double matmul_CL_opt(const size_t plat_id, const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& c);
double matmul_OMP_opt(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& c);

double matmul_CL_img(const size_t plat_id, const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& c);
