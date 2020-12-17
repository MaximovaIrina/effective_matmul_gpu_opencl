#include "Utils.h"

void CHK(int status) {
    if (status != 0) std::cout << "Status = " << status << std::endl;
}

//printf(\"%d, %d \\n\", i, j);

double matmul_CL(const size_t plat_id, const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& c) {
  cl_int error = 0;
  const char* source = 
      "__kernel void matmul(__global int* a, __global int* b, __global int* c) {          \n"
      "  int i = get_global_id(0);                                                        \n"
      "  int j = get_global_id(1);                                                        \n"
      "  int size = get_global_size(0);                                                   \n"
      "  int summ = 0;                                                                    \n"
      "                                                                                   \n"
      "  for (int k = 0; k < size; ++k)                                                   \n"
      "    summ += a[i * size + k] * b[k * size + j];                                     \n"
      "                                                                                   \n"
      "  c[i * size + j] = summ;                                                          \n"
      "}";

  // ����� ���������
  cl_platform_id platforms[NUM_PLATFORMS];
  clGetPlatformIDs(NUM_PLATFORMS, platforms, NULL); 
  cl_platform_id platform = platforms[plat_id];

  char platformName[128];
  clGetPlatformInfo(platform, CL_PLATFORM_NAME, 128, platformName, NULL);
  
  if (plat_id == 0)
      std::cout << "GPU : " << platformName << " GeRorce 940MX\n";
  else 
      std::cout << "CPU : " << platformName << "\n";

  // �������� ���������
  cl_context_properties properties[3] = {
    CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0
  };

  // ������� �������� � ��������� ���������� ��� ���� ����������� �����������
  cl_context context = clCreateContextFromType((platform == NULL) ? NULL : properties,
      plat_id == 0 ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU,
      NULL, NULL, &error);
  CHK(error);

  // ���������� ������ ������� (� ������) ��� �������� ������ ���������
  size_t sizeContext = 0;
  clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &sizeContext);
  
  // �������� ���������� ��� ����������
  cl_device_id device = NULL;
  if (sizeContext > 0) {
      cl_device_id* devices = (cl_device_id*)alloca(sizeContext);
      clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeContext, devices, NULL);
      device = devices[0];
  }

  // �������� ������� ������ ��� ��������� ��������� � ����������
  cl_command_queue queue = clCreateCommandQueue(context, device, 0, &error);
  CHK(error);

  // �������� �������� ��������� � ���� 
  size_t srclen[] = { strlen(source) };
  cl_program program = clCreateProgramWithSource(context, 1, &source, srclen, &error);
  CHK(error);

  // ������� ����������� ���� ��������� ��� ���������� ����������
  error = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  if (error != CL_SUCCESS) {
      std::cout << "Build program failed\n";
      size_t logsize = 0;
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logsize);
      std::vector<char> log(logsize);
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logsize, log.data(), nullptr);
      std::cout << std::string(log.data());
  }

  // ������� ������ ����
  cl_kernel kernel = clCreateKernel(program, "matmul", &error);
  CHK(error);

  // C������ ������ ������ � ���� ������ ��� �������� ���� ��������
  cl_mem aClBuf = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * SIZE * SIZE, NULL, &error);
  CHK(error);
  cl_mem bClBuf = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * SIZE * SIZE, NULL, &error);
  CHK(error);
  cl_mem cClBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * SIZE * SIZE, NULL, &error);
  CHK(error);

  // �������� � ������� ������� ������ �������� ������� � ������ ������
  cl_event event;
  CHK(clEnqueueWriteBuffer(queue, aClBuf, CL_TRUE, 0, sizeof(int) * SIZE * SIZE, a.data(), 0, NULL, &event));
  CHK(clEnqueueWriteBuffer(queue, bClBuf, CL_TRUE, 0, sizeof(int) * SIZE * SIZE, b.data(), 0, NULL, &event));
  CHK(clEnqueueWriteBuffer(queue, cClBuf, CL_TRUE, 0, sizeof(int) * SIZE * SIZE, c.data(), 0, NULL, &event));

  // ������ ���� 
  // ������ ��������� ����
  CHK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &aClBuf));
  CHK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &bClBuf));
  CHK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &cClBuf));

  // ����������� ����������� � ���������� �������� ������ � ������ ����
  const size_t local_ws[] = { BLOCK_SIZE, BLOCK_SIZE };
  const size_t global_ws[] = { SIZE, SIZE };
  auto start = std::chrono::system_clock::now();
  CHK(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_ws, local_ws, 0, NULL, &event));

  // ������� ���������� ���� ������ � �������
  CHK(clFinish(queue));
  auto end = std::chrono::system_clock::now();
  auto time = std::chrono::duration<double>(end - start);

  // �������� ����������� 
  // ����������� ��������������� ������ � ������ �����
  CHK(clEnqueueReadBuffer(queue, cClBuf, CL_TRUE, 0, sizeof(int) * SIZE * SIZE, c.data(), 0, NULL, &event));

  // �c���������� �������������� ��������
  clReleaseMemObject(aClBuf);
  clReleaseMemObject(bClBuf);
  clReleaseMemObject(cClBuf);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  return time.count();
}