#include "Utils.h"

// ��������� ������� ������

double matmul_CL_img(const size_t plat_id, const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& c) {
	cl_int error = 0;
	const char* source =
		"__kernel void matmul_img(__read_only image2d_t a, __read_only image2d_t b, __write_only image2d_t c,	  \n"
		"																			__local int* aTemp, __local int* bTemp) {										      \n"
		"  int i = get_global_id(1);																																			      \n"
		"  int j = get_global_id(0);																																			      \n"
		"																																																				\n"
		"  int iLoc = get_local_id(1);																																		      \n"
		"  int jLoc = get_local_id(0);  																																	      \n"
		"																																																				\n"
		"  int size = get_global_size(0);																																				\n"
		"  int block_size = get_local_size(0);																																	\n"
		"																																																				\n"
		"  int cIJ = 0;        												  								  																		  \n"
		"																																																				\n"
		"  int numBlocks = size / block_size;																																		\n"
		"																																																				\n"
		"	  for (int block = 0; block < numBlocks; ++block) {																										\n"
		"	  	int blockRow = block * block_size + iLoc;																													\n"
		"	  	int blockCol = block * block_size + jLoc;																													\n"
		"	  																							 																											\n"
		"	  	int2 aCoord = { i, blockCol };																																		\n"
		"	  	int2 bCoord = { blockRow, j };																																		\n"
		"	  																																																		\n"
		"	  	aTemp[iLoc * block_size + jLoc] = read_imagei(a, aCoord).x;																				\n"
		"	  	bTemp[iLoc * block_size + jLoc] = read_imagei(b, bCoord).x;																				\n"
		"	  																																																		\n"
		"	  	barrier(CLK_LOCAL_MEM_FENCE);																																			\n"
		"	  	for (int k = 0; k < block_size; k++)																															\n"
		"	  		cIJ += aTemp[iLoc * block_size + k] * bTemp[iLoc * block_size + k];															\n"
		"	  	barrier(CLK_LOCAL_MEM_FENCE);																																			\n"
		"	  }																																																		\n"
		"    																																																		\n"
		"	 int2 cCoord = {i , j};																																								\n"
		"	 write_imagei(c, cCoord, cIJ);                																												\n"
		"}";

	// ����� ���������
	cl_platform_id platforms[NUM_PLATFORMS];
	clGetPlatformIDs(NUM_PLATFORMS, platforms, NULL);
	cl_platform_id platform = platforms[plat_id];

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
	cl_kernel kernel = clCreateKernel(program, "matmul_img", &error);
	CHK(error);

	// C������ ������ ������ � ���� IMAGE ��� �������� ���� ��������
	cl_image_format format = { CL_R , CL_SIGNED_INT32 };
	_cl_image_desc desc = {};
	desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	desc.image_width = SIZE;
	desc.image_height = SIZE;
	cl_mem aClImg = clCreateImage(context, CL_MEM_READ_ONLY, &format, &desc, NULL, &error);
	CHK(error);
	cl_mem bClImg = clCreateImage(context, CL_MEM_READ_ONLY, &format, &desc, NULL, &error);
	CHK(error);
	cl_mem cClImg = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, NULL, &error);
	CHK(error);;


	// �������� � ������� ������� ������ �������� ������� � ������ ������
	size_t Origin[] = { 0, 0, 0 };
	size_t Region[] = { SIZE, SIZE, 1 };
	cl_event event;
	CHK(clEnqueueWriteImage(queue, aClImg, CL_TRUE, Origin, Region, 0, 0, a.data(), NULL, NULL, &event));
	CHK(clEnqueueWriteImage(queue, bClImg, CL_TRUE, Origin, Region, 0, 0, b.data(), NULL, NULL, &event));
	CHK(clEnqueueWriteImage(queue, cClImg, CL_TRUE, Origin, Region, 0, 0, c.data(), NULL, NULL, &event));

	// ������ ���� 
	// ������ ��������� ����
	CHK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &aClImg));
	CHK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &bClImg));
	CHK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &cClImg));
	CHK(clSetKernelArg(kernel, 3, sizeof(int) * BLOCK_SIZE * BLOCK_SIZE, NULL));
	CHK(clSetKernelArg(kernel, 4, sizeof(int) * BLOCK_SIZE * BLOCK_SIZE, NULL));

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
	CHK(clEnqueueReadImage(queue, cClImg, CL_TRUE, Origin, Region, 0, 0, c.data(), NULL, NULL, &event));

	// �c���������� �������������� ��������
	clReleaseMemObject(aClImg);
	clReleaseMemObject(bClImg);
	clReleaseMemObject(cClImg);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	return time.count();
}