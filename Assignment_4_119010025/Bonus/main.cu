#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#define DATAFILE "./data.bin"
#define OUTFILE "./snapshot.bin"

#define VOLUME_SIZE 1087488 //4096+34816+1048576=1062KB
#define SUPERBLOCK_SIZE 4096 //32K/8 bits = 4 K
#define DATA_BLOCK_SIZE 1048576

#define FCB_ENTRIES 1024
#define DATA_BLOCK_NUM 32768

#define PER_FCB_SIZE 34 //34 bytes per FCB
#define PER_STORAGE_BLOCK_SIZE 32

#define MAX_PER_FILENAME_SIZE 20
#define MAX_FILE_NUM 1024

#define DATA_BLOCK_VOLUME_OFFSET 38912 //4096+34816

#define DIR_STRUCTURE_SIZE 4096  // additional directory structure space

// data input and output
__device__ __managed__ uchar input[DATA_BLOCK_SIZE];
__device__ __managed__ uchar output[DATA_BLOCK_SIZE];

// volume (disk storage)
__device__ __managed__ uchar volume[VOLUME_SIZE+DIR_STRUCTURE_SIZE];


__device__ void user_program(FileSystem* fs, uchar* input, uchar* output);


__global__ void mykernel(uchar* input, uchar* output) {

	// Initilize the file system
	FCB root_FCB;
	STACK stack1;
	FileSystem fs;
	fs_init(&fs, volume, &root_FCB, &stack1, SUPERBLOCK_SIZE, PER_FCB_SIZE, FCB_ENTRIES,
		VOLUME_SIZE, PER_STORAGE_BLOCK_SIZE, MAX_PER_FILENAME_SIZE,
		MAX_FILE_NUM, DATA_BLOCK_SIZE, DATA_BLOCK_VOLUME_OFFSET, DATA_BLOCK_NUM);

	// user program the access pattern for testing file operations
	user_program(&fs, input, output);
}

__host__ void write_binaryFile(char* fileName, void* buffer, int bufferSize)
{
	FILE* fp;
	fp = fopen(fileName, "wb");
	fwrite(buffer, 1, bufferSize, fp);
	fclose(fp);
}

__host__ int load_binaryFile(char* fileName, void* buffer, int bufferSize)
{
	FILE* fp;
	fp = fopen(fileName, "rb");

	if (!fp)
	{
		printf("***Unable to open file %s***\n", fileName);
		exit(1);
	}

	//Get file length
	fseek(fp, 0, SEEK_END);
	int fileLen = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	if (fileLen > bufferSize)
	{
		printf("****invalid testcase!!****\n");
		printf("****software warrning: the file: %s size****\n", fileName);
		printf("****is greater than buffer size****\n");
		exit(1);
	}

	//Read file contents into buffer
	fread(buffer, fileLen, 1, fp);
	fclose(fp);
	return fileLen;
}

int main() {
	cudaError_t cudaStatus;
	load_binaryFile(DATAFILE, input, DATA_BLOCK_SIZE);

	// Launch to GPU kernel with single thread
	cudaDeviceSetLimit(cudaLimitStackSize, 62768);
	mykernel << <1, 1 >> > (input, output);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "mykernel launch failed: %s\n",
			cudaGetErrorString(cudaStatus));
		return 0;
	}

	cudaDeviceSynchronize();
	cudaDeviceReset();

	write_binaryFile(OUTFILE, output, DATA_BLOCK_SIZE);
	

	return 0;
}
