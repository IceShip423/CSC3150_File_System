#ifndef VIRTUAL_MEMORY_H
#define VIRTUAL_MEMORY_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>

typedef unsigned char uchar;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;

#define G_WRITE 1
#define G_READ 0

#define LS_D 0
#define LS_S 1
#define RM 2
#define MKDIR 3
#define CD 4
#define PWD 5
#define CD_P 6
#define RM_RF 7

struct FCB { // 32B
	char filename[20];
	u32 staring_block;
	u32 size; // in byte
	u16 modified_time;
	u8 open_mode;
	u8 allocated_blocks;
};

struct BitMap {
	u32 data[1024]; 
	inline __device__ bool is_free(u32 bitnum)
	{
		return data[bitnum / 32] >> (bitnum % 32) & 1;// 1 is free
	}
	inline __device__ bool set_empty(u32 bitnum)
	{
		data[bitnum / 32] |= 1 << (bitnum % 32);
	}
	inline __device__ bool set_allocated(u32 bitnum)
	{
		data[bitnum / 32] &= ~(1 << (bitnum % 32));
	}
	__device__ void init()
	{
		for (int i = 0; i < 1024; ++i)
		{
			data[i] = 0xFFFFFFFF;
		}
	}
	__device__ u32 FindFree(u32 start_bitnum)
	{
#define MAX_BLOCK_NUM 2<<15
		u32 i;
		for (i = start_bitnum; i < MAX_BLOCK_NUM; ++i)
		{
			if (is_free(i)) break;
		}
		return i;
	}
	__device__ u32 FindAllocated(u32 start_bitnum)
	{
		u32 i;
		for (i = start_bitnum; i < MAX_BLOCK_NUM; ++i)
		{
			if (!is_free(i)) break;
		}
		return i;
#undef MAX_BLOCK_NUM
	}
};

struct FileSystem {
	uchar* volume;
	BitMap* bitmap;
	FCB* fcb[1024];

	int SUPERBLOCK_SIZE;
	int FCB_SIZE;
	int FCB_ENTRIES;
	int VOLUME_SIZE;
	int STORAGE_BLOCK_SIZE;
	int MAX_FILENAME_SIZE;
	int MAX_FILE_NUM;
	int MAX_FILE_SIZE_TOT;
	int FILE_BASE_ADDRESS;
};




__device__ void fs_init(FileSystem* fs, uchar* volume, int SUPERBLOCK_SIZE,
	int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
	int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE,
	int MAX_FILE_NUM, int MAX_FILE_SIZE_TOT, int FILE_BASE_ADDRESS);

__device__ u32 fs_open(FileSystem* fs, char* s, int op);
__device__ void fs_read(FileSystem* fs, uchar* output, u32 size, u32 fp);
__device__ u32 fs_write(FileSystem* fs, uchar* input, u32 size, u32 fp);
__device__ void fs_gsys(FileSystem* fs, int op);
__device__ void fs_gsys(FileSystem* fs, int op, char* s);

__device__ void user_program(FileSystem* fs, uchar* input, uchar* output);
__device__ void user_program_b(FileSystem* fs, uchar* input, uchar* output);

#endif
