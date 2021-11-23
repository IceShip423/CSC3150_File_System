#ifndef FILE_SYSTEM_H
#define FILE_SYSTEM_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include <cstdio>

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
	char filename[20]; // 20B
	u16 size; // 2B 
	u16 staring_block; // 2B
	u16 FCB_idx; // 2B
	u16 created_time; // 2B
	u16 modified_time; // 2B
	u8 open_mode; // 1B
	u8 allocated_blocks; // 1B
};

struct BitMap {
	u32 data[1024];
	inline __device__ bool is_free(u32 bit_idx)
	{
		return (data[bit_idx / 32] >> (bit_idx % 32)) & 1;// 1 is free
	}
	inline __device__ void set_free(u32 bit_idx)
	{
		data[bit_idx / 32] |= 1 << (bit_idx % 32);
	}
	inline __device__ void set_allocated(u32 bit_idx)
	{
		data[bit_idx / 32] &= ~(1 << (bit_idx % 32));
	}
	__device__ void init()
	{
		for (int i = 0; i < 1024; ++i)
		{
			data[i] = 0xFFFFFFFF;
		}
	}
	__device__ u32 FindFree(u32 start_bit_idx)
	{
		u32 i;
		for (i = start_bit_idx; i < 32768; ++i)
		{
			if (is_free(i)) break;
		}
		return i;
	}
	__device__ u32 FindAllocated(u32 start_bit_idx)
	{
		u32 i;
		for (i = start_bit_idx; i < 32768; ++i)
		{
			if (!is_free(i)) break;
		}
		return i;
#undef t_BLOCK_NUM
	}
};

struct FileSystem {
	uchar* volume;
	BitMap* bitmap;
	FCB* fcb[1024];

	int SUPERBLOCK_SIZE;
	int PER_FCB_SIZE;
	int FCB_ENTRIES;
	int VOLUME_SIZE;
	int PER_STORAGE_BLOCK_SIZE;
	int MAX_PER_FILENAME_SIZE;
	int MAX_FILE_NUM;
	int DATA_BLOCK_SIZE;
	int DATA_BLOCK_VOLUME_OFFSET;
	int DATA_BLOCK_NUM;
};




__device__ void fs_init(FileSystem* fs, uchar* volume, int SUPERBLOCK_SIZE,
	int PER_FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
	int PER_STORAGE_BLOCK_SIZE, int MAX_PER_FILENAME_SIZE,
	int MAX_FILE_NUM, int DATA_BLOCK_SIZE, int DATA_BLOCK_VOLUME_OFFSET, int DATA_BLOCK_NUM);

__device__ u32 fs_open(FileSystem* fs, char* s, int op);
__device__ void fs_read(FileSystem* fs, uchar* output, u32 size, u32 fp);
__device__ u32 fs_write(FileSystem* fs, uchar* input, u32 size, u32 fp);
__device__ void fs_gsys(FileSystem* fs, int op);
__device__ void fs_gsys(FileSystem* fs, int op, char* s);

__device__ void compact(FileSystem* fs);
__device__ void show_FCB( FCB* t_FCB);

#endif
