#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__device__ __managed__ u32 gtime = 0;

void RESET_FCB(FCB* t_FCB);

#pragma region Other Methods

__device__ void fs_init(FileSystem* fs, uchar* volume, int SUPERBLOCK_SIZE,
	int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
	int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE,
	int MAX_FILE_NUM, int MAX_FILE_SIZE_TOT, int FILE_BASE_ADDRESS)
{
	// init variables
	fs->volume = volume;
	fs->bitmap = (BitMap*)volume;
	fs->bitmap->init();
	for (int i = 0; i < FCB_ENTRIES; ++i)
	{
		fs->fcb[i] = (FCB*)(volume + SUPERBLOCK_SIZE + i * FCB_SIZE);
		RESET_FCB(fs->fcb[i]);
	}

	// init constants
	fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
	fs->FCB_SIZE = FCB_SIZE;
	fs->FCB_ENTRIES = FCB_ENTRIES;
	fs->VOLUME_SIZE = VOLUME_SIZE;
	fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
	fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
	fs->MAX_FILE_NUM = MAX_FILE_NUM;
	fs->MAX_FILE_SIZE_TOT = MAX_FILE_SIZE_TOT;
	fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;

}

__device__ FCB* FindFile(FileSystem* fs, char* s)
{
	for (int i = 0; i < fs->FCB_ENTRIES; ++i)
	{
		if (fs->fcb[i]->filename == s)
		{
			return fs->fcb[i];
		}
	}
	return NULL;
}

__device__ uchar* DataBlockIdx_ptr(FileSystem* fs, u32 block_idx)
{
	return fs->volume + fs->FILE_BASE_ADDRESS + block_idx * fs->STORAGE_BLOCK_SIZE;
}


__device__ void my_memcpy(uchar* dst, uchar* src, size_t count)
{
	for (int i = 0; i < count; ++i)
	{
		dst[i] = src[i];
	}
}

__device__ void my_memclean(uchar* dst, size_t count)
{
	for (int i = 0; i < count; ++i)
	{
		dst[i] = 0;
	}
}

inline __device__ void DataBlock_Clean(FileSystem* fs, u32 block_idx)
{
	my_memclean(DataBlockIdx_ptr(fs, block_idx), fs->STORAGE_BLOCK_SIZE);
}

__device__ u32 fs_close(FileSystem* fs, u32 fp)
{
	FCB* t_FCB = (FCB*)fp;
	t_FCB->open_mode = G_READ;
	// get upper bound
	u32 block_num_used = (t_FCB->size + fs->STORAGE_BLOCK_SIZE + 1) / fs->STORAGE_BLOCK_SIZE;
	for (int i = block_num_used; i < t_FCB->allocated_blocks; ++i)
	{
		fs->bitmap->set_empty(t_FCB->staring_block + i);
	}
}

__device__ char* my_strcpy(char* dest, const char* src) {
	int i = 0;
	do {
		dest[i] = src[i];
	} while (src[i++] != 0);
	return dest;
}

#pragma endregion

#pragma region Space Free

__device__ void Free_DataBlock_BitMap(FileSystem* fs, FCB* t_FCB)
{
	for (u32 idx = t_FCB->staring_block; idx < t_FCB->staring_block + t_FCB->allocated_blocks; ++idx)
	{
		DataBlock_Clean(fs, idx);
		fs->bitmap->set_empty(idx);
	}
}

__device__ void RESET_FCB(FCB* t_FCB)
{
	t_FCB->allocated_blocks = 0;
	strcpy(t_FCB->filename, "");
	t_FCB->modified_time = 0;
	t_FCB->open_mode = G_READ;
	t_FCB->size = 0;
	t_FCB->staring_block = 0;
}

#pragma endregion

#pragma region Free Space Management

__device__ void compact(FileSystem* fs)
{
	// do something
}


__device__ void allocate_blocks(FileSystem* fs, u32 start_block, u32 blocknum)
{
	for (u32 i = 0; i < blocknum; ++i)
	{
		fs->bitmap->set_allocated(start_block + i);
	}
}

#pragma endregion

#pragma region Space Allocation

#define MAX_BLOCK_NUM 2<<15
__device__ u32 FindFreeBlock(FileSystem* fs, u32 block_num)
{
	for (int i = 0; i < MAX_BLOCK_NUM;)
	{
		u32 start_block = fs->bitmap->FindFree(i);
		u32 end_block = fs->bitmap->FindAllocated(start_block);
		if (end_block - start_block > block_num)
		{
			allocate_blocks(fs, start_block, block_num);
			return start_block;
		}
		i = end_block;
	}
	compact(fs);
	u32 start_block = fs->bitmap->FindFree(0);
	u32 end_block = fs->bitmap->FindAllocated(start_block);
	if (end_block - start_block > block_num)
	{
		allocate_blocks(fs, start_block, block_num);
		return start_block;
	}
	// error
	printf("find free space error\n");
}
#undef MAX_BLOCK_NUM

__device__ u32 FindFreeFCB(FileSystem* fs)
{
	for (int i = 0; i < fs->FCB_ENTRIES; ++i)
	{
		if (fs->fcb[i]->filename == "")
		{
			return i;
		}
	}
	// error
	printf("find FCB error\n");
	return 0;
}


#pragma endregion



#pragma region User APIs

__device__ u32 fs_open(FileSystem* fs, char* s, int op)
{
	gtime++;
	FCB* t_FCB = FindFile(fs, s);
	if (t_FCB == NULL)
	{
		if (op == G_READ)
		{
			// error
			printf("read from nonexisting file\n");
			return 0;
		}
		// allocate new position
		u32 block_num_wanted = 1024 / fs->STORAGE_BLOCK_SIZE;
		u32 start_block = FindFreeBlock(fs, block_num_wanted);
		u32 t_FCB_idx = FindFreeFCB(fs);
		t_FCB = fs->fcb[t_FCB_idx];
		my_strcpy(t_FCB->filename, s);
		t_FCB->size = 0;
		t_FCB->staring_block = start_block;
		t_FCB->modified_time = gtime;
		t_FCB->allocated_blocks = block_num_wanted;
	}
	t_FCB->open_mode = op;
	return (u32)t_FCB;
}


__device__ void fs_read(FileSystem* fs, uchar* output, u32 size, u32 fp)
{
	gtime++;
	FCB* t_FCB = (FCB*)fp;
	if (t_FCB->open_mode != G_READ)
	{
		// error
		printf("read in write mode error\n");
		return;
	}
	uchar* starting_address = DataBlockIdx_ptr(fs, t_FCB->staring_block);
	my_memcpy(output, starting_address, size);
	fs_close(fs, fp); // close file after write
}

__device__ u32 fs_write(FileSystem* fs, uchar* input, u32 size, u32 fp)
{
	gtime++;
	FCB* t_FCB = (FCB*)fp;
	if (t_FCB->open_mode != G_WRITE)
	{
		// error
		printf("write in read mode error\n");
		return 0;
	}
	uchar* starting_address = DataBlockIdx_ptr(fs, t_FCB->staring_block);
	//1212
	my_memcpy(starting_address, input, size);
	t_FCB->size = size;
	t_FCB->modified_time = gtime;
	return fp;
}

__device__ void fs_gsys(FileSystem* fs, int op)
{
	gtime++;
	if (op == LS_D) // order by modified time
	{
		printf("=====Sort by modified time=======\n");
		for (int i = 0; i < fs->FCB_ENTRIES; ++i)
		{
			if (fs->fcb[i]->filename != "")
			{
				printf("%s  %d\n", fs->fcb[i]->filename, fs->fcb[i]->modified_time);
			}
		}
		printf("=====END=======\n");
	}
	else if (op == LS_S) // order by size
	{
		printf("=======Sort by size=======\n");
		for (int i = 0; i < fs->FCB_ENTRIES; ++i)
		{
			if (fs->fcb[i]->filename != "")
			{
				printf("%s  %d\n", fs->fcb[i]->filename, fs->fcb[i]->size);
			}
		}
		printf("=====END=======\n");
	}

}

__device__ void fs_gsys(FileSystem* fs, int op, char* s) // RM
{
	gtime++;
	FCB* t_FCB;
	if ((t_FCB = FindFile(fs, s)) == NULL)
	{
		// error
		printf("rm nonexisting file error\n");
		return;
	}
	Free_DataBlock_BitMap(fs, t_FCB);
	RESET_FCB(t_FCB);
}

#pragma endregion
