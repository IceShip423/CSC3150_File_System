#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <string.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>


__device__ __managed__ u32 gtime = 0;

__device__  void RESET_FCB(FCB* t_FCB);
__device__ bool my_strcmp(const char* x, const char* y);


#pragma region Other Methods

__device__ void fs_init(FileSystem* fs, uchar* volume, int SUPERBLOCK_SIZE,
	int PER_FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
	int PER_STORAGE_BLOCK_SIZE, int MAX_PER_FILENAME_SIZE,
	int MAX_FILE_NUM, int DATA_BLOCK_SIZE, int DATA_BLOCK_VOLUME_OFFSET, int DATA_BLOCK_NUM)
{
	// init constants
	fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
	fs->PER_FCB_SIZE = PER_FCB_SIZE;
	fs->FCB_ENTRIES = FCB_ENTRIES;
	fs->VOLUME_SIZE = VOLUME_SIZE;
	fs->PER_STORAGE_BLOCK_SIZE = PER_STORAGE_BLOCK_SIZE;
	fs->MAX_PER_FILENAME_SIZE = MAX_PER_FILENAME_SIZE;
	fs->MAX_FILE_NUM = MAX_FILE_NUM;
	fs->DATA_BLOCK_SIZE = DATA_BLOCK_SIZE;
	fs->DATA_BLOCK_VOLUME_OFFSET = DATA_BLOCK_VOLUME_OFFSET;
	fs->DATA_BLOCK_NUM = DATA_BLOCK_NUM;

	// init variables
	fs->volume = volume;
	fs->bitmap = (BitMap*)volume;
	fs->bitmap->init();
	for (int i = 0; i < FCB_ENTRIES; ++i)
	{
		fs->fcb[i] = (FCB*)(volume + SUPERBLOCK_SIZE + i * PER_FCB_SIZE);
		fs->fcb[i]->FCB_idx = i;
		RESET_FCB(fs->fcb[i]);
	}
	assert(((uchar*)(&fs->fcb[FCB_ENTRIES - 1]->allocated_blocks) - volume) == SUPERBLOCK_SIZE + PER_FCB_SIZE * FCB_ENTRIES - 1); // struct correctness
	assert(sizeof(*(fs->fcb[0])) == 32);// struct correctness
}

__device__ FCB* FindFile(FileSystem* fs, char* s)
{
	for (int i = 0; i < fs->FCB_ENTRIES; ++i)
	{
		if (my_strcmp(fs->fcb[i]->filename, s))
		{
			return fs->fcb[i];
		}
	}
	return NULL;
}

inline __device__ uchar* DataBlockIdx_ptr(FileSystem* fs, u32 block_idx)
{
	return fs->volume + (fs->DATA_BLOCK_VOLUME_OFFSET + block_idx * fs->PER_STORAGE_BLOCK_SIZE);
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

__device__ void my_strcpy(char* dest, const char* src) {
	int i = 0;
	do {
		dest[i] = src[i];
	} while (src[i++] != 0);
}

__device__ bool my_strcmp(const char* x, const char* y)
{
	while (*x != 0 && *y != 0)
	{
		if (*x != *y) return false;
		x++;
		y++;
	}
	return *x == 0 && *y == 0;
}

#pragma endregion

#pragma region Space Free

inline __device__ void DataBlock_Clean(FileSystem* fs, u32 block_idx)
{
	my_memclean(DataBlockIdx_ptr(fs, block_idx), fs->PER_STORAGE_BLOCK_SIZE);
}

__device__ void Free_DataBlock_BitMap(FileSystem* fs, FCB* t_FCB)
{
	for (u32 idx = t_FCB->staring_block; idx < t_FCB->staring_block + t_FCB->allocated_blocks; ++idx)
	{
		DataBlock_Clean(fs, idx);
		fs->bitmap->set_free(idx);
	}
}

__device__ void RESET_FCB(FCB* t_FCB)
{
	t_FCB->allocated_blocks = 0;
	my_strcpy(t_FCB->filename, "");
	t_FCB->created_time = 0;
	t_FCB->modified_time = 0;
	t_FCB->open_mode = G_READ;
	t_FCB->size = 0;
	t_FCB->staring_block = 0;
}

__host__ __device__ bool FCB_start_block_cmp(const FCB* o1, const FCB* o2)
{
	return o1->size > o2->size;
}

__device__ void Block_Migrate(FileSystem* fs, u16 dst_block_idx, u16 src_block_idx)
{
	if (dst_block_idx == src_block_idx) return;
	uchar* dst_add = DataBlockIdx_ptr(fs, dst_block_idx);
	uchar* src_add = DataBlockIdx_ptr(fs, src_block_idx);
	my_memcpy(dst_add, src_add, fs->DATA_BLOCK_SIZE);
	fs->bitmap->set_allocated(dst_block_idx);
	fs->bitmap->set_free(src_block_idx);
}

__device__ void compact(FileSystem* fs)
{
	printf("[log] Do compact\n");
	FCB* obs[1024];
	int cnt = 0;
	for (int i = 0; i < fs->FCB_ENTRIES; ++i)
	{
		if (!my_strcmp(fs->fcb[i]->filename, ""))
		{
			obs[cnt++] = (fs->fcb[i]);
		}
	}
	thrust::sort(obs, obs + cnt, FCB_start_block_cmp);
	u16 st_block = 0;
	for (int i = 0; i < cnt; ++i)
	{
		FCB* t_FCB = obs[i];
		for (int j = 0; j < t_FCB->allocated_blocks; j++)
		{
			Block_Migrate(fs, st_block + j, t_FCB->staring_block + j);
		}
		t_FCB->staring_block = st_block;
		st_block += t_FCB->allocated_blocks;
	}
}

#pragma endregion

#pragma region Space Allocation

__device__ void allocate_blocks(FileSystem* fs, u32 start_block, u32 blocknum)
{
	for (u32 i = 0; i < blocknum; ++i)
	{
		fs->bitmap->set_allocated(start_block + i);
	}
}

__device__ u32 FindFreeBlock(FileSystem* fs, u32 block_num)
{
	for (int i = 0; i < fs->DATA_BLOCK_NUM;)
	{
		u32 start_block = fs->bitmap->FindFree(i);
		u32 end_block = fs->bitmap->FindAllocated(start_block);
		if (end_block - start_block > block_num)
		{
			allocate_blocks(fs, start_block, block_num);
			return start_block;
		}
		i = end_block + 1;
	}
	compact(fs);
	u32 start_block = fs->bitmap->FindFree(0);
	u32 end_block = fs->bitmap->FindAllocated(start_block);
	assert(end_block - start_block >= block_num);
	allocate_blocks(fs, start_block, block_num);
	return start_block;
}

__device__ u32 FindFreeFCB(FileSystem* fs)
{
	for (int i = 0; i < fs->FCB_ENTRIES; ++i)
	{
		if (my_strcmp(fs->fcb[i]->filename, ""))
		{
			return i;
		}
	}
	assert(false);// find FCB error
	return 0;
}

#pragma endregion

#pragma region Debug

inline __device__ void show_FCB(FCB* t_FCB)
{
	printf("[FCB] %s [b]%d  [t]%d  [m]%d  [sz]%d  [st]%d\n", t_FCB->filename, t_FCB->allocated_blocks, t_FCB->modified_time, t_FCB->open_mode, t_FCB->size, t_FCB->staring_block);
}

#pragma endregion

#pragma region User APIs

__device__ void fs_close(FileSystem* fs, u32 fp)
{
	FCB* t_FCB = fs->fcb[fp];
	t_FCB->open_mode = G_READ;
	// get upper bound
	u32 block_num_used = (t_FCB->size + fs->PER_STORAGE_BLOCK_SIZE - 1) / fs->PER_STORAGE_BLOCK_SIZE;
	for (int i = block_num_used; i < t_FCB->allocated_blocks; ++i)
	{
		fs->bitmap->set_free(t_FCB->staring_block + i);
	}
	t_FCB->allocated_blocks = block_num_used;
}

__device__ u32 fs_open(FileSystem* fs, char* s, int op)
{
	gtime++;
	FCB* t_FCB = FindFile(fs, s);
	if (t_FCB == NULL)
	{
		assert(op == G_WRITE);
		// allocate new position
		u32 block_num_wanted = 1024 / fs->PER_STORAGE_BLOCK_SIZE;
		u32 start_block = FindFreeBlock(fs, block_num_wanted);
		u32 t_FCB_idx = FindFreeFCB(fs);
		t_FCB = fs->fcb[t_FCB_idx];
		my_strcpy(t_FCB->filename, s);
		t_FCB->size = 0;
		t_FCB->staring_block = start_block;
		t_FCB->created_time = gtime;
		t_FCB->modified_time = gtime;
		t_FCB->allocated_blocks = block_num_wanted;
	}
	t_FCB->open_mode = op;
	return t_FCB->FCB_idx;
}


__device__ void fs_read(FileSystem* fs, uchar* output, u32 size, u32 fp)
{
	gtime++;
	FCB* t_FCB = fs->fcb[fp];
	assert(t_FCB->open_mode == G_READ);
	uchar* starting_address = DataBlockIdx_ptr(fs, t_FCB->staring_block);
	my_memcpy(output, starting_address, size);
}

__device__ u32 fs_write(FileSystem* fs, uchar* input, u32 size, u32 fp)
{
	gtime++;
	FCB* t_FCB = fs->fcb[fp];
	assert(t_FCB->open_mode == G_WRITE);
	uchar* starting_address = DataBlockIdx_ptr(fs, t_FCB->staring_block);
	my_memcpy(starting_address, input, size);
	t_FCB->size = size;
	t_FCB->modified_time = gtime;
	fs_close(fs, fp); // close file after write
	return fp;
}

inline __host__ __device__ bool FCB_modified_time_cmp(const FCB* o1, const FCB* o2)
{
	return (o1->modified_time == o2->modified_time) ? (o1->created_time < o2->created_time) : (o1->modified_time > o2->modified_time);
}

inline __host__ __device__ bool FCB_size_cmp(const FCB* o1, const FCB* o2)
{
	return (o1->size == o2->size) ? (o1->created_time < o2->created_time) : (o1->size > o2->size);
}


__device__ void fs_gsys(FileSystem* fs, int op)
{
	gtime++;
	FCB* obs[1024];
	int cnt = 0;
	for (int i = 0; i < fs->FCB_ENTRIES; ++i)
	{
		if (!my_strcmp(fs->fcb[i]->filename, ""))
		{
			obs[cnt++] = (fs->fcb[i]);
		}
	}
	if (op == LS_D) // order by modified time
	{
		thrust::sort(obs, obs + cnt, FCB_modified_time_cmp);
		printf("===sort by modified time===\n");
		for (int i = 0; i < cnt; ++i)
		{
			printf("%s\n", obs[i]->filename);
		}
	}
	else if (op == LS_S) // order by size
	{
		thrust::sort(obs, obs + cnt, FCB_size_cmp);
		printf("===sort by size===\n");
		for (int i = 0; i < cnt; ++i)
		{
			printf("%s %d\n", obs[i]->filename, obs[i]->size);
		}
	}

}

__device__ void fs_gsys(FileSystem* fs, int op, char* s) // RM
{
	gtime++;
	FCB* t_FCB = FindFile(fs, s);
	assert(t_FCB != NULL);
	Free_DataBlock_BitMap(fs, t_FCB);
	RESET_FCB(t_FCB);
}

#pragma endregion
