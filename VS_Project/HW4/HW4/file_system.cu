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

#define FCB_IDX_NULL 0xFFFF
#define EDGE_IDX_NULL 0xFFFF

#pragma region Other Methods

__device__ void fs_init(FileSystem* fs, uchar* volume, FCB* root_FCB,STACK* FCB_stack, int SUPERBLOCK_SIZE,
	int PER_FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
	int PER_STORAGE_BLOCK_SIZE, int MAX_PER_FILENAME_SIZE,
	int MAX_FILE_NUM, int DATA_BLOCK_SIZE, int DATA_BLOCK_VOLUME_OFFSET, int DATA_BLOCK_NUM)
{
	printf("gaga");
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

	printf("gaga");
	// init variables
	fs->volume = volume;
	fs->bitmap = (BitMap*)volume;
	fs->bitmap->init();
	int per_edge_size = sizeof(EDGE);
	//printf("per_edge_size: %d\n", per_edge_size);
	for (int i = 0; i < FCB_ENTRIES; ++i) // max edge entries = max FCB entries
	{
		// FCB
		fs->fcb[i] = (FCB*)(volume + SUPERBLOCK_SIZE + i * PER_FCB_SIZE);
		fs->fcb[i]->FCB_idx = i;
		RESET_FCB(fs->fcb[i]);
		// EDGE
		fs->edge[i] = (EDGE*)(volume + VOLUME_SIZE + i * per_edge_size);
		fs->edge[i]->FCB_idx = FCB_IDX_NULL;
		fs->edge[i]->next_edge = EDGE_IDX_NULL;
	}
	printf("gaga");
	// root
	fs->root_FCB = root_FCB;
	RESET_FCB(fs->root_FCB);
	fs->root_FCB->open_mode = DIR;
	fs->FCB_stack = FCB_stack;
	fs->FCB_stack->cnt = 0;
	fs->FCB_stack->push(fs->root_FCB);
	// struct correctness
	//printf("%lld\n", sizeof(*(fs->fcb[0])));
	assert(sizeof(*(fs->fcb[0])) == 32);
	assert(sizeof(*(fs->edge[0])) == 4);

}

__device__ FCB* FindFile(FileSystem* fs, char* s, FCB* t_dir)
{
	assert(t_dir->open_mode == DIR);
	for (u16 edge_idx = t_dir->first_edge_idx; edge_idx != EDGE_IDX_NULL; edge_idx = fs->edge[edge_idx]->next_edge)
	{
		printf("3434[%d]",edge_idx);
		FCB* child_FCB = fs->fcb[fs->edge[edge_idx]->FCB_idx];
		if (my_strcmp(child_FCB->filename, s))
		{
			return child_FCB;
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

__device__ bool my_strcmp(const char* x, const char* y) //a mutable pointer to an immutable character
{
	while (*x != 0 && *y != 0)
	{
		if (*x != *y) return false;
		x++;
		y++;
	}
	return *x == 0 && *y == 0;
}

__device__ u16 my_strlen(const char* x)
{
	u16 len = 0;
	while (*(x++) != 0)len++;
	return len;
}

#pragma endregion

#pragma region Space Free

inline __device__ void DataBlock_Clean(FileSystem* fs, u32 block_idx)
{
	my_memclean(DataBlockIdx_ptr(fs, block_idx), fs->PER_STORAGE_BLOCK_SIZE);
}

__device__ void Free_DataBlock_BitMap(FileSystem* fs, FCB* t_FCB)
{
	for (u32 idx = t_FCB->starting_block; idx < t_FCB->starting_block + t_FCB->allocated_blocks; ++idx)
	{
		DataBlock_Clean(fs, idx);
		fs->bitmap->set_free(idx);
	}
}

__device__ void delete_in_directory(FileSystem* fs, FCB* parent_FCB, u16 FCB_idx)
{
	if (fs->edge[parent_FCB->first_edge_idx]->FCB_idx == FCB_idx)
	{
		u16 edge_idx = parent_FCB->first_edge_idx;
		parent_FCB->first_edge_idx = fs->edge[edge_idx]->next_edge;
		fs->edge[edge_idx]->FCB_idx = FCB_IDX_NULL;
		fs->edge[edge_idx]->next_edge = EDGE_IDX_NULL;
		return;
	}
	u16 previous_edge_idx = parent_FCB->first_edge_idx;
	for (u16 edge_idx = fs->edge[previous_edge_idx]->next_edge; edge_idx != EDGE_IDX_NULL; edge_idx = fs->edge[edge_idx]->next_edge)
	{
		if (fs->edge[edge_idx]->FCB_idx == FCB_idx)
		{
			fs->edge[previous_edge_idx]->next_edge = fs->edge[edge_idx]->next_edge;
			fs->edge[edge_idx]->FCB_idx = FCB_IDX_NULL;
			fs->edge[edge_idx]->next_edge = EDGE_IDX_NULL;
			return;
		}
		previous_edge_idx = edge_idx;
	}
	assert(false);
}

__device__ void RESET_FCB(FCB* t_FCB)
{
	t_FCB->allocated_blocks = 0;
	my_strcpy(t_FCB->filename, "");
	t_FCB->modified_time = 0;
	t_FCB->open_mode = G_READ;
	t_FCB->size = 0;
	t_FCB->starting_block = 0;
	t_FCB->first_edge_idx = EDGE_IDX_NULL;
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
			Block_Migrate(fs, st_block + j, t_FCB->starting_block + j);
		}
		t_FCB->starting_block = st_block;
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

#pragma region Directory

__device__ u16 find_empty_edge(FileSystem* fs)
{
	for (int i = 0; i < fs->FCB_ENTRIES; ++i)
	{
		if (fs->edge[i]->FCB_idx == FCB_IDX_NULL)
		{
			return i;
		}
	}
	assert(false);  // error
	return 0;
}

__device__ void dir_add_edge(FileSystem* fs, FCB* parent_dir, FCB* child)
{
	assert(parent_dir->open_mode == DIR);
	u16 edge_idx = find_empty_edge(fs);
	fs->edge[edge_idx]->FCB_idx = child->FCB_idx;
	fs->edge[edge_idx]->next_edge = parent_dir->first_edge_idx;
	parent_dir->first_edge_idx = edge_idx;
}

__device__ u16 calculate_directory_size(FileSystem* fs, FCB* t_dir)
{
	assert(t_dir->open_mode == DIR);
	u16 size = 0;
	for (u16 edge_idx = t_dir->first_edge_idx; edge_idx != EDGE_IDX_NULL; edge_idx = fs->edge[edge_idx]->next_edge)
	{
		FCB* child_FCB = fs->fcb[fs->edge[edge_idx]->FCB_idx];
		size += my_strlen(child_FCB->filename) + 1; // include '\0'
	}
	return size;
}

__device__ void remove_file(FileSystem* fs, FCB* t_FCB)
{
	assert(t_FCB != NULL);
	assert(t_FCB->open_mode == G_READ || t_FCB->open_mode == G_WRITE);
	Free_DataBlock_BitMap(fs, t_FCB);
	RESET_FCB(t_FCB);
	delete_in_directory(fs, fs->FCB_stack->top(), t_FCB->FCB_idx);
}

__device__ void remove_dir(FileSystem* fs, FCB* t_FCB)
{
	assert(t_FCB != NULL);
	assert(t_FCB->open_mode == DIR);
	for (u16 edge_idx = t_FCB->first_edge_idx; edge_idx != EDGE_IDX_NULL; edge_idx = fs->edge[edge_idx]->next_edge)
	{
		FCB* child_FCB = fs->fcb[fs->edge[edge_idx]->FCB_idx];
		if (child_FCB->open_mode == DIR)
		{
			remove_dir(fs, t_FCB);
		}
		else
		{
			remove_file(fs, t_FCB);
		}
	}
	RESET_FCB(t_FCB);
	delete_in_directory(fs, fs->FCB_stack->top(), t_FCB->FCB_idx);
}

#pragma endregion

#pragma region Debug

inline __device__ void show_FCB(FCB* t_FCB)
{
	printf("[FCB] %s [b]%d  [t]%d  [m]%d  [sz]%d  [st]%d\n", t_FCB->filename, t_FCB->allocated_blocks, t_FCB->modified_time, t_FCB->open_mode, t_FCB->size, t_FCB->starting_block);
}

#pragma endregion

#pragma region User APIs

__device__ u32 fs_close(FileSystem* fs, u32 fp)
{
	FCB* t_FCB = fs->fcb[fp];
	t_FCB->open_mode = G_READ;
	// get upper bound
	u32 block_num_used = (t_FCB->size + fs->PER_STORAGE_BLOCK_SIZE - 1) / fs->PER_STORAGE_BLOCK_SIZE;
	for (int i = block_num_used; i < t_FCB->allocated_blocks; ++i)
	{
		fs->bitmap->set_free(t_FCB->starting_block + i);
	}
	t_FCB->allocated_blocks = block_num_used;
}

__device__ u32 fs_open(FileSystem* fs, char* s, int op)
{
	FCB* t_FCB = FindFile(fs, s, fs->FCB_stack->top());
	printf("12121");
	if (t_FCB == NULL)
	{
		assert(op == G_WRITE);
		// allocate new position
		u32 block_num_wanted = 1024 / fs->PER_STORAGE_BLOCK_SIZE;
		u32 start_block = FindFreeBlock(fs, block_num_wanted);
		u32 t_FCB_idx = FindFreeFCB(fs);
		printf("[new]  start_block:%d  t_FCB_idx:%d\n", start_block, t_FCB_idx);
		t_FCB = fs->fcb[t_FCB_idx];
		my_strcpy(t_FCB->filename, s);
		t_FCB->size = 0;
		t_FCB->starting_block = start_block;
		t_FCB->modified_time = gtime++;
		t_FCB->allocated_blocks = block_num_wanted;
		t_FCB->first_edge_idx = EDGE_IDX_NULL;
		// directory
		dir_add_edge(fs, fs->FCB_stack->top(), t_FCB);
		fs->FCB_stack->top()->size = calculate_directory_size(fs, fs->FCB_stack->top());
		fs->FCB_stack->top()->modified_time = gtime++;
	}
	t_FCB->open_mode = op;
	//show_FCB(t_FCB);
	return t_FCB->FCB_idx;
}


__device__ void fs_read(FileSystem* fs, uchar* output, u32 size, u32 fp)
{
	FCB* t_FCB = fs->fcb[fp];
	assert(t_FCB->open_mode == G_READ);
	uchar* starting_address = DataBlockIdx_ptr(fs, t_FCB->starting_block);
	my_memcpy(output, starting_address, size);
	//show_FCB(t_FCB);
}

__device__ u32 fs_write(FileSystem* fs, uchar* input, u32 size, u32 fp)
{
	FCB* t_FCB = fs->fcb[fp];
	assert(t_FCB->open_mode == G_WRITE);
	uchar* starting_address = DataBlockIdx_ptr(fs, t_FCB->starting_block);
	my_memcpy(starting_address, input, size);
	t_FCB->size = size;
	t_FCB->modified_time = gtime++;
	fs_close(fs, fp); // close file after write
	//show_FCB(t_FCB);
	return fp;
}

inline __host__ __device__ bool FCB_modified_time_cmp(const FCB* o1, const FCB* o2)
{
	return o1->modified_time > o2->modified_time;
}

inline __host__ __device__ bool FCB_size_cmp(const FCB* o1, const FCB* o2)
{
	return o1->size > o2->size;
}


__device__ void fs_gsys(FileSystem* fs, int op)
{
	if (op == LS_D || op == LS_S)
	{
		FCB* obs[1024];
		int cnt = 0;
		for (u16 edge_idx = fs->FCB_stack->top()->first_edge_idx; edge_idx != EDGE_IDX_NULL; edge_idx = fs->edge[edge_idx]->next_edge)
		{
			FCB* child_FCB = fs->fcb[fs->edge[edge_idx]->FCB_idx];
			obs[cnt++] = child_FCB;
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
				printf("%s  %d\n", obs[i]->filename, obs[i]->size);
				// show_FCB(obs[i]);
			}
		}
	}
	else if (op == CD_P)
	{
		fs->FCB_stack->pop();
	}
	else if (op == PWD)
	{
		printf("CURRENT DIR: ");
		for (int i = 0; i < fs->FCB_stack->cnt; ++i)
		{
			printf("%s/", fs->FCB_stack->data[i]->filename);
		}
		printf("\n");
	}
}

__device__ void fs_gsys(FileSystem* fs, int op, char* s) // RM
{
	if (op == RM)
	{
		FCB* t_FCB = FindFile(fs, s, fs->FCB_stack->top());
		remove_file(fs, t_FCB);
		fs->FCB_stack->top()->size = calculate_directory_size(fs, fs->FCB_stack->top());
		fs->FCB_stack->top()->modified_time = gtime++;
	}
	else if (op == MKDIR)
	{
		// allocate new position
		u32 t_FCB_idx = FindFreeFCB(fs);
		FCB* t_FCB = fs->fcb[t_FCB_idx];
		my_strcpy(t_FCB->filename, s);
		t_FCB->size = 0;
		t_FCB->modified_time = gtime++;
		t_FCB->first_edge_idx = EDGE_IDX_NULL;
		t_FCB->open_mode = DIR;
		// directory
		dir_add_edge(fs, fs->FCB_stack->top(), t_FCB);
		fs->FCB_stack->top()->size = calculate_directory_size(fs, fs->FCB_stack->top());
		fs->FCB_stack->top()->modified_time = gtime++;
	}
	else if (op == CD)
	{
		FCB* t_FCB = FindFile(fs, s, fs->FCB_stack->top());
		assert(t_FCB != NULL);
		fs->FCB_stack->push(t_FCB);
	}
	else if (op == RM_RF)
	{
		FCB* t_FCB = FindFile(fs, s, fs->FCB_stack->top());
		remove_dir(fs, t_FCB);
		fs->FCB_stack->top()->size = calculate_directory_size(fs, fs->FCB_stack->top());
		fs->FCB_stack->top()->modified_time = gtime++;
	}
}

#pragma endregion
