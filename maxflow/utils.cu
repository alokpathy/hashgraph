#pragma once

#define WARP_SIZE 32
#include "allocator.h"
#include "matrix.h"
#include "cub/cub/cub.cuh"

#include <utility>

#include <fstream> //TODO remove

#define FORWARD 1
#define BACKWARD 0

#define N_MAX_BLOCKS 65534
#define WARP_SIZE 32

#include "config.h"

using cub::KeyValuePair;
typedef KeyValuePair<int,flow_t> kvpid;

//
//Custom atomic operations
//

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                        __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif


//TODO temp solution
#define MIN_NONZERO_V 0.00000001
__device__ __inline__ double isZeroAfterAtomicAdd(double * address, double val) {
	return ((atomicAdd(address, val) + val) <= MIN_NONZERO_V); //TODO temp solution
}

__device__ __inline__ bool isZero(double val) {
	return (val <= MIN_NONZERO_V); //+0.0 and -0.0, negative value not a pb
}

__device__ __inline__ double isZeroAfterAtomicAdd(uint32_t * address, uint32_t val) {
	return ((atomicAdd(address, val) + val) <= 0); //TODO temp solution
}

__device__ __inline__ bool isZero(uint32_t val) {
	return (val <= 0);
}

//TODO this is the version f->void
template<typename F, typename ...Args> 
__device__ void iterate_on_edges(int u, const csr_graph &g, F f, Args&&... args) {
	int start	    = g.row_offsets[u];
	int end 	    = g.row_offsets[u+1];
	int end_last_block   = start + blockDim.x * (((end-start) + blockDim.x - 1) / blockDim.x);

	for(int idx = start + blockDim.x * blockIdx.x + threadIdx.x;
	   	idx <= end_last_block;
		idx += blockDim.x * gridDim.x) {
		int i_edge 	= (idx < end) ? idx 			: -1;
		int v 		= (idx < end) ? g.col_indices[idx] 	: -1;  	
		
		f(u, v, i_edge, std::forward<Args>(args)...); //we should be able to early exit
	}	
}

//TODO this is actually the version f->void
template<typename F, typename ...Args>
__device__ void iterate_on_edges(int u, const csr_subgraph &sg, F f, Args&&... args) {
	
	int start	    = sg.edge_offsets[u];
	int end 	    = sg.edge_offsets[u+1];
	int end_last_block   = start + blockDim.x * (((end-start) + blockDim.x - 1) / blockDim.x);

	for(int idx = start + blockDim.x * blockIdx.x + threadIdx.x;
	   	idx <= end_last_block;
		idx += blockDim.x * gridDim.x) {
		int i_edge 	= (idx < end) ? sg.parent_edge_indices[idx] 	: -1;
		int v 		= (idx < end) ? sg.col_indices[idx] 		: -1;  	
		
		f(u, v, i_edge, std::forward<Args>(args)...); //we should be able to early exit
	}	
			

}


//TODO use std::function instead of template
template<typename F>
__global__ void apply_level_kernel(int *elements, int start, int end, F f) {
	for(int idx = start + blockIdx.z * blockDim.z + threadIdx.z;
		idx < end;
		idx += blockDim.z * gridDim.z) {
		int u = elements[idx];
		f(idx, u);
	}
}


template<typename F>
__global__ void apply_level_kernel(int start, int end, F f) {
	for(int u = start + blockIdx.z * blockDim.z + threadIdx.z;
		u < end;
		u += blockDim.z * gridDim.z) {
			
		f(u);
	}
}

//iterate through levels in graph
//TODO use std::function instead of template
//TODO overload sans q_bfs (just offsets)

template<int DIRECTION, typename F>
void iterate_on_levels(int start_level, 
				int end_level,
				int *elements,
				const int* levels_offset, 
				F f, 
				dim3 localGrid, 
				dim3 localBlock,
				int elts_per_thread=1,
				size_t shared_memory=0,
				cudaStream_t stream=0) {
	
	for(int level = start_level;
		((DIRECTION == FORWARD) && (level <= end_level))
		|| ((DIRECTION == BACKWARD) && (level >= end_level));
		level += (DIRECTION == FORWARD) ? 1 : -1) {
		int start = levels_offset[level];
		int end = levels_offset[level+1];
		int num_items = end - start;
	
		int nthreads_z = (num_items + elts_per_thread -1)/elts_per_thread;
		dim3 grid, block;
	
		block.x = localBlock.x;
		block.y = localBlock.y;
		block.z = min(512/block.x/block.y, 64);

		grid.x = localGrid.x;
		grid.y = localGrid.y;
		grid.z = min((nthreads_z + block.z - 1)/block.z, N_MAX_BLOCKS);
			
		//printf("level with elements : level : %i ; start : %i ; end : %i ; size : %i \n", level, start, end, num_items);
		apply_level_kernel<<<grid,block,shared_memory,stream>>>(elements, start, end, f);
	}
		
}

template<int DIRECTION, typename F>
__device__ void d_iterate_on_levels(const int start_level, 
				const int end_level,
				const int* levels_offset, 
				F f) {
	
	for(int level = start_level;
		((DIRECTION == FORWARD) && (level <= end_level))
		|| ((DIRECTION == BACKWARD) && (level >= end_level));
		level += (DIRECTION == FORWARD) ? 1 : -1) {
		int start = levels_offset[level];
		int end = levels_offset[level+1];

		//if(threadIdx.x == 0 && threadIdx.z == 0)
		//	printf("\n on device : level : %i ; start : %i ; end : %i \n", level, start, end);
		for(int u = start + blockIdx.z * blockDim.z + threadIdx.z;
			u < end;
			u += blockDim.z * gridDim.z) { 
			f(u);
		}

		__syncthreads(); 
	}
}


template<int DIRECTION, typename F>
__global__ void d_iterate_on_levels_kernel(const int start_level, 
				const int end_level,
				const int* levels_offset, 
				F f){
	d_iterate_on_levels<DIRECTION>(start_level, end_level, levels_offset, f);
}


template<int DIRECTION, typename F>
__host__ void iterate_on_levels(int start_level, 
				int end_level,
				const int* levels_offset, 
				F f, 
				dim3 localGrid, 
				dim3 localBlock,
				int elts_per_thread=4,
				size_t shared_memory=0,
				cudaStream_t stream=0,
				int max_level_width=INT_MAX //can be used to block-level sync
				) {
	dim3 block;
	
	block.x = localBlock.x;
	block.y = localBlock.y;
	block.z = min(256/block.x/block.y,64); //only 256, shared mem pb


	if(max_level_width <= block.z) {
		dim3 grid(1,1,1); 

		d_iterate_on_levels_kernel<DIRECTION><<<grid,block>>>(start_level, end_level, levels_offset, f);
			
		return;
	}	

	for(int level = start_level;
		((DIRECTION == FORWARD) && (level <= end_level))
		|| ((DIRECTION == BACKWARD) && (level >= end_level));
		level += (DIRECTION == FORWARD) ? 1 : -1) {
		int start = levels_offset[level];
		int end = levels_offset[level+1];
		int num_items = end - start;
	
		int nthreads_z = (num_items + elts_per_thread -1)/elts_per_thread;
		//printf("level : %i ; start : %i ; end : %i ; size : %i \n", level, start, end, num_items);
		dim3 grid;
		grid.x = localGrid.x;
		grid.y = localGrid.y;
		grid.z = min((nthreads_z + block.z - 1)/block.z, N_MAX_BLOCKS);
		apply_level_kernel<<<grid,block,shared_memory,stream>>>(start, end, f);
	}
		
}

void PartitionFlagged(int *d_out, 
			int *d_in, 
			char *d_flags, 
			int num_items, 
			int *d_num_selected_out,
			void *d_temp_storage,
			size_t temp_storage_bytes) {

	// Run selection
	cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items);
}

void ExclusiveSum(int *d_out, int *d_in, int num_items, void *d_temp_storage, size_t temp_storage_bytes) {
	// Run exclusive prefix sum
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, 0);
	
}


void SegmentedReduce(int *d_out, char *d_in, int *d_offsets, int num_segments) {
	// Determine temporary device storage requirements
	void     *d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;
	cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out,
			num_segments, d_offsets, d_offsets + 1);

	// Allocate temporary storage
	cudaMalloc(&d_temp_storage, temp_storage_bytes);

	// Run sum-reduction
	cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out,
			num_segments, d_offsets, d_offsets + 1);

}



template<typename F>
__global__ void apply_on_graph_kernel(int n, F f) {
	for(int u = blockIdx.z*blockDim.z + threadIdx.z;
		u < n;
		u += blockDim.z*gridDim.z) {

		f(u);
	}
}


//call on every node
template<typename F>
void apply_on_graph(int n, F f, dim3 localGrid, dim3 localBlock, int memory, cudaStream_t stream) {
	dim3 block,grid;

	block.x = localBlock.x;
	block.y = localBlock.y;
	block.z = min(512/block.x/block.y, 64);
	
	grid.x = localGrid.x; 
	grid.y = localGrid.y;
	grid.z = min((n + block.z - 1) / block.z, N_MAX_BLOCKS);
	apply_on_graph_kernel<<<grid,block,memory,stream>>>(n, f);
}
	
template<typename F>
void apply_on_graph(int n, F f, cudaStream_t stream) {
	dim3 grid, block;
	grid.x = grid.y = block.x = block.y = 1;
	apply_on_graph(n, f, grid, block, 0, stream);
}


