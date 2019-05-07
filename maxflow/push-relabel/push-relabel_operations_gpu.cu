//Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#include <stdio.h>
#include <limits.h>
//#include "../utils.cu"
#include "vector_types.h"
#include "push-relabel_operations.h"

//Number of times the push/relabel method is applied between two global relabeling
#define KERNEL_CYCLES 10

//Number of threads to process the reductions in discharge
//Must be less than warpsize 
//Only power of two (lots of divisions)
#define THREADS_PER_VERTEX 4
#define THREADS_PER_BLOCK 128
#define BLOCK_Y_SIZE (THREADS_PER_BLOCK / THREADS_PER_VERTEX)


//Min on int3 using first int
__inline__ __device__ int3 min3_x(int3 i1, int3 i2) {
	return (i1.x < i2.x) ? i1 : i2;
}

//Reducing using min inside of warps, by group of THREADS_PER_VERTEX


__inline__ __device__ int3 warpReduceMin(int3 val) {
	for(int offset = THREADS_PER_VERTEX/2; offset > 0; offset >>= 1) { 
		//Reducing only in same groups
		int3 other;
		other.x =  __shfl_down(val.x, offset, THREADS_PER_VERTEX);
		other.y =  __shfl_down(val.y, offset, THREADS_PER_VERTEX);
		other.z =  __shfl_down(val.z, offset, THREADS_PER_VERTEX);
		//the min3_x function has to be branch-free
		val = min3_x(val, other);		
	}
	return val;
}

//#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
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
//#endif

__host__ __device__ int blockIdyForNode(int u) {
	return (u / BLOCK_Y_SIZE);
}

__global__ void discharge_kernel(int blocky_offset, double *cf, int num_nodes, int s, int t, int *row_offsets, int *col_indices, int *reverse, double *e, int *h, int *active_blocks) {
	//branch diverging inside the kernel	
	
	//ID of the current block of nodes
	int blocky = blocky_offset + blockIdx.y; 
	//Node concerned by this thread
	int u 	   = blocky * blockDim.y + threadIdx.y;
	
	#ifdef ACTIVE_FLAGS 
	__shared__ bool is_block_active;	
	if(threadIdx.x == 0 && threadIdx.y ==0) {
		is_block_active = atomicExch(&active_blocks[blocky], 0);
	}
	__syncthreads();
	
	if(!is_block_active) return;
	is_block_active = false;	
	#endif

	//Shared between threads of the x-axis
	//8 byte aligned	
	extern __shared__ double sh[];
	double *local_e = sh;
	int *local_h = (int*)&sh[blockDim.y];

	
	//Vertex concerned by the discharge op
	int local_u = threadIdx.y;

	if (u >= num_nodes || u == s || u == t) return;

	int cycle = KERNEL_CYCLES + 1;
	
	while(--cycle) {
		if(threadIdx.x == 0) {
			local_h[local_u] = h[u];
			local_e[local_u] = e[u];
			//no sync, same warp
		}
		//if node is not active, nothing to do		
		if(local_e[local_u] <= 0.0 || local_h[local_u] == num_nodes) break;		
		
		#if LOG_LEVEL > 3
		if(threadIdx.x == 0)
			printf("%i is active : e=%f le=%f lh=%i h=%i \n", u, e[u], local_e[local_u], local_h[local_u], h[u]);
		#endif

		//Looking for lowest neighbor
		//First is neighbor height
		//Second is correcponding edge index
		int3 thread_candidate;
		thread_candidate.x = INT_MAX;
		//TODO put row_offsets in shared ?
		int end_edge = row_offsets[u+1];
		for(int i_edge = row_offsets[u] + threadIdx.x; 
			i_edge < end_edge; 
			i_edge += blockDim.x) {
			
			int neighbor = col_indices[i_edge];
			int3 candidate;	
			//setting neighbor height
			//if neighbor cannot be chosen, use identity element of min	
			candidate.x = (cf[i_edge] > 0 && h[neighbor] < num_nodes) 
							? h[neighbor] 
							:  INT_MAX; 
			candidate.y = i_edge;
			candidate.z = neighbor;
			thread_candidate = min3_x(
				thread_candidate,
				candidate
			);
		}
		

		//using warpReduceMin
		//TODO if blockDim.x is bigger than a warp, use blockDim.y = 1 and blockReduceMin
		#if THREADS_PER_VERTEX > 1
		//printf("Y=%i ; T:%i : we have a candidate for %i with h %i \n",threadIdx.y, threadIdx.x, u, thread_candidate.x);	
		thread_candidate = warpReduceMin(thread_candidate);	
		#endif

		//Only first of blocks has the result
		if(threadIdx.x == 0) {

			//printf("the winner for %i is h=%i \n", u, thread_candidate.x);
			
			//Retrieving result	
			int h_neighbor = thread_candidate.x;		
			int i_edge   = thread_candidate.y;
			int neighbor = thread_candidate.z;

			if(h_neighbor == INT_MAX) break; //no admissible neighbor 

			//If pushing is possible, push
			if(local_h[local_u] > h_neighbor) {
				//pushing from u to neighbor 
				double d = min(local_e[local_u], cf[i_edge]);
				
				#if LOG_LEVEL > 3
				printf("Pushing %f/%f from %i to %i, h = %i and %i, eu = %f \n", d, cf[i_edge], u, col_indices[i_edge], local_h[local_u], h_neighbor, local_e[local_u]);
				#endif

				atomicAdd(&cf[reverse[i_edge]], d); //TODO directly set mask ? 
				atomicAdd(&cf[i_edge], -d);
				atomicAdd(&e[neighbor], d);
				local_e[local_u] = atomicAdd(&e[u], -d) - d; //postfix operator

				#ifdef ACTIVE_FLAGS 
				if(local_e[local_u] > 0) 
					is_block_active = 1;
				
				active_blocks[blockIdyForNode(neighbor)] = 1;	
				#endif
			} else { //if we can't push, relabel to lowest + 1 (key to the lock free algo)
				#if LOG_LEVEL > 3
				printf("Relabeling %i  from %i to %i, excess was %f \n", u, local_h[local_u], h_neighbor+1, local_e[local_u]);
				#endif
					h[u] = local_h[local_u] = h_neighbor + 1;
					is_block_active = 1;
			
			}		
		}	
	}

	
	//setting global flag for current block
	#ifdef ACTIVE_FLAGS 
	__syncthreads();
	
	//Not sure if its alive
	//if(threadIdx.x == 0 && threadIdx.y ==0) {
	//TODO we only need to do it once
	if(is_block_active)
		active_blocks[blocky] = 1;
		//printf("------------ SACTIVATING %i \n", blockIdx.y);
	//}
	#endif


}


__global__ void remove_violating_edges_kernel(double *cf, int num_nodes, int *row_offsets, int *col_indices, int *reverse, double *e, int *h, int *active_blocks) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	if (u >= num_nodes) return;
	
	for (int i = row_offsets[u]; i < row_offsets[u+1]; ++i) {
		int v = col_indices[i];
		double cf_uv = cf[i];
		
		//random accesses global memory
		if(cf_uv > 0 && h[u] > h[v] + 1) {
			printf("%i is violating edge \n", u);	
			int vu = reverse[i];
			double eu = atomicAdd(&e[u], -cf_uv) - cf_uv;
			double ev = atomicAdd(&e[v], cf_uv) + cf_uv;
			atomicAdd(&cf[vu], cf_uv);
			cf[i] = 0;
			
			#ifdef ACTIVE_FLAGS 
			//we cant turn off the associated blocks
			//other vertices may be active
			#endif
		}

	}
}


__global__ void global_gap_relabeling_post_bfs_kernel(int* global_relabel_h, int *h, double *e, int num_nodes) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;

	if (u >= num_nodes) return;

	if(global_relabel_h[u] == -1) { //we cant go from the node to the sink
		h[u] = num_nodes;
	} else {
		if(global_relabel_h[u] > h[u])
			h[u] = global_relabel_h[u];
	}
}




//Wrapper functions


//Discharge : apply the push/relabel ops
//return true if there is still active nodes in the graph
bool discharge(double *cf, int num_nodes, int s, int t, int *row_offsets, int *col_indices, int *reverse, double *e, int *h, int *q, int *active_blocks) {
		
		//first and last y block to execute, [start; end[
		int block_idy_start = 0, block_idy_end = 0;
		int total_y_blocks = blockIdyForNode(num_nodes) + 1;

		#if LOG_LEVEL > 4	
		cudaDeviceSynchronize(); //e need to be ready	
		printf("Actives nodes : ");
		for(int u=0; u != num_nodes; ++u)
			if(e[u] > 0)
				printf("\t%i", u);
		printf("\n");		
		#endif		

		#ifdef ACTIVE_FLAGS
		
		cudaDeviceSynchronize(); //active_blocks need to be ready	

		#if LOG_LEVEL > 4	
		printf("Actives blocks : ");
		#endif		

		int y=0;
		for(; y != total_y_blocks; ++y)
			if(active_blocks[y]) {
				#if LOG_LEVEL > 4	
				printf("\t%i:[%i;%i[", y, y*BLOCK_Y_SIZE, (y+1)*BLOCK_Y_SIZE);
				#endif		
				block_idy_start = y;
				block_idy_end = y + 1;
				break;
			}
		for(++y; y != total_y_blocks; ++y)
			if(active_blocks[y]) {
				#if LOG_LEVEL > 4	
				printf("\t%i:[%i;%i[", y, y*BLOCK_Y_SIZE, (y+1)*BLOCK_Y_SIZE);
				#endif		
				block_idy_end = y + 1;
			}
		
		#if LOG_LEVEL > 4	
		printf("\n");
		printf("Discharge : from %i to %i, %f%% of the vertices \n",
		block_idy_start * BLOCK_Y_SIZE,
		block_idy_end * BLOCK_Y_SIZE,
		100 * (double)(block_idy_end - block_idy_start) / total_y_blocks);
		#endif	
		
		#else
		block_idy_start = 0;
		block_idy_end = total_y_blocks+1;
		#endif
	
		
		dim3 block(THREADS_PER_VERTEX, BLOCK_Y_SIZE);
		dim3 grid(1, (block_idy_end - block_idy_start));
		int shared_memory_size = block.y * sizeof(int) + block.y * sizeof(double);
		discharge_kernel<<<grid, block, shared_memory_size>>>(block_idy_start, cf, num_nodes, s, t,  row_offsets, col_indices, reverse, e, h, active_blocks);
		cudaDeviceSynchronize();	
		
		//TODO use indep flag ?
		for(y=0; y != total_y_blocks; ++y)
			if(active_blocks[y])
				return true;

		return false;
}	
	
void remove_violating_edges(double *cf, int num_nodes, int *row_offsets, int *col_indices, int *reverse, double *e, int *h, int *active_blocks) {
		cudaDeviceSynchronize(); //TODO	
		remove_violating_edges_kernel<<<(num_nodes + 255)/256, 256>>>(cf, num_nodes, row_offsets, col_indices, reverse, e, h, active_blocks);
}		
 
void global_gap_relabeling_post_bfs(int *global_relabel_h, int* h, double *e, int num_nodes) {
		cudaDeviceSynchronize(); //TODO	
		global_gap_relabeling_post_bfs_kernel<<<(num_nodes + 255)/256, 256>>>(global_relabel_h, h, e, num_nodes);
}
	
