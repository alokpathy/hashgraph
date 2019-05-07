// Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.

#include <stdio.h>
#include "../matrix.h"
#include "bfs.h"

#include "nvToolsExt.h"

//Generic tools handling masks and fill

#define THREADS_PER_VERTEX_BFS	4

#define N_BLOCKS_MAX 65535


template<typename T>
__global__ void fill_kernel(int size, T *data, T value)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < size) data[tid] = value;
}

template<typename T>
void fill(int size, T *data, T value) {
	fill_kernel<<<(size + 255)/256, 256>>>(size, data, value);
	cudaDeviceSynchronize();
}

// main bfs kernel: finds next frontier using blockDim.x threads per vertex
//level_width[i] contain the max width of the ith level
__global__ void next_frontier(int start, int end, int *d_next, int *row_offsets, int *col_indices, char *mask, int t, int *q, int *output, bool mark_pred, int bfs_level, int *found)
{
	for(int idx = start + blockIdx.y * blockDim.y + threadIdx.y;
	idx < end;
	idx += blockDim.y * gridDim.y) {
		// current frontier
		int u = q[idx];
		// writing node level TODO optional

		// loop over neighbor vertices
		for (int i = row_offsets[u] + threadIdx.x; i < row_offsets[u+1]; i += blockDim.x) {
			// next frontier
			int v = col_indices[i];
			// only advance if we haven't visited & mask allows it
			if (output[v] == -5 && mask[i]) {
				// critical section below to avoid collisions from multiple threads
				if (atomicCAS(&output[v], -5, mark_pred ? u : (bfs_level+1)) == -5) {		
					// add new vertex to our queue
					int pos = atomicAdd(d_next, 1);
					q[pos] = v;
					
					if (v == t) {
						// early exit if we found the path
						*found = 1;
						return;
					}
				}
			}
		}
	}
}

//BFS GPU naive implementation
int bfs(int *row_offsets, int *col_indices, int num_nodes, int num_edges, int s, int t, int *q, int *output, int output_type, char *mask, int *bfs_offsets)
{
	nvtxRangePushA("BFS");	
	// set all vertices as undiscovered (-5)
	fill(num_nodes, output, -5);
	// start with source vertex
	q[0] = s;
	bool mark_pred = (output_type == BFS_MARK_PREDECESSOR);
  	output[s] = mark_pred ? s : 0;
	
	// found flag (zero-copy memory)
	static int *found = NULL;
	if (!found) cudaMallocHost(&found, sizeof(int));
	*found = 0;
	
	static int *d_next = NULL;
	if (!d_next)  cudaMalloc(&d_next, sizeof(int));
	
	int h_start = 0, h_end = 1;

	cudaMemcpy(d_next, &h_end, sizeof(int), cudaMemcpyHostToDevice);	
	
	int bfs_level = 0;

	int off_idx = 0;
	bfs_offsets[off_idx++] = 0;		

	
	dim3 block(THREADS_PER_VERTEX_BFS, 128 / THREADS_PER_VERTEX_BFS);
	do {
		// calculate grid size
		int nitems;
		nitems = h_end - h_start;

#if LOG_LEVEL > 4
		printf("  bfs level %i: %i vertices :\n", bfs_level, nitems);
		for(int i=h_start; i!=h_end; ++i)
			printf("%i\t", q[i]);
		printf("\n");
			
#endif

		dim3 grid(1, min((nitems + block.y-1) / block.y, N_BLOCKS_MAX));
		next_frontier<<<grid, block>>>(h_start, h_end, d_next, row_offsets, col_indices, mask, t, q, output, mark_pred, bfs_level, found);

		bfs_offsets[off_idx++] = h_end;		
		h_start = h_end;
		cudaMemcpy(&h_end, d_next, sizeof(int), cudaMemcpyDeviceToHost);
		++bfs_level;
	} while(h_start < h_end && *found == 0);

	bfs_offsets[off_idx++] = h_end;		

	#if LOG_LEVEL > 1
	if (*found)
		printf("bfs: traversed vertices %d (%.0f%%), ", h_end, (double)100.0*h_end/num_nodes);
	#endif

	nvtxRangePop();	
	return (*found);
}

