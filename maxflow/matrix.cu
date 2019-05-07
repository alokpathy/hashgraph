#include "matrix.h"

//CUB histogram is too complex
__global__ count_values(int *values, int *counts, int size) {
	for(int u = blockIdx.x * blockDim.x + threadIdx.x;
		u < size;
		u += blockDim.x * gridDim.x) 
		atomicAdd(&counts[values[u]], 1);

}

__global__ set_idx_origin(int *row_offsets, int *col_indices, int *in_edge_idx_offsets, int *in_edge_idx, int *edge_origin, int n) {
	for(int u = blockIdx.y * blockDim.y + threadIdx.y;
		u < n;
		u += blockDim.y * gridDim.y) {
		
		for(int i_edge = row_offsets[u];
			i_edge < row_offsets[u+1];
			i_edge += blockDim.x) {
			int v = col_indices[i_edge];
			int pos = atomicAdd(&in_edge_idx_offsets[v], -1) - 1; 
			in_edge_idx[pos] = i_edge;
			edge_origin[i_edge] = u;
		}

	}

}

#define SET_IN_THREADS_PER_VERTEX 4
#define N_THREADS 512
#define SET_IN_BLOCK_Y (N_THREADS / SET_IN_THREADS_PER_VERTEX)
#define N_BLOCKS_MAX 65535

void csr_graph_reverse::set_in_edge(const csr_graph &g) {
	if(g.n <= 0) return;
	cudaMemset(in_edge_idx_offsets, 0, sizeof(int) * (g.n + 1));

	dim3 grid1D(min(N_BLOCKS_MAX, g.n /N_THREADS));
	dim3 block1D(N_THREADS); 
	count_values<<<grid1D,block1D>>>(g.col_indices, in_edge_idx_offsets, g.nnz);

	//Copied gtom cub doc
	// Determine temporary device storage requirements
	void     *d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;
	cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, in_edge_idx_offsets, in_edge_idx_offsets, g.n + 1);
	// Allocate temporary storage
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, in_edge_idx_offsets, in_edge_idx_offsets, g.n + 1);

	
	cudaMemcpy(&in_edge_idx_offsets[g.n], &in_edge_idx_offsets[g.n-1], sizeof(int), cudaMemcpyDeviceToDevice);
	
	dim3 grid2D(1, min(N_BLOCKS_MAX, g.n/SET_IN_BLOCK_Y));
	dim3 block2D(SET_IN_THREADS_PER_VERTEX, SET_IN_BLOCK_Y); 
	set_idx_origin<<<grid2D, block2D>>>(g.row_offsets, g.col_indices, in_edge_idx_offsets, in_edge_idx, edge_origin, g.n);
}
