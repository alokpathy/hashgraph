#include <limits>
#include <cfloat> //for cuda
#include "../cub/cub/cub.cuh"
#include "../allocator.h"

#include <stdlib.h> //TODO just for debug (exit)

#define THREADS_PER_VERTEX 32
#define N_BLOCK_THREADS 512 //number of threads in a block
#define BLOCK_Y_SIZE (N_BLOCK_THREADS / THREADS_PER_VERTEX)

//init edge events	
cudaEvent_t memset_1, memset_2, start_init;
//argmin events
cudaEvent_t argmin_out, start_get_min;
//push and pull events
cudaEvent_t start_move_flow, end_move_flow;

void createEvents() {
	cudaEventCreate(&memset_1);
	cudaEventCreate(&memset_2);
	cudaEventCreate(&start_init);
	cudaEventCreate(&start_get_min);

	cudaEventCreate(&argmin_out);

	cudaEventCreate(&start_move_flow);
	cudaEventCreate(&end_move_flow);
}



#include "../utils.cu"
#include "get_subgraph.cu"
#include "find_node_to_push.cu"
#include "prune.cu"
#include "push_pull.cu"
#include "device_gppp.cu"

#define N_THREADS 512

#define N_MAX_BLOCKS 65534

dim3 getBlock2D() {
	return dim3(THREADS_PER_VERTEX, BLOCK_Y_SIZE);
}
dim3 getGrid2D(int n) {
	return dim3(1, min((n + BLOCK_Y_SIZE -1) / BLOCK_Y_SIZE, N_MAX_BLOCKS));
}
dim3 getBlock1D() {
	return dim3(N_THREADS);
}

dim3 getGrid1D(int n) {
	return dim3 (min((n + N_THREADS - 1) / N_THREADS, N_MAX_BLOCKS));
}

// 
// Push/Pull function
// wiq_prune, ll be called by iterate_(in|out)_neighbors
// returns true if the caller should stop iterate
//
// DIRECTION :
// FORWARD : push from u to v
// BACKWARD : pull from u to v


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//
// Reduce-push-pull-prune kernel
// Used if layered graph is small enough
//




 
//
//
//
//


