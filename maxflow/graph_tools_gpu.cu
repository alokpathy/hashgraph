// Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.

#include <stdio.h>


#define N_BLOCKS_MAX 65535
#define N_THREADS 512

template<typename T>
__global__ void fill_kernel(int size, T *data, T value)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < size) data[tid] = value;
}

__global__ void setup_mask_unsaturated_kernel(int num_edges, char *mask, double *cf)
{
	for(int u= threadIdx.x + blockIdx.x * blockDim.x;
		u < num_edges;
		u += blockDim.x * gridDim.x) 
		mask[u] = (cf[u] > 0);
}


//TODO memory issue ? reverse_edge can be kind of chatotics
__global__ void setup_mask_unsaturated_backward_kernel(int num_edges, int *mask, double *cf, int *reverse_edge) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < num_edges) mask[reverse_edge[tid]] = (cf[tid] > 0);
}


template<typename T>
void fill(int size, T *data, T value) {
	fill_kernel<<<(size + 255)/256, 256>>>(size, data, value);
	cudaDeviceSynchronize();
}

bool setup_mask_unsaturated(int num_edges, char *mask, double *cf) {
	setup_mask_unsaturated_kernel<<<min((num_edges + N_THREADS)/N_THREADS, N_BLOCKS_MAX), N_THREADS>>>(num_edges, mask, cf);
	return true;
}

void setup_mask_unsaturated_backward(int num_edges, int *mask, double *cf, int *reverse_edge) {
  	setup_mask_unsaturated_backward_kernel<<<(num_edges + 255)/256, 256>>>(num_edges, mask, cf, reverse_edge);
	cudaDeviceSynchronize();
}


template void fill<int>(int,int*,int);
template void fill<double>(int,double*,double);
template void fill<char>(int,char*,char);
