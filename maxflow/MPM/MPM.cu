#include "MPM.h"
#include "../bfs/bfs.h"

//Implementations of MPM functions members
#include "get_subgraph.cu"
#include "push_pull.cu"
#include "prune.cu"

#include <time.h>

#include <cuda_profiler_api.h>
#include "nvToolsExt.h"

#include "../config.h"

#define GPUID 0 
#define N_BLOCKS_MAX 65535
#define N_THREADS 512


MPM::MPM(csr_graph& _g) : g(_g) {
	//TODO reduce number of mallocs
	q_bfs  = (int*)my_malloc((g.n+1) * sizeof(int)); 
	h  = (int*)my_malloc((g.n) * sizeof(int)); 

	node_mask = (char*)my_malloc(g.n * sizeof(char)); 	
	queue_mask = (char*)my_malloc(g.n * sizeof(char)); 	
	prune_mask = (char*)my_malloc(g.n * sizeof(char)); 	
	have_been_pruned = (char*)my_malloc(g.n * sizeof(char)); 	

	node_g_to_sg  	= (int*)my_malloc(g.n * sizeof(int)); //TODO reuse Bfs
	node_sg_to_g  	= (int*)my_malloc(g.n * sizeof(int));

	edge_mask  	= (char*)my_malloc(g.nnz * sizeof(char));
	edge_mask_orig  = (char*)my_malloc(g.nnz * sizeof(char));
	reverse_edge_map  = (int*)my_malloc(g.nnz * sizeof(int));

	cudaMalloc(&d_total_flow, sizeof(flow_t));
	e = (flow_t*)my_malloc(g.n * sizeof(flow_t)); 	

	//buffer for degree_in and degree_out
	degree 	= (flow_t*)my_malloc((2 * g.n) * sizeof(flow_t)); 	

	bfs_offsets 	= (int*)my_malloc((g.n+1) * sizeof(int)); 	
	sg_level_offsets 	= (int*)my_malloc((g.n+1) * sizeof(int)); 	

	cudaMalloc(&d_nsg, sizeof(int));	

	cudaMallocHost(&d_node_to_push, sizeof(int));
	cudaMallocHost(&d_flow_to_push, sizeof(flow_t));

	cudaStreamCreate(&st1);
	cudaStreamCreate(&st2);

	cudaMemset(d_total_flow, 0, sizeof(flow_t));	
	cudaMemset(e, 0, sizeof(flow_t) * g.n);
	cudaMemset(prune_mask, 0, sizeof(char) * g.n);


	buf1  = (int*)my_malloc((g.n+1) * sizeof(int)); 
	buf2  = (int*)my_malloc((g.n+1) * sizeof(int)); 

	sg_in.resize(g.n, g.nnz);
	sg_out.resize(g.n, g.nnz);

	cf = g.vals_cap; //TODO alloc and copy

	//CUB memory
	//Device Reduce

	cudaMalloc(&d_ppd, sizeof(post_prune_data));

	cub::DeviceReduce::ArgMin(d_min_reduce, min_reduce_size, degree, &d_ppd->d_min, 2*g.n);
	cudaMalloc(&d_min_reduce, min_reduce_size);
	
	//Partition (get subgraph)
	cub::DevicePartition::Flagged(d_storage_partition, size_storage_partition, buf1, queue_mask, buf2, d_nsg, g.n);
	cudaMalloc(&d_storage_partition, size_storage_partition);
	
	//Exclusive sum (get subgraph)
	cub::DeviceScan::ExclusiveSum(d_storage_exclusive_sum, size_storage_exclusive_sum, buf1, buf2, g.n);
	cudaMalloc(&d_storage_exclusive_sum, size_storage_exclusive_sum);
	

	//Building reverse edge map
	for(int u=0; u != g.n; ++u) {
		for (int i = g.row_offsets[u]; i < g.row_offsets[u+1]; ++i) {
			int v = g.col_indices[i];
			int uv = i;
			int vu = g.edge(v,u); 
			reverse_edge_map[uv] = vu;
		}
	}
	memFetch();
	cudaDeviceSynchronize();
}

__global__ void setup_mask_unsaturated_kernel(int num_edges, char *mask, flow_t *cf)
{
	for(int u= threadIdx.x + blockIdx.x * blockDim.x;
		u < num_edges;
		u += blockDim.x * gridDim.x) 
		mask[u] = (cf[u] > 0);
}



bool setup_mask_unsaturated(int num_edges, char *mask, flow_t *cf) {
	setup_mask_unsaturated_kernel<<<min((num_edges + N_THREADS)/N_THREADS, N_BLOCKS_MAX), N_THREADS>>>(num_edges, mask, cf);
	return true;
}

//Main algorithm loop
flow_t MPM::maxflow(int _s, int _t, float *elapsed_time) {
	s = _s;
	t = _t;

	//TODO create cf
	setup_mask_unsaturated(g.nnz, edge_mask_orig, cf);

	
		int nsg; //number of nodes in subgraphh

	cudaDeviceSynchronize();
	
	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC, &start);
	cudaProfilerStart();
	while(bfs(g.row_offsets, g.col_indices, g.n, g.nnz, s, t, q_bfs, h, BFS_MARK_DEPTH, edge_mask_orig, bfs_offsets)) {
		cudaDeviceSynchronize();
		cudaMemcpy(&ht, &h[t], sizeof(int), cudaMemcpyDeviceToHost);
		init_level_graph(nsg);
		cudaDeviceSynchronize();
		
		nvtxRangePushA("saturate_subgraph");	
		//Find node to push - usually done end of prune, but the first need to be done here
		cub::DeviceReduce::ArgMin(d_min_reduce, min_reduce_size, degree_in+1, &(d_ppd->d_min), 2*(sg_in.n-1), st1);

		cudaMemcpy(&h_ppd, d_ppd, sizeof(post_prune_data), cudaMemcpyDeviceToHost);
		do {
			push_and_pull();
			prune();
		} while(!h_ppd.s_t_pruned);		
		nvtxRangePop();	

	}

	flow_t h_total_flow;
	cudaMemcpy(&h_total_flow, d_total_flow, sizeof(flow_t), cudaMemcpyDeviceToHost);

	cudaProfilerStop();
	clock_gettime(CLOCK_MONOTONIC, &end);
	*elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9;

	return h_total_flow;
}

void MPM::memFetch() {
	cudaMemPrefetchAsync(q_bfs, g.n * sizeof(int), 0, st1); 
	cudaMemPrefetchAsync(h, (g.n) * sizeof(int), 0, st1); 
	
	cudaMemPrefetchAsync(node_mask, g.n * sizeof(char), 0, st1); 	
	cudaMemPrefetchAsync(queue_mask, g.n * sizeof(char), 0, st1); 	
	cudaMemPrefetchAsync(prune_mask, g.n * sizeof(char), 0, st1); 	
	cudaMemPrefetchAsync(have_been_pruned, g.n * sizeof(char), 0, st1); 	

	cudaMemPrefetchAsync(node_g_to_sg, g.n * sizeof(int), 0, st1); //TODO reuse Bfs
	cudaMemPrefetchAsync(node_sg_to_g, g.n * sizeof(int), 0, st1);
	
	cudaMemPrefetchAsync(edge_mask, g.nnz * sizeof(char), 0, st1);
	cudaMemPrefetchAsync(edge_mask_orig, g.nnz * sizeof(char), 0, st1);
	cudaMemPrefetchAsync(reverse_edge_map, g.nnz * sizeof(int), 0, st1);
	
	cudaMemPrefetchAsync(e, g.n * sizeof(flow_t), 0, st1); 	
	
	cudaMemPrefetchAsync(bfs_offsets, (g.n+1) * sizeof(int), 0, st1); 	
	cudaMemPrefetchAsync(sg_level_offsets, (g.n+1) * sizeof(int), 0, st1); 	
	
	cudaMemPrefetchAsync(buf1, (g.n+1) * sizeof(int), 0, st1); 	
	cudaMemPrefetchAsync(buf2, (g.n+1) * sizeof(int), 0, st1); 	
	
	cudaMemPrefetchAsync(g.row_offsets, g.n * sizeof(int), 0, st1); 	
	cudaMemPrefetchAsync(g.col_indices, g.nnz * sizeof(int), 0, st1); 	
	cudaMemPrefetchAsync(cf, g.nnz * sizeof(flow_t), 0, st1); 	
}

MPM::~MPM() {
	//TODO free on host

} 
