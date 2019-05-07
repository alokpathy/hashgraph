#pragma once
#include "../matrix.h"
#include <vector>
#include "../cub/cub/cub.cuh"

#include "../config.h"


using std::vector;

using cub::KeyValuePair;
typedef KeyValuePair<int,flow_t> kvpid;

struct post_prune_data {
	kvpid d_min;
	int prune_flag; //used to know if we're done with prune
	int s_t_pruned;
};

class MPM {
public:
	MPM(csr_graph &g);
	flow_t maxflow(int s, int t, float *time);
	__host__ __device__ virtual ~MPM();

	//Should be private function members
	void init_level_graph(int& nsg);
	void get_node_to_push();
	void push_and_pull();
	void prune();
	void write_edges();
	void memFetch();
	
	const csr_graph g;	

	//Current query
	int s, t;

	//BFS
	int *q_bfs;
	int *h;
	int *bfs_offsets;	
	int ht;
	
	int *d_nsg; //size subgraph, on device

	int *d_node_to_push;
	flow_t *d_flow_to_push;
	
	char *node_mask; // nodes active in layer graph
	char *queue_mask; // nodes active in layer graph (using their bfs queue idx)
	char *prune_mask; // nodes to prune - layer graph indexes
	char *have_been_pruned;
	char *edge_mask; // nodes to prune - layer graph indexes
	char *edge_mask_orig; // nodes to prune - layer graph indexes

	flow_t *d_total_flow; //total flow pushed so far - on device
	flow_t *e;	//local excess - used in push/pull
	flow_t *cf;	//edge cap - edge flow	

	//Degree
	flow_t *degree;
	flow_t *degree_in;
	flow_t *degree_out;

	//Layer graph
	csr_subgraph sg_in, sg_out;	
	int *node_sg_to_g;	
	int *node_g_to_sg;	
	int *sg_level_offsets;
	int max_level_width;
	
	cudaStream_t st1, st2; //streams used by kernels

	//CUB
	void *d_storage_partition = NULL;
	size_t size_storage_partition = 0;	

	void *d_storage_exclusive_sum = NULL;
	size_t size_storage_exclusive_sum = 0;	

	//Buffers - we may not need them
	int *buf1, *buf2;

	//Can be removed if memory becomes a pb
	int *reverse_edge_map; 

	//Used by cub funcs in get_node_min
	post_prune_data h_ppd, *d_ppd;

	void *d_min_reduce = NULL; 
	size_t min_reduce_size = 0;
	
};


