// Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#include "../matrix.h"
#include "../allocator.h"
#include <stdio.h>
#include "push-relabel_operations.h" 
#include "../bfs/bfs.h"
#include "../graph_tools.h"
#include <unistd.h>

//#define REORDER_GRAPH

//Push-relabel max flow implementation
//GPU implementation
double maxflowimplementation(csr_graph* orig_g, int s, int t) {
	
	int *h = (int*)my_malloc(orig_g->n * sizeof(int));  	// nodes height
	int *q = (int*)my_malloc(orig_g->n * sizeof(int));  	// bfs vertices queue
	
	#ifdef REORDER_GRAPH
	csr_graph* g = new csr_graph;
	reorder_memory(orig_g, g, q, h, &s, &t); 
	#else
	csr_graph* g = orig_g;
	#endif

	double *e = (double*)my_malloc(g->n * sizeof(double));  	// flow excess
	int *global_relabel_h = (int*)my_malloc(g->n * sizeof(int));  	// nodes height
	
	int *reverse = (int*)my_malloc(g->nnz * sizeof(int)); //reverse edge id
	double *cf = g->vals_cap; //using vals_cap as cf, vals_flow will not be valid after this function 
	int *p = (int*)my_malloc(g->n * sizeof(int));		// parent vertices
	int *mask = (int*)my_malloc(g->nnz * sizeof(int));	// edge mask (used in Gunrock only)

	//Keeping track of blocks containing active nodes
	int *active_blocks = NULL;
	#ifdef ACTIVE_FLAGS
	int n_node_blocks = blockIdyForNode(g->n) + 1;
	active_blocks = (int*)my_malloc(n_node_blocks * sizeof(int));
	fill(n_node_blocks, active_blocks, 0);	
	#endif


	fill(g->n, h, 0); 
	fill(g->n, global_relabel_h, 0); 
	fill(g->n, e, 0.0);

	h[s] = g->n;	
	//Saturating out-edges from source
	
	for (int i = g->row_offsets[s]; i < g->row_offsets[s+1]; ++i) {
		int u = g->col_indices[i];
		int su = i;
		int us = g->edge(u,s); 
		double vf_su = g->vals_cap[su];
		e[u] = vf_su;
		g->vals_cap[us] += vf_su;
		g->vals_cap[su] -= vf_su;
		active_blocks[blockIdyForNode(u)] = 1;
		#if LOG_LEVEL > 3
		printf("init : pushing %f from %i to %i \n", vf_su, s, u); 
		#endif
	}

	//Naive reverse edges finding
	//We could do it linearly in the kernel
	for(int u=0; u != g->n; ++u) {
		for (int i = g->row_offsets[u]; i < g->row_offsets[u+1]; ++i) {
			int v = g->col_indices[i];
			int uv = i;
			int vu = g->edge(v,u); 
			reverse[uv] = vu;
		}
	}
	double oldet = -1; //forcing init global relabel
	
	do {
		//cudaDeviceSynchronize();	
		//Mark as valid an edge if its reversal is valid
		if(e[t] > oldet) {	
		printf("--------- Global relabeling --------- \n");
		oldet = e[t];
		
		//Violating edges can exists during lock-free push relabel
		//We need to treat them (saturate them) before bfs
		//BFS will make them valid
		//can be avoided by using while(e[u] > 0) in discharge kernel
		remove_violating_edges(cf, g->n, g->row_offsets, g->col_indices, reverse, e, h, active_blocks);
	
		setup_mask_unsaturated_backward(g->nnz, mask, cf, reverse);
	
		//Backwards BFS from sink (global relabeling)
		bfs(g->row_offsets, g->col_indices, g->n, g->nnz, t, -1, q, global_relabel_h, BFS_MARK_DEPTH, mask);
		
		//Post bfs global relabeling + gap relabeling
		global_gap_relabeling_post_bfs(global_relabel_h, h, e, g->n);
		cudaDeviceSynchronize();	
		}
		
		#if LOG_LEVEL > 2
		printf("Finished push session, s=%f, t=%f \n", e[s], e[t]);
		#endif
		
	} while(discharge(cf, g->n, s, t,  g->row_offsets, g->col_indices, reverse, e, h, q, active_blocks));
	
	#if LOG_LEVEL > 2
	printf("Finished push-relabel algorithm, s=%f, t=%f \n", e[s], e[t]);
	#endif


	return e[t];
}
