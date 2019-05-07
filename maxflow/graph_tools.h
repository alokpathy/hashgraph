// Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.

#pragma once 

#include "allocator.h"

#include <vector>
#include <tuple>
#include <algorithm>
#include <numeric>

using std::vector;
using std::tuple;
using std::tie;
using std::get;
using std::make_tuple;

typedef tuple<int,double,double> idd;

template<typename T>
void fill(int size, T *data, T value);

bool setup_mask_unsaturated(int num_edges, char *mask, double *cf); 

void setup_mask_unsaturated_backward(int num_edges, int *mask, double *cf, int *reverse_edge); 

/*
//reorder based on bfs from s
void inline reorder_memory(csr_graph* in, csr_graph* out, int *q, int *h, int *s, int *t) {
	//TODO lambdas in BFS
	printf("Start reordering \n");
	//alocating new graph 
	//using too much memory, pre BFS ?
	out->row_offsets = (int*)my_malloc((in->n+1) * sizeof(int));
	out->col_indices = (int*)my_malloc(in->nnz * sizeof(int));
	out->vals_cap = (double*)my_malloc(in->nnz * sizeof(double));
	out->vals_flow = (double*)my_malloc(in->nnz * sizeof(double));

	//used to store old node id -> new node id 
	int* eq = (int*)malloc(in->n * sizeof(int));		

	#pragma omp parallel for 
	for (int i = 0; i < in->n; i++) h[i] = -1;

	// start with source vertex
	q[0] = *t;
	eq[*t] = 0;
	h[*t] = 0;

	int start_idx = 0;
	int i_node = 1;
	int i_edge = 0;
	int end_idx = i_node;
	int found = 0;
	int bfs_level = 0;

	vector<idd> current_node_edges;

	while(!found && start_idx < end_idx) {

	//printf("Level %i : %i nodes \n", bfs_level,  end_idx - start_idx); 
		for(int idx = start_idx; idx < end_idx; idx++) {
			int u = q[idx];
			int new_u = idx;
			out->row_offsets[new_u] = i_edge;
			//printf("Start index %i ----- : \n", new_u);
			
			current_node_edges.clear();
			
			for (int i = in->row_offsets[u]; i < in->row_offsets[u+1]; i++) {
				int v = in->col_indices[i];
				if(__sync_val_compare_and_swap(&h[v], -1, bfs_level+1) == -1) { 
						int new_v    = __sync_fetch_and_add (&i_node, 1);
						q[new_v] = v;
						eq[v] = new_v;
				}

				current_node_edges.push_back(make_tuple(eq[v], in->vals_cap[i], in->vals_flow[i]));
				//printf("------- to edge %i ----- : \n", eq[v]);
			}
			
			std::sort(current_node_edges.begin(), 
				  current_node_edges.end(),
				  [] (const idd& t1, const idd& t2) { return get<0>(t1) < get<0>(t2); }  );
			//printf("Edges of node %i : \n", u);
			for(size_t k=0; k != current_node_edges.size(); ++k) {
				tie(out->col_indices[i_edge],
					out->vals_cap[i_edge],
					out->vals_flow[i_edge]) = current_node_edges[k];
				++i_edge;
				//printf("----> %i : %f \n", out->col_indices[i_edge-1], out->vals_cap[i_edge-1]);
			}			
		}
	start_idx = end_idx;
    	end_idx = i_node;
    	++bfs_level;
	}
	
	out->row_offsets[i_node] = i_edge;
	out->n = i_node;
	out->nnz = i_edge;
	*s = eq[*s];
	*t = eq[*t];

	free(eq);
	
	printf("Reordering memory : from %i to %i \n", in->n, out->n);
}

*/






