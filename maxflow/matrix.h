// Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.

#pragma once

#define INF 1e10
#include <algorithm>
#include "allocator.h"
#include <fstream>
#include "config.h"

using std::min;


struct csr_graph
{
  int n, nnz;
  int *row_offsets;
  int *col_indices;
  flow_t *vals_cap;	// capacity
  flow_t *vals_flow;	// maxflow

  // set edge weight = 1/degree
  void set_edge_weights_rcp_degree()
  {
    for (int i = 0; i < n; i++) {
      int degree = row_offsets[i+1] - row_offsets[i];
      for (int j = row_offsets[i]; j < row_offsets[i+1]; j++)
        vals_cap[j] = (flow_t)1.0/degree;
    }
  }

  // function returns edge id for i->j using binary search
  int edge(int i, int j) const
  {
    // use binary search here
    int low = row_offsets[i];
    int high = row_offsets[i+1]-1;
    while (high > low) {
      int mid = (low + high)/2;
      if (j == col_indices[mid]) return mid;
      if (j < col_indices[mid])
        high = mid-1;
      else
        low = mid+1;
    }

    return (col_indices[low] == j) ? low : -1;    
  }

};

struct csr_subgraph {
	int n, nnz;
	int *edge_offsets = NULL;
	int *parent_edge_indices = NULL;
	int *col_indices = NULL;

	csr_subgraph() { }		

	void resize(int _n, int _nnz) {
		n 	= _n;
		nnz 	= _nnz;
		clean();
		edge_offsets = (int*)my_malloc(n * sizeof(int)); 
		parent_edge_indices = (int*)my_malloc((nnz+1) * sizeof(int)); 
		col_indices = (int*)my_malloc(nnz * sizeof(int)); 
	}
	
	void clean() {
		if(edge_offsets) 	my_free(edge_offsets);
		if(parent_edge_indices) my_free(parent_edge_indices);
		if(col_indices) 	my_free(col_indices);
	}

};


