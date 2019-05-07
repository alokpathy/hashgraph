// Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.

#pragma once

#include "../matrix.h"

//TODO inheritance nvgraph csr DS

//csr graph using same edge idx than its parent
struct csr_graph_mpm : public csr_graph_reverse {

	//nodes degree
	double *degree_in;
	double *degree_out;

	//Mask to know if an edge is active
	char *edge_mask;
	int *edge_mask_orig; //active in the orig graph TODO refactoring	
	//Distance source-node
	int *h;
	
	int *buf1;
	int *buf2;

	int *sg_level_offsets;

	int ht; //depth of sink		
	//buffer must be of size at least g->n
	csr_graph_mpm(const csr_graph& g, int *buffer) : 
		csr_graph_reverse(g, buffer),
		cf(g.vals_cap) //TODO copy if we want the details
	{
		
		buf1  	= (int*)my_malloc(g.n * sizeof(int)); //TODO use bitset
		buf2  	= (int*)my_malloc(g.n * sizeof(int)); //TODO use bitset
		sg_level_offsets = (int*)my_malloc((g.n+1) * sizeof(int)); //TODO use bitset
		
		h = (int*)my_malloc(g.n * sizeof(int));
	}


	virtual void memFetch(int deviceID) {
		cudaMemPrefetchAsync(cf, nnz * sizeof(double), deviceID, 0);
		cudaMemPrefetchAsync(edge_mask, nnz * sizeof(int), deviceID, 0);
		
		cudaMemPrefetchAsync(buf2, n * sizeof(int), deviceID, 0);
		cudaMemPrefetchAsync(buf1, n * sizeof(int), deviceID, 0);
	
		cudaMemPrefetchAsync(node_g_to_sg, n * sizeof(int), deviceID, 0);
		cudaMemPrefetchAsync(node_sg_to_g, n * sizeof(int), deviceID, 0);
		cudaMemPrefetchAsync(sg_level_offsets, (n+1) * sizeof(int), deviceID, 0);
		
		csr_graph_reverse::memFetch(deviceID);
	}	

	
	//mess with the copy of struct to gpu
	/*
	virtual ~csr_graph_mpm() {
		my_free(degree_in);
		my_free(degree_out);
		my_free(edge_mask);
		my_free(h);
	}
	*/
};


