// Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#include "../matrix.h"
#include "../allocator.h"
#include <stdio.h>
#include "../graph_tools.h"
#include "../bfs/bfs.h"
#include <time.h>

//Edmonds-karp implementation
//TODO separate CPU and GPU implem ? 

double maxflowimplementation(csr_graph* g, int s, int t, float *time) {
	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC, &start);



	int it = 0;		// number of augmented paths
	double fm = 0.0;	
	int *q = (int*)my_malloc(g->n * sizeof(int));  	// bfs vertices queue
	int *p = (int*)my_malloc(g->n * sizeof(int));		// parent vertices
	int *h = (int*)my_malloc(g->n * sizeof(int)); // depth of nodes - TODO remove
	char *mask = (char*)my_malloc(g->nnz * sizeof(int));	// edge mask (used in Gunrock only)
	double *cf = g->vals_cap;

	int* level_width = (int*)my_malloc(g->n * sizeof(int)); // depth of nodes - TODO remove
 //not used here	
	// find shortest augmented paths in c-f
	setup_mask_unsaturated(g->nnz, mask, cf);
	while (bfs(g->row_offsets, g->col_indices, g->n, g->nnz, s, t, q, p, BFS_MARK_PREDECESSOR, mask, level_width)) {
		// backtrack to find the max flow we can push through
		int v = t;
		double mf = INF;

		while (v != s) {
			int u = p[v];
			int i = g->edge(u,v);
			mf = min(mf, cf[i]);
			v = u;
		} 
		// update flow value
		fm = fm + mf;

		// backtrack and update flow graph
		v = t;
		int len = 0;
		while (v != s) {
			int u = p[v];
			int uv = g->edge(u,v);
			int vu = g->edge(v,u);

			cf[uv] -= mf;
			mask[uv] = (cf[uv] > 0);
			cf[g->edge(v,u)] += mf;
			mask[vu] = (cf[vu] > 0);

			v = u;
			len++;
		}

		// output more stats
#if LOG_LEVEL > 1
		printf("path length %d, aug flow %g\n", len, mf);
#endif

#if LOG_LEVEL > 2
		printf("aug path vertices: ");
		v = t;
		while (v != s) { printf("%i ", v+1); v = p[v]; }
		printf("%i \n", v+1);
#endif

		// count number of iterations
		it++;
	}
#if LOG_LEVEL > 0
	printf("%i augmenting paths\n", it);
#endif

	my_free(q);
	my_free(p);
	my_free(mask);

	clock_gettime(CLOCK_MONOTONIC, &end);
	*time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9;

	return fm;
} 
