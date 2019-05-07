// Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../allocator.h"
#include "../matrix.h"
#include "bfs.h"

//BFS CPU implementation
int bfs(int *row_offsets, int *col_indices, int num_nodes, int num_edges, int s, int t, int *q, int *output, int output_type, char *mask, int *bfs_offsets)
{
	int found = 0;
	#pragma omp parallel num_threads(1)
	{
		int edges = 0;  // count number of visited edges

		// set all vertices as undiscovered (-1)
		for (int i = 0; i < num_nodes; i++) output[i] = -1;

		// start with source vertex
		q[0] = s;
		output[s] = (output_type == BFS_MARK_PREDECESSOR) ? s : 0;

		#if LOG_LEVEL > 3
		int *bfs_level = (int*)my_malloc(num_nodes * sizeof(int));
		int *bfs_vertices = (int*)my_malloc(num_nodes * sizeof(int));
		int *bfs_edges = (int*)my_malloc(num_nodes * sizeof(int));
		memset(bfs_level, 0, num_nodes * sizeof(int));
		memset(bfs_vertices, 0, num_nodes * sizeof(int));
		memset(bfs_edges, 0, num_nodes * sizeof(int));
		bfs_vertices[0] = 1;
		bfs_edges[0] = row_offsets[s+1] - row_offsets[s];
		#endif

		int idx = -1;
		int size = 1;
		while (idx+1 < size && !found) {
			idx = idx+1;
			int u = q[idx];
			for (int i = row_offsets[u]; i < row_offsets[u+1]; i++) {
				int v = col_indices[i];
				edges++;
				if (output[v] == -1 && mask[i]) {
					output[v] = (output_type == BFS_MARK_PREDECESSOR) ? u : output[u]+1;
					#if LOG_LEVEL > 3
					bfs_level[v] = bfs_level[u] + 1;
					bfs_vertices[bfs_level[v]]++;
					bfs_edges[bfs_level[v]] += row_offsets[v+1] - row_offsets[v];
					#endif

					if (v == t) {
						found = 1;
						#if LOG_LEVEL > 1
						printf("bfs: traversed vertices %d (%.0f%%), traversed edges %d (%.0f%%), ", size, (double)100.0*size/num_nodes, edges, (double)100*edges/num_edges);
						#endif
						
						#if LOG_LEVEL > 3
						printf("\n");
						for (int i = 0; i < bfs_level[v]; i++)
							printf("  bfs level %i: %i vertices, %i edges\n", i, bfs_vertices[i], bfs_edges[i]);
						#endif
						break;
					}
					q[size] = v;
					size++;
				}
			}
		}
		#if LOG_LEVEL > 3
		my_free(bfs_level);
		my_free(bfs_vertices);
		my_free(bfs_edges);
		#endif
	}

	return found;
}

