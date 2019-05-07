// Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../matrix.h"
#include "bfs.h"

//BFS CPU OMP implementation
int bfs(int *row_offsets, int *col_indices, int num_nodes, int num_edges, int s, int t, int *q, int *output, int output_type, char *mask, int* bfs_offsets)
{
  // set all vertices as undiscovered (-1)
  #pragma omp parallel for 
  for (int i = 0; i < num_nodes; i++) output[i] = -1;

  // start with source vertex
  q[0] = s;
  bool mark_pred = (output_type == BFS_MARK_PREDECESSOR);
  output[s] = mark_pred ? s : 0;
  
  int size = 1;
  int start_idx = 0;
  int end_idx = size;
  int found = 0;
  int bfs_level = 0;
  
  while(!found && start_idx < end_idx) {
    
    #pragma omp parallel for 
    for(int idx = start_idx; idx < end_idx; idx++) {
    int u = q[idx];
      
    for (int i = row_offsets[u]; i < row_offsets[u+1]; i++) {
        int v = col_indices[i];
          if(output[v] == -1 && mask[i]) {
	  if(mask[i] && __sync_val_compare_and_swap(&output[v], -1, mark_pred ? u : (bfs_level+1)) == -1) { 
            if (v == t) {
              found = 1;
              break;
            }
            int pos = __sync_fetch_and_add (&size, 1);
            q[pos] = v;
          }
	}
      }
    }

    start_idx = end_idx;
    end_idx = size;
    ++bfs_level;
  }
  return found;
}

