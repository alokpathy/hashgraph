// Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.

/*
   LOG_LEVEL:
     0: no output
     1: one line result
     2: print augmented path stats
     3: print augmented path vertices
     4: bfs level stats for each path
*/

#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <assert.h>

#include "allocator.h"
#include "IO/gr_reader.h"
#include "matrix.h"

#include "config.h"

extern flow_t maxflowimplementation(csr_graph* g, int s, int t, float *time);

flow_t max_flow(int argc, char **argv)
{
  csr_graph g;		// main graph structure
  flow_t fm = 0;	// value of max flow


  read_gr(argv[1], g);
  // read source/target
  int s = atoi(argv[2])-1;
  int t = atoi(argv[3])-1;

  // setup flow network with zeros
  g.vals_flow = (flow_t*)my_malloc(g.nnz * sizeof(flow_t));
  memset(g.vals_flow, 0, g.nnz * sizeof(flow_t));
	
  // start timer

  float time;
  fm =  maxflowimplementation(&g, s, t, &time);

  // stop timer
  int fm_i = (int)fm;
  printf("max flow = %i\n", fm_i);
  printf("time: %.3f s\n", time);


  // write final flow network for debug purposes
  //write_csr(argv[4], f.n, f.n, f.nnz, f.rows, f.cols, f.vals);

  // free memory
  my_free(g.row_offsets);
  my_free(g.col_indices);
  my_free(g.vals_cap);
  my_free(g.vals_flow);
 
  return fm;
}

void print_help()
{
  printf("Usage: ./maxflow <input matrix file> <source id> <target id> [<random seed>]\n");
  printf("       if random seed is not specified the weights are set as 1/degree for each vertex\n");
}

int main(int argc, char **argv)
{
  if (argc < 4)
    print_help();
  else{
    max_flow(argc, argv);  
}
return 0;
}
