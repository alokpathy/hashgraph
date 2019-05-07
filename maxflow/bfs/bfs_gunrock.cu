// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

// Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.

/**
 * @file
 * test_bfs.cu
 *
 * @brief Simple test driver program for breadth-first search.
 */

#include <stdio.h>
#include <string>
#include <deque>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>
#include <gunrock/util/track_utils.cuh>

// BFS includes
#include <gunrock/app/bfs/bfs_enactor.cuh>
#include <gunrock/app/bfs/bfs_problem.cuh>
#include <gunrock/app/bfs/bfs_functor.cuh>

// Operator includes
#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>

#include <moderngpu.cuh>

// graph structure
#include "../matrix.h"

//Generic tools handling masks and fill
#include "bfs_tools.cu" 

using namespace gunrock;
using namespace gunrock::app;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::bfs;

void ref_bfs_mask(const int src_node, const int dst_node, const int num_nodes, const int num_edges, const int *row_offsets, const int *col_indices, const int *col_mask, int *parents)
{
  int *q = (int*)malloc(num_nodes * sizeof(int));
  q[0] = src_node;
  parents[src_node] = src_node;
  int idx = -1;
  int size = 1;
  int found = 0;
  while (idx+1 < size && !found) {
    idx++;
    int u = q[idx];
    for (int i = row_offsets[u]; i < row_offsets[u+1]; i++) {
      int v = col_indices[i];
      if (parents[v] == -1 && col_mask[i]) {
        parents[v] = u;
        if (v == dst_node) {
          found = 1;
          break;
        }
        else {
          q[size] = v;
          size++;
        }
      }
    }
  }
}

cudaError_t bfs_mask(int src_node, int dst_node, int num_nodes, int num_edges, int *row_offsets, int *col_indices, int *col_mask, int *parents)
{
#if 0
  // TODO: use Gunrock's customized BFS here
  ref_bfs_mask(src_node, dst_node, num_nodes, num_edges, row_offsets, col_indices, col_mask, parents);

  return cudaSuccess;
#else
  typedef int VertexId;
  typedef int SizeT;
  typedef int Value;
  typedef BFSProblem <VertexId,SizeT,Value,
                      true, // MARK_PREDECESSORS
                      true> // IDEMPOTENCE
                      Problem;
  typedef BFSEnactor <Problem> Enactor;

  cudaError_t retval = cudaSuccess;

  Info<VertexId, SizeT, Value> *info = new Info<VertexId, SizeT, Value>;


  info->InitBase2("BFS");
  ContextPtr *context = (ContextPtr*)info->context;
  cudaStream_t *streams = (cudaStream_t*)info->streams;

  int *gpu_idx = new int[1];
  gpu_idx[0] = 0;

  Problem *problem = new Problem(false, false); //no direction optimized, no undirected
  if (retval = util::GRError(problem->Init(
    false, //stream_from_host (depricated)
    row_offsets,
    col_indices,
    col_mask,
    parents,
    num_nodes,
    num_edges,
    1,
    NULL,
    "random",
    streams),
    "BFS Problem Init failed", __FILE__, __LINE__)) return retval;

  Enactor *enactor = new Enactor(1, gpu_idx);

  if (retval = util::GRError(enactor->Init(context, problem),
  "BFS Enactor Init failed.", __FILE__, __LINE__)) return retval;

  if (retval = util::GRError(problem->Reset(
  src_node, enactor->GetFrontierType()),
  "BFS Problem Reset failed", __FILE__, __LINE__))
  return retval;

  if (retval = util::GRError(enactor->Reset(),
  "BFS Enactor Reset failed", __FILE__, __LINE__))
  return retval;

  if (retval = util::GRError(enactor->Enact(src_node),
  "BFS Enact failed", __FILE__, __LINE__)) return retval;

  // free memory
  delete info;
  delete problem;
  delete enactor;

  return retval;
#endif
}

//BFS gunrock implementation
int bfs(csr_graph *g, int s, int t, int *q, int *p, int *mask)
{
  // set all vertices as undiscovered (-1)
  fill<-1><<<(g->n + 255)/256, 256>>>(g->n, p);
  cudaDeviceSynchronize();

  // setup mask, TODO: move this step inside Gunrock to reduce BW
  setup_mask<<<(g->nnz + 255)/256, 256>>>(g->nnz, mask, g->vals_cap, g->vals_flow);

  // run bfs (with mask)
  bfs_mask(s, t, g->n, g->nnz, g->row_offsets, g->col_indices, mask, p);

  // check if path exists
  return (p[t] != -1);  
}
// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
