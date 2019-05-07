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

//Generic tools handling fill
#include "../graph_tools.h" 

#include "bfs.h"

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

int bfs(int *row_offsets, int *col_indices, int num_nodes, int num_edges, int src_node, int dst_node, int *q, int *output, int output_type, int *col_mask)
{
  fill(num_nodes, output, -1);
  cudaDeviceSynchronize();

  bool mark_pred = (output_type == BFS_MARK_PREDECESSOR);
#if 0
  // TODO: use Gunrock's customized BFS here
  ref_bfs_mask(src_node, dst_node, num_nodes, num_edges, row_offsets, col_indices, col_mask, parents);

  return cudaSuccess;
#else
  typedef int VertexId;
  typedef int SizeT;
  typedef int Value;
  typedef BFSProblem <VertexId,SizeT,Value,
                      false, // MARK_PREDECESSORS
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
    output,
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

   if (retval = util::GRError(problem->Extract(output, NULL),
  "BFS Extract failed", __FILE__, __LINE__)) return retval;
 

  // free memory
  delete info;
  delete problem;
  delete enactor;
  //check if path exists


	//MAX_INT default value for src dis TODO	
   return (dst_node >= 0 && dst_node < num_nodes) && (output[dst_node] != -1);  
#endif
}


// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
