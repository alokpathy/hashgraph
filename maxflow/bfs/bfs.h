#pragma once

#include <vector>

using std::vector;
//Output types

#define BFS_MARK_PREDECESSOR 0
#define BFS_MARK_DEPTH 1

int bfs(int *row_offsets, int *col_indices, int num_nodes, int num_edges, int src_node, int dst_node, int *q, int *output, int output_type, char *col_mask, int *bfs_offsets);

