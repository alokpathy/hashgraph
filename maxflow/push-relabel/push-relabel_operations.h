bool discharge(double *cf, int num_nodes, int s, int t, int *row_offsets, int *col_indices, int *reverse, double *e, int *h, int *q, int *active_blocks);

void remove_violating_edges(double *cf, int num_nodes, int *row_offsets, int *col_indices, int *reverse, double *e, int *h, int *active_blocks);
 
void global_gap_relabeling_post_bfs(int *global_relabel_h, int* h, double *e, int num_nodes);


//GPU conf
//Keeps track of active nodes
//Execute only those that are actives
#define ACTIVE_FLAGS

#ifdef USE_GPU
int blockIdyForNode(int u);
#endif 
