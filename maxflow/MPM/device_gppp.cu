//Meta operation
//loop find node/push pull/prune in device mem



#define RPPP_X 32 
#define RPPP_Y 16
#define RPPP_NTHREADS (RPPP_X * RPPP_Y)

__global__ void reduce_push_pull_prune(csr_graph g, 
					csr_subgraph sg_in,
					 csr_subgraph sg_out, 
					int ht,
					int *s_t_pruned,
					double *degree_in,
					double *degree_out,
					int *h,
					int *node_sg_to_g,
					char *edge_mask,
					char *edge_mask_orig,
					int *reverse_edge_map,
					double *cf,
					double *e,
					char *prune_mask,
					char *have_been_pruned,
					int *sg_level_offsets,
					double *d_total_flow) {
	int n = sg_in.n;
	/*
	__shared__ char s_t_pruned;
	__shared__ double degree_in[25];
	__shared__ double degree_out[25];

	//use __ballot and bitset	
	__shared__ char edge_mask[12]; // bank conflits - use ballot for writing
	//Dot not use mask for nodes for now - too complicated 
	//Not useful for 32-bits vals
	//Load offsets ?

	//only one thread can set a bit to 1, and random access : bitset with atomicOr 
	//bitset
	__shared__ char prune_mask[12];	
	__shared__ char have_been_pruned[12];	
	__shared__ double e[12];
	*/

	//Init
	int ithread = threadIdx.x + (blockDim.x) * threadIdx.y;

	if(ithread == 0) {
		*s_t_pruned = 0;
	}	
	__syncthreads();	
	//End init

	do {
		//
		// Step 1 : Find node to push
		//
		__shared__ int node_to_push;
		__shared__ double flow_to_push;
		device_get_node_to_push<RPPP_X,RPPP_Y>(degree_in, 
							degree_out, 
							n, 
							node_to_push, 
							flow_to_push,
							ithread,
							d_total_flow);

		int level_node_to_push = h[node_sg_to_g[node_to_push]];

		__syncthreads();
		//
		// Step 2 : Push/Pull
		//

		device_push_pull<RPPP_X,RPPP_Y>(degree_in,
				degree_out,
				edge_mask,
				edge_mask_orig,
				reverse_edge_map,
				cf,
				e,
				prune_mask,
				node_to_push,
				flow_to_push,
				level_node_to_push,
				sg_in,
				sg_out,
				sg_level_offsets,
				ht,
				ithread);
	
		__syncthreads();
		
		//
		// Step 3 : Prune 
		//
		device_prune<RPPP_X,RPPP_Y>(ithread, 
			RPPP_NTHREADS,
			degree_in,
			degree_out,
			cf,
			prune_mask,
			have_been_pruned,
			edge_mask,
			sg_in,
			sg_out,
			s_t_pruned);
			
		__syncthreads();

		break;
	} while(!*s_t_pruned);
}

