#include "MPM.h"


#include "nvToolsExt.h"

#include "../config.h"

#define PRUNE_DIM_X 8

//
//Prune : removing nodes will null throughput
//First level is detected while executing push/pull
//

__device__ bool edge_op_prune(const int node_pruned,
			const int node_to_update,
			const int i_edge,
			char *prune_mask,
			flow_t *degree_to_update,
			flow_t *cf,
			char *edge_mask,
			const csr_subgraph &sg_in,
			const csr_subgraph &sg_out,
			post_prune_data *d_ppd,
			const int cur_flag) {

	if(i_edge == -1 || !edge_mask[i_edge]) return false; //if this is not a real edge, quit now
	flow_t cf_edge = cf[i_edge];
	
	if(!isZero(cf_edge)) {
		edge_mask[i_edge] = 0;				
		flow_t new_degree = atomicAdd(&degree_to_update[node_to_update], -cf_edge) - cf_edge; //TODO shouldnt have atomics
		if(isZero(new_degree)) { 
			prune_mask[node_to_update] = 1;	
			d_ppd->prune_flag = cur_flag;
		}
	}


	return false; //we have to iterate through all edges, we're not done

}


template<typename F1, typename F2>
__device__ void node_op_prune(const int u,
			const csr_subgraph &sg_in,
			const csr_subgraph &sg_out,
			F1 f_forward_edge,
			F2 f_backward_edge,
			char *prune_mask,
			flow_t *in_degree,
			flow_t *out_degree,
			post_prune_data *d_ppd,
			const int flag) {

	if(!prune_mask[u]) return;			
	
	prune_mask[u] = 0;
	//have_been_pruned[u] = 1;
	//if(threadIdx.x == 0)
	//	printf("pruning %i \n", u);	

	if(u == 0 || u == (sg_in.n-1)) { //s is 0, t is sg_in-1
		d_ppd->s_t_pruned = 1;	return;	
	}

	//those two operations dont need to be serial
	//Deleting edges and updating neighbor's d_out
	//TODO warp div	
	iterate_on_edges(u, sg_out, f_forward_edge, flag);		
	iterate_on_edges(u, sg_in, f_backward_edge, flag);		
	
	in_degree[u] 	= FLOW_INF; 
	out_degree[u] 	= FLOW_INF;
}


void MPM::prune() {
	
	auto f_forward_edge = [*this] __device__ (const int node_pruned,
			const int node_to_update,
			const int i_edge,
			const int flag) {
		return edge_op_prune(node_pruned, node_to_update, i_edge, prune_mask, degree_in, cf, edge_mask, sg_in, sg_out, d_ppd, flag);	
	};	

	auto f_backward_edge = [*this] __device__ (const int node_pruned,
			const int node_to_update,
			const int i_edge,
			const int flag) {
		return edge_op_prune(node_pruned, node_to_update, i_edge, prune_mask, degree_out, cf, edge_mask, sg_out, sg_in, d_ppd, flag);	//TODO remove sg_in, sg_out
	};

	auto f_node_flag = [*this, f_backward_edge, f_forward_edge] __device__ (const int node, const int flag) {
		node_op_prune(node, sg_in, sg_out, f_forward_edge, f_backward_edge, prune_mask, degree_in, degree_out, d_ppd, flag);
	};


	nvtxRangePushA("prune");	

	//
	// End reduce
	//

	dim3 localBlock, localGrid;
	localBlock.x = PRUNE_DIM_X;
	localBlock.y = 1;
	
	localGrid.x = 1;
	localGrid.y = 1;

	bool done = false;
	int niters = 1; //do 3 iters at first
	int last_flag = 0;

	while(!done) {
		for(int it=0; it != niters; ++it) {
			++last_flag;
			auto f_node = [last_flag, f_node_flag] __device__ (const int node) {
				f_node_flag(node, last_flag);
			};
			apply_on_graph(sg_in.n, f_node, localGrid, localBlock, 0, st1); 
		}
		//bug on cub	

		//Find node to push
		cub::DeviceReduce::ArgMin(d_min_reduce, min_reduce_size, degree_in+1, &d_ppd->d_min, 2*(sg_in.n-1), st1);
		cudaMemcpyAsync(&h_ppd, d_ppd, sizeof(post_prune_data), cudaMemcpyDeviceToHost, st1);
		cudaStreamSynchronize(st1);
		done = (h_ppd.prune_flag != last_flag) || h_ppd.s_t_pruned;
		niters *= 2;
		//if(!done)
		//	printf("lets go again - iter=%i \n", niters);
	}
	

	//printf("s_t_pruned : %i \n", h_ppd.s_t_pruned);
	
	//TODO could be a simple memset on have_been_pruned if *s_t_pruned
	nvtxRangePop();
}



