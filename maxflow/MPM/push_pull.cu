
#include "../config.h"
#include "nvToolsExt.h"

//
// Pull/Push flow from/to source
//

#define MOVE_FLOW_DIM_X 32
#define MOVE_FLOW_DIM_Y 1

__host__ __device__ int imin_to_node_to_push(int imin, int nsg) {
	imin += 1;;
	return (imin < nsg) ? imin : (imin - nsg);	
}

template<typename F1, typename F2>
__global__ void first_push_pull(post_prune_data *d_ppd, 
			flow_t *d_total_flow,
			csr_graph g,
			int nsg,
			int *h,
			flow_t *e,
			int *node_sg_to_g,
			F1 f_node_push,
			F2 f_node_pull) {

	int node_to_push = imin_to_node_to_push(d_ppd->d_min.key, nsg); 
	flow_t flow_to_push = d_ppd->d_min.value;
	
	//Flag need to be reset before next prune
	d_ppd->prune_flag = 0;

	int ithread = threadIdx.y * blockDim.x + threadIdx.x;
	
	if(ithread == 0) {	
		*d_total_flow += flow_to_push;
		//printf("---------- pushing %i with %f -  gn=%i \n", node_to_push, flow_to_push, nsg);
	}

	switch(threadIdx.y) {
		case 0: //push
		f_node_push(node_to_push, flow_to_push);
		break;
		case 1: //pull
		f_node_pull(node_to_push, flow_to_push);
		break;
	}
}

template<int THREADS_VERTEX>
__device__ bool edge_op_move_flow(const int from, 
				const int to, 
				const int i_edge, 
				flow_t &to_push, 
				flow_t *degree_to,
				char *edge_mask,
				char *edge_mask_orig,
				char *prune_mask,
				flow_t *cf,
				flow_t *e,
				const int *reverse_edge_map,
				const int sg_t) {

	flow_t cf_edge = (i_edge != -1 && edge_mask[i_edge]) ? cf[i_edge] : 0;

	//
	// Exclusive sum of edges available capacities (cf = cap - flow)
	// If exclusive sum is >= to_push -> nothing to do for this thread
	// Else if exclusive + cf_edge <= to_push do a full push
	// Else do a partial push of to_push - exclusive
	//

	typedef cub::WarpScan<flow_t,THREADS_VERTEX> WarpScan;
	__shared__ typename WarpScan::TempStorage temp_storage[512/THREADS_VERTEX]; //TODO size, multiple thread
	int ithread = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);
	int i_logical_warp = ithread / THREADS_VERTEX;
	flow_t aggregate, exclusive_sum;
	exclusive_sum = cf_edge;
	WarpScan(temp_storage[i_logical_warp]).ExclusiveSum(exclusive_sum, exclusive_sum, aggregate);	
	//printf("(%i,W:%i,XD:%i) U:%i Value:%f Scan:%f \n", threadIdx.x, i_logical_warp, THREADS_VERTEX, from, cf_edge, exclusive_sum);
	//Do not kill the threads with cf_edge = 0
	//We need to update to_push	

	//if(i_edge != -1)
	//printf("raw : %i -> %i : %i : %f (%f, %i)  \n", from, to, i_edge, cf_edge, cf[i_edge], edge_mask[i_edge]);

	if(!isZero(cf_edge)) {
		flow_t local_push = 0;
		int rev_i_edge;
		if(exclusive_sum < to_push) { 
			local_push = min(cf_edge, to_push - exclusive_sum);
			rev_i_edge = reverse_edge_map[i_edge];
		}
		if(isZero(cf_edge - local_push)) {
			//i_edge is now saturated
			cf[i_edge] = 0;
			//printf("edge %i is going down \n", i_edge);
			edge_mask[i_edge] = 0;	
			edge_mask_orig[i_edge] = 0;
	
			//rev_i_edge cant be in layer graph (only edges to next level)	
			cf[rev_i_edge] += cf_edge;	
			edge_mask_orig[rev_i_edge] = 1;

		} else if(local_push > 0) {
			//partial push on i_edge
			cf[i_edge] -= local_push;
			cf[rev_i_edge] += local_push;	
			edge_mask_orig[rev_i_edge] = 1;
		}
		if(local_push > 0) {
				
			//printf("moving %f from %i to %i - exlu sum : %f  \n", local_push, from, to, exclusive_sum);				

			//Assign local_push flow to to

			//Multiple nodes on the same level can have an edge going to to
			//We need atomics here
			
			//We don't push to s or t
			//Avoiding useless atomics + avoiding memset (to set e[t'] = 0 at the end)
			if(to != 0 && to != sg_t) 
				atomicAdd(&e[to], local_push);

			//if(MOVE_FLOW_MASK)	
			//	move_flow_mask[to] = 1;		

			//Multiple nodes on the same level can have an edge going to to
			//We need atomics here
			flow_t new_degree_to   = atomicAdd(&degree_to[to], -local_push) - local_push; //atomicAdd is postfix
			if(isZero(new_degree_to)) 	
				prune_mask[to] 	 = 1;
			//printf("new degree from %i %f, to %i %f \n", from, degree_from[from], to, new_degree_to);

		}
	}

	to_push -= aggregate;
	return (to_push <= 0); //we're done if nothing left to push
}




void MPM::push_and_pull() {
	auto f_edge_push = [*this] __device__ (const int from, const int to, const int i_edge, flow_t &to_push) {
		edge_op_move_flow<MOVE_FLOW_DIM_X>(from, to, i_edge, to_push, degree_in, edge_mask, edge_mask_orig, prune_mask, cf, e, reverse_edge_map, sg_in.n-1);			
	};
	
	auto f_edge_pull = [*this] __device__ (const int from, const int to, const int i_edge, flow_t &to_pull) {
		edge_op_move_flow<MOVE_FLOW_DIM_X>(from, to, i_edge, to_pull, degree_out, edge_mask, edge_mask_orig, prune_mask, cf, e, reverse_edge_map, sg_in.n-1);			
	};
	
	auto f_node_push = [*this, f_edge_push] __device__ (const int u, flow_t to_push) {
		if(isZero(to_push)) return; //it is an exact 0
		//printf("will push %f from %i \n", to_push, u);
		flow_t pushed = to_push;
		iterate_on_edges(u, sg_out, f_edge_push, to_push);

		//printf("(%i) Post Push \n", threadIdx.x);
		if(threadIdx.x == 0) {
			flow_t new_degree = (degree_out[u] -= pushed);
			//printf("%p new degree out[%i] = %f \n", degree_out, u, new_degree);
			if(isZero(new_degree))	{
				prune_mask[u] = 1;
			} 
			e[u] = 0; 
		}

	};

	auto f_node_pull = [*this, f_edge_pull] __device__ (const int u, flow_t to_pull) {
		if(isZero(to_pull)) return; //it is an exact 0
		//printf("will pull %f from %i \n", to_pull, u);
		flow_t pulled = to_pull;
		iterate_on_edges(u, sg_in, f_edge_pull, to_pull);

		if(threadIdx.x == 0) {
			flow_t new_degree = (degree_in[u] -= pulled);
			//printf("%p new degree in[%i] = %f \n", degree_in, u, new_degree);
			if(isZero(new_degree))	{ 
				prune_mask[u] = 1; 
			}

			e[u] = 0;
		}

	};
	
	auto f_node_push_e = [*this, f_node_push] __device__ (const int u) {
		f_node_push(u, e[u]);
	};
	
	auto f_node_pull_e = [*this, f_node_pull] __device__ (const int u) {
		f_node_pull(u, e[u]);
	};

	nvtxRangePushA("push_pull");	
	

	//Launching first push pull
	dim3 sgrid, sblock;	
	sgrid.x =1, sgrid.y = 1, sgrid.z = 1;
	sblock.x = 32, sblock.y = 2, sblock.z = 1;

	first_push_pull<<<sgrid,sblock,0,st1>>>(d_ppd,
				d_total_flow,
				g,
				sg_in.n,
				h,
				e,
				node_sg_to_g,
				f_node_push,
				f_node_pull);
	
	//Computing h[node_to_push] in the meantime
	kvpid h_min = h_ppd.d_min;
	int node_to_push = imin_to_node_to_push(h_min.key, sg_in.n); 
	//printf("on host : %i (%f) \n", node_to_push, h_min.value);
	int level_node_to_push = 0;
	//could do a binary search
	while(node_to_push >= sg_level_offsets[level_node_to_push+1])
		level_node_to_push++;	

	dim3 grid, block;
	block.x = MOVE_FLOW_DIM_X;	
	block.y = MOVE_FLOW_DIM_Y;
	
	grid.x = 1;
	grid.y = 1;
	//cudaEvent_t pull_done;	
	//cudaEventCreate(&pull_done);

	iterate_on_levels<FORWARD>(level_node_to_push+1, ht-1, sg_level_offsets, f_node_push_e, grid, block, 1, 0, st1, max_level_width);
	
	iterate_on_levels<BACKWARD>(level_node_to_push-1, 1, sg_level_offsets, f_node_pull_e, grid, block, 1, 0, st1, max_level_width);

	//cudaStreamWaitEvent(st1, pull_done, 0);	
	//cudaEventRecord(pull_done, st2);	
	

	nvtxRangePop();

	
}



