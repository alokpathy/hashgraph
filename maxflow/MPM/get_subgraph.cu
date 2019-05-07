#include "../cub/cub/cub.cuh"
#include "../utils.cu"

#include "../config.h"
#include "nvToolsExt.h"

#define INIT_G_BLOCK_X 32 
#define WRITE_EDGES_DIM_X 32 
#define NTHREADS 512

__global__ void reverse_hash(int *reversed, int *hash, int num_items) {
	for(int u = blockDim.x * blockIdx.x + threadIdx.x;
		u < num_items;
		u += blockDim.x * gridDim.x) {

		reversed[hash[u]] = u;
	}	
}

void MPM::write_edges() {
	auto f_edge = [*this] __device__ (const int from, 
					const int to, 
					const int i_edge, 
					flow_t &degree_in_thread, 
					flow_t &degree_out_thread, 
					int &in_edge_offset,
					int &out_edge_offset) {


		int rev_i_edge = (i_edge != -1) ? reverse_edge_map[i_edge] : -1;	
	
		int is_valid_out_edge = (i_edge != -1) && edge_mask[i_edge];	
		int is_valid_in_edge  = (rev_i_edge != -1) && edge_mask[rev_i_edge]; 		
		
		typedef cub::WarpScan<int,WRITE_EDGES_DIM_X> WarpScan;
		__shared__ typename WarpScan::TempStorage temp_storage_scan[NTHREADS/WRITE_EDGES_DIM_X];
	
		// Compute exclusive warp-wide prefix sums
		
		int ithread = threadIdx.x + blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;
		int warpid = ithread / WRITE_EDGES_DIM_X; 

		int idx_in_edge = is_valid_in_edge, idx_out_edge = is_valid_out_edge;		
		int n_in_edge_in_warp, n_out_edge_in_warp;
		WarpScan(temp_storage_scan[warpid]).ExclusiveSum(idx_in_edge,idx_in_edge, n_in_edge_in_warp);		
		__syncthreads();
		WarpScan(temp_storage_scan[warpid]).ExclusiveSum(idx_out_edge,idx_out_edge, 	n_out_edge_in_warp);		
		
		//printf("(%i) u:%i edge:%i ; is in:%i (scan:%i) ; is out:%i (scan:%i) \n", threadIdx.x, from, i_edge, is_valid_in_edge, idx_in_edge, is_valid_out_edge, idx_out_edge);
		//scan is done, lets return inactive edges
		//printf("u=%i, tx=%i, active=%i, sum=%i, i_edge=%i \n", from, threadIdx.x, is_edge_active, write_idx_thread, i_edge);
		
		if(is_valid_out_edge) {
			//Computing degree
			degree_out_thread += cf[i_edge];
			
			//Writing edges
			int write_idx = out_edge_offset + idx_out_edge;
			sg_out.parent_edge_indices[write_idx] = i_edge;
			sg_out.col_indices[write_idx] = node_g_to_sg[to]; 
			
			//printf("(%i,%i,%i) writing edge=%i (ig:%i)  %i -> %i (g:%i) \n", threadIdx.x, threadIdx.y, threadIdx.z, write_idx, i_edge, from, g.node_g_to_sg[to], to); 	
		} else if(is_valid_in_edge) {
			degree_in_thread += cf[rev_i_edge];
			
			//Writing edges
			int write_idx = in_edge_offset + idx_in_edge;
			sg_in.parent_edge_indices[write_idx] = rev_i_edge;
			sg_in.col_indices[write_idx] = node_g_to_sg[to]; 
		}
		
		out_edge_offset += n_out_edge_in_warp;
		in_edge_offset  += n_in_edge_in_warp;
		
	};

	auto f_node = [*this, f_edge] __device__ (const int u) {
		typedef cub::WarpReduce<flow_t> WarpReduce;
		__shared__ typename WarpReduce::TempStorage temp_storage_reduce[NTHREADS/WRITE_EDGES_DIM_X];
		flow_t in_degree_thread = 0, out_degree_thread = 0;
		int out_edge_offset = sg_out.edge_offsets[u], in_edge_offset = sg_in.edge_offsets[u];
		int u_g = node_sg_to_g[u];//u in g;
		
		iterate_on_edges(u_g, g, f_edge, in_degree_thread, out_degree_thread, in_edge_offset, out_edge_offset);	

		int ithread = threadIdx.x + blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;
		int warpid = ithread / WRITE_EDGES_DIM_X; 

			
		flow_t total_in_degree 	= WarpReduce(temp_storage_reduce[warpid]).Sum(in_degree_thread);
		__syncthreads();
		flow_t total_out_degree 	= WarpReduce(temp_storage_reduce[warpid]).Sum(out_degree_thread);
		__syncthreads();
	
		if(threadIdx.x == 0) {
			degree_in[u] = total_in_degree;
			degree_out[u] = total_out_degree;
		}
	};


	dim3 grid,block;

	block.x = WRITE_EDGES_DIM_X;
	block.y = 1;
	grid.x = 1;
	grid.y = 1;	
	

	apply_on_graph(sg_in.n, f_node, grid, block, 0, st1);
}

void MPM::init_level_graph(int &nsg) {
	nvtxRangePushA("get_subgraph");	
	
	cudaMemset(edge_mask, 0,  sizeof(char) * g.nnz);	
	cudaMemset(node_mask, 0, sizeof(char) * g.n);	
	cudaMemset(queue_mask, 0, sizeof(char) * g.n);	
	cudaMemset(d_ppd, 0, sizeof(post_prune_data)); //resetting s_t_pruned and prune_flag
	
	auto f_edge_init = [*this] __device__ (const int from, const int to, const int i_edge, int *n_in_edges, int *n_out_edges) {
		if(i_edge != -1) {
			int hto 	= h[to];
			int hfrom 	= h[from];
			if((hto + 1) == hfrom) { // going backward
				int rev_i_edge = reverse_edge_map[i_edge];
				
				if(edge_mask_orig[rev_i_edge]) {
					edge_mask[rev_i_edge] = 1; //the edge is part of subgraph
					node_mask[to] = 1; //to is part of subgraph
					atomicAdd(n_in_edges, 1);
					
				}
			} else if(edge_mask[i_edge] && (hfrom + 1) == hto) { //going forward
				atomicAdd(n_out_edges, 1);
			}
		}
	};

	
	auto f_node_init = [*this, f_edge_init] __device__ (int idx, int u) {
		if(!node_mask[u]) return;	
		
		if(threadIdx.x == 0) {	
			sg_in.edge_offsets[idx] = 0;
			sg_out.edge_offsets[idx] = 0;
		}
		__syncthreads(); //this code shouldnt diverge

		iterate_on_edges(u, g, f_edge_init, &sg_in.edge_offsets[idx], &sg_out.edge_offsets[idx]);

		if(threadIdx.x == 0) {	
			queue_mask[idx] = 1;
			//printf("Node %i, in=%i, out=%i \n", u, sg_in.edge_offsets[idx], sg_out.edge_offsets[idx]);
		}	
		//printf("node %i, direction %i, sum %f \n", u, direction, degree_set[u]);		
	
	};

	//localgrid and block : what we actually need in our lambdas, the z-axis will be use and defined internaly in iterate_on_levels
	dim3 localGrid(1, 1);
	dim3 localBlock(INIT_G_BLOCK_X, 1);

	char one = 1;
	cudaMemcpy(&node_mask[t], &one, sizeof(char), cudaMemcpyHostToDevice); 
	iterate_on_levels<BACKWARD>(ht,
				0,
				q_bfs,
				bfs_offsets, 
				f_node_init,
				localGrid, 
				localBlock,
				1, // nodes per thread
				0,
				0);
	//Creating new list containing only vertices in layered graph

		//TODO we dont need g.n, we just need bfs_offsets[ht...]

	cudaEvent_t nsg_on_host;

	//List nodes in layer
	PartitionFlagged(node_sg_to_g, q_bfs, queue_mask, g.n, d_nsg, d_storage_partition, size_storage_partition);
	cudaMemcpy(&nsg, d_nsg, sizeof(int), cudaMemcpyDeviceToHost);
	//in edge count in layer
	PartitionFlagged(buf1, sg_in.edge_offsets, queue_mask, g.n, d_nsg, d_storage_partition, size_storage_partition);
	//out edge count in layer
	PartitionFlagged(buf2, sg_out.edge_offsets, queue_mask, g.n, d_nsg, d_storage_partition, size_storage_partition);


	sg_in.n = nsg;
	sg_out.n = nsg;
	//printf("nsg is %i \n", nsg);
	
	degree_in = degree;
	degree_out = degree + nsg;
	//printf("in layered graph : %i \n", nsg);
	//TODO we could resize sg here (n)
 
	//
	// Compute offsets
	//
	ExclusiveSum(sg_in.edge_offsets, buf1, nsg+1, d_storage_exclusive_sum, size_storage_exclusive_sum); //we dont care about whats in g.buf[nsg] (exclusive sum), but we need the nsg+1 first elements of the sum
	ExclusiveSum(sg_out.edge_offsets, buf2, nsg+1, d_storage_exclusive_sum, size_storage_exclusive_sum);

	//TODO we could resize sg here (nnz)

	/*	
		
	cudaDeviceSynchronize();
	for(int i=0; i != nsg; ++i) {
		printf("sg_node %i (h=%i), g_node %i, in_edge %i, out_edge %i offset in %i, offset out %i \n",
			i,
			h[node_sg_to_g[i]],
			node_sg_to_g[i],
			buf1[i],
			buf2[i],
			sg_in.edge_offsets[i],
			sg_out.edge_offsets[i]);

	}
	*/

	dim3 block, grid;
	block.x = 256;
	grid.x = min((nsg + block.x - 1)/block.x, N_MAX_BLOCKS);
	reverse_hash<<<grid,block>>>(node_g_to_sg, node_sg_to_g, nsg);

	//TODO cudaLaunch returns 0x7 in write_edges
	write_edges();

/*		
	cudaDeviceSynchronize();
	printf("bfs off : \n");
	for(int i=0; i!=ht+2; ++i)
		printf("%i\t", bfs_offsets[i]);
	printf("\n");
*/	

	SegmentedReduce(buf1, queue_mask, bfs_offsets, ht+2);	
	
	

/*		
	printf("counts : \n");
	for(int i=0; i!=ht+2; ++i)
		printf("%i\t", buf1[i]);
	printf("\n");
		
*/
	ExclusiveSum(sg_level_offsets, buf1, ht+2, d_storage_exclusive_sum, size_storage_exclusive_sum);
	
	cudaDeviceSynchronize();
	//CPU pagefaults
	max_level_width = 0;
	for(int i=0; i != (ht+1); ++i)
		max_level_width = max(max_level_width, sg_level_offsets[i+1] - sg_level_offsets[i]);
	//printf("max level width : %i \n", max_level_width);

	/*	
	cudaDeviceSynchronize();
	printf("SCAN : \n");
	for(int i=0; i!=ht+2; ++i)
		printf("%i\t", g.sg_level_offsets[i]);
	printf("\n");
		


	
	cudaDeviceSynchronize();
	printf("levels offsets: \n");
	for(int i=0; i != ht+2; ++i)
		printf("(%i) %i : %i\n", i, buf1[i], sg_level_offsets[i]);
	printf("\n");
	*/	
/*

	for(int i=0; i != nsg; ++i) {
		printf("sg_node %i (h=%i), g_node %i, din=%f, dout=%f \n out edges : ",
			i,
			h[node_sg_to_g[i]],
			node_sg_to_g[i],
			degree_in[i],
			degree_out[i]);

		for(int i_edge = sg_out.edge_offsets[i]; 
			i_edge != sg_out.edge_offsets[i+1];
			++i_edge)
			printf("%i (g:%i)\t", sg_out.col_indices[i_edge], node_sg_to_g[sg_out.col_indices[i_edge]]);

		printf("\n in edges :");
		for(int i_edge = sg_in.edge_offsets[i]; 
			i_edge != sg_in.edge_offsets[i+1];
			++i_edge)
			printf("%i (g:%i)\t", sg_in.col_indices[i_edge], node_sg_to_g[sg_in.col_indices[i_edge]]);

		printf("\n");	

	}

*/	
	//TODO we use it in prune
	cudaMemset(queue_mask, 0, sizeof(char) * g.n);		
	cudaDeviceSynchronize();

	nvtxRangePop();	
}
