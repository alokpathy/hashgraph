#include "MPM.h"
#include "../utils.cu"



void MPM::get_node_to_push() {	

	//printf("height winner : %i \n", *h_h_node_to_push);
}

template<int BLOCK_DIM_X, int BLOCK_DIM_Y=1, int BLOCK_DIM_Z=1>
__device__ void device_get_node_to_push(double *in_degree,
				double *out_degree,
				int n,
				int &node_to_push,
				double &flow_to_push,
				const int ithread,
				double *d_total_flow) {
	cub::ArgMin argmin;
	kvpid argmin_in = blockArgMin<BLOCK_DIM_X,BLOCK_DIM_Y,BLOCK_DIM_Z>(in_degree+1, n-1); //avoiding source	
	argmin_in.key += 1;
	kvpid argmin_out = blockArgMin<BLOCK_DIM_X,BLOCK_DIM_Y,BLOCK_DIM_Z>(out_degree, n-1); 
	
	if(ithread == 0) {
		kvpid m = argmin(argmin_in, argmin_out);

		node_to_push = m.key;
		flow_to_push = m.value;
		*d_total_flow += flow_to_push;
		printf("-> pushing %i with %f (in d) \n", node_to_push, flow_to_push);
	}
}
