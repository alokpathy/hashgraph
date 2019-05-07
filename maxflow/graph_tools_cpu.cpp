// Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.

template<typename T>
void fill(int size, T *data, T value) {
	for(int i=0; i != size; ++i) 
		data[i] = value;
}

void setup_mask_unsaturated(int num_edges, char *mask, double *cf) {
	for(int i=0; i != num_edges; ++i)
		mask[i] = (cf[i] > 0);
}

void setup_mask_unsaturated_backward(int num_edges, int *mask, double *cf, int *reverse_edge) {
	for(int i=0; i != num_edges; ++i)
		mask[reverse_edge[i]] = (cf[i] > 0);
}


template void fill<int>(int,int*,int);
template void fill<double>(int,double*,double);
