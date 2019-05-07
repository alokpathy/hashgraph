#include "MPM.h"
#include "../matrix.h"

double maxflowimplementation(csr_graph* g, int s, int t, float *elapsed_time) {
	MPM mpm(*g);
	
	return mpm.maxflow(s,t,elapsed_time);
}
