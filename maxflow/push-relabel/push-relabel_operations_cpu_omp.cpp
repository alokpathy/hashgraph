#include <climits>
#include <algorithm>
#include <functional>

#define DISCHARGE_CYCLES 1000

//returns true is the node is active
inline bool is_active(int u, int* h, double* e, int num_nodes) {
	return (h[u] < num_nodes && e[u] > 0);
}

void discharge(double *cf, int num_nodes, int s, int t, int *row_offsets, int *col_indices, int *reverse, double *e, int *h, int *q) {
	for(int it=0; it < DISCHARGE_CYCLES; ++it) {
	int size = 0;	
	#pragma omp parallel for
	for(int u=0; u < num_nodes; ++u)
		if(u != s && u != t) {
			if(!is_active(u, h, e, num_nodes)) continue;
			if(size != 0 && h[u] > h[q[0]]) //if the new element is higher than those in the queue
				size = 0; //reset queue (highest elements first)
			int pos = __sync_fetch_and_add(&size, 1);
			q[pos] = u;
		}
 	if(!size) break;
	

	//TODO floating points comparaison		
	#pragma omp parallel for
	for(int i=0; i < size; ++i) {
		int u = q[i];
		while(e[u] > 0) {
			int hu = h[u]; //height of u	
			double eu = e[u]; //excess of u

			#if LOG_LEVEL > 3
			printf("%i is active with e = %f, h=%i\n", u, eu, hu);	
			#endif

			//Looking for lowest neighbor
			int hp = INT_MAX; 
			int vp;
			int uvp;
			double cf_uvp;

			//TODO parallel reduction ?
			//row_offsets in shared ?
			for (int i = row_offsets[u]; i < row_offsets[u+1]; ++i) {
				double cf_uv = cf[i];
				int v = col_indices[i];
				int hpp = h[v];
				if(cf_uv <= 0.0 || hpp == num_nodes) continue;

				if(hpp < hp) {
					vp = v;
					hp = hpp;
					uvp = i;
					cf_uvp = cf_uv;
				}
			}

			if(hp == INT_MAX) break; //No outgoing edges (can happen if the node is linked to the source)
			
			//If pushing is possible, push
			if(hu > hp) {
				//pushing from u to vp
				double d = std::min(eu, cf_uvp);
				

				#if LOG_LEVEL > 3
				printf("Pushing %f/%f from %i to %i, h = %i and %i, eu = %f \n", d,cf_uvp, u, vp, hu, hp, eu);
				#endif

				//sync fetch and add doesnt work on doubles
				#pragma omp atomic	
				cf[reverse[uvp]] += d;
				#pragma omp atomic	
				cf[uvp] -= d;
				#pragma omp atomic	
				e[vp] += d;
				#pragma omp atomic	
				e[u] -= d;

			} else { //if we can't push, relabel to lowest + 1 (key to the lock free algo)
				h[u] = hp + 1;
				#if LOG_LEVEL > 3
				printf("Relabeling %i  from %i to %i, excess was %f \n", u, hu, hp+1, eu);
				#endif
			}			
		}
	}
	}
}

void remove_violating_edges(double *cf, int num_nodes, int *row_offsets, int *col_indices, int *reverse, double *e, int *h) {
	#pragma omp parallel for
	for(int u=0; u < num_nodes; ++u) {

		for (int i = row_offsets[u]; i < row_offsets[u+1]; ++i) {
			int v = col_indices[i];
			double cf_uv = cf[i];
			if(cf_uv > 0 && h[u] > h[v] + 1) {
				
				#if LOG_LEVEL > 3
				printf("%i is violating edge \n", u);	
				#endif

				int vu = reverse[i];
				#pragma omp atomic	
				e[u] -= cf_uv;
				#pragma omp atomic	
				e[v] += cf_uv;
				#pragma omp atomic	
				cf[vu] += cf_uv;
				
				cf[i] = 0;
			}
		}
	}
}
 
void update_excess_total(int *h, double *e, int num_nodes, int *node_mask, double *ExcessTotal) {
	#pragma omp parallel for
	for(int u=0; u < num_nodes; ++u) {
		if(h[u] != -1) continue;
		h[u] = num_nodes;
		if(node_mask[u]) continue; 
		node_mask[u] = 1;
		if(e[u] <= 0.0)  continue;
		#pragma omp atomic	
		*ExcessTotal -= e[u];

		#if LOG_LEVEL > 3
		printf("Removing edge with excess : %f \n", e[u]);
		#endif
	}

}
	
