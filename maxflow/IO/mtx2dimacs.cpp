#include "../matrix.h"
#include "matrix_io.h"

int main(int argc, char **argv)
{
	if(argc < 3) {
		printf("Usage : %s <input> <output> \n", argv[0]);
		exit(1);
	}

	csr_graph g;		// main graph structure

	// read mtx entry
	read_mm_matrix(argv[1], &g.n, &g.n, &g.nnz, &g.row_offsets, &g.col_indices, &g.vals_cap);

	//Set edges to degrees for cap
	g.set_edge_weights_rcp_degree();

	std::ofstream out(argv[2]);


	out << "p max " << g.n << " " << g.nnz << "\n";


	for(int u = 0; u < g.n; ++u) {
		for(int i_edge = g.row_offsets[u]; i_edge != g.row_offsets[u+1]; ++i_edge) { 
			int cap = (1000000000.0 * g.vals_cap[i_edge]);
			int v = g.col_indices[i_edge]; 
			out << "a " << u+1 << " " << v+1 << " " << cap  << "\n";		

			if(g.edge(v, u) == -1) {
				out << "a " << v+1 << " " << u+1 << " " << "0"  << "\n";		
			}
		}
	}

	return 0;
}
