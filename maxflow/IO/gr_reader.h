#include <fstream>
#include <string>
#include "../allocator.h"
#include "../matrix.h"
#include "../config.h"
#include <cstdlib>
#include <iostream>

using std::string;
using std::ifstream;
using std::cout;

void read_gr(const string& filename, csr_graph& g) {
	ifstream iff(filename);
	uint64_t header[4];
	
	iff.read((char*)header, sizeof(uint64_t) * 4);

	uint64_t n, nnz;
	n = header[2];
	nnz = header[3];

	uint64_t *degrees = (uint64_t*)malloc(sizeof(uint64_t) * n);

	iff.read((char*)degrees, sizeof(uint64_t) * n);

	uint32_t *outs = (uint32_t*)malloc(sizeof(uint32_t) * nnz);
	iff.read((char*)outs, sizeof(uint32_t) * nnz);

	//Inc Sum -> Ex sum
	int last = 0;
	for(int i=0; i != n; ++i) {
		int here = degrees[i] - last;
		last = degrees[i];
		degrees[i] -= here;		
	}	

	uint32_t buf;
	if(nnz & 1)
		iff.read((char*)&buf, sizeof(uint32_t)); //align on 64 bits

	uint32_t *w = (uint32_t*)malloc(sizeof(uint32_t) * nnz);
	iff.read((char*)w, sizeof(uint32_t) * nnz);

	
	//Copying into g 
	//If the data types are coherent, we could load directly in those

	g.row_offsets = (int*)my_malloc(sizeof(int) * n);	
	g.col_indices = (int*)my_malloc(sizeof(int) * nnz);	
	g.vals_cap    = (flow_t*)my_malloc(sizeof(flow_t) * nnz);	

	g.n = n;
	g.nnz = nnz;


	for(int i=0; i != n; ++i)
		g.row_offsets[i] = (int)degrees[i];
	
	for(int i=0; i != nnz; ++i)
		g.col_indices[i] = (int)outs[i];

	for(int i=0; i != nnz; ++i)
		g.vals_cap[i] = (flow_t)w[i];

	free(degrees);
	free(outs);
	free(w);
	/*
	for(int u=0; u != 20; ++u) {
		cout << "Node " << u << " : ";
		for(int i = g.row_offsets[u]; i != g.row_offsets[u+1]; ++i) {
			cout << g.col_indices[i] << "(" << g.vals_cap[i] << ")\t";

		}
		cout << "\n";
	}
	*/
}
