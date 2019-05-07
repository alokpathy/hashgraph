#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <assert.h>

#include "../allocator.h"
#include "../matrix_io.h"
#include "../matrix.h"
#include "../MPM/MPM.h"

#include <sstream>

#include "../boost-push-relabel/push-relabel.h"

using std::stringstream;

void do_benchmarks()
{
	vector<pair<int,int>> st_roadCA = {{1,1000}};

	map<string, vector<pair<int,int>>> todo;
	todo.insert({"data/roadNet-CA.mtx", st_roadCA);	

	for(auto& g_sts : todo) {
		string g_path = g_sts.first;
		vector<pair<int,int>> ls_s_t = g_sts.second;

		csr_graph g;
		// read capacity graph, generate symmetric entries
		read_mm_matrix(argv[1], &g.n, &g.n, &g.nnz, &g.row_offsets, &g.col_indices, &g.vals_cap);
		if (argc == 4)
			g.set_edge_weights_rcp_degree();


		//Using MPM
		MPM mpm(*g);

		for(auto st : ls_s_t) {
			int s = st.first;
			int t = st.second;
	
			stringstream dimacs;
			export_to_dimacs(dimacs, g, s, t);
	
			float pr_time, mpm_time;
			boost_push_relabel(dimacs, &pr_time);
			mpm.maxflow(s, t, &mpm_time);
		}		

		my_free(g.row_offsets);
		my_free(g.col_indices);
		my_free(g.vals_cap);
		mpm.clean();
	}
}


int main(int argc, char **argv)
{
	do_benchmarks();
}
