#include <boost/config.hpp>
#include <iostream>
#include <string>
#include <boost/graph/push_relabel_max_flow.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/read_dimacs.hpp>
#include <boost/graph/graph_utility.hpp>
#include <time.h>
#include <fstream>

using std::istream;

int main() {
	using namespace boost;

	typedef adjacency_list_traits<vecS, vecS, directedS> Traits;
	typedef adjacency_list<listS, vecS, directedS, 
		property<vertex_name_t, std::string>,
		property<edge_capacity_t, double,
		property<edge_residual_capacity_t, double,
		property<edge_reverse_t, Traits::edge_descriptor> > >
			> Graph;

	std::ifstream dimacs("../data/dimacs_autoexport.dat");

	Graph g;

	property_map<Graph, edge_capacity_t>::type 
		capacity = get(edge_capacity, g);
	property_map<Graph, edge_reverse_t>::type 
		rev = get(edge_reverse, g);
	property_map<Graph, edge_residual_capacity_t>::type 
		residual_capacity = get(edge_residual_capacity, g);

	Traits::vertex_descriptor s, t;
	read_dimacs_max_flow(g, capacity, rev, s, t, dimacs);


	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC, &start);

	long flow;
#if defined(BOOST_MSVC) && BOOST_MSVC <= 1300
	// Use non-named parameter version
	property_map<Graph, vertex_index_t>::type 
		indexmap = get(vertex_index, g);
	flow = push_relabel_max_flow(g, s, t, capacity, residual_capacity, rev, indexmap);
#else
	flow = push_relabel_max_flow(g, s, t);
#endif

	clock_gettime(CLOCK_MONOTONIC, &end);
	float time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9;

  double f_as_d = (double)flow / 1000000000.0;
  
 printf("max flow = %f\n", f_as_d);
  printf("time: %.3f s\n", time);


	/*
	   std::cout << "c flow values:" << std::endl;
	   graph_traits<Graph>::vertex_iterator u_iter, u_end;
	   graph_traits<Graph>::out_edge_iterator ei, e_end;
	   for (boost::tie(u_iter, u_end) = vertices(g); u_iter != u_end; ++u_iter)
	   for (boost::tie(ei, e_end) = out_edges(*u_iter, g); ei != e_end; ++ei)
	   if (capacity[*ei] > 0)
	   std::cout << "f " << *u_iter << " " << target(*ei, g) << " " 
	   << (capacity[*ei] - residual_capacity[*ei]) << std::endl;
	 */
	return 0;
}

