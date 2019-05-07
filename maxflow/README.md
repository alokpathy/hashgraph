# Maxflow

This framework implements various maximum flow algorithms on GPU.

## Motivation

Existing GPU graph libraries like Gunrock and nvGraph are both missing a few important graph primitives including the maximum flow, which is frequently used in network analysis, image segmentation, clustering, bipartite matching, and other problems. There is also an interesting application of maximum flow algorithm to community detection problem in social networks. There are a lot of algorithms developed to compute the maximum flow so the task will be to investigate their appropriate parallel implementations, find bottlenecks, optimize and benchmark on a set of graphs with different characteristics and explore a few real applications. If things go well, we might consider integration into nvGraph as the final step, although the work will be mostly focused on new algorithms development and analysis.

## Build instructions

Update Makefile as necessary then run `make` to build everything, or use `make <version>` to build a specific version only: `cpu`, `gpu_naive`, `gpu_gunrock`.

Note that if you're trying to build `gpu_gunrock` you will need to clone recursively to fetch `gunrock` submodule. Then also build Gunrock with all dependencies using cmake.

## Running examples

Download data sets using a shell script in `data/get_data.sh`.

```
Usage: ./maxflow <input matrix file> <source id> <target id> [<random seed>]
       if random seed is not specified the weights are set as 1/degree for each vertex
```

There is also a test script `test.py` which runs various pre-defined examples and validates results. You will need to create a soft link `maxflow_gpu` to `maxflow_gpu_naive` or `maxflow_gpu_gunrock` to use the script.

## Development plan

- [ ] Study maximum flow algorithms and relevant papers (2 weeks)
- [ ] Implementation of selected maximum flow algorithms on GPU (2-4 weeks)
- [ ] Study applications: bipartite matching, community detection (2 weeks)
- [ ] Integration into nvGraph (2 weeks)

