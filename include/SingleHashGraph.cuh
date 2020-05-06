/*
 * Copyright (c) 2017-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <set>

#include <cuda_profiler_api.h> //--profile-from-start off

// #include <moderngpu/memory.hxx>
// #include <moderngpu/kernel_sortedsearch.hxx>
// #include <moderngpu/kernel_mergesort.hxx>
// #include <moderngpu/kernel_merge.hxx>
// #include <moderngpu/kernel_scan.hxx>
// #include <moderngpu/kernel_segsort.hxx>
// #include <moderngpu/kernel_load_balance.hxx>

#include <cuda_runtime_api.h>

// #include "rmm/rmm.h"
// #include "rmm.h"

#include <cub/cub.cuh>

using hkey_t = int64_t;
using index_t = int32_t;
using HashKey = uint32_t;

// using namespace mgpu;

struct keyval
{
  hkey_t  key;
  // index_t ind; 
};

// Overload Modern GPU memory allocation and free to use RMM
// class rmm_mgpu_context_t : public mgpu::standard_context_t
// {
// public:
//   rmm_mgpu_context_t(bool print_prop = true, cudaStream_t stream_ = 0) :
//     mgpu::standard_context_t(print_prop, stream_) {}
//   ~rmm_mgpu_context_t() {}
// 
//   virtual void* alloc(size_t size, memory_space_t space) {
//     void *p = nullptr;
//     if(size) {
//       if (memory_space_device == space) {
//         if (RMM_SUCCESS != RMM_ALLOC(&p, size, stream()))
//           throw cuda_exception_t(cudaPeekAtLastError());
//       }
//       else {
//         cudaError_t result = cudaMallocHost(&p, size);
//         if (cudaSuccess != result) throw cuda_exception_t(result);
//       }
//     }
//     return p;
//   }
// 
//   virtual void free(void* p, memory_space_t space) {
//     if (p) {
//       if (memory_space_device == space) {
//         if (RMM_SUCCESS != RMM_FREE(p, stream()))
//           throw cuda_exception_t(cudaPeekAtLastError());
//       }
//       else {
//         cudaError_t result = cudaFreeHost(&p);
//         if (cudaSuccess != result) throw cuda_exception_t(result);
//       }
//     }
//   }
// };

class SingleHashGraph {
public:
    // SingleHashGraph(int64_t countSize, int64_t maxkey, context_t &context, int64_t tableSize);
    SingleHashGraph(int64_t countSize, int64_t maxkey, int64_t tableSize, int64_t lrbBins);
    ~SingleHashGraph();

    // void build(int64_t countSize, context_t &context, int64_t tableSize);
    void build(int64_t countSize, int64_t tableSize);

private:

    // mem_t<hkey_t>  d_vals;
    // mem_t<HashKey>  d_hash;
    // mem_t<index_t>  d_counter;
    // mem_t<index_t>  d_offset;
    // mem_t<keyval>   d_edges;

    hkey_t*  d_vals;
    HashKey*  d_hash;
    index_t*  d_counter;
    index_t*  d_offset;
    keyval*   d_edges;

    int64_t lrbBins;
    index_t *d_lrbCounter;
    index_t *d_lrbCounterPrefix;
    keyval *d_lrbArray;
};
