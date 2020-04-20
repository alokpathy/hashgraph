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

#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>

#include <omp.h>

// #include "rmm.h"
// #include "rmm.hpp"
// #include "rmm/rmm_api.h"
// #include "rmm/detail/memory_manager.hpp"


// #include <moderngpu/kernel_load_balance.hxx>

#include <cuda_runtime_api.h>

// #include "rmm/rmm.h"
// #include "rmm.h"

#include <cub/cub.cuh>

// using namespace mgpu;

#define CHECK_ERROR(str) \
	{cudaDeviceSynchronize(); cudaError_t err; err = cudaGetLastError(); if(err!=0) {printf("ERROR %s:  %d %s\n", str, err, cudaGetErrorString(err)); fflush(stdout); exit(0);}}

// #define CUDA_PROFILE
// #define HOST_PROFILE

//#define INDEX_TRACK
//#define MANAGED_MEM
// #define B32

#ifdef B32
using hkey_t = uint32_t;
using index_t = int64_t;
using HashKey = uint32_t;
#else
using hkey_t = int64_t;
using index_t = int64_t;
using HashKey = int64_t;
#endif


struct keyval_key
{
  hkey_t  key;
};

struct keyval_ind
{
  hkey_t  key;
  // uint64_t gpuId;
  index_t ind; 
};

struct keypair
{
  // index_t left; 
  index_t right; 
};

struct inputData
{
  hkey_t *d_keys;
  HashKey *d_hash;
  index_t len;
};

#ifdef INDEX_TRACK
typedef keyval_ind keyval;
#else
typedef keyval_key keyval;
#endif

inline bool operator==(const keyval &kv1, const keyval &kv2) {
  // return kv1.key == kv2.key && kv1.ind == kv2.ind;
  return kv1.key == kv2.key;
}

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

class MultiHashGraph {
public:
    MultiHashGraph(inputData *h_dVals, index_t countSize, index_t maxkey, 
                      // context_t &context, index_t tableSize,
                      index_t tableSize,
                      index_t binCount, index_t lrbBins, index_t gpuCount); ~MultiHashGraph();

    void build(bool buildSplits, index_t tid);
    void buildSingle();

    static void intersect(MultiHashGraph &mgA, MultiHashGraph &mgB, index_t *h_Common,
                              keypair **h_dOutput, index_t tid);

    void destroyMulti();

    char **h_dFinalKeys;
    index_t *h_hashOff;
    index_t *h_counterOff;
    index_t *h_offsetOff;
    index_t *h_edgesOff;
    index_t *h_lrbOff;

    // Structures for allocating bins to GPUs.
    // Public so that another HG can use the same splits.
    index_t *h_binSplits;
    index_t **h_dBinSplits;

    // int64_t *d_Common;
    // int64_t *d_GlobalCounter;
    index_t **h_dCommon;
    index_t **h_dGlobalCounter;

    index_t countSize;
    // int64_t tableSize;
    index_t tableSize;
    index_t gpuCount;

    // Public for correctness check
    hkey_t *h_vals;

    char **h_dCountCommon;
    char *uvmPtr;
    char *uvmPtrIntersect;
    index_t *prefixArray;
    index_t *prefixArrayIntersect;
    index_t totalSize;
    index_t totalSizeIntersect;

    size_t **h_dExSumTemp;
    size_t exSumTempBytes;


private:

    // mem_t<hkey_t>  d_vals;
    // hkey_t  **h_dVals;
    inputData *h_dVals;
    hkey_t  *d_vals;
    // mem_t<HashKey>  d_hash;
    // mem_t<index_t>  d_counter;
    // mem_t<index_t>  d_offset;
    // mem_t<keyval>   d_edges;


    // Structures for initial binning
    index_t *h_binSizes;
    // index_t *d_binSizes;
    index_t **h_dBinSizes;
    index_t **h_hBinSizes;

    index_t *h_psBinSizes;
    index_t *d_psBinSizes;

    // Allocating physical bins for binning keys
    hkey_t **h_keyBins;
    index_t *d_binCounter;
    index_t *h_binCounter;

    // Structures for keeping keys on each GPU.
    hkey_t **h_dKeys;

    // Structures for storing and binning hashes on each GPU.
    HashKey **h_dHashes;
    index_t **h_dBinCounter;
    index_t **h_hBinCounter;

    // Structures for prefix summing hash bins across GPUs (on host).
    index_t *h_hashBinSize;
    index_t *h_psHashBinSize;

    // Structure for allocating hash bins for each GPU.
    index_t *h_hashSplits;

    // Structure for sending hash bin allocations to each GPU.
    // Keeps track of which bins go to which GPU.
    index_t **h_dHashSplits;

    // Structures for counting the key/hash buffer sizes on each GPU.
    index_t **h_dBufferCounter;

    index_t **h_bufferCounter;

    // Used for initial key binning
    // hkey_t **h_dKeyBinBuff; 
    keyval **h_dKeyBinBuff; 
    HashKey **h_dHashBinBuff; 
    index_t **h_dKeyBinOff;
    index_t **h_hKeyBinOff;

    // Actual key/hash buffers per GPU on each GPU for hash values.
    hkey_t **h_dKeyBuff;

    HashKey **h_dHashBuff;
    index_t **h_dOffset;
    index_t **h_hOffset;

    // size_t *h_keyPitches;
    // size_t *h_hashPitches;

    // Final, consolidated list of key/hashes on each GPU.
    // hkey_t **h_dFinalKeys;
    HashKey **h_dFinalHash;
    index_t **h_dFinalCounter;
    index_t **h_hFinalCounter;

    index_t **h_dFinalOffset;
    index_t **h_hFinalOffset;

    // HashGraph construction structures.
    index_t **h_dOffsets;
    index_t **h_dCounter;
    // keyval **h_dEdges;
    hkey_t **h_dEdges;

    // LRB constructs
    index_t **h_dLrbCounter;
    index_t **h_dLrbCountersPrefix;

    index_t maxkey;
    index_t binCount;
    index_t lrbBins;

    bool multiDestroyed = false;
};
