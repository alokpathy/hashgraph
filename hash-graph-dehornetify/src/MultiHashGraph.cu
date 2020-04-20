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
#include "MultiHashGraph.cuh"
#include "MultiHashGraphDeviceOperators.cuh"
#include "MultiHashGraphHostOperators.cuh"

#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <chrono>

#include <algorithm>

#include <cuda_profiler_api.h> //--profile-from-start off

// #include <moderngpu/memory.hxx>
// #include <moderngpu/kernel_sortedsearch.hxx>
// #include <moderngpu/kernel_mergesort.hxx>
// #include <moderngpu/kernel_merge.hxx>
// #include <moderngpu/kernel_scan.hxx>
// #include <moderngpu/kernel_segsort.hxx>

#include <cuda_runtime_api.h>
#include <cub/cub.cuh>

#include <omp.h>

// #include "rmm/rmm.h"
// #include "rmm.h"
// using namespace mgpu;
using namespace std::chrono;


// Uncomment once we remove "using namespace hornets_nest"
// const int BLOCK_SIZE_OP2 = 256;

// #define ERROR_CHECK
// #define PRINT_KEYS
#define LRB_BUILD

#ifdef HOST_PROFILE
uint64_t tidFocused = 2;
#endif

// #define DEBUG

MultiHashGraph::MultiHashGraph(inputData *h_dVals, index_t countSize, index_t maxkey, 
                                    // context_t &context, index_t tableSize, 
                                    index_t tableSize, 
                                    index_t binCount, index_t lrbBins, 
                                    index_t gpuCount) {

  
  index_t binRange = std::ceil(maxkey / ((float)binCount));
  BLOCK_COUNT = std::ceil(countSize / ((float) BLOCK_SIZE_OP2));
  BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

  std::cout << "bin_count: " << binCount << std::endl;
  std::cout << "bin_range: " << binRange << std::endl;
  std::cout << "table_size: " << tableSize << std::endl;

  std::cout << "BLOCK_COUNT: " << BLOCK_COUNT << " BLOCKS_SIZE: " << BLOCK_SIZE_OP2 << "\n";

  h_vals = new hkey_t[countSize]();

  // Input is arrays of key on different devices.
  index_t avgKeyCount = std::ceil(countSize / ((double) gpuCount));
  index_t h_valIdx = 0;

  // h_dKeyBinBuff = new hkey_t*[gpuCount]();
  h_dKeyBinBuff = new keyval*[gpuCount]();

  index_t seed = 0;
  for (index_t i = 0; i < gpuCount; i++) {
    cudaSetDevice(i);

    index_t lo = avgKeyCount * i;
    index_t hi = avgKeyCount * (i + 1);
    hi = std::min(hi, countSize);

    index_t keyCount = hi - lo;

    cudaMemcpy(h_vals + h_valIdx, h_dVals[i].d_keys, keyCount * sizeof(hkey_t),
                  cudaMemcpyDeviceToHost);

    h_valIdx += keyCount;

    cudaMalloc(&h_dKeyBinBuff[i], keyCount * sizeof(keyval));
  }

#ifdef PRINT_KEYS
  std::sort(h_vals, h_vals + countSize);
  std::cout << "keys: " << std::endl;
  for (uint32_t i = 0; i < countSize; i++) {
    std::cout << h_vals[i] << " ";
  }
  std::cout << std::endl;
#endif

  // Structures for initial binning
  h_binSizes = new index_t[binCount](); // Consolidated bin sizes across devices
  h_hBinSizes = new index_t*[gpuCount](); // Bin sizes per device
  h_dBinSizes = new index_t*[gpuCount]();
  for (index_t i = 0; i < gpuCount; i++) {
    cudaSetDevice(i);
    cudaMalloc(&h_dBinSizes[i], binCount * sizeof(index_t));
    h_hBinSizes[i] = new index_t[binCount]();
  }

  h_psBinSizes = new index_t[binCount + 1]();

  // Structures for allocating bins to GPUs (i.e. hash ranges).
  h_binSplits = new index_t[gpuCount + 1]();
  h_dBinSplits = new index_t*[gpuCount]();
  for (index_t i = 0; i < gpuCount; i++) {
    cudaSetDevice(i);
    cudaMalloc(&h_dBinSplits[i], (gpuCount + 1) * sizeof(index_t));
  }
  cudaSetDevice(0);

  // Structures for counting the key/hash buffer sizes on each GPU.
  h_bufferCounter = new index_t*[gpuCount]();
  for (index_t i = 0; i < gpuCount; i++) {
    h_bufferCounter[i] = new index_t[gpuCount]();
  }

  h_dBufferCounter = new index_t*[gpuCount]();
  for (index_t i = 0; i < gpuCount; i++) {
    cudaSetDevice(i);
    cudaMalloc(&h_dBufferCounter[i], gpuCount * sizeof(index_t));
    cudaMemset(h_dBufferCounter[i], 0, gpuCount * sizeof(index_t));
  }
  cudaSetDevice(0);

  h_hKeyBinOff = new index_t*[gpuCount]();
  h_dKeyBinOff = new index_t*[gpuCount]();
  for (index_t i = 0; i < gpuCount; i++) {
    cudaSetDevice(i);
    cudaMalloc(&h_dKeyBinOff[i], (gpuCount + 1) * sizeof(index_t));
    h_hKeyBinOff[i] = new index_t[gpuCount + 1]();
  }

  // h_dFinalKeys = new hkey_t*[gpuCount]();
  h_dFinalKeys = new char*[gpuCount]();

  h_hFinalCounter = new index_t*[gpuCount]();
  h_dFinalOffset = new index_t*[gpuCount]();
  h_hFinalOffset = new index_t*[gpuCount]();
  for (index_t i = 0; i < gpuCount; i++) {
    cudaSetDevice(i);
    h_hFinalCounter[i] = new index_t[gpuCount]();
    h_hFinalOffset[i] = new index_t[gpuCount]();
    cudaMalloc(&h_dFinalOffset[i], (gpuCount + 1) * sizeof(index_t));
  }

  h_dOffsets = new index_t*[gpuCount]();
  h_dCounter = new index_t*[gpuCount]();
  h_dEdges = new hkey_t*[gpuCount]();

  h_dExSumTemp = new size_t*[gpuCount]();
  // exSumTempBytes = 1279;
  // exSumTempBytes = 2000;
  // exSumTempBytes = 3000000;
  exSumTempBytes = std::max(2048L, (long)(tableSize / 10));
  // exSumTempBytes = tableSize / 10;
  for (index_t i = 0; i < gpuCount; i++) {
    cudaSetDevice(i);
    cudaMalloc(&h_dExSumTemp[i], exSumTempBytes);
  }

  this->h_dVals = h_dVals;
  this->countSize = countSize;
  this->maxkey = maxkey;
  this->tableSize = tableSize;
  this->binCount = binCount;
  this->lrbBins = lrbBins;
  this->gpuCount = gpuCount;

  h_hashOff = new index_t[gpuCount]();
  h_counterOff = new index_t[gpuCount]();
  h_offsetOff = new index_t[gpuCount]();
  h_edgesOff = new index_t[gpuCount]();
  h_lrbOff = new index_t[gpuCount]();

  // cudaMalloc(&d_Common, 1 * sizeof(index_t));
  // cudaMemset(d_Common, 0, 1 * sizeof(index_t));

  // cudaMalloc(&d_GlobalCounter, 1 * sizeof(index_t));
  // cudaMemset(d_GlobalCounter, 0, 1 * sizeof(index_t));
  h_dCommon = new index_t*[gpuCount]();
  h_dGlobalCounter = new index_t*[gpuCount]();
  for (index_t i = 0; i < gpuCount; i++) {
    cudaSetDevice(i);
    cudaMalloc(&h_dCommon[i], 1 * sizeof(index_t));
    cudaMemset(h_dCommon[i], 0, 1 * sizeof(index_t));

    cudaMalloc(&h_dGlobalCounter[i], 1 * sizeof(index_t));
    cudaMemset(h_dGlobalCounter[i], 0, 1 * sizeof(index_t));
  }

#ifdef LRB_BUILD
  h_dLrbCounter = new index_t*[gpuCount]();
  h_dLrbCountersPrefix = new index_t*[gpuCount]();

  for (index_t i = 0; i < gpuCount; i++) {
    cudaSetDevice(i);
    cudaMalloc(&h_dLrbCounter[i], (lrbBins + 2) * sizeof(index_t));
    cudaMemset(h_dLrbCounter[i], 0, (lrbBins + 2) * sizeof(index_t));

    cudaMalloc(&h_dLrbCountersPrefix[i], (lrbBins + 2) * sizeof(index_t));
    cudaMemset(h_dLrbCountersPrefix[i], 0,(lrbBins + 2) * sizeof(index_t));
  }
#endif

#ifdef MANAGED_MEM
  index_t size = countSize * sizeof(keyval) + 
                     countSize * sizeof(HashKey) +
                     (2 * countSize * sizeof(keyval)) +
                     (2 * (tableSize + gpuCount) * sizeof(index_t));
  std::cout << "managed alloc size: " << size << std::endl;
  this->totalSize = size;
  uvmPtr = nullptr;

  cudaMallocManaged(&uvmPtr, size);
  index_t equalChunk = size / gpuCount;
  for (index_t i = 0; i < gpuCount; i++) {
    cudaSetDevice(i);
    cudaMemPrefetchAsync(uvmPtr + equalChunk * i, equalChunk, i);
  }
  prefixArray = new index_t[gpuCount + 1]();
  
  h_dCountCommon = new char*[gpuCount]();
#endif

  CHECK_ERROR("constructor");

}

MultiHashGraph::~MultiHashGraph() {
#if 0
  if (!multiDestroyed) {
    destroyMulti();
  }
  delete[] h_vals;
#endif
}

void MultiHashGraph::destroyMulti() {
  // cudaFree(d_binSizes);
#if 0
  cudaFree(d_psBinSizes);

  for (uindex_t i = 0; i < gpuCount; i++) {
    cudaSetDevice(i);
    cudaFree(h_dKeys[i]);
    cudaFree(h_dBinCounter[i]);
    cudaFree(h_dHashes[i]);
    cudaFree(h_dBufferCounter[i]);
    cudaFree(h_dHashSplits[i]);
    delete[] h_hBinCounter[i];
  }
  cudaSetDevice(0);
  delete[] h_dKeys;
  delete[] h_dHashSplits;
  delete[] h_dBinCounter;
  delete[] h_dHashes;
  delete[] h_hBinCounter;
  delete[] h_dBufferCounter;

  for (uindex_t i = 0; i < binCount; i++) {
    delete[] h_keyBins[i];
  }

  delete[] h_keyBins;
  delete[] h_binSizes;
  cudaFree(d_binCounter);
  delete[] h_binCounter;
  delete[] h_binSplits;

  multiDestroyed = true;
#endif
}

bool compareByKey(const keyval &kv1, const keyval &kv2) {
  return kv1.key < kv2.key;
}

// void lrbBuildMultiTable(hkey_t *d_vals, HashKey *d_hash, index_t *d_counter, 
void lrbBuildMultiTable(keyval *d_vals, HashKey *d_hash, index_t *d_counter, 
// void lrbBuildMultiTable(keyval *d_vals, keyval *d_hash, index_t *d_counter, 
                          index_t *d_offSet, keyval *d_edges, index_t *d_splits, 
                          index_t valCount, index_t tableSize, index_t ogTableSize, 
                          keyval *d_lrbArray, index_t *d_lrbCounters, 
                          index_t *d_lrbCountersPrefix, size_t *d_exSumTemp,
                          size_t exSumTempBytes, index_t lrbBins, index_t lrbBinSize, 
                          index_t devNum) {

  cudaMemset(d_counter, 0, (tableSize + 1) * sizeof(index_t));
  void*  _d_temp_storage     { nullptr };
  size_t _temp_storage_bytes { 0 };



  hashValuesD<<<BLOCK_COUNT, BLOCK_SIZE_OP2>>>(valCount, d_vals, d_hash, 
                                                  (HashKey) ogTableSize, devNum);
  decrHash<<<BLOCK_COUNT, BLOCK_SIZE_OP2>>>(d_hash, valCount, d_splits, 
                                                devNum);

  lrbCountHashD<<<BLOCK_COUNT, BLOCK_SIZE_OP2>>>(valCount, d_hash, d_lrbCounters, 
                                                    lrbBinSize);
  _d_temp_storage = nullptr; _temp_storage_bytes = 0;

  cub::DeviceScan::ExclusiveSum(NULL, _temp_storage_bytes, d_lrbCounters, 
                                    d_lrbCountersPrefix, lrbBins + 1);

  if (_temp_storage_bytes > exSumTempBytes) {
    std::cerr << "ERROR: NOT ENOUGH TEMP SPACE ALLOCATED" << std::endl;
  }


  cub::DeviceScan::ExclusiveSum(d_exSumTemp, _temp_storage_bytes, d_lrbCounters, 
                                    d_lrbCountersPrefix, lrbBins + 1);

  cudaMemcpy(d_lrbCountersPrefix + lrbBins, &lrbBins, sizeof(index_t), 
                cudaMemcpyHostToDevice);

  cudaMemset(d_lrbCounters, 0, (lrbBins + 1) * sizeof(index_t));

  lrbRehashD<<<BLOCK_COUNT, BLOCK_SIZE_OP2>>>(valCount, d_vals, d_hash, 
                                                    d_lrbCounters, d_lrbArray, 
                                                    d_lrbCountersPrefix, lrbBinSize,
                                                    devNum);

  lrbCountHashGlobalD<<<BLOCK_COUNT, BLOCK_SIZE_OP2>>>(valCount, d_counter, 
                                                            d_lrbArray, d_splits, 
                                                            ogTableSize, devNum);

  _d_temp_storage = nullptr; _temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(_d_temp_storage, _temp_storage_bytes,d_counter, 
                                    d_offSet, tableSize);
  // cudaMalloc(&_d_temp_storage, _temp_storage_bytes);
  // // RMM_ALLOC(&_d_temp_storage, _temp_storage_bytes, 0);
  // cub::DeviceScan::ExclusiveSum(_d_temp_storage, _temp_storage_bytes,d_counter, 
  //                                    d_offSet, tableSize);

  if (_temp_storage_bytes > exSumTempBytes) {
    std::cerr << "ERROR: NOT ENOUGH TEMP SPACE ALLOCATED" << std::endl;
    std::cerr << _temp_storage_bytes << " " << exSumTempBytes << std::endl;
  }
  cub::DeviceScan::ExclusiveSum(d_exSumTemp, _temp_storage_bytes,d_counter, 
                                      d_offSet, tableSize);
  cudaMemcpy(d_offSet + tableSize, &valCount, sizeof(index_t), cudaMemcpyHostToDevice);
  // cudaFree(_d_temp_storage);
  // RMM_FREE(_d_temp_storage, 0);

  cudaMemset(d_counter, 0, tableSize * sizeof(index_t));

  lrbCopyToGraphD<<<BLOCK_COUNT, BLOCK_SIZE_OP2>>>(valCount, d_counter, d_offSet, 
                                                      d_edges, d_lrbArray, d_splits, 
                                                      ogTableSize, devNum);
}

#ifndef LRB_BUILD
void buildMultiTable(hkey_t *d_vals, HashKey *d_hash, index_t *d_counter, 
	              index_t *d_offSet, keyval *d_edges, index_t *d_splits, index_t valCount, 
                      index_t tableSize, index_t devNum) {

  void*  _d_temp_storage     { nullptr };
  size_t _temp_storage_bytes { 0 };

  decrHash<<<BLOCK_COUNT, BLOCK_SIZE_OP2>>>(d_vals, d_hash, valCount, d_splits, devNum);

  countHashD<<<BLOCK_COUNT, BLOCK_SIZE_OP2>>>(valCount, d_hash, d_counter);

  cub::DeviceScan::ExclusiveSum(_d_temp_storage, _temp_storage_bytes, d_counter, 
                                    d_offSet, tableSize);

  cudaMalloc(&_d_temp_storage, _temp_storage_bytes);
  cub::DeviceScan::ExclusiveSum(_d_temp_storage, _temp_storage_bytes,d_counter, 
                                    d_offSet, tableSize);
  // d_counter = fill(0, (size_t)tableSize, context);
  cudaMemset(d_counter, 0, tableSize * sizeof(index_t));
  cudaMemcpy(d_offSet + tableSize, &valCount, sizeof(index_t), cudaMemcpyHostToDevice);

  if (_d_temp_storage > 0) {
    cudaFree(_d_temp_storage);
  }

  copyToGraphD<<<BLOCK_COUNT, BLOCK_SIZE_OP2>>>(valCount, d_vals, d_hash, d_counter, d_offSet, 
                                                    d_edges, tableSize);
}
#endif

void MultiHashGraph::build(bool findSplits, index_t tid) {
  // index_t binRange = std::ceil(maxkey / ((float)binCount));
  index_t binRange = std::ceil(tableSize / ((float)binCount));
  
  cudaSetDevice(0);
  cudaEvent_t start, stop;

  float buildTime = 0.0f; // milliseoncds
  high_resolution_clock::time_point t1;
  high_resolution_clock::time_point t2;

  // Hash all keys on each device.
  cudaSetDevice(tid);

  basicHashD<<<BLOCK_COUNT, BLOCK_SIZE_OP2>>>(h_dVals[tid].len, h_dVals[tid].d_keys,
                                                  // h_dVals[tid].d_keys + h_dVals[tid].len, tableSize);
                                                  h_dVals[tid].d_hash, tableSize);
#ifdef ERROR_CHECK
  cudaDeviceSynchronize();
  CHECK_ERROR("hashValues");
#endif

  if (findSplits) {
#ifdef HOST_PROFILE
    t1 = high_resolution_clock::now();
#endif
    // Count the number of keys in each key bin and determine the hash range per device.
    countBinSizes(h_dVals, h_hBinSizes, h_dBinSizes, h_binSizes, h_psBinSizes, h_binSplits,
                      h_dBinSplits, countSize, tableSize, binRange, binCount, gpuCount, tid);
#ifdef HOST_PROFILE
    if (tid == tidFocused) {
      cudaDeviceSynchronize();
      t2 = high_resolution_clock::now();
      buildTime = duration_cast<milliseconds>( t2 - t1 ).count();
      std::cout << "countBinSizes time: " << (buildTime / 1000.0) << std::endl;
    }
#endif

#ifdef ERROR_CHECK
    cudaDeviceSynchronize();
    CHECK_ERROR("countBinSizes");
#endif
  }

#ifdef HOST_PROFILE
  if (tid == tidFocused) {
    t1 = high_resolution_clock::now();
  }
#endif
  // Count the number of keys that each GPU needs to ship to each other GPU based
  // on ranges.
  countKeyBuffSizes(h_dVals, h_dBinSplits, h_bufferCounter, h_dBufferCounter, gpuCount, tid);
#ifdef HOST_PROFILE
  if (tid == tidFocused) {
    cudaDeviceSynchronize();
    t2 = high_resolution_clock::now();
    buildTime = duration_cast<milliseconds>( t2 - t1 ).count();
    std::cout << "countKeyBuff time: " << (buildTime / 1000.0) << std::endl;
  }
#endif


#ifdef ERROR_CHECK
  cudaDeviceSynchronize();
  CHECK_ERROR("countKeyBuffSizes");
#endif

#ifdef HOST_PROFILE
  if (tid == tidFocused) {
    t1 = high_resolution_clock::now();
  }
#endif
  // On each GPU, buffer all the keys going to each other GPU.
  populateKeyBuffs(h_dVals, h_dKeyBinBuff, h_dKeyBinOff, h_hKeyBinOff, 
                      h_dBufferCounter, h_bufferCounter, h_dBinSplits, h_dExSumTemp,
                      exSumTempBytes, gpuCount, tid);
#ifdef HOST_PROFILE
  if (tid == tidFocused) {
    cudaDeviceSynchronize();
    t2 = high_resolution_clock::now();
    buildTime = duration_cast<milliseconds>( t2 - t1 ).count();
    std::cout << "populateKeyBuffs time: " << (buildTime / 1000.0) << std::endl;
  }
#endif
#ifdef ERROR_CHECK
  cudaDeviceSynchronize();
  CHECK_ERROR("populateKeyBuffs");
#endif


#ifdef INDEX_TRACK
  cudaFree(h_dVals[tid].d_keys);
  cudaFree(h_dVals[tid].d_hash);
  // RMM_FREE(h_dVals[tid].d_keys, 0);
  // RMM_FREE(h_dVals[tid].d_hash, 0);
#endif

#ifdef HOST_PROFILE
  if (tid == tidFocused) {
    t1 = high_resolution_clock::now();
  }
#endif

  // On each GPU, count the number of keys that will get shipped to it.
#ifdef MANAGED_MEM
  countFinalKeys(h_bufferCounter, h_dFinalKeys, h_hFinalCounter,
                    h_hFinalOffset, h_dFinalOffset, h_binSplits, gpuCount, tid,
                    uvmPtr, prefixArray, totalSize);
#else
  countFinalKeys(h_bufferCounter, h_dFinalKeys, h_hFinalCounter,
                    h_hFinalOffset, h_dFinalOffset, h_binSplits, gpuCount, tid);
#endif

#ifdef HOST_PROFILE
  if (tid == tidFocused) {
    cudaDeviceSynchronize();
    t2 = high_resolution_clock::now();
    buildTime = duration_cast<milliseconds>( t2 - t1 ).count();
    std::cout << "countFinalKeys time: " << (buildTime / 1000.0) << std::endl;
  }
#endif
#ifdef ERROR_CHECK
  cudaDeviceSynchronize();
  CHECK_ERROR("countFinalKeys");
#endif

#ifdef HOST_PROFILE
  if (tid == tidFocused) {
    t1 = high_resolution_clock::now();
  }
#endif

  #pragma omp barrier

  // Ship all the keys to their respective GPUs.
  allToAll(h_dVals, h_dFinalKeys, h_hFinalOffset, h_dKeyBinBuff, 
              h_hKeyBinOff, h_hFinalCounter, gpuCount, tid);

#ifdef HOST_PROFILE
  if (tid == tidFocused) {
    cudaDeviceSynchronize();
    t2 = high_resolution_clock::now();
    buildTime = duration_cast<milliseconds>( t2 - t1 ).count();
    std::cout << "allToAll time: " << (buildTime / 1000.0) << std::endl;
  }
#endif

  // #pragma omp barrier
  // cudaDeviceSynchronize();

#ifdef ERROR_CHECK
  cudaDeviceSynchronize();
  CHECK_ERROR("allToAll");
#endif

#ifdef HOST_PROFILE
  if (tid == tidFocused) {
    t1 = high_resolution_clock::now();
  }
#endif

  // Build hashgraph on each GPU.
  index_t hashRange = h_binSplits[tid + 1] - h_binSplits[tid];
  // cudaMalloc(&h_dOffsets[tid], 2 * (hashRange + 1) * sizeof(index_t));

  index_t keyCount = h_hFinalOffset[tid][gpuCount];

#ifdef LRB_BUILD
  index_t lrbBinSize = std::ceil(hashRange / (float)(lrbBins));
  
  if (lrbBinSize == 0) {
    // std::cout << "ERROR: TOO MANY LRB BINS" << std::endl;
    printf("ERROR tid: %ld hashRange: %ld\n", tid, hashRange);
    exit(0);
  }

  h_hashOff[tid] = keyCount * sizeof(keyval);
  h_counterOff[tid] = (keyCount * sizeof(keyval)) +
                        (keyCount * sizeof(HashKey)) +
                        (2 * keyCount * sizeof(keyval)) + 
                        ((hashRange + 1) * sizeof(index_t));
  h_offsetOff[tid] = (keyCount * sizeof(keyval)) + 
                        (keyCount * sizeof(HashKey)) +
                        (2 * keyCount * sizeof(keyval));
  h_edgesOff[tid] =  (keyCount * sizeof(keyval)) + 
                        (keyCount * sizeof(HashKey));
  h_lrbOff[tid] =    (keyCount * sizeof(keyval)) +
                        (keyCount * sizeof(HashKey)) +
                        (keyCount * sizeof(keyval));

  lrbBuildMultiTable((keyval *) h_dFinalKeys[tid], 
                          (HashKey *) (h_dFinalKeys[tid] + h_hashOff[tid]), // d_hash
                          (index_t *)(h_dFinalKeys[tid] + h_counterOff[tid]), // d_counter
                          (index_t *)(h_dFinalKeys[tid] + h_offsetOff[tid]), // d_offSet
                          (keyval *)(h_dFinalKeys[tid] + h_edgesOff[tid]), // d_edges
                          h_dBinSplits[tid], keyCount, hashRange, tableSize, 
                          (keyval *)(h_dFinalKeys[tid] + h_lrbOff[tid]), // d_lrbArray
                          h_dLrbCounter[tid], 
                          h_dLrbCountersPrefix[tid], h_dExSumTemp[tid], exSumTempBytes, 
                          lrbBins, lrbBinSize, tid);
#ifdef HOST_PROFILE
  if (tid == tidFocused) {
    cudaDeviceSynchronize();
    t2 = high_resolution_clock::now();
    buildTime = duration_cast<milliseconds>( t2 - t1 ).count();
    std::cout << "building time: " << (buildTime / 1000.0) << std::endl;
  }
#endif

#else

  buildMultiTable(h_dFinalKeys[tid], h_dFinalHash[tid], h_dCounter[tid], h_dOffsets[tid], 
                     h_dEdges[tid], h_dBinSplits[tid], keyCount, hashRange, tid);

#endif
                    
  cudaSetDevice(0);
#ifdef ERROR_CHECK
  cudaDeviceSynchronize();
  CHECK_ERROR("build error");
#endif
}

void MultiHashGraph::intersect(MultiHashGraph &mhgA, MultiHashGraph &mhgB, index_t *h_Common,
                                    keypair **h_dOutput, index_t tid) {

  index_t gpuCount = mhgA.gpuCount;

  cudaSetDevice(tid);

  index_t *d_offsetA  = (index_t *)(mhgA.h_dFinalKeys[tid] + mhgA.h_offsetOff[tid]);
  keyval  *d_edgesA   = (keyval  *)(mhgA.h_dFinalKeys[tid] + mhgA.h_edgesOff[tid]);
  index_t *d_counterA = (index_t *)(mhgA.h_dFinalKeys[tid] + mhgA.h_counterOff[tid]);

  index_t *d_offsetB  = (index_t *)(mhgB.h_dFinalKeys[tid] + mhgB.h_offsetOff[tid]);
  keyval  *d_edgesB   = (keyval  *)(mhgB.h_dFinalKeys[tid] + mhgB.h_edgesOff[tid]);
  index_t *d_counterB = (index_t *)(mhgB.h_dFinalKeys[tid] + mhgB.h_counterOff[tid]);

  index_t *d_Common = mhgA.h_dCommon[tid];
  index_t *d_GlobalCounter = mhgA.h_dGlobalCounter[tid];

  size_t *d_exSumTemp = mhgA.h_dExSumTemp[tid];
  size_t exSumTempBytes = mhgA.exSumTempBytes;

  index_t tableSize = mhgA.h_binSplits[tid + 1] - mhgA.h_binSplits[tid];
  if (tableSize != mhgB.h_binSplits[tid + 1] - mhgB.h_binSplits[tid]) {
    std::cerr << "ERROR: TABLE SIZE NOT SAME BETWEN TWO HG'S" << std::endl;
    exit(0);
  }

  // TODO: might be able to reuse stuff from cudaMalloc() from building.
  index_t *d_countCommon = nullptr;
  index_t *d_outputPositions = nullptr;

  // cudaMalloc(&d_countCommon, (size_t)(tableSize + 1) * sizeof(index_t));
  // cudaMalloc(&d_outputPositions, (size_t)(tableSize + 1) * sizeof(index_t));
#ifdef MANAGED_MEM
  d_countCommon = (index_t *) mhgA.h_dCountCommon[tid];
  cudaMemPrefetchAsync(d_countCommon, 
                            mhgA.prefixArrayIntersect[tid + 1] - mhgA.prefixArrayIntersect[tid], 
                            tid);
#else
  cudaMalloc(&d_countCommon, (size_t)(2 * ((tableSize + 1) * sizeof(index_t))));
#endif
  d_outputPositions = d_countCommon + tableSize + 1;

  // RMM_ALLOC(&d_countCommon, (size_t)(tableSize + 1) * sizeof(index_t), 0);
  // RMM_ALLOC(&d_outputPositions, (size_t)(tableSize + 1) * sizeof(index_t), 0);

  // cudaMemsetAsync(d_countCommon, 0, (size_t)(tableSize + 1) * sizeof(index_t));
  // cudaMemsetAsync(d_outputPositions, 0, (size_t)(tableSize + 1) * sizeof(index_t));
  cudaMemsetAsync(d_countCommon, 0, (size_t)(2 * ((tableSize + 1) * sizeof(index_t))));

  simpleIntersect<<<BLOCK_COUNT, BLOCK_SIZE_OP2>>>(tableSize, d_offsetA, d_edgesA, d_offsetB,
                                                        d_edgesB, d_countCommon, NULL, true);
  // forAll (vertices,simpleIntersect<true>{d_offSetA.data(),d_edgesA.data(), d_offSetB.data(),
  //                                              d_edgesB.data(),d_countCommon.data(),NULL});


  // index_t h_Common;
  void *_d_temp_storage=nullptr; size_t _temp_storage_bytes=0;

  _d_temp_storage=nullptr; _temp_storage_bytes=0;
  cub::DeviceReduce::Sum(_d_temp_storage, _temp_storage_bytes, d_countCommon, 
                              d_Common, tableSize);

  if (_temp_storage_bytes > exSumTempBytes) {
    std::cerr << "ERROR: NOT ENOUGH TEMP SPACE ALLOCATED" << std::endl;
  }
  // RMM_ALLOC(&_d_temp_storage, _temp_storage_bytes, 0);
  cub::DeviceReduce::Sum(d_exSumTemp, _temp_storage_bytes, d_countCommon, 
                              d_Common, tableSize);
  cudaMemcpy(&h_Common[tid], d_Common, 1 * sizeof(index_t), cudaMemcpyDeviceToHost);
  // gpu::copyToHost<int32_t>(d_Common.data(), 1, &h_Common);
  // RMM_FREE(_d_temp_storage, 0);
  // gpu::free(_d_temp_storage);

  _d_temp_storage=nullptr; _temp_storage_bytes=0;
  cub::DeviceScan::ExclusiveSum(_d_temp_storage, _temp_storage_bytes, d_countCommon, 
                              d_outputPositions, tableSize);

  if (_temp_storage_bytes > exSumTempBytes) {
    std::cerr << "ERROR: NOT ENOUGH TEMP SPACE ALLOCATED" << std::endl;
  }

  // RMM_ALLOC(&_d_temp_storage, _temp_storage_bytes, 0);
  cub::DeviceScan::ExclusiveSum(d_exSumTemp, _temp_storage_bytes, d_countCommon, 
                              d_outputPositions, tableSize);
  // RMM_FREE(_d_temp_storage, 0);
  // gpu::free(_d_temp_storage);


  // printf("Size of the ouput is : %ld\n", h_Common[tid]); fflush(stdout);

  if (h_Common[tid] > 0) {
    // d_output =  mem_t<keypair>(h_Common,context,memory_space_device);
    cudaMalloc(&h_dOutput[tid], h_Common[tid] * sizeof(keypair));
    // RMM_ALLOC(&h_dOutput[tid], h_Common[tid] * sizeof(keypair), 0);
  }


  simpleIntersect<<<BLOCK_COUNT, BLOCK_SIZE_OP2>>>(tableSize, d_offsetA, d_edgesA, d_offsetB,
                                                        d_edgesB, d_outputPositions, 
                                                        h_dOutput[tid], false);
  // forAll (tableSize,simpleIntersect<false>{d_offSetA.data(),d_edgesA.data(),
  //         d_offSetB.data(),d_edgesB.data(),d_outputPositions.data(),
  //         d_output.data()});

#ifdef ERROR_CHECK
  cudaDeviceSynchronize();
  CHECK_ERROR("intersect error");
#endif
}

void MultiHashGraph::buildSingle() {

    cudaSetDevice(0);
    std::cout << "single countSize: " << countSize << std::endl;
    std::cout << "single tableSize: " << tableSize << std::endl;

    // mem_t<HashKey>  d_hashA(countSize, context, memory_space_device);
    // mem_t<int32_t>  d_counterA = fill((int32_t)0, (size_t)(tableSize+1), context);
    // mem_t<index_t>  d_offsetA = fill((index_t)0, (size_t)(tableSize+1), context);
    // mem_t<keyval>   d_edgesA(countSize,context,memory_space_device);
    HashKey *d_hashA;
    int32_t *d_counterA;
    index_t * d_offsetA;
    keyval *d_edgesA;

    cudaMalloc(&d_hashA, countSize * sizeof(HashKey));
    cudaMalloc(&d_counterA, (tableSize + 1) * sizeof(int32_t));
    cudaMalloc(&d_offsetA, (tableSize + 1) * sizeof(index_t));
    cudaMalloc(&d_edgesA, countSize * sizeof(keyval));
    
    cudaMemset(d_counterA, 0, (tableSize + 1) * sizeof(int32_t));
    cudaMemset(d_offsetA, 0, (tableSize + 1) * sizeof(index_t));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float buildTime = 0.0f; // milliseoncds

    cudaMalloc(&d_vals, countSize * sizeof(hkey_t));
    cudaMemcpy(d_vals, h_vals, countSize * sizeof(hkey_t), cudaMemcpyHostToDevice); 

    cudaEventRecord(start);

    buildTable(d_vals, d_hashA, d_counterA, d_offsetA, d_edgesA, (index_t)countSize, 
                  (index_t)(tableSize));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&buildTime, start, stop);

    std::cout << "single buildTable() time: " << (buildTime / 1000.0) << "\n"; // seconds

    index_t *h_offset = new index_t[tableSize + 1]();
    HashKey *h_hash = new HashKey[countSize]();
    keyval *h_edges = new keyval[countSize]();

    cudaMemcpy(h_offset, d_offsetA, (tableSize + 1) * sizeof(index_t), 
                        cudaMemcpyDeviceToHost);

    cudaMemcpy(h_hash, d_hashA, countSize * sizeof(HashKey),
                        cudaMemcpyDeviceToHost);
    cudaMemcpy(h_edges, d_edgesA, countSize * sizeof(keyval),
                        cudaMemcpyDeviceToHost);

    // Everything in multi-GPU HG is in single-GPU HG
    for (index_t i = 0; i < gpuCount; i++) {
      cudaSetDevice(i);

      index_t hashRange = h_binSplits[i + 1] - h_binSplits[i];
      index_t keyCount = h_hFinalOffset[i][gpuCount];

      index_t *h_hOffsets = new index_t[hashRange + 1]();
      keyval *h_hEdges = new keyval[keyCount]();

      cudaMemcpy(h_hOffsets, h_dFinalKeys[i] + h_offsetOff[i],
                                               (hashRange + 1) * sizeof(index_t), 
                                               cudaMemcpyDeviceToHost);

      cudaMemcpy(h_hEdges, h_dFinalKeys[i] + h_edgesOff[i],
                                             keyCount * sizeof(keyval), 
                                             cudaMemcpyDeviceToHost);

      for (index_t j = 0; j < hashRange; j++) {

        index_t hash = j + h_binSplits[i];

        index_t multiDegree = h_hOffsets[j + 1] - h_hOffsets[j];
        index_t singleDegree = h_offset[hash + 1] - h_offset[hash];

        if (multiDegree != singleDegree) {
         std::cerr << "Degree error hash: " << hash  << " multi: " << 
              multiDegree << " single: " << singleDegree << "\n";
        }

        std::vector<hkey_t> multiGPU;
        for (index_t k = h_hOffsets[j]; k < h_hOffsets[j + 1]; k++) {
          keyval edge = h_hEdges[k];
          multiGPU.push_back(edge.key);
        }

        std::vector<hkey_t> singleGPU;
        for (index_t k = h_offset[hash]; k < h_offset[hash + 1]; k++) {
          keyval edge = h_edges[k];
          singleGPU.push_back(edge.key);
        }

        std::sort(multiGPU.begin(), multiGPU.end());
        std::sort(singleGPU.begin(), singleGPU.end());

        if (multiGPU != singleGPU) {
          std::cerr << "List error\n";

          std::cerr << "multiGPU:\n";
          for (hkey_t kv : multiGPU) {
            std::cerr << kv << " ";
          }
          std::cerr << std::endl;

          std::cerr << "singleGPU:\n";
          for (hkey_t kv : singleGPU) {
            std::cerr << kv << " ";
          }
          std::cerr << std::endl;
        }
      }
    }
    cudaSetDevice(0);

    cudaFree(d_vals);
    CHECK_ERROR("buildSingle");
}
