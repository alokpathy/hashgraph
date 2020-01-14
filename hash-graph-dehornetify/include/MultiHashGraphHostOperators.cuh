#include "MultiHashGraph.cuh"

int BLOCK_COUNT = -1;
int BLOCK_SIZE_OP2 = 64; // TODO: Double check this.

// A recursive binary search function. It returns location of x in given array arr[l..r] is present, 
// otherwise it returns the bin id with the smallest value larger than x
size_t binarySearch(uint64_t *bins, size_t l, size_t r, size_t x) { 
  if (r >= l) { 
    size_t mid = l + (r - l) / 2; 
  
    // If the element is present at the middle itself 
    if (bins[mid] == x) 
        return mid; 

    // If element is smaller than mid, then it can only be present in left subarray 
    if (bins[mid] > x) 
        return binarySearch(bins, l, mid - 1, x); 

    // Else the element can only be present in right subarray 
    return binarySearch(bins, mid + 1, r, x); 
  } 
  
  // We reach here when element is not present in array and return the bin id 
  // of the smallest value greater than x
  return l; 
}

void countBinSizes(inputData *h_dVals, uint64_t **h_hBinSizes, uint64_t **h_dBinSizes, 
                      uint64_t *h_binSizes, uint64_t *h_psBinSizes, uint64_t *h_binSplits,
                      uint64_t **h_dBinSplits, int64_t countSize, int64_t tableSize, 
                      uint64_t binRange, uint64_t binCount, uint64_t gpuCount, uint64_t tid) {

  // Count bin sizes for the keys stored on each GPU.
  // Bins are of hash values.

  countBinSizes<<<BLOCK_COUNT, BLOCK_SIZE_OP2>>>(h_dVals[tid].d_hash, h_dVals[tid].len, 
  // countBinSizes<<<BLOCK_COUNT, BLOCK_SIZE_OP2>>>(h_dVals[tid].d_keys + h_dVals[tid].len, h_dVals[tid].len, 
                                                    h_dBinSizes[tid], binRange);

#ifdef DEBUG
  #pragma omp barrier
  #pragma omp master
  {
    std::cout << "h_binSizes:" << std::endl;
    for (uint64_t i = 0; i < binCount; i++) {
      std::cout << h_binSizes[i] << " ";
    }
    std::cout << std::endl;
  } // debug master
  #pragma omp barrier
#endif

  // Consolidate bin sizes across GPUs.
  cudaMemcpyAsync(h_hBinSizes[tid], h_dBinSizes[tid], binCount * sizeof(uint64_t), 
                      cudaMemcpyDeviceToHost);

  #pragma omp barrier

  #pragma omp master
  {
    for (uint64_t i = 0; i < gpuCount; i++) {
      for (uint64_t j = 0; j < binCount; j++) {
        h_binSizes[j] += h_hBinSizes[i][j];
      }
    }

    h_psBinSizes[0] = 0;
    for (uint64_t i = 1; i < binCount; i++) {
      h_psBinSizes[i] = h_psBinSizes[i - 1] + h_binSizes[i - 1];
    }
    h_psBinSizes[binCount] = countSize;
  } // master

  #pragma omp barrier

#ifdef DEBUG
  #pragma omp barrier
  #pragma omp master
  {
    std::cout << "h_binSizes:" << std::endl;
    for (uint64_t i = 0; i < binCount; i++) {
      std::cout << h_binSizes[i] << "\n";
    }
    std::cout << std::endl;
  } // debug master
  #pragma omp barrier
#endif

  // Find split points in prefix sum to determine what bins should go to each GPU.
  // TODO: This can probably be parallelized (maybe on device?)
  // might not be worth though, not a lot of work
  uint64_t avgKeyCount = std::ceil(countSize / ((float)gpuCount));
  uint64_t upperVal = avgKeyCount * (tid + 1);
  uint64_t upperIdx = binarySearch(h_psBinSizes, 0, binCount, upperVal);

  int64_t minRange = upperIdx * binRange; 
  h_binSplits[tid + 1] = std::min(minRange, tableSize);

  #pragma omp barrier

  cudaMemcpyAsync(h_dBinSplits[tid], h_binSplits, (gpuCount + 1) * sizeof(uint64_t),
                      cudaMemcpyHostToDevice);
#ifdef DEBUG
  #pragma omp barrier
  #pragma omp master
  {
    std::cout << "h_binSplits" << std::endl;
    for (uint64_t i = 0; i < gpuCount + 1; i++) {
      std::cout << h_binSplits[i] << " ";
    }
    std::cout << std::endl;
  } // debug master
  #pragma omp barrier
#endif
  
}

void countKeyBuffSizes(inputData *h_dVals, uint64_t **h_dBinSplits, 
                          uint64_t **h_bufferCounter, uint64_t **h_dBufferCounter, 
                          uint64_t gpuCount, uint64_t tid) {

  // Clear counters
  cudaMemset(h_dBufferCounter[tid], 0, gpuCount * sizeof(uint64_t));

  // TODO: replace this with partitionRelabel from cuSort.
  countKeyBuffSizes<<<BLOCK_COUNT, BLOCK_SIZE_OP2>>>(h_dVals[tid].d_hash, h_dVals[tid].len,
  // countKeyBuffSizes<<<BLOCK_COUNT, BLOCK_SIZE_OP2>>>(h_dVals[tid].d_keys + h_dVals[tid].len, h_dVals[tid].len,
                                                        h_dBufferCounter[tid],
                                                        h_dBinSplits[tid], 
                                                        gpuCount);

  // cudaMemcpyAsync(h_bufferCounter[tid], h_dBufferCounter[tid], gpuCount * sizeof(uint64_t),
  cudaMemcpy(h_bufferCounter[tid], h_dBufferCounter[tid], gpuCount * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost);

#ifdef DEBUG
  std::cout << "h_keyBins" << std::endl;
  for (uint64_t i = 0; i < binCount; i++) {
    std::cout << "bin" << i << ": ";
    for (uint64_t j = 0; j < h_binCounter[i]; j++) {
      std::cout << h_keyBins[i][j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  delete[] h_binCounter;
#endif
  
}

// void populateKeyBuffs(inputData *h_dVals, hkey_t **h_dKeyBinBuff, 
void populateKeyBuffs(inputData *h_dVals, keyval **h_dKeyBinBuff, 
                          uint64_t **h_dKeyBinOff,
                          uint64_t **h_hKeyBinOff, uint64_t **h_dBufferCounter, 
                          uint64_t **h_bufferCounter, uint64_t **h_dBinSplits, 
                          size_t **h_dExSumTemp, size_t exSumTempBytes, 
                          uint64_t gpuCount, uint64_t tid) {

  // Compute offset into original key array based on binning
  void*  _d_temp_storage     { nullptr };
  size_t _temp_storage_bytes { 0 };

  cub::DeviceScan::ExclusiveSum(NULL, _temp_storage_bytes, h_dBufferCounter[tid], 
                                    h_dKeyBinOff[tid], gpuCount);

  if (_temp_storage_bytes > exSumTempBytes) {
    std::cerr << "ERROR: NOT ENOUGH TEMP SPACE ALLOCATED" << std::endl;
  }

  cub::DeviceScan::ExclusiveSum(h_dExSumTemp[tid], _temp_storage_bytes, h_dBufferCounter[tid], 
                                    h_dKeyBinOff[tid], gpuCount);

  // cudaMemcpyAsync(h_hKeyBinOff[tid], h_dKeyBinOff[tid], (gpuCount + 1) * sizeof(uint64_t),
  cudaMemcpy(h_hKeyBinOff[tid], h_dKeyBinOff[tid], (gpuCount + 1) * sizeof(uint64_t),
                    cudaMemcpyDeviceToHost);

  // Reset counters to 0 to actually fill buffers.
  cudaMemset(h_dBufferCounter[tid], 0, gpuCount * sizeof(uint64_t));


  // Buffer keys values according to which GPU they should be sent to based on hash range.
  binHashValues<<<BLOCK_COUNT, BLOCK_SIZE_OP2>>>(h_dVals[tid].len, h_dVals[tid].d_keys, 
                                                    h_dVals[tid].d_hash,
                                                    // h_dVals[tid].d_keys + h_dVals[tid].len,
                                                    h_dKeyBinBuff[tid],
                                                    h_dKeyBinOff[tid], h_dBinSplits[tid], 
                                                    h_dBufferCounter[tid], gpuCount, tid);

  // Copy counters and offsets to host
  // cudaMemcpyAsync(h_bufferCounter[tid], h_dBufferCounter[tid], gpuCount * sizeof(uint64_t),
  cudaMemcpy(h_bufferCounter[tid], h_dBufferCounter[tid], gpuCount * sizeof(uint64_t),
                    cudaMemcpyDeviceToHost);

#ifdef DEBUG
  #pragma omp barrier
  #pragma omp master
  {
    std::cout << "buffSizes:" << "\n";
    for (int64_t i = 0; i < gpuCount; i++) {
      std::cout << "gpu: " << i << " count: " << h_bufferCounter[i] << "\n";
    }
    std::cout << std::endl;
  }
  #pragma omp barrier
#endif
}

// void countFinalKeys(uint64_t **h_bufferCounter, hkey_t **h_dFinalKeys,
#ifdef MANAGED_MEM
void countFinalKeys(uint64_t **h_bufferCounter, char **h_dFinalKeys,
                        uint64_t **h_hFinalCounters, 
                        uint64_t **h_hFinalOffset, uint64_t **h_dFinalOffset, 
                        uint64_t *h_binSplits, uint64_t gpuCount, uint64_t tid,
                        char *uvmPtr, uint64_t *prefixArray, uint64_t totalSize) {
#else
void countFinalKeys(uint64_t **h_bufferCounter, char **h_dFinalKeys,
                        uint64_t **h_hFinalCounters, 
                        uint64_t **h_hFinalOffset, uint64_t **h_dFinalOffset, 
                        uint64_t *h_binSplits, uint64_t gpuCount, uint64_t tid) {
#endif

  // h_hFinalCounters is the transpose of h_bufferCounter
  // h_hFinalCounters[i][j] is the number of keys GPU i receives from GPU j.
  for (uint64_t j = 0; j < gpuCount; j++) {
    h_hFinalCounters[tid][j] = h_bufferCounter[j][tid];
  }

  // Prefix sum over all final counters.
  h_hFinalOffset[tid][0] = 0;
  for (uint64_t j = 1; j < gpuCount + 1; j++) {
    h_hFinalOffset[tid][j] = h_hFinalOffset[tid][j - 1] + h_hFinalCounters[tid][j - 1];
  }

  // cudaMemcpyAsync(h_dFinalOffset[tid], h_hFinalOffset[tid], (gpuCount + 1) * sizeof(uint64_t),
  //                     cudaMemcpyHostToDevice);

  uint64_t keyCount = h_hFinalOffset[tid][gpuCount];
  uint64_t hashRange = h_binSplits[tid + 1] - h_binSplits[tid];

#ifdef DEBUG
  #pragma omp barrier
  #pragma omp master
  {
    std::cout << "buffSizes:" << "\n";
    for (int64_t i = 0; i < gpuCount; i++) {
      std::cout << "gpu: " << i << " count: " << h_hFinalOffset[i][gpuCount] << "\n";
    }
    std::cout << std::endl;
  }
  #pragma omp barrier
#endif

  // cudaMalloc(&h_dFinalKeys[tid], (4 * keyCount * sizeof(hkey_t)) + 
  // [ keys | hash | edges | lrbArray | offset | counter ]
  // (len = keyCount, type = keyval)    (len = hashRange + 1, type = index_t)
  // except hash, hash is type HashKey
  // cudaMalloc(&h_dFinalKeys[tid], (4 * keyCount * sizeof(keyval)) +
  //                               (2 * (hashRange + 1) * sizeof(index_t)));
  // RMM_ALLOC(&h_dFinalKeys[tid], keyCount * sizeof(keyval) + 
#ifdef MANAGED_MEM
  #pragma omp barrier
  #pragma omp master
  {
    prefixArray[0] = 0;
    for (uint64_t i = 1; i < gpuCount; i++) {
      uint64_t tidKeyCount = h_hFinalOffset[i - 1][gpuCount];
      uint64_t tidHashRange = h_binSplits[i] - h_binSplits[i - 1];
      uint64_t size = tidKeyCount * sizeof(keyval) + 
                                 tidKeyCount * sizeof(HashKey) +
                                 (2 * tidKeyCount * sizeof(keyval)) +
                                 (2 * (tidHashRange + 1) * sizeof(index_t));

      prefixArray[i] = prefixArray[i - 1] + size;
    }
    prefixArray[gpuCount] = totalSize;

    h_dFinalKeys[0] = uvmPtr;
    for (uint64_t i = 1; i < gpuCount; i++) {
      h_dFinalKeys[i] = uvmPtr + prefixArray[i];
    }
  }
  #pragma omp barrier

  cudaMemPrefetchAsync(h_dFinalKeys[tid], prefixArray[tid + 1] - prefixArray[tid], tid);

#else
  cudaMalloc(&h_dFinalKeys[tid], keyCount * sizeof(keyval) + 
                                 keyCount * sizeof(HashKey) +
                                 (2 * keyCount * sizeof(keyval)) +
                                 (2 * (hashRange + 1) * sizeof(index_t)));
  uint64_t size = keyCount * sizeof(keyval) + 
                                 keyCount * sizeof(HashKey) +
                                 (2 * keyCount * sizeof(keyval)) +
                                 (2 * (hashRange + 1) * sizeof(index_t));
#endif
}

// void allToAll(inputData *h_dVals, hkey_t **h_dFinalKeys,
void allToAll(inputData *h_dVals, char **h_dFinalKeys,
                  // uint64_t **h_hFinalOffset, hkey_t **h_dKeyBinBuff,
                  uint64_t **h_hFinalOffset, keyval **h_dKeyBinBuff,
                  uint64_t **h_hKeyBinOff, uint64_t **h_hFinalCounters, uint64_t gpuCount,
                  uint64_t tid) {

  for (uint64_t j = 0; j < gpuCount; j++) {
    // Ship keys + hashes from GPU i to GPU j
    uint64_t keyCount = h_hFinalCounters[tid][j];
    // uint64_t keyCount = h_hFinalCounters[j][tid];
    // cudaMemcpyAsync(h_dFinalKeys[j] + h_hFinalOffset[j][tid], h_dKeyBinBuff[tid] + h_hKeyBinOff[tid][j],
    //                  keyCount * sizeof(keyval), cudaMemcpyDeviceToDevice);
    // cudaMemcpyAsync(h_dFinalKeys[j] + (h_hFinalOffset[j][tid] * sizeof(keyval)),
    //                   h_dKeyBinBuff[tid] + h_hKeyBinOff[tid][j],
    //                   keyCount * sizeof(keyval), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(h_dFinalKeys[tid] + (h_hFinalOffset[tid][j] * sizeof(keyval)),
                      h_dKeyBinBuff[j] + h_hKeyBinOff[j][tid],
                      keyCount * sizeof(keyval), cudaMemcpyDeviceToDevice);
  }
}

template<typename hkey_t, typename HashKey,  typename index_t>
void buildTable(hkey_t *d_vals, HashKey *d_hash, int32_t *d_counter, 
	            index_t *d_offSet, keyval *d_edges, index_t valCount, 
                    index_t tableSize, int64_t valsOffset=0) {

  void*  _d_temp_storage     { nullptr };
  size_t _temp_storage_bytes { 0 };

  // hashValuesD<<<BLOCK_COUNT, BLOCK_SIZE_OP2>>>(valCount, d_vals + valsOffset, 
  //                                                  d_hash.data(), (HashKey) tableSize);
  basicHashD<<<BLOCK_COUNT, BLOCK_SIZE_OP2>>>(valCount, d_vals + valsOffset, 
                                                  d_hash, (HashKey) tableSize);

  countHashD32<<<BLOCK_COUNT, BLOCK_SIZE_OP2>>>(valCount, d_hash, d_counter);
  cub::DeviceScan::ExclusiveSum(_d_temp_storage, _temp_storage_bytes,d_counter, 
                                    d_offSet, tableSize);
  cudaMalloc(&_d_temp_storage, _temp_storage_bytes);
  cub::DeviceScan::ExclusiveSum(_d_temp_storage, _temp_storage_bytes,d_counter, 
                                    d_offSet, tableSize);
  // d_counter = fill(0, (size_t)tableSize, context);
  cudaMemset(d_counter, 0, tableSize * sizeof(int32_t));
  cudaMemcpy(d_offSet + tableSize, &valCount, sizeof(index_t), cudaMemcpyHostToDevice);
  cudaFree(_d_temp_storage);

  copyToGraphD32<<<BLOCK_COUNT, BLOCK_SIZE_OP2>>>(valCount, d_vals + valsOffset, 
                                    d_hash, d_counter, d_offSet, 
                                    d_edges, tableSize);
}

