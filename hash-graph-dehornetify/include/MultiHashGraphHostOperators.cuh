#include "MultiHashGraph.cuh"

int BLOCK_COUNT = -1;
int BLOCK_SIZE_OP2 = 64; // TODO: Double check this.

// A recursive binary search function. It returns location of x in given array arr[l..r] is present, 
// otherwise it returns the bin id with the smallest value larger than x
size_t binarySearch(index_t *bins, size_t l, size_t r, size_t x) { 
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

void countBinSizes(inputData *h_dVals, index_t **h_hBinSizes, index_t **h_dBinSizes, 
                      index_t *h_binSizes, index_t *h_psBinSizes, index_t *h_binSplits,
                      index_t **h_dBinSplits, index_t countSize, index_t tableSize, 
                      index_t binRange, index_t binCount, index_t gpuCount, index_t tid) {

  // Count bin sizes for the keys stored on each GPU.
  // Bins are of hash values.

  countBinSizes<<<BLOCK_COUNT, BLOCK_SIZE_OP2>>>(h_dVals[tid].d_hash, h_dVals[tid].len, 
  // countBinSizes<<<BLOCK_COUNT, BLOCK_SIZE_OP2>>>(h_dVals[tid].d_keys + h_dVals[tid].len, h_dVals[tid].len, 
                                                    h_dBinSizes[tid], binRange);

  // Consolidate bin sizes across GPUs.
  cudaMemcpyAsync(h_hBinSizes[tid], h_dBinSizes[tid], binCount * sizeof(index_t), 
                      cudaMemcpyDeviceToHost);

  #pragma omp barrier

  #pragma omp master
  {
    for (index_t i = 0; i < gpuCount; i++) {
      for (index_t j = 0; j < binCount; j++) {
        h_binSizes[j] += h_hBinSizes[i][j];
      }
    }

    h_psBinSizes[0] = 0;
    for (index_t i = 1; i < binCount; i++) {
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
    for (index_t i = 0; i < binCount; i++) {
      if (h_binSizes[i] > 0) {
        std::cout << "i: " << i << " " << h_binSizes[i] << " ";
      }
    }
    std::cout << std::endl;
  } // debug master
  #pragma omp barrier
#endif

  // Find split points in prefix sum to determine what bins should go to each GPU.
  // TODO: This can probably be parallelized (maybe on device?)
  // might not be worth though, not a lot of work
  index_t avgKeyCount = std::ceil(countSize / ((float)gpuCount));
  index_t upperVal = avgKeyCount * (tid + 1);
  index_t upperIdx = binarySearch(h_psBinSizes, 0, binCount, upperVal);

  index_t minRange = upperIdx * binRange; 
  h_binSplits[tid + 1] = std::min(minRange, tableSize);

  #pragma omp barrier

  cudaMemcpyAsync(h_dBinSplits[tid], h_binSplits, (gpuCount + 1) * sizeof(index_t),
                      cudaMemcpyHostToDevice);

#ifdef DEBUG
  #pragma omp barrier
  #pragma omp master
  {
    std::cout << "h_binSplits" << std::endl;
    for (index_t i = 0; i < gpuCount + 1; i++) {
      std::cout << h_binSplits[i] << " ";
    }
    std::cout << std::endl;
  } // debug master
  #pragma omp barrier
#endif
  
}

void countKeyBuffSizes(inputData *h_dVals, index_t **h_dBinSplits, 
                          index_t **h_bufferCounter, index_t **h_dBufferCounter, 
                          index_t gpuCount, index_t tid) {

  // Clear counters
  cudaMemset(h_dBufferCounter[tid], 0, gpuCount * sizeof(index_t));

  // TODO: replace this with partitionRelabel from cuSort.
  countKeyBuffSizes<<<BLOCK_COUNT, BLOCK_SIZE_OP2>>>(h_dVals[tid].d_hash, h_dVals[tid].len,
  // countKeyBuffSizes<<<BLOCK_COUNT, BLOCK_SIZE_OP2>>>(h_dVals[tid].d_keys + h_dVals[tid].len, h_dVals[tid].len,
                                                        h_dBufferCounter[tid],
                                                        h_dBinSplits[tid], 
                                                        gpuCount);

  // cudaMemcpyAsync(h_bufferCounter[tid], h_dBufferCounter[tid], gpuCount * sizeof(index_t),
  cudaMemcpy(h_bufferCounter[tid], h_dBufferCounter[tid], gpuCount * sizeof(index_t),
                        cudaMemcpyDeviceToHost);


#ifdef DEBUG
  std::cout << "h_keyBins" << std::endl;
  for (index_t i = 0; i < binCount; i++) {
    std::cout << "bin" << i << ": ";
    for (index_t j = 0; j < h_binCounter[i]; j++) {
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
                          index_t **h_dKeyBinOff,
                          index_t **h_hKeyBinOff, index_t **h_dBufferCounter, 
                          index_t **h_bufferCounter, index_t **h_dBinSplits, 
                          size_t **h_dExSumTemp, size_t exSumTempBytes, 
                          index_t gpuCount, index_t tid) {

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

  // cudaMemcpyAsync(h_hKeyBinOff[tid], h_dKeyBinOff[tid], (gpuCount + 1) * sizeof(index_t),
  cudaMemcpy(h_hKeyBinOff[tid], h_dKeyBinOff[tid], (gpuCount + 1) * sizeof(index_t),
                    cudaMemcpyDeviceToHost);

  // Reset counters to 0 to actually fill buffers.
  cudaMemset(h_dBufferCounter[tid], 0, gpuCount * sizeof(index_t));


  // Buffer keys values according to which GPU they should be sent to based on hash range.
  binHashValues<<<BLOCK_COUNT, BLOCK_SIZE_OP2>>>(h_dVals[tid].len, h_dVals[tid].d_keys, 
                                                    h_dVals[tid].d_hash,
                                                    // h_dVals[tid].d_keys + h_dVals[tid].len,
                                                    h_dKeyBinBuff[tid],
                                                    h_dKeyBinOff[tid], h_dBinSplits[tid], 
                                                    h_dBufferCounter[tid], gpuCount, tid);

  // Copy counters and offsets to host
  // cudaMemcpyAsync(h_bufferCounter[tid], h_dBufferCounter[tid], gpuCount * sizeof(index_t),
  cudaMemcpy(h_bufferCounter[tid], h_dBufferCounter[tid], gpuCount * sizeof(index_t),
                    cudaMemcpyDeviceToHost);

#ifdef DEBUG
  #pragma omp barrier
  #pragma omp master
  {
    std::cout << "buffSizes:" << "\n";
    for (index_t i = 0; i < gpuCount; i++) {
      std::cout << "gpu: " << i << " count: " << h_bufferCounter[i] << "\n";
    }
    std::cout << std::endl;
  }
  #pragma omp barrier
#endif
}

// void countFinalKeys(index_t **h_bufferCounter, hkey_t **h_dFinalKeys,
#ifdef MANAGED_MEM
void countFinalKeys(index_t **h_bufferCounter, char **h_dFinalKeys,
                        index_t **h_hFinalCounters, 
                        index_t **h_hFinalOffset, index_t **h_dFinalOffset, 
                        index_t *h_binSplits, index_t gpuCount, index_t tid,
                        char *uvmPtr, index_t *prefixArray, index_t totalSize) {
#else
void countFinalKeys(index_t **h_bufferCounter, char **h_dFinalKeys,
                        index_t **h_hFinalCounters, 
                        index_t **h_hFinalOffset, index_t **h_dFinalOffset, 
                        index_t *h_binSplits, index_t gpuCount, index_t tid) {
#endif

  // h_hFinalCounters is the transpose of h_bufferCounter
  // h_hFinalCounters[i][j] is the number of keys GPU i receives from GPU j.
  #pragma omp barrier
  for (index_t j = 0; j < gpuCount; j++) {
    h_hFinalCounters[tid][j] = h_bufferCounter[j][tid];
  }

  // Prefix sum over all final counters.
  h_hFinalOffset[tid][0] = 0;
  for (index_t j = 1; j < gpuCount + 1; j++) {
    h_hFinalOffset[tid][j] = h_hFinalOffset[tid][j - 1] + h_hFinalCounters[tid][j - 1];
  }

  // cudaMemcpyAsync(h_dFinalOffset[tid], h_hFinalOffset[tid], (gpuCount + 1) * sizeof(index_t),
  //                     cudaMemcpyHostToDevice);

  index_t keyCount = h_hFinalOffset[tid][gpuCount];
  index_t hashRange = h_binSplits[tid + 1] - h_binSplits[tid];

#ifdef DEBUG
  #pragma omp barrier
  #pragma omp master
  {
    std::cout << "buffSizes:" << "\n";
    for (index_t i = 0; i < gpuCount; i++) {
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
    for (index_t i = 1; i < gpuCount; i++) {
      index_t tidKeyCount = h_hFinalOffset[i - 1][gpuCount];
      index_t tidHashRange = h_binSplits[i] - h_binSplits[i - 1];
      index_t size = tidKeyCount * sizeof(keyval) + 
                                 tidKeyCount * sizeof(HashKey) +
                                 (2 * tidKeyCount * sizeof(keyval)) +
                                 (2 * (tidHashRange + 1) * sizeof(index_t));

      prefixArray[i] = prefixArray[i - 1] + size;
    }
    prefixArray[gpuCount] = totalSize;

    h_dFinalKeys[0] = uvmPtr;
    for (index_t i = 1; i < gpuCount; i++) {
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
  index_t size = keyCount * sizeof(keyval) + 
                                 keyCount * sizeof(HashKey) +
                                 (2 * keyCount * sizeof(keyval)) +
                                 (2 * (hashRange + 1) * sizeof(index_t));
#endif
}

// void allToAll(inputData *h_dVals, hkey_t **h_dFinalKeys,
void allToAll(inputData *h_dVals, char **h_dFinalKeys,
                  // index_t **h_hFinalOffset, hkey_t **h_dKeyBinBuff,
                  index_t **h_hFinalOffset, keyval **h_dKeyBinBuff,
                  index_t **h_hKeyBinOff, index_t **h_hFinalCounters, index_t gpuCount,
                  index_t tid) {

  for (index_t j = 0; j < gpuCount; j++) {
    // Ship keys + hashes from GPU i to GPU j
    index_t keyCount = h_hFinalCounters[tid][j];
    // index_t keyCount = h_hFinalCounters[j][tid];
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
                    index_t tableSize, index_t valsOffset=0) {

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

