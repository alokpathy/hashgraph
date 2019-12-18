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

#include <algorithm>
#include <unistd.h>

// #define RAND_KEYS
// #define PRINT_KEYS

// #define BUILD_TEST

struct prg {
  hkey_t lo, hi;

  __host__ __device__ prg(hkey_t _lo=0, hkey_t _hi=0) : lo(_lo), hi(_hi) {};

  __host__ __device__ hkey_t operator()(unsigned long long index) const {
    thrust::default_random_engine rng(index);
    thrust::uniform_int_distribution<hkey_t> dist(lo, hi);
    rng.discard(index);
    return dist(rng);
  }
};

// A recursive binary search function. It returns location of x in given array arr[l..r] is present, 
// otherwise it returns the bin id with the smallest value larger than x
int64_t binarySearch(hkey_t *bins, int32_t l, int64_t r, int32_t x) {
  if (r >= l) { 
    int64_t mid = l + (r - l) / 2; 
  
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

void enablePeerAccess(uint32_t gpuCount) {
  // Enable P2P access between each pair of GPUs.
  for (int64_t j = 0; j < gpuCount; j++) {
    cudaSetDevice(j);
    for (int64_t i = 0; i < gpuCount; i++) {
      if (j != i) {
        int isCapable;
        cudaDeviceCanAccessPeer(&isCapable, j, i);
        if (isCapable == 1) {
          cudaError_t err = cudaDeviceEnablePeerAccess(i, 0);
          if (err == cudaErrorPeerAccessAlreadyEnabled) {
            cudaGetLastError();
          }
        }
      }
    }
  }
} 

void generateInput(inputData *h_dVals, int64_t countSize, int64_t maxkey, uint32_t gpuCount,
                        uint64_t seed) {
  std::cout << "generating input" << std::endl;

  uint64_t avgKeyCount = std::ceil(countSize / ((double) gpuCount));
  for (uint64_t i = 0; i < gpuCount; i++) {
    cudaSetDevice(i);

    int64_t lo = avgKeyCount * i;
    int64_t hi = avgKeyCount * (i + 1);
    hi = std::min(hi, countSize);

    uint64_t keyCount = hi - lo;

    cudaMalloc(&h_dVals[i].d_keys, keyCount * sizeof(hkey_t));
    cudaMalloc(&h_dVals[i].d_hash, keyCount * sizeof(HashKey));
    // RMM_ALLOC(&h_dVals[i].d_keys, keyCount * sizeof(hkey_t), 0);
    // RMM_ALLOC(&h_dVals[i].d_hash, keyCount * sizeof(HashKey), 0);

#ifdef RAND_KEYS
    // Randomly generate input keys on each device.
    thrust::counting_iterator<hkey_t> index_sequence_begin(seed);
    thrust::transform(thrust::device, index_sequence_begin, index_sequence_begin + keyCount,
                        h_dVals[i].d_keys, prg(0, maxkey));
#else
    hkey_t *h_tmpKeys = new hkey_t[keyCount]();
    for (uint64_t j = lo; j < hi; j++) {
      h_tmpKeys[j - lo] = j;
    }
    cudaMemcpy(h_dVals[i].d_keys, h_tmpKeys, keyCount * sizeof(hkey_t), cudaMemcpyHostToDevice);
#endif

    h_dVals[i].len = keyCount;

#ifdef PRINT_KEYS
    std::cout << "keys gpu " << i << std::endl;
    thrust::device_ptr<hkey_t> td_keys = thrust::device_pointer_cast(h_dVals[i].d_keys);
    for (uint32_t j = 0; j < keyCount; j++) {
      std::cout << *(td_keys + j) << " ";
    }
    std::cout << std::endl;
#endif

    seed += keyCount;

  }
  std::cout << "done generating input" << std::endl;
}

int main(int argc, char **argv) {

  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);

  std::cout << "deviceCount: " << deviceCount << std::endl;

  char hostname[HOST_NAME_MAX];
  gethostname(hostname, HOST_NAME_MAX);

  std::cout << "hostname: " << hostname << std::endl;

  int64_t countSizeA = 1L << 24;
  HashKey maxkey = 1L << 26;

  uint32_t binCount = 16000;
  uint32_t gpuCount = 4;

  index_t lrbBins = -1;

  bool checkCorrectness = false;
  bool buildTest = false;

  int64_t countSizeB = 1L << 22;

  if (argc >= 2 && argc < 9) {
    std::cerr << "Please specify all arguments.\n";
    return 1;
  }

  if (argc >= 3) {
    int64_t size = strtoull(argv[1], NULL, 0);
    countSizeA = size;

    int64_t key = strtoull(argv[2], NULL, 0);
    maxkey = key;

    binCount = atoi(argv[3]);
    gpuCount = atoi(argv[4]);

    lrbBins = strtoull(argv[5], NULL, 0);

    // char *correctnessFlag = atoi(argv[5]);
    // if (correctnessFlag > 0) {
    if (!strcmp(argv[6], "check")) {
      checkCorrectness = true;
    }
    
    countSizeB = strtoull(argv[7], NULL, 0);

    if (!strcmp(argv[8], "build")) {
      buildTest = true;
    }
  } 

  HashKey tableSize = maxkey;

  std::cout << "countSizeA: " << countSizeA << std::endl;
  std::cout << "maxkey: " << maxkey << std::endl;

  // rmm_mgpu_context_t contextA;
  // rmm_mgpu_context_t contextB;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float buildTime = 0.0f; // milliseoncds

  // enablePeerAccess(gpuCount);

  // rmmOptions_t rmmO;

  // rmmO.initial_pool_size = 1L << 60;
  // rmmO.allocation_mode = PoolAllocation;
  // rmmO.enable_logging = false;
  // rmmO.num_devices = 16;

  // int *devices = (int *)malloc(gpuCount * sizeof(int));
  // for (int64_t i = 0; i < gpuCount; i++) {
  //   devices[i] = i;
  // }
  // 
  // rmmO.devices = devices;

  // rmmInitialize(&rmmO);


  if (buildTest) {
    inputData *h_dVals = new inputData[gpuCount]();
    generateInput(h_dVals, countSizeA, maxkey, gpuCount, 0);

    // MultiHashGraph mhg(h_dVals, countSizeA, maxkey, contextA, tableSize, binCount, lrbBins, gpuCount);
    MultiHashGraph mhg(h_dVals, countSizeA, maxkey, tableSize, binCount, lrbBins, gpuCount);

    omp_set_num_threads(gpuCount);

    cudaSetDevice(0);
    cudaEventRecord(start);

    #pragma omp parallel
    {
      uint64_t tid = omp_get_thread_num();
      mhg.build(true, tid);
    } // pragma

    cudaSetDevice(0);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&buildTime, start, stop);

    std::cout << "multi buildTable() time: " << (buildTime / 1000.0) << "\n"; // seconds

    if (checkCorrectness) {
      mhg.destroyMulti();
      mhg.buildSingle();
    }
  } else {
    inputData *h_dValsA = new inputData[gpuCount]();
    inputData *h_dValsB = new inputData[gpuCount]();

    generateInput(h_dValsA, countSizeA, maxkey, gpuCount, 0);
    generateInput(h_dValsB, countSizeB, maxkey, gpuCount, countSizeA);

    std::cout << "hashgraph constructors" << std::endl;
    // MultiHashGraph mhgA(h_dValsA, countSizeA, maxkey, contextA, tableSize, binCount, lrbBins, gpuCount);
    // MultiHashGraph mhgB(h_dValsB, countSizeB, maxkey, contextB, tableSize, binCount, lrbBins, gpuCount);
    MultiHashGraph mhgA(h_dValsA, countSizeA, maxkey, tableSize, binCount, lrbBins, gpuCount);
    MultiHashGraph mhgB(h_dValsB, countSizeB, maxkey, tableSize, binCount, lrbBins, gpuCount);
    std::cout << "done hashgraph constructors" << std::endl;

#ifdef MANAGED_MEM
    std::cout << "managed mem constructors" << std::endl;
    uint64_t size = 2 * (tableSize + gpuCount) * sizeof(int64_t);
    cudaMallocManaged(&mhgA.uvmPtrIntersect, size);
    mhgA.prefixArrayIntersect = new uint64_t[gpuCount + 1]();
    mhgA.totalSizeIntersect = size;
    std::cout << "done managed mem constructors" << std::endl;
#endif

    keypair **h_dOutput = new keypair*[gpuCount]();
    int64_t *h_Common = new int64_t[gpuCount]();

    omp_set_num_threads(gpuCount);

    cudaSetDevice(0);
    cudaEventRecord(start);

    #pragma omp parallel
    {
      uint64_t tid = omp_get_thread_num();
      mhgA.build(true, tid);

      #pragma omp master
      {
        mhgB.h_binSplits = mhgA.h_binSplits; // small memory leak.
        mhgB.h_dBinSplits = mhgA.h_dBinSplits;

#ifdef MANAGED_MEM
        mhgA.prefixArrayIntersect[0] = 0;
        for (uint64_t i = 1; i < gpuCount; i++) {
          uint64_t tidHashRange = mhgA.h_binSplits[i] - mhgA.h_binSplits[i - 1];
          uint64_t size = 2 * (tidHashRange + 1) * sizeof(int64_t);
          mhgA.prefixArrayIntersect[i] = mhgA.prefixArrayIntersect[i - 1] + size;
        }
        mhgA.prefixArrayIntersect[gpuCount] = mhgA.totalSizeIntersect;

        mhgA.h_dCountCommon[0] = mhgA.uvmPtrIntersect;
        for (uint64_t i = 1; i < gpuCount; i++) {
          mhgA.h_dCountCommon[i] = mhgA.uvmPtrIntersect + 
                                        mhgA.prefixArrayIntersect[i];
        }
#endif
      } // master

      #pragma omp barrier

      mhgB.build(false, tid); // Build second HG but use same splits as first HG.

      #pragma omp barrier

      MultiHashGraph::intersect(mhgA, mhgB, h_Common, h_dOutput, tid);
    } // pragma

    cudaSetDevice(0);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&buildTime, start, stop);

    std::cout << "multi intersect() time: " << (buildTime / 1000.0) << "\n"; // seconds

    if (checkCorrectness) {
      mhgA.buildSingle();
      mhgB.buildSingle();
      
      int64_t outputSize = 0;
      for (int64_t i = 0; i < gpuCount; i++) {
        outputSize += h_Common[i];
      }

      keypair *h_output = new keypair[outputSize]();
      int64_t h_idx = 0;
      for (int64_t i = 0; i < gpuCount; i++) {
        cudaSetDevice(i);
        cudaMemcpy(h_output + h_idx, h_dOutput[i], h_Common[i] * sizeof(keypair),
                            cudaMemcpyDeviceToHost);
        h_idx += h_Common[i];
      }

      std::vector<hkey_t> result;
      result.reserve(outputSize);
      for (int64_t i = 0; i < outputSize; i++) {
        result.push_back(h_output[i].right);
      }

      if (result.size() != result.capacity()) {
        std::cerr << "ERROR: RESULT ERROR" << std::endl;
        exit(0);
      }

      std::sort(mhgA.h_vals, mhgA.h_vals + countSizeA);
      std::sort(mhgB.h_vals, mhgB.h_vals + countSizeB);

      std::vector<hkey_t> ans;
      ans.reserve(outputSize);
      for (int64_t i = 0; i < countSizeA; i++) {
        int64_t ogIdx = binarySearch(mhgB.h_vals, 0, countSizeB - 1, mhgA.h_vals[i]);

        int64_t idx = ogIdx;
        while (idx >= 0 && mhgB.h_vals[idx] == mhgA.h_vals[i]) {
          ans.push_back(mhgA.h_vals[i]);
          idx--;
        }

        idx = ogIdx + 1;
        while (idx < countSizeB && mhgB.h_vals[idx] == mhgA.h_vals[i]) {
          ans.push_back(mhgA.h_vals[i]);
          idx++;
        }
        // for (int64_t j = 0; j < countSizeB; j++) {
        //   if (mhgA.h_vals[i] == mhgB.h_vals[j]) {
        //     ans.push_back(mhgA.h_vals[i]);
        //   }

        //   if (mhgA.h_vals[i] < mhgB.h_vals[j]) {
        //     break;
        //   }
        // } 
      }

      if (ans.size() != outputSize) {
        std::cerr << "ERROR: INTERSECT OUTPUT HAS INCORRECT SIZE" << std::endl;
        std::cerr << "ansSize: " << ans.size() << " outputSize: " << outputSize << std::endl;
        // exit(0);
      }

      std::sort(result.begin(), result.end());
      std::sort(ans.begin(), ans.end());

      if (result != ans) {
        std::cerr << "ERROR: INTERSECT OUTPUT HAS INCORRECT CONTENT" << std::endl;
        
        std::cout << "output: " << std::endl;
        for (auto i = result.begin(); i != result.end(); ++i) {
            std::cout << *i << " ";
        }
        std::cout << std::endl;

        std::cout << "ans: " << std::endl;
        for (auto i = ans.begin(); i != ans.end(); ++i) {
            std::cout << *i << " ";
        }
        std::cout << std::endl;

        exit(0);
      }
    }
  }
}
