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

#include <cuda_runtime_api.h>

// #define ID_HASH

__forceinline__  __host__ __device__ uint32_t rotl32( uint32_t x, int8_t r ) {
  return (x << r) | (x >> (32 - r));
}
__forceinline__ __host__ __device__ uint32_t fmix32( uint32_t h ) {
  h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;
  return h;
}
__forceinline__  __host__ __device__ uint32_t hash_murmur(const HashKey& key) {

#ifdef ID_HASH
  return (uint32_t) key;
#endif

  constexpr int len = sizeof(int);
  const uint8_t * const data = (const uint8_t*)&key;
  constexpr int nblocks = len / 4;
  uint32_t h1 = 0;
  constexpr uint32_t c1 = 0xcc9e2d51;
  constexpr uint32_t c2 = 0x1b873593;
  //----------
  // body
  const uint32_t * const blocks = (const uint32_t *)(data + nblocks*4);
  for(int i = -nblocks; i; i++)
  {
    uint32_t k1 = blocks[i];//getblock32(blocks,i);
    k1 *= c1;
    k1 = rotl32(k1,15);
    k1 *= c2;
    h1 ^= k1;
    h1 = rotl32(h1,13); 
    h1 = h1*5+0xe6546b64;
  }
  //----------
  // tail
  const uint8_t * tail = (const uint8_t*)(data + nblocks*4);
  uint32_t k1 = 0;
  switch(len & 3)
  {
    case 3: k1 ^= tail[2] << 16;
    case 2: k1 ^= tail[1] << 8;
    case 1: k1 ^= tail[0];
            k1 *= c1; k1 = rotl32(k1,15); k1 *= c2; h1 ^= k1;
  };
  //----------
  // finalization
  h1 ^= len;
  h1 = fmix32(h1);
  return h1;
}

// __global__ void basicHashD(uint64_t valCount, hkey_t *valsArr, HashKey *hashArr, int64_t tableSize) {
__global__ void basicHashD(index_t valCount, hkey_t *valsArr, HashKey *hashArr, index_t tableSize) {
  int64_t     id = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;

  for (auto i = id; i < valCount; i += stride) {
    hashArr[i] = (HashKey)(hash_murmur(valsArr[i]) % tableSize);
  }    
}

// __global__ void hashValuesD(uint64_t valCount, keyval *valsArr, keyval *hashArr, int64_t tableSize) {
// __global__ void hashValuesD(uint64_t valCount, keyval *valsArr, HashKey *hashArr, int64_t tableSize,
__global__ void hashValuesD(index_t valCount, keyval *valsArr, HashKey *hashArr, index_t tableSize,
                                index_t devNum) {
  int64_t     id = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;

  for (auto i = id; i < valCount; i += stride) {
    // hashArr[i].key = (HashKey)(hash_murmur(valsArr[i].key) % tableSize);
    hashArr[i] = (HashKey)(hash_murmur(valsArr[i].key) % tableSize);
  }    
}

// __global__ void countHashD(uint64_t valCount, HashKey *hashArr, index_t *countArr) {
__global__ void countHashD(index_t valCount, HashKey *hashArr, index_t *countArr) {
  int64_t     id = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;
  for (auto i = id; i < valCount; i += stride) {
    atomicAdd((unsigned long long int*)(countArr + hashArr[i]), 1);
    // atomicAdd((countArr + hashArr[i]), 1);
  }    
}

__global__ void countHashD32(index_t valCount, HashKey *hashArr, int32_t *countArr) {
  int64_t     id = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;
  for (auto i = id; i < valCount; i += stride) {
    atomicAdd(countArr + hashArr[i], 1);
  }    
}

// __global__ void copyToGraphD(uint64_t valCount, hkey_t *valsArr, HashKey *hashArr, index_t *countArr,
__global__ void copyToGraphD(index_t valCount, hkey_t *valsArr, HashKey *hashArr, index_t *countArr,
                                index_t *offsetArr, keyval *edges, HashKey tableSize) {

  int     id = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (auto i = id; i < valCount; i += stride) {
    HashKey hashVal=hashArr[i];
    int pos = atomicAdd((unsigned long long int*)(countArr + hashVal),1)+offsetArr[hashVal];
    // int pos = atomicAdd((countArr + hashVal),1)+offsetArr[hashVal];
#ifdef INDEX_TRACK
    edges[pos]={valsArr[i],i};
#else
    edges[pos] = { valsArr[i] };
#endif
  }    
}

__global__ void copyToGraphD32(index_t valCount, hkey_t *valsArr, HashKey *hashArr, int32_t *countArr,
                                index_t *offsetArr, keyval *edges, index_t tableSize) {

  int     id = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (auto i = id; i < valCount; i += stride) {
    HashKey hashVal=hashArr[i];
    int pos = atomicAdd(countArr + hashVal,1)+offsetArr[hashVal];
    // edges[pos] = valsArr[i];
    // edges[pos]={valsArr[i],i};
#ifdef INDEX_TRACK
    edges[pos]={valsArr[i],i};
#else
    edges[pos] = { valsArr[i] };
#endif
    // edges[pos]={valsArr[i], 0, i};
  }    
}

__global__ void countBinSizes(HashKey *d_vals, index_t size, index_t *d_binSizes, 
                                index_t binRange) {

  int64_t     id = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;

  for (index_t i = id; i < size; i += stride) {
    // index_t bin = d_vals[i] / binRange;
    index_t bin = d_vals[i] / binRange;
    atomicAdd((unsigned long long int*)(&d_binSizes[bin]), 1);
  }
}

__global__ void fillSequence(hkey_t *d_vals, int64_t size) {

  int     id = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (auto i = id; i < size; i += stride) {
    d_vals[i] = i;
  }
}

__global__ void countHashBinSizes(HashKey *d_vals, index_t size, index_t *d_binSizes, 
                                index_t binRange) {

  int64_t     id = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;
  for (auto i = id; i < size; i += stride) {
    index_t bin = d_vals[i] / binRange;
    atomicAdd((unsigned long long int*)(&d_binSizes[bin]), 1);
  }
}

__global__ void countBufferSizes(index_t *hashSplits, index_t size, index_t *bufferCounter, 
                                    index_t gpuCount, HashKey *hashVals) {

  int64_t     id = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;
  for (auto i = id; i < size; i += stride) {
    HashKey hash = hashVals[i];
    // TODO: This might make things slow.
    for (index_t j = 0; j < gpuCount; j++) {
      if (hashSplits[j] <= hash && hash < hashSplits[j + 1]) {
        atomicAdd((unsigned long long int*)(&bufferCounter[j]), 1);
        break;
      }
    }
  }
}

// __global__ void countKeyBuffSizes(HashKey *hashVals, index_t size, index_t *counter, 
//                                       index_t *splits, index_t gpuCount) {
// 
//   int64_t     id = blockIdx.x * blockDim.x + threadIdx.x;
//   int64_t stride = blockDim.x * gridDim.x;
//   for (auto i = id; i < size; i += stride) {
//     HashKey hash = hashVals[i];
//     // TODO: This might make things slow.
//     for (index_t j = 0; j < gpuCount; j++) {
//       if (hash < splits[j + 1]) {
//         atomicAdd((unsigned long long int*)(&counter[j]), 1);
//         break;
//       }
//     }
//   }
// }

__global__ void countKeyBuffSizes(HashKey *hashVals, index_t size_, index_t *counter, 
                                      index_t *splits, index_t gpuCount_) {
  int     id = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  uint32_t gpuCount = gpuCount_;
  int size=size_;
  __shared__ index_t internalCounters[16];
  if(threadIdx.x<gpuCount){
    internalCounters[threadIdx.x]=0;
  }
  __syncthreads();
  for (auto i = id; i < size; i += stride) {
    HashKey hash = hashVals[i];
    // TODO: This might make things slow.
    for (uint32_t j = 0; j < gpuCount; j++) {
      if (hash < splits[j + 1]) {
        // atomicAdd((unsigned long long int*)(&counter[j]), 1);
        atomicAdd((unsigned long long int*)(&internalCounters[j]), 1);
        break;
      }
    }
  }
  __syncthreads();
  if(threadIdx.x<gpuCount){
      atomicAdd((unsigned long long int*)(&counter[threadIdx.x]), internalCounters[threadIdx.x]);
  }
}


__global__ void binKeyValues(index_t size, hkey_t *keys, hkey_t *keyBuff, index_t *offsets, 
                                  index_t *splits, index_t *counter, index_t gpuCount) {

  int64_t     id = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;
  for (auto i = id; i < size; i += stride) {
    hkey_t key = keys[i];
    for (index_t j = 0; j < gpuCount; j++) {
      if (splits[j] <= key && key < splits[j + 1]) {
        index_t pos = atomicAdd((unsigned long long int*)(&counter[j]), 1);
        index_t off = offsets[j];
        keyBuff[off + pos] = key;
        break;
      }
    }
  }
}

// __global__ void binHashValues(index_t size, hkey_t *keys, HashKey *hashes, 
//                                   // hkey_t *keyBuff, index_t *offsets, 
//                                   keyval *keyBuff, index_t *offsets, 
//                                   index_t *hashSplits, index_t *counter, 
//                                   index_t gpuCount, index_t tid) {
// 
//   int64_t     id = blockIdx.x * blockDim.x + threadIdx.x;
//   int64_t stride = blockDim.x * gridDim.x;
//   for (auto i = id; i < size; i += stride) {
//     hkey_t key = keys[i];
//     HashKey hash = hashes[i];
//     for (index_t j = 0; j < gpuCount; j++) {
//       if (hash < hashSplits[j + 1]) {
//         index_t pos = atomicAdd((unsigned long long int*)(&counter[j]), 1);
//         index_t off = offsets[j];
// #ifdef INDEX_TRACK
//         keyBuff[off + pos] = {key, i};
// #else
//         keyBuff[off + pos] = { key };
// #endif
//         // keyBuff[off + pos] = {key, gpuCount, i};
//         // keyBuff[off + pos] = {key, i};
//         break;
//       }
//     }
//   }
//   if (id == 0) {
//     offsets[gpuCount] = size;
//   }
// }
__global__ void binHashValues(index_t size, hkey_t *keys, HashKey *hashes, 
                                  // hkey_t *keyBuff, index_t *offsets, 
                                  keyval *keyBuff, index_t *offsets, 
                                  index_t *hashSplits, index_t *counter, 
                                  index_t gpuCount, index_t tid) {
  int64_t     id = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;
  __shared__ index_t internalCounters[16];
  __shared__ index_t posGPUs[16];
  if(threadIdx.x<gpuCount){
    internalCounters[threadIdx.x]=0;
  }
  __syncthreads();
  for (auto i = id; i < size; i += stride) {
    hkey_t key = keys[i];
    HashKey hash = hashes[i];
    for (index_t j = 0; j < gpuCount; j++) {
      if (hash < hashSplits[j + 1]) {
        index_t pos = atomicAdd((unsigned long long int*)(&internalCounters[j]), 1);
        break;
      }
    }
  }
  __syncthreads();
  if(threadIdx.x<gpuCount){
        posGPUs[threadIdx.x] = atomicAdd((unsigned long long int*)(&counter[threadIdx.x]), internalCounters[threadIdx.x]);
  }
  __syncthreads();
  for (auto i = id; i < size; i += stride) {
    hkey_t key = keys[i];
    HashKey hash = hashes[i];
    index_t pos, off;
    for (index_t j = 0; j < gpuCount; j++) {
      if (hash < hashSplits[j + 1]) {
        pos = atomicAdd((unsigned long long int*)(&posGPUs[j]), 1);
        off = offsets[j];
        break;
      }
    }
#ifdef INDEX_TRACK
    keyBuff[off + pos] = {key, i};
#else
    keyBuff[off + pos] = { key };
#endif
  }
  if (id == 0) {
    offsets[gpuCount] = size;
  }
}

// __global__ void binHashValues(keyhash *buffer, hkey_t *keys, index_t size, index_t *hashes, 
// __global__ void binHashValues(hkey_t *keyBuff, HashKey *hashBuff, hkey_t *keys, index_t size, 
//                                   HashKey *hashes, index_t *hashSplits, index_t *counter, 
//                                   index_t gpuCount, size_t keyPitch, size_t hashPitch) {
// 
//   int64_t     id = blockIdx.x * blockDim.x + threadIdx.x;
//   int64_t stride = blockDim.x * gridDim.x;
//   for (auto i = id; i < size; i += stride) {
//     index_t key = keys[i];
//     index_t hash = hashes[i];
//     // TODO: This might make things slow.
//     for (index_t j = 0; j < gpuCount; j++) {
//       if (hashSplits[j] <= hash && hash < hashSplits[j + 1]) {
//         index_t pos = atomicAdd(&counter[j], 1);
//         hkey_t *keyRow = (hkey_t*)((char*)keyBuff + j * keyPitch);
//         HashKey *hashRow = (HashKey*)((char*)hashBuff + j * hashPitch);
//         keyRow[pos] = key;
//         hashRow[pos] = hash;
//       }
//     }
//   }
// }

// __global__ void decrHash(hkey_t *vals, HashKey *hashes, size_t size, index_t *splits, index_t devNum) {
__global__ void decrHash(HashKey *hashes, size_t size, index_t *splits, index_t devNum) {

  int64_t     id = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;
  for (auto i = id; i < size; i += stride) {
    index_t minHash = splits[devNum];
    // hashes[i].key -= minHash;
    hashes[i] -= minHash;
  }
}

__global__ void binKeyValues(hkey_t *keyBuff, hkey_t *keys, size_t size, index_t *counter, 
                                index_t binRange, size_t pitch) {

  int64_t     id = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;
  for (auto i = id; i < size; i += stride) {
    hkey_t key = keys[i];
    index_t bin = key / binRange;

    hkey_t *row = (hkey_t*)((char*)keyBuff + bin * pitch);
    index_t pos = atomicAdd((unsigned long long int*)(&counter[bin]), 1);
    row[pos] = key;
  }
}

/*** LRB-Specific Kernels ***/
__global__ void lrbCountHashD(index_t valCount, HashKey *hashArr, index_t *d_lrbCounters, 
// __global__ void lrbCountHashD(index_t valCount, keyval *hashArr, index_t *d_lrbCounters, 
                                  index_t lrbBinSize) {
  int     id = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (auto i = id; i < valCount; i += stride) {
      index_t ha = (index_t)(hashArr[i] / lrbBinSize);
      // index_t ha = (index_t)(hashArr[i].key / lrbBinSize);
      atomicAdd((unsigned long long int*)(d_lrbCounters + ha),1);
      // atomicAdd((d_lrbCounters + ha),1);
  }    
}

// __global__ void lrbRehashD(index_t valCount, hkey_t *valsArr, HashKey *hashArr, 
__global__ void lrbRehashD(index_t valCount, keyval *valsArr, HashKey *hashArr, 
                              index_t *d_lrbCounters, keyval *d_lrbHashReordered, 
                              // index_t *d_lrbCounters, hkey_t *d_lrbHashReordered, 
                              index_t *d_lrbCountersPrefix, index_t lrbBinSize,
                              index_t devNum) {

  int     id = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (auto i = id; i < valCount; i += stride) {
      // HashKey ha = hashArr[i].key/lrbBinSize;
      HashKey ha = hashArr[i]/lrbBinSize;
      index_t pos = atomicAdd((unsigned long long int*)(d_lrbCounters + ha),1)+ d_lrbCountersPrefix[ha];
      // index_t pos = atomicAdd((d_lrbCounters + ha),1)+ d_lrbCountersPrefix[ha];
      // d_lrbHashReordered[pos]={valsArr[i],i};
      d_lrbHashReordered[pos] = valsArr[i];
  }
}

__global__ void lrbCountHashGlobalD(index_t valCount, index_t *countArr, 
                                        keyval *d_lrbHashReordered, index_t *splits,
                                        index_t tableSize, index_t devNum) {

  int     id = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (auto i = id; i < valCount; i += stride) {
      HashKey hash = hash_murmur(d_lrbHashReordered[i].key);
      HashKey ha = (hash % tableSize) - splits[devNum];
      atomicAdd((unsigned long long int*)(countArr + ha),1);
      // atomicAdd((countArr + ha),1);
  }    
}

__global__ void lrbCopyToGraphD(index_t valCount, index_t *countArr, index_t *offsetArr, 
                                    keyval *edges, keyval *d_lrbHashReordered, index_t *splits,
                                    index_t tableSize, index_t devNum) {

  int     id = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (auto i = id; i < valCount; i += stride) {
      HashKey hash = hash_murmur(d_lrbHashReordered[i].key);
      HashKey hashVal = (hash % tableSize) - splits[devNum];
      // HashKey hashVal=d_lrbHashReordered[i].key;
      int pos = atomicAdd((unsigned long long int*)(countArr + hashVal),1)+offsetArr[hashVal];
      // int pos = atomicAdd((countArr + hashVal),1)+offsetArr[hashVal];
      edges[pos]=d_lrbHashReordered[i];
  }    
}
__global__ void lrbCountHashD32(index_t valCount, HashKey *hashArr, int32_t *d_lrbCounters, 
                                  index_t lrbBinSize) {
  int     id = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  
  for (auto i = id; i < valCount; i += stride) {
      index_t ha = (index_t)(hashArr[i]/lrbBinSize);
      atomicAdd(d_lrbCounters + ha,1);
  }    
}

__global__ void lrbRehashD32(index_t valCount, hkey_t *valsArr, HashKey *hashArr, 
                              int32_t *d_lrbCounters, keyval *d_lrbHashReordered, 
                              int32_t *d_lrbCountersPrefix, index_t lrbBinSize) {

  int     id = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (auto i = id; i < valCount; i += stride) {
      HashKey ha = hashArr[i]/lrbBinSize;
      // if(blockIdx.x==0)
      //     printf("%d ", ha);
      index_t pos = atomicAdd(d_lrbCounters + ha,1)+ d_lrbCountersPrefix[ha];
      // d_lrbHashReordered[pos]={valsArr[i],i};
      d_lrbHashReordered[pos] = { valsArr[i] };
  }    
}

__global__ void lrbCountHashGlobalD32(index_t valCount, int32_t *countArr, 
                                        keyval *d_lrbHashReordered, index_t tableSize) {

  int     id = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (auto i = id; i < valCount; i += stride) {
      HashKey ha = hash_murmur(d_lrbHashReordered[i].key)%tableSize;
      // HashKey ha = d_lrbHashReordered[i].key;
      atomicAdd(countArr + ha,1);
  }    
}

__global__ void lrbCopyToGraphD32(index_t valCount, int32_t *countArr, index_t *offsetArr, 
                                    keyval *edges, keyval *d_lrbHashReordered, 
                                    index_t tableSize) {

  int     id = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (auto i = id; i < valCount; i += stride) {
      HashKey hashVal = hash_murmur(d_lrbHashReordered[i].key)%tableSize;
      // HashKey hashVal=d_lrbHashReordered[i].key;
      int pos = atomicAdd(countArr + hashVal,1)+offsetArr[hashVal];
      edges[pos]=d_lrbHashReordered[i];
  }    
}

/*** INTERSECTION-specific KERNELS ***/
__global__ void simpleIntersect(index_t valCount, index_t *offsetA, keyval *edgesA,
                                    index_t *offsetB, keyval *edgesB, index_t  *counter,
                                    keypair *pairs, bool countOnly) {

  int     id = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (auto i = id; i < valCount; i += stride) {
    int64_t sizeA = offsetA[i + 1] - offsetA[i];
    int64_t sizeB = offsetB[i + 1] - offsetB[i];
    if (sizeA == 0 || sizeB == 0) {
      continue;
    }

    for (index_t ia = 0; ia < sizeA; ia++) {
      hkey_t aKey = edgesA[offsetA[i] + ia].key;
      for (index_t ib = 0; ib < sizeB; ib++) {
        hkey_t bKey = edgesB[offsetB[i] + ib].key;

        if (aKey == bKey) {
          if (countOnly) {
            counter[i]++;
          } else {
            // pairs[counter[i]++] = {i, edgesB[offsetB[i] + ib].ind};
            // pairs[counter[i]++] = {i, edgesB[offsetB[i] + ib].key};
            pairs[counter[i]++] = { edgesB[offsetB[i] + ib].key };
          }
        }
      }
    }
  }
}

#if 0
__global__ void lrbCountHashD(index_t valCount, HashKey *hashArr, index_t *d_lrbCounters, index_t lrbBinSize) {
  int     id = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (auto i = id; i < valCount; i += stride) {
      index_t ha = index_t(hashArr[i]/lrbBinSize);
      atomicAdd((unsigned long long int*)(d_lrbCounters + ha),1);
  }    
}

__global__ void lrbRehashD(index_t valCount, hkey_t *valsArr, HashKey *hashArr, 
                              index_t *d_lrbCounters, keyval *d_lrbHashReordered, 
                              index_t *d_lrbCountersPrefix, index_t lrbBinSize) {

  int     id = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (auto i = id; i < valCount; i += stride) {
      HashKey ha = hashArr[i]/lrbBinSize;
      // if(blockIdx.x==0)
      //     printf("%d ", ha);
      index_t pos = atomicAdd((unsigned long long int*)(d_lrbCounters + ha),1)+ d_lrbCountersPrefix[ha];
      d_lrbHashReordered[pos]={valsArr[i],i};
  }    
}

__global__ void lrbCountHashGlobalD(index_t valCount, index_t *countArr, keyval *d_lrbHashReordered, index_t tableSize) {

  int     id = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (auto i = id; i < valCount; i += stride) {
      HashKey ha = hash_murmur(d_lrbHashReordered[i].key)%tableSize;
      // HashKey ha = d_lrbHashReordered[i].key;
      atomicAdd((unsigned long long int*)(countArr + ha),1);
  }    
}

__global__ void lrbCopyToGraphD(index_t valCount, index_t *countArr, index_t *offsetArr, keyval *edges, 
                                      keyval *d_lrbHashReordered, index_t tableSize) {

  int     id = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (auto i = id; i < valCount; i += stride) {
      HashKey hashVal = hash_murmur(d_lrbHashReordered[i].key)%tableSize;
      // HashKey hashVal=d_lrbHashReordered[i].key;
      int pos = atomicAdd((unsigned long long int*)(countArr + hashVal),1)+offsetArr[hashVal];
      edges[pos]=d_lrbHashReordered[i];
  }    
}
#endif
