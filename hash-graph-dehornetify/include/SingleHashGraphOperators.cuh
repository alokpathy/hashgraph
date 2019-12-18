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

__global__ void hashValuesD(index_t valCount, hkey_t *valsArr, HashKey *hashArr, HashKey tableSize) {
  int     id = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (auto i = id; i < valCount; i += stride) {
    // hashArr[i] = hash_murmur(valsArr[i]) % tableSize;
    hashArr[i] = hash_murmur(valsArr[i]) % tableSize;
    // hashArr[i] = (HashKey)(hash_murmur(valsArr[i].key) % tableSize);
  }    
}

__global__ void countHashD(index_t valCount, HashKey *hashArr, index_t *countArr) {
  int     id = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (auto i = id; i < valCount; i += stride) {
    atomicAdd(countArr + hashArr[i], 1);
  }    
}

__global__ void copyToGraphD(index_t valCount, hkey_t *valsArr, HashKey *hashArr, index_t *countArr,
                                index_t *offsetArr, keyval *edges, index_t tableSize) {

  int     id = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (auto i = id; i < valCount; i += stride) {
    HashKey hashVal=hashArr[i];
    int pos = atomicAdd(countArr + hashVal,1)+offsetArr[hashVal];
    // edges[pos]={valsArr[i],i};
    edges[pos]={valsArr[i]};
  }    
}

__global__ void lrbCountHashD(index_t valCount, HashKey *hashArr, int32_t *d_lrbCounters, 
                                  index_t lrbBinSize) {
  int     id = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  
  for (auto i = id; i < valCount; i += stride) {
      index_t ha = (index_t)(hashArr[i]/lrbBinSize);
      atomicAdd(d_lrbCounters + ha,1);
  }    
}

__global__ void lrbRehashD(index_t valCount, hkey_t *valsArr, HashKey *hashArr, 
// __global__ void lrbRehashD(index_t valCount, keyval *valsArr, HashKey *hashArr, 
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

__global__ void lrbCountHashGlobalD(index_t valCount, int32_t *countArr, 
                                        keyval *d_lrbHashReordered, index_t tableSize) {

  int     id = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (auto i = id; i < valCount; i += stride) {
      HashKey ha = hash_murmur(d_lrbHashReordered[i].key)%tableSize;
      // HashKey ha = d_lrbHashReordered[i].key;
      atomicAdd(countArr + ha,1);
  }    
}

__global__ void lrbCopyToGraphD(index_t valCount, int32_t *countArr, index_t *offsetArr, 
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
