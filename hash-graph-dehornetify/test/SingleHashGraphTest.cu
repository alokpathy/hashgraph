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
#include "SingleHashGraph.cuh"

int main(int argc, char **argv) {

  int64_t countSize = 1L << 24;
  int64_t maxkey = 1L << 26;
  int64_t tableSize = maxkey;

  if (argc >= 2 && argc < 3) {
    std::cerr << "Please specify all arguments.\n";
    return 1;
  }

  if (argc >= 2) {
    uint32_t sizeExp = atoi(argv[1]);
    countSize = 1L << sizeExp;

    uint32_t keyExp = atoi(argv[2]);
    maxkey = 1L << keyExp;
  } 
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float buildTime = 0.0f; // milliseoncds

  rmm_mgpu_context_t context;

  std::cout << "before1" << std::endl;
  SingleHashGraph shg(countSize, maxkey, context, tableSize); 
  std::cout << "after1" << std::endl;

  cudaEventRecord(start);

  shg.build(countSize, context, tableSize);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&buildTime, start, stop);

  std::cout << "single buildTable() time: " << (buildTime / 1000.0) << "\n"; // seconds
}
