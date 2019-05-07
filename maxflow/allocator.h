// Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#pragma once

#include <stdlib.h>
#include <stdio.h>

static long tot_malloc_bytes = 0;

#ifndef USE_GPU

inline void* my_malloc(long size)
{
#if (LOG_LEVEL > 1) && (LOG_LEVEL < 4)
  tot_malloc_bytes += size;
  printf("Memory allocation footprint %.3f MB\n", ((float) tot_malloc_bytes)/(1<<20));
#endif
  void *ptr = malloc(size);
  return ptr;
}

inline void my_free(void *ptr)
{
  free(ptr);
}

#else

#include <cuda_runtime.h>

inline void* my_malloc(int size)
{
#if (LOG_LEVEL > 1) && (LOG_LEVEL < 4)
  tot_malloc_bytes += size;
  printf("Unified memory allocation footprint %.3f MB\n", ((float) tot_malloc_bytes)/(1<<20));
#endif
  void *ptr;
  cudaMallocManaged(&ptr, size);
  return ptr;
}

inline void my_free(void *ptr)
{
  cudaFree(ptr);
}

#endif

