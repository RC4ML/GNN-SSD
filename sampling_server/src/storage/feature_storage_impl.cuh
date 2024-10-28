
#ifndef FEATURE_STORAGE_IMPL_H_
#define FEATURE_STORAGE_IMPL_H_

#include <cstdint>
#include <string>
#include <vector>

#include "iostack.cuh"
#include "userqueue.cuh"

// Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError()                                       \
  {                                                            \
    cudaError_t e = cudaGetLastError();                        \
    if (e != cudaSuccess) {                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                           \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  }
  

#endif