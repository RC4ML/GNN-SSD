#ifndef OPERATOR_IMPL_H
#define OPERATOR_IMPL_H
#include <time.h>
#include <iostream>

#include "memorypool.cuh"
#include "graph_storage.cuh"
#include "feature_storage.cuh"
#include "cache.cuh"

extern "C"
void BatchGenerate(
	cudaStream_t    strm_hdl, 
	FeatureStorage* feature,
	UnifiedCache*   cache,
	MemoryPool*     memorypool,
	int32_t         batch_size, 
	int32_t         counter, 
  int32_t         part_id,
  int32_t         dev_id,
  int32_t         mode,
  bool            is_presc,
  int32_t 		    hop_num
);

extern "C"											
void RandomSample(
  cudaStream_t    strm_hdl, 
  GraphStorage*   graph,
  UnifiedCache*   cache,
  MemoryPool*     memorypool,
  int32_t         count,
  int32_t         dev_id,
  int32_t         op_id,
  bool            is_presc
);

extern "C"
void FeatureCacheLookup(
  cudaStream_t    strm_hdl,
  UnifiedCache*   cache, 
  MemoryPool*     memorypool,
  int32_t         op_id,
  int32_t         dev_id
);

extern "C"
void IOSubmit(
	cudaStream_t    strm_hdl, 
	FeatureStorage* feature,
  MemoryPool*     memorypool,
	int32_t			    op_id,
	int32_t         dev_id
);

extern "C"
void IOComplete(
  cudaStream_t    strm_hdl, 
  FeatureStorage* feature,
  UnifiedCache*   cache, 
  MemoryPool*     memorypool,
  int32_t         dev_id,
  int32_t         mode
);

#endif