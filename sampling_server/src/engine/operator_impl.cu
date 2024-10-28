#include "operator_impl.cuh"
#include "device_launch_parameters.h"
#include "system_config.cuh"

#include <cuda_runtime.h>
#include <iostream>
#include <random>

#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/linear_congruential_engine.h>

#define OP_THREAD_NUM 1024
#define SH_MEM_SIZE 1024

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

//assume no duplicate
__global__ void batch_generate(
	int32_t* batch_ids, 
	int32_t* labels, 
	int32_t batch_size, 
	int32_t counter, 
	int32_t* all_ids, 
	int32_t* all_labels, 
	int32_t total_cap,
	int32_t* position_map,
	uint32_t* accessed_map)
{
	int32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
	if(idx < batch_size){
		if((batch_size * counter + idx) >= total_cap){
			batch_ids[idx] = -1;
			labels[idx] = -1;
		}else{
			int32_t src_id = all_ids[(batch_size * counter + idx)%(total_cap)];
			batch_ids[idx] = src_id;
			// accessed_map[src_id] = 1;
			int32_t bitmap_idx = src_id / 32;
			int32_t bitmap_off = src_id % 32;
			uint32_t bitmap_data = (1 << bitmap_off);
			atomicOr(accessed_map + bitmap_idx, bitmap_data);
			position_map[src_id] = idx;
			labels[idx] = all_labels[(batch_size * counter + idx)%(total_cap)];
		}	
	}
}

__global__ void counter_update(
	int32_t* node_counter, 			// for the inputs of feature extraction and graph sampling
	int32_t* edge_counter, 			// for the inputs of graph sampling
	int32_t op_id, 
	int32_t size,
	int32_t hop_num)
{
	if(op_id == 0){
		node_counter[0] = 0;										//start offset for feature extraction
		node_counter[1] = size;										//sampled nodes num for feature extraction
		node_counter[INTRABATCH_CON * 3 + (op_id / INTRABATCH_CON)] = node_counter[0] + node_counter[1];
		node_counter[INTRABATCH_CON * 3 - 1] = hop_num;
	}else if((op_id > 0) && (op_id % INTRABATCH_CON == 0)){
		node_counter[0] = node_counter[0] + node_counter[1];		
		node_counter[1] = node_counter[INTRABATCH_CON * 2];			
		node_counter[INTRABATCH_CON * 2] = 0;						//reset
		node_counter[INTRABATCH_CON * 2 + 1] = node_counter[0] + node_counter[1];					//total
		//hop1 start (0 C0), hop2 start (C0 C1), hop3 start (C0+C1 C2)
		//hop1 done (C0 C1), hop2 done (C0+C1 C2), hop3 done (C0+C1+C2 C3)
		edge_counter[0] = edge_counter[0] + edge_counter[1];		//input offset for graph sampling
		edge_counter[1] = edge_counter[2];							//input vertex size of graph sampling
		edge_counter[2] = 0;										//reset
		//hop1 start (0 0 0), hop2 start (0 B1 0), hop3 start (B1 B2 0)
		//hop1 done (0 B1 0), hop2 done (B1 B2 0), hop3 done (B1+B2 B3 0)
		node_counter[INTRABATCH_CON * 3 + (op_id / INTRABATCH_CON)] = node_counter[0] + node_counter[1];
		edge_counter[INTRABATCH_CON * 3 + (op_id / INTRABATCH_CON)] = edge_counter[0] + edge_counter[1];	
	}else if(op_id % INTRABATCH_CON > 0){
		node_counter[(op_id % INTRABATCH_CON) * 2] = node_counter[0];										
		node_counter[(op_id % INTRABATCH_CON) * 2 + 1] = node_counter[1];	
	}else{
		printf("Sampling Parameters Error\n");
	}
}

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
	bool 			is_presc,
	int32_t 		hop_num)
{
	int32_t* all_ids 	= nullptr;
	int32_t* all_labels = nullptr;
	int32_t total_cap 	= 0;
	if(mode == TRAINMODE){
		all_ids 	= feature->GetTrainingSetIds(dev_id);
		all_labels 	= feature->GetTrainingLabels(dev_id);
		total_cap 	= feature->TrainingSetSize(dev_id);
	}else if(mode == VALIDMODE){
		all_ids 	= feature->GetValidationSetIds(dev_id);
		all_labels 	= feature->GetValidationLabels(dev_id);
		total_cap 	= feature->ValidationSetSize(dev_id);
	}else if(mode == TESTMODE){
		all_ids 	= feature->GetTestingSetIds(dev_id);
		all_labels 	= feature->GetTestingLabels(dev_id);
		total_cap 	= feature->TestingSetSize(dev_id);
	}else{
		std::cout<<"invalid mode: "<<mode<<"\n";
	}

	int32_t total_node_num 		= feature->TotalNodeNum();

	int32_t* batch_ids 			= memorypool->GetSampledIds();
	int32_t* labels 			= memorypool->GetLabels();
	uint32_t* accessed_map 		= memorypool->GetAccessedMap();
	int32_t* position_map 		= memorypool->GetPositionMap();
	int32_t* node_counter 		= memorypool->GetNodeCounter();
	int32_t* edge_counter 		= memorypool->GetEdgeCounter();
	int32_t* agg_src_ids 		= memorypool->GetAggSrcId();
	int32_t* agg_dst_ids 		= memorypool->GetAggDstId();
	int32_t* agg_src_off 		= memorypool->GetAggSrcOf();
	int32_t* agg_dst_off 		= memorypool->GetAggDstOf();
	int32_t* cache_index 		= memorypool->GetCacheSearchBuffer();
	int32_t* sampled_ids 		= memorypool->GetSampledIds();
	float* dst_float_buffer 	= memorypool->GetFloatFeatures();
	int32_t op_id				= 0;

	if(all_ids == nullptr){
		std::cout<<"invalid src id ptr\n";
		return;
	}
	if(all_labels == nullptr){
		std::cout<<"invalid label ptr\n";
		return;
	}
	cudaCheckError();

	cudaMemsetAsync(accessed_map, 0, int64_t(int64_t((total_node_num / 32) + 1) * int64_t(sizeof(uint32_t))), (strm_hdl));
	//cudaMemsetAsync(position_map, 0, int64_t(int64_t(total_node_num) * int64_t(sizeof(int32_t))), (strm_hdl));
	cudaCheckError();
	// cache->Finalize(dev_id);
	cudaMemsetAsync(node_counter, 0, 16 * sizeof(int32_t), (strm_hdl));
	cudaMemsetAsync(edge_counter, 0, 16 * sizeof(int32_t), (strm_hdl));
	cudaCheckError();

	int32_t size = ((batch_size*(counter+1)) >= total_cap) ? (total_cap - batch_size * counter) : batch_size;
	dim3 bg_block((size - 1)/OP_THREAD_NUM + 1, 1);
	dim3 bg_thread(OP_THREAD_NUM, 1);
	batch_generate<<<bg_block, bg_thread, 0, (strm_hdl)>>>(batch_ids, labels, size, counter, all_ids, all_labels, total_cap, position_map, accessed_map);
	cudaCheckError();

	counter_update<<<1, 1, 0, (strm_hdl)>>>(node_counter, edge_counter, 0, size, hop_num);
	cudaCheckError();
	if(!is_presc){
		cache->FindFeat(sampled_ids, cache_index, node_counter, op_id, strm_hdl, dev_id);
		cudaCheckError();
	}

}

/////////random sampler//////////
__global__ void random_sample(
	int32_t*  sampled_ids,
	int32_t   op_id,
	int64_t** csr_node_index, 
	int32_t** csr_dst_node_ids,
	char*     partition_index,
	int32_t*  parition_offset,
	int32_t   count, 
	int32_t   partition_count, 
	int32_t*  agg_src_ids,
	int32_t*  agg_dst_ids,
	uint32_t*  accessed_map,
	int32_t*  position_map,
	int32_t*  node_counter,
	int32_t*  edge_counter,
	int32_t   dev_id
	)
{	
	/*the direction for agg is reversed*/
	__shared__ int32_t sh_agg_src_ids[SH_MEM_SIZE];
	__shared__ int32_t sh_agg_dst_ids[SH_MEM_SIZE];
	__shared__ int32_t sh_sampled_id[SH_MEM_SIZE];
	/*local offset: 0, local node offset; 1, global node offset; 2, local edge offset; 3, global edge offset */
	__shared__ int32_t local_offset[4];
	int32_t* input_ids = nullptr;
	int32_t batch_size = 0;
	if(op_id == INTRABATCH_CON){
		input_ids = sampled_ids;
		batch_size = node_counter[1];
	}else if(op_id > INTRABATCH_CON){
		input_ids = agg_src_ids + edge_counter[0];
		batch_size = edge_counter[1];
	}
	for(int32_t idx = threadIdx.x + blockDim.x * blockIdx.x; idx < batch_size * count; idx += gridDim.x * blockDim.x){
		if(threadIdx.x == 0){
            local_offset[0] = 0;
			local_offset[1] = 0;
            local_offset[2] = 0;
            local_offset[3] = 0;
        }
        __syncthreads();
		int32_t sample_src_id = input_ids[idx/count];
		int32_t sample_dst_id;
		if(sample_src_id >= 0){
			int32_t neighbor_offset = idx%count;
			int32_t part_id = partition_index[idx/count];
			int32_t part_offset = parition_offset[idx/count];
			int64_t start_index;
			int32_t col_size;
			if(part_id < 0){
				start_index = csr_node_index[partition_count][sample_src_id];
				col_size = csr_node_index[partition_count][(sample_src_id + 1)] - start_index;
			}else{
				start_index = csr_node_index[part_id][part_offset];
				col_size = csr_node_index[part_id][(part_offset + 1)] - start_index;
			}

			if(neighbor_offset >= col_size){
				sample_dst_id = -1;
			}else{
				thrust::minstd_rand engine;
				engine.discard(idx);
				thrust::uniform_int_distribution<> dist(0, col_size - 1);
				int32_t dst_index = dist(engine);
				if(part_id < 0){
					sample_dst_id = csr_dst_node_ids[partition_count][(int64_t(start_index + int64_t(dst_index)))];
				}else{
					sample_dst_id = csr_dst_node_ids[part_id][(int64_t(start_index + int64_t(dst_index)))];
				}
				if(sample_dst_id >= 0){
					int32_t bitmap_idx = sample_dst_id / 32;
					int32_t bitmap_off = sample_dst_id % 32;
					uint32_t bitmap_data = (1 << bitmap_off);
					uint32_t old_bitmap_data = atomicOr(accessed_map + bitmap_idx, bitmap_data);
					uint32_t is_accessed = (old_bitmap_data >> bitmap_off) % 2;
					// int32_t acc_count = atomicAdd(accessed_map + sample_dst_id, 1);
					if(is_accessed == 0){//first time node is sampled
						int32_t node_off = atomicAdd(local_offset, 1);
						sh_sampled_id[node_off] = sample_dst_id;
					}
					int32_t edge_off = atomicAdd(local_offset + 2, 1);
					sh_agg_src_ids[edge_off] = sample_dst_id;
					sh_agg_dst_ids[edge_off] = sample_src_id;
				}
			}
		}
		__syncthreads();
		if(threadIdx.x == 0){
            local_offset[1] = atomicAdd(node_counter + INTRABATCH_CON * 2, local_offset[0]);//global node count current hop
			local_offset[3] = atomicAdd(edge_counter + 2, local_offset[2]);//global edge count current hop
		}
        __syncthreads();
		if(threadIdx.x < local_offset[0]){
			int32_t node_base = local_offset[1] + node_counter[0] + node_counter[1];
			int32_t dst_id = sh_sampled_id[threadIdx.x]; 
			sampled_ids[node_base + threadIdx.x] = dst_id;
			position_map[dst_id] = node_base + threadIdx.x;
		}
		__syncthreads();
		if(threadIdx.x < local_offset[2]){
			int32_t edge_base = local_offset[3] + edge_counter[0] + edge_counter[1];
			agg_src_ids[edge_base + threadIdx.x] = sh_agg_src_ids[threadIdx.x];
			agg_dst_ids[edge_base + threadIdx.x] = sh_agg_dst_ids[threadIdx.x]; 
		}
		__syncthreads();
	}
}

__global__ void construct_graph(int32_t* agg_src_ids, int32_t* agg_dst_ids, 
								int32_t* agg_src_off, int32_t* agg_dst_off,
								int32_t* position_map, int32_t* edge_counter, int32_t* node_counter, int32_t op_id, int32_t dev_id){
	int32_t edge_num = edge_counter[2];
	int32_t edge_off = edge_counter[0] + edge_counter[1];
	for(int32_t idx = threadIdx.x + blockDim.x * blockIdx.x; idx < edge_num; idx += gridDim.x * blockDim.x){
		int32_t src_id = agg_src_ids[edge_off + idx];
		int32_t dst_id = agg_dst_ids[edge_off + idx];
		int32_t src_of = position_map[src_id];
		int32_t dst_of = position_map[dst_id];
		agg_src_off[edge_off + idx] = src_of;
		agg_dst_off[edge_off + idx] = dst_of;
	}
}



/////////random sampler//////////
__global__ void pre_sample(
	int32_t*  sampled_ids,
	int32_t   op_id,
	int64_t* csr_node_index, 
	int32_t* csr_dst_node_ids,
	char*     partition_index,
	int32_t*  parition_offset,
	int32_t   count, 
	int32_t   partition_count, 
	int32_t*  agg_src_ids,
	int32_t*  agg_dst_ids,
	uint32_t*  accessed_map,
	int32_t*  position_map,
	int32_t*  node_counter,
	int32_t*  edge_counter,
	int32_t   dev_id,
	unsigned long long int*  edge_access_time
	)
{	
	/*the direction for agg is reversed*/
	__shared__ int32_t sh_agg_src_ids[SH_MEM_SIZE];
	__shared__ int32_t sh_agg_dst_ids[SH_MEM_SIZE];
	__shared__ int32_t sh_sampled_id[SH_MEM_SIZE];
	/*local offset: 0, local node offset; 1, global node offset; 2, local edge offset; 3, global edge offset */
	__shared__ int32_t local_offset[4];
	int32_t* input_ids = nullptr;
	int32_t batch_size = 0;
	if(op_id == INTRABATCH_CON){
		input_ids = sampled_ids;
		batch_size = node_counter[1];
	}else if(op_id > INTRABATCH_CON){
		input_ids = agg_src_ids + edge_counter[0];
		batch_size = edge_counter[1];
	}
	for(int32_t idx = threadIdx.x + blockDim.x * blockIdx.x; idx < batch_size * count; idx += gridDim.x * blockDim.x){
		if(threadIdx.x == 0){
            local_offset[0] = 0;
			local_offset[1] = 0;
            local_offset[2] = 0;
            local_offset[3] = 0;
        }
        __syncthreads();
		int32_t sample_src_id = input_ids[idx/count];
		int32_t sample_dst_id;
		if(sample_src_id >= 0){
			int32_t neighbor_offset = idx%count;
			int64_t start_index = csr_node_index[sample_src_id];
			int32_t col_size = csr_node_index[(sample_src_id + 1)] - start_index;
			if(neighbor_offset >= col_size){
				sample_dst_id = -1;
			}else{
				thrust::minstd_rand engine;
				engine.discard(idx);
				thrust::uniform_int_distribution<> dist(0, col_size - 1);
				int32_t dst_index = dist(engine);
				sample_dst_id = csr_dst_node_ids[(int64_t(start_index + int64_t(dst_index)))];
				if(sample_dst_id >= 0){
					atomicAdd(edge_access_time + sample_src_id, 1);

					int32_t bitmap_idx = sample_dst_id / 32;
					int32_t bitmap_off = sample_dst_id % 32;
					uint32_t bitmap_data = (1 << bitmap_off);
					uint32_t old_bitmap_data = atomicOr(accessed_map + bitmap_idx, bitmap_data);
					uint32_t is_accessed = (old_bitmap_data >> bitmap_off) % 2;
					// int32_t acc_count = atomicAdd(accessed_map + sample_dst_id, 1);
					// int32_t acc_count = atomicAdd(accessed_map + sample_dst_id, 1);
					if(is_accessed == 0){//first time node is sampled
						int32_t node_off = atomicAdd(local_offset, 1);
						sh_sampled_id[node_off] = sample_dst_id;
					}
					int32_t edge_off = atomicAdd(local_offset + 2, 1);
					sh_agg_src_ids[edge_off] = sample_dst_id;
					sh_agg_dst_ids[edge_off] = sample_src_id;
				}
			}
		}
		__syncthreads();
		if(threadIdx.x == 0){
            local_offset[1] = atomicAdd(node_counter + INTRABATCH_CON * 2, local_offset[0]);//global node count current hop
			local_offset[3] = atomicAdd(edge_counter + 2, local_offset[2]);//global edge count current hop
		}
        __syncthreads();
		if(threadIdx.x < local_offset[0]){
			int32_t node_base = local_offset[1] + node_counter[0] + node_counter[1];
			int32_t dst_id = sh_sampled_id[threadIdx.x]; 
			sampled_ids[node_base + threadIdx.x] = dst_id;
			position_map[dst_id] = node_base + threadIdx.x;
		}
		__syncthreads();
		if(threadIdx.x < local_offset[2]){
			int32_t edge_base = local_offset[3] + edge_counter[0] + edge_counter[1];
			agg_src_ids[edge_base + threadIdx.x] = sh_agg_src_ids[threadIdx.x];
			agg_dst_ids[edge_base + threadIdx.x] = sh_agg_dst_ids[threadIdx.x]; 
		}
		__syncthreads();
	}
}


extern "C"											
void RandomSample(
  cudaStream_t    strm_hdl, 
  GraphStorage*   graph,
  UnifiedCache*   cache,
  MemoryPool*     memorypool,
  int32_t         count,
  int32_t         dev_id,
  int32_t         op_id,
  bool            is_presc) 
{		

	if(graph == nullptr){
		std::cout<<"invalid storage ptr\n";
		return;
	}

	int32_t** csr_dst_node_ids 		= graph -> GetCSRNodeMatrix(dev_id);
	int64_t** csr_node_index  		= graph -> GetCSRNodeIndex(dev_id);
	int32_t partition_count 		= graph -> GetPartitionCount();
	char* partition_index 			= graph -> PartitionIndex(dev_id);
	int32_t* parition_offset 		= graph -> PartitionOffset(dev_id);

	uint32_t* accessed_map 			= memorypool->GetAccessedMap();
	int32_t* position_map 			= memorypool->GetPositionMap();
	int32_t* node_counter 			= memorypool->GetNodeCounter();
	int32_t* edge_counter 			= memorypool->GetEdgeCounter();
	int32_t* sampled_ids 			= memorypool->GetSampledIds();
	int32_t* agg_src_ids 			= memorypool->GetAggSrcId();
	int32_t* agg_dst_ids 			= memorypool->GetAggDstId();
	int32_t* agg_src_off 			= memorypool->GetAggSrcOf();
	int32_t* agg_dst_off 			= memorypool->GetAggDstOf();
	char* tmp_partition_index  		= memorypool->GetTmpPartIdx();
	int32_t* tmp_parition_offset	= memorypool->GetTmpPartOff();
	int32_t* cache_index 			= memorypool->GetCacheSearchBuffer();

    dim3 block_num(16, 1);
    dim3 thread_num(OP_THREAD_NUM, 1);
	if(!is_presc){
		int32_t* h_node_counter = (int32_t*)malloc(16*sizeof(int32_t));
		cudaMemcpy(h_node_counter, node_counter, 64, cudaMemcpyDeviceToHost);
		cudaCheckError();

		int32_t* h_edge_counter = (int32_t*)malloc(16*sizeof(int32_t));
		cudaMemcpy(h_edge_counter, edge_counter, 64, cudaMemcpyDeviceToHost);
		cudaCheckError();

		int32_t* input_ids = nullptr;
		int32_t batch_size = 0;
		if(op_id == INTRABATCH_CON){
			input_ids = sampled_ids;
			batch_size = h_node_counter[1];
		}else if(op_id > INTRABATCH_CON){
			input_ids = agg_src_ids + h_edge_counter[0];
			batch_size = h_edge_counter[1];
		}
		cache -> FindTopo(input_ids, tmp_partition_index, tmp_parition_offset, batch_size, op_id, strm_hdl, dev_id);
		cudaCheckError();
		free(h_node_counter);
		free(h_edge_counter);
		random_sample<<<block_num, thread_num, 0, (strm_hdl)>>>(sampled_ids, op_id, csr_node_index, csr_dst_node_ids, 
																											tmp_partition_index, tmp_parition_offset,
																											count, partition_count,
																											agg_src_ids, agg_dst_ids,
																											accessed_map,
																											position_map,
																											node_counter,
																											edge_counter, 
																											dev_id);	
		cudaCheckError();
	}else{
		int32_t* pin_csr_dst_node_ids = graph -> GetCSRNodeMatrixCPU();
		int64_t* pin_csr_node_index  = graph -> GetCSRNodeIndexCPU();
		unsigned long long int* edge_access_time = cache->GetEdgeAccessedMap(dev_id);
		pre_sample<<<block_num, thread_num, 0, (strm_hdl)>>>(sampled_ids, op_id, pin_csr_node_index, pin_csr_dst_node_ids, 
																											partition_index, parition_offset,
																											count, partition_count,
																											agg_src_ids, agg_dst_ids,
																											accessed_map,
																											position_map,
																											node_counter,
																											edge_counter, 
																											dev_id,
																											edge_access_time);	
	}
	
	cudaCheckError();
	construct_graph<<<block_num, thread_num, 0, (strm_hdl)>>>(agg_src_ids, agg_dst_ids, agg_src_off, agg_dst_off,
															  position_map, edge_counter, node_counter, op_id, dev_id);
	cudaCheckError();

	counter_update<<<1, 1, 0, (strm_hdl)>>>(node_counter, edge_counter, op_id, 0, 0);		
	cudaCheckError();	

	if(!is_presc){
		cache->FindFeat(sampled_ids, cache_index, node_counter, op_id, strm_hdl, dev_id);
		cudaCheckError();
	}

}

extern "C"
void FeatureCacheLookup(
  cudaStream_t    strm_hdl,
  UnifiedCache*   cache, 
  MemoryPool*     memorypool,
  int32_t         op_id,
  int32_t         dev_id)
{	
	int32_t* sampled_ids 		= memorypool->GetSampledIds();
	int32_t* cache_index 		= memorypool->GetCacheSearchBuffer();
	float* dst_float_buffer 	= memorypool->GetFloatFeatures();
	int32_t* node_counter 		= memorypool->GetNodeCounter();
	int32_t* edge_counter 		= memorypool->GetEdgeCounter();

	counter_update<<<1, 1, 0, (strm_hdl)>>>(node_counter, edge_counter, op_id, 0, 0);		
	cudaCheckError();
	cache->FeatCacheLookup(sampled_ids, cache_index, node_counter, dst_float_buffer, op_id, dev_id, strm_hdl);
	cudaCheckError();
}	

extern "C"
void IOSubmit(
	cudaStream_t    strm_hdl, 
	FeatureStorage* feature,
  	MemoryPool*     memorypool,
	int32_t			op_id,
	int32_t         dev_id)
{
	int32_t* sampled_ids 		= memorypool->GetSampledIds();
	int32_t* cache_index 		= memorypool->GetCacheSearchBuffer();
	float* dst_float_buffer 	= memorypool->GetFloatFeatures();
	int32_t* node_counter 		= memorypool->GetNodeCounter();
	int32_t* edge_counter 		= memorypool->GetEdgeCounter();

	counter_update<<<1, 1, 0, (strm_hdl)>>>(node_counter, edge_counter, op_id, 0, 0);		
	cudaCheckError();
	feature->IOSubmit(sampled_ids, cache_index, node_counter, dst_float_buffer, op_id, dev_id, strm_hdl);
	cudaCheckError();
}


__global__ void ClearPosMap(int32_t* position_map, int32_t* sampled_ids, int32_t* node_counter){
	int32_t batch_size = node_counter[INTRABATCH_CON * 2 + 1];
	for(int32_t idx = threadIdx.x + blockDim.x * blockIdx.x; idx < batch_size; idx += gridDim.x * blockDim.x){
		int32_t sampled_id = sampled_ids[idx];
		position_map[sampled_id] = 0;
	}
}

extern "C"
void IOComplete(
  cudaStream_t    strm_hdl, 
  FeatureStorage* feature,
  UnifiedCache*   cache, 
  MemoryPool*     memorypool,
  int32_t         dev_id,
  int32_t         mode)
{
	if(mode == TRAINMODE){
		int32_t* agg_dst_id = memorypool->GetAggDstId();
		int32_t* agg_src_id = memorypool->GetAggSrcId();
		int32_t* node_counter = memorypool->GetNodeCounter();
		int32_t* edge_counter = memorypool->GetEdgeCounter();
		int32_t* sampled_ids = memorypool->GetSampledIds();
		int32_t* agg_src_off = memorypool->GetAggSrcOf();
		int32_t* agg_dst_off = memorypool->GetAggDstOf();
		int32_t* position_map = memorypool->GetPositionMap();

		feature->IOComplete(op_id, dev_id, strm_hdl);
		cudaCheckError();
		
		dim3 block_num(32, 1);
		dim3 thread_num(OP_THREAD_NUM, 1);
		ClearPosMap<<<block_num, thread_num, 0, strm_hdl>>>(position_map, sampled_ids, node_counter);
		cache -> CacheProfiling(sampled_ids, agg_src_id, agg_dst_id, agg_src_off, agg_dst_off, node_counter, edge_counter, strm_hdl, dev_id);
	}
}
