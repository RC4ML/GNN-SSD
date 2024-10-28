#include <vector>
#include "graph_storage.cuh"
#include "feature_storage.cuh"
#include "cache.cuh"
#include "ipc_service.h"

class StorageManagement {
public:
  
  void Initialze(int32_t shard_count);

  GraphStorage* GetGraph();

  FeatureStorage* GetFeature();

  UnifiedCache* GetCache(); 

  IPCEnv* GetIPCEnv();

  int32_t Shard_To_Device(int32_t part_id);

  int32_t Shard_To_Partition(int32_t part_id);

  int32_t Central_Device();

private:
  void EnableP2PAccess();

  void ConfigPartition(BuildInfo* info, int32_t shard_count);

  void ReadMetaFIle(BuildInfo* info);

  void LoadGraph(BuildInfo* info);

  void LoadFeature(BuildInfo* info);
  
  int32_t central_device_;
  std::vector<int> shard_to_device_;
  std::vector<int> shard_to_partition_;
  int32_t partition_;

  int64_t cache_edge_num_;
  int64_t edge_num_;
  int32_t node_num_;

  int32_t training_set_num_;
  int32_t validation_set_num_;
  int32_t testing_set_num_;

  int32_t float_feature_len_;

  int64_t cache_memory_;

  std::string dataset_path_;
  int32_t raw_batch_size_;
  int32_t epoch_;
  int32_t num_ssd_;
  int32_t num_queues_per_ssd_;
  int32_t cpu_cache_capacity_;//for legion ssd
  int32_t gpu_cache_capacity_;//for legion ssd
  
  GraphStorage* graph_;
  FeatureStorage* feature_;
  UnifiedCache* cache_;
  IPCEnv* env_;
};


