#include "operator.h"
#include "operator_impl.cuh"
#include "storage_management.cuh"
#include "graph_storage.cuh"
#include "feature_storage.cuh"
#include "ipc_service.h"
#include "cache.cuh"
#include "memorypool.cuh"

class BatchGenerateOP : public Operator {
public:
    BatchGenerateOP(int op_id){
        op_id_ = op_id;
    }
    void run(OpParams* params) override {
        FeatureStorage* feature     = (FeatureStorage*)(params->feature);
        UnifiedCache* cache         = (UnifiedCache*)(params->cache);
        MemoryPool* memorypool      = (MemoryPool*)(params->memorypool);
        IPCEnv* env                 = (IPCEnv*)(params->env);
        int32_t device_id           = params->device_id;
        int32_t mode                = memorypool->GetCurrentMode();
        int32_t iter                = memorypool->GetIter();
        int32_t batch_size          = env->GetCurrentBatchsize(device_id, mode);
        bool is_presc               = params->is_presc;
        int32_t hop_num             = params->hop_num;

        BatchGenerate(params->stream, feature, cache, memorypool, batch_size, iter, device_id, device_id, mode, is_presc, hop_num);
        cudaEventRecord(((params->event)), ((params->stream)));
        cudaCheckError();
    }
private:
    int op_id_;
};

Operator* NewBatchGenerateOP(int op_id){
    return new BatchGenerateOP(op_id);
}

class RandomSampleOP : public Operator {
public:
    RandomSampleOP(int op_id){
        op_id_ = op_id;
    }
    void run(OpParams* params) override {
        MemoryPool* memorypool      = (MemoryPool*)(params->memorypool);
        GraphStorage* graph         = (GraphStorage*)(params->graph);
        UnifiedCache* cache         = (UnifiedCache*)(params->cache);
        bool is_presc               = params->is_presc;
        int32_t count               = params->neighbor_count;
        int32_t device_id           = params->device_id;

        RandomSample(params->stream, graph, cache, memorypool, count, device_id, op_id_, is_presc);
        cudaEventRecord(((params->event)), ((params->stream)));
        cudaCheckError();
    }
private:
    int op_id_;
};

Operator* NewRandomSampleOP(int op_id){
    return new RandomSampleOP(op_id);
}

class CacheLookupOP : public Operator {
public:
    CacheLookupOP(int op_id){
        op_id_ = op_id;
    }
    void run(OpParams* params) override {
        UnifiedCache* cache         = (UnifiedCache*)(params->cache);
        MemoryPool* memorypool      = (MemoryPool*)(params->memorypool);
        int32_t device_id           = params->device_id;

        FeatureCacheLookup(params->stream, cache, memorypool, op_id_, device_id);
        cudaEventRecord(((params->event)), ((params->stream)));
        cudaCheckError();
    }
private:
    int op_id_;
};

Operator* NewCacheLookupOP(int op_id){
    return new CacheLookupOP(op_id);
}

class SSDIOSubmitOP : public Operator {
public:
    SSDIOSubmitOP(int op_id) {
        op_id_ = op_id;
    }
    void run(OpParams* params) override {
        FeatureStorage* feature     = (FeatureStorage*)(params->feature);
        MemoryPool* memorypool      = (MemoryPool*)(params->memorypool);
        int32_t device_id           = params->device_id;

        IOSubmit(params->stream, feature, memorypool, op_id_, device_id);
        cudaEventRecord(((params->event)), ((params->stream)));
        cudaCheckError();
    }
private:
    int op_id_;
};

Operator* NewSSDIOSubmitOP(int op_id){
    return new SSDIOSubmitOP(op_id);
}

class SSDIOCompleteOP : public Operator {
public:
    SSDIOCompleteOP(int op_id){
        op_id_ = op_id;
    }
    void run(OpParams* params) override {
        FeatureStorage* feature     = (FeatureStorage*)(params->feature);
        UnifiedCache* cache         = (UnifiedCache*)(params->cache);
        MemoryPool* memorypool      = (MemoryPool*)(params->memorypool);
        int mode                    = memorypool->GetCurrentMode();
        int32_t device_id           = params->device_id;

        IOComplete(params->stream, feature, cache, memorypool, device_id, mode);
        cudaEventRecord(((params->event)), ((params->stream)));
        cudaCheckError();
    }
private:
    int op_id_;
};

Operator* NewSSDIOCompleteOP(int op_id){
    return new SSDIOCompleteOP(op_id);
}

