#ifndef FEATURE_STORAGE_H_
#define FEATURE_STORAGE_H_

#include "buildinfo.h"

class FeatureStorage {
public: 
    virtual ~FeatureStorage() = default;

    virtual void Build(BuildInfo* info) = 0;
    virtual void Finalize() = 0;

    virtual int32_t* GetTrainingSetIds(int32_t part_id) const = 0;
    virtual int32_t* GetValidationSetIds(int32_t part_id) const = 0;
    virtual int32_t* GetTestingSetIds(int32_t part_id) const = 0;

    virtual int32_t* GetTrainingLabels(int32_t part_id) const = 0;
    virtual int32_t* GetValidationLabels(int32_t part_id) const = 0;
    virtual int32_t* GetTestingLabels(int32_t part_id) const = 0;

    virtual int32_t TrainingSetSize(int32_t part_id) const = 0;
    virtual int32_t ValidationSetSize(int32_t part_id) const = 0;
    virtual int32_t TestingSetSize(int32_t part_id) const = 0;

    virtual int32_t TotalNodeNum() const = 0;
    virtual float* GetAllFloatFeature() const = 0;
    virtual int32_t GetFloatFeatureLen() const = 0;

    virtual void Print(BuildInfo* info) = 0;
    
    virtual void IOSubmit(int32_t* sampled_ids, int32_t* cache_index,
                  int32_t* node_counter, float* dst_float_buffer,
                  int32_t op_id, int32_t dev_id, cudaStream_t strm_hdl) = 0;

    virtual void IOComplete(int32_t op_id, int32_t dev_id, cudaStream_t strm_hdl) = 0;

};

extern "C" 
FeatureStorage* NewCompleteFeatureStorage();

#endif  