#pragma once
#ifndef GRAPH_STORAGE_H_
#define GRAPH_STORAGE_H_

#include "buildinfo.h"

class GraphStorage {
public: 
    virtual ~GraphStorage() = default;
    //build
    virtual void Build(BuildInfo* info) = 0;
    virtual void GraphCache(int32_t* QT, int32_t Ki, int32_t Kg, int32_t capacity) = 0;
    virtual void Finalize() = 0;
    //CSR
    virtual int32_t GetPartitionCount() const = 0;
	  virtual int64_t** GetCSRNodeIndex(int32_t dev_id) const = 0;
	  virtual int32_t** GetCSRNodeMatrix(int32_t dev_id) const = 0;
    virtual int64_t* GetCSRNodeIndexCPU() const = 0;
    virtual int32_t* GetCSRNodeMatrixCPU() const = 0;
    virtual int64_t Src_Size(int32_t part_id) const = 0;
    virtual int64_t Dst_Size(int32_t part_id) const = 0;
    virtual char* PartitionIndex(int32_t dev_id) const = 0;
    virtual int32_t* PartitionOffset(int32_t dev_id) const = 0;
};
extern "C" 
GraphStorage* NewCompleteGraphStorage();

#endif  // GRAPH_STORAGE_H_