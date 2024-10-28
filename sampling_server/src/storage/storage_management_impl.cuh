#ifndef STORAGE_MANAGEMENT_IMPL_H_
#define STORAGE_MANAGEMENT_IMPL_H_

#include <algorithm>
#include <functional>
#include <iostream>
#include <cuda_runtime.h>

#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <numeric>
#include <math.h>
#include <thread>
#include <numeric>
#include <chrono>
#include <random>

#include <cstdint>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <sys/types.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>



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


void mmap_trainingset_read(std::string &training_file, std::vector<int32_t>& training_set_ids){
    int64_t t_idx = 0;
    int32_t fd = open(training_file.c_str(), O_RDONLY);
    if(fd == -1){
        std::cout<<"cannout open file: "<<training_file<<"\n";
    }
    // int64_t buf_len = lseek(fd, 0, SEEK_END);
    int64_t buf_len = int64_t(int64_t(training_set_ids.size()) * 4); 
    const int32_t* buf = (int32_t *)mmap(NULL, buf_len, PROT_READ, MAP_PRIVATE, fd, 0);
    const int32_t* buf_end = buf + buf_len/sizeof(int32_t);
    int32_t temp;
    while(buf < buf_end){
        temp = *buf;
        training_set_ids[t_idx++] = temp;
        buf++;
    }
    close(fd);
    return;
}

int32_t mmap_partition_read(std::string &partition_file, int32_t* partition_index){
    int64_t part_idx = 0;
    int32_t fd = open(partition_file.c_str(), O_RDONLY);
    // if(fd == -1){
    //     std::cout<<"cannout open file: "<<partition_file<<"\n";
    // }
    int64_t buf_len = lseek(fd, 0, SEEK_END);
    const int32_t* buf = (int32_t *)mmap(NULL, buf_len, PROT_READ, MAP_PRIVATE, fd, 0);
    const int32_t* buf_end = buf + buf_len/sizeof(int32_t);
    int32_t temp;
    while(buf < buf_end){
        temp = *buf;
        partition_index[part_idx++] = temp;
        buf++;
    }
    close(fd);
    return fd;
}

void mmap_indptr_read(std::string &indptr_file, int64_t* indptr){
    int64_t indptr_index = 0;
    int32_t fd = open(indptr_file.c_str(), O_RDONLY);
    if(fd == -1){
        std::cout<<"cannout open file: "<<indptr_file<<"\n";
    }
    int64_t buf_len = lseek(fd, 0, SEEK_END);
    const int64_t *buf = (int64_t *)mmap(NULL, buf_len, PROT_READ, MAP_PRIVATE, fd, 0);
    const int64_t* buf_end = buf + buf_len/sizeof(int64_t);
    int64_t temp;
    while(buf < buf_end){
        temp = *buf;
        indptr[indptr_index++] = temp;
        buf++;
    }
    close(fd);
    return;
}

void mmap_indices_read(std::string &indices_file, int32_t* indices){
    int64_t indices_index = 0;
    int32_t fd = open(indices_file.c_str(), O_RDONLY);
    if(fd == -1){
        std::cout<<"cannout open file: "<<indices_file<<"\n";
    }
    int64_t buf_len = lseek(fd, 0, SEEK_END);
    const int32_t *buf = (int32_t *)mmap(NULL, buf_len, PROT_READ, MAP_PRIVATE, fd, 0);
    const int32_t* buf_end = buf + buf_len/sizeof(int32_t);
    int32_t temp;
    while(buf < buf_end){
        temp = *buf;
        indices[indices_index++] = temp;
        buf++;
    }
    close(fd);
    return;
}

void mmap_features_read(std::string &features_file, float* features){
    int64_t n_idx = 0;
    int32_t fd = open(features_file.c_str(), O_RDONLY);
    if(fd == -1){
        std::cout<<"cannout open file: "<<features_file<<"\n";
    }
    int64_t buf_len = lseek(fd, 0, SEEK_END);
    const float *buf = (float *)mmap(NULL, buf_len, PROT_READ, MAP_PRIVATE, fd, 0);
    const float* buf_end = buf + buf_len/sizeof(float);
    float temp;
    while(buf < buf_end){
        temp = *buf;
        features[n_idx++] = temp;
        buf++;
    }
    close(fd);
    return;
}

void mmap_labels_read(std::string &labels_file, std::vector<int32_t>& labels){
    int64_t n_idx = 0;
    int32_t fd = open(labels_file.c_str(), O_RDONLY);
    if(fd == -1){
        std::cout<<"cannout open file: "<<labels_file<<"\n";
    }
    int64_t buf_len = lseek(fd, 0, SEEK_END);
    const int32_t *buf = (int32_t *)mmap(NULL, buf_len, PROT_READ, MAP_PRIVATE, fd, 0);
    const int32_t* buf_end = buf + buf_len/sizeof(int32_t);
    int32_t temp;
    while(buf < buf_end){
        temp = *buf;
        labels[n_idx++] = temp;
        buf++;
    }
    close(fd);
    return;
}

#endif