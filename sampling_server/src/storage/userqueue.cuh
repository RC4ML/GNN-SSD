#include <iostream>
#include <climits>
// #include <iomanip>
#include <stdlib.h>
#include <cuda.h>
#include <vector>
#include <algorithm>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/shuffle.h>
#include <unistd.h>
#include <cub/cub.cuh>
// #include

#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/linear_congruential_engine.h>

#define my_ULLONG_MAX (~0ULL)
// #define MAX_ITEMS 16//when setting to 32 will cause compilation error
using namespace std;

typedef uint64_t app_addr_t;

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


struct cache_id{
    uint64_t buffer_id;
    uint64_t ssd_id;
    __forceinline__
    __host__ __device__
    cache_id(){
        buffer_id=my_ULLONG_MAX;
        ssd_id=my_ULLONG_MAX;
    }
    __forceinline__
    __host__ __device__
    cache_id(uint64_t ssd,uint64_t buffer):ssd_id(ssd),buffer_id(buffer){}
    
    __host__ __device__
    bool operator < (const cache_id& lhs)const{
        return this->ssd_id<lhs.ssd_id;
    }
};

struct Compare{
    __device__ 
    bool operator()(const IOReq& a, const IOReq& b) const
        {
            return a.start_lb < b.start_lb;
        }

    __device__ 
    bool operator()(const cache_id& a,const cache_id& b) const{
        return a.ssd_id < b.ssd_id;
    }
};

__global__
void dequeue_kernel(int32_t* node_counter, int op_id, IOReq *d_ret, int32_t* p_miss_cnt, int32_t* input_ids, int32_t* cache_index, int32_t float_feature_len,
                    float *dst_float_buffer, int32_t num_ssd);

__global__
void dequeuerandom_kernel(int32_t* node_counter, int op_id, IOReq *d_ret, int32_t* p_miss_cnt, int32_t* input_ids, int32_t* cache_index, int32_t float_feature_len,
                    float *dst_float_buffer, int32_t num_ssd);

struct UserQueue{
    void *ssd_start_addr;
    uint64_t grid_size{80};//cuda param
    uint64_t block_size{1024};

    uint64_t ssd_block_size{ITEM_SIZE};//block size
    uint64_t init_buffer_cnt_;
    
    IOReq *d_ret_;
    uint64_t *d_ret_ssd_;

    int32_t* cache_miss_;
    int32_t* p_miss_cnt_;

    // __forceinline__
    __host__ __device__
    UserQueue(uint64_t grid_size,uint64_t block_size,uint64_t merge_lenth,uint64_t ssd_block_size,uint64_t init_buffer_cnt):
            grid_size(grid_size),block_size(block_size),
            ssd_block_size(ssd_block_size),init_buffer_cnt_(init_buffer_cnt){
            cudaMalloc((void **)&p_miss_cnt_,sizeof(uint64_t));

            cudaMalloc((void **)&d_split_flag_,sizeof(uint64_t)*init_buffer_cnt_);
            cudaMalloc((void **)&d_ret_,sizeof(IOReq)*init_buffer_cnt_);
            cudaMalloc((void **)&d_ret_ssd_,sizeof(uint64_t)*init_buffer_cnt_); 
            cudaMalloc((void **)&cache_miss_,sizeof(int32_t)*init_buffer_cnt_);            
            };

    __forceinline__
    __host__ __device__
    ~UserQueue(){
        //maybe free the cuda memory
    }
    __host__
    IOReq* dequeuerandom(int32_t* node_counter, int32_t op_id, int32_t* cache_index, int32_t* input_ids, int32_t* p_miss_cnt, float* dst_float_buffer, int32_t float_feature_len, int num_ssd, cudaStream_t stream){
        
        dequeuerandom_kernel<<<grid_size, block_size, 0, stream>>>(node_counter, op_id, d_ret_, p_miss_cnt_, input_ids, cache_index, float_feature_len, dst_float_buffer, num_ssd);
        cudaMemcpyAsync(p_miss_cnt,p_miss_cnt_,sizeof(int32_t),cudaMemcpyDeviceToDevice, stream);
        cudaMemsetAsync(p_miss_cnt_, 0, sizeof(int32_t), stream);
        // int32_t* h_miss_cnt = (int32_t*)malloc(sizeof(int32_t));
        // cudaMemcpy(h_miss_cnt, p_miss_cnt, sizeof(int32_t), cudaMemcpyDeviceToHost);
        // std::cout<<"op "<<op_id<<" cache miss "<<h_miss_cnt[0]<<"\n";
        return d_ret_;
    }
    __host__
    IOReq* dequeue(int32_t* node_counter, int32_t op_id, int32_t* cache_index, int32_t* input_ids, int32_t* p_miss_cnt, float* dst_float_buffer, int32_t float_feature_len, int num_ssd, cudaStream_t stream){
        
        dequeue_kernel<<<grid_size, block_size, 0, stream>>>(node_counter, op_id, d_ret_, p_miss_cnt_, input_ids, cache_index, float_feature_len, dst_float_buffer, num_ssd);
        cudaMemcpyAsync(p_miss_cnt,p_miss_cnt_,sizeof(int32_t),cudaMemcpyDeviceToDevice, stream);
        cudaMemsetAsync(p_miss_cnt_, 0, sizeof(int32_t), stream);
        // int32_t* h_miss_cnt = (int32_t*)malloc(sizeof(int32_t));
        // cudaMemcpy(h_miss_cnt, p_miss_cnt, sizeof(int32_t), cudaMemcpyDeviceToHost);
        // std::cout<<"op "<<op_id<<" cache miss "<<h_miss_cnt[0]<<"\n";
        return d_ret_;
    }
};


__global__
void dequeue_kernel(int32_t* node_counter, int op_id, IOReq *d_ret, int32_t* p_miss_cnt, int32_t* input_ids, int32_t* cache_index, int32_t float_feature_len,
                    float *dst_float_buffer, int32_t num_ssd){
    uint64_t thread_id=blockIdx.x*blockDim.x+threadIdx.x;  
    int32_t  node_off = node_counter[(op_id % INTRABATCH_CON) * 2];
    int32_t  batch_size = node_counter[(op_id % INTRABATCH_CON) * 2 + 1];
    int feature_block_size = float_feature_len * sizeof(float) / ITEM_SIZE;            
    for(;thread_id<batch_size;thread_id+=blockDim.x*gridDim.x){
        int32_t input_id = input_ids[thread_id];
        int32_t cache_idx = cache_index[thread_id];
        if(cache_idx < 0){
            uint64_t offset = atomicAdd(p_miss_cnt, 1);
            d_ret[offset].start_lb = uint64_t(input_id % num_ssd) * NUM_LBS_PER_SSD + uint64_t(input_id / num_ssd) * feature_block_size;//raid0, for p5510
            d_ret[offset].num_items = feature_block_size;
            for(int j = 0; j < feature_block_size; j++){
                d_ret[offset].dest_addr[j] = (app_addr_t)(dst_float_buffer + (int64_t(node_off) * float_feature_len) + (1ll * thread_id * float_feature_len + j * ITEM_SIZE) / sizeof(float));
            }
        }
    }
}

__global__
void dequeuerandom_kernel(int32_t* node_counter, int op_id, IOReq *d_ret, int32_t* p_miss_cnt, int32_t* input_ids, int32_t* cache_index, int32_t float_feature_len,
                    float *dst_float_buffer, int32_t num_ssd){
    uint64_t thread_id=blockIdx.x*blockDim.x+threadIdx.x;  
    int32_t  node_off = node_counter[(op_id % INTRABATCH_CON) * 2];
    int32_t  batch_size = node_counter[(op_id % INTRABATCH_CON) * 2 + 1];
    int feature_block_size = float_feature_len * sizeof(float) / ITEM_SIZE;            
    for(;thread_id<batch_size;thread_id+=blockDim.x*gridDim.x){
        thrust::minstd_rand engine;
        engine.discard(thread_id);
        thrust::uniform_int_distribution<> dist(0, 1000000000 - 1);
        int32_t input_id = dist(engine);
        // int32_t input_id = input_ids[thread_id];
        int32_t cache_idx = cache_index[thread_id];
        if(cache_idx < 0){
            uint64_t offset = atomicAdd(p_miss_cnt, 1);
            d_ret[offset].start_lb = uint64_t(input_id % num_ssd) * NUM_LBS_PER_SSD + (input_id % (NUM_LBS_PER_SSD/(feature_block_size)))*feature_block_size;//uint64_t(input_id / num_ssd) * feature_block_size;//raid0
            d_ret[offset].num_items = feature_block_size;
            for(int j = 0; j < feature_block_size; j++){
                d_ret[offset].dest_addr[j] = (app_addr_t)(dst_float_buffer + (int64_t(node_off) * float_feature_len) + (1ll * thread_id * float_feature_len + j * ITEM_SIZE) / sizeof(float));
            }
        }
    }
}

