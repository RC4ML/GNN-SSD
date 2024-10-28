#ifndef IOSTACK_H
#define IOSTACK_H

#pragma once
#include "ssdqp.cuh"
#include "system_config.cuh"
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <sys/mman.h>
#define _CUDA
#include "ioctl.h"
#include <sys/ioctl.h>
#include <assert.h>
#include <iostream>

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



typedef uint64_t app_addr_t;
struct IOReq
{
    uint64_t start_lb;
    app_addr_t dest_addr[MAX_ITEMS];
    int num_items;

    __forceinline__ __host__ __device__ IOReq(){};

    __forceinline__
        __host__ __device__
        IOReq(uint64_t ssd, uint64_t length) : start_lb(ssd), num_items(length)
    {
        for (int i = 0; i < num_items; i++)
            dest_addr[i] = ~0ULL;
    }

    __host__ __device__ bool operator<(const IOReq &lhs) const
    {
        return this->start_lb < lhs.start_lb;
    }

    __forceinline__
        __host__ __device__ ~IOReq()
    {
        // delete[] gpu_addr;
    }
};

struct IOHandler
{
    int hop_id;
    int* num_reqs;
};


__device__ int req_id_to_ssd_id(int req_id, int num_ssds, int* ssd_num_reqs_prefix_sum)
{
    int ssd_id = 0;
    for (; ssd_id < num_ssds; ssd_id++)
        if (ssd_num_reqs_prefix_sum[ssd_id] > req_id)
            break;
    return ssd_id;
}

__global__ static void submit_io_req_kernel(IOReq *reqs, int32_t* num_reqs, int num_ssds, int num_queues_per_ssd, SSDQueuePair *ssdqp, uint64_t *prp1, uint64_t **IO_buf_base, uint64_t *prp2, int* ssd_num_reqs_prefix_sum)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int i = thread_id; i < num_reqs[0]; i += num_threads)
    {
        int ssd_id = req_id_to_ssd_id(i, num_ssds, ssd_num_reqs_prefix_sum);
        if (ssd_id >= num_ssds)
            break;
        int req_offset = i - (ssd_id == 0 ? 0 : ssd_num_reqs_prefix_sum[ssd_id - 1]);
        int queue_id = req_offset / (QUEUE_DEPTH - 1);
        assert(queue_id < num_queues_per_ssd);
        int global_queue_id = ssd_id * num_queues_per_ssd + queue_id;
        int id_in_queue = req_offset % (QUEUE_DEPTH - 1);
        int queue_pos = (ssdqp[global_queue_id].sq_tail + id_in_queue) % QUEUE_DEPTH;

        uint64_t io_addr = prp1[ssd_id] + queue_id * QUEUE_IOBUF_SIZE + queue_pos * MAX_IO_SIZE; // assume contiguous!
        uint64_t io_addr2 = io_addr / HOST_PGSZ * HOST_PGSZ + HOST_PGSZ;
        if (reqs[i].num_items * ITEM_SIZE > HOST_PGSZ * 2)
        {
            int prp_size_per_ssd = PRP_SIZE * QUEUE_DEPTH * num_queues_per_ssd / HOST_PGSZ * HOST_PGSZ + HOST_PGSZ;
            uint64_t offset = queue_id * PRP_SIZE * QUEUE_DEPTH + queue_pos * PRP_SIZE;
            io_addr2 = prp2[ssd_id * prp_size_per_ssd / HOST_PGSZ + offset / HOST_PGSZ] + offset % HOST_PGSZ;
        }
        ssdqp[global_queue_id].fill_sq(
            ssdqp[global_queue_id].cmd_id + id_in_queue,                   // command id
            queue_pos,                                                     // position in SQ
            OPCODE_READ,                                                   // opcode
            io_addr,                                                       // prp1
            io_addr2,                                                      // prp2
            reqs[i].start_lb & 0xffffffff,                                 // start lb low
            (reqs[i].start_lb >> 32) & 0xffffffff,                         // start lb high
            RW_RETRY_MASK | (reqs[i].num_items * ITEM_SIZE / LB_SIZE - 1), // number of LBs
            i                                                              // req id
        );
    }
}

__global__ static void ring_sq_doorbell_kernel(int num_ssds, int num_queues_per_ssd, SSDQueuePair *ssdqp, int *ssd_num_reqs, int *ssd_num_reqs_prefix_sum, int32_t* num_reqs)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int i = thread_id; i < num_reqs[0]; i += num_threads)
    {
        int ssd_id = req_id_to_ssd_id(i, num_ssds, ssd_num_reqs_prefix_sum);
        if (ssd_id >= num_ssds)
            break;
        int req_offset = i - (ssd_id == 0 ? 0 : ssd_num_reqs_prefix_sum[ssd_id - 1]);
        int queue_id = req_offset / (QUEUE_DEPTH - 1);
        assert(queue_id < num_queues_per_ssd);
        int global_queue_id = ssd_id * num_queues_per_ssd + queue_id;
        int id_in_queue = req_offset % (QUEUE_DEPTH - 1);

        if (id_in_queue == 0)
        {
            int cnt = ssd_num_reqs[ssd_id] - queue_id * (QUEUE_DEPTH - 1);
            if (cnt > QUEUE_DEPTH - 1)
                cnt = QUEUE_DEPTH - 1;
            ssdqp[global_queue_id].cmd_id += cnt;
            ssdqp[global_queue_id].sq_tail = (ssdqp[global_queue_id].sq_tail + cnt) % QUEUE_DEPTH;
            // printf("thread %d ssd %d queue %d end req %d cnt %d\n", thread_id, ssd_id, queue_id, ssd_num_reqs_prefix_sum[ssd_id], cnt);
            *ssdqp[global_queue_id].sqtdbl = ssdqp[global_queue_id].sq_tail;
        }
    }
}

__global__ static void copy_io_req_kernel(IOReq *reqs, int32_t* num_reqs, int num_ssds, int num_queues_per_ssd, SSDQueuePair *ssdqp, uint64_t *prp1, uint64_t **IO_buf_base, uint64_t *prp2, int* ssd_num_reqs_prefix_sum)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = thread_id / WARP_SIZE;
    int lane_id = thread_id % WARP_SIZE;
    int num_warps = blockDim.x * gridDim.x / WARP_SIZE;
    for (int i = warp_id; i < num_reqs[0]; i += num_warps)
    {
        int ssd_id = req_id_to_ssd_id(i, num_ssds, ssd_num_reqs_prefix_sum);
        int req_offset = i - (ssd_id == 0 ? 0 : ssd_num_reqs_prefix_sum[ssd_id - 1]);
        int queue_id = req_offset / (QUEUE_DEPTH - 1);
        int global_queue_id = ssd_id * num_queues_per_ssd + queue_id;
        int id_in_queue = req_offset % (QUEUE_DEPTH - 1);
        int complete_id = ssdqp[global_queue_id].num_completed + id_in_queue;
        int queue_pos = complete_id % QUEUE_DEPTH;

        if (lane_id == 0)
        {
            // printf("polling req %d ssd %d queue %d complete_id %d queue_pos %d num_completed %d\n", i, ssd_id, queue_id, complete_id, queue_pos, ssdqp[global_queue_id].num_completed);
            uint32_t current_phase = (complete_id / QUEUE_DEPTH) & 1;
            while (((ssdqp[global_queue_id].cq[queue_pos * 4 + 3] & PHASE_MASK) >> 16) == current_phase)
                ;
            uint32_t status = ssdqp[global_queue_id].cq[queue_pos * 4 + 3];
            uint32_t cmd_id = status & CID_MASK;
            if ((status >> 17) & SC_MASK)
            {
                printf("thread %d cq[%d] status: 0x%x, cid: %d\n", thread_id, queue_pos, (status >> 17) & SC_MASK, cmd_id);
                assert(0);
            }
        }

        int cmd_id = ssdqp[global_queue_id].cq[queue_pos * 4 + 3] & CID_MASK;
        int req_id = ssdqp[global_queue_id].cmd_id_to_req_id[cmd_id % QUEUE_DEPTH];
        int sq_pos = ssdqp[global_queue_id].cmd_id_to_sq_pos[cmd_id % QUEUE_DEPTH];
        for (int j = 0; j < reqs[req_id].num_items; j++)
            for (int k = lane_id; k < ITEM_SIZE / 8; k += WARP_SIZE)
                ((uint64_t *)reqs[req_id].dest_addr[j])[k] = IO_buf_base[ssd_id][queue_id * QUEUE_DEPTH * MAX_IO_SIZE / 8 + sq_pos * MAX_IO_SIZE / 8 + j * ITEM_SIZE / 8 + k];
    }
}

__global__ static void ring_cq_doorbell_kernel(int num_ssds, int num_queues_per_ssd, SSDQueuePair *ssdqp, int *ssd_num_reqs, int *ssd_num_reqs_prefix_sum, int32_t* num_reqs)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int i = thread_id; i < num_reqs[0]; i += num_threads)
    {
        int ssd_id = req_id_to_ssd_id(i, num_ssds, ssd_num_reqs_prefix_sum);
        if (ssd_id >= num_ssds)
            break;
        int req_offset = i - (ssd_id == 0 ? 0 : ssd_num_reqs_prefix_sum[ssd_id - 1]);
        int queue_id = req_offset / (QUEUE_DEPTH - 1);
        assert(queue_id < num_queues_per_ssd);
        int global_queue_id = ssd_id * num_queues_per_ssd + queue_id;
        int id_in_queue = req_offset % (QUEUE_DEPTH - 1);

        if (id_in_queue == 0)
        {
            int cnt = ssd_num_reqs[ssd_id] - queue_id * (QUEUE_DEPTH - 1);
            if (cnt > QUEUE_DEPTH - 1)
                cnt = QUEUE_DEPTH - 1;
            ssdqp[global_queue_id].num_completed += cnt;
            ssdqp[global_queue_id].cq_head = (ssdqp[global_queue_id].cq_head + cnt) % QUEUE_DEPTH;
            *ssdqp[global_queue_id].cqhdbl = ssdqp[global_queue_id].cq_head;
            // printf("queue %d num_completed %d cq_head %d\n", global_queue_id, ssdqp[global_queue_id].num_completed, ssdqp[global_queue_id].cq_head);
        }
    }
}

__global__ static void rw_data_kernel(uint32_t opcode, int ssd_id, uint64_t start_lb, uint64_t num_lb, int num_queues_per_ssd, SSDQueuePair *ssdqp, uint64_t *prp1, uint64_t **IO_buf_base, uint64_t *prp2)
{
    uint32_t cid;
    int prp_size_per_ssd = PRP_SIZE * QUEUE_DEPTH * num_queues_per_ssd / HOST_PGSZ * HOST_PGSZ + HOST_PGSZ;
    ssdqp[ssd_id * num_queues_per_ssd].submit(cid, opcode, prp1[ssd_id], LB_SIZE * num_lb <= HOST_PGSZ * 2 ? prp1[ssd_id] + HOST_PGSZ : prp2[ssd_id * prp_size_per_ssd / HOST_PGSZ], start_lb & 0xffffffff, (start_lb >> 32) & 0xffffffff, RW_RETRY_MASK | (num_lb - 1));
    uint32_t status;
    ssdqp[ssd_id * num_queues_per_ssd].poll(status, cid);
    if (status != 0)
    {
        printf("read/write failed with status 0x%x\n", status);
        assert(0);
    }
}

__global__ static void preprocess_io_req_1(IOReq *reqs, int32_t* num_reqs, int num_ssds, int num_queues_per_ssd, int* ssd_num_reqs)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int i = tid; i < num_reqs[0]; i += num_threads)
    {
        int ssd_id = reqs[i].start_lb / NUM_LBS_PER_SSD;
        // assert(ssd_id < num_ssds);
        if(ssd_id < num_ssds && ssd_id >= 0){
            atomicAdd(&ssd_num_reqs[ssd_id], 1);
        }else{
            printf("ssd_id: %d\n", ssd_id);
        }
    }
}

__global__ static void preprocess_io_req_2(IOReq *reqs, int32_t* num_reqs, int num_ssds, int num_queues_per_ssd, int* ssd_num_reqs, int* ssd_num_reqs_prefix_sum)
{
    for (int i = 0; i < num_ssds; i++)
    {
        // assert(ssd_num_reqs[i] <= num_queues_per_ssd * (QUEUE_DEPTH - 1));
        if(ssd_num_reqs[i] > num_queues_per_ssd * (QUEUE_DEPTH - 1)){
            printf("ssd_num_reqs[%d]: %d\n", i, ssd_num_reqs[i]);
        }
        ssd_num_reqs_prefix_sum[i] = ssd_num_reqs[i];
        if (i > 0)
            ssd_num_reqs_prefix_sum[i] += ssd_num_reqs_prefix_sum[i - 1];
    }
}

__device__ int req_ids[MAX_SSDS_SUPPORTED];
__global__ static void distribute_io_req_1(IOReq *reqs, int32_t* num_reqs, int num_ssds, int num_queues_per_ssd, IOReq *distributed_reqs, int* ssd_num_reqs_prefix_sum)
{
    for (int i = 0; i < num_ssds; i++)
        req_ids[i] = i ? ssd_num_reqs_prefix_sum[i - 1] : 0;
}

__global__ static void distribute_io_req_2(IOReq *reqs, int32_t* num_reqs, int num_ssds, int num_queues_per_ssd, IOReq *distributed_reqs)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int i = tid; i < num_reqs[0]; i += num_threads)
    {
        int ssd_id = reqs[i].start_lb / NUM_LBS_PER_SSD;
        // assert(ssd_id < num_ssds);
        if(ssd_id < num_ssds && ssd_id >= 0){
            int req_id = atomicAdd(&req_ids[ssd_id], 1);
            distributed_reqs[req_id] = reqs[i];
            distributed_reqs[req_id].start_lb %= NUM_LBS_PER_SSD;
        }
    }
}

__global__ static void distribute_io_req_3(IOReq *reqs, int32_t* num_reqs, int num_ssds, int num_queues_per_ssd, IOReq *distributed_reqs, int* ssd_num_reqs_prefix_sum)
{
    for (int i = 0; i < num_ssds; i++){
        if(req_ids[i] != ssd_num_reqs_prefix_sum[i]){
            printf("req id %d %d\n", req_ids[i], ssd_num_reqs_prefix_sum[i]);
        }
        // assert(req_ids[i] == ssd_num_reqs_prefix_sum[i]);
    }

}

class IOStack
{
public:
    IOStack(int num_ssds, int num_queues_per_ssd) : num_ssds_(num_ssds), num_queues_per_ssd_(num_queues_per_ssd)
    {
        // alloc device variables
        CHECK(cudaMalloc(&d_ssdqp_, num_ssds_ * num_queues_per_ssd_ * sizeof(SSDQueuePair)));
        CHECK(cudaMalloc(&d_prp1_, num_ssds_ * sizeof(uint64_t)));
        int prp_size_per_ssd = PRP_SIZE * QUEUE_DEPTH * num_queues_per_ssd_ / HOST_PGSZ * HOST_PGSZ + HOST_PGSZ;
        CHECK(cudaMalloc(&d_prp2_, num_ssds_ * prp_size_per_ssd / HOST_PGSZ * sizeof(uint64_t)));
        CHECK(cudaMalloc(&d_IO_buf_base_, num_ssds_ * sizeof(uint64_t *)));
        h_IO_buf_base_ = (uint64_t **)malloc(sizeof(uint64_t *) * num_ssds_);
        CHECK(cudaMalloc(&distributed_reqs_, 4000000 * sizeof(IOReq)));
        CHECK(cudaMalloc(&ssd_num_reqs_, MAX_SSDS_SUPPORTED * sizeof(int)));
        CHECK(cudaMalloc(&ssd_num_reqs_prefix_sum_, MAX_SSDS_SUPPORTED * sizeof(int)));

        // init ssds
        for (int i = 0; i < num_ssds_; i++)
            init_ssd(i);
    }

    ~IOStack()
    {
        CHECK(cudaFree(d_ssdqp_));
        CHECK(cudaFree(d_prp1_));
        CHECK(cudaFree(d_IO_buf_base_));
        munmap(reg_ptr_, REG_SIZE);
        free(h_admin_queue_);
        // CHECK(cudaFree(d_io_queue_));
        // CHECK(cudaFree(d_io_buf_));
        free(h_IO_buf_base_);
    }

    IOHandler* submit_io_req(IOReq *reqs, int32_t* num_reqs, int hop_id, cudaStream_t stream)
    {
        
        cudaMemsetAsync(ssd_num_reqs_, 0, sizeof(int) * num_ssds_, stream);
        cudaCheckError();

        cudaMemsetAsync(ssd_num_reqs_prefix_sum_, 0, sizeof(int) * num_ssds_, stream);
        cudaCheckError();

        preprocess_io_req_1<<<32, NUM_THREADS_PER_BLOCK, 0, stream>>>(reqs, num_reqs, num_ssds_, num_queues_per_ssd_, ssd_num_reqs_);
        cudaCheckError();
        preprocess_io_req_2<<<1, 1, 0, stream>>>(reqs, num_reqs, num_ssds_, num_queues_per_ssd_, ssd_num_reqs_, ssd_num_reqs_prefix_sum_);
        cudaCheckError();

        distribute_io_req_1<<<1, 1, 0, stream>>>(reqs, num_reqs, num_ssds_, num_queues_per_ssd_, distributed_reqs_, ssd_num_reqs_prefix_sum_);
        cudaCheckError();

        distribute_io_req_2<<<32, NUM_THREADS_PER_BLOCK, 0, stream>>>(reqs, num_reqs, num_ssds_, num_queues_per_ssd_, distributed_reqs_);
        cudaCheckError();

        distribute_io_req_3<<<1, 1, 0, stream>>>(reqs, num_reqs, num_ssds_, num_queues_per_ssd_, distributed_reqs_, ssd_num_reqs_prefix_sum_);
        cudaCheckError();

        int num_blocks = 1;
        submit_io_req_kernel<<<num_blocks, NUM_THREADS_PER_BLOCK, 0, stream>>>(distributed_reqs_, num_reqs, num_ssds_, num_queues_per_ssd_, d_ssdqp_, d_prp1_, d_IO_buf_base_, d_prp2_, ssd_num_reqs_prefix_sum_);
        cudaCheckError();

        ring_sq_doorbell_kernel<<<num_blocks, NUM_THREADS_PER_BLOCK, 0, stream>>>(num_ssds_, num_queues_per_ssd_, d_ssdqp_, ssd_num_reqs_, ssd_num_reqs_prefix_sum_, num_reqs);
        cudaCheckError();

        return &IOHandler(hop_id, num_reqs);

    }

    void complete_io_req(std::vector<IOHandler*> handlers, int32_t submission_times, cudaStream_t stream)
    {
        int num_blocks = 32;
        for(int i = 0; i < submission_times; i++){
            copy_io_req_kernel<<<num_blocks, NUM_THREADS_PER_BLOCK, 0, stream>>>(distributed_reqs_, handlers[i].num_reqs, num_ssds_, num_queues_per_ssd_, d_ssdqp_, d_prp1_, d_IO_buf_base_, d_prp2_, ssd_num_reqs_prefix_sum_);
            cudaCheckError();

            ring_cq_doorbell_kernel<<<num_blocks, NUM_THREADS_PER_BLOCK, 0, stream>>>(num_ssds_, num_queues_per_ssd_, d_ssdqp_, ssd_num_reqs_, ssd_num_reqs_prefix_sum_, handlers[i].num_reqs);
            cudaCheckError();
        }
    }

    void read_data(int ssd_id, uint64_t start_lb, uint64_t num_lb)
    {
        rw_data_kernel<<<1, 1>>>(OPCODE_READ, ssd_id, start_lb, num_lb, num_queues_per_ssd_, d_ssdqp_, d_prp1_, d_IO_buf_base_, d_prp2_);
    }

    void write_data(int ssd_id, uint64_t start_lb, uint64_t num_lb)
    {
        rw_data_kernel<<<1, 1>>>(OPCODE_WRITE, ssd_id, start_lb, num_lb, num_queues_per_ssd_, d_ssdqp_, d_prp1_, d_IO_buf_base_, d_prp2_);
    }

    uint64_t **get_d_io_buf_base()
    {
        return d_IO_buf_base_;
    }

    uint64_t **get_h_io_buf_base()
    {
        return h_IO_buf_base_;
    }

    
    int num_ssds_;
    int num_queues_per_ssd_;
    SSDQueuePair *d_ssdqp_;
    uint64_t *d_prp1_, *d_prp2_;
    uint64_t **d_IO_buf_base_, **h_IO_buf_base_;
    void *reg_ptr_;
    void *h_admin_queue_;
    void *d_io_queue_;
    void *d_io_buf_;
    IOReq *distributed_reqs_;
    int* ssd_num_reqs_; 
    int* ssd_num_reqs_prefix_sum_;

    void init_ssd(int ssd_id)
    {
        // open file and map BAR
        fprintf(stderr, "setting up SSD %d\n", ssd_id);
        char fname[20];
        sprintf(fname, "/dev/libnvm%d", ssd_id);
        int fd = open(fname, O_RDWR);
        if (fd < 0)
        {
            fprintf(stderr, "Failed to open: %s\n", strerror(errno));
            exit(1);
        }
        reg_ptr_ = mmap(NULL, REG_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_LOCKED, fd, 0);
        if (reg_ptr_ == MAP_FAILED)
        {
            fprintf(stderr, "Failed to mmap: %s\n", strerror(errno));
            exit(1);
        }
        CHECK(cudaHostRegister(reg_ptr_, REG_SIZE, cudaHostRegisterIoMemory));

        // reset controller
        uint64_t h_reg_ptr = (uint64_t)reg_ptr_;
        *(uint32_t *)(h_reg_ptr + REG_CC) &= ~REG_CC_EN;
        while (*(uint32_t volatile *)(h_reg_ptr + REG_CSTS) & REG_CSTS_RDY)
            ;
        fprintf(stderr, "reset done\n");

        // set admin_qp queue attributes
        *(uint32_t *)(h_reg_ptr + REG_AQA) = ((ADMIN_QUEUE_DEPTH - 1) << 16) | (ADMIN_QUEUE_DEPTH - 1);
        posix_memalign(&h_admin_queue_, HOST_PGSZ, HOST_PGSZ * 2);
        memset(h_admin_queue_, 0, HOST_PGSZ * 2);
        nvm_ioctl_map req; // convert to physical address
        req.vaddr_start = (uint64_t)h_admin_queue_;
        req.n_pages = 2;
        req.ioaddrs = (uint64_t *)malloc(sizeof(uint64_t) * 2);
        int err = ioctl(fd, NVM_MAP_HOST_MEMORY, &req);
        if (err)
        {
            fprintf(stderr, "Failed to map admin_qp queue: %s\n", strerror(errno));
            exit(1);
        }
        uint64_t asq = (uint64_t)h_admin_queue_;
        *(uint64_t *)(h_reg_ptr + REG_ASQ) = req.ioaddrs[0];
        uint64_t acq = (uint64_t)h_admin_queue_ + HOST_PGSZ;
        *(uint64_t *)(h_reg_ptr + REG_ACQ) = req.ioaddrs[1];
        SSDQueuePair admin_qp((volatile uint32_t *)asq, (volatile uint32_t *)acq, BROADCAST_NSID, (uint32_t *)(h_reg_ptr + REG_SQTDBL), (uint32_t *)(h_reg_ptr + REG_CQHDBL), ADMIN_QUEUE_DEPTH);
        fprintf(stderr, "set admin_qp queue attributes done\n");

        // enable controller
        *(uint32_t *)(h_reg_ptr + REG_CC) |= REG_CC_EN;
        while (!(*(uint32_t volatile *)(h_reg_ptr + REG_CSTS) & REG_CSTS_RDY))
            ;
        fprintf(stderr, "enable controller done\n");

        // set number of I/O queues
        uint32_t cid;
        admin_qp.submit(cid, OPCODE_SET_FEATURES, 0x0, 0x0, FID_NUM_QUEUES,
                        ((num_queues_per_ssd_ - 1) << 16) | (num_queues_per_ssd_ - 1));
        uint32_t status;
        admin_qp.poll(status, cid);
        if (status != 0)
        {
            fprintf(stderr, "set number of queues failed with status 0x%x\n", status);
            exit(1);
        }
        fprintf(stderr, "set number of queues done!\n");

        // create I/O queues
        int sq_size = QUEUE_DEPTH * SQ_ITEM_SIZE;
        assert(sq_size % HOST_PGSZ == 0);
        uint64_t *phys = cudaMallocAlignedMapped(d_io_queue_, sq_size * 2 * num_queues_per_ssd_, fd); // 2 stands for SQ and CQ
        CHECK(cudaMemset(d_io_queue_, 0, sq_size * 2 * num_queues_per_ssd_));
        for (int i = 0; i < num_queues_per_ssd_; i++)
        {
            uint64_t sq = (uint64_t)d_io_queue_ + sq_size * (2 * i);
            uint64_t cq = (uint64_t)d_io_queue_ + sq_size * (2 * i + 1);
            int qid = i + 1;
            int offset = sq_size * (2 * i + 1);
            uint64_t prp1 = phys[offset / DEVICE_PGSZ] + offset % DEVICE_PGSZ;
            admin_qp.submit(cid, OPCODE_CREATE_IO_CQ, prp1, 0x0, ((QUEUE_DEPTH - 1) << 16) | qid, 0x1);
            admin_qp.poll(status, cid);
            if (status != 0)
            {
                fprintf(stderr, "create I/O CQ failed with status 0x%x\n", status);
                exit(1);
            }
            offset = sq_size * (2 * i);
            prp1 = phys[offset / DEVICE_PGSZ] + offset % DEVICE_PGSZ;
            admin_qp.submit(cid, OPCODE_CREATE_IO_SQ, prp1, 0x0, ((QUEUE_DEPTH - 1) << 16) | qid, (qid << 16) | 0x1);
            admin_qp.poll(status, cid);
            if (status != 0)
            {
                fprintf(stderr, "create I/O SQ failed with status 0x%x\n", status);
                exit(1);
            }
            uint32_t *cmd_id_to_req_id;
            CHECK(cudaMalloc(&cmd_id_to_req_id, QUEUE_DEPTH * 4));
            uint32_t *cmd_id_to_sq_pos;
            CHECK(cudaMalloc(&cmd_id_to_sq_pos, QUEUE_DEPTH * 4));
            bool *sq_entry_busy;
            CHECK(cudaMalloc(&sq_entry_busy, QUEUE_DEPTH));
            CHECK(cudaMemset(sq_entry_busy, 0, QUEUE_DEPTH));
            SSDQueuePair current_qp((volatile uint32_t *)sq, (volatile uint32_t *)cq, 0x1, (uint32_t *)(h_reg_ptr + REG_SQTDBL + DBL_STRIDE * qid), (uint32_t *)(h_reg_ptr + REG_CQHDBL + DBL_STRIDE * qid), QUEUE_DEPTH, cmd_id_to_req_id, cmd_id_to_sq_pos, sq_entry_busy);
            CHECK(cudaMemcpy(d_ssdqp_ + ssd_id * num_queues_per_ssd_ + i, &current_qp, sizeof(SSDQueuePair), cudaMemcpyHostToDevice));
        }
        // free(phys);
        fprintf(stderr, "create I/O queues done!\n");

        // alloc IO buffer
        phys = cudaMallocAlignedMapped(d_io_buf_, (size_t)QUEUE_IOBUF_SIZE * num_queues_per_ssd_, fd);
        CHECK(cudaMemcpy(d_prp1_ + ssd_id, phys, sizeof(uint64_t), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_IO_buf_base_ + ssd_id, &d_io_buf_, sizeof(uint64_t), cudaMemcpyHostToDevice));
        h_IO_buf_base_[ssd_id] = (uint64_t *)d_io_buf_;
        // for (int i = 0; i < QUEUE_IOBUF_SIZE * num_queues_per_ssd_ / DEVICE_PGSZ; i++)
        //     printf("%lx\n", phys[i]);

        // build PRP list
        assert(PRP_SIZE <= HOST_PGSZ);
        int prp_size_per_ssd = PRP_SIZE * QUEUE_DEPTH * num_queues_per_ssd_ / HOST_PGSZ * HOST_PGSZ + HOST_PGSZ; // 1 table per ssd in host memory
        // printf("prp_size_per_ssd: %d\n", prp_size_per_ssd);
        void *tmp;
        posix_memalign(&tmp, HOST_PGSZ, prp_size_per_ssd);
        memset(tmp, 0, prp_size_per_ssd);
        uint64_t *prp = (uint64_t *)tmp;
        for (int i = 0; i < QUEUE_DEPTH * num_queues_per_ssd_; i++)
            for (int j = 1; j < NUM_PRP_ENTRIES; j++)
            {
                int prp_idx = i * NUM_PRP_ENTRIES + j;
                int offset = i * MAX_IO_SIZE + j * HOST_PGSZ;
                prp[prp_idx - 1] = phys[offset / DEVICE_PGSZ] + offset % DEVICE_PGSZ;
            }
        // Fill in each PRP table
        // free(phys);
        // for (int i = 0; i < QUEUE_DEPTH * num_queues_per_ssd_ * NUM_PRP_ENTRIES; i++)
        //     printf("%lx\n", prp[i]);
        req.vaddr_start = (uint64_t)prp;
        req.n_pages = prp_size_per_ssd / HOST_PGSZ;
        req.ioaddrs = (uint64_t *)malloc(sizeof(uint64_t) * req.n_pages);
        // req.ioaddrs[0] is a physical pointer to PRP table
        err = ioctl(fd, NVM_MAP_HOST_MEMORY, &req);
        if (err)
        {
            fprintf(stderr, "Failed to map: %s\n", strerror(errno));
            exit(1);
        }
        // for (int i = 0; i < req.n_pages; i++)
        //     printf("%lx ", req.ioaddrs[i]);
        // printf("\n");
        CHECK(cudaMemcpy(d_prp2_ + ssd_id * req.n_pages, req.ioaddrs, req.n_pages * sizeof(uint64_t), cudaMemcpyHostToDevice));
        // d_prp2_ is an array of physical pointer to PRP table
    }

    uint64_t *cudaMallocAlignedMapped(void *&vaddr, size_t size, int fd)
    {
        size = size / DEVICE_PGSZ * DEVICE_PGSZ + DEVICE_PGSZ;
        uint64_t *ptr;
        CHECK(cudaMalloc(&ptr, size + DEVICE_PGSZ));
        vaddr = (void *)((uint64_t)ptr / DEVICE_PGSZ * DEVICE_PGSZ + DEVICE_PGSZ);
        int flag = 0;
        if ((uint64_t)vaddr != (uint64_t)ptr)
        {
            flag = 1;
        }
        nvm_ioctl_map req;
        req.ioaddrs = (uint64_t *)malloc(sizeof(uint64_t) * (size / DEVICE_PGSZ + flag));
        req.n_pages = size / DEVICE_PGSZ + flag;
        req.vaddr_start = (uint64_t)ptr;
        int err = ioctl(fd, NVM_MAP_DEVICE_MEMORY, &req);
        if (err)
        {
            printf("Failed to map: %s\n", strerror(errno));
            return nullptr;
        }
        return req.ioaddrs + flag;
    }
};

#endif
