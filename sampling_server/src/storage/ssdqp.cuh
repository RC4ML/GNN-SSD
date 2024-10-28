#pragma once
#include <stdint.h>
#include <assert.h>
#include "system_config.cuh"
class SSDQueuePair
{
public:
    volatile uint32_t *sq;
    volatile uint32_t *cq;
    uint32_t sq_tail;
    uint32_t cq_head;
    uint32_t cmd_id; // also number of commands submitted
    uint32_t namespace_id;
    uint32_t *sqtdbl, *cqhdbl;
    uint32_t *cmd_id_to_req_id;
    uint32_t *cmd_id_to_sq_pos;
    bool *sq_entry_busy;
    uint32_t queue_depth;
    uint32_t num_completed;

    __host__ __device__ SSDQueuePair()
    {
    }

    __host__ __device__ SSDQueuePair(volatile uint32_t *sq, volatile uint32_t *cq, uint32_t namespace_id, uint32_t *sqtdbl, uint32_t *cqhdbl, uint32_t queue_depth, uint32_t *cmd_id_to_req_id = nullptr, uint32_t *cmd_id_to_sq_pos = nullptr, bool *sq_entry_busy = nullptr)
        : sq(sq), cq(cq), sq_tail(0), cq_head(0), cmd_id(0), namespace_id(namespace_id), sqtdbl(sqtdbl), cqhdbl(cqhdbl), cmd_id_to_req_id(cmd_id_to_req_id), cmd_id_to_sq_pos(cmd_id_to_sq_pos), sq_entry_busy(sq_entry_busy), queue_depth(queue_depth), num_completed(0)
    {
    }

    __host__ __device__ void submit(uint32_t &cid, uint32_t opcode, uint64_t prp1, uint64_t prp2, uint32_t dw10, uint32_t dw11, uint32_t dw12 = 0)
    {
        // printf("%lx %lx %x %x %x %x %x\n", prp1, prp2, dw10, dw11, dw12, opcode, cmd_id);
        fill_sq(cmd_id, sq_tail, opcode, prp1, prp2, dw10, dw11, dw12);
        sq_tail = (sq_tail + 1) % queue_depth;
        *sqtdbl = sq_tail;
        cid = cmd_id;
        cmd_id = (cmd_id + 1) & CID_MASK;
    }

    __host__ __device__ void fill_sq(uint32_t cid, uint32_t pos, uint32_t opcode, uint64_t prp1, uint64_t prp2, uint32_t dw10, uint32_t dw11, uint32_t dw12 = 0, uint32_t req_id = 0xffffffff)
    {
        // if (req_id == 1152)
        //     printf("%lx %lx %x %x %x %x %x %x\n", prp1, prp2, dw10, dw11, dw12, opcode, cid, namespace_id);
        sq[pos * 16 + 0] = opcode | (cid << 16);
        sq[pos * 16 + 1] = namespace_id;
        sq[pos * 16 + 6] = prp1 & 0xffffffff;
        sq[pos * 16 + 7] = prp1 >> 32;
        sq[pos * 16 + 8] = prp2 & 0xffffffff;
        sq[pos * 16 + 9] = prp2 >> 32;
        sq[pos * 16 + 10] = dw10;
        sq[pos * 16 + 11] = dw11;
        sq[pos * 16 + 12] = dw12;
        if (cmd_id_to_req_id)
            cmd_id_to_req_id[cid % queue_depth] = req_id;
        if (cmd_id_to_sq_pos)
            cmd_id_to_sq_pos[cid % queue_depth] = pos;
        if (sq_entry_busy)
            sq_entry_busy[pos] = true;
    }

    __host__ __device__ void poll(uint32_t &code, uint32_t cid)
    {
        uint32_t current_phase = ((cmd_id - 1) / queue_depth) & 1;
        uint32_t status = cq[cq_head * 4 + 3];
        while (((status & PHASE_MASK) >> 16) == current_phase)
            status = cq[cq_head * 4 + 3];
        if ((status & CID_MASK) != cid)
        {
            printf("expected cid: %d, actual cid: %d\n", cid, status & CID_MASK);
            assert(0);
        }
        cq_head = (cq_head + 1) % queue_depth;
        *cqhdbl = cq_head;
        code = (status >> 17) & SC_MASK;
        num_completed++;
    }

    __device__ uint32_t poll_range(int expected_sq_head, bool should_break)
    {
        // printf("cmd_id: %d, size: %d, current_phase: %d\n", cmd_id, size, current_phase);
        int i;
        uint32_t last_sq_head = ~0U;
        int last_num_completed = num_completed;
        int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
        for (i = cq_head; (num_completed & CID_MASK) != cmd_id; i = (i + 1) % queue_depth)
        {
            uint32_t current_phase = (num_completed / queue_depth) & 1;
            uint32_t status = cq[i * 4 + 3];
            uint64_t start = clock64();
            while (((status & PHASE_MASK) >> 16) == current_phase)
            {
                status = cq[i * 4 + 3];
                if (clock64() - start > 1000000000)
                {
                    printf("timeout sq_tail=%d, cq_head=%d, i=%d, num_completed=%d, cmd_id=%d\n", sq_tail, cq_head, i, num_completed, cmd_id);
                    printf("last_sq_head: %d, expected_sq_head: %d\n", last_sq_head, expected_sq_head);
                    // int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
                    // if (thread_id)
                    //     return 0;
                    // for (int m = 0; m < queue_depth; m++)
                    // {
                    //     printf("SQE %d\n", m);
                    //     for (int n = 0; n < 16; n++)
                    //         printf("DW%2d, %08x\n", n, sq[m * 16 + n]);
                    // }
                    // for (int m = 0; m < queue_depth; m++)
                    // {
                    //     printf("CQE %d\n", m);
                    //     for (int n = 0; n < 4; n++)
                    //         printf("DW%2d, %08x\n", n, cq[m * 4 + n]);
                    // }
                    return 1;
                }
            }
            int cmd_id = status & CID_MASK;
            int sq_pos = cmd_id_to_sq_pos[cmd_id % queue_depth];
            if ((status >> 17) & SC_MASK)
            {
                printf("cq[%d] status: 0x%x, cid: %d\n", i, (status >> 17) & SC_MASK, status & CID_MASK);
                int req_id = cmd_id_to_req_id[cmd_id % queue_depth];
                printf("req_id: %d, sq_pos: %d\n", req_id, sq_pos);
                // for (int i = 0; i < 16; i++)
                //     printf("%08x ", sq[sq_pos * 16 + i]);
                // printf("\n");
                return (status >> 17) & SC_MASK;
            }
            last_sq_head = cq[i * 4 + 2] & SQ_HEAD_MASK;
            sq_entry_busy[sq_pos] = false;
            // printf("thread %d freed sq_pos %d\n", thread_id, sq_pos);
            num_completed++;
            if (should_break && ((cq[i * 4 + 2] & SQ_HEAD_MASK) - expected_sq_head + queue_depth) % queue_depth <= WARP_SIZE)
            {
                // printf("cq[%d] sq_head: %d, expected_sq_head: %d\n", i, cq[i * 4 + 2] & SQ_HEAD_MASK, expected_sq_head);
                i = (i + 1) % queue_depth;
                if (num_completed - last_num_completed > 64)
                    printf("%d: %d completed\n", thread_id, num_completed - last_num_completed);
                break;
            }
        }
        if (i != cq_head)
        {
            cq_head = i;
            // printf("cq_head is %p, set cqhdbl to %d\n", cqhdbl, cq_head);
            *cqhdbl = cq_head;
        }
        return 0;
    }

    __host__ __device__ uint32_t poll_until_sq_entry_free(int expected_sq_pos)
    {
        int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
        int last_num_completed = num_completed;
        // printf("thread %d want to free sq_pos: %d num_completed %d cmd_id %d\n", thread_id, expected_sq_pos, num_completed, cmd_id);
        int i;
        for (i = cq_head; (num_completed & CID_MASK) != cmd_id; i = (i + 1) % queue_depth)
        {
            uint32_t current_phase = (num_completed / queue_depth) & 1;
            uint32_t status = cq[i * 4 + 3];
            while (((status & PHASE_MASK) >> 16) == current_phase)
                status = cq[i * 4 + 3];
            int cmd_id = status & CID_MASK;
            int sq_pos = cmd_id_to_sq_pos[cmd_id % queue_depth];
            if ((status >> 17) & SC_MASK)
            {
                printf("cq[%d] status: 0x%x, cid: %d\n", i, (status >> 17) & SC_MASK, status & CID_MASK);
                int req_id = cmd_id_to_req_id[cmd_id % queue_depth];
                printf("req_id: %d, sq_pos: %d\n", req_id, sq_pos);
                // for (int i = 0; i < 16; i++)
                //     printf("%08x ", sq[sq_pos * 16 + i]);
                // printf("\n");
                return (status >> 17) & SC_MASK;
            }
            sq_entry_busy[sq_pos] = false;
            // printf("thread %d manually freed sq_pos %d\n", thread_id, sq_pos);
            num_completed++;
            if (sq_pos == expected_sq_pos)
            {
                cq_head = (i + 1) % queue_depth;
                // printf("cq_head is %p, set cqhdbl to %d\n", cqhdbl, cq_head);
                *cqhdbl = cq_head;
                if (num_completed - last_num_completed > 64)
                    printf("%d: %d completed\n", thread_id, num_completed - last_num_completed);
                return 0;
            }
        }
        // printf("thread %d failed to free sq_pos %d\n", thread_id, expected_sq_pos);
        return 1;
    }
};
