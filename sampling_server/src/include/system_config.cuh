#pragma once

/*-----------------------------iostack parameters---------------------------*/
#include <stdio.h>
#define REG_SIZE 0x4000   // BAR 0 mapped size
#define REG_CC 0x14       // addr: controller configuration
#define REG_CC_EN 0x1     // mask: enable controller
#define REG_CSTS 0x1c     // addr: controller status
#define REG_CSTS_RDY 0x1  // mask: controller ready
#define REG_AQA 0x24      // addr: admin queue attributes
#define REG_ASQ 0x28      // addr: admin submission queue base addr
#define REG_ACQ 0x30      // addr: admin completion queue base addr
#define REG_SQTDBL 0x1000 // addr: submission queue 0 tail doorbell
#define REG_CQHDBL 0x1004 // addr: completion queue 0 sq_tail doorbell
#define DBL_STRIDE 8
#define PHASE_MASK 0x10000 // mask: phase tag
#define HOST_PGSZ 0x1000
#define DEVICE_PGSZ 0x10000
#define CID_MASK 0xffff           // mask: command id
#define SC_MASK 0xff              // mask: status code
#define BROADCAST_NSID 0 // broadcast namespace id
#define OPCODE_SET_FEATURES 0x09
#define OPCODE_CREATE_IO_CQ 0x05
#define OPCODE_CREATE_IO_SQ 0x01
#define OPCODE_READ 0x02
#define OPCODE_WRITE 0x01
#define FID_NUM_QUEUES 0x07
#define LB_SIZE 0x200
#define RW_RETRY_MASK 0x80000000
#define SQ_ITEM_SIZE 64
#define WARP_SIZE 32
#define SQ_HEAD_MASK 0xffff

#define MAX_IO_SIZE 4096
#define ITEM_SIZE 512
#define MAX_ITEMS (MAX_IO_SIZE / ITEM_SIZE)
#define NUM_THREADS_PER_BLOCK 512
#define ADMIN_QUEUE_DEPTH 64
#define QUEUE_DEPTH 4096
#define QUEUE_IOBUF_SIZE (MAX_IO_SIZE * QUEUE_DEPTH)
#define NUM_PRP_ENTRIES (MAX_IO_SIZE / HOST_PGSZ)
#define PRP_SIZE (NUM_PRP_ENTRIES * sizeof(uint64_t))
#define NUM_LBS_PER_SSD 0x100000000
#define MAX_SSDS_SUPPORTED 16


#define INTERBATCH_CON 2 //inter-batch pipeline concurrency 
#define INTRABATCH_CON 3 //intra-batch pipeline concurrency

#define MAX_DEVICE 8
#define MEMORY_USAGE 7
#define TRAINMODE 0
#define VALIDMODE 1
#define TESTMODE  2

#define CACHEMISS_FLAG -2
#define CACHECPU_FLAG -1

#define CHECK(ans) gpuAssert((ans), __FILE__, __LINE__)

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(1);
    }
}
