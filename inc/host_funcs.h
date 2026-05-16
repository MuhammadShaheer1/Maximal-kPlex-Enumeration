#ifndef CUTS_HOST_FUNCS_H
#define CUTS_HOST_FUNCS_H
#ifndef NVTX3_USE_CHECKED_OVERLOADS_FOR_GET
#define NVTX3_USE_CHECKED_OVERLOADS_FOR_GET 0
#endif
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include "device_funcs.h"
#include "kPlexEnum.h"
#include "gpu_memory_allocation.h"
#include "free_memories.h"

template <typename T>
graph<T> peelGraph(const graph<T> &g, bool *const mark, int *const resNei)
{
    const int n = g.n;
#pragma omp parallel
    {
        thread_local std::queue<int> Q;
#pragma omp for
        for (int i = 0; i < n; ++i)
        {
            if (g.degree[i] < bd)
            {
                mark[i] = false;
                Q.push(i);
            }
            else
            {
                mark[i] = true;
                resNei[i] = g.degree[i];
            }
        }
        while (Q.size())
        {
            const int ele = Q.front();
            Q.pop();
            for (int i = g.offsets[ele]; i < g.offsets[ele + 1]; ++i)
            {
                const int nei = g.neighbors[i];
                if (mark[nei])
                {
                    int old = resNei[nei];
                    while (!utils::CAS(&resNei[nei], old, old - 1))
                    {
                        old = resNei[nei];
                    }
                    if (old == bd)
                    {
                        mark[nei] = false;
                        Q.push(nei);
                    }
                }
            }
        }
    }
    int *const map = new int[n];
#pragma omp parallel for
    for (int i = 0; i < n; ++i)
        map[i] = i;
    _seq<int> leadList = sequence::pack(map, mark, n);
    const int pn = leadList.n;
#pragma omp parallel for
    for (int i = 0; i < pn; ++i)
        map[leadList.A[i]] = i;
    // vertex<int> *vertices = newA(vertex<int>, pn);
    uintT *newOffsets = newA(uintT, pn + 1);
    uintT *newDegrees = newA(uintT, pn);

#pragma omp parallel for
    for (int i = 0; i < pn; i++)
    {
        int ori = leadList.A[i];
        int count = 0;
        for (uintT j = g.offsets[ori]; j < g.offsets[ori + 1]; j++)
        {
            int nei = g.neighbors[j];
            if (mark[nei])
                count++;
        }
        newDegrees[i] = count;
    }

    newOffsets[0] = 0;
    for (int i = 1; i < pn + 1; i++)
    {
        newOffsets[i] = newOffsets[i - 1] + newDegrees[i - 1];
    }

    uintT totalEdges = newOffsets[pn];
    uintT *newNeighbors = newA(uintT, totalEdges);

#pragma omp parallel for
    for (int i = 0; i < pn; i++)
    {
        int ori = leadList.A[i];
        int cursor = newOffsets[i];
        for (uintT j = g.offsets[ori]; j < g.offsets[ori + 1]; j++)
        {
            int nei = g.neighbors[j];
            if (mark[nei])
            {
                newNeighbors[cursor++] = map[nei];
            }
        }
    }

    delete[] map;
    leadList.del();

    graph<T> newGraph;
    printf("\npn = %d\n", pn);
    newGraph.n = pn;
    newGraph.m = totalEdges;
    newGraph.offsets = newOffsets;
    newGraph.neighbors = newNeighbors;
    newGraph.degree = newDegrees;
    return newGraph;
}

void computeOffsets(S_pointers &s, unsigned int *d_blk_counter)
{
    thrust::device_vector<unsigned> d_keys;
    if (d_keys.size() == 0)
    {
        d_keys.resize(WARPS * MAX_BLK_SIZE);
        thrust::transform(
            thrust::device,
            thrust::make_counting_iterator<unsigned>(0),
            thrust::make_counting_iterator<unsigned>(WARPS * MAX_BLK_SIZE),
            d_keys.begin(),
            [] __device__(unsigned idx)
            { return idx / MAX_BLK_SIZE; });
    }

    auto deg_ptr = thrust::device_pointer_cast(s.degree);
    auto ldeg_ptr = thrust::device_pointer_cast(s.l_degree);
    auto off_ptr = thrust::device_pointer_cast(s.offsets);
    auto loff_ptr = thrust::device_pointer_cast(s.l_offsets);

    thrust::exclusive_scan_by_key(
        thrust::device,
        d_keys.begin(),
        d_keys.end(),
        deg_ptr,
        off_ptr);
    thrust::exclusive_scan_by_key(
        thrust::device,
        d_keys.begin(),
        d_keys.end(),
        ldeg_ptr,
        loff_ptr);
}

void checkCudaError(int kernel)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error in %d: %s\n", kernel, cudaGetErrorString(err));
        return;
    }
}

void spillToHost(T_pointers &t, unsigned int* d_tail_A, HostTaskBuffer& hostBuf)
{
    unsigned int tail = 0;
    cudaMemcpy(&tail, d_tail_A, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    unsigned int toMove = std::min(unsigned(STAGING_CHUNK), tail);

    unsigned int deviceStart = tail - toMove;

    // cudaMemcpyAsync(h_task_stage, d_tasks_A + deviceStart, toMove * sizeof(HostTask), cudaMemcpyDeviceToHost, stream);

    // cudaStreamSynchronize(stream);
    Task tmp;
    for (unsigned int i = 0; i < toMove; i++)
    {
        unsigned int idx = deviceStart + i;

        cudaMemcpy(&tmp, t.d_tasks_A + idx, sizeof(Task), cudaMemcpyDeviceToHost);

        HostTask& h = hostBuf.tasks[hostBuf.size+i];
        h.idx = tmp.idx;
        h.PlexSz = tmp.PlexSz;
        h.CandSz = tmp.CandSz;
        h.ExclSz = tmp.ExclSz;

        cudaMemcpy(h.labels, tmp.labels, MAX_BLK_SIZE * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h.neiInG, tmp.neiInG, MAX_BLK_SIZE * sizeof(uint16_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h.neiInP, tmp.neiInP, MAX_BLK_SIZE * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    }

    // memcpy(hostBuf.tasks + hostBuf.size, h_task_stage, toMove * sizeof(HostTask));

    hostBuf.size += toMove;

    tail -= toMove;
    cudaMemcpy(d_tail_A, &tail, sizeof(unsigned int), cudaMemcpyHostToDevice);
}

void reFillTasks(T_pointers& t, unsigned int* d_tail_A, HostTaskBuffer& hostBuf)
{
    unsigned int tail;
    cudaMemcpy(&tail, d_tail_A, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    unsigned int fromHost = std::min(unsigned(STAGING_CHUNK), hostBuf.size);

    unsigned int hostStart = hostBuf.size - fromHost;

    // memcpy(h_task_stage, hostBuf.tasks + hostStart, fromHost * sizeof(HostTask));

    // cudaMemcpyAsync(d_tasks_A + tail, h_task_stage, fromHost * sizeof(HostTask), cudaMemcpyHostToDevice, stream);

    // cudaStreamSynchronize(stream);

    for (unsigned int i = 0; i < fromHost; i++)
    {
        HostTask &h = hostBuf.tasks[hostStart + i];

        unsigned int idx = tail + i;

        uint8_t* labels_dev = t.d_all_labels_A + idx*MAX_BLK_SIZE;
        uint16_t* neiInG_dev = t.d_all_neiInG_A + idx*MAX_BLK_SIZE;
        uint16_t* neiInP_dev = t.d_all_neiInP_A + idx * MAX_BLK_SIZE;

        cudaMemcpy(labels_dev, h.labels, MAX_BLK_SIZE * sizeof(uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(neiInG_dev, h.neiInG, MAX_BLK_SIZE * sizeof(uint16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(neiInP_dev, h.neiInP, MAX_BLK_SIZE * sizeof(uint16_t), cudaMemcpyHostToDevice);

        Task tmp;
        tmp.idx = h.idx;
        tmp.PlexSz = h.PlexSz;
        tmp.CandSz = h.CandSz;
        tmp.ExclSz = h.ExclSz;
        tmp.labels = labels_dev;
        tmp.neiInG = neiInG_dev;
        tmp.neiInP = neiInP_dev;

        cudaMemcpy(t.d_tasks_A + idx, &tmp, sizeof(Task), cudaMemcpyHostToDevice);
    }

    hostBuf.size -= fromHost;

    tail += fromHost;
    cudaMemcpy(d_tail_A, &tail, sizeof(unsigned int), cudaMemcpyHostToDevice);
}
//original
// void initializeBNB(int initialN, T_pointers &task_pointers, P_pointers plex_pointers, S_pointers subgraph_pointers, unsigned int *d_blk, unsigned int *d_left, unsigned int *d_blk_counter, unsigned int *d_left_counter, uint8_t *commonMtx, unsigned int *plex_count, uint16_t* d_sat, uint16_t* d_commons, uint32_t* d_uni, unsigned long long* cycles, uint32_t* d_adj, int* d_abort, unsigned int* global_count)
// {
//     cudaMemset(d_abort, 0, sizeof(int));
//     int h_abort = 0;
//     unsigned int head = 0;
//     // cudaMemcpy(&tail_max, task_pointers.d_tail_A, sizeof(unsigned int), cudaMemcpyDeviceToHost);
//     while (true)
//     {
//         unsigned int tail;
//         // unsigned int plex;
//         cudaMemcpy(&tail, task_pointers.d_tail_A, sizeof(unsigned int), cudaMemcpyDeviceToHost);
//         // cudaMemcpy(&plex, plex_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
//         // printf("tail: %u\n", tail);
//         if (tail == 0)
//             break;

//         unsigned int batch;
//         batch = std::min((unsigned)5*WARPS, tail);
//         // else batch = std::min((unsigned)3*WARPS, tail);

//         head = tail - batch;
        
//         chkerr(cudaMemcpy(task_pointers.d_tail_B, &batch, sizeof(unsigned int), cudaMemcpyHostToDevice));
//         chkerr(cudaMemcpy(task_pointers.d_tasks_B, task_pointers.d_tasks_A + head, batch * sizeof(Task), cudaMemcpyDeviceToDevice));
//         chkerr(cudaMemcpy(task_pointers.d_all_labels_B, task_pointers.d_all_labels_A + head * MAX_BLK_SIZE, batch * MAX_BLK_SIZE * sizeof(uint8_t), cudaMemcpyDeviceToDevice));
//         chkerr(cudaMemcpy(task_pointers.d_all_neiInG_B, task_pointers.d_all_neiInG_A + head * MAX_BLK_SIZE, batch * MAX_BLK_SIZE * sizeof(uint16_t), cudaMemcpyDeviceToDevice));
//         chkerr(cudaMemcpy(task_pointers.d_all_neiInP_B, task_pointers.d_all_neiInP_A + head * MAX_BLK_SIZE, batch * MAX_BLK_SIZE * sizeof(uint16_t), cudaMemcpyDeviceToDevice));

//         tail = head;
//         cudaMemcpy(task_pointers.d_tail_A, &tail, sizeof(tail), cudaMemcpyHostToDevice);
//         bool flip = false;

//         while (true)
//         {
//             unsigned int *tail_in = flip ? task_pointers.d_tail_C : task_pointers.d_tail_B;
//             unsigned int *tail_out = flip ? task_pointers.d_tail_B : task_pointers.d_tail_C;
//             Task *Q_in = flip ? task_pointers.d_tasks_C : task_pointers.d_tasks_B;
//             Task *Q_out = flip ? task_pointers.d_tasks_B : task_pointers.d_tasks_C;
//             uint8_t *lab_out = flip ? task_pointers.d_all_labels_B : task_pointers.d_all_labels_C;
//             uint16_t *nei_out = flip ? task_pointers.d_all_neiInG_B : task_pointers.d_all_neiInG_C;
//             uint16_t *P_out = flip ? task_pointers.d_all_neiInP_B : task_pointers.d_all_neiInP_C;

//             cudaMemcpy(&tail, tail_in, sizeof(unsigned int), cudaMemcpyDeviceToHost);
//             // printf("tail inside: %d\n", tail);
//             if (tail == 0)
//                 break;
//             cudaMemset(tail_out, 0, sizeof(unsigned int));
//             unsigned int numTasks = tail;
//             unsigned int waves = (numTasks) / WARPS + 1;

//             for (unsigned int w = 0; w < waves; w++)
//             {
//                 BNB<<<BLK_NUMS, BLK_DIM>>>(w, plex_pointers, subgraph_pointers, d_blk, d_left, d_blk_counter, d_left_counter, commonMtx, Q_in, Q_out, task_pointers.d_tasks_A, numTasks, 0, tail_out, task_pointers.d_tail_A, lab_out, nei_out, P_out, task_pointers.d_all_labels_A, task_pointers.d_all_neiInG_A, task_pointers.d_all_neiInP_A, plex_count, d_sat, d_commons, d_uni, cycles, d_adj, d_abort, global_count);
//                 cudaMemcpy(&h_abort, d_abort, sizeof(int), cudaMemcpyDeviceToHost);
//                 if (h_abort) 
//                 {
//                     printf("Maximum Capacity Reached on level %d\n", initialN);
//                     break;
//                 }
//                 // cudaMemcpy(&tail, task_pointers.d_tail_A, sizeof(unsigned int), cudaMemcpyDeviceToHost);
//                 // printf("tail: %d, capacity: %u\n", tail, MAX_CAP/4);
//                 // if (h_abort)
//                 // {
//                     // printf("Maximum Capacity Reached on level %d\n", initialN);
//                     // printf("Copying Some Tasks To Host Memory with size: %u\n", hostBuf.size);
//                     // spillToHost(task_pointers, task_pointers.d_tail_A, hostBuf);
//                     // cudaMemset(d_abort, 0, sizeof(int));
//                 // }
//             }
//             cudaDeviceSynchronize();
//             checkCudaError(initialN);
//             if (h_abort) break;
//             // cudaMemcpy(&tail, task_pointers.d_tail_A, sizeof(unsigned int), cudaMemcpyDeviceToHost);
//             // if (tail == 0) break;
//             flip = !flip;
//         }
//         if(h_abort) break;
//     }
//     // printf("tailmax: %d\n", tail_max);
//     cudaMemset(task_pointers.d_tail_A, 0, sizeof(unsigned int));
//     cudaMemset(task_pointers.d_tail_B, 0, sizeof(unsigned int));
//     cudaMemset(task_pointers.d_tail_C, 0, sizeof(unsigned int));
// }

// With tiny task
void initializeBNB(int initialN, T_pointers &task_pointers, P_pointers plex_pointers, S_pointers subgraph_pointers, unsigned int *d_blk, unsigned int *d_left, unsigned int *d_blk_counter, unsigned int *d_left_counter, uint8_t *commonMtx, unsigned int *plex_count, uint16_t* bnb_neiInG, uint16_t* bnb_neiInP, uint16_t* d_sat, uint16_t* d_commons, uint32_t* d_uni, unsigned long long* cycles, uint32_t* d_adj, int* d_abort, unsigned int* global_count)
{
    cudaMemset(d_abort, 0, sizeof(int));
    int h_abort = 0;
    chkerr(cudaMemset(task_pointers.d_tail_B, 0, sizeof(unsigned int)));

    auto drainTinyFrontier = [&](Task* parent_tasks, Task* checkpoint_tasks, uint8_t* checkpoint_labels, uint16_t* checkpoint_neiInG, uint16_t* checkpoint_neiInP, unsigned int* checkpoint_tail, unsigned int checkpoint_cap) -> bool
    {
        while (true)
        {
            bool flip = false;
            while (true)
            {
                unsigned int *tail_in = flip ? task_pointers.d_tiny_tail_C : task_pointers.d_tiny_tail_B;
                unsigned int *tail_out = flip ? task_pointers.d_tiny_tail_B : task_pointers.d_tiny_tail_C;
                TinyTask *Q_in = flip ? task_pointers.d_tiny_tasks_C : task_pointers.d_tiny_tasks_B;
                TinyTask *Q_out = flip ? task_pointers.d_tiny_tasks_B : task_pointers.d_tiny_tasks_C;

                unsigned int tail = 0;
                cudaMemcpy(&tail, tail_in, sizeof(unsigned int), cudaMemcpyDeviceToHost);
                if (tail == 0)
                    break;
                cudaMemset(tail_out, 0, sizeof(unsigned int));
                unsigned int numTasks = tail;
                unsigned int waves = (numTasks + WARPS - 1) / WARPS;

                for (unsigned int w = 0; w < waves; w++)
                {
                    BNB<<<BLK_NUMS, BLK_DIM>>>(w, plex_pointers, subgraph_pointers, d_blk, d_left, d_blk_counter, d_left_counter, commonMtx, parent_tasks, Q_in, Q_out, task_pointers.d_tiny_tasks_A, numTasks, tail_out, task_pointers.d_tiny_tail_A, task_pointers.Delta, task_pointers.d_delta_tail, task_pointers.d_replay_stack, checkpoint_tasks, checkpoint_labels, checkpoint_neiInG, checkpoint_neiInP, checkpoint_tail, checkpoint_cap, plex_count, bnb_neiInG, bnb_neiInP, d_sat, d_commons, d_uni, cycles, d_adj, d_abort, global_count);
                    cudaMemcpy(&h_abort, d_abort, sizeof(int), cudaMemcpyDeviceToHost);
                    if (h_abort)
                    {
                        unsigned int h_tail_in = 0;
                        unsigned int h_tail_out = 0;
                        unsigned int h_tiny_overflow_tail = 0;
                        unsigned int h_delta_tail = 0;
                        unsigned int h_checkpoint_tail = 0;
                        cudaMemcpy(&h_tail_in, tail_in, sizeof(unsigned int), cudaMemcpyDeviceToHost);
                        cudaMemcpy(&h_tail_out, tail_out, sizeof(unsigned int), cudaMemcpyDeviceToHost);
                        cudaMemcpy(&h_tiny_overflow_tail, task_pointers.d_tiny_tail_A, sizeof(unsigned int), cudaMemcpyDeviceToHost);
                        cudaMemcpy(&h_delta_tail, task_pointers.d_delta_tail, sizeof(unsigned int), cudaMemcpyDeviceToHost);
                        cudaMemcpy(&h_checkpoint_tail, task_pointers.d_checkpoint_tail, sizeof(unsigned int), cudaMemcpyDeviceToHost);
                        if (h_abort == 1)
                            printf("Maximum TinyTask Capacity Reached on level %d\n", initialN);
                        else if (h_abort == 2)
                            printf("Maximum Delta Capacity Reached on level %d\n", initialN);
                        else if (h_abort == 3)
                            printf("Maximum Replay Stack Capacity Reached on level %d\n", initialN);
                        else if (h_abort == 4)
                            printf("Invalid Linked Delta Log on level %d\n", initialN);
                        else if (h_abort == 5)
                            printf("Maximum Checkpoint Task Capacity Reached on level %d\n", initialN);
                        else
                            printf("Maximum Capacity Reached on level %d\n", initialN);
                        printf("TinyTask tails: in=%u out=%u overflow=%u/%u, Delta=%u/%u, checkpoints=%u/%u\n",
                               h_tail_in,
                               h_tail_out,
                               h_tiny_overflow_tail,
                               (unsigned int)TINY_OVERFLOW_CAP,
                               h_delta_tail,
                               (unsigned int)DELTA_CAP,
                               h_checkpoint_tail,
                               (unsigned int)MAX_CAP);
                        break;
                    }
                }
                cudaDeviceSynchronize();
                checkCudaError(initialN);
                if (h_abort) break;
                flip = !flip;
            }
            if (h_abort) break;

            unsigned int tiny_tail = 0;
            cudaMemcpy(&tiny_tail, task_pointers.d_tiny_tail_A, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            if (tiny_tail == 0) break;

            unsigned int tiny_batch = std::min((unsigned)5*WARPS, tiny_tail);
            unsigned int tiny_head = tiny_tail - tiny_batch;
            chkerr(cudaMemcpy(task_pointers.d_tiny_tasks_B, task_pointers.d_tiny_tasks_A + tiny_head, tiny_batch * sizeof(TinyTask), cudaMemcpyDeviceToDevice));
            chkerr(cudaMemcpy(task_pointers.d_tiny_tail_B, &tiny_batch, sizeof(unsigned int), cudaMemcpyHostToDevice));
            chkerr(cudaMemset(task_pointers.d_tiny_tail_C, 0, sizeof(unsigned int)));

            tiny_tail = tiny_head;
            chkerr(cudaMemcpy(task_pointers.d_tiny_tail_A, &tiny_tail, sizeof(unsigned int), cudaMemcpyHostToDevice));
        }

        return h_abort == 0;
    };

    auto processTaskQueue = [&](Task* parent_tasks, unsigned int* parent_tail, Task* checkpoint_tasks, uint8_t* checkpoint_labels, uint16_t* checkpoint_neiInG, uint16_t* checkpoint_neiInP, unsigned int* checkpoint_tail, unsigned int checkpoint_cap) -> bool
    {
        while (true)
        {
            unsigned int tail = 0;
            cudaMemcpy(&tail, parent_tail, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            if (tail == 0) break;

            unsigned int batch = std::min((unsigned)5*WARPS, tail);
            unsigned int head = tail - batch;

            chkerr(cudaMemset(task_pointers.d_delta_tail, 0, sizeof(unsigned int)));
            chkerr(cudaMemset(task_pointers.d_tiny_tail_A, 0, sizeof(unsigned int)));
            chkerr(cudaMemset(task_pointers.d_tiny_tail_C, 0, sizeof(unsigned int)));

            unsigned int threads = 256;
            unsigned int blocks = (batch + threads - 1) / threads;
            seedInitialTinyTasks<<<blocks, threads>>>(parent_tasks, task_pointers.d_tiny_tasks_B, head, batch);
            cudaDeviceSynchronize();
            checkCudaError(initialN);

            chkerr(cudaMemcpy(task_pointers.d_tiny_tail_B, &batch, sizeof(unsigned int), cudaMemcpyHostToDevice));

            tail = head;
            cudaMemcpy(parent_tail, &tail, sizeof(tail), cudaMemcpyHostToDevice);

            if (!drainTinyFrontier(parent_tasks, checkpoint_tasks, checkpoint_labels, checkpoint_neiInG, checkpoint_neiInP, checkpoint_tail, checkpoint_cap)) return false;
        }

        return h_abort == 0;
    };


    if (processTaskQueue(task_pointers.d_tasks_A, task_pointers.d_tail_A, task_pointers.d_tasks_B, task_pointers.d_all_labels_B, task_pointers.d_all_neiInG_B, task_pointers.d_all_neiInP_B, task_pointers.d_tail_B, CHECKPOINT_TASK_CAP))
    {
        while (!h_abort)
        {
            unsigned int tail_B = 0;
            cudaMemcpy(&tail_B, task_pointers.d_tail_B, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            if (tail_B == 0) break;

            chkerr(cudaMemset(task_pointers.d_tail_A, 0, sizeof(unsigned int)));
            if (!processTaskQueue(task_pointers.d_tasks_B, task_pointers.d_tail_B, task_pointers.d_tasks_A, task_pointers.d_all_labels_A, task_pointers.d_all_neiInG_A, task_pointers.d_all_neiInP_A, task_pointers.d_tail_A, MAX_CAP)) break;

            unsigned int tail_A = 0;
            cudaMemcpy(&tail_A, task_pointers.d_tail_A, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            if (tail_A == 0) break;

            chkerr(cudaMemset(task_pointers.d_tail_B, 0, sizeof(unsigned int)));
            if (!processTaskQueue(task_pointers.d_tasks_A, task_pointers.d_tail_A, task_pointers.d_tasks_B, task_pointers.d_all_labels_B, task_pointers.d_all_neiInG_B, task_pointers.d_all_neiInP_B, task_pointers.d_tail_B, CHECKPOINT_TASK_CAP)) break;
        }
    }
    // printf("tailmax: %d\n", tail_max);
    cudaMemset(task_pointers.d_tail_A, 0, sizeof(unsigned int));
    cudaMemset(task_pointers.d_tail_B, 0, sizeof(unsigned int));
    cudaMemset(task_pointers.d_tiny_tail_A, 0, sizeof(unsigned int));
    cudaMemset(task_pointers.d_tiny_tail_B, 0, sizeof(unsigned int));
    cudaMemset(task_pointers.d_tiny_tail_C, 0, sizeof(unsigned int));
    cudaMemset(task_pointers.d_delta_tail, 0, sizeof(unsigned int));
    // cudaMemset(task_pointers.d_checkpoint_tail, 0, sizeof(unsigned int));
}

void initializeBNB2(int initialN, T_pointers &task_pointers, P_pointers plex_pointers, S_pointers subgraph_pointers, unsigned int *d_blk, unsigned int *d_left, unsigned int *d_blk_counter, unsigned int *d_left_counter, uint8_t *commonMtx, unsigned int *plex_count, uint16_t* d_sat, uint16_t* d_commons, uint32_t* d_uni, unsigned long long* cycles, uint32_t* d_adj, int* d_abort, HostTaskBuffer& hostBuf, HostTask* h_task_stage, unsigned int* state, unsigned int* res, unsigned int* recExcl, unsigned int* recCand)
{
    // cudaMemset(d_abort, 0, sizeof(int));
    // int h_abort = 0;
    unsigned int tail_max;
    cudaMemcpy(&tail_max, task_pointers.d_tail_A, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int head = 0;
    while (true)
    {
        unsigned int tail;
        // unsigned int plex;
        cudaMemcpy(&tail, task_pointers.d_tail_A, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        // cudaMemcpy(&plex, plex_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        // printf("tail: %u\n", tail);
        if (tail > tail_max)
            tail_max = tail;
        if (tail == 0)
            break;

        unsigned int batch;
        batch = std::min((unsigned)5*WARPS, tail);
        // else batch = std::min((unsigned)3*WARPS, tail);

        head = tail - batch;
        
        chkerr(cudaMemcpy(task_pointers.d_tail_B, &batch, sizeof(unsigned int), cudaMemcpyHostToDevice));
        chkerr(cudaMemcpy(task_pointers.d_tasks_B, task_pointers.d_tasks_A + head, batch * sizeof(Task), cudaMemcpyDeviceToDevice));
        chkerr(cudaMemcpy(task_pointers.d_all_labels_B, task_pointers.d_all_labels_A + head * MAX_BLK_SIZE, batch * MAX_BLK_SIZE * sizeof(uint8_t), cudaMemcpyDeviceToDevice));
        chkerr(cudaMemcpy(task_pointers.d_all_neiInG_B, task_pointers.d_all_neiInG_A + head * MAX_BLK_SIZE, batch * MAX_BLK_SIZE * sizeof(uint16_t), cudaMemcpyDeviceToDevice));
        chkerr(cudaMemcpy(task_pointers.d_all_neiInP_B, task_pointers.d_all_neiInP_A + head * MAX_BLK_SIZE, batch * MAX_BLK_SIZE * sizeof(uint16_t), cudaMemcpyDeviceToDevice));

        tail = head;
        cudaMemcpy(task_pointers.d_tail_A, &tail, sizeof(tail), cudaMemcpyHostToDevice);
        bool flip = false;

        while (true)
        {
            unsigned int *tail_in = flip ? task_pointers.d_tail_C : task_pointers.d_tail_B;
            unsigned int *tail_out = flip ? task_pointers.d_tail_B : task_pointers.d_tail_C;
            Task *Q_in = flip ? task_pointers.d_tasks_C : task_pointers.d_tasks_B;
            Task *Q_out = flip ? task_pointers.d_tasks_B : task_pointers.d_tasks_C;
            uint8_t *lab_out = flip ? task_pointers.d_all_labels_B : task_pointers.d_all_labels_C;
            uint16_t *nei_out = flip ? task_pointers.d_all_neiInG_B : task_pointers.d_all_neiInG_C;
            uint16_t *P_out = flip ? task_pointers.d_all_neiInP_B : task_pointers.d_all_neiInP_C;

            cudaMemcpy(&tail, tail_in, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            // printf("tail inside: %d\n", tail);
            if (tail == 0)
                break;
            cudaMemset(tail_out, 0, sizeof(unsigned int));
            unsigned int numTasks = tail;
            unsigned int waves = (numTasks) / WARPS + 1;

            for (unsigned int w = 0; w < waves; w++)
            {
                BNB2<<<BLK_NUMS, BLK_DIM>>>(w, plex_pointers, subgraph_pointers, d_blk, d_left, d_blk_counter, d_left_counter, commonMtx, Q_in, Q_out, task_pointers.d_tasks_A, numTasks, 0, tail_out, task_pointers.d_tail_A, lab_out, nei_out, P_out, task_pointers.d_all_labels_A, task_pointers.d_all_neiInG_A, task_pointers.d_all_neiInP_A, plex_count, d_sat, d_commons, d_uni, cycles, d_adj, d_abort, state, res, recExcl, recCand);
            }
            cudaDeviceSynchronize();
            checkCudaError(initialN);
            flip = !flip;
        }
    }
    printf("tailmax: %d\n", tail_max);
    // cudaMemset(task_pointers.d_tail_A, 0, sizeof(unsigned int));
    // cudaMemset(task_pointers.d_tail_B, 0, sizeof(unsigned int));
    // cudaMemset(task_pointers.d_tail_C, 0, sizeof(unsigned int));
}

inline int find_pos_sorted(unsigned int* neighbors, unsigned int* offsets, unsigned int u, unsigned int v)
{
    unsigned int b = offsets[u];
    unsigned int e = offsets[u+1];
    auto* first = neighbors + b;
    auto* end = neighbors + e;

    auto* it = std::lower_bound(first, end, v);
    if (it == end || *it != v) return -1;
    return int(it - neighbors);
}

void truss_peeling(unsigned int* neighbors, unsigned int* offsets, unsigned int* degrees, vector<pair<int, int>> &Q_e, int n, int* m)
{
    while(!Q_e.empty())
    {
        auto [u, v] = Q_e.back();
        Q_e.pop_back();

        // printf("u : %d, v: %d\n", u, v);

        int begin = offsets[u];
        int end = offsets[u+1];
        int degree = degrees[u];

        // printf("u begin : %d, end: %d, degree: %d\n", begin, end, degree);

        // int pos = -1;
        // for (int i = 0; i < degree && (begin+i) < m[0]; i++)
        // { 
        //     if (neighbors[begin+i] == v)
        //     {
        //         pos = begin+i;
        //         break;
        //     }
        // }

        int pos = find_pos_sorted(neighbors, offsets, u, v);

        // printf("pos: %d\n", pos);

        if (pos != -1)
        {
            for (int i = pos; i < m[0] - 1; i++)
            {
                neighbors[i] = neighbors[i+1];
            }
            m[0] -= 1;

            for (int i = u+1; i <= n; i++)
            {
                offsets[i] -= 1;
            }
            degrees[u]--;
        }
    }
}

void fast_truss_peeling(unsigned int* neighbors, unsigned int* offsets, unsigned int* degrees, vector<pair<int, int>> &Q_e, int n, int* m, vector<int> &triangles)
{
    std::vector<uint8_t> dead(m[0], 0);
    while(!Q_e.empty())
    {
        auto [u, v] = Q_e.back();
        Q_e.pop_back();

        // printf("u : %d, v: %d\n", u, v);

        int pos_uv = find_pos_sorted(neighbors, offsets, u, v);
        int pos_vu = find_pos_sorted(neighbors, offsets, v, u);
        if (pos_uv >= 0 && !dead[pos_uv])
        {
            dead[pos_uv] = 1;
            degrees[u]--;
        }
        if (pos_vu >= 0 && !dead[pos_vu])
        {
            dead[pos_vu] = 1;
            degrees[v]--;
        }

        unsigned int iu = offsets[u], eu = offsets[u+1];
        unsigned int iv = offsets[v], ev = offsets[v+1];

        while (iu < eu && iv < ev)
        {
            while (iu < eu && dead[iu]) ++iu;
            while (iv < ev && dead[iv]) ++iv;

            if (iu >= eu || iv >= ev) break;

            unsigned int a = neighbors[iu];
            unsigned int b = neighbors[iv];

            if (a == b)
            {
                unsigned int w = a;

                int pos_uw = iu;
                int pos_vw = iv;

                if (!dead[pos_uw])
                {
                    int prev = triangles[pos_uw];
                    if (prev > 0)
                    {
                        int now = prev - 1;
                        triangles[pos_uw] = now;
                        if (prev == (lb - 2 * k)){
                            Q_e.emplace_back(u, w);
                        }
                    }
                }

                if (!dead[pos_vw])
                {
                    int prev = triangles[pos_vw];
                    if (prev > 0)
                    {
                        int now = prev - 1;
                        triangles[pos_vw] = now;
                        if (prev == (lb - 2 * k))
                        {
                            Q_e.emplace_back(v, w);
                        }
                    }
                }
                iu++;
                iv++;
            }
            else if (a < b) iu++;
            else iv++;
        }
    }

    unsigned int write = 0;
    for (int u = 0; u < n; u++)
    {
        unsigned int b = offsets[u];
        unsigned int e = offsets[u+1];
        offsets[u] = write;
        for (unsigned int i = b; i < e; i++)
        {
            if (!dead[i])
            {
                neighbors[write] = neighbors[i];
                triangles[write] = triangles[i];
                ++write;
            }
        }
    }
    offsets[n] = write;
    m[0] = write;
    triangles.resize(write);
}

void fast_truss_peeling_parallel(unsigned int* neighbors, unsigned int* offsets, unsigned int* degrees, vector<pair<int, int>> &Q_e, int n, int* m, vector<int> &triangles)
{
    const int M = m[0];
    std::vector<int> dead(M, 0);

    std::vector<std::pair<int, int>> curr_frontier = Q_e;
    std::vector<std::pair<int, int>> next_frontier;

    Q_e.clear();

    while(!curr_frontier.empty())
    {
        // int nthreads = 1;
        // #pragma omp parallel{
        //     #pragma omp single
        //     nthreads = omp_get_num_threads();
        // }
        int nthreads = omp_get_max_threads();
        std::vector<std::vector<std::pair<int, int>>> tls_next(nthreads);

        #pragma omp parallel for
        for (int idx = 0; idx < curr_frontier.size(); idx++)
        {
            const int tid = omp_get_thread_num();
            auto [u, v] = curr_frontier[idx];

            int pos_uv = find_pos_sorted(neighbors, offsets, u, v);
            if (pos_uv >= 0)
            {
                int was = 0;
                #pragma omp atomic capture
                {
                    was = dead[pos_uv];
                    dead[pos_uv] = 1;
                }
                if (!was)
                {
                    #pragma omp atomic
                    degrees[u]--;
                }
            }

            int pos_vu = find_pos_sorted(neighbors, offsets, v, u);
            if (pos_vu >= 0)
            {
                int was = 0;
                #pragma omp atomic capture
                {
                    was = dead[pos_vu];
                    dead[pos_vu] = 1;
                }
                if (!was)
                {
                    #pragma omp atomic
                    degrees[v]--;
                }
            }

            unsigned int iu = offsets[u];
            unsigned int eu = offsets[u+1];
            unsigned int iv = offsets[v];
            unsigned int ev = offsets[v+1];
            // Reconsider it
            while (iu < eu && iv < ev)
            {
                while (iu < eu && dead[iu]) iu++;
                while (iv < ev && dead[iv]) iv++;

                if (iu >= eu || iv >= ev) break;

                unsigned int a = neighbors[iu];
                unsigned int b = neighbors[iv];

                if (a == b)
                {
                    unsigned int w = a;

                    int pos_uw = iu;
                    int pos_vw = iv;

                    if (!dead[pos_uw])
                    {
                        int prev = 0;
                        #pragma omp atomic capture
                        {
                            prev = triangles[pos_uw];
                            triangles[pos_uw] = prev - 1;
                        }
                        if (prev > 0 && prev == (lb - 2 * k))
                        {
                            tls_next[tid].emplace_back(u, w);
                        }
                    }

                    if (!dead[pos_vw])
                    {
                        int prev = 0;
                        #pragma omp atomic capture
                        {
                            prev = triangles[pos_vw];
                            triangles[pos_vw] = prev - 1;
                        }
                        if (prev > 0 && prev == (lb - 2 * k))
                        {
                            tls_next[tid].emplace_back(v, w);
                        }
                    }

                    iu++;
                    iv++;
                }
                else if (a < b) iu++;
                else iv++;
            }
        }

        next_frontier.clear();
        size_t total = 0;
        for (auto &v: tls_next) total += v.size();
        next_frontier.reserve(total);
        for(auto &v: tls_next)
        {
            next_frontier.insert(next_frontier.end(), std::make_move_iterator(v.begin()), std::make_move_iterator(v.end()));
        }
        curr_frontier.swap(next_frontier);
    }

    unsigned int write = 0;
    for (int u = 0; u < n; u++)
    {
        unsigned int b = offsets[u];
        unsigned int e = offsets[u+1];
        offsets[u] = write;
        for (unsigned int i = b; i < e; i++)
        {
            if (!dead[i])
            {
                neighbors[write] = neighbors[i];
                triangles[write] = triangles[i];
                write++;
            }
        }
    }
    offsets[n] = write;
    m[0] = write;
    triangles.resize(write);
}




void initHostTaskBuffer(HostTaskBuffer &buf, unsigned int capacity)
{
    buf.capacity = capacity;
    buf.size = 0;
    // cudaHostAlloc(&buf.tasks, capacity * sizeof(HostTask), cudaHostAllocDefault);
    buf.tasks = new HostTask[capacity];
}



void decomposableSearch(const graph<int> &g)
{
    int *dpos = new int[g.n];
    int *dseq = new int[g.n];
    bool *mark = new bool[g.n];
    int *resNei = new int[g.n];
    omp_set_num_threads(16);
    #pragma omp parallel for
    for (int i = 0; i < g.n; ++i)
        dpos[i] = INT_MAX;
    unsigned int *validblk;
    unsigned int h_validblk;

    float time_0 = 0;
    // float time_1 = 0;
    // float time_2 = 0;
    // float time_3 = 0;
    // float time_4 = 0;
    // float time_5 = 0;
    // float time_6 = 0;
    // float time_7 = 0;
    cudaEvent_t event_start;
    cudaEvent_t event_stop;
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);
    cudaEventRecord(event_start);

    graph<intT> peelG = peelGraph(g, mark, resNei);
    int pn = peelG.n;
    volatile bool *const ready = (volatile bool *)mark;

    #pragma omp parallel for
    for (int i = 0; i < pn; ++i)
        mark[i] = false;

/// K-Truss Logic

// printf("n: %d, m: %d, q: %d, k: %d\n", pn, peelG.m, lb, k);
    // cudaEventRecord(event_start);
    if (truss)
    {
    vector<int> triangles(peelG.m, 0);
    vector<int> counts(pn, 0);

    #pragma omp parallel for
    for (int i = 0; i < pn; i++)
    {
        const int begin = peelG.offsets[i];
        const int end = peelG.offsets[i+1];
        for (int p = begin; p < end; p++)
        {
            const int v = peelG.neighbors[p];
            const int b2 = peelG.offsets[v];
            const int e2 = peelG.offsets[v+1];

            unsigned int iu = begin, iv = b2;
            int common = 0;
            while (iu < end && iv < e2)
            {
                unsigned int a = peelG.neighbors[iu];
                unsigned int b = peelG.neighbors[iv];
                if (a == b)
                {
                    common++;
                    iu++;
                    iv++;
                }
                else if (a < b) iu++;
                else iv++;
            }
            triangles[p] = common;
            if (i < v && common < (lb - 2 * k))
            {
                counts[i]++;
            }
        }
    }

    vector<int> w(pn+1, 0);
    for (int i = 0; i < pn; i++) w[i+1] = w[i] + counts[i];

    vector<pair<int, int>> Q_e(w[pn]);

    #pragma omp parallel for
    for (int i = 0; i < pn; i++)
    {
        int u = w[i];
        int begin = peelG.offsets[i];
        int end = peelG.offsets[i+1];
        for (int p = begin; p < end; p++)
        {
            int v = peelG.neighbors[p];
            if (i < v && triangles[p] < (lb - 2 * k))
            {
                Q_e[u++] = {i, v};
            }
        }
    }

// fast_truss_peeling_parallel(peelG.neighbors, peelG.offsets, peelG.degree, Q_e, pn, &peelG.m, triangles);
fast_truss_peeling(peelG.neighbors, peelG.offsets, peelG.degree, Q_e, pn, &peelG.m, triangles);
peelG = peelGraph(peelG, mark, resNei);
pn = peelG.n;
    }

// #pragma omp master
//     {
        ListLinearHeap<int> *linear_heap = new ListLinearHeap<int>(pn, pn - 1);
        linear_heap->init(pn, pn - 1);
        for (int i = 0; i < pn; ++i)
        {
            linear_heap->insert(i, peelG.degree[i]);
        }
        for (int i = 0; i < pn; i++)
        {
            int u, key;
            linear_heap->pop_min(u, key);
            dpos[u] = i;
            dseq[i] = u;
            ready[i] = true;
            for (int j = 0; j < peelG.degree[u]; j++)
            {
                const int nei = peelG.neighbors[peelG.offsets[u] + j];
                if (dpos[nei] == INT_MAX)
                {
                    linear_heap->decrement(nei);
                }
            }
        }
        delete linear_heap;
    // }

    cudaEventRecord(event_stop);
    cudaEventSynchronize(event_stop);
    float time_milli_sec = 0;
    cudaEventElapsedTime(&time_milli_sec, event_start, event_stop);
    time_0 += time_milli_sec;

    P_pointers plex_pointers;
    plex_pointers.k = k;
    plex_pointers.lb = lb;
    plex_pointers.bd = bd;
    plex_pointers.thres = thres;

    G_pointers graph_pointers;
    D_pointers degen_pointers;
    S_pointers subgraph_pointers;
    T_pointers task_pointers;

    printf("Start copying graph to GPU....\n");
    copy_graph_to_gpu<intT>(peelG, dpos, dseq, graph_pointers, degen_pointers, subgraph_pointers);
    printf("Done copying graph to GPU....\n");

    unsigned int *d_blk;
    unsigned int *d_blk_counter;
    unsigned int *d_left;
    unsigned int *d_left_counter;
    uint32_t *d_visited;
    unsigned int *d_hopSz;
    unsigned int *global_count;
    unsigned int h_global;
    unsigned int *left_count;
    unsigned int *plex_count;
    uint8_t *commonMtx;
    unsigned int h_plex_count;

    uint16_t *d_sat;
    uint16_t *d_commons;
    uint32_t *d_uni;
    uint32_t *d_adj;

    unsigned int* d_res;
    unsigned int* d_res2;
    unsigned int* d_br;
    unsigned int* d_state;
    unsigned int* d_state2;
    unsigned int* d_v2delete;
    unsigned int* d_len;
    unsigned int* d_sz;

    unsigned int* recCand1;
    unsigned int* recCand2;
    unsigned int* recExcl;
    unsigned int* recCand;

    uint16_t* neiInG;
    uint16_t* neiInP; 
    uint16_t* bnb_neiInG;
    uint16_t* bnb_neiInP;

    unsigned long long* cycles;
    int *d_abort_flag = nullptr;
    int* d_abort2 = nullptr;
    int* d_abort3 = nullptr;


    cudaMalloc(&d_res, WARPS * MAX_DEPTH * sizeof(unsigned int));
    cudaMalloc(&d_res2, WARPS * MAX_DEPTH * sizeof(unsigned int));
    cudaMalloc(&d_br, WARPS * MAX_DEPTH * sizeof(unsigned int));
    cudaMalloc(&d_state, WARPS * MAX_DEPTH * sizeof(unsigned int));
    cudaMalloc(&d_state2, WARPS * MAX_DEPTH * sizeof(unsigned int));
    cudaMalloc(&d_v2delete, WARPS * MAX_DEPTH * sizeof(unsigned int));
    cudaMalloc(&d_len, WARPS * sizeof(unsigned int));
    cudaMalloc(&d_sz, WARPS * sizeof(unsigned int));

    cudaMalloc(&recCand1, WARPS * MAX_BLK_SIZE * sizeof(unsigned int));
    cudaMalloc(&recCand2, WARPS * MAX_BLK_SIZE * sizeof(unsigned int));
    cudaMalloc(&recExcl, WARPS * MAX_BLK_SIZE * sizeof(unsigned int));
    cudaMalloc(&recCand, WARPS * MAX_BLK_SIZE * sizeof(unsigned int));

    cudaMalloc(&neiInG, WARPS * MAX_BLK_SIZE * sizeof(uint16_t));
    cudaMalloc(&neiInP, WARPS * MAX_BLK_SIZE * sizeof(uint16_t));

    cudaMalloc(&bnb_neiInG, WARPS * MAX_BLK_SIZE * sizeof(uint16_t));
    cudaMalloc(&bnb_neiInP, WARPS * MAX_BLK_SIZE * sizeof(uint16_t));

    cudaMalloc(&d_sat, WARPS * MAX_BLK_SIZE * sizeof(uint16_t));
    cudaMalloc(&d_commons, WARPS * MAX_BLK_SIZE * sizeof(uint16_t));
    cudaMalloc(&d_uni, WARPS * 32 * sizeof(uint32_t));
    cudaMalloc(&d_adj, ADJSIZE * WARPS * sizeof(uint32_t));

    thrust::device_ptr<unsigned int> deg_ptr(subgraph_pointers.degree);
    thrust::device_ptr<unsigned int> off_ptr(subgraph_pointers.offsets);

    cudaMalloc(&global_count, sizeof(unsigned int));
    cudaMemset(global_count, 0, sizeof(unsigned int));

    cudaMalloc(&left_count, sizeof(unsigned int));
    cudaMemset(left_count, 0, sizeof(unsigned int));

    cudaMalloc(&plex_count, sizeof(unsigned int));
    cudaMemset(plex_count, 0, sizeof(unsigned int));

    cudaMalloc(&validblk, sizeof(unsigned int));
    cudaMemset(validblk, 0, sizeof(unsigned int));
    
    cudaMalloc(&d_blk, MAX_BLK_SIZE * WARPS * sizeof(unsigned int));
    cudaMalloc(&d_blk_counter, WARPS * sizeof(unsigned int));

    cudaMalloc(&d_left, MAX_BLK_SIZE * WARPS * sizeof(unsigned int));
    cudaMalloc(&d_left_counter, WARPS * sizeof(unsigned int));
    cudaMalloc(&d_hopSz, WARPS * sizeof(unsigned int));
    size_t totalBytes = size_t(WARPS) * CAP * sizeof(uint8_t);
    cudaMalloc(&commonMtx, totalBytes);

    int range = (pn/32)+1;
    //printf("Range: %d\n", range);
    cudaMalloc(&d_visited, range * WARPS * sizeof(uint32_t));
    cudaMalloc(&cycles, BLK_NUMS * sizeof(unsigned long long));
    //cudaMalloc(&d_count, pn * WARPS * sizeof(unsigned int));

    cudaMemset(d_blk_counter, 0, WARPS * sizeof(unsigned int));
    cudaMemset(d_left_counter, 0, WARPS * sizeof(unsigned int));
    cudaMemset(d_hopSz, 0, WARPS * sizeof(unsigned int));
    cudaMemset(d_visited, 0, range * WARPS * sizeof(uint32_t));
    //cudaMemset(d_count, 0, pn * WARPS * sizeof(uint16_t));
    cudaMemset(commonMtx, 0, totalBytes);
    cudaMemset(recCand1, 0, WARPS * MAX_BLK_SIZE * sizeof(unsigned int));
    cudaMemset(recCand2, 0, WARPS * MAX_BLK_SIZE * sizeof(unsigned int));
    cudaMemset(recExcl, 0, WARPS * MAX_BLK_SIZE * sizeof(unsigned int));
    cudaMemset(recCand, 0, WARPS * MAX_BLK_SIZE * sizeof(unsigned int));
    cudaMemset(d_uni, 0, WARPS * 32 * sizeof(uint32_t));
    cudaMemset(d_adj, 0, ADJSIZE * WARPS * sizeof(uint32_t));
    cudaMemset(cycles, 0, 40 * sizeof(unsigned long long));

    size_t capacity = MAX_CAP;

    chkerr(cudaMalloc(&task_pointers.d_tasks_A, capacity * sizeof(Task)));
    chkerr(cudaMalloc(&task_pointers.d_all_labels_A, capacity * MAX_BLK_SIZE * sizeof(uint8_t)));
    chkerr(cudaMalloc(&task_pointers.d_all_neiInG_A, capacity * MAX_BLK_SIZE * sizeof(uint16_t)));
    chkerr(cudaMalloc(&task_pointers.d_all_neiInP_A, capacity * MAX_BLK_SIZE * sizeof(uint16_t)));
    chkerr(cudaMalloc(&task_pointers.d_tail_A, sizeof(unsigned int)));
    chkerr(cudaMemset(task_pointers.d_tail_A, 0, sizeof(unsigned int)));

    capacity = TINY_OVERFLOW_CAP;
    chkerr(cudaMalloc(&task_pointers.d_tiny_tasks_A, capacity * sizeof(TinyTask)));
    chkerr(cudaMalloc(&task_pointers.d_tiny_tail_A, sizeof(unsigned int)));
    chkerr(cudaMemset(task_pointers.d_tiny_tail_A, 0, sizeof(unsigned int)));
    chkerr(cudaMalloc(&task_pointers.Delta, (size_t)DELTA_CAP * sizeof(BranchLog)));
    chkerr(cudaMalloc(&task_pointers.d_delta_tail, sizeof(unsigned int)));
    chkerr(cudaMemset(task_pointers.d_delta_tail, 0, sizeof(unsigned int)));
    chkerr(cudaMalloc(&task_pointers.d_replay_stack, (size_t)WARPS * REPLAY_STACK_CAP * sizeof(unsigned int)));
    chkerr(cudaMalloc(&task_pointers.d_checkpoint_tail, sizeof(unsigned int)));
    chkerr(cudaMemset(task_pointers.d_checkpoint_tail, 0, sizeof(unsigned int)));

    size_t oneTask = sizeof(Task) + MAX_BLK_SIZE * sizeof(uint8_t) + 2 * MAX_BLK_SIZE * sizeof(uint16_t) + 2 * sizeof(unsigned int);
    // printf("One task takes %zu memory\n", oneTask);

    size_t capacity2 = TINY_FRONTIER_CAP;
    size_t checkpoint_capacity = CHECKPOINT_TASK_CAP;
    chkerr(cudaMalloc(&task_pointers.d_tasks_B, checkpoint_capacity * sizeof(Task)));
    chkerr(cudaMalloc(&task_pointers.d_all_labels_B, checkpoint_capacity * MAX_BLK_SIZE * sizeof(uint8_t)));
    chkerr(cudaMalloc(&task_pointers.d_all_neiInG_B, checkpoint_capacity * MAX_BLK_SIZE * sizeof(uint16_t)));
    chkerr(cudaMalloc(&task_pointers.d_all_neiInP_B, checkpoint_capacity * MAX_BLK_SIZE * sizeof(uint16_t)));
    chkerr(cudaMalloc(&task_pointers.d_tail_B, sizeof(unsigned int)));

    chkerr(cudaMalloc(&task_pointers.d_tiny_tasks_B, capacity2 * sizeof(TinyTask)));
    chkerr(cudaMalloc(&task_pointers.d_tiny_tail_B, sizeof(unsigned int)));
    chkerr(cudaMemset(task_pointers.d_tiny_tail_B, 0, sizeof(unsigned int)));

    // cudaMalloc(&task_pointers.d_tasks_C, capacity2 * sizeof(Task));
    // cudaMalloc(&task_pointers.d_all_labels_C, capacity2 * MAX_BLK_SIZE * sizeof(uint8_t));
    // cudaMalloc(&task_pointers.d_all_neiInG_C, capacity2 * MAX_BLK_SIZE * sizeof(uint16_t));
    // cudaMalloc(&task_pointers.d_all_neiInP_C, capacity2 * MAX_BLK_SIZE * sizeof(uint16_t));
    // cudaMalloc(&task_pointers.d_tail_C, sizeof(unsigned int));

    chkerr(cudaMalloc(&task_pointers.d_tiny_tasks_C, capacity2 * sizeof(TinyTask)));
    chkerr(cudaMalloc(&task_pointers.d_tiny_tail_C, sizeof(unsigned int)));
    chkerr(cudaMemset(task_pointers.d_tiny_tail_C, 0 , sizeof(unsigned int)));

    // allocate_tiny_task_queues(task_pointers);

    cudaMalloc(&d_abort_flag, sizeof(int));
    cudaMalloc(&d_abort2, sizeof(int));
    cudaMalloc(&d_abort3, sizeof(int));
    cudaMemset(d_abort_flag, 0, sizeof(int));
    cudaMemset(d_abort2, 0, sizeof(int));

    // unsigned int* d_resume_checked;
    // unsigned int* d_resume_errors;

    // chkerr(cudaMalloc(&d_resume_checked, sizeof(unsigned int)));
    // chkerr(cudaMalloc(&d_resume_errors, sizeof(unsigned int)));

    graph<intT> subg;

    HostTaskBuffer buf;
    // initHostTaskBuffer(buf, 10*MAX_CAP);
    
    HostTask* h_task_stage = nullptr;
    // cudaHostAlloc(&h_task_stage, STAGING_CHUNK * sizeof(HostTask), cudaHostAllocDefault);

    cudaEventRecord(event_start);
    

    // unsigned long long* h_cycles;
    // h_cycles = (unsigned long long *)malloc(40 * sizeof(unsigned long long));
    // printf("Total Iterations: %d\n", (pn/WARPS)+1);
    int h_abort = 1;

    // Total nodes = 27000, warps = 4454, 

    unsigned int tail_max = 0;

    for (int i = 0; i < (pn/WARPS)+1; i++)
    {
        // printf("Iteration: %d/%d\n", i+1, (pn/WARPS)+1);
        decompose<<<BLK_NUMS, BLK_DIM>>>(i, plex_pointers, graph_pointers, degen_pointers, d_blk, d_blk_counter, d_left, d_left_counter, d_visited, global_count, left_count, validblk, d_hopSz, cycles);
        cudaDeviceSynchronize();
        checkCudaError(0);

        // cudaEventRecord(event_stop);
        // cudaEventSynchronize(event_stop);
        // float time_milli_sec = 0;
        // cudaEventElapsedTime(&time_milli_sec, event_start, event_stop);
        // time_1 += time_milli_sec;
        // cudaEventRecord(event_start);

        calculateDegrees<<<BLK_NUMS, BLK_DIM>>>(i, plex_pointers, graph_pointers, subgraph_pointers, d_blk, d_blk_counter, d_left, d_left_counter, global_count, left_count);
        cudaDeviceSynchronize();
        checkCudaError(1);

        // cudaEventRecord(event_stop);
        // cudaEventSynchronize(event_stop);
        // time_milli_sec = 0;
        // cudaEventElapsedTime(&time_milli_sec, event_start, event_stop);
        // time_2 += time_milli_sec;
        // cudaEventRecord(event_start);

        computeOffsets(subgraph_pointers, d_blk_counter);
        cudaDeviceSynchronize();
        checkCudaError(2);

        // cudaEventRecord(event_stop);
        // cudaEventSynchronize(event_stop);
        // time_milli_sec = 0;
        // cudaEventElapsedTime(&time_milli_sec, event_start, event_stop);
        // time_3 += time_milli_sec;
        // cudaEventRecord(event_start);

        fillNeighbors<<<BLK_NUMS, BLK_DIM>>>(i, subgraph_pointers, plex_pointers, graph_pointers, d_blk, d_blk_counter, d_left, d_left_counter, d_hopSz, commonMtx, d_adj);
        cudaDeviceSynchronize();
        checkCudaError(3);

        // cudaEventRecord(event_stop);
        // cudaEventSynchronize(event_stop);
        // time_milli_sec = 0;
        // cudaEventElapsedTime(&time_milli_sec, event_start, event_stop);
        // time_4 += time_milli_sec;
        // cudaEventRecord(event_start);

        buildCommonMtx<<<BLK_NUMS, BLK_DIM>>>(i, plex_pointers, subgraph_pointers, graph_pointers, commonMtx, d_hopSz);
        cudaDeviceSynchronize();
        checkCudaError(4);

        // cudaEventRecord(event_stop);
        // cudaEventSynchronize(event_stop);
        // time_milli_sec = 0;
        // cudaEventElapsedTime(&time_milli_sec, event_start, event_stop);
        // time_1 += time_milli_sec;
        // cudaEventRecord(event_start);

        
        cudaMemset(d_abort_flag, 0, sizeof(int));
        h_abort = 1;
        while(h_abort)
        {
            cudaMemset(d_abort2, 0, sizeof(int));
            kSearch<<<BLK_NUMS, BLK_DIM>>>(i, plex_pointers, subgraph_pointers, graph_pointers, task_pointers, d_blk_counter, d_res, d_br, d_state, d_len, d_sz, neiInG, neiInP, plex_count, commonMtx, recCand1, recCand2, d_v2delete, d_adj, cycles, d_abort2, d_abort_flag, global_count);
            cudaDeviceSynchronize();
            checkCudaError(5);
            cudaMemcpy(&h_abort, d_abort2, sizeof(int), cudaMemcpyDeviceToHost);

            // cudaEventRecord(event_stop);
            // cudaEventSynchronize(event_stop);
            // time_milli_sec = 0;
            // cudaEventElapsedTime(&time_milli_sec, event_start, event_stop);
            // time_2 += time_milli_sec;
            // cudaEventRecord(event_start);

        //     unsigned int *d_debug_expanded, *d_debug_spilled;
        //     cudaMalloc(&d_debug_expanded, sizeof(unsigned int));
        //     cudaMalloc(&d_debug_spilled, sizeof(unsigned int));
        //     cudaMemset(d_debug_expanded, 0, sizeof(unsigned int));
        //     cudaMemset(d_debug_spilled, 0, sizeof(unsigned int));

        //     unsigned int h_debug_tail_A = 0;

        //     chkerr(cudaMemcpy(&h_debug_tail_A,
        //           task_pointers.d_tail_A,
        //           sizeof(unsigned int),
        //           cudaMemcpyDeviceToHost));

        //     unsigned int debug_tasks = min(h_debug_tail_A, 64u);

        //     chkerr(cudaMemset(task_pointers.d_tiny_tail_A, 0, sizeof(unsigned int)));
        //     chkerr(cudaMemset(task_pointers.d_delta_tail_A, 0, sizeof(unsigned int)));

        //     printf("Launching BNB_localDFS_debug with %u tasks\n", debug_tasks);
        //     fflush(stdout);

        //     BNB_localDFS_debug<<<debug_tasks, 32>>>(
        //         subgraph_pointers,
        //         task_pointers.d_tasks_A,
        //         debug_tasks,
        //         0,
        //         task_pointers.d_tiny_tasks_A,
        //         task_pointers.d_tiny_tail_A,
        //         TINY_SMALL_CAP,
        //         task_pointers.d_delta_log_A,
        //         task_pointers.d_delta_tail_A,
        //         DELTA_SMALL_CAP,
        //         d_debug_expanded,
        //         d_debug_spilled
        //     );

        //     cudaError_t launch_err = cudaGetLastError();
        //     if (launch_err != cudaSuccess) {
        //         printf("BNB_localDFS_debug launch error: %s\n",
        //             cudaGetErrorString(launch_err));
        //         fflush(stdout);
        //         exit(1);
        //     }

        //     cudaError_t sync_err = cudaDeviceSynchronize();
        //     if (sync_err != cudaSuccess) {
        //         printf("BNB_localDFS_debug sync error: %s\n",
        //             cudaGetErrorString(sync_err));
        //         fflush(stdout);
        //         exit(1);
        //     }

        //     printf("Finished BNB_localDFS_debug\n");
        //     fflush(stdout);

        //     unsigned int h_debug_expanded = 0;
        //     unsigned int h_debug_spilled = 0;

        //     cudaMemcpy(&h_debug_expanded, d_debug_expanded, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        //     cudaMemcpy(&h_debug_spilled, d_debug_spilled, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        //    unsigned int h_tiny_tail_A = 0;
        //     unsigned int h_delta_tail_A = 0;

        //     chkerr(cudaMemcpy(&h_tiny_tail_A,
        //                     task_pointers.d_tiny_tail_A,
        //                     sizeof(unsigned int),
        //                     cudaMemcpyDeviceToHost));

        //     chkerr(cudaMemcpy(&h_delta_tail_A,
        //                     task_pointers.d_delta_tail_A,
        //                     sizeof(unsigned int),
        //                     cudaMemcpyDeviceToHost));

        //     printf("BNB_localDFS_debug tasks=%u expanded=%u spilled=%u tiny_tail=%u delta_tail=%u\n",
        //         debug_tasks,
        //         h_debug_expanded,
        //         h_debug_spilled,
        //         h_tiny_tail_A,
        //         h_delta_tail_A);

        //     if (h_tiny_tail_A > 0) {
        //         TinyTask h_tt;

        //         cudaMemcpy(&h_tt,
        //                 task_pointers.d_tiny_tasks_A,
        //                 sizeof(TinyTask),
        //                 cudaMemcpyDeviceToHost);

        //         printf("First TinyTask: idx=%d parent=%u plex=%u cand=%u excl=%u log_off=%u log_len=%u depth=%u hash=%llu\n",
        //                 h_tt.idx,
        //                 h_tt.parent_task_pos,
        //                 h_tt.plex_sz,
        //                 h_tt.cand_sz,
        //                 h_tt.excl_sz,
        //                 h_tt.log_off,
        //                 h_tt.log_len,
        //                 h_tt.depth,
        //                 (unsigned long long)h_tt.state_hash);

        //         Delta h_delta[LOCAL_BNB_DEPTH];

        //         cudaMemcpy(h_delta,
        //                 task_pointers.d_delta_log_A + h_tt.log_off,
        //                 h_tt.log_len * sizeof(Delta),
        //                 cudaMemcpyDeviceToHost);

        //         for (unsigned int i = 0; i < h_tt.log_len; ++i) {
        //             printf("  delta[%u]: v=%u old=%u new=%u neiInP_delta=%d neiInG_delta=%d\n",
        //                 i,
        //                 h_delta[i].v,
        //                 h_delta[i].old_label,
        //                 h_delta[i].new_label,
        //                 h_delta[i].neiInP_delta,
        //                 h_delta[i].neiInG_delta);
        //         }
        //     }

        //     chkerr(cudaMemset(d_resume_checked, 0, sizeof(unsigned int)));
        //     chkerr(cudaMemset(d_resume_errors, 0, sizeof(unsigned int)));

        //     if (h_tiny_tail_A > 0) {
        //         printf("Launching resumeTinyTasks_debug with %u tiny tasks\n", h_tiny_tail_A);

        //         resumeTinyTasks_debug<<<h_tiny_tail_A, 32>>>(
        //             subgraph_pointers,
        //             task_pointers.d_tasks_A,
        //             task_pointers.d_tiny_tasks_A,
        //             h_tiny_tail_A,
        //             task_pointers.d_delta_log_A,
        //             d_resume_checked,
        //             d_resume_errors
        //         );

        //         chkerr(cudaDeviceSynchronize());
        //         checkCudaError(777);
        //     }

        //     unsigned int h_resume_checked = 0;
        //     unsigned int h_resume_errors = 0;

        //     cudaMemcpy(&h_resume_checked,
        //             d_resume_checked,
        //             sizeof(unsigned int),
        //             cudaMemcpyDeviceToHost);

        //     cudaMemcpy(&h_resume_errors,
        //             d_resume_errors,
        //             sizeof(unsigned int),
        //             cudaMemcpyDeviceToHost);

        //     printf("resumeTinyTasks_debug checked=%u errors=%u\n",
        //         h_resume_checked,
        //         h_resume_errors);

        //     chkerr(cudaMemset(task_pointers.d_tiny_tail_B, 0, sizeof(unsigned int)));
        //     chkerr(cudaMemset(task_pointers.d_delta_tail_B, 0, sizeof(unsigned int)));

        //     unsigned int* d_continue_checked;
        //     unsigned int* d_continue_errors;
        //     unsigned int* d_continue_spilled;
        //     unsigned int* d_continue_overflow;

        //     chkerr(cudaMalloc(&d_continue_checked, sizeof(unsigned int)));
        //     chkerr(cudaMalloc(&d_continue_errors, sizeof(unsigned int)));
        //     chkerr(cudaMalloc(&d_continue_spilled, sizeof(unsigned int)));
        //     chkerr(cudaMalloc(&d_continue_overflow, sizeof(unsigned int)));

        //     chkerr(cudaMemset(d_continue_checked, 0, sizeof(unsigned int)));
        //     chkerr(cudaMemset(d_continue_errors, 0, sizeof(unsigned int)));
        //     chkerr(cudaMemset(d_continue_spilled, 0, sizeof(unsigned int)));
        //     chkerr(cudaMemset(d_continue_overflow, 0, sizeof(unsigned int)));

        //     if (h_tiny_tail_A > 0) {
        //         printf("Launching resumeTinyTasks_continue_debug with %u tiny tasks\n",
        //             h_tiny_tail_A);

        //         resumeTinyTasks_continue_debug<<<h_tiny_tail_A, 32>>>(
        //             subgraph_pointers,
        //             task_pointers.d_tasks_A,
        //             task_pointers.d_tiny_tasks_A,
        //             h_tiny_tail_A,
        //             task_pointers.d_delta_log_A,
        //             task_pointers.d_tiny_tasks_B,
        //             task_pointers.d_tiny_tail_B,
        //             TINY_MAX_CAP,
        //             task_pointers.d_delta_log_B,
        //             task_pointers.d_delta_tail_B,
        //             DELTA_MAX_CAP,
        //             d_continue_checked,
        //             d_continue_errors,
        //             d_continue_spilled,
        //             d_continue_overflow
        //         );

        //         chkerr(cudaDeviceSynchronize());
        //         checkCudaError(778);
        //     }

        //     unsigned int h_continue_checked = 0;
        //     unsigned int h_continue_errors = 0;
        //     unsigned int h_continue_spilled = 0;
        //     unsigned int h_continue_overflow = 0;
        //     unsigned int h_tiny_tail_B = 0;
        //     unsigned int h_delta_tail_B = 0;

        //     cudaMemcpy(&h_continue_checked,
        //             d_continue_checked,
        //             sizeof(unsigned int),
        //             cudaMemcpyDeviceToHost);

        //     cudaMemcpy(&h_continue_errors,
        //             d_continue_errors,
        //             sizeof(unsigned int),
        //             cudaMemcpyDeviceToHost);

        //     cudaMemcpy(&h_continue_spilled,
        //             d_continue_spilled,
        //             sizeof(unsigned int),
        //             cudaMemcpyDeviceToHost);

        //     chkerr(cudaMemcpy(&h_continue_overflow, d_continue_overflow, sizeof(unsigned int), cudaMemcpyDeviceToHost));

        //     cudaMemcpy(&h_tiny_tail_B,
        //             task_pointers.d_tiny_tail_B,
        //             sizeof(unsigned int),
        //             cudaMemcpyDeviceToHost);

        //     cudaMemcpy(&h_delta_tail_B,
        //             task_pointers.d_delta_tail_B,
        //             sizeof(unsigned int),
        //             cudaMemcpyDeviceToHost);

        //     printf("resumeTinyTasks_continue_debug checked=%u errors=%u spilled=%u overflow=%u tiny_tail_B=%u delta_tail_B=%u\n",
        //         h_continue_checked,
        //         h_continue_errors,
        //         h_continue_spilled,
        //         h_continue_overflow,
        //         h_tiny_tail_B,
        //         h_delta_tail_B);

        //     if (h_tiny_tail_B > 0) {
        //         TinyTask h_tt_B;

        //         cudaMemcpy(&h_tt_B,
        //                 task_pointers.d_tiny_tasks_B,
        //                 sizeof(TinyTask),
        //                 cudaMemcpyDeviceToHost);

        //         printf("First TinyTask B: idx=%d parent=%u plex=%u cand=%u excl=%u log_off=%u log_len=%u depth=%u hash=%llu\n",
        //             h_tt_B.idx,
        //             h_tt_B.parent_task_pos,
        //             h_tt_B.plex_sz,
        //             h_tt_B.cand_sz,
        //             h_tt_B.excl_sz,
        //             h_tt_B.log_off,
        //             h_tt_B.log_len,
        //             h_tt_B.depth,
        //             (unsigned long long)h_tt_B.state_hash);

        //         Delta h_delta_B[16];

        //         unsigned int to_print = h_tt_B.log_len;
        //         if (to_print > 16) to_print = 16;

        //         cudaMemcpy(h_delta_B,
        //                 task_pointers.d_delta_log_B + h_tt_B.log_off,
        //                 to_print * sizeof(Delta),
        //                 cudaMemcpyDeviceToHost);

        //         for (unsigned int i = 0; i < to_print; ++i) {
        //             printf("  B delta[%u]: v=%u old=%u new=%u neiInP_delta=%d neiInG_delta=%d\n",
        //                 i,
        //                 h_delta_B[i].v,
        //                 h_delta_B[i].old_label,
        //                 h_delta_B[i].new_label,
        //                 h_delta_B[i].neiInP_delta,
        //                 h_delta_B[i].neiInG_delta);
        //         }
        //     }

        //     unsigned int* d_resume_checked = nullptr;
        //     unsigned int* d_resume_errors  = nullptr;

        //     cudaMalloc(&d_resume_checked, sizeof(unsigned int));
        //     cudaMalloc(&d_resume_errors,  sizeof(unsigned int));

        //     cudaMemset(d_resume_checked, 0, sizeof(unsigned int));
        //     cudaMemset(d_resume_errors,  0, sizeof(unsigned int));

        //     if (h_tiny_tail_B > 0) {
        //         printf("Launching resumeTinyTasks_debug on TinyTask B with %u tiny tasks\n",
        //             h_tiny_tail_B);

        //         resumeTinyTasks_debug<<<h_tiny_tail_B, 32>>>(
        //             subgraph_pointers,
        //             task_pointers.d_tasks_A,
        //             task_pointers.d_tiny_tasks_B,
        //             h_tiny_tail_B,
        //             task_pointers.d_delta_log_B,
        //             d_resume_checked,
        //             d_resume_errors
        //         );

        //         cudaError_t step9_launch_err = cudaGetLastError();
        //         if (step9_launch_err != cudaSuccess) {
        //             printf("resumeTinyTasks_debug_B launch error: %s\n",
        //                 cudaGetErrorString(step9_launch_err));
        //         }

        //         cudaError_t step9_sync_err = cudaDeviceSynchronize();
        //         if (step9_sync_err != cudaSuccess) {
        //             printf("resumeTinyTasks_debug_B sync error: %s\n",
        //                 cudaGetErrorString(step9_sync_err));
        //         }
        //     } else {
        //         printf("Skipping Step 9: h_tiny_tail_B is 0\n");
        //     }

        //     h_resume_checked = 0;
        //     h_resume_errors  = 0;

        //     cudaMemcpy(&h_resume_checked,
        //             d_resume_checked,
        //             sizeof(unsigned int),
        //             cudaMemcpyDeviceToHost);

        //     cudaMemcpy(&h_resume_errors,
        //             d_resume_errors,
        //             sizeof(unsigned int),
        //             cudaMemcpyDeviceToHost);

        //     printf("resumeTinyTasks_debug_B checked=%u errors=%u\n",
        //         h_resume_checked,
        //         h_resume_errors);

        //     const unsigned int EXTRA_GENS = 4;

        //     TinyTask* cur_tiny = task_pointers.d_tiny_tasks_B;
        //     Delta* cur_log = task_pointers.d_delta_log_B;
        //     unsigned int cur_tail = h_tiny_tail_B;
        //     const char* cur_name = "B";

        //     for (unsigned int gen = 0; gen < EXTRA_GENS; gen++)
        //     {
        //         if (cur_tail == 0)
        //         {
        //             printf("Stopping: Current TinyTask queue %s is empty\n", cur_name);
        //             break;
        //         }

        //         const bool out_is_A = (cur_tiny == task_pointers.d_tiny_tasks_B);

        //         TinyTask* out_tiny = out_is_A ? task_pointers.d_tiny_tasks_A : task_pointers.d_tiny_tasks_B;

        //         Delta* out_log = out_is_A ? task_pointers.d_delta_log_A : task_pointers.d_delta_log_B;

        //         unsigned int* out_tiny_tail = out_is_A ? task_pointers.d_tiny_tail_A : task_pointers.d_tiny_tail_B;

        //         unsigned int* out_delta_tail = out_is_A ? task_pointers.d_delta_tail_A : task_pointers.d_delta_tail_B;

        //         const char* out_name = out_is_A ? "A" : "B";

        //         chkerr(cudaMemset(out_tiny_tail, 0, sizeof(unsigned int)));
        //         chkerr(cudaMemset(out_delta_tail, 0, sizeof(unsigned int)));

        //         chkerr(cudaMemset(d_continue_checked, 0, sizeof(unsigned int)));
        //         chkerr(cudaMemset(d_continue_errors, 0, sizeof(unsigned int)));
        //         chkerr(cudaMemset(d_continue_spilled, 0, sizeof(unsigned int)));
        //         chkerr(cudaMemset(d_continue_overflow, 0, sizeof(unsigned int)));

        //         printf("gen = %u, continuing TinyTask %s -> %s with %u tiny tasks\n", gen + 1, cur_name, out_name, cur_tail);

        //         // resumeTinyTasks_continue_debug<<<cur_tail, 32>>>(subgraph_pointers, task_pointers.d_tasks_A, cur_tiny, cur_tail, cur_log, out_tiny, out_tiny_tail, TINY_MAX_CAP, out_log, out_delta_tail, DELTA_MAX_CAP, d_continue_checked, d_continue_errors, d_continue_spilled, d_continue_overflow);

        //         // chkerr(cudaDeviceSynchronize());
        //         // checkCudaError(790 + gen);

        //             bool cuda_failed = false;

        //             for (unsigned int chunk_start = 0; chunk_start < cur_tail; chunk_start += CONTINUE_CHUNK)
        //             {
        //                 unsigned int h_overflow_now = 0;
        //                 chkerr(cudaMemcpy(&h_overflow_now, d_continue_overflow, sizeof(unsigned int), cudaMemcpyDeviceToHost));

        //                 if (h_overflow_now != 0)
        //                 {
        //                     printf("Stopping Chunks Early: Overflow already detected at chunk_start: %u\n", chunk_start);
        //                     break;
        //                 }

        //                 unsigned int h_errors_now = 0;
        //                 chkerr(cudaMemcpy(&h_errors_now, d_continue_errors, sizeof(unsigned int), cudaMemcpyDeviceToHost));

        //                 if (h_errors_now != 0)
        //                 {
        //                     printf("Stopping Chunks Early: Continue Errors already detected at chunk_start=%u\n", chunk_start);
        //                     break;
        //                 }

        //                 const unsigned int chunk_count = std::min(CONTINUE_CHUNK, cur_tail - chunk_start);
        //                 printf("Chunk start = %u count = %u\n", chunk_start, chunk_count);

        //                 unsigned int h_tiny_tail_now = 0;
        //                 unsigned int h_delta_tail_now = 0;

        //                 chkerr(cudaMemcpy(&h_tiny_tail_now, out_tiny_tail, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        //                 chkerr(cudaMemcpy(&h_delta_tail_now, out_delta_tail, sizeof(unsigned int), cudaMemcpyDeviceToHost));

        //                 TinyTask h_first_curr_tt;
        //                 chkerr(cudaMemcpy(&h_first_curr_tt, cur_tiny+chunk_start, sizeof(TinyTask), cudaMemcpyDeviceToHost));

        //                 const unsigned int max_children_per_input = 1u << LOCAL_BNB_DEPTH;
        //                 const unsigned int next_log_len = (unsigned int)h_first_curr_tt.log_len + LOCAL_BNB_DEPTH;

        //                 const unsigned long long worst_new_tiny = (unsigned long long)chunk_count * max_children_per_input;

        //                 const unsigned long long worst_new_delta = worst_new_tiny * next_log_len;

        //                 const unsigned long long worst_tiny_tail = (unsigned long long)h_tiny_tail_now + worst_new_tiny;

        //                 const unsigned long long worst_delta_tail = (unsigned long long)h_delta_tail_now + worst_new_delta;

        //                 if (worst_tiny_tail > (unsigned long long) TINY_MAX_CAP || worst_delta_tail > (unsigned long long)DELTA_MAX_CAP)
        //                 {
        //                     printf("Stopping chunks before launch: conservative capacity limit at chunk_start=%u, chunk_count=%u, current_tiny=%u/%u, current_delta=%u/%u, worst_new_tiny=%llu, worst_new_delta=%llu, next_log_len=%u\n", chunk_start, chunk_count, h_tiny_tail_now, TINY_MAX_CAP, h_delta_tail_now, DELTA_MAX_CAP, worst_new_tiny, worst_new_delta, next_log_len);

        //                     unsigned int one = 1;
        //                     chkerr(cudaMemcpy(d_continue_overflow, &one, sizeof(unsigned int), cudaMemcpyHostToDevice));

        //                     break;
        //                 }

        //                 resumeTinyTasks_continue_debug<<<chunk_count, 32>>>(subgraph_pointers, task_pointers.d_tasks_A, cur_tiny + chunk_start, chunk_count, cur_log, out_tiny, out_tiny_tail, TINY_MAX_CAP, out_log, out_delta_tail, DELTA_MAX_CAP, d_continue_checked, d_continue_errors, d_continue_spilled, d_continue_overflow);

        //                 cudaError_t launch_err = cudaGetLastError();
        //                 if (launch_err != cudaSuccess)
        //                 {
        //                     printf("Continue Chunk Launch Error at chunk start=%u: %s\n", chunk_start, cudaGetErrorString(launch_err));
        //                     cuda_failed = true;
        //                     break;
        //                 }
                        
        //                 cudaError_t sync_err = cudaDeviceSynchronize();
        //                 if (sync_err != cudaSuccess)
        //                 {
        //                     printf("Continue Chunk Sync Error at chunk_start=%u: %s\n", chunk_start, cudaGetErrorString(sync_err));
        //                     cuda_failed = true;
        //                     break;
        //                 }
        //             }
        //             if (cuda_failed){
        //                 printf("Stopping Generation %u because CUDA failed during chunked continuation\n", gen+1);
        //                 break;
        //             }

                

        //         h_continue_checked = 0;
        //         h_continue_errors = 0;
        //         h_continue_spilled = 0;
        //         h_continue_overflow = 0;
        //         unsigned int h_out_tiny_tail = 0;
        //         unsigned int h_out_delta_tail = 0;

        //         chkerr(cudaMemcpy(&h_continue_checked, d_continue_checked, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        //         chkerr(cudaMemcpy(&h_continue_errors, d_continue_errors, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        //         chkerr(cudaMemcpy(&h_continue_spilled, d_continue_spilled, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        //         chkerr(cudaMemcpy(&h_continue_overflow, d_continue_overflow, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        //         chkerr(cudaMemcpy(&h_out_tiny_tail, out_tiny_tail, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        //         chkerr(cudaMemcpy(&h_out_delta_tail, out_delta_tail, sizeof(unsigned int), cudaMemcpyDeviceToHost));

        //         printf("Continue %s->%s checked=%u errors=%u spilled=%u overflow=%u tiny_tail_%s=%u delta_tail_%s=%u\n", cur_name, out_name, h_continue_checked, h_continue_errors, h_continue_spilled, h_continue_overflow, out_name, h_out_tiny_tail, out_name, h_out_delta_tail);

        //         if (h_continue_errors != 0) {
        //             printf("Stopping: Continue Errors detected in generation %u\n", gen+1);
        //         }

        //         if (h_continue_overflow != 0) {
        //             printf("Stopping generation loop because continuation overflow occurred: "
        //                 "tiny_tail=%u tiny_cap=%u delta_tail=%u delta_cap=%u spilled=%u\n",
        //                 h_out_tiny_tail,
        //                 TINY_MAX_CAP,
        //                 h_out_delta_tail,
        //                 DELTA_MAX_CAP,
        //                 h_continue_spilled);
        //             break;
        //         }

        //         if (h_out_tiny_tail > 0)
        //         {
        //             TinyTask h_tt_out;

        //             chkerr(cudaMemcpy(&h_tt_out, out_tiny, sizeof(TinyTask), cudaMemcpyDeviceToHost));
        //             printf("First TinyTask %s: idx=%d, parent=%u, plex=%u, cand=%u, excl=%u, log_off=%u, log_len=%u, depth=%u, hash=%llu\n", out_name, h_tt_out.idx, h_tt_out.parent_task_pos, h_tt_out.plex_sz, h_tt_out.cand_sz, h_tt_out.excl_sz, h_tt_out.log_off, h_tt_out.log_len, h_tt_out.depth, (unsigned long long)h_tt_out.state_hash);

        //         }

        //         chkerr(cudaMemset(d_resume_checked, 0, sizeof(unsigned int)));
        //         chkerr(cudaMemset(d_resume_errors, 0, sizeof(unsigned int)));

        //         if (h_out_tiny_tail > 0)
        //         {
        //             printf("Validating TinyTask %s with %u tiny tasks\n", out_name, h_out_tiny_tail);

        //             if (h_continue_overflow != 0) {
        //                 printf("Skipping validation because continuation overflow occurred: "
        //                     "tiny_tail=%u tiny_cap=%u delta_tail=%u delta_cap=%u spilled=%u\n",
        //                     h_out_tiny_tail,
        //                     TINY_MAX_CAP,
        //                     h_out_delta_tail,
        //                     DELTA_MAX_CAP,
        //                     h_continue_spilled);
        //                 break;
        //             }

        //             if (h_continue_errors != 0) {
        //                 printf("Skipping validation because continuation errors occurred: errors=%u\n",
        //                     h_continue_errors);
        //                 break;
        //             }

        //             bool validate_cuda_failed = false;

        //             for (unsigned int chunk_start = 0; chunk_start < h_out_tiny_tail; chunk_start += CONTINUE_CHUNK)
        //             {

        //                 const unsigned int chunk_count = std::min(CONTINUE_CHUNK, h_out_tiny_tail - chunk_start);
        //                 printf("Validate Chunk start = %u count = %u\n", chunk_start, chunk_count);

        //                 resumeTinyTasks_debug<<<chunk_count, 32>>>(subgraph_pointers, task_pointers.d_tasks_A, out_tiny + chunk_start, chunk_count, out_log, d_resume_checked, d_resume_errors);

        //                 cudaError_t launch_err = cudaGetLastError();
        //                 if (launch_err != cudaSuccess)
        //                 {
        //                     printf("Validate Chunk Launch Error at chunk start=%u: %s\n", chunk_start, cudaGetErrorString(launch_err));
        //                     validate_cuda_failed = true;
        //                     break;
        //                 }
                        
        //                 cudaError_t sync_err = cudaDeviceSynchronize();
        //                 if (sync_err != cudaSuccess)
        //                 {
        //                     printf("Validate Chunk Sync Error at chunk_start=%u: %s\n", chunk_start, cudaGetErrorString(sync_err));
        //                     validate_cuda_failed = true;
        //                     break;
        //                 }
        //             }
        //             if (validate_cuda_failed){
        //                 printf("Stopping Generation %u because Validation CUDA failed\n", gen+1);
        //             }
        //         }

        //         h_resume_checked = 0;
        //         h_resume_errors = 0;

        //         chkerr(cudaMemcpy(&h_resume_checked, d_resume_checked, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        //         chkerr(cudaMemcpy(&h_resume_errors, d_resume_errors, sizeof(unsigned int), cudaMemcpyDeviceToHost));

        //         printf("Validate %s Checked=%u Errors=%u\n", out_name, h_resume_checked, h_resume_errors);

        //         if (h_resume_errors != 0)
        //         {
        //             printf("Stopping: Replay Validation Errors Detected in queue %s\n", out_name);
        //             break;
        //         }

        //         cur_tiny = out_tiny;
        //         cur_log = out_log;
        //         cur_tail = h_out_tiny_tail;
        //         cur_name = out_name;


        //     }

            int* tmp = d_abort_flag;
            d_abort_flag = d_abort2;
            d_abort2 = tmp;

            initializeBNB(6, task_pointers, plex_pointers, subgraph_pointers, d_blk, d_left, d_blk_counter, d_left_counter, commonMtx, plex_count, bnb_neiInG, bnb_neiInP, d_sat, d_commons, d_uni, cycles, d_adj, d_abort3, global_count);
            // initializeBNB2(6, task_pointers, plex_pointers, subgraph_pointers, d_blk, d_left, d_blk_counter, d_left_counter, commonMtx, plex_count, d_sat, d_commons, d_uni, cycles, d_adj, d_abort3, buf, h_task_stage, d_state2, d_res2, recExcl, recCand);

            // cudaEventRecord(event_stop);
            // cudaEventSynchronize(event_stop);
            // time_milli_sec = 0;
            // cudaEventElapsedTime(&time_milli_sec, event_start, event_stop);
            // time_3 += time_milli_sec;
            // cudaEventRecord(event_start);
        }

        // kSearch3<<<BLK_NUMS, BLK_DIM>>>(i, plex_pointers, subgraph_pointers, graph_pointers, task_pointers, d_left, d_blk_counter, d_left_counter, d_res, d_br, d_state, d_len, d_sz, neiInG, neiInP, plex_count, commonMtx, recCand1, recCand2, recExcl, recCand, d_v2delete, d_adj, d_sat, d_commons, d_uni, global_count);
        
        // }
        cudaMemset(d_blk_counter, 0, WARPS * sizeof(unsigned int));
        cudaMemset(d_left_counter, 0, WARPS * sizeof(unsigned int));
        cudaMemset(d_hopSz, 0, WARPS * sizeof(unsigned int));
        cudaMemset(d_visited, 0, range * WARPS * sizeof(uint32_t));
        cudaMemset(d_adj, 0, ADJSIZE * WARPS * sizeof(uint32_t));
        //cudaMemset(d_count, 0, pn * WARPS * sizeof(uint16_t));
        cudaMemset(commonMtx, 0, totalBytes);
        cudaMemset(d_sz, 0, WARPS * sizeof(unsigned int));

        // cudaEventRecord(event_stop);
        // cudaEventSynchronize(event_stop);
        // time_milli_sec = 0;
        // cudaEventElapsedTime(&time_milli_sec, event_start, event_stop);
        // time_7 += time_milli_sec;
        // cudaEventRecord(event_start);
    }

    cudaEventRecord(event_stop);
    cudaEventSynchronize(event_stop);
    time_milli_sec = 0;
    cudaEventElapsedTime(&time_milli_sec, event_start, event_stop);
    time_0 += time_milli_sec;
    printf("Total Time Elapsed: %f ms\n", time_0);
    
    // printf("Time 0: %f, Time 1: %f, Time 2: %f, Time 3: %f, Time 4: %f, Time 5: %f, Time 6: %f\n", time_0, time_1, time_2, time_3, time_4, time_5, time_6);
    
    cudaMemcpy(&h_plex_count, plex_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_validblk, validblk, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_global, global_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("Total Valid Blocks: %d, Maximal k-Plexes: %u\n", h_validblk, h_plex_count);
    // printf("Total tasks generated: %u\n", h_global);
    printf("\nKernel Launch Successfully\n");
    free_graph_gpu_memory(graph_pointers, degen_pointers);
    // free_tiny_task_queues(task_pointers);
    // cudaFree(d_resume_checked);
    // cudaFree(d_resume_errors);
    // delete[] buf.tasks;
    // cudaFreeHost(h_task_stage);
}

#endif // CUTS_HOST_FUNCS_H