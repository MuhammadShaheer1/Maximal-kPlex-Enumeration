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

void initializeBNB(int initialN, T_pointers &task_pointers, P_pointers plex_pointers, S_pointers subgraph_pointers, unsigned int *d_blk, unsigned int *d_left, unsigned int *d_blk_counter, unsigned int *d_left_counter, uint8_t *commonMtx, unsigned int *plex_count, uint16_t* d_sat, uint16_t* d_commons, uint32_t* d_uni, unsigned long long* cycles, uint32_t* d_adj)
{
    unsigned int head = 0;
    while (true)
    {
        unsigned int tail;
        cudaMemcpy(&tail, task_pointers.d_tail_A, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        // printf("tail: %d\n", tail);
        if (tail == 0)
            break;
        unsigned int batch = std::min((unsigned)4*WARPS, tail);

        head = tail - batch;
        
        cudaMemcpy(task_pointers.d_tail_B, &batch, sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(task_pointers.d_tasks_B, task_pointers.d_tasks_A + head, batch * sizeof(Task), cudaMemcpyDeviceToDevice);
        cudaMemcpy(task_pointers.d_all_labels_B, task_pointers.d_all_labels_A + head * MAX_BLK_SIZE, batch * MAX_BLK_SIZE * sizeof(uint8_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(task_pointers.d_all_neiInG_B, task_pointers.d_all_neiInG_A + head * MAX_BLK_SIZE, batch * MAX_BLK_SIZE * sizeof(uint16_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(task_pointers.d_all_neiInP_B, task_pointers.d_all_neiInP_A + head * MAX_BLK_SIZE, batch * MAX_BLK_SIZE * sizeof(uint16_t), cudaMemcpyDeviceToDevice);

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
                BNB<<<BLK_NUMS, BLK_DIM>>>(w, plex_pointers, subgraph_pointers, d_blk, d_left, d_blk_counter, d_left_counter, commonMtx, Q_in, Q_out, task_pointers.d_tasks_A, numTasks, 0, tail_out, task_pointers.d_tail_A, lab_out, nei_out, P_out, task_pointers.d_all_labels_A, task_pointers.d_all_neiInG_A, task_pointers.d_all_neiInP_A, plex_count, d_sat, d_commons, d_uni, cycles, d_adj);
            }
            cudaDeviceSynchronize();
            checkCudaError(6);
            flip = !flip;
        }
    }
}

void decomposableSearch(const graph<int> &g)
{
    int *dpos = new int[g.n];
    int *dseq = new int[g.n];
    bool *mark = new bool[g.n];
    int *resNei = new int[g.n];
#pragma omp parallel for
    for (int i = 0; i < g.n; ++i)
        dpos[i] = INT_MAX;
    unsigned int *validblk;
    unsigned int h_validblk;

    float time_0 = 0;
    cudaEvent_t event_start;
    cudaEvent_t event_stop;
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);
    cudaEventRecord(event_start);

    graph<intT> peelG = peelGraph(g, mark, resNei);
    const int pn = peelG.n;
    volatile bool *const ready = (volatile bool *)mark;

#pragma omp for
    for (int i = 0; i < pn; ++i)
        mark[i] = false;
#pragma omp master
    {
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
    }

    cudaEventRecord(event_stop);
    cudaEventSynchronize(event_stop);
    float time_milli_sec = 0;
    cudaEventElapsedTime(&time_milli_sec, event_start, event_stop);
    time_0 += time_milli_sec;

    P_pointers plex_pointers;
    plex_pointers.k = k;
    plex_pointers.lb = lb;
    plex_pointers.bd = bd;

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
    unsigned int *left_count;
    unsigned int *plex_count;
    uint8_t *commonMtx;
    unsigned int h_plex_count;

    uint16_t *d_sat;
    uint16_t *d_commons;
    uint32_t *d_uni;
    uint32_t *d_adj;

    unsigned int* d_res;
    unsigned int* d_br;
    unsigned int* d_state;
    unsigned int* d_v2delete;

    unsigned int* recCand1;
    unsigned int* recCand2;

    uint16_t* neiInG;
    uint16_t* neiInP; 

    unsigned long long* cycles;

    cudaMalloc(&d_res, WARPS * MAX_DEPTH * sizeof(unsigned int));
    cudaMalloc(&d_br, WARPS * MAX_DEPTH * sizeof(unsigned int));
    cudaMalloc(&d_state, WARPS * MAX_DEPTH * sizeof(unsigned int));
    cudaMalloc(&d_v2delete, WARPS * MAX_DEPTH * sizeof(unsigned int));

    cudaMalloc(&recCand1, WARPS * MAX_BLK_SIZE * sizeof(unsigned int));
    cudaMalloc(&recCand2, WARPS * MAX_BLK_SIZE * sizeof(unsigned int));

    cudaMalloc(&neiInG, WARPS * MAX_BLK_SIZE * sizeof(uint16_t));
    cudaMalloc(&neiInP, WARPS * MAX_BLK_SIZE * sizeof(uint16_t));

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
    cudaMemset(d_uni, 0, WARPS * 32 * sizeof(uint32_t));
    cudaMemset(d_adj, 0, ADJSIZE * WARPS * sizeof(uint32_t));
    cudaMemset(cycles, 0, 40 * sizeof(unsigned long long));

    size_t capacity = MAX_CAP;

    cudaMalloc(&task_pointers.d_tasks_A, capacity * sizeof(Task));
    cudaMalloc(&task_pointers.d_all_labels_A, capacity * MAX_BLK_SIZE * sizeof(uint8_t));
    cudaMalloc(&task_pointers.d_all_neiInG_A, capacity * MAX_BLK_SIZE * sizeof(uint16_t));
    cudaMalloc(&task_pointers.d_all_neiInP_A, capacity * MAX_BLK_SIZE * sizeof(uint16_t));
    cudaMalloc(&task_pointers.d_tail_A, sizeof(unsigned int));
    cudaMemset(task_pointers.d_tail_A, 0, sizeof(unsigned int));

    size_t capacity2 = SMALL_CAP;
    cudaMalloc(&task_pointers.d_tasks_B, capacity2 * sizeof(Task));
    cudaMalloc(&task_pointers.d_all_labels_B, capacity2 * MAX_BLK_SIZE * sizeof(uint8_t));
    cudaMalloc(&task_pointers.d_all_neiInG_B, capacity2 * MAX_BLK_SIZE * sizeof(uint16_t));
    cudaMalloc(&task_pointers.d_all_neiInP_B, capacity2 * MAX_BLK_SIZE * sizeof(uint16_t));
    cudaMalloc(&task_pointers.d_tail_B, sizeof(unsigned int));

    cudaMalloc(&task_pointers.d_tasks_C, capacity2 * sizeof(Task));
    cudaMalloc(&task_pointers.d_all_labels_C, capacity2 * MAX_BLK_SIZE * sizeof(uint8_t));
    cudaMalloc(&task_pointers.d_all_neiInG_C, capacity2 * MAX_BLK_SIZE * sizeof(uint16_t));
    cudaMalloc(&task_pointers.d_all_neiInP_C, capacity2 * MAX_BLK_SIZE * sizeof(uint16_t));
    cudaMalloc(&task_pointers.d_tail_C, sizeof(unsigned int));

    graph<intT> subg;
    cudaEventRecord(event_start);
    // float time_1 = 0;
    // float time_2 = 0;
    // float time_3 = 0;
    // float time_4 = 0;
    // float time_5 = 0;
    // float time_6 = 0;
    // float time_7 = 0;

    // unsigned long long* h_cycles;
    // h_cycles = (unsigned long long *)malloc(40 * sizeof(unsigned long long));

    for (int i = 0; i < (pn/WARPS)+1; i++)
    {
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
        // time_5 += time_milli_sec;
        // cudaEventRecord(event_start);

        kSearch<<<BLK_NUMS, BLK_DIM>>>(i, plex_pointers, subgraph_pointers, graph_pointers, task_pointers, d_blk_counter, d_res, d_br, d_state, neiInG, neiInP, plex_count, commonMtx, recCand1, recCand2, d_v2delete, d_adj, cycles);
        cudaDeviceSynchronize();
        checkCudaError(5);

        // cudaEventRecord(event_stop);
        // cudaEventSynchronize(event_stop);
        // time_milli_sec = 0;
        // cudaEventElapsedTime(&time_milli_sec, event_start, event_stop);
        // time_6 += time_milli_sec;
        // cudaEventRecord(event_start);

        initializeBNB(0, task_pointers, plex_pointers, subgraph_pointers, d_blk, d_left, d_blk_counter, d_left_counter, commonMtx, plex_count, d_sat, d_commons, d_uni, cycles, d_adj);

        cudaMemset(d_blk_counter, 0, WARPS * sizeof(unsigned int));
        cudaMemset(d_left_counter, 0, WARPS * sizeof(unsigned int));
        cudaMemset(d_hopSz, 0, WARPS * sizeof(unsigned int));
        cudaMemset(d_visited, 0, range * WARPS * sizeof(uint32_t));
        cudaMemset(d_adj, 0, ADJSIZE * WARPS * sizeof(uint32_t));
        //cudaMemset(d_count, 0, pn * WARPS * sizeof(uint16_t));
        cudaMemset(commonMtx, 0, totalBytes);

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

    // cudaMemcpy(h_cycles, cycles, 40 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    // int device;
    // cudaGetDevice(&device);
    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, device);
    // // for (int i = 0; i < 40; i++)
    // // {
    //     double us = (double) h_cycles[0] / prop.clockRate;
    //     printf("Block %d: %f ms\n", 0, us);
    // // }
    
    // printf("Time 0: %f, Time 1: %f, Time 2: %f, Time 3: %f, Time 4: %f, Time 5: %f, Time 6: %f, Time 7: %f\n", time_0, time_1, time_2, time_3, time_4, time_5, time_6, time_7);
    
    cudaMemcpy(&h_plex_count, plex_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_validblk, validblk, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("Total Valid Blocks: %d, Maximal k-Plexes: %d\n", h_validblk, h_plex_count);
    printf("\nKernel Launch Successfully\n");
    free_graph_gpu_memory(graph_pointers, degen_pointers);
}

#endif // CUTS_HOST_FUNCS_H