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

void initializeBNB(int initialN, T_pointers &task_pointers, P_pointers plex_pointers, S_pointers subgraph_pointers, unsigned int *d_blk, unsigned int *d_left, unsigned int *d_blk_counter, unsigned int *d_left_counter, uint8_t *commonMtx, unsigned int *plex_count, uint16_t* d_sat, uint16_t* d_commons, uint32_t* d_uni, unsigned long long* cycles, uint32_t* d_adj, int* d_abort, HostTaskBuffer& hostBuf, HostTask* h_task_stage, cudaStream_t stream = 0)
{
    cudaMemset(d_abort, 0, sizeof(int));
    int h_abort = 0;
    unsigned int head = 0;
    while (true)
    {
        unsigned int tail;
        // unsigned int plex;
        cudaMemcpy(&tail, task_pointers.d_tail_A, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        // cudaMemcpy(&plex, plex_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        // printf("tail: %u, iteration: %d, hostSize: %u\n", tail, initialN, hostBuf.size);
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
                BNB<<<BLK_NUMS, BLK_DIM>>>(w, plex_pointers, subgraph_pointers, d_blk, d_left, d_blk_counter, d_left_counter, commonMtx, Q_in, Q_out, task_pointers.d_tasks_A, numTasks, 0, tail_out, task_pointers.d_tail_A, lab_out, nei_out, P_out, task_pointers.d_all_labels_A, task_pointers.d_all_neiInG_A, task_pointers.d_all_neiInP_A, plex_count, d_sat, d_commons, d_uni, cycles, d_adj, d_abort);
                cudaMemcpy(&h_abort, d_abort, sizeof(int), cudaMemcpyDeviceToHost);
                // if (h_abort) 
                // {
                //     printf("Maximum Capacity Reached on level %d\n", initialN);
                //     break;
                // }
                cudaMemcpy(&tail, task_pointers.d_tail_A, sizeof(unsigned int), cudaMemcpyDeviceToHost);
                // printf("tail: %d, capacity: %u\n", tail, MAX_CAP/4);
                if (h_abort)
                {
                    // printf("Maximum Capacity Reached on level %d\n", initialN);
                    // printf("Copying Some Tasks To Host Memory with size: %u\n", hostBuf.size);
                    // spillToHost(task_pointers, task_pointers.d_tail_A, hostBuf);
                    // cudaMemset(d_abort, 0, sizeof(int));
                }
            }
            cudaDeviceSynchronize();
            checkCudaError(initialN);
            // if (h_abort) break;
            // cudaMemcpy(&tail, task_pointers.d_tail_A, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            // if (tail == 0) break;
            flip = !flip;
        }
        // if(h_abort) break;
    }
    cudaMemset(task_pointers.d_tail_A, 0, sizeof(unsigned int));
    cudaMemset(task_pointers.d_tail_B, 0, sizeof(unsigned int));
    cudaMemset(task_pointers.d_tail_C, 0, sizeof(unsigned int));
}

void initializeBNB2(int initialN, T_pointers &task_pointers, P_pointers plex_pointers, S_pointers subgraph_pointers, unsigned int *d_blk, unsigned int *d_left, unsigned int *d_blk_counter, unsigned int *d_left_counter, uint8_t *commonMtx, unsigned int *plex_count, uint16_t* d_sat, uint16_t* d_commons, uint32_t* d_uni, unsigned long long* cycles, uint32_t* d_adj, int* d_abort, HostTaskBuffer& hostBuf, HostTask* h_task_stage, unsigned int* state, unsigned int* res)
{
    cudaMemset(d_abort, 0, sizeof(int));
    int h_abort = 0;
    unsigned int head = 0;
    while (true)
    {
        unsigned int tail;
        // unsigned int plex;
        cudaMemcpy(&tail, task_pointers.d_tail_A, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        // cudaMemcpy(&plex, plex_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        // printf("tail: %u, iteration: %d, hostSize: %u\n", tail, initialN, hostBuf.size);
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
                BNB2<<<BLK_NUMS, BLK_DIM>>>(w, plex_pointers, subgraph_pointers, d_blk, d_left, d_blk_counter, d_left_counter, commonMtx, Q_in, Q_out, task_pointers.d_tasks_A, numTasks, 0, tail_out, task_pointers.d_tail_A, lab_out, nei_out, P_out, task_pointers.d_all_labels_A, task_pointers.d_all_neiInG_A, task_pointers.d_all_neiInP_A, plex_count, d_sat, d_commons, d_uni, cycles, d_adj, d_abort, state, res);
                cudaMemcpy(&h_abort, d_abort, sizeof(int), cudaMemcpyDeviceToHost);
                // if (h_abort) 
                // {
                //     printf("Maximum Capacity Reached on level %d\n", initialN);
                //     break;
                // }
                cudaMemcpy(&tail, task_pointers.d_tail_A, sizeof(unsigned int), cudaMemcpyDeviceToHost);
                // printf("tail: %d, capacity: %u\n", tail, MAX_CAP/4);
                if (h_abort)
                {
                    // printf("Maximum Capacity Reached on level %d\n", initialN);
                    // printf("Copying Some Tasks To Host Memory with size: %u\n", hostBuf.size);
                    // spillToHost(task_pointers, task_pointers.d_tail_A, hostBuf);
                    // cudaMemset(d_abort, 0, sizeof(int));
                }
            }
            cudaDeviceSynchronize();
            checkCudaError(initialN);
            // if (h_abort) break;
            // cudaMemcpy(&tail, task_pointers.d_tail_A, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            // if (tail == 0) break;
            flip = !flip;
        }
        // if(h_abort) break;
    }
    cudaMemset(task_pointers.d_tail_A, 0, sizeof(unsigned int));
    cudaMemset(task_pointers.d_tail_B, 0, sizeof(unsigned int));
    cudaMemset(task_pointers.d_tail_C, 0, sizeof(unsigned int));
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
        while (iv < eu && dead[iv]) ++iv;

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
                neighbors[write++] = neighbors[i];
                triangles[write] = triangles[i];
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

            while (iu >= eu && iv >= ev)
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

printf("n: %d, m: %d, q: %d, k: %d\n", pn, peelG.m, lb, k);
    // vector<pair<int, int>> Q_e;
    // vector<int> triangles;
    // for (int i = 0; i < pn; i++)
    // {
    //     int begin = peelG.offsets[i];
    //     int end = peelG.offsets[i+1];

    //     // vector<int> neighbor(peelG.neighbors + begin, peelG.neighbors+end);
        
    //     for (int p = begin; p < end; p++)
    //     {
    //         int v = peelG.neighbors[p];
    //         int begin2 = peelG.offsets[v];
    //         int end2 = peelG.offsets[v+1];

    //         unsigned iu = begin, iv = begin2;
    //         int common = 0;
    //         while (iu < end && iv < end2)
    //         {
    //             unsigned a = peelG.neighbors[iu];
    //             unsigned b = peelG.neighbors[iv];

    //             if (a == b)
    //             {
    //                 common++;
    //                 iu++;
    //                 iv++;
    //             }
    //             else if (a < b) iu++;
    //             else iv++;
    //         }
    //         triangles.push_back(common);
    //         if (i < v && common < (lb - 2 * k)) Q_e.emplace_back(i, v); // q - 2 * k 
    //     }
    // }

    // printf("Edges to be pruned: %d\n", Q_e.size());



    //// PARALLEL
    // while (true)
    // {

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

    // printf("Edges to be pruned: %d\n", int(Q_e.size()));

// if (Q_e.size() == 0) break;

// printf("Neighbors: ");
// for (int i = 0; i < peelG.m; i++)
// {
//     printf("%d ", peelG.neighbors[i]);
// }
// printf("\n");

// printf("Offsets: ");
// for (int i = 0; i < peelG.n+1; i++)
// {
//     printf("%d ", peelG.offsets[i]);
// }
// printf("\n");

// printf("Degree: ");
// for (int i = 0; i < peelG.n; i++)
// {
//     printf("%d ", peelG.degree[i]);
// }
// printf("\n");

// printf("Before: \n");
// for (int i = 0; i < Q_e.size(); i++)
// {
//     printf("%d %d\n", Q_e[i].first, Q_e[i].second);
// }

fast_truss_peeling_parallel(peelG.neighbors, peelG.offsets, peelG.degree, Q_e, pn, &peelG.m, triangles);
peelG = peelGraph(peelG, mark, resNei);
pn = peelG.n;
// }

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

    unsigned long long* cycles;
    int *d_abort_flag = nullptr;
    int* d_abort2 = nullptr;


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

    cudaMalloc(&d_abort_flag, sizeof(int));
    cudaMalloc(&d_abort2, sizeof(int));
    cudaMemset(d_abort_flag, 0, sizeof(int));
    cudaMemset(d_abort2, 0, sizeof(int));

    graph<intT> subg;

    HostTaskBuffer buf;
    // initHostTaskBuffer(buf, 10*MAX_CAP);
    
    HostTask* h_task_stage = nullptr;
    // cudaHostAlloc(&h_task_stage, STAGING_CHUNK * sizeof(HostTask), cudaHostAllocDefault);

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
    // printf("Total Iterations: %d\n", (pn/WARPS)+1);
    int h_abort = 0;

    // Total nodes = 27000, warps = 4454, 

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
        // time_5 += time_milli_sec;
        // cudaEventRecord(event_start);
        //-----
        // while(h_abort = 1)
        // {
        cudaMemset(d_abort_flag, 0, sizeof(int));
        kSearch<<<BLK_NUMS, BLK_DIM>>>(i, plex_pointers, subgraph_pointers, graph_pointers, task_pointers, d_blk_counter, d_res, d_br, d_state, d_len, d_sz, neiInG, neiInP, plex_count, commonMtx, recCand1, recCand2, d_v2delete, d_adj, cycles, d_abort_flag);
        // kSearch3<<<BLK_NUMS, BLK_DIM>>>(i, plex_pointers, subgraph_pointers, graph_pointers, task_pointers, d_left, d_blk_counter, d_left_counter, d_res, d_br, d_state, d_len, d_sz, neiInG, neiInP, plex_count, commonMtx, recCand1, recCand2, recExcl, recCand, d_v2delete, d_adj, d_sat, d_commons, d_uni);
        cudaDeviceSynchronize();
        checkCudaError(5);

        
        // cudaMemcpy(&h_abort, d_abort_flag, sizeof(int), cudaMemcpyDeviceToHost);
        // if (h_abort)
        // {
        //     printf("Memory FULL: Maximum Capacity Reached in kSearch\n");
            // break;
            // printf("Remaining iterations: %d\n", (pn/warps+1)-i);
            // printf("Remaining nodes: %d\n", nodes - i * warps);
            // nodes = nodes - i * warps;
            // if (change)
            // {
            //     blk_nums = int(blk_nums / 2);
            //     change = false;
            // }
            // else 
            // {
            // blk_dim = int(blk_dim / 2);
            // change = true;
            // }
            // warps = blk_nums * (blk_dim/32);
            // i--;
            // remaining = remaining + (nodes / warps);
            // printf("blk_num: %d, blk_dim: %d, warps: %d, remaining: %d\n", blk_nums, blk_dim, warps, remaining);
            // cudaMemset(d_abort_flag, 0, sizeof(int));
            // continue;
        // }

        // cudaEventRecord(event_stop);
        // cudaEventSynchronize(event_stop);
        // time_milli_sec = 0;
        // cudaEventElapsedTime(&time_milli_sec, event_start, event_stop);
        // time_6 += time_milli_sec;
        // cudaEventRecord(event_start);

        initializeBNB2(6, task_pointers, plex_pointers, subgraph_pointers, d_blk, d_left, d_blk_counter, d_left_counter, commonMtx, plex_count, d_sat, d_commons, d_uni, cycles, d_adj, d_abort2, buf, h_task_stage, d_state2, d_res2);
        
        // if (h_abort)
        // {
        // printf("Restarting\n");
        // cudaMemset(d_abort2, 0, sizeof(int));
        // kSearch2<<<BLK_NUMS, BLK_DIM>>>(i, plex_pointers, subgraph_pointers, graph_pointers, task_pointers, d_blk_counter, d_res, d_br, d_state, d_len, d_sz, neiInG, neiInP, plex_count, commonMtx, recCand1, recCand2, d_v2delete, d_adj, cycles, d_abort_flag, d_abort2);
        // cudaDeviceSynchronize();
        // checkCudaError(7);
        // cudaMemcpy(&h_abort, d_abort2, sizeof(int), cudaMemcpyDeviceToHost);
        // if (h_abort) 
        // {
        //     printf("Memory again failed\n");
        //     // break;
        // }
        // initializeBNB(8, task_pointers, plex_pointers, subgraph_pointers, d_blk, d_left, d_blk_counter, d_left_counter, commonMtx, plex_count, d_sat, d_commons, d_uni, cycles, d_adj, d_abort2, buf, h_task_stage);

        // }
        
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
    printf("Total Valid Blocks: %d, Maximal k-Plexes: %u\n", h_validblk, h_plex_count);
    printf("\nKernel Launch Successfully\n");
    free_graph_gpu_memory(graph_pointers, degen_pointers);
    // delete[] buf.tasks;
    // cudaFreeHost(h_task_stage);
}

#endif // CUTS_HOST_FUNCS_H