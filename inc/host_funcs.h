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
graph<T> peelGraph(const graph<T> &g,bool* const mark,int * const resNei){
    const int n=g.n;
    #pragma omp parallel
    {
        thread_local std::queue<int> Q;
        #pragma omp for
        for(int i=0;i<n;++i){
            if(g.degree[i]<bd){
                mark[i]=false;
                Q.push(i);
            }
            else{
                mark[i]=true;
                resNei[i]=g.degree[i];
            }
        }
        while(Q.size()){
            const int ele=Q.front();
            Q.pop();
            for(int i=g.offsets[ele];i<g.offsets[ele+1];++i){
                const int nei=g.neighbors[i];
                if(mark[nei]){
                    int old=resNei[nei];
                    while(!utils::CAS(&resNei[nei],old,old-1)){
                        old=resNei[nei];
                    }
                    if(old==bd){
                        mark[nei]=false;
                        Q.push(nei);
                    }
                }
            }
        }
    }
    int* const map=new int[n];
     #pragma omp parallel for
    for(int i=0;i<n;++i)map[i]=i;
    _seq<int> leadList = sequence::pack(map, mark, n);
    const int pn=leadList.n;
    #pragma omp parallel for
    for(int i=0;i<pn;++i)map[leadList.A[i]]=i; 
    //vertex<int> *vertices = newA(vertex<int>, pn);
    uintT *newOffsets = newA(uintT, pn+1);
    uintT *newDegrees = newA(uintT, pn);

    #pragma omp parallel for
    for (int i = 0; i < pn; i++)
    {
        int ori = leadList.A[i];
        int count = 0;
        for (uintT j = g.offsets[ori]; j < g.offsets[ori+1]; j++)
        {
            int nei = g.neighbors[j];
            if (mark[nei]) count++;
        }
        newDegrees[i] = count;
    }

    newOffsets[0] = 0;
    for (int i = 1; i < pn+1; i++)
    {
        newOffsets[i] = newOffsets[i-1] + newDegrees[i - 1];
    }

    uintT totalEdges = newOffsets[pn];
    uintT *newNeighbors = newA(uintT, totalEdges);

    #pragma omp parallel for
    for (int i = 0; i < pn; i++)
    {
        int ori = leadList.A[i];
        int cursor = newOffsets[i];
        for (uintT j = g.offsets[ori]; j < g.offsets[ori+1]; j++)
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
    if (d_keys.size()==0)
    {
        d_keys.resize(WARPS*MAX_BLK_SIZE);
        thrust::transform(
            thrust::device,
            thrust::make_counting_iterator<unsigned>(0),
            thrust::make_counting_iterator<unsigned>(WARPS*MAX_BLK_SIZE),
            d_keys.begin(),
            [] __device__ (unsigned idx) {return idx / MAX_BLK_SIZE;}
        );
    }

    auto deg_ptr = thrust::device_pointer_cast(s.degree);
    auto ldeg_ptr = thrust::device_pointer_cast(s.l_degree);
    auto off_ptr = thrust::device_pointer_cast(s.offsets);
    auto loff_ptr = thrust::device_pointer_cast(s.l_offsets);

    // thrust::device_vector<unsigned> d_len(WARPS);
    // cudaMemcpy(thrust::raw_pointer_cast(d_len.data()), d_blk_counter, WARPS*sizeof(unsigned), cudaMemcpyDeviceToDevice);

    // unsigned* raw_d_len = thrust::raw_pointer_cast(d_len.data());

    // thrust::transform(
    //     thrust::device,
    //     thrust::make_counting_iterator<unsigned>(0),
    //     thrust::make_counting_iterator<unsigned>(WARPS*MAX_BLK_SIZE),
    //     deg_ptr,
    //     deg_ptr,
    //     [raw_d_len] __device__ (unsigned idx, unsigned val){
    //         unsigned w = idx / MAX_BLK_SIZE;
    //         unsigned base = w * MAX_BLK_SIZE;
    //         unsigned off = idx - base;
    //         return (off < raw_d_len[w]) ? val : 0u;
    //     }
    // );
    // thrust::transform(
    //     thrust::device,
    //     thrust::make_counting_iterator<unsigned>(0),
    //     thrust::make_counting_iterator<unsigned>(WARPS*MAX_BLK_SIZE),
    //     ldeg_ptr,
    //     ldeg_ptr,
    //     [raw_d_len] __device__ (unsigned idx, unsigned val){
    //         unsigned w = idx / MAX_BLK_SIZE;
    //         unsigned base = w * MAX_BLK_SIZE;
    //         unsigned off = idx - base;
    //         return (off < raw_d_len[w]) ? val : 0u;
    //     }
    // );
    thrust::exclusive_scan_by_key(
        thrust::device,
        d_keys.begin(),
        d_keys.end(),
        deg_ptr,
        off_ptr
    );
    thrust::exclusive_scan_by_key(
        thrust::device,
        d_keys.begin(),
        d_keys.end(),
        ldeg_ptr,
        loff_ptr
    );
}

int plexCnt = 0;

bool isNeighbor(int u, int v, const graph<int> &g)
{
    int begin = g.offsets[u];
    int end = g.offsets[u+1];
    for (int i = begin; i < end; i++)
    {
        if (g.neighbors[i] == v) return true;
    }
    return false;
}

void update_missing_add(int v, vector<int> &missing, const vector<int>& U, const graph<int> &g)
{
    for (int u: U)
    {
        if (!isNeighbor(v, u, g))
            missing[u]++;
    }
}

void update_missing_remove(int v, vector<int>& missing, const vector<int>& U, const graph<int> &g)
{
    for (int u: U)
    {
        if (!isNeighbor(v, u, g))
            missing[u]--;
    }
}

bool isKplex(int v, vector<int>& missing, const vector<int>& U, const graph<int> &g)
{
    if (missing[v] > (k-1))
    {
        return false;
    }

    for (int u: U)
    {
        if (missing[u] == (k-1) && !isNeighbor(v, u, g))
            return false;
    }
    return true;
}

bool isMaximal(vector<int> missing, vector<int> P, vector<int> left, const graph<int> &g)
{
    //printf("Count: ");
    for (int u: left)
    {
        int count = 0;
        for (int v: P)
        {
            if (!isNeighbor(v, u, g)) count++;
        }
        //printf("%d ", count);
        if (count > k - 1) continue;
        bool validExtension = true;
        for (int v: P)
        {
            if (!isNeighbor(v, u, g))
            {
                if (missing[v] >= (k-1))
                {
                    validExtension = false;
                    break;
                }
            }
        }
        if (validExtension)
        {
            return false;
        }
    }
    //printf("\n");
    return true;
}

void bKplex(vector<int> &P, vector<int> &C, vector<int> &X, vector<int> &left, vector<int> &missing, const graph<int> &g)
{
    // printf("PSize: %d, CSize: %d, XSize: %d, LeftSz: %d\n", P.size(), C.size(), X.size(), left.size());
    // printf("Maximal k-plexes: %d\n", plexCnt);
    // printf("P: ");
    // for (int u: P)
    // {
    //     printf("%d ", u);
    // }
    // printf("\n");
    // printf("C: ");
    // for (int u: C)
    // {
    //     printf("%d ", u);
    // }
    // printf("\n");
    // printf("X: ");
    // for (int u: X)
    // {
    //     printf("%d ", u);
    // }
    // printf("\n");
    // printf("Missing: ");
    // for (int u: missing)
    // {
    //     printf("%d ", u);
    // }
    // printf("\n");
    if (C.empty() && X.empty())
    {
        if (P.size() >= lb && isMaximal(missing, P, left, g))
        {
            // printf("Maximal k-plex found\n");
            plexCnt++;
        }
        return;
    }
    vector<int> candList = C;
    for (int v: candList)
    {
        auto it = find(C.begin(), C.end(), v);
        if (it == C.end()) return;

        C.erase(it);

        vector<int> C2, X2;
        vector<int> missing2 = missing;

        update_missing_add(v, missing2, C, g);
        update_missing_add(v, missing2, X, g);
        update_missing_add(v, missing2, P, g);

        P.push_back(v);

        for (int u: C)
        {
            if (isKplex(u, missing2, P, g))
                C2.push_back(u);
        }

        for (int u: X)
        {
            if (isKplex(u, missing2, P, g))
                X2.push_back(u);
        }

        bKplex(P, C2, X2, left, missing2, g);

        P.pop_back();
        X.push_back(v);
        //C.erase(C.begin() + i);
        //--i;
    }


}

void BKCPUAlgorithm(const graph<int> &g, unsigned int* blk, unsigned int blkCount, unsigned int* leftBase, unsigned int leftCount)
{

    vector<int> P;
    vector<int> C(blk+1, blk+blkCount);
    vector<int> X;
    vector<int> missing(g.n, 0);
    //int arr2[1] = {21};
    vector<int> left(leftBase, leftBase+leftCount);
    
    P.push_back(blk[0]);
    update_missing_add(blk[0], missing, C, g);
    vector<int> C2;
    for (int u: C)
    {
        if (isKplex(u, missing, P, g))
            C2.push_back(u);
    }

    bKplex(P, C2, X, left, missing, g);
}

void decomposableSearch(const graph<int> &g)
{
    int *dpos = new int[g.n];
    int *dseq = new int[g.n];
    bool *mark = new bool[g.n];
    int *resNei = new int[g.n];
    #pragma omp parallel for
    for(int i=0;i<g.n;++i)dpos[i]=INT_MAX;
    // size_t cntMaxPlex=0;
    unsigned int *validblk;
    unsigned int h_validblk;
    
    graph<intT> peelG = peelGraph(g,mark,resNei);
    const int pn=peelG.n;
    volatile bool* const ready=(volatile bool*)mark;

    #pragma omp for
    for(int i=0;i<pn;++i) mark[i]=false;
    #pragma omp master
    {
        ListLinearHeap<int> *linear_heap = new ListLinearHeap<int>(pn, pn-1); 
        linear_heap->init(pn, pn-1);
        for(int i=0;i<pn;++i){
            linear_heap->insert(i,peelG.degree[i]);
        }
        for (int i = 0; i < pn; i++) {
            int u, key;
            linear_heap->pop_min(u, key);
            dpos[u] = i;
            dseq[i] = u;
            ready[i] = true;
            for (int j = 0; j < peelG.degree[u]; j++){
                //const int nei = peelG.V[u].Neighbors[j];
                const int nei = peelG.neighbors[peelG.offsets[u]+j];
                if(dpos[nei]==INT_MAX) {
                    linear_heap->decrement(nei);
                }
            }
        }
        delete linear_heap;
    }

    P_pointers plex_pointers;
    plex_pointers.k = k;
    plex_pointers.lb = lb;
    plex_pointers.bd = bd;

    G_pointers graph_pointers;
    D_pointers degen_pointers;
    S_pointers subgraph_pointers;

    printf("Start copying graph to GPU....\n");
    copy_graph_to_gpu<intT>(peelG, dpos, dseq, graph_pointers, degen_pointers, subgraph_pointers);
    printf("Done copying graph to GPU....\n");

    unsigned int *d_blk;
    unsigned int *d_left;
    unsigned int *d_blk_counter;
    unsigned int *d_left_counter;
    unsigned int *d_visited;
    unsigned int *d_count;
    unsigned int *d_hopSz;
    unsigned int *global_count;
    unsigned int *left_count;
    unsigned int *plex_count;
    unsigned int *neiInG;
    unsigned int *neiInP;
    unsigned int *nonNeigh;
    unsigned int *nonNeighLeft;
    unsigned int *stack;
    unsigned int *depth;
    unsigned int h_global_count;
    unsigned int h_left_count;
    unsigned int h_plex_count;

    thrust::device_ptr<unsigned int> deg_ptr (subgraph_pointers.degree);
    thrust::device_ptr<unsigned int> off_ptr (subgraph_pointers.offsets);

    cudaMalloc(&global_count, sizeof(unsigned int));
    cudaMemset(global_count,0,sizeof(unsigned int));

    cudaMalloc(&left_count, sizeof(unsigned int));
    cudaMemset(left_count,0,sizeof(unsigned int));

    cudaMalloc(&plex_count, sizeof(unsigned int));
    cudaMemset(plex_count,0,sizeof(unsigned int));

    cudaMalloc(&validblk, sizeof(unsigned int));
    cudaMemset(validblk,0,sizeof(unsigned int));
    // blk = the seed node + it's direct neighbors + it's two-hop neighbors
    cudaMalloc(&d_blk, MAX_BLK_SIZE * WARPS * sizeof(unsigned int)); // 40 with 1024 threads eachs. each block has 32 warps
    cudaMalloc(&d_blk_counter, WARPS*sizeof(unsigned int));

    cudaMalloc(&nonNeigh, MAX_BLK_SIZE*WARPS*sizeof(unsigned int));
    cudaMalloc(&nonNeighLeft, MAX_BLK_SIZE*WARPS*sizeof(unsigned int));
    cudaMalloc(&stack, MAX_BLK_SIZE*WARPS*sizeof(unsigned int));
    cudaMalloc(&depth, MAX_BLK_SIZE*WARPS*sizeof(unsigned int));
    cudaMalloc(&neiInG, MAX_BLK_SIZE * WARPS * sizeof(unsigned int));
    cudaMalloc(&neiInP, MAX_BLK_SIZE * WARPS * sizeof(unsigned int));

    cudaMalloc(&d_left, MAX_BLK_SIZE*WARPS*sizeof(unsigned int));
    cudaMalloc(&d_left_counter, WARPS * sizeof(unsigned int));
    cudaMalloc(&d_hopSz, WARPS * sizeof(unsigned int));

    cudaMalloc(&d_visited, pn * WARPS * sizeof(int)); // 40 million nodes, 1 million nodes
    cudaMalloc(&d_count, pn * WARPS * sizeof(int));
    
    cudaMemset(d_blk_counter, 0, WARPS*sizeof(unsigned int));
    cudaMemset(d_left_counter, 0, WARPS*sizeof(unsigned int));
    cudaMemset(nonNeigh, 0, MAX_BLK_SIZE * WARPS*sizeof(unsigned int));
    cudaMemset(nonNeighLeft, 0, MAX_BLK_SIZE * WARPS*sizeof(unsigned int));
    cudaMemset(depth, 0, MAX_BLK_SIZE * WARPS*sizeof(unsigned int));
    cudaMemset(d_hopSz, 0, WARPS * sizeof(unsigned int));
    cudaMemset(d_visited, 0, pn * WARPS * sizeof(unsigned int)); // creating a binary structure, threshold and adaptive based on size
    cudaMemset(d_count, 0, pn * WARPS * sizeof(unsigned int));

    //------------CPU Testing only--------
    // unsigned int *h_blk          = (unsigned int*)malloc(MAX_BLK_SIZE * WARPS * sizeof(unsigned int));
    // unsigned int *h_left         = (unsigned int*)malloc(MAX_BLK_SIZE * WARPS * sizeof(unsigned int));
    // unsigned int *h_blk_counter  = (unsigned int*)malloc(      WARPS * sizeof(unsigned int));
    // unsigned int *h_left_counter = (unsigned int*)malloc(      WARPS * sizeof(unsigned int));
    //--------------------------------------

    cudaEvent_t event_start;
    cudaEvent_t event_stop;
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);
    cudaEventRecord(event_start);
    //printf("Number of iterations: %f\n", double(pn/WARPS));
    
    // cudaDeviceSetLimit(cudaLimitStackSize, 32*1024);
    for (int i = 0; i < (pn/WARPS)+1; i++) 
    {
        decompose<<<BLK_NUMS, BLK_DIM>>>(i, plex_pointers, graph_pointers, degen_pointers, d_blk, d_blk_counter, d_left, d_left_counter, d_visited, d_count, global_count, left_count, validblk, d_hopSz);
        cudaDeviceSynchronize();
        cudaMemset(global_count,0,sizeof(unsigned int));
        cudaMemset(left_count,0,sizeof(unsigned int));
        calculateDegrees<<<BLK_NUMS, BLK_DIM>>>(i, plex_pointers, graph_pointers, subgraph_pointers, d_blk, d_blk_counter, d_left, d_left_counter, global_count, left_count);
        cudaDeviceSynchronize();
        computeOffsets(subgraph_pointers, d_blk_counter);
        cudaDeviceSynchronize();
        fillNeighbors<<<BLK_NUMS, BLK_DIM>>>(i, subgraph_pointers, graph_pointers, d_blk, d_blk_counter, d_left, d_left_counter, d_hopSz, neiInG);
        cudaDeviceSynchronize();
        cudaMemset(global_count,0,sizeof(unsigned int));
        //kSearch<<<BLK_NUMS, BLK_DIM>>>(i, plex_pointers, subgraph_pointers, graph_pointers, d_blk, d_left, neiInG, neiInP, d_hopSz, plex_count, nonNeigh, depth);
        BKIterative<<<BLK_NUMS, BLK_DIM>>>(i, plex_pointers, subgraph_pointers, graph_pointers, d_blk, d_left, d_left_counter, plex_count, nonNeigh, nonNeighLeft, depth, stack, global_count);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return;
    }
    cudaDeviceSynchronize();
    // cudaMemcpy(h_blk,          d_blk,          MAX_BLK_SIZE * WARPS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_left,         d_left,         MAX_BLK_SIZE * WARPS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_blk_counter,  d_blk_counter,         WARPS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_left_counter, d_left_counter,        WARPS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < WARPS; i++)
    // {
    //     printf("i = %d\n", i);
    //     if (i >= g.n) break;
    //     //printf("Hello before accessing local chunk of %d with %d\n", i, g.n);
    //     unsigned int * blkBase = h_blk + i * MAX_BLK_SIZE;
    //     unsigned int blkCount = h_blk_counter[i];
    //     unsigned int * leftBase = h_left + i * MAX_BLK_SIZE;
    //     unsigned int leftCount = h_left_counter[i];
    //     //if (blkCount - 1 < bd) continue;
    //     //printf("Hello after accessing local chunk of %d\n", i);
    //     // unsigned int blkBase[6] = {2,3,1,0,4,5};
    //     // unsigned int blkCount = 6;
    //     // unsigned int leftBase[0] = {};
    //     // unsigned int leftCount = 0;
    //     BKCPUAlgorithm(peelG, blkBase, blkCount, leftBase, leftCount);
    //     printf("%d is done with maximal k-plexes: %d\n", i, plexCnt);
    //     //printf("out of bK\n");
    // }
    // //printf("Out of for loop\n");
    // printf("Maximal k-Plexes: %d\n", plexCnt);

    cudaEventRecord(event_stop);
    cudaEventSynchronize(event_stop);
    float time_milli_sec = 0;
    cudaEventElapsedTime(&time_milli_sec, event_start, event_stop);
    printf("Total Time Elapsed: %f ms\n", time_milli_sec);

    cudaMemcpy(&h_global_count, global_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_left_count, left_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_plex_count, plex_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_validblk, validblk, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("Total Valid Blocks: %d, Total Block Size: %d, Total Left Size: %d, Maximal k-Plexes: %d\n", h_validblk, h_global_count, h_left_count, h_plex_count);
    printf("\nKernel Launch Successfully\n");
    free_graph_gpu_memory(graph_pointers, degen_pointers);
}

#endif //CUTS_HOST_FUNCS_H