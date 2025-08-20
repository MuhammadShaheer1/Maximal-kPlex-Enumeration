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

int plexCnt = 0;
unsigned int blkCnt = 0;
unsigned int leftCnt = 0;
unsigned int *d_block;
std::queue<State> taskQueue;
std::queue<State2> tQueue;
// std::queue<Task> TQ;

bool isNeighbor(int u, int v, const graph<int> &g)
{
    int begin = g.offsets[u];
    int end = g.offsets[u + 1];
    for (int i = begin; i < end; i++)
    {
        if (g.neighbors[i] == v)
            return true;
    }
    return false;
}

bool isLeftNeighbor(int u, int v, const graph<int> &g)
{
    int begin = g.offsetsLeft[u];
    int end = g.offsetsLeft[u + 1];
    for (int i = begin; i < end; i++)
    {
        if (g.neighborsLeft[i] == v)
            return true;
    }
    return false;
}

void update_missing_add(int v, vector<int> &missing, const vector<int> &U, const graph<int> &g)
{
    for (int u : U)
    {
        if (!isNeighbor(v, u, g))
            missing[u]++;
    }
}

void update_missing_remove(int v, vector<int> &missing, const vector<int> &U, const graph<int> &g)
{
    for (int u : U)
    {
        if (!isNeighbor(v, u, g))
            missing[u]--;
    }
}

bool isKplex(int v, vector<int> &missing, const vector<int> &U, const graph<int> &g)
{
    if (missing[v] > (k - 1))
    {
        return false;
    }

    for (int u : U)
    {
        if (missing[u] == (k - 1) && !isNeighbor(v, u, g))
            return false;
    }
    return true;
}

bool isKplex4(int v, vector<int> &neiInP, Stack<int> &P, const graph<int> &g)
{
    if (neiInP[v] + k < (P.sz + 1))
    {
        return false;
    }

    for (int i = 0; i < P.sz; i++)
    {
        int u = P.members[i];
        if (neiInP[u] + k == P.sz && !isNeighbor(v, u, g))
            return false;
    }
    return true;
}

bool isMaximal(vector<int> missing, vector<int> P, vector<int> left, const graph<int> &g)
{
    // printf("Count: ");
    for (int u : left)
    {
        int count = 0;
        for (int v : P)
        {
            if (!isNeighbor(v, u, g))
                count++;
        }
        // printf("%d ", count);
        if (count > k - 1)
            continue;
        bool validExtension = true;
        for (int v : P)
        {
            if (!isNeighbor(v, u, g))
            {
                if (missing[v] >= (k - 1))
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
    // printf("\n");
    return true;
}

void bKplex(vector<int> &P, vector<int> &C, vector<int> &X, vector<int> &left, vector<int> &missing, const graph<int> &g)
{
    printf("Entering BK Plex\n");
    printf("PSize: %d, CSize: %d, XSize: %d, LeftSz: %d\n", P.size(), C.size(), X.size(), left.size());
    printf("Maximal k-plexes: %d\n", plexCnt);
    printf("P: ");
    for (int u : P)
    {
        printf("%d ", u);
    }
    printf("\n");
    printf("C: ");
    for (int u : C)
    {
        printf("%d ", u);
    }
    printf("\n");
    printf("X: ");
    for (int u : X)
    {
        printf("%d ", u);
    }
    printf("\n");
    printf("Missing: ");
    for (int u : missing)
    {
        printf("%d ", u);
    }
    printf("\n");
    // printf("Hello! into BK plex\n");
    if (C.empty() && X.empty())
    {
        if (P.size() >= lb && isMaximal(missing, P, left, g))
        {
            // printf("Maximal k-plex found\n");
            // printf("P: ");
            // for (int u: P)
            // {
            //     printf("%d ", u);
            // }
            // printf("\n");
            plexCnt++;
        }
        return;
    }
    vector<int> candList = C;
    for (int v : candList)
    {
        auto it = find(C.begin(), C.end(), v);
        if (it == C.end())
            return;

        C.erase(it);

        vector<int> C2, X2;
        vector<int> missing2 = missing;

        update_missing_add(v, missing2, C, g);
        update_missing_add(v, missing2, X, g);
        update_missing_add(v, missing2, P, g);

        P.push_back(v);
        for (int u : C)
        {
            if (isKplex(u, missing2, P, g))
                C2.push_back(u);
        }

        for (int u : X)
        {
            if (isKplex(u, missing2, P, g))
                X2.push_back(u);
        }
        // taskQueue.emplace(P, C2, X2, missing2, left);
        bKplex(P, C2, X2, left, missing2, g);

        P.pop_back();
        X.push_back(v);
        // C.erase(C.begin() + i);
        //--i;
    }
}

bool isKplex3(int v, unsigned int *missing, uint8_t *labels, const graph<int> &g)
{
    if (missing[v] > (k - 1))
    {
        return false;
    }

    for (int i = 0; i < g.n; i++)
    {
        if (labels[i] == P)
        {
            if (missing[i] == (k - 1) && !isNeighbor(v, i, g))
                return false;
        }
    }
    return true;
}

bool isKplexPC(int v, unsigned int *missing, uint8_t *labels, const graph<int> &g, unsigned int PlexSz)
{
    if (missing[v] + k < PlexSz + 1 /* || missing[v] + k < max(lb, PlexSz) + 1*/)
    {
        return false;
    }

    for (int i = 0; i < g.n; i++)
    {
        if (labels[i] == P || labels[i] == C)
        {
            if (missing[i] + k == (PlexSz) && !isNeighbor(v, i, g))
                return false;
        }
    }
    return true;
}

bool isMaximal3(unsigned int *missing, uint8_t *labels, unsigned int *left, unsigned int leftCount, const graph<int> &g)
{
    // printf("Count: ");
    for (int u = 0; u < leftCount; u++)
    {
        int count = 0;
        for (int v = 0; v < g.n; v++)
        {
            if (labels[v] == P)
            {
                if (!isLeftNeighbor(v, u, g))
                    count++;
            }
        }
        // printf("%d ", count);
        if (count > k - 1)
            continue;
        bool validExtension = true;
        for (int v = 0; v < g.n; v++)
        {
            if (labels[v] == P)
            {
                if (!isLeftNeighbor(v, u, g))
                {
                    if (missing[v] >= (k - 1))
                    {
                        validExtension = false;
                        break;
                    }
                }
            }
        }
        if (validExtension)
        {
            return false;
        }
    }
    // printf("\n");
    return true;
}

bool isMaximalPC(unsigned int *missing, unsigned int PlexSz, uint8_t *labels, unsigned int *left, unsigned int leftCount, const graph<int> &g)
{
    // printf("Count: ");
    for (int u = 0; u < leftCount; u++)
    {
        int count = 0;
        for (int v = 0; v < g.n; v++)
        {
            if (labels[v] == P || labels[v] == C)
            {
                if (!isLeftNeighbor(v, u, g))
                    count++;
            }
        }
        // printf("%d ", count);
        if (count > k - 1)
            continue;
        bool validExtension = true;
        for (int v = 0; v < g.n; v++)
        {
            if (labels[v] == P || labels[v] == C)
            {
                if (!isLeftNeighbor(v, u, g))
                {
                    if (missing[v] + k < PlexSz + 1)
                    {
                        validExtension = false;
                        break;
                    }
                }
            }
        }
        if (validExtension)
        {
            return false;
        }
    }
    // printf("\n");
    return true;
}

void subG(int i, unsigned int *neiInG, const graph<int> &g)
{
    for (int j = 0; j < g.n; j++)
    {
        if (i == j)
            continue;
        if (isNeighbor(i, j, g))
            neiInG[j]--;
    }
}

void addG(int i, unsigned int *neiInG, const graph<int> &g)
{
    for (int j = 0; j < g.n; j++)
    {
        if (i == j)
            continue;
        if (isNeighbor(i, j, g))
            neiInG[j]++;
    }
}


void BKCPUAlgorithm(const graph<int> &g, unsigned int *blk, unsigned int blkCount, unsigned int *leftBase, unsigned int leftCount)
{

    vector<int> P;
    vector<int> C(blk + 1, blk + blkCount);
    vector<int> X;
    vector<int> missing(g.n, 0);
    // int arr2[1] = {21};
    vector<int> left(leftBase, leftBase + leftCount);

    P.push_back(blk[0]);
    update_missing_add(blk[0], missing, C, g);
    vector<int> C2;
    for (int u : C)
    {
        if (isKplex(u, missing, P, g))
            C2.push_back(u);
    }

    bKplex(P, C2, X, left, missing, g);
}

// bool upperBound(Stack<int> &P, VtxSet &C1, vector<int> &neiInG, vector<int> &neiInP, vector<int> &nonadjInP, const graph<int> &g)
// {
//     int cnt = P.sz;
//     for (int t = 0; t < P.sz; t++)
//     {
//         int u = P.members[t];
//         if (neiInG[u] + k < lb)
//             return false;
//         nonadjInP[t] = P.sz - neiInP[u];
//     }
//     //return true;
//     if (!g.proper)
//         return true;

//     for (int i = 0; i < C1.sz; i++)
//     {
//         int max_noadj = -1;
//         int max_index = 0;
//         int ele = C1.members[i];
//         for (int j = 1; j < P.sz; j++)
//         {
//             int v = P.members[j];
//             if (!isNeighbor(ele, v, g) && nonadjInP[j] > max_noadj)
//             {
//                 max_noadj = nonadjInP[j];
//                 max_index = j;
//             }
//         }
//         if (max_noadj < k)
//         {
//             cnt++;
//             nonadjInP[max_index]++;
//         }
//         if (cnt >= lb)
//             return true;
//     }
//     return cnt >= lb;
// }

// void subG2(int i, vector<int> &neiInG, const graph<int> &g)
// {
//     for (int j = 0; j < g.degree[i]; j++)
//     {
//         int nei = g.neighbors[g.offsets[i] + j];
//         neiInG[nei]--;
//     }
// }

// void addG2(int i, vector<int> &neiInG, const graph<int> &g)
// {
//     for (int j = 0; j < g.degree[i]; j++)
//     {
//         int nei = g.neighbors[g.offsets[i] + j];
//         neiInG[nei]++;
//     }
// }
// void updateCand1KFake(int &recCand1, const int v2add, VtxSet &C1, const graph<int> &g, vector<int> &neiInG)
// {
//     recCand1 = C1.sz;
//     for (int i = 0; i < C1.sz;)
//     {
//         int ele = C1.members[i];
//         if (g.proper && UNLINK2EQUAL > g.commonMtx[v2add * g.n + ele])
//         {
//             C1.fakeRemove(ele);
//             subG2(ele, neiInG, g);
//         }
//         else
//             ++i;
//     }
//     recCand1 -= C1.sz;
// }

// void updateCand2Fake(int &recCand2, const int v2add, VtxSet &C2, const graph<int> &g)
// {
//     recCand2 = C2.sz;
//     for (int i = 0; i < C2.sz;)
//     {
//         int ele = C2.members[i];
//         if (g.proper && UNLINK2EQUAL > g.commonMtx[v2add * g.n + ele])
//         {
//             C2.fakeRemove(ele);
//         }
//         else
//             ++i;
//     }
//     recCand2 -= C2.sz;
// }

// void updateExclK(int &recExcl, const int v2add, VtxSet &X, const graph<int> &g, Stack<int> &exclStack)
// {
//     recExcl = X.sz;
//     for (int i = 0; i < X.sz;)
//     {
//         int ele = X.members[i];
//         if (g.proper && UNLINK2MORE > g.commonMtx[v2add * g.n + ele])
//         {
//             X.remove(ele);
//             exclStack.push(ele);
//         }
//         else
//             ++i;
//     }
//     recExcl -= X.sz;
// }

// void fakeRecoverAdd(int len, VtxSet &C1, const graph<int> &g, vector<int> &neiInG)
// {
//     int *cursor = C1.members + C1.sz;
//     for (int i = 0; i < len; i++)
//     {
//         const int ele = *cursor;
//         addG2(ele, neiInG, g);
//         cursor++;
//     }
//     C1.sz += len;
// }

// void fakeRecoverAddC2(int len, VtxSet &C2, const graph<int> &g)
// {
//     int *cursor = C2.members + C2.sz;
//     for (int i = 0; i < len; i++)
//     {
//         const int ele = *cursor;
//         cursor++;
//     }
//     C2.sz += len;
// }

// void kSearchIter(int idx, int res, Stack<int> &P, VtxSet &C1, VtxSet &C2, VtxSet &X, const graph<int> &g, vector<int> &neiInG, vector<int> &neiInP, vector<int> &nonadjInP, Stack<int> &exclStack)
// {
//     vector<Frame> stack;
//     Frame f(res, 1, 0);
//     stack.push_back(f);
//     int v2delete;
//     int v2add;
//     while (!stack.empty())
//     {
//         Frame f = stack.back();
//         stack.pop_back();
//         switch (f.state)
//         {
//         case 0:
//             //printf("case 0\n");
//             if (C2.sz == 0)
//             {
//                 if (P.sz + C1.sz < lb)
//                     continue;
//                 if (P.sz > 1 && !upperBound(P, C1, neiInG, neiInP, nonadjInP, g))
//                     continue;
//                 uint8_t *labels = new uint8_t[g.n];
//                 unsigned int *neiInG2 = new unsigned int[g.n];
//                 unsigned int *neiInP2 = new unsigned int[g.n];

//                 for (int i = 0; i < g.n; i++)
//                 {
//                     labels[i] = 3;
//                 }

//                 for (int i = 0; i < P.sz; i++)
//                 {
//                     labels[P.members[i]] = 0;
//                 }
//                 for (int i = 0; i < C1.sz; i++)
//                 {
//                     labels[C1.members[i]] = 1;
//                 }
//                 for (int i = 0; i < X.sz; i++)
//                 {
//                     labels[X.members[i]] = 2;
//                 }
//                 for (int i = 0; i < C2.sz; i++)
//                 {
//                     labels[C2.members[i]] = 5;
//                 }
//                 for (int i = 0; i < g.n; i++)
//                 {
//                     neiInG2[i] = neiInG[i];
//                     neiInP2[i] = neiInP[i];
//                 }
//             //     printf("Task No. %d\n", TQ.size());
//             // printf("idx: %d, PlexSz: %d, CandSz: %d, ExclSz: %d\n", idx, P.sz, C1.sz, X.sz);
//             // printf("labels: \n");
//             // for (int i = 0; i < g.n; i++)
//             // {
//             //   printf("%d ", labels[i]);
//             // }
//             // printf("\n");
//             // printf("neiInG: \n");
//             // for (int i = 0; i < g.n; i++)
//             // {
//             //   printf("%d ", neiInG2[i]);
//             // }
//             // printf("\n");
//             // printf("neiInP: \n");
//             // for (int i = 0; i < g.n; i++)
//             // {
//             //   printf("%d ", neiInP2[i]);
//             // }
//             // printf("\n");
//                 TQ.emplace(idx, P.sz, C1.sz, X.sz, labels, neiInG2, neiInP2);
//                 continue;
//             }

//             v2delete = C2.pop_back();
//             X.add(v2delete);
//             f.v2delete = v2delete;
//             f.state = 1;
//             stack.push_back(f);
//             f.state = 0;
//             f.v2adds = {};
//             stack.push_back(f);
//             continue;
//         case 1:
//             //printf("Case 1\n");
//             X.remove(f.v2delete);
//             C2.add(f.v2delete);
//             f.state = 2;
//             stack.push_back(f);
//             continue;

//         case 2:
//             //printf("Case 2\n");
//             if (f.br < f.res)
//             {
//                 v2add = C2.pop_back();
//                 P.push(v2add);
//                 for (int i = 0; i < g.degree[v2add]; i++)
//                 {
//                     int nei = g.neighbors[g.offsets[v2add] + i];
//                     neiInP[nei]++;
//                 }
//                 addG2(v2add, neiInG, g);
//                 f.v2adds.push_back(v2add);
//                 if (C2.sz)
//                 {
//                     v2delete = C2.pop_back();
//                     X.add(v2delete);
//                     f.v2delete = v2delete;
//                     f.state = 3;
//                     stack.push_back(f);
//                     f.res = f.res - f.br;
//                     f.br = 1;
//                     f.state = 0;
//                     f.v2adds = {};
//                     stack.push_back(f);
//                     continue;
//                 }
//                 else 
//                 {
//                     f.state = 4;
//                     stack.push_back(f);
//                     f.res = f.res - f.br;
//                     f.br = 1;
//                     f.state = 0;
//                     f.v2adds = {};
//                     stack.push_back(f);
//                     continue;
//                 }
//             }
//             else
//             {
//                 f.state = 4;
//                 stack.push_back(f);
//                 continue;
//             }
        
//         case 3:
//             //printf("Case 3\n");
//             X.remove(f.v2delete);
//             C2.add(f.v2delete);
//             f.br++;
//             f.state = 2;
//             stack.push_back(f);
//             continue;

//         case 4:
//             //printf("Case 4\n");
//             if (f.br == f.res)
//             {
//                 v2add = C2.pop_back();
//                 P.push(v2add);
//                 addG2(v2add, neiInG, g);

//                 for (int i = 0; i < g.degree[v2add]; i++)
//                 {
//                     int nei = g.neighbors[g.offsets[v2add] + i];
//                     neiInP[nei]++;
//                 }
//                 f.v2adds.push_back(v2add);
//                 VtxSet C12(g.n);
//                 VtxSet C22(g.n);
//                 vector<int> newNeiInG;
//                 newNeiInG = neiInG;
//                 for (int i = 0; i < C1.sz; i++)
//                 {
//                     int u = C1.members[i];
//                     if (isKplex4(u, neiInP, P, g))
//                         C12.add(u);
//                     else
//                         subG2(u, newNeiInG, g);
//                 }
//                 // for (int i = 0; i < C2.sz; i++)
//                 // {
//                 //     int u = C2.members[i];
//                 //     if (isKplex4(u, missing, P, g))
//                 //         C22.add(u);
//                 // }
//                 // printf("C1 Sz: %d\n", C12.sz);
//                 if (P.sz + C12.sz < lb)
//                 {
//                     f.state = 5;
//                     stack.push_back(f);
//                     continue;
//                 }
//                 if (P.sz > 1 && !upperBound(P, C12, newNeiInG, neiInP, nonadjInP, g))
//                 {
//                     f.state = 5;
//                     stack.push_back(f);
//                     continue;
//                 }
//                 uint8_t *labels = new uint8_t[g.n];
//                 unsigned int *neiInG2 = new unsigned int[g.n];
//                 unsigned int *neiInP2 = new unsigned int[g.n];

//                 for (int i = 0; i < g.n; i++)
//                 {
//                     labels[i] = 3;
//                 }

//                 for (int i = 0; i < P.sz; i++)
//                 {
//                     labels[P.members[i]] = 0;
//                 }
//                 for (int i = 0; i < C12.sz; i++)
//                 {
//                     labels[C12.members[i]] = 1;
//                 }
//                 for (int i = 0; i < X.sz; i++)
//                 {
//                     labels[X.members[i]] = 2;
//                 }
//                 for (int i = 0; i < C2.sz; i++)
//                 {
//                     labels[C2.members[i]] = 5;
//                 }
//                 for (int i = 0; i < g.n; i++)
//                 {
//                     neiInG2[i] = newNeiInG[i];
//                     neiInP2[i] = neiInP[i];
//                 }
//             //     printf("Task No. %d\n", TQ.size());
//             // printf("idx: %d, PlexSz: %d, CandSz: %d, ExclSz: %d\n", idx, P.sz, C1.sz, X.sz);
//             // printf("labels: \n");
//             // for (int i = 0; i < g.n; i++)
//             // {
//             //   printf("%d ", labels[i]);
//             // }
//             // printf("\n");
//             // printf("neiInG: \n");
//             // for (int i = 0; i < g.n; i++)
//             // {
//             //   printf("%d ", neiInG2[i]);
//             // }
//             // printf("\n");
//             // printf("neiInP: \n");
//             // for (int i = 0; i < g.n; i++)
//             // {
//             //   printf("%d ", neiInP2[i]);
//             // }
//             // printf("\n");
//                 TQ.emplace(idx, P.sz, C12.sz, X.sz, labels, neiInG2, neiInP2);
//                 f.state = 5;
//                 stack.push_back(f);
//                 continue;
//             }
//             else
//             {
//                 f.state = 5;
//                 stack.push_back(f);
//                 continue;
//             }

//         case 5:
//             //printf("Case 5\n");
//             for (int i = f.br; i >= 1; i--)
//             {
//                 v2add = f.v2adds.back();
//                 f.v2adds.pop_back();
//                 C2.add(v2add);
//                 if (P.top() != v2add) printf("v2add logic is not correct\n");
//                 P.pop();
//                 subG2(v2add, neiInG, g);
//                 for (int i = 0; i < g.degree[v2add]; i++)
//                 {
//                     int nei = g.neighbors[g.offsets[v2add] + i];
//                     neiInP[nei]--;
//                 }
//             }
        
//         // default:
//         //     printf("State value is corrupted\n");
//         //     break;
//         }
//     }
//     //printf("kSearch Running successfully\n");
// }

// void kSearchCPU(int idx, int res, Stack<int> &P, VtxSet &C1, VtxSet &C2, VtxSet &X, const graph<int> &g, vector<int> &neiInG, vector<int> &neiInP, vector<int> &nonadjInP, Stack<int> &exclStack)
// {
//     // printf("PSize: %d, C1Size: %d, C2Size: %d, XSize: %d, res: %d\n", P.sz, C1.sz, C2.sz, X.sz, res);
//     // printf("P: ");
//     // for (int i = 0; i < P.sz; i++)
//     // {
//     //     printf("%d ", P.members[i]);
//     // }
//     // printf("\n");
//     // printf("C1: ");
//     // for (int i = 0; i < C1.sz; i++)
//     // {
//     //     printf("%d ", C1.members[i]);
//     // }
//     // printf("\n");
//     // printf("C2: ");
//     // for (int i = 0; i < C2.sz; i++)
//     // {
//     //     printf("%d ", C2.members[i]);
//     // }
//     // printf("\n");
//     //  printf("X: ");
//     // for (int i = 0; i < X.sz; i++)
//     // {
//     //     printf("%d ", X.members[i]);
//     // }
//     // printf("\n");
//     int recExcl = 0, recExclTmp;
//     int recCand1[K_LIMIT], recCand2[K_LIMIT];
//     if (C2.sz == 0)
//     {
//         if (P.sz + C1.sz < lb)
//             return;
//         if (P.sz > 1 && !upperBound(P, C1, neiInG, neiInP, nonadjInP, g))
//             return;
//         uint8_t *labels = new uint8_t[g.n];
//         unsigned int *neiInG2 = new unsigned int[g.n];
//         unsigned int *neiInP2 = new unsigned int[g.n];

//         for (int i = 0; i < g.n; i++)
//         {
//             labels[i] = 3;
//         }

//         for (int i = 0; i < P.sz; i++)
//         {
//             labels[P.members[i]] = 0;
//         }
//         for (int i = 0; i < C1.sz; i++)
//         {
//             labels[C1.members[i]] = 1;
//         }
//         for (int i = 0; i < X.sz; i++)
//         {
//             labels[X.members[i]] = 2;
//         }
//         for (int i = 0; i < C2.sz; i++)
//         {
//             labels[C2.members[i]] = 5;
//         }
//         for (int i = 0; i < g.n; i++)
//         {
//             neiInG2[i] = neiInG[i];
//             neiInP2[i] = neiInP[i];
//         }
//         // printf("Emplacing a task\n");
//         // printf("neiInP: ");
//         // for (int i = 0; i < g.n; i++)
//         // {
//         //     printf("%d ", neiInP[i]);
//         // }
//         // printf("\n");
//         TQ.emplace(idx, P.sz, C1.sz, X.sz, labels, neiInG2, neiInP2);
//         return;
//     }

//     int v2delete = C2.pop_back();
//     X.add(v2delete);
//     // C2.pop_back();
//     // X.push_back(v2delete);
//     kSearchCPU(idx, res, P, C1, C2, X, g, neiInG, neiInP, nonadjInP, exclStack);
//     // X.pop_back();
//     // C2.push_back(v2delete);
    
//     X.remove(v2delete);
//     C2.add(v2delete);
//     int br = 1;
//     for (; br < res; br++)
//     {
//         // int v2add = C2.back();
//         // C2.pop_back();
//         int v2add = C2.pop_back();
//         P.push(v2add);
//         // P.push_back(v2add);
//         for (int i = 0; i < g.degree[v2add]; i++)
//         {
//             int nei = g.neighbors[g.offsets[v2add] + i];
//             neiInP[nei]++;
//         }
//         addG2(v2add, neiInG, g);
//         //printf("Pruning 1 C1Sz: %d\n", C1.sz);
//         updateCand1KFake(recCand1[br], v2add, C1, g, neiInG);
//         updateCand2Fake(recCand2[br], v2add, C2, g);
//         updateExclK(recExclTmp, v2add, X, g, exclStack);
//         recExcl += recExclTmp;
//         if (C2.sz)
//         {
//             v2delete = C2.pop_back();
//             X.add(v2delete);
//             //     C22.pop_back();
//             //     X.push_back(v2delete);
//             kSearchCPU(idx, res - br, P, C1, C2, X, g, neiInG, neiInP, nonadjInP, exclStack);
//             X.remove(v2delete);
//             C2.add(v2delete);
//             // X.pop_back();
//             // C22.push_back(v2delete);
//         }
//         else
//         {
//             kSearchCPU(idx, res - br, P, C1, C2, X, g, neiInG, neiInP, nonadjInP, exclStack);
//             break;
//         }
//     }
//     if (br == res)
//     {
//         recCand2[br] = 0;
//         // int v2add = C2.back();
//         // C2.pop_back();
//         int v2add = C2.pop_back();
//         P.push(v2add);

//         addG2(v2add, neiInG, g);

//         for (int i = 0; i < g.degree[v2add]; i++)
//         {
//             int nei = g.neighbors[g.offsets[v2add] + i];
//             // if (isNeighbor(i, v2add, g)) neiInP[i]++;
//             neiInP[nei]++;
//         }
//         // printf("before pruning 2: C1 Sz: %d\n", C1.sz);
//         // updateCand1KFake(recCand1[br], v2add, C1, g, neiInG);
//         // printf("C1 Sz: %d\n", C1.sz);
//         VtxSet C12(g.n);
//         VtxSet C22(g.n);
//         vector<int> newNeiInG;
//         newNeiInG = neiInG;
//         for (int i = 0; i < C1.sz; i++)
//         {
//             int u = C1.members[i];
//             if (isKplex4(u, neiInP, P, g))
//                 C12.add(u);
//             else
//                 subG2(u, newNeiInG, g);
//         }
//         // for (int i = 0; i < C2.sz; i++)
//         // {
//         //     int u = C2.members[i];
//         //     if (isKplex4(u, missing, P, g))
//         //         C22.add(u);
//         // }
//         // printf("C1 Sz: %d\n", C12.sz);
//         if (P.sz + C12.sz < lb)
//         {
//             goto restore;
//         }
//         if (P.sz > 1 && !upperBound(P, C12, newNeiInG, neiInP, nonadjInP, g))
//             goto restore;
//         uint8_t *labels = new uint8_t[g.n];
//         unsigned int *neiInG2 = new unsigned int[g.n];
//         unsigned int *neiInP2 = new unsigned int[g.n];

//         for (int i = 0; i < g.n; i++)
//         {
//             labels[i] = 3;
//         }

//         for (int i = 0; i < P.sz; i++)
//         {
//             labels[P.members[i]] = 0;
//         }
//         for (int i = 0; i < C12.sz; i++)
//         {
//             labels[C12.members[i]] = 1;
//         }
//         for (int i = 0; i < X.sz; i++)
//         {
//             labels[X.members[i]] = 2;
//         }
//         for (int i = 0; i < C2.sz; i++)
//         {
//             labels[C2.members[i]] = 5;
//         }
//         for (int i = 0; i < g.n; i++)
//         {
//             neiInG2[i] = newNeiInG[i];
//             neiInP2[i] = neiInP[i];
//         }
//         // printf("Emplacing2\n");
//         TQ.emplace(idx, P.sz, C12.sz, X.sz, labels, neiInG2, neiInP2);
//         // TQ.emplace(idx, labels, nonNeigh, P.size(), C12.size(), X.size());
//         // tQueue.emplace(P, C12, X, missing, left, blk);
//     }
// restore:
//     for (int i = br; i >= 1; i--)
//     {
//         // printf("Recovering\n");
//         // fakeRecoverAdd(recCand1[i], C1, g, neiInG);
//         // printf("C1Sz: %d\n", C1.sz);
//         // fakeRecoverAddC2(recCand2[i], C2, g);
//         // if (i != res)
//         // {
//         //     //printf("Recovering\n");
//         //     fakeRecoverAdd(recCand1[i], C1, g, neiInG);
//         // }
//         int v2add = P.top();
//         C2.add(v2add);
//         P.pop();
//         // P.pop_back();
//         subG2(v2add, neiInG, g);
//         // for (int j = 0; j < g.n; j++)
//         // {
//         //     if (isNeighbor(j, v2add, g)) neiInP[j]--;
//         // }
//         for (int i = 0; i < g.degree[v2add]; i++)
//         {
//             int nei = g.neighbors[g.offsets[v2add] + i];
//             neiInP[nei]--;
//         }
//     }
//     // for (int i = 0; i < recExcl; i++)
//     // {
//     //     X.add(exclStack.top());
//     //     exclStack.pop();
//     // }
// }

// void BKCPUAlgorithm2(int idx, const graph<int> &g)
// {
//     Stack<int> P(g.n);
//     VtxSet C1(g.n);
//     VtxSet C2(g.n);
//     VtxSet X(g.n);
//     Stack<int> exclStack(g.n);
//     vector<int> neiInG(g.degreeHop, g.degreeHop + g.n);
//     vector<int> neiInP(g.n, 0);
//     vector<int> nonadjInP(g.n, 0);
//     P.push(0);
//     for (int j = 1; j < g.n; j++)
//     {
//         if (j < g.hopSz)
//             C1.add(j);
//         else
//             C2.add(j);
//     }
//     for (int i = 0; i < g.n; i++)
//     {
//         if (isNeighbor(0, i, g))
//         {
//             neiInP[i]++;
//         }
//     }
//     // // printf("%d has %d neighbors\n", blk[0], g.degree[blk[0]]);
//     kSearchIter(idx, k - 1, P, C1, C2, X, g, neiInG, neiInP, nonadjInP, exclStack);
// }

// void allocateHostpointers(H_pointers &h, S_pointers s, unsigned int *d_blk_counter, unsigned int *d_hopSz, uint8_t *commonMtx)
// {
//     cudaMemcpy(h.h_degree, s.degree, MAX_BLK_SIZE * WARPS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
//     cudaMemcpy(h.h_degree_hop, s.degreeHop, MAX_BLK_SIZE * WARPS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
//     cudaMemcpy(h.h_neighbors, s.neighbors, MAX_BLK_SIZE * WARPS * AVG_DEGREE * sizeof(unsigned int), cudaMemcpyDeviceToHost);
//     cudaMemcpy(h.h_offsets, s.offsets, MAX_BLK_SIZE * WARPS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
//     cudaMemcpy(h.h_blk_counter, d_blk_counter, WARPS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
//     cudaMemcpy(h.h_hopSz, d_hopSz, WARPS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
//     cudaMemcpy(h.h_proper, s.proper, WARPS * sizeof(bool), cudaMemcpyDeviceToHost);
//     size_t capacity = size_t(WARPS) * CAP * sizeof(uint8_t);
//     cudaMemcpy(h.h_commonMtx, commonMtx, capacity, cudaMemcpyDeviceToHost);
//     //printf("common mtx: %d\n", h.h_commonMtx[1041+2927*CAP]);

//     // capacity = size_t(2230) * CAP;
//     // uint8_t* hMatrix = h.h_commonMtx + capacity;
//     // unsigned int counter = h.h_blk_counter[2230];
//     // printf("Counter: %d\n", counter);

//     //   printf("Common Matrix: \n");
//     //   for (int a = 0; a < counter; a++)
//     //     {
//     //       for (int b = 0; b < counter; b++)
//     //         {
//     //           printf("%d ", hMatrix[a*counter+b]);
//     //         }
//     //       printf("\n\n");
//     //     }
// }

// void initializeSubg(int i, graph<int> &subg, H_pointers host_pointers)
// {
//     unsigned int blkCount = host_pointers.h_blk_counter[i];
//     subg.degree = host_pointers.h_degree + i * MAX_BLK_SIZE;
//     subg.neighbors = host_pointers.h_neighbors + i * MAX_BLK_SIZE * AVG_DEGREE;
//     subg.offsets = host_pointers.h_offsets + i * MAX_BLK_SIZE;
//     subg.n = blkCount;
//     subg.degreeHop = host_pointers.h_degree_hop + i * MAX_BLK_SIZE;
//     subg.proper = host_pointers.h_proper[i];
//     subg.hopSz = host_pointers.h_hopSz[i];
//     subg.commonMtx = host_pointers.h_commonMtx + i * CAP;
// }

// void allocateTasks(T_pointers &t, H_pointers host_pointers, vector<Task> tasks)
// {
//     for (int i = 0; i < tasks.size(); i++)
//     {
//         auto &h = tasks[i];

//         cudaMemcpy(t.d_all_labels_A + i * MAX_BLK_SIZE, h.labels, host_pointers.h_blk_counter[h.idx] * sizeof(uint8_t), cudaMemcpyHostToDevice);

//         cudaMemcpy(t.d_all_neiInG_A + i * MAX_BLK_SIZE, h.neiInG, host_pointers.h_blk_counter[h.idx] * sizeof(unsigned int), cudaMemcpyHostToDevice);

//         cudaMemcpy(t.d_all_neiInP_A + i * MAX_BLK_SIZE, h.neiInP, host_pointers.h_blk_counter[h.idx] * sizeof(unsigned int), cudaMemcpyHostToDevice);

//         Task t1;
//         t1.idx = h.idx;
//         t1.PlexSz = h.PlexSz;
//         t1.CandSz = h.CandSz;
//         t1.ExclSz = h.ExclSz;
//         t1.labels = t.d_all_labels_A + i * MAX_BLK_SIZE;
//         t1.neiInG = t.d_all_neiInG_A + i * MAX_BLK_SIZE;
//         t1.neiInP = t.d_all_neiInP_A + i * MAX_BLK_SIZE;

//         cudaMemcpy(t.d_tasks_A + i, &t1, sizeof(Task), cudaMemcpyHostToDevice);
//     }
// }

void checkCudaError(int kernel)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error in %d: %s\n", kernel, cudaGetErrorString(err));
        return;
    }
}

void initializeBNB(int initialN, T_pointers &task_pointers, P_pointers plex_pointers, S_pointers subgraph_pointers, unsigned int *d_blk, unsigned int *d_left, unsigned int *d_blk_counter, unsigned int *d_left_counter, uint8_t *commonMtx, unsigned int *plex_count, unsigned int* nonAdjInP, uint16_t* d_sat, uint16_t* d_commons, uint32_t* d_uni, unsigned long long* cycles, uint32_t* d_adj)
{
    //cudaMemcpy(task_pointers.d_tail_A, &initialN, sizeof(unsigned int), cudaMemcpyHostToDevice);
    unsigned int head = 0;
    while (true)
    {
        unsigned int tail;
        cudaMemcpy(&tail, task_pointers.d_tail_A, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        //printf("tail: %d\n", tail);
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
            //printf("tail inside: %d\n", tail);
            if (tail == 0)
                break;
            cudaMemset(tail_out, 0, sizeof(unsigned int));
            unsigned int numTasks = tail;
            unsigned int waves = (numTasks) / WARPS + 1;

            for (unsigned int w = 0; w < waves; w++)
            {
                //printf("wave %d\n", w);
                // cudaEvent_t event_start;
                // cudaEvent_t event_stop;
                // cudaEventCreate(&event_start);
                // cudaEventCreate(&event_stop);
                // cudaEventRecord(event_start);
                BNB<<<BLK_NUMS, BLK_DIM>>>(w, plex_pointers, subgraph_pointers, d_blk, d_left, d_blk_counter, d_left_counter, commonMtx, Q_in, Q_out, task_pointers.d_tasks_A, numTasks, 0, tail_out, task_pointers.d_tail_A, lab_out, nei_out, P_out, task_pointers.d_all_labels_A, task_pointers.d_all_neiInG_A, task_pointers.d_all_neiInP_A, plex_count, nonAdjInP, d_sat, d_commons, d_uni, cycles, d_adj);
                // cudaEventRecord(event_stop);
                // cudaEventSynchronize(event_stop);
                // float time_milli_sec = 0;
                // cudaEventElapsedTime(&time_milli_sec, event_start, event_stop);
                // time[0] += time_milli_sec;
            }
            cudaDeviceSynchronize();
            checkCudaError(6);
            //cudaMemcpy(&tail, tail_out, sizeof(unsigned int), cudaMemcpyDeviceToHost);
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
    // size_t cntMaxPlex=0;
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
                // const int nei = peelG.V[u].Neighbors[j];
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
    H_pointers host_pointers;
    T_pointers task_pointers;

    host_pointers.h_degree = (unsigned int *)malloc(MAX_BLK_SIZE * WARPS * sizeof(unsigned int));
    host_pointers.h_degree_hop = (unsigned int *)malloc(MAX_BLK_SIZE * WARPS * sizeof(unsigned int));
    host_pointers.h_offsets = (unsigned int *)malloc(MAX_BLK_SIZE * WARPS * sizeof(unsigned int));
    host_pointers.h_neighbors = (unsigned int *)malloc(MAX_BLK_SIZE * WARPS * AVG_DEGREE * sizeof(unsigned int));
    host_pointers.h_blk_counter = (unsigned int *)malloc(WARPS * sizeof(unsigned int));
    host_pointers.h_proper = (bool *)malloc(WARPS * sizeof(bool));
    host_pointers.h_hopSz = (unsigned int *)malloc(WARPS * sizeof(unsigned int));
    size_t hostCapacity = size_t(WARPS) * CAP * sizeof(uint8_t);
    host_pointers.h_commonMtx = (uint8_t *)malloc(hostCapacity);

    printf("Start copying graph to GPU....\n");
    copy_graph_to_gpu<intT>(peelG, dpos, dseq, graph_pointers, degen_pointers, subgraph_pointers);
    printf("Done copying graph to GPU....\n");

    // size_t words_per_warp = (pn/32)+1;
    // size_t total_words = WARPS * words_per_warp;

    //printf("words_per_warp: %d\n", words_per_warp);

    unsigned int *d_blk;
    unsigned int *d_blk_counter;
    unsigned int *d_left;
    unsigned int *d_left_counter;
    uint8_t *d_visited;
    //unsigned int *d_count;
    unsigned int *d_hopSz;
    unsigned int *global_count;
    unsigned int *left_count;
    unsigned int *plex_count;
    uint8_t *commonMtx;
    unsigned int h_global_count;
    unsigned int h_left_count;
    unsigned int h_plex_count;

    uint16_t *d_sat;
    uint16_t *d_commons;
    uint32_t *d_uni;
    uint32_t *d_adj;

    unsigned int* d_res;
    unsigned int* d_br;
    unsigned int* d_state;
    unsigned int* d_v2delete;
    // unsigned int* d_v2adds;

    unsigned int* recCand1;
    unsigned int* recCand2;
    unsigned int* recExcl;

    uint16_t* neiInG;
    uint16_t* neiInP;
    unsigned int* nonAdjInP;

    

    unsigned long long* cycles;

    cudaMalloc(&d_res, WARPS * MAX_DEPTH * sizeof(unsigned int));
    cudaMalloc(&d_br, WARPS * MAX_DEPTH * sizeof(unsigned int));
    cudaMalloc(&d_state, WARPS * MAX_DEPTH * sizeof(unsigned int));
    cudaMalloc(&d_v2delete, WARPS * MAX_DEPTH * sizeof(unsigned int));
    // cudaMalloc(&d_v2adds, WARPS * STACKSIZE * sizeof(unsigned int));

    cudaMalloc(&recCand1, WARPS * MAX_BLK_SIZE * sizeof(unsigned int));
    cudaMalloc(&recCand2, WARPS * MAX_BLK_SIZE * sizeof(unsigned int));
    cudaMalloc(&recExcl, WARPS * MAX_BLK_SIZE * sizeof(unsigned int));

    cudaMalloc(&neiInG, WARPS * MAX_BLK_SIZE * sizeof(uint16_t));
    cudaMalloc(&neiInP, WARPS * MAX_BLK_SIZE * sizeof(uint16_t));
    cudaMalloc(&nonAdjInP, WARPS * MAX_BLK_SIZE * sizeof(unsigned int));

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

    cudaMalloc(&d_visited, pn * WARPS * sizeof(uint8_t));
    cudaMalloc(&cycles, BLK_NUMS * sizeof(unsigned long long));
    //cudaMalloc(&d_count, pn * WARPS * sizeof(unsigned int));

    cudaMemset(d_blk_counter, 0, WARPS * sizeof(unsigned int));
    cudaMemset(d_left_counter, 0, WARPS * sizeof(unsigned int));
    cudaMemset(d_hopSz, 0, WARPS * sizeof(unsigned int));
    cudaMemset(d_visited, 0, pn * WARPS * sizeof(uint8_t));
    //cudaMemset(d_count, 0, pn * WARPS * sizeof(uint16_t));
    cudaMemset(commonMtx, 0, totalBytes);
    cudaMemset(recCand1, 0, WARPS * MAX_BLK_SIZE * sizeof(unsigned int));
    cudaMemset(recCand2, 0, WARPS * MAX_BLK_SIZE * sizeof(unsigned int));
    cudaMemset(recExcl, 0, WARPS * MAX_BLK_SIZE * sizeof(unsigned int));
    cudaMemset(d_uni, 0, WARPS * 32 * sizeof(uint32_t));
    cudaMemset(subgraph_pointers.Pset, 0, WARPS * 32 * sizeof(uint32_t));
    cudaMemset(subgraph_pointers.Cset, 0, WARPS * 32 * sizeof(uint32_t));
    cudaMemset(subgraph_pointers.C2set, 0, WARPS * 32 * sizeof(uint32_t));
    cudaMemset(subgraph_pointers.Xset, 0, WARPS * 32 * sizeof(uint32_t));
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

    // cudaEvent_t event_start;
    // cudaEvent_t event_stop;
    // cudaEventCreate(&event_start);
    // cudaEventCreate(&event_stop);
    cudaEventRecord(event_start);
    //printf("Number of iterations: %f\n", double(pn / WARPS));
    float time_1 = 0;
    float time_2 = 0;
    float time_3 = 0;
    float time_4 = 0;
    float time_5 = 0;
    float time_6 = 0;
    float time_7 = 0;

    unsigned long long* h_cycles;
    h_cycles = (unsigned long long *)malloc(40 * sizeof(unsigned long long));


    // cudaDeviceSetLimit(cudaLimitStackSize, 32 * 1024);
    // int maxShmPerBlock = 0;
    // cudaDeviceGetAttribute(&maxShmPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    // printf("Shared Memory: %d\n", maxShmPerBlock);
    //cudaDeviceSynchronize();
    for (int i = 1; i < /*(pn/WARPS)+*/2; i++)
    {
        decompose<<<BLK_NUMS, BLK_DIM>>>(i, plex_pointers, graph_pointers, degen_pointers, d_blk, d_blk_counter, d_left, d_left_counter, d_visited, global_count, left_count, validblk, d_hopSz);
        // cudaDeviceSynchronize();
        // checkCudaError(0);
        // cudaMemset(global_count, 0, sizeof(unsigned int));
        // cudaMemset(left_count, 0, sizeof(unsigned int));

        // cudaEventRecord(event_stop);
        // cudaEventSynchronize(event_stop);
        // float time_milli_sec = 0;
        // cudaEventElapsedTime(&time_milli_sec, event_start, event_stop);
        // time_1 += time_milli_sec;
        // cudaEventRecord(event_start);

        calculateDegrees<<<BLK_NUMS, BLK_DIM>>>(i, plex_pointers, graph_pointers, subgraph_pointers, d_blk, d_blk_counter, d_left, d_left_counter, global_count, left_count);
        // cudaDeviceSynchronize();
        // checkCudaError(1);

        // cudaEventRecord(event_stop);
        // cudaEventSynchronize(event_stop);
        // time_milli_sec = 0;
        // cudaEventElapsedTime(&time_milli_sec, event_start, event_stop);
        // time_2 += time_milli_sec;
        // cudaEventRecord(event_start);

        computeOffsets(subgraph_pointers, d_blk_counter);
        // cudaDeviceSynchronize();
        // checkCudaError(2);

        // cudaEventRecord(event_stop);
        // cudaEventSynchronize(event_stop);
        // time_milli_sec = 0;
        // cudaEventElapsedTime(&time_milli_sec, event_start, event_stop);
        // time_3 += time_milli_sec;
        // cudaEventRecord(event_start);

        fillNeighbors<<<BLK_NUMS, BLK_DIM>>>(i, subgraph_pointers, plex_pointers, graph_pointers, d_blk, d_blk_counter, d_left, d_left_counter, d_hopSz, commonMtx, d_adj);
        // cudaDeviceSynchronize();
        // checkCudaError(3);

        // cudaEventRecord(event_stop);
        // cudaEventSynchronize(event_stop);
        // time_milli_sec = 0;
        // cudaEventElapsedTime(&time_milli_sec, event_start, event_stop);
        // time_4 += time_milli_sec;
        // cudaEventRecord(event_start);

        buildCommonMtx<<<BLK_NUMS, BLK_DIM>>>(i, plex_pointers, subgraph_pointers, graph_pointers, commonMtx, d_hopSz);
        // cudaDeviceSynchronize();
        // checkCudaError(4);

        // cudaEventRecord(event_stop);
        // cudaEventSynchronize(event_stop);
        // time_milli_sec = 0;
        // cudaEventElapsedTime(&time_milli_sec, event_start, event_stop);
        // time_5 += time_milli_sec;
        // cudaEventRecord(event_start);

        kSearch<<<BLK_NUMS, BLK_DIM>>>(i, plex_pointers, subgraph_pointers, graph_pointers, task_pointers, d_blk, d_blk_counter, d_left, d_left_counter, d_res, d_br, d_state, d_hopSz, neiInG, neiInP, plex_count, commonMtx, recCand1, recCand2, recExcl, nonAdjInP, d_v2delete, d_adj, cycles);
        // cudaDeviceSynchronize();
        // checkCudaError(5);

        // cudaEventRecord(event_stop);
        // cudaEventSynchronize(event_stop);
        // time_milli_sec = 0;
        // cudaEventElapsedTime(&time_milli_sec, event_start, event_stop);
        // time_6 += time_milli_sec;
        // cudaEventRecord(event_start);
        //allocateHostpointers(host_pointers, subgraph_pointers, d_blk_counter, d_hopSz, commonMtx);
        
        // for (int j = 12; j < 13; j++)
        // {
        //     if (j >= peelG.n)
        //         break;
        //     initializeSubg(j, subg, host_pointers);
        //     if (subg.n < lb)
        //         continue;
        //     BKCPUAlgorithm2(j, subg);
        // }
        // printf("Queue size of %d is %d\n",i, TQ.size());
        // std::vector<Task> tasks;
        // while (!TQ.empty())
        // {
        //     tasks.push_back(std::move(TQ.front()));
        //     TQ.pop();
        // }
        // unsigned int initialN = tasks.size();

        // allocateTasks(task_pointers, host_pointers, tasks);
        initializeBNB(0, task_pointers, plex_pointers, subgraph_pointers, d_blk, d_left, d_blk_counter, d_left_counter, commonMtx, plex_count, nonAdjInP, d_sat, d_commons, d_uni, cycles, d_adj);
        cudaMemset(d_blk_counter, 0, WARPS * sizeof(unsigned int));
        cudaMemset(d_left_counter, 0, WARPS * sizeof(unsigned int));
        cudaMemset(d_hopSz, 0, WARPS * sizeof(unsigned int));
        cudaMemset(d_visited, 0, pn * WARPS * sizeof(uint8_t));
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

    cudaMemcpy(h_cycles, cycles, 40 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    // for (int i = 0; i < 40; i++)
    // {
        double us = (double) h_cycles[0] / prop.clockRate;
        printf("Block %d: %f ms\n", 0, us);
    // }
    
    // printf("Time 0: %f, Time 1: %f, Time 2: %f, Time 3: %f, Time 4: %f, Time 5: %f, Time 6: %f, Time 7: %f\n", time_0, time_1, time_2, time_3, time_4, time_5, time_6, time_7);

    // cudaMemcpy(&h_global_count, global_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(&h_left_count, left_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_plex_count, plex_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_validblk, validblk, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("Total Valid Blocks: %d, Maximal k-Plexes: %d\n", h_validblk, h_plex_count);
    printf("\nKernel Launch Successfully\n");
    free_graph_gpu_memory(graph_pointers, degen_pointers);
}

#endif // CUTS_HOST_FUNCS_H