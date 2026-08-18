// #include "../inc/device_funcs.h"
#include "../inc/device_funcs.h"

#ifndef ENABLE_EDGE_POTENTIAL_BOUND
#define ENABLE_EDGE_POTENTIAL_BOUND 1
#endif

__device__ bool basic_search(unsigned int node, unsigned int *buffer, unsigned int len)
{
  for (int idx = 0; idx < len; idx++)
  {
    if (node == buffer[idx]) return true;
  }
  return false;
}

__device__ bool binary_search(unsigned int node, unsigned int *buffer, unsigned int len)
{
  unsigned int lo = 0, hi = len;

  while (lo < hi)
  {
    unsigned int mid = lo + ((hi - lo) >> 1);
    unsigned int v = buffer[mid];
    if (v < node) lo = mid + 1;
    else hi = mid;
  }

  return (lo < len) && (buffer[lo] == node);
}

__global__ void decompose(int batch_id, P_pointers p, G_pointers g, D_pointers d, unsigned int *d_blk, unsigned int *d_blk_counter, unsigned int *d_left, unsigned int *d_left_counter, uint32_t *visited, unsigned int *global_count, unsigned int *left_count, unsigned int *validblk, unsigned int* d_hopSz, unsigned long long* cycles)
{
  unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int warp_id = (global_index / 32);
  unsigned int lane_id = threadIdx.x % 32;
  unsigned int local_warp_id = threadIdx.x >> 5;

  long cond = long(g.n) - long(p.lb) + 2;
  if ((warp_id + WARPS * batch_id) >= cond) return;

  int vstart = d.dseq[warp_id + WARPS * batch_id];
  int idx;
  unsigned int* blkBase = d_blk + warp_id * MAX_BLK_SIZE;
  unsigned int* counterBase = d_blk_counter + warp_id;
  unsigned int* leftBase = d_left + warp_id * MAX_LEFT_SIZE;
  unsigned int* left_counter = d_left_counter + warp_id;
  unsigned int* hopSz = d_hopSz + warp_id;

  int range = (g.n / 32) + 1;
  uint32_t *visitedBase = visited + warp_id * range;

  __shared__ unsigned int sh_blkCount[WARPS_EACH_BLK];
  __shared__ unsigned int sh_leftCount[WARPS_EACH_BLK];

  if (lane_id == 0) sh_blkCount[local_warp_id] = 0;
  if (lane_id == 0) sh_leftCount[local_warp_id] = 0;

  int lb_2k = p.lb - 2 * p.k;
  if (lane_id == 0)
  {
    idx = atomicAdd(&sh_blkCount[local_warp_id], 1);
    blkBase[idx] = vstart;
    atomicOr((unsigned int*)&visitedBase[vstart >> 5], (1u << (vstart & 31)));
  }
  __syncwarp();

  int start = 0;
  int position = d.dpos[vstart];
  int deg = g.degree[vstart];
  int off = g.offsets[vstart];

  for (int t = start; t < deg; t += 32)
  {
    int i = t + lane_id;
    if (i < deg)
    {
      int nei = g.neighbors[off + i];
      if (position < d.dpos[nei])
      {
        idx = atomicAdd(&sh_blkCount[local_warp_id], 1);
        if (idx < MAX_BLK_SIZE)
        {
          blkBase[idx] = nei;
          atomicOr((unsigned int*)&visitedBase[nei >> 5], (1u << (nei & 31)));
        }
        else
        {
          atomicSub(&sh_blkCount[local_warp_id], 1);
        }
      }
      else
      {
        atomicOr((unsigned int*)&visitedBase[nei >> 5], (1u << (nei & 31)));
      }
    }
  }
  __syncwarp();

  size_t sz;
  int hop = 0;
  do {
    sz = sh_blkCount[local_warp_id];
    if (sz - 1 < p.bd) goto CLEAN;
    for (int i = lane_id + 1; i < sh_blkCount[local_warp_id]; i += 32)
    {
      size_t cnt = 0;
      unsigned int u = blkBase[i];
      for (int j = 1; j < sh_blkCount[local_warp_id]; j++)
      {
        unsigned int v = blkBase[j];
        if (binary_search(v, g.neighbors + g.offsets[u], g.degree[u])) cnt++;
      }
      if (cnt < lb_2k)
      {
        atomicAnd((unsigned int*)&visitedBase[u >> 5], ~(1u << (u & 31)));
      }
    }
    if (lane_id == 0)
    {
      int writeIdx = 1;
      int oldCount = sh_blkCount[local_warp_id];
      for (int i = 1; i < oldCount; i++)
      {
        unsigned int u = blkBase[i];
        if (visitedBase[u >> 5] & (1u << (u & 31)))
        {
          blkBase[writeIdx++] = u;
        }
      }
      sh_blkCount[local_warp_id] = writeIdx;
    }
    __syncwarp();
  } while (sh_blkCount[local_warp_id] < sz);

  if (lane_id == 0) hopSz[0] = sh_blkCount[local_warp_id];
  __syncwarp();

  for (int t = start; t < deg; t += 32)
  {
    const int neighbor_idx = t + lane_id;
    if (neighbor_idx < deg)
    {
      const int direct_left = g.neighbors[off + neighbor_idx];
      if (position >= d.dpos[direct_left])
      {
        size_t cnt = 0;
        for (int j = 1; j < sh_blkCount[local_warp_id]; j++)
        {
          const unsigned int v = blkBase[j];
          if (binary_search(v, g.neighbors + g.offsets[direct_left], g.degree[direct_left])) cnt++;
        }
        if (cnt >= lb_2k + 1)
        {
          idx = atomicAdd(&sh_leftCount[local_warp_id], 1);
          if (idx < MAX_LEFT_SIZE)
          {
            leftBase[idx] = direct_left;
            atomicOr((unsigned int*)&visitedBase[direct_left >> 5], (1u << (direct_left & 31)));
          }
          else
          {
            atomicSub(&sh_leftCount[local_warp_id], 1);
          }
        }
      }
    }
  }
  __syncwarp();

  hop = hopSz[0];
  for (int i = 0; i < deg; i++)
  {
    int nei = g.neighbors[off + i];
    if (position < d.dpos[nei])
    {
      for (int j = lane_id; j < g.degree[nei]; j += 32)
      {
        int twoHop = g.neighbors[g.offsets[nei] + j];
        if (visitedBase[twoHop >> 5] & (1u << (twoHop & 31))) continue;
        const int thr = lb_2k + ((position < d.dpos[twoHop]) ? 2 : 3);
        if (min(g.degree[twoHop], hop - 1) < thr) continue;
        int cnt = 0;
        for (int k = 1; k < hop; k++)
        {
          unsigned int v = blkBase[k];
          if (binary_search(v, g.neighbors + g.offsets[twoHop], g.degree[twoHop]))
          {
            if (++cnt >= thr) break;
          }
          int rem = hop - 1 - k;
          if (cnt + rem < thr) break;
        }

        if ((visitedBase[twoHop >> 5] & (1u << (twoHop & 31))) == 0)
        {
          atomicOr((unsigned int*)&visitedBase[twoHop >> 5], (1u << (twoHop & 31)));
          if (position < d.dpos[twoHop])
          {
            if (cnt >= lb_2k + 2)
            {
              idx = atomicAdd(&sh_blkCount[local_warp_id], 1);
              if (idx < MAX_BLK_SIZE)
              {
                blkBase[idx] = twoHop;
              }
              else
              {
                atomicSub(&sh_blkCount[local_warp_id], 1);
              }
            }
          }
          else
          {
            if (cnt >= lb_2k + 3)
            {
              idx = atomicAdd(&sh_leftCount[local_warp_id], 1);
              if (idx < MAX_LEFT_SIZE)
              {
                leftBase[idx] = twoHop;
              }
              else
              {
                atomicSub(&sh_leftCount[local_warp_id], 1);
              }
            }
          }
        }
      }
      __syncwarp();
    }
  }

  __syncwarp();

  if (lane_id == 0)
  {
    if (sh_blkCount[local_warp_id] > MAX_BLK_SIZE)
    {
      printf("Block Size is greater than the constant\n");
    }
    else if (sh_leftCount[local_warp_id] > MAX_LEFT_SIZE)
    {
      printf("Left Size is greater than constant\n");
    }
    else if (sh_blkCount[local_warp_id] >= p.lb)
    {
      atomicAdd(&validblk[0], 1);
      atomicAdd(&counterBase[0], sh_blkCount[local_warp_id]);
      atomicAdd(&left_counter[0], sh_leftCount[local_warp_id]);
    }
    else
    {
      hopSz[0] = 0;
    }
  }
CLEAN:
  __syncwarp();
}


__global__ void calculateDegrees(int i , P_pointers p, G_pointers g, S_pointers s, unsigned int *d_blk, unsigned int *d_blk_counter, unsigned int *d_left, unsigned int *d_left_counter, uint32_t *visited, unsigned int *global_count, unsigned int *left_count)
{
  unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int warp_id = (global_index / 32);
  unsigned int lane_id = threadIdx.x % 32;
  if ((warp_id+WARPS*i) >= (g.n-p.lb+2)) return;
  unsigned int* blkBase = d_blk + warp_id * MAX_BLK_SIZE;
  unsigned int* counterBase = d_blk_counter + warp_id;
  const int range = (g.n / 32) + 1;
  uint32_t* visitedBase = visited + warp_id * range;

  unsigned int* local_n = s.n + warp_id;
  // s.n[warp_id] must always reflect THIS wave's block count, even when it's
  // below p.lb and everything past this point gets skipped -- buildCommonMtx()
  // reads s.n[warp_id] unconditionally (it has no matching < p.lb guard).
  if (lane_id == 0) local_n[0] = counterBase[0];

  if (counterBase[0] < p.lb) return;

  unsigned int* degreeBase = s.degree + warp_id * (MAX_BLK_SIZE);

  int counter;

  if (lane_id == 0)
  {
    counter = counterBase[0];
  }

  counter = __shfl_sync(0xFFFFFFFF, counter, 0);

  for (int idx = lane_id; idx < counter; idx+=32)
  {
    unsigned int origin = blkBase[idx];
    int ne = 0;
    for (int j = 0; j < g.degree[origin];j++)
    {
      unsigned int nei = g.neighbors[g.offsets[origin]+j];
      if ((visitedBase[nei >> 5] & (1u << (nei & 31))) == 0) continue;
      if (basic_search(nei, blkBase, counter)) ne++;
    }

    degreeBase[idx] = ne;
  }
}

__global__ void fillNeighbors(int i, S_pointers s, P_pointers p, G_pointers g, unsigned int *d_blk, unsigned int *d_blk_counter, unsigned int *d_left, unsigned int *d_left_counter, unsigned int *d_hopSz, uint8_t* commonMtx, uint32_t* d_adj, uint32_t* d_left_adj, uint32_t* d_local_left_adj)
{
  unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int warp_id = (global_index / 32);
  unsigned int lane_id = threadIdx.x % 32;

  if ((warp_id+WARPS*i) >= (g.n-p.lb+2)) return;
  int value = warp_id+WARPS*i;
  unsigned int* blkBase = d_blk + warp_id * MAX_BLK_SIZE;
  unsigned int* counterBase = d_blk_counter + warp_id;

  unsigned int* offsetsBase = s.offsets + warp_id * (MAX_BLK_SIZE);

  unsigned int* degreeBase = s.degree + warp_id * (MAX_BLK_SIZE);
  unsigned int* degreeHop = s.degreeHop + warp_id * (MAX_BLK_SIZE);

  unsigned int* leftBase = d_left + warp_id * MAX_LEFT_SIZE;
  unsigned int* left_counter = d_left_counter + warp_id;

  unsigned int* hopSz = d_hopSz + warp_id;

  unsigned int* neighborsBase = s.neighbors + (size_t)warp_id * MAX_BLK_SIZE * AVG_DEGREE;

  unsigned int* local_n = s.n + warp_id;
  unsigned int* local_m = s.m + warp_id;
  unsigned int* PlexSz = s.PSize + warp_id;
  size_t capacity = size_t(warp_id) * CAP;
  uint8_t* commonMtxBase = commonMtx + capacity;

  uint32_t* adjList = d_adj + warp_id * ADJSIZE;
  uint32_t* leftAdjBase = d_left_adj + warp_id * LEFT_ADJ_SIZE;
  uint32_t* localLeftAdjBase = d_local_left_adj + warp_id * LOCAL_LEFT_ADJ_SIZE;

  unsigned int* CandSz = s.CSize + warp_id;
  unsigned int* Cand2Sz = s.C2Size + warp_id;
  unsigned int* ExclSz = s.XSize + warp_id;

  unsigned int* plex = s.P + warp_id * MAX_BLK_SIZE;
  unsigned int* cand1 = s.C + warp_id * MAX_BLK_SIZE;
  unsigned int* cand2 = s.C2 + warp_id * MAX_BLK_SIZE;
  // unsigned int* excl = s.X + warp_id * MAX_BLK_SIZE;

  int counter;
  int l_counter;
  int hop;
  if (lane_id == 0)
  {
    counter = counterBase[0];
    l_counter = left_counter[0];
    hop = hopSz[0];
  }

  counter = __shfl_sync(0xFFFFFFFF, counter, 0);
  l_counter = __shfl_sync(0xFFFFFFFF, l_counter, 0);
  hop = __shfl_sync(0xFFFFFFFF, hop, 0);

  for (int idx = lane_id; idx < counter; idx+=32)
  {
    unsigned int origin = blkBase[idx];
    int cnt = 0;
    unsigned int offset = offsetsBase[idx];
    for (int j = 0; j < counter; j++)
    {
      unsigned int nei = blkBase[j];
      if (binary_search(nei, g.neighbors+g.offsets[origin], g.degree[origin]))
      {
        int v = idx*counter+j;
        int v2 = j*counter + idx;
        atomicOr(&adjList[v >> 5], 1u << (v & 31));
        atomicOr(&adjList[v2 >> 5], 1u << (v2 & 31));
        commonMtxBase[v] = 1;
        neighborsBase[offset+cnt] = j;
        cnt++;
      }
      if (j == hop-1) degreeHop[idx] = cnt;
    }

    for (int j = 0; j < l_counter; j++)
    {
      unsigned int nei = leftBase[j];
      if (binary_search(nei, g.neighbors+g.offsets[origin], g.degree[origin]))
      {
        atomicOr(&leftAdjBase[(size_t)j * MASK_WORDS + (idx >> 5)], 1u << (idx & 31));
        atomicOr(&localLeftAdjBase[(size_t)idx * LEFT_MASK_WORDS + (j >> 5)], 1u << (j & 31));
      }
    }
  }

  __syncwarp();

  for (int i = lane_id + 1; i < hop; i += 32)
  {
    cand1[i-1] = i;
  }
  for (int i = lane_id + hop; i < counter; i += 32)
  {
    cand2[i-hop] = i;
  }
  if (lane_id == 0)
  {
    plex[0] = 0;
    CandSz[0] = hop - 1;
    Cand2Sz[0] = counter - hop;
    ExclSz[0] = 0;
    PlexSz[0] = 1;
  }
  local_m[0] = offsetsBase[counter];

  if (value == 6 && lane_id == 0)
  {
    // printf("Blks: ");
    // for (int i = 0; i < counterBase[0]; i++)
    // {
    //   printf("%d ", blkBase[i]);
    // }
    // printf("\n");
    // printf("Lefts: ");
    // for (int i = 0; i < left_counter[0]; i++)
    // {
    //   printf("%d ", leftBase[i]);
    // }
    // printf("\n");
    // printf("Degree of Node %d: ", blkBase[0]);
    // for (int i = 0; i < counterBase[0]; i++)
    // {
    //   printf("%d ", degreeBase[i]);
    // }
    // printf("\n");

    // printf("Offsets Array: ");
    // for (int i = 0; i < counterBase[0]+1; i++)
    // {
    //   printf("%d ", offsetsBase[i]);
    // }
    // printf("\n");

    // printf("Left Degree of Node %d: ", blkBase[0]);
    // for (int i = 0; i < counterBase[0]; i++)
    // {
    //   printf("%d ", l_degreeBase[i]);
    // }
    // printf("\n");

    // printf("Left Offsets Array: ");
    // for (int i = 0; i < counterBase[0]+1; i++)
    // {
    //   printf("%d ", l_offsetsBase[i]);
    // }
    // printf("\n");

    // printf("Degree Hop Array: ");
    // for (int i = 0; i < counterBase[0]; i++)
    // {
    //   printf("%d ", degreeHop[i]);
    // }
    // printf("\n");

    // // printf("neiInG Array: ");
    // // for (int i = 0; i < counterBase[0]; i++)
    // // {
    // //   printf("%d ", neiInGBase[i]);
    // // }
    // // printf("\n");

    // printf("Neighors Array: ");
    // for (int i = 0; i < offsetsBase[counterBase[0]]; i++)
    // {
    //   printf("%d ", neighborsBase[i]);
    // }
    // printf("\n");

    // printf("Left Neighors Array: ");
    // for (int i = 0; i < l_offsetsBase[counterBase[0]]; i++)
    // {
    //   printf("%d ", l_neighborsBase[i]);
    // }
    // printf("\n");

    // // //---------------BNB----------------
    // // //printf("n = %d, m = %d, Plex Size = %d, Cand1 Size = %d, Cand2 Size = %d, Excl Size = %d\n", local_n[0], local_m[0], PlexSz[0], Cand1Sz[0], Cand2Sz[0], ExclSz[0]);
    // // //-----------------BK-----------
    // printf("n = %d, m = %d, Plex Size = %d, Cand Size = %d, Cand2 Sz = %d, Excl Size = %d, hopSz: %d, leftCounter: %d\n", local_n[0], local_m[0], PlexSz[0], CandSz[0], Cand2Sz[0], ExclSz[0], hopSz[0], left_counter[0]);
    // printf("Labels Array: ");
    // for (int i = 0; i < local_n[0]; i++)
    // {
    //   printf("%d ", labelsBase[i]);
    // }
    // printf("\n");

    // printf("Cand1: ");
    // for (int i = 0; i < CandSz[0]-1; i++)
    // {
    //   printf("%d ", cand1[i]);
    // }
    // printf("\n");

    // printf("Cand2: ");
    // for (int i = 0; i < Cand2Sz[0]; i++)
    // {
    //   printf("%d ", cand2[i]);
    // }
    // printf("\n");
    // printf("Common Matrix: \n");
    // for (int i = 0; i < counterBase[0]; i++)
    // {
    //   for (int j = 0; j < counterBase[0]; j++)
    //   {
    //     printf("%d ", commonMtxBase[i*counterBase[0]+j]);
    //   }
    //   printf("\n\n");
    // }
  }
}

__device__ __forceinline__ bool adjContains(const uint32_t* adjList, unsigned int n, unsigned int row, unsigned int col)
{
  const size_t check = (size_t)row * n + col;
  return ((adjList[check >> 5] >> (check & 31u)) & 1u) != 0;
}

__device__ unsigned int buildCriticalPList(int lane_id, int k, unsigned int PlexSz, unsigned int* plex, uint16_t* neiInP, uint16_t* criticalP)
{
  unsigned int criticalPSz = 0;
  for (unsigned int base = 0; base < PlexSz; base += 32)
  {
    const unsigned int idx = base + lane_id;
    unsigned int v = 0;
    bool pred = false;
    if (idx < PlexSz)
    {
      v = plex[idx];
      pred = (neiInP[v] + k == PlexSz);
    }

    const unsigned int mask = __ballot_sync(0xFFFFFFFFu, pred);
    const unsigned int rank = __popc(mask & ((1u << lane_id) - 1u));
    if (pred) criticalP[criticalPSz + rank] = (uint16_t)v;
    criticalPSz += __popc(mask);
  }
  __syncwarp();
  return criticalPSz;
}

__device__ bool isKplex2(int lane_id, int v, int k, unsigned int PlexSz, uint16_t* neiInP, unsigned int* plex, unsigned int n, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeBase, uint32_t* adjList)
{
  unsigned mask = __activemask();
  if (neiInP[v] + k <  (PlexSz+1))
  {
    return false;
  }
  bool localPass = true;
  for (int i = lane_id; i < PlexSz; i+=32)
  {
    const int u = plex[i];
    if (neiInP[u] + k == PlexSz && !adjContains(adjList, n, u, v)) localPass = false;
  }
  if (__any_sync(mask, !localPass)) return false;
  return true;
}

__device__ bool isKplex3CriticalList(int v, int k, unsigned int PlexSz, uint16_t* neiInP, const uint16_t* criticalP, unsigned int criticalPSz, unsigned int n, const uint32_t* adjList)
{
  if (neiInP[v] + k < (PlexSz + 1))
  {
    return false;
  }

  for (unsigned int i = 0; i < criticalPSz; i++)
  {
    if (!adjContains(adjList, n, criticalP[i], v)) return false;
  }

  return true;
}

__device__ bool isKplexPC2(int v, int k, unsigned int totalSz, unsigned int PlexSz, unsigned int CandSz, uint16_t* missing, unsigned int* plex, unsigned int* cand, unsigned int n, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeBase, uint32_t* adjList)
{
  if (missing[v] + k < (totalSz + 1))
  {
    return false;
  }
  for (int i = 0; i < PlexSz; i++)
  {
    const int u = plex[i];
    if (missing[u] + k == (totalSz) && !adjContains(adjList, n, u, v))
        return false;
  }
  for (int i = 0; i < CandSz; i++)
  {
    const int u = cand[i];
    if (missing[u] + k == (totalSz) && !adjContains(adjList, n, u, v))
        return false;
  }
  return true;
}

__device__ void subG(int lane_id, int j, uint16_t* neiInG, unsigned int n, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeBase)
{
  for (int i = lane_id; i < degreeBase[j]; i+=32)
  {
    int nei = neighborsBase[offsetsBase[j]+i];
    neiInG[nei]--;
  }
}

__device__ void addG(int lane_id, int j, uint16_t* neiInG, unsigned int n, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeBase)
{
  for (int i = lane_id; i < degreeBase[j]; i+=32)
  {
    int nei = neighborsBase[offsetsBase[j]+i];
    neiInG[nei]++;
  }
  __syncwarp();
}

__device__ void clearMaximalMasks(unsigned int lane_id, uint32_t* solutionMask, uint32_t* criticalMask, uint32_t* candidateMask, unsigned int leftMaskWords = LEFT_MASK_WORDS)
{
  for (int w = lane_id; w < MASK_WORDS; w += 32)
  {
    solutionMask[w] = 0;
    criticalMask[w] = 0;
  }
  for (unsigned int w = lane_id; w < leftMaskWords; w += 32)
  {
    candidateMask[w] = 0;
  }
  __syncwarp();
}

__device__ void addMaskBit(uint32_t* mask, unsigned int v)
{
  atomicOr(&mask[v >> 5], 1u << (v & 31));
}

__device__ bool canAddLeftByBitset(unsigned int lane_id, unsigned int u, int k, unsigned int totalSz, const uint32_t* leftAdjBase, const uint32_t* solutionMask, const uint32_t* criticalMask)
{
  const uint32_t* row = leftAdjBase + (size_t)u * MASK_WORDS;

  unsigned int adjCnt = 0;
  for (int w = 0; w < MASK_WORDS; w++)
  {
    const uint32_t adj = row[w];
    adjCnt += __popc(adj & solutionMask[w]);
    if ((criticalMask[w] & ~adj) != 0) return false;
  }
  return adjCnt + (unsigned int)k >= totalSz + 1;
}

__device__ __forceinline__ unsigned int solutionVertexAt(unsigned int pos, unsigned int PlexSz, unsigned int* plex, unsigned int* cand)
{
  return (pos < PlexSz) ? plex[pos] : cand[pos - PlexSz];
}


__device__ void buildLeftCandidateMaskByBitset(unsigned int lane_id, int k, unsigned int nsat, const uint16_t* local_sat, unsigned int PlexSz, unsigned int* plex, unsigned int* cand, const uint32_t* localLeftAdjBase, uint32_t* candidateMask, unsigned int leftMaskWords = LEFT_MASK_WORDS)
{
  for (unsigned int w = lane_id; w < leftMaskWords; w += 32)
  {
    uint32_t bits = 0;
    if (nsat > 0)
    {
      bits = localLeftAdjBase[(size_t)local_sat[0] * leftMaskWords + w];
      for (unsigned int i = 1; i < nsat; i++)
      {
        bits &= localLeftAdjBase[(size_t)local_sat[i] * leftMaskWords + w];
      }
    }

    if (nsat == 0 || nsat == 1)
    {
      uint32_t firstKBits = 0;
      for (int t = 0; t < k; t++)
      {
        const unsigned int u = solutionVertexAt(t, PlexSz, plex, cand);
        firstKBits |= localLeftAdjBase[(size_t)u * leftMaskWords + w];
      }
      bits = (nsat == 0) ? firstKBits : (bits & firstKBits);
    }

    candidateMask[w] = bits;
  }
  __syncwarp();
}

__device__ bool hasAddableLeftByCandidateMask(unsigned int lane_id, int k, unsigned int totalSz, unsigned int left_count, const uint32_t* leftAdjBase, const uint32_t* solutionMask, const uint32_t* criticalMask, const uint32_t* candidateMask, unsigned int leftMaskWords = LEFT_MASK_WORDS)
{
  bool found = false;
  for (unsigned int w = lane_id; w < leftMaskWords; w += 32)
  {
    uint32_t bits = candidateMask[w];
    while (bits)
    {
      const unsigned int bit = __ffs(bits) - 1;
      const unsigned int u = (unsigned int)w * 32u + bit;
      if (u < left_count && canAddLeftByBitset(lane_id, u, k, totalSz, leftAdjBase, solutionMask, criticalMask))
      {
        found = true;
      }
      bits &= bits - 1;
    }
  }
  return __any_sync(0xFFFFFFFFu, found);
}

// leftAdjBase/localLeftAdjBase are always non-null at every live call site, so
// this takes the bitset fast path below. The older non-bitset fallback was dead
// code and has been removed along with l_neighbors itself.
__device__ bool isMaximal_opt(unsigned int lane_id, int k, unsigned int PlexSz, unsigned int* left, unsigned int left_count, uint16_t* nonNeigh, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeBase, unsigned int* plex, unsigned int n, uint16_t* local_sat, uint32_t* uni, const uint32_t* leftAdjBase, const uint32_t* localLeftAdjBase, unsigned int leftMaskWords = LEFT_MASK_WORDS)
{
  const unsigned FULL_MASK = 0xFFFFFFFFu;

  bool max = true;
  uint32_t* solutionMask = uni;
  uint32_t* criticalMask = uni + MASK_WORDS;
  uint32_t* candidateMask = uni + 2 * MASK_WORDS;
  clearMaximalMasks(lane_id, solutionMask, criticalMask, candidateMask, leftMaskWords);

  int nsat = 0;
  for (int base = 0; base < PlexSz; base += 32)
  {
    int i = base + lane_id;
    bool pred = false;
    uint16_t v = 0;
    if (i < PlexSz)
    {
      v = plex[i];
      pred = nonNeigh[v] + k < PlexSz + 1;
      addMaskBit(solutionMask, v);
      if (pred) addMaskBit(criticalMask, v);
    }
    unsigned mask = __ballot_sync(FULL_MASK, pred);

    int off = __shfl_sync(FULL_MASK, nsat, 0);
    if (lane_id == 0) nsat += __popc(mask);

    int l_off = __popc(mask & ((1u << lane_id) - 1));
    if (pred) local_sat[off + l_off] = v;
  }
  nsat = __shfl_sync(FULL_MASK, nsat, 0);
  __syncwarp();

  buildLeftCandidateMaskByBitset(lane_id, k, nsat, local_sat, PlexSz, plex, nullptr, localLeftAdjBase, candidateMask, leftMaskWords);
  if (hasAddableLeftByCandidateMask(lane_id, k, PlexSz, left_count, leftAdjBase, solutionMask, criticalMask, candidateMask, leftMaskWords))
  {
    if (lane_id == 0) max = false;
  }
  max = __shfl_sync(FULL_MASK, max, 0);
  return max;
}

// See isMaximal_opt's comment: leftAdjBase/localLeftAdjBase are always non-null
// at every live call site, so this takes the bitset fast path.
__device__ bool isMaximalPC_opt(unsigned int lane_id, int k, unsigned int PlexSz, unsigned int CandSz, unsigned int totalSz, unsigned int* left, unsigned int left_count, uint16_t* nonNeigh, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeBase, unsigned int* plex, unsigned int* cand,  unsigned int n, uint16_t* local_sat, uint32_t* uni, const uint32_t* leftAdjBase, const uint32_t* localLeftAdjBase, unsigned int leftMaskWords = LEFT_MASK_WORDS)
{
  const unsigned FULL_MASK = 0xFFFFFFFFu;

  bool max = true;
  uint32_t* solutionMask = uni;
  uint32_t* criticalMask = uni + MASK_WORDS;
  uint32_t* candidateMask = uni + 2 * MASK_WORDS;
  clearMaximalMasks(lane_id, solutionMask, criticalMask, candidateMask, leftMaskWords);

  int nsat = 0;
  for (int base = 0; base < PlexSz; base += 32)
  {
    int i = base + lane_id;
    bool pred = false;
    uint16_t v = 0;
    if (i < PlexSz)
    {
      v = plex[i];
      pred = nonNeigh[v] + k < totalSz + 1;
      addMaskBit(solutionMask, v);
      if (pred) addMaskBit(criticalMask, v);
    }
    unsigned mask = __ballot_sync(FULL_MASK, pred);

    int off = __shfl_sync(FULL_MASK, nsat, 0);
    if (lane_id == 0) nsat += __popc(mask);

    int l_off = __popc(mask & ((1u << lane_id) - 1));
    if (pred) local_sat[off + l_off] = v;
  }

  for (int base = 0; base < CandSz; base += 32)
  {
    int i = base + lane_id;
    bool pred = false;
    uint16_t v = 0;
    if (i < CandSz)
    {
      v = cand[i];
      pred = (nonNeigh[v] + k < totalSz + 1);
      addMaskBit(solutionMask, v);
      if (pred) addMaskBit(criticalMask, v);
    }
    unsigned mask = __ballot_sync(FULL_MASK, pred);
    int off = __shfl_sync(FULL_MASK, nsat, 0);
    if (lane_id == 0) nsat += __popc(mask);
    nsat = __shfl_sync(FULL_MASK, nsat, 0);
    int l_off = __popc(mask & ((1u << lane_id) - 1));
    if (pred) local_sat[off + l_off] = v;
  }
  nsat = __shfl_sync(FULL_MASK, nsat, 0);
  __syncwarp();

  buildLeftCandidateMaskByBitset(lane_id, k, nsat, local_sat, PlexSz, plex, cand, localLeftAdjBase, candidateMask, leftMaskWords);
  if (hasAddableLeftByCandidateMask(lane_id, k, totalSz, left_count, leftAdjBase, solutionMask, criticalMask, candidateMask, leftMaskWords))
  {
    if (lane_id == 0) max = false;
  }
  max = __shfl_sync(FULL_MASK, max, 0);
  return max;
}

  __device__ bool upperBoundK2(int lane_id, int k, int lb, unsigned int* plex, uint16_t* neiInG, unsigned int PlexSz)
{
  bool ok = true;
  for (int i = lane_id; i < PlexSz; i+=32)
  {
      const int u = plex[i];
      if (neiInG[u] + k < lb)
      {
        ok = false;
        break;
      }
  }
  bool any_fail = __any_sync(0xFFFFFFFF, !ok);
  if (any_fail)
  {
    return false;
  }
  return true;
  }

  __device__ bool upperBound2(int lane_id, int k, int lb, unsigned int* plex, uint16_t* neiInG, unsigned int PlexSz)
{
  bool ok = true;
  for (int i = lane_id; i < PlexSz; i+=32)
  {
    unsigned int v = plex[i];
      if (neiInG[v] + k < lb)
      {
        ok = false;
        break;
      }
  }
  bool any_fail = __any_sync(0xFFFFFFFF, !ok);
  if (any_fail)
  {
    return false;
  }
  return true;
}

__device__ unsigned int computeEdgePotentialSum(int lane_id, unsigned int* plex, unsigned int PlexSz, unsigned int* cand, unsigned int CandSz, uint16_t* neiInG)
{
#if ENABLE_EDGE_POTENTIAL_BOUND
  unsigned int localDegreeSum = 0;
  for (unsigned int i = lane_id; i < PlexSz; i += 32)
  {
    localDegreeSum += neiInG[plex[i]];
  }
  for (unsigned int i = lane_id; i < CandSz; i += 32)
  {
    localDegreeSum += neiInG[cand[i]];
  }

  for (int offset = 16; offset > 0; offset >>= 1)
  {
    localDegreeSum += __shfl_down_sync(0xFFFFFFFFu, localDegreeSum, offset);
  }

  return __shfl_sync(0xFFFFFFFFu, localDegreeSum, 0);
#else
  return 0;
#endif
}

__device__ bool edgePotentialBoundCached(int k, int lb, unsigned int PlexSz, unsigned int CandSz, unsigned int edgePotential)
{
#if ENABLE_EDGE_POTENTIAL_BOUND
  if (PlexSz + CandSz < (unsigned int)lb) return false;
  if (lb <= k) return true;

  const unsigned int requiredDegree = (unsigned int)lb * (unsigned int)(lb - k);
  return edgePotential >= requiredDegree;
#else
  return true;
#endif
}

__device__ unsigned int subtractRemovedVertexPotential(int lane_id, unsigned int edgePotential, unsigned int v, uint16_t* neiInG)
{
#if ENABLE_EDGE_POTENTIAL_BOUND
  if (lane_id == 0)
  {
    const unsigned int decrement = 2u * (unsigned int)neiInG[v];
    edgePotential = (edgePotential > decrement) ? (edgePotential - decrement) : 0u;
  }
  return __shfl_sync(0xFFFFFFFFu, edgePotential, 0);
#else
  return edgePotential;
#endif
}

__device__ void updateCand13(int lane_id, unsigned int* cand1, uint8_t* commonMtx, unsigned int* recCand1Base, uint16_t* neiInGBase, int sz, int n, int v2add, unsigned int* Cand1Sz, unsigned int* neighborsBase, unsigned int* degreeBase, unsigned int* offsetsBase)
{
  const uint8_t* row = commonMtx + (size_t)v2add * n;

  int read  = 0;
  int write = 0;
  int size = Cand1Sz[0];

  while (read < size)
  {
    const int take = min(32, size - read);
    const bool active = (lane_id < take);
    //const int idx = i + lane_id;

    unsigned int v = 0;
    if (active) v = cand1[read+lane_id];

    const bool keep = active && !(row[v] < UNLINK2EQUAL);
    // printf("keep: %d, lane_id: %d, unlink_equal: %d, row: %d, active: %d, v2add: %d, n: %d\n", keep, lane_id, UNLINK2EQUAL, row[v], active, v2add, n);

    const unsigned activemask = __ballot_sync(0xFFFFFFFF, active);
    unsigned keepmask = __ballot_sync(0xFFFFFFFF, keep);
    unsigned dropmask = activemask ^ keepmask;

    const int keep_rank = __popc(keepmask & ((1u << lane_id) - 1));
    const int num_keep  = __popc(keepmask);

    if (active && keep)
    {
      cand1[write + keep_rank] = v;
    }


    while (dropmask)
    {
      const int leader = __ffs(dropmask) - 1;
      const unsigned vdrop = __shfl_sync(0xFFFFFFFF, v, leader);
      recCand1Base[vdrop] = (unsigned)(sz - 1);
      subG(lane_id, vdrop, neiInGBase, n, neighborsBase, offsetsBase, degreeBase);
      dropmask &= (dropmask - 1);
    }

    if (lane_id == 0)
    {
      read += take;
      write += num_keep;
    }
    read  = __shfl_sync(0xFFFFFFFF, read, 0);
    write = __shfl_sync(0xFFFFFFFF, write, 0);
  }
  if (lane_id == 0)
  {
    Cand1Sz[0] = write;
  }
  __syncwarp();
}

__device__ void updateCand23(int lane_id, unsigned int* __restrict__ cand2, const uint8_t* __restrict__ commonRow, 
                            unsigned int* __restrict__ recCand2Base, int sz, unsigned int*  Cand2Sz)
{
  int write = 0;
  unsigned mask;
  for (int idx = lane_id; idx < Cand2Sz[0]; idx += 32)
  {
    unsigned int v = cand2[idx];
    bool keep = false;
    keep = !(UNLINK2EQUAL > commonRow[v]);

    mask = __activemask();
    unsigned keepMask = __ballot_sync(mask, keep);
    int rankKeep = __popc(keepMask & ((1u << lane_id) - 1));
    int keepCnt = __popc(keepMask);

    if (keep) cand2[write + rankKeep] = v;
    if (!keep) recCand2Base[v] = sz-1;
    write += keepCnt;
  }

  if (lane_id == 0) Cand2Sz[0] = write;
  __syncwarp();
}

__device__ void recoverCand12(int lane_id, unsigned int* cand1, unsigned int* recCand1Base, uint16_t* neiInGBase, int sz, int n, unsigned int* Cand1Sz, unsigned int* neighborsBase, unsigned int* degreeBase, unsigned int* offsetsBase)
{
  for (int base = 0; base < n; base += 32)
  {
    const int i = base + lane_id;
    const bool inrange = (i < n);
    const bool keep = inrange && (recCand1Base[i] == (unsigned)(sz - 1));

    unsigned mask = __ballot_sync(0xFFFFFFFF, keep);
    if (!mask) continue;

    int out_base = 0;
    if (lane_id == 0)
    {
      const int cnt = __popc(mask);
      out_base = Cand1Sz[0];
      Cand1Sz[0] = out_base + cnt;
    }

    out_base = __shfl_sync(0xFFFFFFFF, out_base, 0);

    const int rank = __popc(mask & ((1u << lane_id) - 1));
    if (keep)
    {
      cand1[out_base + rank] = i;
      recCand1Base[i] = 0;
    }

    while (mask)
    {
      const int leader = __ffs(mask) - 1;
      const int v = __shfl_sync(0xFFFFFFFF, i, leader);
      addG(lane_id, v, neiInGBase, n, neighborsBase, offsetsBase, degreeBase);
      mask &= (mask - 1);
    }
  }
  __syncwarp();
}

__device__ void recoverCand23(int lane_id, int n, unsigned int* cand2, unsigned int* recCand2Base, int sz, unsigned int* Cand2Sz)
{
  const unsigned target = (unsigned)(sz - 1);
  const int rounds = (n + 31) >> 5;
  for (int t = 0; t < rounds; t++)
  {
    int i = (t << 5) + lane_id;
    bool in = (i < n);
    unsigned tag   = in ? recCand2Base[i] : 0u;
    unsigned match = in && (tag == target);

    unsigned amask = __activemask();
    unsigned hit = __ballot_sync(amask, match);
    if (!hit) continue;

    int base = 0;
    if (lane_id == 0)
    {
      int cnt = __popc(hit);
      base = *Cand2Sz;
      *Cand2Sz = base + cnt;
    }
    
    base = __shfl_sync(amask, base, 0);

    if (match)
    {
      int rank = __popc(hit & ((1u << lane_id) - 1));
      cand2[base + rank] = i;
      recCand2Base[i] = 0;
    }
  }
}

__device__ void initializePCX(int lane_id, const uint8_t* __restrict__ labelsBase, unsigned int n, unsigned int* __restrict__ plex, unsigned int* __restrict__ cand, unsigned int* __restrict__ excl)
{
  const unsigned FULL = 0xFFFFFFFFu;

  int p_base = 0, c_base = 0, x_base = 0;

  for (int i0 = 0; i0 < n; i0 += 32)
  {
    int i = i0 + lane_id;
    uint8_t lbl = (i < n) ? labelsBase[i] : 0xFF;
    // if (lane_id == 0) printf("lbl: %d\n", lbl);
    unsigned p_mask = __ballot_sync(FULL, lbl == P);
    unsigned c_mask = __ballot_sync(FULL, lbl == C);
    unsigned x_mask = __ballot_sync(FULL, lbl == X);

    int p_rank = __popc(p_mask & ((1u << lane_id) - 1));
    int c_rank = __popc(c_mask & ((1u << lane_id) - 1));
    int x_rank = __popc(x_mask & ((1u << lane_id) - 1));

    int p_wbase = __shfl_sync(FULL, p_base, 0);
    int c_wbase = __shfl_sync(FULL, c_base, 0);
    int x_wbase = __shfl_sync(FULL, x_base, 0);

    // if (lane_id == 0) printf("plex: %d\n", plex[p_base]);
    // if (lane_id == 0) printf("lbl: %d\n", labelsBase[i]);

    if (lbl == P) plex[p_wbase + p_rank] = i;
    if (lbl == C) cand[c_wbase + c_rank] = i;
    if (lbl == X) excl[x_wbase + x_rank] = i;

    if (lane_id == 0)
    {
      p_base += __popc(p_mask);
      c_base += __popc(c_mask);
      x_base += __popc(x_mask);
    }

  }
  
}


__device__ bool reserve_task_slot(int lane_id, Task* tasks, Task* global_tasks, unsigned int* tailPtr, unsigned int* global_tail, uint8_t* d_all_labels, uint16_t* d_all_neiInG, uint16_t* d_all_neiInP, uint8_t* global_labels, uint16_t* global_neiInG, uint16_t* global_neiInP, int* abort, Task*& localTasks, uint8_t*& labels, uint16_t*& all_neiInG, uint16_t*& all_neiInP, unsigned int& pos)
{
  localTasks = tasks;
  labels = d_all_labels;
  all_neiInG = d_all_neiInG;
  all_neiInP = d_all_neiInP;

  if (lane_id == 0) pos = atomicAdd(&tailPtr[0], 1u);
  pos = __shfl_sync(0xFFFFFFFFu, pos, 0);

  if (pos + 1 > (SMALL_CAP)-WARPS)
  {
    if (lane_id == 0)
    {
      atomicSub(&tailPtr[0], 1u);
      pos = atomicAdd(&global_tail[0], 1u);
    }
    pos = __shfl_sync(0xFFFFFFFFu, pos, 0);
    if (pos + 1 >= (MAX_CAP)-WARPS)
    {
      if (lane_id == 0) abort[0] = 1;
      __syncwarp();
      return false;
    }
    localTasks = global_tasks;
    labels = global_labels;
    all_neiInG = global_neiInG;
    all_neiInP = global_neiInP;
  }

  return true;
}

__device__ bool enqueue_exclude_task_from_arrays(int lane_id, int task_idx, unsigned int n, unsigned int pivot, unsigned int* plex, unsigned int PlexSz, unsigned int* cand, unsigned int CandSz, unsigned int* excl, unsigned int ExclSz, uint16_t* neiInG, uint16_t* neiInP, unsigned int edgePotential, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeBase, Task* tasks, Task* global_tasks, unsigned int* tailPtr, unsigned int* global_tail, uint8_t* d_all_labels, uint16_t* d_all_neiInG, uint16_t* d_all_neiInP, uint8_t* global_labels, uint16_t* global_neiInG, uint16_t* global_neiInP, int* abort, unsigned int* global_count)
{
  Task* localTasks = tasks;
  uint8_t* labels = d_all_labels;
  uint16_t* all_neiInG = d_all_neiInG;
  uint16_t* all_neiInP = d_all_neiInP;
  unsigned int pos = 0;

  if (CandSz == 0) return false;
  if (!reserve_task_slot(lane_id, tasks, global_tasks, tailPtr, global_tail, d_all_labels, d_all_neiInG, d_all_neiInP, global_labels, global_neiInG, global_neiInP, abort, localTasks, labels, all_neiInG, all_neiInP, pos)) return false;


  uint8_t* childLabels = labels + (size_t)pos * MAX_BLK_SIZE;
  uint16_t* childNeiInG = all_neiInG + (size_t)pos * MAX_BLK_SIZE;
  uint16_t* childNeiInP = all_neiInP + (size_t)pos * MAX_BLK_SIZE;

  for (unsigned int j = lane_id; j < n; j += 32)
  {
    childLabels[j] = U;
    childNeiInG[j] = neiInG[j];
    childNeiInP[j] = neiInP[j];
  }
  __syncwarp();

  for (unsigned int j = lane_id; j < PlexSz; j += 32) childLabels[plex[j]] = P;
  for (unsigned int j = lane_id; j < CandSz; j += 32) childLabels[cand[j]] = C;
  for (unsigned int j = lane_id; j < ExclSz; j += 32) childLabels[excl[j]] = X;
  __syncwarp();

  const unsigned int childEdgePotential = subtractRemovedVertexPotential(lane_id, edgePotential, pivot, neiInG);

  if (lane_id == 0) childLabels[pivot] = X;
  __syncwarp();

  subG(lane_id, pivot, childNeiInG, n, neighborsBase, offsetsBase, degreeBase);
  __syncwarp();

  if (lane_id == 0)
  {
    Task &nt = localTasks[pos];
    nt.idx = task_idx;
    nt.PlexSz = PlexSz;
    nt.CandSz = CandSz - 1;
    nt.ExclSz = ExclSz + 1;
    nt.edgePotential = childEdgePotential;
    nt.labels = childLabels;
    nt.neiInG = childNeiInG;
    nt.neiInP = childNeiInP;
    atomicAdd(&global_count[0], 1);
  }

  return true;
}

__device__ bool remove_unsorted_value(unsigned int* data, unsigned int& sz, unsigned int value)
{
  for (unsigned int i = 0; i < sz; i++)
  {
    if (data[i] == value)
    {
      data[i] = data[sz - 1];
      sz--;
      return true;
    }
  }
  return false;
}

__device__ bool apply_include_branch_unsorted(int lane_id, int k, int lb, unsigned int n, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeBase, uint8_t* commonMtx, unsigned int* plex, unsigned int& PlexSz, unsigned int* cand, unsigned int& CandSz, unsigned int* excl, unsigned int& ExclSz, uint16_t* neiInP, uint16_t* neiInG, int minIndex, uint32_t* adjList, uint16_t* criticalP, unsigned int& edgePotential)
{
  int moved = 1;
  if (lane_id == 0)
  {
    plex[PlexSz++] = minIndex;
    moved = remove_unsorted_value(cand, CandSz, minIndex) ? 1 : 0;
  }

  moved = __shfl_sync(0xFFFFFFFFu, moved, 0);
  PlexSz = __shfl_sync(0xFFFFFFFFu, PlexSz, 0);
  CandSz = __shfl_sync(0xFFFFFFFFu, CandSz, 0);
  __syncwarp();

  if (!moved) return false;

  for (int j = lane_id; j < degreeBase[minIndex]; j += 32)
  {
    const int nei = neighborsBase[offsetsBase[minIndex]+j];
    neiInP[nei]++;
  }
  __syncwarp();

  const unsigned int criticalPSz = buildCriticalPList(lane_id, k, PlexSz, plex, neiInP, criticalP);
  const uint8_t* row = commonMtx + (size_t)minIndex * n;

  int read  = 0;
  int write = 0;
  int size = CandSz;

  while (read < size)
  {
    const int take = min(32, size - read);
    const bool active = (lane_id < take);

    unsigned int v = 0;
    if (active) v = cand[read+lane_id];

    const bool keep = active && !(row[v] < UNLINK2EQUAL) && isKplex3CriticalList(v, k, PlexSz, neiInP, criticalP, criticalPSz, n, adjList);

    const unsigned activemask = __ballot_sync(0xFFFFFFFFu, active);
    unsigned keepmask = __ballot_sync(0xFFFFFFFFu, keep);
    unsigned dropmask = activemask ^ keepmask;

    const int keep_rank = __popc(keepmask & ((1u << lane_id) - 1));
    const int num_keep  = __popc(keepmask);

    if (active && keep) cand[write + keep_rank] = v;

    while (dropmask)
    {
      const int leader = __ffs(dropmask) - 1;
      const unsigned vdrop = __shfl_sync(0xFFFFFFFFu, v, leader);
      edgePotential = subtractRemovedVertexPotential(lane_id, edgePotential, vdrop, neiInG);
      subG(lane_id, vdrop, neiInG, n, neighborsBase, offsetsBase, degreeBase);
      dropmask &= (dropmask - 1);
    }

    if (lane_id == 0)
    {
      read += take;
      write += num_keep;
    }
    read = __shfl_sync(0xFFFFFFFFu, read, 0);
    write = __shfl_sync(0xFFFFFFFFu, write, 0);
  }

  CandSz = write;
  CandSz = __shfl_sync(0xFFFFFFFFu, CandSz, 0);
  __syncwarp();

  bool ub = upperBound2(lane_id, k, lb, plex, neiInG, PlexSz);
  if (!ub) return false;
  if (!edgePotentialBoundCached(k, lb, PlexSz, CandSz, edgePotential)) return false;

  read = 0;
  write = 0;
  size = ExclSz;

  while (read < size)
  {
    const int take = min(32, size - read);
    const bool active = (lane_id < take);

    unsigned int v = 0;
    if (active) v = excl[read+lane_id];

    const bool keep = active && !(row[v] < UNLINK2MORE) && isKplex3CriticalList(v, k, PlexSz, neiInP, criticalP, criticalPSz, n, adjList);

    const unsigned keepmask = __ballot_sync(0xFFFFFFFFu, keep);
    const int keep_rank = __popc(keepmask & ((1u << lane_id) - 1));
    const int num_keep = __popc(keepmask);

    if (active && keep) excl[write + keep_rank] = v;

    if (lane_id == 0)
    {
      read += take;
      write += num_keep;
    }
    read = __shfl_sync(0xFFFFFFFFu, read, 0);
    write = __shfl_sync(0xFFFFFFFFu, write, 0);
  }

  ExclSz = write;
  ExclSz = __shfl_sync(0xFFFFFFFFu, ExclSz, 0);
  __syncwarp();

  return true;
}

__device__ bool enqueue_task_from_arrays(int lane_id, int task_idx, unsigned int n, unsigned int* plex, unsigned int PlexSz, unsigned int* cand, unsigned int CandSz, unsigned int* excl, unsigned int ExclSz, uint16_t* neiInG, uint16_t* neiInP, unsigned int edgePotential, Task* tasks, Task* global_tasks, unsigned int* tailPtr, unsigned int* global_tail, uint8_t* d_all_labels, uint16_t* d_all_neiInG, uint16_t* d_all_neiInP, uint8_t* global_labels, uint16_t* global_neiInG, uint16_t* global_neiInP, int* abort, unsigned int* global_count)
{
  Task* localTasks = tasks;
  uint8_t* labels = d_all_labels;
  uint16_t* all_neiInG = d_all_neiInG;
  uint16_t* all_neiInP = d_all_neiInP;
  unsigned int pos = 0;

  if (!reserve_task_slot(lane_id, tasks, global_tasks, tailPtr, global_tail, d_all_labels, d_all_neiInG, d_all_neiInP, global_labels, global_neiInG, global_neiInP, abort, localTasks, labels, all_neiInG, all_neiInP, pos)) return false;

  uint8_t* childLabels = labels + (size_t)pos * MAX_BLK_SIZE;
  uint16_t* childNeiInG = all_neiInG + (size_t)pos * MAX_BLK_SIZE;
  uint16_t* childNeiInP = all_neiInP + (size_t)pos * MAX_BLK_SIZE;

  for (unsigned int j = lane_id; j < n; j += 32)
  {
    childLabels[j] = U;
    childNeiInG[j] = neiInG[j];
    childNeiInP[j] = neiInP[j];
  }
  __syncwarp();

  for (unsigned int j = lane_id; j < PlexSz; j += 32) childLabels[plex[j]] = P;
  for (unsigned int j = lane_id; j < CandSz; j += 32) childLabels[cand[j]] = C;
  for (unsigned int j = lane_id; j < ExclSz; j += 32) childLabels[excl[j]] = X;
  __syncwarp();

  if (lane_id == 0)
  {
    Task &nt = localTasks[pos];
    nt.idx = task_idx;
    nt.PlexSz = PlexSz;
    nt.CandSz = CandSz;
    nt.ExclSz = ExclSz;
    nt.edgePotential = edgePotential;
    nt.labels = childLabels;
    nt.neiInG = childNeiInG;
    nt.neiInP = childNeiInP;
    atomicAdd(&global_count[0], 1);
  }

  return true;
}

__global__ void rebaseTaskQueuePointers(Task* tasks, uint8_t* labels, uint16_t* neiInG, uint16_t* neiInP, unsigned int N)
{
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N) return;

  tasks[tid].labels = labels + (size_t)tid * MAX_BLK_SIZE;
  tasks[tid].neiInG = neiInG + (size_t)tid * MAX_BLK_SIZE;
  tasks[tid].neiInP = neiInP + (size_t)tid * MAX_BLK_SIZE;
}

__global__ void BNB(int i, P_pointers p, S_pointers s, unsigned int* d_blk, unsigned int* d_left, unsigned int* d_blk_counter, unsigned int* d_left_counter, uint8_t* commonMtx, Task* tasks, Task* outTasks, Task* global_tasks, unsigned int N, unsigned int head, unsigned int* tailPtr, unsigned int* global_tail, uint8_t* d_all_labels, uint16_t* d_all_neiInG, uint16_t* d_all_neiInP, uint8_t* global_labels, uint16_t* global_neiInG, uint16_t* global_neiInP, unsigned int* plex_count, uint16_t* d_bnb_neiInG, uint16_t* d_bnb_neiInP, uint16_t* d_sat, uint32_t* d_uni, unsigned long long* cycles, uint32_t* d_adj, uint32_t* d_left_adj, uint32_t* d_local_left_adj, int* abort, unsigned int* global_count)
{ 
  unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int warp_id = (global_index / 32);
  unsigned int lane_id = threadIdx.x % 32;
  unsigned int local_warp_id = threadIdx.x >> 5;

  unsigned int task_pos = head + warp_id + WARPS * i;
  if (task_pos >= N) return;

  int k = p.k;
  int q = p.lb;
  Task t = tasks[task_pos];
  uint8_t* labelsBase = t.labels;

  unsigned int* leftBase = d_left + t.idx * MAX_LEFT_SIZE;
  uint16_t* neiInG = d_bnb_neiInG + warp_id * MAX_BLK_SIZE;
  uint16_t* neiInP = d_bnb_neiInP + warp_id * MAX_BLK_SIZE;
  unsigned int* left_count = d_left_counter + t.idx;
  unsigned int* local_n = d_blk_counter + t.idx;
  unsigned int PlexSz = t.PlexSz;
  unsigned int CandSz = t.CandSz;
  unsigned int ExclSz = t.ExclSz;
  unsigned int edgePotential = t.edgePotential;

  size_t capacity = size_t(t.idx) * CAP;
  uint8_t* commonMtxBase = commonMtx + capacity;

  unsigned int* degreeBase = s.degree + t.idx * MAX_BLK_SIZE;
  unsigned int* offsetsBase = s.offsets + t.idx * MAX_BLK_SIZE;
  unsigned int* neighborsBase = s.neighbors + (size_t)t.idx * MAX_BLK_SIZE * AVG_DEGREE;

  unsigned int* plex = s.PB + warp_id * MAX_BLK_SIZE;
  unsigned int* cand = s.CB + warp_id * MAX_BLK_SIZE;
  unsigned int* excl = s.XB + warp_id * MAX_BLK_SIZE;

  uint16_t* local_sat = d_sat + warp_id * MAX_BLK_SIZE;
  uint32_t* local_uni = d_uni + warp_id * MAXIMAL_MASK_WORDS;

  uint32_t* adjList = d_adj + t.idx * ADJSIZE;
  uint32_t* leftAdjBase = d_left_adj + t.idx * LEFT_ADJ_SIZE;
  uint32_t* localLeftAdjBase = d_local_left_adj + t.idx * LOCAL_LEFT_ADJ_SIZE;

  unsigned int n;
  if (lane_id == 0) n = local_n[0];
  n = __shfl_sync(0xFFFFFFFF, n, 0);

  for (unsigned int j = lane_id; j < n; j += 32)
  {
    neiInG[j] = t.neiInG[j];
    neiInP[j] = t.neiInP[j];
  }
  
  initializePCX(lane_id, labelsBase, n, plex, cand, excl);
  __syncwarp();

  for (int local_step = 0; local_step < p.local_bnb_steps; local_step++)
  {

    if (abort[0]) return;

    if (PlexSz + CandSz < q) return;
    if (!edgePotentialBoundCached(k, q, PlexSz, CandSz, edgePotential)) return;

    if (CandSz == 0)
    {
      if (ExclSz == 0 && PlexSz >= q)
      {
        __syncwarp();
        bool maximal = isMaximal_opt(lane_id, k, PlexSz, leftBase, left_count[0], neiInP, neighborsBase, offsetsBase, degreeBase, plex, n, local_sat, local_uni, leftAdjBase, localLeftAdjBase);
        if (maximal && lane_id == 0) atomicAdd(&plex_count[0], 1);
      }
      return;
    }
    __syncwarp();

    int minnei_Plex = INT_MAX;
    int pivot = -1;
    int minnei_Cand = INT_MAX;

    for (int i = lane_id; i < PlexSz; i+=32)
    {
      const int v = plex[i];
      if (neiInG[v] < minnei_Plex)
      {
        minnei_Plex = neiInG[v];
        pivot = v;
      }
    }

      

    for (int offset = 16; offset > 0; offset >>= 1)
    {
      int otherMin = __shfl_down_sync(0xFFFFFFFF, minnei_Plex, offset);
      int otherIdx = __shfl_down_sync(0xFFFFFFFF, pivot, offset);
      if (otherMin < minnei_Plex || (otherMin == minnei_Plex && otherIdx < pivot))
      {
        minnei_Plex = otherMin;
        pivot = otherIdx;
      }
    }

    minnei_Plex = __shfl_sync(0xFFFFFFFF, minnei_Plex, 0);
    pivot = __shfl_sync(0xFFFFFFFF, pivot, 0);

    int pivot_plex = pivot;
      
    if (minnei_Plex + k  < max(q, PlexSz)) return;

    if (minnei_Plex + k < PlexSz + CandSz)
    {     
      minnei_Cand = INT_MAX;
      pivot = -1;
      for (int i = lane_id; i < CandSz; i+=32)
      {
        const int v = cand[i];
        if (!adjContains(adjList, n, v, pivot_plex))
        {
          if (neiInG[v] < minnei_Cand)
          {
            minnei_Cand = neiInG[v];
            pivot = v;
          }
          else if (neiInG[v] == minnei_Cand && neiInP[pivot] > neiInP[v])
          {
            pivot = v;
          }
        }
      }
          
      for (int offset = 16; offset > 0; offset >>= 1)
      {
        int otherMin = __shfl_down_sync(0xFFFFFFFF, minnei_Cand, offset);
        int otherIdx = __shfl_down_sync(0xFFFFFFFF, pivot, offset);
        if (otherMin < minnei_Cand || (otherMin == minnei_Cand && otherIdx != -1 && neiInP[pivot] > neiInP[otherIdx]))
        {
          minnei_Cand = otherMin;
          pivot = otherIdx;
        }
      }

      minnei_Cand = __shfl_sync(0xFFFFFFFF, minnei_Cand, 0);
      pivot = __shfl_sync(0xFFFFFFFF, pivot, 0);

      if (pivot == -1)
      {
        pivot = cand[CandSz - 1];
      }

      if (!enqueue_exclude_task_from_arrays(lane_id, t.idx, n, pivot, plex, PlexSz, cand, CandSz, excl, ExclSz, neiInG, neiInP, edgePotential, neighborsBase, offsetsBase, degreeBase, outTasks, global_tasks, tailPtr, global_tail, d_all_labels, d_all_neiInG, d_all_neiInP, global_labels, global_neiInG, global_neiInP, abort, global_count)) return;

      const bool includeOk = apply_include_branch_unsorted(lane_id, k, q, n, neighborsBase, offsetsBase, degreeBase, commonMtxBase, plex, PlexSz, cand, CandSz, excl, ExclSz, neiInP, neiInG, pivot, adjList, local_sat, edgePotential);
      if (!includeOk) return;
      continue;
    }
    
    int minnei = minnei_Plex;

    for (int i = lane_id; i < CandSz; i+=32)
    {
      const int v = cand[i];
      if (neiInG[v] < minnei)
      {
        minnei = neiInG[v];
        pivot = v;
      }
      else if (neiInG[v] == minnei && neiInP[pivot] > neiInP[v])
      {
        pivot = v;
      }
    }

    for(int offset = 16; offset > 0; offset >>= 1)
    {
      int otherMin = __shfl_down_sync(0xFFFFFFFF, minnei, offset);
      int otherIdx = __shfl_down_sync(0xFFFFFFFF, pivot, offset);

      if (otherMin < minnei || (otherMin == minnei && otherIdx != -1 && neiInP[pivot] > neiInP[otherIdx]))
      {
        minnei = otherMin;
        pivot = otherIdx;
      }
    }
    minnei = __shfl_sync(0xFFFFFFFF, minnei, 0);
    pivot = __shfl_sync(0xFFFFFFFF, pivot, 0);
    if (minnei >= (PlexSz + CandSz - k))
    {
      if (PlexSz + CandSz < q) return;
      bool flag = false;
          
      for (int i = lane_id; i < ExclSz; i+=32)
      {
        const int v = excl[i];
        if (isKplexPC2(v, k, PlexSz+CandSz, PlexSz, CandSz, neiInG, plex, cand, n, neighborsBase, offsetsBase, degreeBase, adjList))
        {
          flag = true;
        }
      }
          
      if (__any_sync(0xFFFFFFFF, flag)) return;

      bool maximal = isMaximalPC_opt(lane_id, k, PlexSz, CandSz, PlexSz+CandSz, leftBase, left_count[0], neiInG, neighborsBase, offsetsBase, degreeBase, plex, cand, n, local_sat, local_uni, leftAdjBase, localLeftAdjBase);
      if (maximal)
      {
        if (lane_id == 0) atomicAdd(&plex_count[0], 1);
      }
          
      return;
    }

    if (!enqueue_exclude_task_from_arrays(lane_id, t.idx, n, pivot, plex, PlexSz, cand, CandSz, excl, ExclSz, neiInG, neiInP, edgePotential, neighborsBase, offsetsBase, degreeBase, outTasks, global_tasks, tailPtr, global_tail, d_all_labels, d_all_neiInG, d_all_neiInP, global_labels, global_neiInG, global_neiInP, abort, global_count)) return;
    const bool includeOk = apply_include_branch_unsorted(lane_id, k, q, n, neighborsBase, offsetsBase, degreeBase, commonMtxBase, plex, PlexSz, cand, CandSz, excl, ExclSz, neiInP, neiInG, pivot, adjList, local_sat, edgePotential);
    if (!includeOk) return;
  }

  if (PlexSz + CandSz < q) return;
  if (!edgePotentialBoundCached(k, q, PlexSz, CandSz, edgePotential)) return;
  if (!enqueue_task_from_arrays(lane_id, t.idx, n, plex, PlexSz, cand, CandSz, excl, ExclSz, neiInG, neiInP, edgePotential, outTasks, global_tasks, tailPtr, global_tail, d_all_labels, d_all_neiInG, d_all_neiInP, global_labels, global_neiInG, global_neiInP, abort, global_count)) return;
  __syncwarp();
}
__device__ int commonEle(int i, int j, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeHop)
{
  int begin1 = offsetsBase[i];
  int begin2 = offsetsBase[j];
  int sz1 = degreeHop[i];
  int sz2 = degreeHop[j];

  int a = 0, b = 0;
  int szdest = 0;

  while (a < sz1 && b < sz2)
  {
    if (neighborsBase[begin1 + a] < neighborsBase[begin2+b]) a++;
    else if (neighborsBase[begin1 + a] > neighborsBase[begin2+b]) b++;
    else{
      szdest++;
      a++, b++;
    }
  }

  return szdest;

}

// commonEle(i,j) can never exceed either vertex's own degreeHop (an
// intersection can't be larger than either set it's drawn from), so if
// min(degreeHop[i], degreeHop[j]) already falls below the threshold we're
// classifying against, the "less than" outcome is guaranteed without running
// the merge-based intersection at all. This matters here specifically: at
// large q the blocks buildCommonMtx runs on are routinely near-complete-clique
// sized, so commonEle's O(degree) cost per pair over O(n^2) pairs is the
// dominant cost of the whole pipeline -- this skip is exact (not an
// approximation) and never changes the classification, it just avoids the
// merge whenever the answer is already decided.
__device__ uint8_t classifyPair(int i, int j, bool alreadyAdjacent, int thresIfAdjacent, int thresIfNotAdjacent, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeHop)
{
  const int T = alreadyAdjacent ? thresIfAdjacent : thresIfNotAdjacent;
  const int minDeg = min(degreeHop[i], degreeHop[j]);
  if (minDeg < T)
  {
    return alreadyAdjacent ? LINK2LESS : UNLINK2LESS;
  }

  const int common = commonEle(i, j, neighborsBase, offsetsBase, degreeHop);
  if (alreadyAdjacent)
  {
    if (common > T) return LINK2MORE;
    if (common == T) return LINK2EQUAL;
    return LINK2LESS;
  }
  else
  {
    if (common > T) return UNLINK2MORE;
    if (common == T) return UNLINK2EQUAL;
    return UNLINK2LESS;
  }
}

__global__ void buildCommonMtx(int idx, P_pointers p, S_pointers s, G_pointers g, uint8_t* commonMtx, unsigned int* d_hopSz)
{
  unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int warp_id = (global_index / 32);
  unsigned int lane_id = threadIdx.x % 32;

  if ((warp_id+WARPS*idx) >= (g.n-p.lb+2)) return;

  int k = p.k;
  int lb = p.lb;
  unsigned int* local_n = s.n + warp_id;
  unsigned int *hopSz = d_hopSz + warp_id;

  unsigned int* offsetsBase = s.offsets + warp_id * MAX_BLK_SIZE;
  unsigned int* neighborsBase = s.neighbors + (size_t)warp_id * MAX_BLK_SIZE * AVG_DEGREE;
  unsigned int* degreeHop = s.degreeHop + warp_id * MAX_BLK_SIZE;

  size_t capacity = size_t(warp_id) * CAP;
  uint8_t* commonMtxBase = commonMtx + capacity;
  const int thresPP1=lb-k-2*max(k-2,0),thresPP2=lb-k-2*max(k-3,0);
  const int thresPC1=lb-2*k-max(k-2,0),thresPC2=lb-k-1-max(k-2,0)-max(k-3,0);
  const int thresCC1=lb-2*k-(k-1),thresCC2=lb-2*k+2-(k-1);

  int hop;
  int n;
  if (lane_id == 0) 
  {
    hop = hopSz[0];
    n = local_n[0];
  }
  hop = __shfl_sync(0xFFFFFFFF, hop, 0);
  n = __shfl_sync(0xFFFFFFFF, n, 0);
    
  for (int i = lane_id; i < hop; i+=32)
  {
    for (int j = 0; j < i; j++)
    {
      const bool alreadyAdjacent = commonMtxBase[i*n+j] != 0;
      commonMtxBase[i*n+j] = classifyPair(i, j, alreadyAdjacent, thresCC1, thresCC2, neighborsBase, offsetsBase, degreeHop);
      commonMtxBase[j*n+i] = commonMtxBase[i*n+j];
    }
  }
  for (int i = hop+lane_id; i < n;i+=32)
  {
    for (int j = 0; j < hop; j++)
    {
      const bool alreadyAdjacent = commonMtxBase[i*n+j] != 0;
      commonMtxBase[i*n+j] = classifyPair(i, j, alreadyAdjacent, thresPC1, thresPC2, neighborsBase, offsetsBase, degreeHop);
      commonMtxBase[j*n+i]=commonMtxBase[i*n+j];
    }
    if (k==2) continue;
    for (int j = hop; j<i; j++)
    {
      const bool alreadyAdjacent = commonMtxBase[i*n+j] != 0;
      commonMtxBase[i*n+j] = classifyPair(i, j, alreadyAdjacent, thresPP1, thresPP2, neighborsBase, offsetsBase, degreeHop);
      commonMtxBase[j*n+i]=commonMtxBase[i*n+j];
    }
  }

  // if (warp_id+WARPS*idx == 4111 && lane_id == 0)
  // {
  // printf("Common Matrix: \n");
  //   for (int i = 0; i < n; i++)
  //   {
  //     for (int j = 0; j < n; j++)
  //     {
  //       printf("%d ", commonMtxBase[i*n+j]);
  //     }
  //     printf("\n\n");
  //   }
  // }
}

__global__ void kSearch(int idx, P_pointers p, S_pointers s, G_pointers g, T_pointers t, unsigned int* d_blk_counter, unsigned int* d_res, unsigned int* d_br, unsigned int* d_state, unsigned int* d_len, unsigned int* d_sz, uint16_t* neiInG, uint16_t* neiInP, unsigned int* plex_count, uint8_t* commonMtx, unsigned int* recCand1, unsigned int* recCand2, unsigned int* d_v2delete, uint32_t* d_adj, unsigned long long* cycles, int* abort_flag, int* d_abort, unsigned int *global_count)
{
  unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int warp_id = (global_index / 32);
  unsigned int lane_id = threadIdx.x % 32;

  if ((warp_id+WARPS*idx) >= (g.n-p.lb+2)) return;

  // if (warp_id >= 10) return;

  int k = p.k;
  int q = p.lb;
  float thres = p.thres;

  unsigned int* counterBase = d_blk_counter + warp_id;

  if (counterBase[0] < q) return;
  

  unsigned int* degreeBase = s.degree + warp_id * MAX_BLK_SIZE;
  unsigned int* offsetsBase = s.offsets + warp_id * MAX_BLK_SIZE;
  unsigned int* neighborsBase = s.neighbors + (size_t)warp_id * MAX_BLK_SIZE * AVG_DEGREE;
  unsigned int* degreeHop = s.degreeHop + warp_id * MAX_BLK_SIZE;

  unsigned int* plex = s.P + warp_id * MAX_BLK_SIZE;
  unsigned int* cand1 = s.C + warp_id * MAX_BLK_SIZE;
  unsigned int* cand2 = s.C2 + warp_id * MAX_BLK_SIZE;
  unsigned int* excl = s.X + warp_id * MAX_BLK_SIZE;

  unsigned int* PlexSz = s.PSize + warp_id;
  unsigned int* Cand1Sz = s.CSize + warp_id;
  unsigned int* Cand2Sz = s.C2Size + warp_id;
  unsigned int* ExclSz = s.XSize + warp_id;

  unsigned int* res = d_res + warp_id * MAX_DEPTH;
  unsigned int* br = d_br + warp_id * MAX_DEPTH;
  unsigned int* state = d_state + warp_id * MAX_DEPTH;
  unsigned int* v2delete = d_v2delete + warp_id * MAX_DEPTH;
  unsigned int* length = d_len + warp_id;
  unsigned int* size = d_sz + warp_id;

  uint16_t* neiInGBase = neiInG + warp_id * MAX_BLK_SIZE;
  uint16_t* neiInPBase = neiInP + warp_id * MAX_BLK_SIZE;

  unsigned int* recCand1Base = recCand1 + warp_id * MAX_BLK_SIZE;
  unsigned int* recCand2Base = recCand2 + warp_id * MAX_BLK_SIZE;

  // t.d_tail_A[0] = 0;

  size_t capacity = size_t(warp_id) * CAP;
  uint8_t* commonMtxBase = commonMtx + capacity;

  uint32_t *adjList = d_adj + ADJSIZE * warp_id;

  int n;
  if (lane_id == 0)
  {
    n = counterBase[0];
  }
  n = __shfl_sync(0xFFFFFFFF, n, 0);

  

  if (lane_id == 0)
  {
  // printf("Common Matrix: \n");
  //   for (int i = 0; i < n; i++)
  //   {
  //     for (int j = 0; j < n; j++)
  //     {
  //       printf("%d ", commonMtxBase[i*n+j]);
  //     }
  //     printf("\n\n");
  //   }
  //   printf("cand1: ");
  //   for (int i = 0; i < sh_Cand1Sz[local_warp_id]; i++)
  //   {
  //     printf("%d ", cand1[i]);
  //   }
  //   printf("\n");

  //   printf("neiInG: ");
  //   for (int i = 0; i < n; i++)
  //   {
  //     printf("%d ", neiInGBase[i]);
  //   }
  //   printf("\n");
  }

  int flag;
  if (lane_id == 0) flag = d_abort[0];
  flag = __shfl_sync(0xFFFFFFFF, flag, 0);
  int sz;

  if (!flag)
  {
    for (int i = lane_id; i < n; i+=32)
    {
      neiInGBase[i] = degreeHop[i];
      neiInPBase[i] = 0;
    }
    
    
    __syncwarp();
    for (int pidx = 0; pidx < PlexSz[0]; pidx++)
    {
      const int pnode = plex[pidx];
      for (int i = lane_id; i < degreeBase[pnode]; i+=32)
      {
        const int nei = neighborsBase[offsetsBase[pnode]+i];
        neiInPBase[nei]++;
      }
      __syncwarp();
    }
    for (int pidx = 1; pidx < PlexSz[0]; pidx++)
    {
      const int pnode = plex[pidx];
      addG(lane_id, pnode, neiInGBase, n, neighborsBase, offsetsBase, degreeBase);
      __syncwarp();
    }
  }


  if (!flag)
  {
    sz = 0;
    res[sz] = k - 1;
    br[sz] = 1;
    state[sz] = 0;
    sz++;
  }
  else 
  {
    sz = size[0];
    if (lane_id == 0) size[0] = 0;
    if (sz == 0) return;
  }
  
  int u, found_idx;
  
  while(sz)
  {
    if (sz >= MAX_DEPTH)
    {
      if (lane_id == 0)
      {
        printf("capacity crossed: %d\n", sz);
      }
      return;
    }
    switch(state[sz-1])
    {
      //reserve my slot, slot -> []
      case 0:
        // printf("state: %d, size: %d, plexsz: %d, cand1sz: %d, cand2sz: %d, exclsz: %d\n", state[sz-1], sz, sh_PlexSz[local_warp_id], sh_Cand1Sz[local_warp_id], sh_Cand2Sz[local_warp_id], sh_ExclSz[local_warp_id]);
        if (Cand2Sz[0] == 0)
        {
          if (!flag)
          {
          if (PlexSz[0] + Cand1Sz[0] < q)
          {
            sz--;
            continue;
          }
          
          bool cond = !upperBoundK2(lane_id, k, q, plex, neiInGBase, PlexSz[0]);
          
          if (PlexSz[0] > 1 && cond)
          {
            sz--;
            continue;
          }
          
          int pos;
          if (lane_id == 0)
          {
            unsigned int* tail = t.d_tail_A;
            pos = atomicAdd(&tail[0], 1u);
          }
          pos = __shfl_sync(0xFFFFFFFF, pos, 0);
          // if (lane_id == 0) printf("pos: %d, max_cap: %d\n", pos, MAX_CAP/4);
          

          uint8_t* newLabels = t.d_all_labels_A + pos * MAX_BLK_SIZE;
          uint16_t* newNeiInG = t.d_all_neiInG_A + pos * MAX_BLK_SIZE;
          uint16_t* newNeiInP = t.d_all_neiInP_A + pos * MAX_BLK_SIZE;
          for (int i = lane_id; i < n; i+=32)
          {
            newLabels[i] = U;
            newNeiInG[i] = neiInGBase[i];
            newNeiInP[i] = neiInPBase[i];
          }
          for (int i = lane_id; i < PlexSz[0]; i+=32)
          {
            const int v = plex[i];
            newLabels[v] = P;
          }
          for (int i = lane_id; i < Cand1Sz[0]; i+=32)
          {
            const int v = cand1[i];
            newLabels[v] = C;
          }
          for (int i = lane_id; i < Cand2Sz[0]; i+=32)
          {
            const int v = cand2[i];
            newLabels[v] = H;
          }
          for (int i = lane_id; i < ExclSz[0]; i+=32)
          {
            const int v = excl[i];
            newLabels[v] = X;
          }

          const unsigned int taskEdgePotential = computeEdgePotentialSum(lane_id, plex, PlexSz[0], cand1, Cand1Sz[0], neiInGBase);
          
          __syncwarp();
          if (lane_id == 0)
          {
            Task &nt = t.d_tasks_A[pos];
            nt.idx = warp_id;
            nt.PlexSz = PlexSz[0];
            nt.CandSz = Cand1Sz[0];
            nt.ExclSz = ExclSz[0];
            nt.edgePotential = taskEdgePotential;
            nt.labels = newLabels;
            nt.neiInG = newNeiInG;
            nt.neiInP = newNeiInP;
            atomicAdd(&global_count[0], 1);
          // __syncwarp();
          // if (warp_id == 0)
          // {
          //   // printf("idx: %d\n", t.idx);
          //   // if (t.idx == 0) return;
          //   // printf("Labels1: %d\n", newLabels[0]);
          //   for (int j = 0; j < MAX_BLK_SIZE; j++)
          //   {
          //     printf("%d ", nt.labels[j]);
          //   }
          //   printf("\n");
          // }
        }
        __syncwarp();
        if (pos+1 > (MAX_CAP*thres)-WARPS)
        {
          // if(lane_id == 0) printf("Maximum Capacity Reached in kSearch\n");
          // atomicExch(abort_flag, 1);
          if (lane_id == 0) abort_flag[0] = 1;
          size[0] = sz;
          state[sz-1] = 0;
          // if (warp_id == 13 && lane_id == 0) printf("%d is returning with sz: %d, state:%d, res: %d, br:%d, plexsz: %d, cand1sz: %d, cand2sz: %d, exclsz: %d\n", warp_id, sz, state[sz-1], res[sz-1], br[sz-1],PlexSz[0], Cand1Sz[0], Cand2Sz[0], ExclSz[0]);
          return;
        }
      }
          // if (lane_id == 0) abort_flag[0] = 0;
          flag = 0;
          sz--;
          continue;
        }
        if (lane_id == 0)
        {
          u = cand2[Cand2Sz[0]-1];
          excl[ExclSz[0]++] = u;
          --Cand2Sz[0];
          v2delete[sz-1] = u;

          res[sz] = res[sz-1];
          br[sz] = 1;
          state[sz] = 0;

          state[sz-1] = 1;
        }
        sz++;
        __syncwarp();
        
        continue;

      case 1:
      // printf("state: %d, size: %d, plexsz: %d, cand1sz: %d, cand2sz: %d, exclsz: %d\n", state[sz-1], sz, sh_PlexSz[local_warp_id], sh_Cand1Sz[local_warp_id], sh_Cand2Sz[local_warp_id], sh_ExclSz[local_warp_id]);      
        if(lane_id == 0)
        {
          state[sz-1] = 2;

          u = v2delete[sz-1];
          cand2[Cand2Sz[0]++] = u;
        }
        u = v2delete[sz-1];
        found_idx = -1;
        for (int base = 0; base < ExclSz[0]; base += 32)
        {
          int idx = base + lane_id;

          bool match = (idx < ExclSz[0]) && (excl[idx] == u);
          unsigned hit = __ballot_sync(0xFFFFFFFF, match);
          if (hit)
          {
            int leader = __ffs(hit) - 1;
            int idx_global = base + leader;
            found_idx = __shfl_sync(0xFFFFFFFF, idx_global, leader);
            break;
          }
        }
          
        if (lane_id == 0 && found_idx >= 0)
        {
          int last = --ExclSz[0];
          int temp = excl[last];
          excl[last] = excl[found_idx];
          excl[found_idx] = temp;
        }
        __syncwarp();
        
        
        continue;
      case 2:
      // printf("state: %d, size: %d, plexsz: %d, cand1sz: %d, cand2sz: %d, exclsz: %d\n", state[sz-1], sz, sh_PlexSz[local_warp_id], sh_Cand1Sz[local_warp_id], sh_Cand2Sz[local_warp_id], sh_ExclSz[local_warp_id]);
        if (br[sz-1] < res[sz-1])
        {
          if (lane_id == 0)
          {
            u = cand2[--Cand2Sz[0]];
            plex[PlexSz[0]++] = u;
          }
          __syncwarp();
          unsigned int node = plex[PlexSz[0]-1];
          for (int i = lane_id; i < degreeBase[node]; i+=32)
          {
            const int nei = neighborsBase[offsetsBase[node]+i];
            neiInPBase[nei]++;
          }
          
          addG(lane_id, node, neiInGBase, n, neighborsBase, offsetsBase, degreeBase);
          // printf("lane_id: %d, node: %d\n", lane_id, node);
          updateCand13(lane_id, cand1, commonMtxBase, recCand1Base, neiInGBase, sz, n, node, &Cand1Sz[0], neighborsBase, degreeBase, offsetsBase);

          const uint8_t* __restrict__ row = commonMtxBase + (size_t) node * n;
          updateCand23(lane_id, cand2, row, recCand2Base, sz, &Cand2Sz[0]);

          if (Cand2Sz[0])
          {
            if (lane_id == 0)
            {
              u = cand2[--Cand2Sz[0]];
              excl[ExclSz[0]++] = u;
              v2delete[sz-1] = u;

              state[sz-1] = 3;
              res[sz] = res[sz-1] - br[sz-1];
              br[sz] = 1;
              state[sz] = 0;
            }
            __syncwarp();
            sz++;
            continue;
          }
          else
          {
            if (lane_id == 0)
            {
              state[sz-1] = 4;
              res[sz] = res[sz-1] - br[sz-1];
              br[sz] = 1;
              state[sz] = 0;
            }
            __syncwarp();
            sz++;
            continue;
          }
        }
        else
        {
          if (lane_id == 0)
          {
            state[sz-1] = 4;
          }
          __syncwarp();
          continue;
        }
      case 3:
      // printf("state: %d, size: %d, plexsz: %d, cand1sz: %d, cand2sz: %d, exclsz: %d\n", state[sz-1], sz, sh_PlexSz[local_warp_id], sh_Cand1Sz[local_warp_id], sh_Cand2Sz[local_warp_id], sh_ExclSz[local_warp_id]);
        if (lane_id == 0)
        {
          br[sz-1]++;
          state[sz-1] = 2;

          u = v2delete[sz-1];
          cand2[Cand2Sz[0]++] = u;
        }
        u = v2delete[sz-1];
        found_idx = -1;
        for (int base = 0; base < ExclSz[0]; base += 32)
        {
          int idx = base + lane_id;

          bool match = (idx < ExclSz[0]) && (excl[idx] == u);
          unsigned hit = __ballot_sync(0xFFFFFFFF, match);
          if (hit)
          {
            int leader = __ffs(hit) - 1;
            int idx_global = base + leader;
            found_idx = __shfl_sync(0xFFFFFFFF, idx_global, leader);
            break;
          }
        }
          
        if (lane_id == 0 && found_idx >= 0)
        {
          int last = --ExclSz[0];
          int temp = excl[last];
          excl[last] = excl[found_idx];
          excl[found_idx] = temp;
        }
        __syncwarp();
        continue;
      case 4:
      // printf("state: %d, size: %d, plexsz: %d, cand1sz: %d, cand2sz: %d, exclsz: %d\n", state[sz-1], sz, sh_PlexSz[local_warp_id], sh_Cand1Sz[local_warp_id], sh_Cand2Sz[local_warp_id], sh_ExclSz[local_warp_id]); 
      if (br[sz-1] == res[sz-1])
      {
        if (!flag)
      { 
          if (lane_id == 0)
          {
            u = cand2[--Cand2Sz[0]];
            plex[PlexSz[0]++] = u;
          }
          __syncwarp();
          unsigned int node = plex[PlexSz[0]-1];
          for (int i = lane_id; i < degreeBase[node]; i+=32)
          {
            const int nei = neighborsBase[offsetsBase[node]+i];
            neiInPBase[nei]++;
          }
          addG(lane_id, node, neiInGBase, n, neighborsBase, offsetsBase, degreeBase);
          updateCand13(lane_id, cand1, commonMtxBase, recCand1Base, neiInGBase, sz, n, node, &Cand1Sz[0], neighborsBase, degreeBase, offsetsBase);
          if (PlexSz[0] + Cand1Sz[0] < q)
          {
            if (lane_id == 0)
            {
              state[sz-1] = 5;
            }
            __syncwarp();
            continue;
          }
          if (PlexSz[0] > 1 && !upperBoundK2(lane_id, k, q, plex, neiInGBase, PlexSz[0]))
          {
            if (lane_id == 0)
            {
              state[sz-1] = 5;
            }
            __syncwarp();
            continue;
          }
          

          int len = 0;
          for (int i = 0; i < Cand1Sz[0]; i++)
          {
            const int v = cand1[i];
            if (!isKplex2(lane_id, v, k, PlexSz[0], neiInPBase, plex, n, neighborsBase, offsetsBase, degreeBase, adjList))
            {
              if (lane_id == 0)
              {
                int temp = cand1[Cand1Sz[0]-1];
                cand1[Cand1Sz[0]-1] = cand1[i];
                cand1[i] = temp;
                Cand1Sz[0]--;
                len++;
              }
              __syncwarp();
              subG(lane_id, v, neiInGBase, n, neighborsBase, offsetsBase, degreeBase);
              i--;
            }
          }
          len = __shfl_sync(0xFFFFFFFF, len, 0);
          if (lane_id == 0) length[0] = len;
          
          int pos;
          if (lane_id == 0)
          {
            unsigned int* tail = t.d_tail_A;
            pos = atomicAdd(&tail[0], 1u);
            state[sz-1] = 5;
          }
          pos = __shfl_sync(0xFFFFFFFF, pos, 0);
          // if (lane_id == 0) printf("pos: %d, max_cap: %d\n", pos, MAX_CAP/4);
          
          size_t baseOff = (size_t)pos * (size_t)MAX_BLK_SIZE;
          uint8_t* newLabels = t.d_all_labels_A + baseOff;
          uint16_t* newNeiInG = t.d_all_neiInG_A + baseOff;
          uint16_t* newNeiInP = t.d_all_neiInP_A + baseOff;
          for (int i = lane_id; i < n; i+=32)
          {
            newLabels[i] = U;
            newNeiInG[i] = neiInGBase[i];
            int temp = neiInPBase[i];
            newNeiInP[i] = temp;
          }
          for (int i = lane_id; i < PlexSz[0]; i+=32)
          {
            const int v = plex[i];
            newLabels[v] = P;
          }
          for (int i = lane_id; i < Cand1Sz[0]; i+=32)
          {
            const int v = cand1[i];
            newLabels[v] = C;
          }
          for (int i = lane_id; i < Cand2Sz[0]; i+=32)
          {
            const int v = cand2[i];
            newLabels[v] = H;
          }
          for (int i = lane_id; i < ExclSz[0]; i+=32)
          {
            const int v = excl[i];
            newLabels[v] = X;
          }
          const unsigned int taskEdgePotential = computeEdgePotentialSum(lane_id, plex, PlexSz[0], cand1, Cand1Sz[0], neiInGBase);
          __syncwarp();
          // if (warp_id == 0 && lane_id == 0)
          // {
          //   // printf("idx: %d\n", t.idx);
          //   // if (t.idx == 0) return;
          //   printf("Labels2: %d\n", newLabels[0]);
          //   // for (int j = 0; j < MAX_BLK_SIZE; j++)
          //   // {
          //   //   printf("%d ", newLabels[j]);
          //   // }
          //   // printf("\n");
          // }
          if (lane_id == 0)
          {
            Task &nt = t.d_tasks_A[pos];
            nt.idx = warp_id;
            nt.PlexSz = PlexSz[0];
            nt.CandSz = Cand1Sz[0];
            nt.ExclSz = ExclSz[0];
            nt.edgePotential = taskEdgePotential;
            nt.labels = newLabels;
            nt.neiInG = newNeiInG;
            nt.neiInP = newNeiInP;
            atomicAdd(&global_count[0], 1);
          }
          __syncwarp();

          if (pos+1 > (MAX_CAP*thres)-WARPS)
          {
            // if(lane_id == 0) printf("Maximum Capacity Reached in kSearch\n");
            // atomicExch(abort_flag, 1);
            if (lane_id == 0) abort_flag[0] = 1;
            size[0] = sz;
            state[sz-1] = 4;
            // if (warp_id == 13 && lane_id == 0) printf("%d is returning with sz: %d, state:%d, res: %d, br:%d, plexsz: %d, cand1sz: %d, cand2sz: %d, exclsz: %d, len: %d\n", warp_id, sz, state[sz-1], res[sz-1], br[sz-1],PlexSz[0], Cand1Sz[0], Cand2Sz[0], ExclSz[0], length[0]);
            return;
          }
        }
        // if (lane_id == 0) abort_flag[0] = 0;
        flag = 0;
          for (int i = 0; i < length[0]; i++)
          {
            if (lane_id == 0) Cand1Sz[0]++;
            __syncwarp();
            const int v = cand1[Cand1Sz[0]-1];
            addG(lane_id, v, neiInGBase, n, neighborsBase, offsetsBase, degreeBase);
          }
          if (lane_id == 0) state[sz-1] = 5;
          continue;
        }
        else
        {
          if (lane_id == 0)
          {
            state[sz-1] = 5;
          }
          __syncwarp();
          continue;
        }
      case 5:
      // printf("state: %d, size: %d, plexsz: %d, cand1sz: %d, cand2sz: %d, exclsz: %d\n", state[sz-1], sz, sh_PlexSz[local_warp_id], sh_Cand1Sz[local_warp_id], sh_Cand2Sz[local_warp_id], sh_ExclSz[local_warp_id]);
        for (int i = br[sz-1]; i >= 1; i--)
        {
          unsigned int node = plex[PlexSz[0]-1];
          if (lane_id == 0)
          {
            cand2[Cand2Sz[0]++] = plex[--PlexSz[0]];
          }
          __syncwarp();
          for (int j = lane_id; j < degreeBase[node]; j+=32)
          {
            const int nei = neighborsBase[offsetsBase[node]+j];
            neiInPBase[nei]--;
          }
          subG(lane_id, node, neiInGBase, n, neighborsBase, offsetsBase, degreeBase);
          __syncwarp();
        }
        
        recoverCand12(lane_id, cand1, recCand1Base, neiInGBase, sz, n, &Cand1Sz[0], neighborsBase, degreeBase, offsetsBase);
        
        recoverCand23(lane_id, n, cand2, recCand2Base, sz, &Cand2Sz[0]);
        __syncwarp();
        
        sz--;
        continue;
    }
  }
}
