// #include "../inc/device_funcs.h"
#include "../inc/device_funcs.h"

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

__global__ void decompose(int i, P_pointers p, G_pointers g, D_pointers d, unsigned int *d_blk, unsigned int *d_blk_counter, unsigned int *d_left, unsigned int *d_left_counter, uint32_t *visited, unsigned int *global_count, unsigned int *left_count, unsigned int *validblk, unsigned int* d_hopSz, unsigned long long* cycles)
{
  unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int warp_id = (global_index / 32);
  unsigned int lane_id = threadIdx.x % 32;

  unsigned int local_warp_id = threadIdx.x >> 5;

  // if (warp_id >= 35) return;

  // unsigned long long t0, t1;

  // if (warp_id + WARPS * i >= 3500) return;
  // if (warp_id + WARPS * i < 291 || warp_id + WARPS * i >= 4390) return;
  // if (warp_id != 2088) return;
    
  long cond = long(g.n) - long(p.lb) + 2;
  if ((warp_id+WARPS*i) >= cond) return;
    
  int vstart = d.dseq[warp_id+WARPS*i];
  int idx;
  unsigned int* blkBase = d_blk + warp_id * MAX_BLK_SIZE;
  unsigned int* counterBase = d_blk_counter + warp_id;

  unsigned int* leftBase = d_left + warp_id * MAX_BLK_SIZE;
  unsigned int* left_counter = d_left_counter + warp_id;
  unsigned int* hopSz = d_hopSz + warp_id;

  int range = (g.n/32)+1;
  uint32_t *visitedBase = visited + warp_id * range;

  __shared__ unsigned int sh_blkCount[WARPS_EACH_BLK];
  __shared__ unsigned int sh_leftCount[WARPS_EACH_BLK];

  if (lane_id == 0) sh_blkCount[local_warp_id] = 0; 
  if (lane_id == 0) sh_leftCount[local_warp_id] = 0;
    
  int lb_2k = p.lb - 2 * p.k; // q - 2*k
  if (lane_id == 0)
  {
    idx = atomicAdd(&sh_blkCount[local_warp_id], 1);
      
    blkBase[idx] = vstart; // seed vertex 
    atomicOr((unsigned int*)&visitedBase[vstart >> 5], (1u << (vstart & 31)));
  }
  __syncwarp();
     
  int start = 0;
  int position  = d.dpos[vstart];
  int hop;
  int deg = g.degree[vstart];
  int off = g.offsets[vstart];

  // while(true)
  // {
      
  //   bool localOverFlow = false;
  //   int overFlowIdx = INT_MAX;
      
    for (int t = start; t < deg; t+=32)
    {
      int i = t+lane_id;
      // unsigned active = __ballot_sync(0xFFFFFFFF, i < deg);
      // if (active == 0) break;
      // if(localOverFlow) break;
      
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
          else{
            atomicSub(&sh_blkCount[local_warp_id], 1);
            // localOverFlow = true;
            // overFlowIdx = i;
            // printf("Above the capacity for blk in %d\n", warp_id);
          }
          
        }
        else
        {
          idx = atomicAdd(&sh_leftCount[local_warp_id], 1);
          if (idx < MAX_BLK_SIZE) 
          {
            leftBase[idx] = nei;
            atomicOr((unsigned int*)&visitedBase[nei >> 5], (1u << (nei & 31)));
          }
          else
          {
            atomicSub(&sh_leftCount[local_warp_id], 1);
            // localOverFlow = true;
            // overFlowIdx = i;
            // printf("Above the capacity for left in %d\n", warp_id);
          }
        }
      }
      // localOverFlow = __any_sync(__activemask(), localOverFlow);
    }
    __syncwarp();

    // int otherMin = overFlowIdx;
    // for(int offset = 16; offset > 0; offset >>= 1)
    // {
    //   otherMin = min(otherMin, __shfl_down_sync(0xFFFFFFFF, otherMin, offset));
    // }
  

    // start = __shfl_sync(0xFFFFFFFF, otherMin, 0);
  
  
  
    size_t sz;
      do{
        sz = sh_blkCount[local_warp_id];
        if (sz - 1 < p.bd) goto CLEAN;
        for (int i = lane_id+1; i < sh_blkCount[local_warp_id]; i+=32)
        {
          size_t cnt = 0;
          unsigned int u = blkBase[i];
          for (int j = 1; j < sh_blkCount[local_warp_id]; j++)
          {
            unsigned int v = blkBase[j];
            if (binary_search(v, g.neighbors+g.offsets[u], g.degree[u])) cnt++;
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
  
      if (lane_id == 0) 
      {
        hopSz[0] = sh_blkCount[local_warp_id];
      }

      __syncwarp();

    
      for (int i = lane_id; i < sh_leftCount[local_warp_id]; i+=32)
      {
        size_t cnt = 0;
        unsigned int u = leftBase[i];
        for (int j = 1; j < sh_blkCount[local_warp_id];j++)
        {
          unsigned int v = blkBase[j];
          if (binary_search(v, g.neighbors+g.offsets[u], g.degree[u])) cnt++;
        }
        if (cnt < lb_2k + 1) atomicAnd((unsigned int*)&visitedBase[u >> 5], ~(1u << (u & 31)));
      }
      __syncwarp();
      if (lane_id == 0)
      {
        int writeIdx = 0;
        int oldCount = sh_leftCount[local_warp_id];
        for (int i = 0; i < oldCount; i++)
        {
          unsigned int u = leftBase[i];
          if (visitedBase[u >> 5] & (1u << (u & 31)))
          {
            leftBase[writeIdx++] = u;
          }
        }
        sh_leftCount[local_warp_id] = writeIdx;
      }
  

      __syncwarp();
  
      // if (!localOverFlow) break;
  // }
      
  hop = hopSz[0];
  for (int i = 0; i < deg; i++)
  {
    int nei = g.neighbors[off + i];
    if (position < d.dpos[nei])
    {
      for (int j = lane_id; j < g.degree[nei]; j+=32)
      {
        int twoHop = g.neighbors[g.offsets[nei]+j];
        if (visitedBase[twoHop >> 5] & (1u << (twoHop & 31))) continue;
        const int thr = lb_2k + ((position < d.dpos[twoHop]) ? 2 : 3);
        if (min(g.degree[twoHop], hop-1) < thr) continue;
        int cnt = 0;
        for (int k = 1; k < hop; k++)
        {
          unsigned int v = blkBase[k];
          if (binary_search(v, g.neighbors+g.offsets[twoHop], g.degree[twoHop]))
          {
            if (++cnt >= thr)
            {
              break;
            }
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
              blkBase[idx] = twoHop;
            }
          }
          else
          {
            if (cnt >= lb_2k + 3)
            {
              idx = atomicAdd(&sh_leftCount[local_warp_id], 1);
              leftBase[idx] = twoHop;
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
    if (sh_blkCount[local_warp_id] >= p.lb){
      atomicAdd(&validblk[0], 1);
      atomicAdd(&counterBase[0], sh_blkCount[local_warp_id]);
      atomicAdd(&left_counter[0], sh_leftCount[local_warp_id]);
    }
  }
  if (lane_id == 0)
  {
    if (sh_blkCount[local_warp_id] > MAX_BLK_SIZE)
    {
      printf("Block Size is greater than the constant\n");
    }
    else if (sh_leftCount[local_warp_id] > MAX_BLK_SIZE)
    {
      printf("Left Size is greater than constant\n");
    }
  }
  CLEAN:
  __syncwarp();
}

__global__ void calculateDegrees(int i , P_pointers p, G_pointers g, S_pointers s, unsigned int *d_blk, unsigned int *d_blk_counter, unsigned int *d_left, unsigned int *d_left_counter, unsigned int *global_count, unsigned int *left_count)
{
  unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int warp_id = (global_index / 32);
  unsigned int lane_id = threadIdx.x % 32;
  if ((warp_id+WARPS*i) >= (g.n-p.lb+2)) return;
  unsigned int* blkBase = d_blk + warp_id * MAX_BLK_SIZE;
  unsigned int* counterBase = d_blk_counter + warp_id;

  unsigned int* leftBase = d_left + warp_id * MAX_BLK_SIZE;
  unsigned int* left_counter = d_left_counter + warp_id;

  if (counterBase[0] < p.lb) return;
  unsigned int* local_n = s.n + warp_id;

  unsigned int* degreeBase = s.degree + warp_id * (MAX_BLK_SIZE);
  unsigned int* l_degreeBase = s.l_degree + warp_id * (MAX_BLK_SIZE);

  int counter;
  int l_counter;

  if (lane_id == 0)
  {
    local_n[0] = counterBase[0];
    counter = counterBase[0];
    l_counter = left_counter[0];
  }

  counter = __shfl_sync(0xFFFFFFFF, counter, 0);
  l_counter = __shfl_sync(0xFFFFFFFF, l_counter, 0);

  for (int idx = lane_id; idx < counter; idx+=32)
  {
    unsigned int origin = blkBase[idx];
    int ne = 0, neLeft = 0;
    for (int j = 0; j < g.degree[origin];j++)
    {
      unsigned int nei = g.neighbors[g.offsets[origin]+j];
      if (basic_search(nei, blkBase, counter))
      {
        ne++;
      }
      else if (basic_search(nei, leftBase, l_counter))
      {
        neLeft++;
      }
    }

    degreeBase[idx] = ne;
    l_degreeBase[idx] = neLeft;
  }
}

__global__ void fillNeighbors(int i, S_pointers s, P_pointers p, G_pointers g, unsigned int *d_blk, unsigned int *d_blk_counter, unsigned int *d_left, unsigned int *d_left_counter, unsigned int *d_hopSz, uint8_t* commonMtx, uint32_t* d_adj)
{
  unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int warp_id = (global_index / 32);
  unsigned int lane_id = threadIdx.x % 32;

  if ((warp_id+WARPS*i) >= (g.n-p.lb+2)) return;
  int value = warp_id+WARPS*i;
  unsigned int* blkBase = d_blk + warp_id * MAX_BLK_SIZE;
  unsigned int* counterBase = d_blk_counter + warp_id;

  unsigned int* offsetsBase = s.offsets + warp_id * (MAX_BLK_SIZE);
  unsigned int* l_offsetsBase = s.l_offsets + warp_id * (MAX_BLK_SIZE);

  unsigned int* degreeBase = s.degree + warp_id * (MAX_BLK_SIZE);
  unsigned int* l_degreeBase = s.l_degree + warp_id * (MAX_BLK_SIZE);
  unsigned int* degreeHop = s.degreeHop + warp_id * (MAX_BLK_SIZE);

  unsigned int* leftBase = d_left + warp_id * MAX_BLK_SIZE;
  unsigned int* left_counter = d_left_counter + warp_id;

  unsigned int* hopSz = d_hopSz + warp_id;

  unsigned int* neighborsBase = s.neighbors + warp_id * (MAX_BLK_SIZE) * (AVG_DEGREE);
  unsigned int* l_neighborsBase = s.l_neighbors + warp_id * (MAX_BLK_SIZE) * (AVG_LEFT_DEGREE);

  unsigned int* local_n = s.n + warp_id;
  unsigned int* local_m = s.m + warp_id;
  unsigned int* PlexSz = s.PSize + warp_id;
  // uint8_t* commonMtxBase = commonMtx + warp_id * CAP;
  size_t capacity = size_t(warp_id) * CAP;
  uint8_t* commonMtxBase = commonMtx + capacity;

  uint32_t* adjList = d_adj + warp_id * ADJSIZE;

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
    unsigned int l_offset = l_offsetsBase[idx];
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

    cnt = 0;
    for (int j = 0; j < l_counter; j++)
    {
      unsigned int nei = leftBase[j];
      if (binary_search(nei, g.neighbors+g.offsets[origin], g.degree[origin]))
      {
        l_neighborsBase[l_offset+cnt] = j;
        cnt++;
      }
    }
  }

  __syncwarp();

  for (int i = lane_id+1; i < hop; i += 32)
  {
    cand1[i-1] = i;
  }
  for (int i = lane_id+hop; i < counter; i+=32)
  {
    cand2[i-hop] = i;
  }
  if (lane_id == 0) plex[0] = 0;
  local_m[0] = offsetsBase[counter];
  CandSz[0] = hop-1;
  Cand2Sz[0] = counter - hop;
  ExclSz[0] = 0;
  PlexSz[0] = 1;

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
    int check = u * n + v;
    if (neiInP[u] + k == PlexSz && !((adjList[check >> 5] >> (check & 31)) & 1u)) localPass = false;
  }
  if (__any_sync(mask, !localPass)) return false;
  return true;
}

__device__ bool isKplex3(int v, int k, unsigned int PlexSz, uint16_t* neiInP, unsigned int* plex, unsigned int n, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeBase, uint32_t* adjList)
{
  if (neiInP[v] + k <  (PlexSz+1))
  {
    return false;
  }
  for (int i = 0; i < PlexSz; i++)
  {
    const int u = plex[i];
    int check = u * n + v;
    if (neiInP[u] + k == PlexSz && !((adjList[check >> 5] >> (check & 31)) & 1u)) return false;
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
    int check = u * n + v;
    if (missing[u] + k == (totalSz) && !((adjList[check >> 5] >> (check & 31)) & 1u))
        return false;
  }
  for (int i = 0; i < CandSz; i++)
  {
    const int u = cand[i];
    int check = u * n + v;
    if (missing[u] + k == (totalSz) && !((adjList[check >> 5] >> (check & 31)) & 1u))
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

__device__ bool canGlobalAdd(uint16_t u, int k, unsigned int PlexSz, unsigned int* plex, unsigned int* l_neighborsBase, unsigned int* l_offsetsBase, unsigned int* l_degreeBase)
{
  int cnt = 0;
  for (int i = 0; i < PlexSz; i++)
  {
    unsigned int v = plex[i];
    if (binary_search(u, l_neighborsBase + l_offsetsBase[v], l_degreeBase[v]))
    {
      cnt++;
    }
  }
  if (cnt + k >= PlexSz + 1) return true;
  return false;
}

__device__ bool canGlobalAdd2(uint16_t u, int k, unsigned int PlexSz, unsigned int CandSz, unsigned int totalSz, unsigned int* plex, unsigned int* cand, unsigned int* l_neighborsBase, unsigned int* l_offsetsBase, unsigned int* l_degreeBase)
{
  int cnt = 0;
  for (int i = 0; i < PlexSz; i++)
  {
    unsigned int v = plex[i];
    if (binary_search(u, l_neighborsBase + l_offsetsBase[v], l_degreeBase[v]))
    {
      cnt++;
    }
  }
  for (int i = 0; i < CandSz; i++)
  {
    unsigned int v = cand[i];
    if (binary_search(u, l_neighborsBase + l_offsetsBase[v], l_degreeBase[v]))
    {
      cnt++;
    }
  }
  if (cnt + k >= totalSz + 1) return true;
  return false;
}
__device__ int intersection(uint16_t* lst1, unsigned int* lst2, int sz1, int sz2, uint16_t* dest)
{
  int i = 0, j = 0;
  int szdest = 0;
  while (i < sz1 && j < sz2)
  {
    if (lst1[i] == lst2[j])
    {
      dest[szdest++] = lst1[i];
      i++;
      j++;
    }
    else if (lst1[i] < lst2[j])
    {
      i++;
    }
    else
    {
      j++;
    }
  }
  return szdest;
}
__device__ bool isMaximal_opt(unsigned int lane_id, int k, unsigned int PlexSz, unsigned int* left, unsigned int left_count, unsigned int* l_neighborsBase, unsigned int* l_offsetsBase, unsigned int* l_degreeBase, uint16_t* nonNeigh, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeBase, unsigned int* plex, unsigned int n, uint16_t* local_sat, uint16_t* local_commons, uint32_t* uni)
{
  const unsigned FULL_MASK = 0xFFFFFFFFu;

  bool max = true;

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
    }
    unsigned mask = __ballot_sync(FULL_MASK, pred);

    int off = __shfl_sync(FULL_MASK, nsat, 0);
    if (lane_id == 0) nsat += __popc(mask);

    int l_off = __popc(mask & ((1u << lane_id) - 1));
    if (pred) local_sat[off + l_off] = v;
  }
  nsat = __shfl_sync(FULL_MASK, nsat, 0);
  if (nsat)
  {
    uint16_t v0 = local_sat[0];
    int start = l_offsetsBase[v0];
    int end = l_offsetsBase[v0+1];
    int szcom;

    for (int j = start + lane_id; j < end; j += 32)
    {
      local_commons[j - start] = l_neighborsBase[j];
    }
    szcom = end - start;
    szcom = __shfl_sync(FULL_MASK, szcom, 0);

    for (int i = 1; i < nsat; i++)
    {
      if (lane_id == 0)
      {
        uint16_t v = local_sat[i];
        szcom = intersection(local_commons, l_neighborsBase+l_offsetsBase[v], szcom, l_degreeBase[v], local_commons);
      }
      szcom = __shfl_sync(FULL_MASK, szcom, 0);
    }

    if (nsat == 1)
    {
      if (lane_id == 0)
      {
        for (int t = 0; t < k; t++)
        {
          unsigned int u = plex[t];
          int s = l_offsetsBase[u];
          int e = l_offsetsBase[u+1];
          for (int j = s; j < e; j++)
          {
            unsigned int v = l_neighborsBase[j];
            uni[v >> 5] |=  (1u << (v & 31));
          }
        }
    }

      int write = 0;
      if (lane_id == 0)
      {
        for (int i = 0; i < szcom; i++)
        {
          uint16_t v = local_commons[i];
          if ((uni[v >> 5] >> (v & 31)) & 1u)
          {
            local_commons[write++] = v;
          }
        }
        szcom = write;
      }
      szcom = __shfl_sync(FULL_MASK, szcom, 0);
    }

    bool found = false;
    for (int i = lane_id; i < szcom; i+=32)
    {
      uint16_t u = local_commons[i];
      if (canGlobalAdd(u, k, PlexSz, plex, l_neighborsBase, l_offsetsBase, l_degreeBase)) found = true;
    }
    if (__any_sync(FULL_MASK, found))
    {
      if (lane_id == 0) max = false;
    }
    for (int i = lane_id; i < 32; i+=32)
    {
      uni[i] = 0;
    }
    __syncwarp();
  }
    else
    {
      int szcom = 0;
      if (lane_id == 0)
      {
        for (int i = 0; i < k; i++)
        {
          unsigned int u;
          u = plex[i];
          int start = l_offsetsBase[u];
          int end = l_offsetsBase[u+1];
          for (int j = start; j < end; j++)
          {
            unsigned int v = l_neighborsBase[j];
            if(uni[v >> 5] >> (v & 31) & 1) continue;
            local_commons[szcom++] = v;
            int wordIdx = v / 32;
            int bitIdx = v % 32;
            uni[wordIdx] |= (1u << bitIdx);
          }
        }
      }
      szcom = __shfl_sync(FULL_MASK, szcom, 0);

      bool found = false;
      for (int i = lane_id; i < szcom; i+=32)
      {
        uint16_t u = local_commons[i];
        if (canGlobalAdd(u, k, PlexSz, plex, l_neighborsBase, l_offsetsBase, l_degreeBase)) found = true;
      }
      if (__any_sync(FULL_MASK, found))
      {
        if (lane_id == 0) max = false;
      }
      for (int i = lane_id; i < 32; i+=32) uni[i] = 0;
    }
  max = __shfl_sync(0xFFFFFFFF, max, 0);

  return max;
}

__device__ bool isMaximalPC_opt(unsigned int lane_id, int k, unsigned int PlexSz, unsigned int CandSz, unsigned int totalSz, unsigned int* left, unsigned int left_count, unsigned int* l_neighborsBase, unsigned int* l_offsetsBase, unsigned int* l_degreeBase, uint16_t* nonNeigh, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeBase, unsigned int* plex, unsigned int* cand,  unsigned int n, uint16_t* local_sat, uint16_t* local_commons, uint32_t* uni)
{
  const unsigned FULL_MASK = 0xFFFFFFFFu;

  bool max = true;

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
    }
    unsigned mask = __ballot_sync(FULL_MASK, pred);
    int off = __shfl_sync(FULL_MASK, nsat, 0);
    if (lane_id == 0) nsat += __popc(mask);
    nsat = __shfl_sync(FULL_MASK, nsat, 0);
    int l_off = __popc(mask & ((1u << lane_id) - 1));
    if (pred) local_sat[off + l_off] = v;
  }
  nsat = __shfl_sync(FULL_MASK, nsat, 0);
  if (nsat)
  {
    uint16_t v0 = local_sat[0];
    int start = l_offsetsBase[v0];
    int end = l_offsetsBase[v0+1];
    int szcom;

    for (int j = start + lane_id; j < end; j += 32)
    {
      local_commons[j - start] = l_neighborsBase[j];
    }
    szcom = end - start;
    szcom = __shfl_sync(FULL_MASK, szcom, 0);

    for (int i = 1; i < nsat; i++)
    {
      if (lane_id == 0)
      {
        uint16_t v = local_sat[i];
        szcom = intersection(local_commons, l_neighborsBase+l_offsetsBase[v], szcom, l_degreeBase[v], local_commons);
      }
      szcom = __shfl_sync(FULL_MASK, szcom, 0);
    }

    if (nsat == 1)
    {
      if (lane_id == 0)
      {
        for (int t = 0; t < k; t++)
        {
          unsigned int u = (PlexSz > t) ? plex[t] : cand[t-PlexSz];
          int s = l_offsetsBase[u];
          int e = l_offsetsBase[u+1];
          for (int j = s; j < e; j++)
          {
            unsigned int v = l_neighborsBase[j];
            uni[v >> 5] |=  (1u << (v & 31));
          }
        }
    }

      int write = 0;
      if (lane_id == 0)
      {
        for (int i = 0; i < szcom; i++)
        {
          uint16_t v = local_commons[i];
          if ((uni[v >> 5] >> (v & 31)) & 1u)
          {
            local_commons[write++] = v;
          }
        }
        szcom = write;
      }
      szcom = __shfl_sync(FULL_MASK, szcom, 0);
    }

    bool found = false;
    for (int i = lane_id; i < szcom; i+=32)
    {
      uint16_t u = local_commons[i];
      if (canGlobalAdd2(u, k, PlexSz, CandSz, totalSz, plex, cand, l_neighborsBase, l_offsetsBase, l_degreeBase)) found = true;
    }
    if (__any_sync(FULL_MASK, found))
    {
      if (lane_id == 0) max = false;
    }
    for (int i = lane_id; i < 32; i+=32)
    {
      uni[i] = 0;
    }
    __syncwarp();
  }
    else
    {
      int szcom = 0;
      if (lane_id == 0)
      {
        for (int i = 0; i < k; i++)
        {
          unsigned int u;
          if (PlexSz > i) u = plex[i];
          else u = cand[i-PlexSz];
          int start = l_offsetsBase[u];
          int end = l_offsetsBase[u+1];
          for (int j = start; j < end; j++)
          {
            unsigned int v = l_neighborsBase[j];
            if(uni[v >> 5] >> (v & 31) & 1) continue;
            local_commons[szcom++] = v;
            int wordIdx = v / 32;
            int bitIdx = v % 32;
            uni[wordIdx] |= (1u << bitIdx);
          }
        }
      }
      szcom = __shfl_sync(FULL_MASK, szcom, 0);

      bool found = false;
      for (int i = lane_id; i < szcom; i+=32)
      {
        uint16_t u = local_commons[i];
        if (canGlobalAdd2(u, k, PlexSz, CandSz, totalSz, plex, cand, l_neighborsBase, l_offsetsBase, l_degreeBase)) found = true;
      }
      if (__any_sync(FULL_MASK, found))
      {
        if (lane_id == 0) max = false;
      }
      for (int i = lane_id; i < 32; i+=32) uni[i] = 0;
    }
  max = __shfl_sync(0xFFFFFFFF, max, 0);

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

__device__ void remove_sorted_value(unsigned int* data, unsigned int& sz, unsigned int value)
{
  for (unsigned int i = 0; i < sz; i++)
  {
    if (data[i] == value)
    {
      for (unsigned int j = i + 1; j < sz; j++)
      {
        data[j - 1] = data[j];
      }
      sz--;
      return;
    }
  }
}

__device__ void insert_sorted_value(unsigned int* data, unsigned int& sz, unsigned int value)
{
  unsigned int pos = sz;
  while (pos > 0 && data[pos - 1] > value)
  {
    data[pos] = data[pos - 1];
    pos--;
  }
  data[pos] = value;
  sz++;
}



// __device__ void enqueue_exclude_child(int lane_id, int idx, unsigned int* local_n, unsigned int* plex, unsigned int PlexSz, unsigned int* cand, unsigned int CandSz, unsigned int* excl, unsigned int ExclSz, int minIndex, Task* tasks, Task* global_tasks, unsigned int* tailPtr, unsigned int* global_tail, uint8_t* d_all_labels, uint16_t* d_all_neiInG, uint16_t* d_all_neiInP, uint8_t* global_labels, uint16_t* global_neiInG, uint16_t* global_neiInP, uint16_t* neiInG, uint16_t* neiInP, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeBase, int* abort, unsigned int* global_count)
// {
//     uint8_t* labels = d_all_labels;
//     uint16_t* all_neiInG = d_all_neiInG;
//     uint16_t* all_neiInP = d_all_neiInP;
//     Task* localTasks = tasks;
//     unsigned int pos2;
//     if (lane_id == 0) pos2 = atomicAdd(&tailPtr[0], 1u);
//     pos2 = __shfl_sync(0xFFFFFFFF, pos2, 0);
//     if (pos2 + 1> (SMALL_CAP)-WARPS) 
//     {
//       if (lane_id == 0)
//       {
//       atomicSub(&tailPtr[0], 1);
//       pos2 = atomicAdd(&global_tail[0], 1u);
//       // printf("position: %d, idx: %d, exclude\n", pos2, idx);
//       }
//       pos2 = __shfl_sync(0xFFFFFFFF, pos2, 0);
//       if (pos2 + 1 >= (MAX_CAP)-WARPS)
//       {
//         if (lane_id == 0)
//         {
//         // printf("Maximum Capacity Reached.\n");
//         // global_tail[0] = 0;
//         abort[0] = 1;
//         }
//         // return;
//       }
//       labels = global_labels;
//       all_neiInG = global_neiInG;
//       all_neiInP = global_neiInP;
//       localTasks = global_tasks;
//     }
//     uint8_t* childLabels2 = labels + pos2*MAX_BLK_SIZE;
//     uint16_t* childNeiInG2 = all_neiInG + pos2*MAX_BLK_SIZE;
//     uint16_t* childNeiInP2 = all_neiInP + pos2*MAX_BLK_SIZE;

//     for (int j = lane_id; j < local_n[0]; j+=32)
//     {
//       childLabels2[j] = U;
//       childNeiInG2[j] = neiInG[j];
//       childNeiInP2[j] = neiInP[j];
//     }

//     for (int j = lane_id; j < PlexSz; j+=32)
//     {
//       unsigned int v = plex[j];
//       childLabels2[v] = P;
//     }
//     for (int j = lane_id; j < CandSz; j+=32)
//     {
//       unsigned int v = cand[j];
//       childLabels2[v] = C;
//     }
//     for (int j = lane_id; j < ExclSz; j+=32)
//     {
//       unsigned int v = excl[j];
//       childLabels2[v] = X;
//     }
//     subG(lane_id, minIndex, childNeiInG2, local_n[0], neighborsBase, offsetsBase, degreeBase);
//     if (lane_id == 0)
//     {
//     Task &nt2 = localTasks[pos2];
//     nt2.idx = idx;
//     childLabels2[minIndex] = X;
//     nt2.PlexSz = PlexSz;
//     nt2.CandSz = CandSz - 1;
//     nt2.ExclSz = ExclSz + 1;
//     nt2.labels = childLabels2;
//     nt2.neiInG = childNeiInG2;
//     nt2.neiInP = childNeiInP2;
//     atomicAdd(&global_count[0], 1);
//     }
// }

__device__ void apply_exclude_branch(int lane_id, unsigned int n, unsigned int* cand, unsigned int& CandSz, unsigned int* excl, unsigned int& ExclSz, uint16_t* neiInG, int minIndex, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeBase)
{
  if (lane_id == 0)
  {
    remove_sorted_value(cand, CandSz, minIndex);
    insert_sorted_value(excl, ExclSz, minIndex);
  }
  CandSz = __shfl_sync(0xFFFFFFFF, CandSz, 0);
  ExclSz = __shfl_sync(0xFFFFFFFF, ExclSz, 0);
  __syncwarp();

  subG(lane_id, minIndex, neiInG, n, neighborsBase, offsetsBase, degreeBase);
  __syncwarp();
}

__device__ bool apply_include_branch(int lane_id, int k, int lb, unsigned int n, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeBase, uint8_t* commonMtx, unsigned int* plex, unsigned int& PlexSz, unsigned int* cand, unsigned int& CandSz, unsigned int* excl, unsigned int& ExclSz, uint16_t* neiInP, uint16_t* neiInG, int minIndex, uint32_t* adjList)
{
  if (lane_id == 0)
  {
    insert_sorted_value(plex, PlexSz, minIndex);
    remove_sorted_value(cand, CandSz, minIndex);
  }
  PlexSz = __shfl_sync(0xFFFFFFFF, PlexSz, 0);
  CandSz = __shfl_sync(0xFFFFFFFF, CandSz, 0);
  __syncwarp();

  for (int j = lane_id; j < degreeBase[minIndex]; j+=32)
  {
    const int nei = neighborsBase[offsetsBase[minIndex]+j];
    neiInP[nei]++;
  }
  __syncwarp();

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

    const bool keep = active && !(row[v] < UNLINK2EQUAL) && (isKplex3(v, k, PlexSz, neiInP, plex, n, neighborsBase, offsetsBase, degreeBase, adjList));

    const unsigned activemask = __ballot_sync(0xFFFFFFFF, active);
    unsigned keepmask = __ballot_sync(0xFFFFFFFF, keep);
    unsigned dropmask = activemask ^ keepmask;

    const int keep_rank = __popc(keepmask & ((1u << lane_id) - 1));
    const int num_keep  = __popc(keepmask);

    if (active && keep)
    {
      cand[write + keep_rank] = v;
    }

    while (dropmask)
    {
      const int leader = __ffs(dropmask) - 1;
      const unsigned vdrop = __shfl_sync(0xFFFFFFFF, v, leader);
      subG(lane_id, vdrop, neiInG, n, neighborsBase, offsetsBase, degreeBase);
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
  CandSz = write;
  __syncwarp();

  bool ub = upperBound2(lane_id, k, lb, plex, neiInG, PlexSz);
  if (!ub) return false;

  read  = 0;
  write = 0;
  size = ExclSz;

  while (read < size)
  {
    const int take = min(32, size - read);
    const bool active = (lane_id < take);

    unsigned int v = 0;
    if (active) v = excl[read+lane_id];

    const bool keep = active && !(row[v] < UNLINK2MORE) && (isKplex3(v, k, PlexSz, neiInP, plex, n, neighborsBase, offsetsBase, degreeBase, adjList));

    const unsigned activemask = __ballot_sync(0xFFFFFFFF, active);
    unsigned keepmask = __ballot_sync(0xFFFFFFFF, keep);

    const int keep_rank = __popc(keepmask & ((1u << lane_id) - 1));
    const int num_keep  = __popc(keepmask);

    if (active && keep)
    {
      excl[write + keep_rank] = v;
    }

    if (lane_id == 0)
    {
      read += take;
      write += num_keep;
    }
    read  = __shfl_sync(0xFFFFFFFF, read, 0);
    write = __shfl_sync(0xFFFFFFFF, write, 0);
  }
  ExclSz = write;
  __syncwarp();
  return true;
}

// __device__ void enqueue_include_child(int lane_id, int idx, int k, int lb, unsigned int* local_n, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeBase, uint8_t* commonMtx, unsigned int* plex, unsigned int& PlexSz, unsigned int* cand, unsigned int& CandSz, unsigned int* excl, unsigned int& ExclSz, uint16_t* neiInP, uint16_t* neiInG, int minIndex, Task* tasks, Task* global_tasks, unsigned int* tailPtr, unsigned int* global_tail, uint8_t* d_all_labels, uint16_t* d_all_neiInG, uint16_t* d_all_neiInP, uint8_t* global_labels, uint16_t* global_neiInG, uint16_t* global_neiInP, unsigned long long* cycles, uint32_t* adjList, uint16_t* local_sat, int* abort, unsigned int* global_count)
// {
//   uint8_t* labels = d_all_labels;
//   uint16_t* all_neiInG = d_all_neiInG;
//   uint16_t* all_neiInP = d_all_neiInP;
//   Task* localTasks = tasks;
    
//   if (lane_id == 0)
//   {
//     plex[PlexSz++] = minIndex;
//     for (int i = 0; i < CandSz; i++)
//     {
//       if(cand[i] == minIndex)
//       {
//         int temp = cand[i];
//         cand[i] = cand[CandSz-1];
//         cand[CandSz-1] = temp;
//         CandSz--;
//         break;
//       }
//     }
//   }
//   PlexSz = __shfl_sync(0xFFFFFFFF, PlexSz, 0);
//   CandSz = __shfl_sync(0xFFFFFFFF, CandSz, 0);

//   __syncwarp();

//   for (int j = lane_id; j < degreeBase[minIndex]; j+=32)
//   {
//     const int nei = neighborsBase[offsetsBase[minIndex]+j];
//     neiInP[nei]++;
//   }
//   __syncwarp();

//   const uint8_t* row = commonMtx + (size_t)minIndex * local_n[0];

//   int read  = 0;
//   int write = 0;
//   int size = CandSz;

//   while (read < size)
//   {
//     const int take = min(32, size - read);
//     const bool active = (lane_id < take);

//     unsigned int v = 0;
//     if (active) v = cand[read+lane_id];

//     const bool keep = active && !(row[v] < UNLINK2EQUAL) && (isKplex3(v, k, PlexSz, neiInP, plex, local_n[0], neighborsBase, offsetsBase, degreeBase, adjList));

//     const unsigned activemask = __ballot_sync(0xFFFFFFFF, active);
//     unsigned keepmask = __ballot_sync(0xFFFFFFFF, keep);
//     unsigned dropmask = activemask ^ keepmask;

//     const int keep_rank = __popc(keepmask & ((1u << lane_id) - 1));
//     const int num_keep  = __popc(keepmask);

//     if (active && keep)
//     {
//       cand[write + keep_rank] = v;
//     }


//     while (dropmask)
//     {
//       const int leader = __ffs(dropmask) - 1;
//       const unsigned vdrop = __shfl_sync(0xFFFFFFFF, v, leader);
//       subG(lane_id, vdrop, neiInG, local_n[0], neighborsBase, offsetsBase, degreeBase);
//       dropmask &= (dropmask - 1);
//     }

//     if (lane_id == 0)
//     {
//       read += take;
//       write += num_keep;
//     }
//     read  = __shfl_sync(0xFFFFFFFF, read, 0);
//     write = __shfl_sync(0xFFFFFFFF, write, 0);
//   }
//   CandSz = write;
//   __syncwarp();

//   bool ub = upperBound2(lane_id, k, lb, plex, neiInG, PlexSz);
    
//   if (ub)
//   {
//     read  = 0;
//     write = 0;
//     size = ExclSz;

//     while (read < size)
//     {
//       const int take = min(32, size - read);
//       const bool active = (lane_id < take);

//       unsigned int v = 0;
//       if (active) v = excl[read+lane_id];

//       const bool keep = active && !(row[v] < UNLINK2MORE) && (isKplex3(v, k, PlexSz, neiInP, plex, local_n[0], neighborsBase, offsetsBase, degreeBase, adjList));

//       const unsigned activemask = __ballot_sync(0xFFFFFFFF, active);
//       unsigned keepmask = __ballot_sync(0xFFFFFFFF, keep);

//       const int keep_rank = __popc(keepmask & ((1u << lane_id) - 1));
//       const int num_keep  = __popc(keepmask);

//       if (active && keep)
//       {
//         excl[write + keep_rank] = v;
//       }

//       if (lane_id == 0)
//       {
//         read += take;
//         write += num_keep;
//       }
//       read  = __shfl_sync(0xFFFFFFFF, read, 0);
//       write = __shfl_sync(0xFFFFFFFF, write, 0);
//     }
//     ExclSz = write;
//     __syncwarp();
      
//     unsigned int pos;
//     if (lane_id == 0) pos = atomicAdd(&tailPtr[0], 1u);
//     pos = __shfl_sync(0xFFFFFFFF, pos, 0);
//     bool use_global = (pos + 1 > (SMALL_CAP)-WARPS);

//     if (use_global) 
//     {
//       if (lane_id == 0)
//       {
//         atomicSub(&tailPtr[0], 1);
//         pos = atomicAdd(&global_tail[0], 1u);
//         // printf("position: %d, idx: %d, include\n", pos, idx);
//       }
//       pos = __shfl_sync(0xFFFFFFFF, pos, 0);
//       if (pos + 1 >= (MAX_CAP)-WARPS)
//       {
//         if (lane_id == 0)
//         {
//         // printf("Maximum Capacity Reached.\n");
//         // global_tail[0] = 0;
//         abort[0] = 1;
//         }
//         // return;
//       }
//       labels = global_labels;
//       all_neiInG = global_neiInG;
//       all_neiInP = global_neiInP;
//       localTasks = global_tasks;
//     }
//     uint8_t* childLabels = labels + pos*MAX_BLK_SIZE;
//     uint16_t* childNeiInG = all_neiInG + pos*MAX_BLK_SIZE;
//     uint16_t* childNeiInP = all_neiInP + pos*MAX_BLK_SIZE;

//     for (int j = lane_id; j < local_n[0]; j+=32)
//     {
//       childLabels[j] = U;
//       childNeiInG[j] = neiInG[j];
//       childNeiInP[j] = neiInP[j];
//     }
//     for (int j = lane_id; j < PlexSz; j+=32)
//     {
//       unsigned int v = plex[j];
//       childLabels[v] = P;
//     }
//     for (int j = lane_id; j < CandSz; j+=32)
//     {
//       unsigned int v = cand[j];
//       childLabels[v] = C;
//     }
//     for (int j = lane_id; j < ExclSz; j+=32)
//     {
//       unsigned int v = excl[j];
//       childLabels[v] = X;
//     }
//     Task &nt = localTasks[pos];
//     nt.idx = idx;

//     __syncwarp();
//     if (lane_id == 0)
//     {
//       nt.PlexSz = PlexSz;
//       nt.CandSz = CandSz;
//       nt.ExclSz = ExclSz;
//       nt.labels = childLabels;
//       nt.neiInG = childNeiInG;
//       nt.neiInP = childNeiInP;
//       atomicAdd(&global_count[0], 1);
//     }        
//   }
//   __syncwarp();
// }

__device__ bool append_delta_decision(int lane_id, const TinyTask& source, unsigned int pivot, unsigned int branch_decision, BranchLog* Delta, unsigned int* delta_tail, int* abort, unsigned int& childLogStart, unsigned int& childLogLen)
{
  childLogLen = source.log_len + 1;
  if (source.log_len >= REPLAY_STACK_CAP)
  {
    if (lane_id == 0) abort[0] = 3;
    __syncwarp();
    return false;
  }

  if(lane_id == 0) childLogStart = atomicAdd(&delta_tail[0], 1u);
  childLogStart = __shfl_sync(0xFFFFFFFFu, childLogStart, 0);

  if (childLogStart >= DELTA_CAP)
  {
    if (lane_id == 0) abort[0] = 2;
    __syncwarp();
    return false;
  }
  if (lane_id == 0)
  {
    Delta[childLogStart].pivot = pivot;
    Delta[childLogStart].branch_decision = branch_decision;
    Delta[childLogStart].prev = source.log_start;
  }

  return true;
}

__device__ bool enqueue_tiny_record(int lane_id, const TinyTask& source, unsigned int PlexSz, unsigned int CandSz, unsigned int ExclSz, TinyTask* tasks, TinyTask* global_tasks, unsigned int* tailPtr, unsigned int* global_tail, int* abort, unsigned int* global_count)
{
  TinyTask* localTasks = tasks;

  unsigned int pos;
  if (lane_id == 0) pos = atomicAdd(&tailPtr[0], 1u);
  pos = __shfl_sync(0xFFFFFFFFu, pos, 0);

  if (pos + 1 > (TINY_FRONTIER_CAP)-WARPS)
  {
    if (lane_id == 0)
    {
      atomicSub(&tailPtr[0], 1);
      pos = atomicAdd(&global_tail[0], 1u);
    }
    pos = __shfl_sync(0xFFFFFFFFu, pos, 0);
    if (pos + 1 >= (TINY_OVERFLOW_CAP) - WARPS)
    {
      if (lane_id == 0) abort[0] = 1;
      __syncwarp();
      return false;
    }
    localTasks = global_tasks;
  }

  if (lane_id == 0)
  {
    TinyTask &nt = localTasks[pos];
    nt.parent_task_pos = source.parent_task_pos;
    nt.log_start = source.log_start;
    nt.log_len = source.log_len;
    nt.plex_sz = PlexSz;
    nt.cand_sz = CandSz;
    nt.excl_sz = ExclSz;
    atomicAdd(&global_count[0], 1);
  }

  return true;
}

__device__ bool reserve_checkpoint_task_slot(int lane_id, unsigned int* checkpoint_tail, unsigned int checkpoint_cap, int* abort, unsigned int& pos)
{
  if (lane_id == 0) pos = atomicAdd(&checkpoint_tail[0], 1u);
  pos = __shfl_sync(0xFFFFFFFFu, pos, 0);

  if (pos + 1 >= (checkpoint_cap) - WARPS)
  {
    if (lane_id == 0) atomicSub(&checkpoint_tail[0], 1u);
    __syncwarp();
    return false;
  }
  return true;
}

__device__ bool enqueue_checkpoint_task(int lane_id, int task_idx, unsigned int n, unsigned int* plex, unsigned int PlexSz, unsigned int* cand, unsigned int CandSz, unsigned int* excl, unsigned int ExclSz, uint16_t* neiInG, uint16_t* neiInP, Task* checkpoint_tasks, uint8_t* checkpoint_labels, uint16_t* checkpoint_neiInG, uint16_t* checkpoint_neiInP, unsigned int* checkpoint_tail, unsigned int checkpoint_cap, int* abort, unsigned int* global_count)
{
  unsigned int pos = 0;

  if (!reserve_checkpoint_task_slot(lane_id, checkpoint_tail, checkpoint_cap, abort, pos)) return false;

  uint8_t* childLabels = checkpoint_labels + (size_t)pos * MAX_BLK_SIZE;
  uint16_t* childNeiInG = checkpoint_neiInG + (size_t)pos * MAX_BLK_SIZE;
  uint16_t* childNeiInP = checkpoint_neiInP + (size_t)pos * MAX_BLK_SIZE;

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
    Task &nt = checkpoint_tasks[pos];
    nt.idx = task_idx;
    nt.PlexSz = PlexSz;
    nt.CandSz = CandSz;
    nt.ExclSz = ExclSz;
    nt.labels = childLabels;
    nt.neiInG = childNeiInG;
    nt.neiInP = childNeiInP;
    atomicAdd(&global_count[0], 1);
  }

  return true;
}

__device__ bool enqueue_checkpoint_exclude_task(int lane_id, int task_idx, unsigned int n, int pivot, unsigned int* plex, unsigned int PlexSz, unsigned int* cand, unsigned int CandSz, unsigned int* excl, unsigned int ExclSz, uint16_t* neiInG, uint16_t* neiInP, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeBase, Task* checkpoint_tasks, uint8_t* checkpoint_labels, uint16_t* checkpoint_neiInG, uint16_t* checkpoint_neiInP, unsigned int* checkpoint_tail, unsigned int checkpoint_cap, int* abort, unsigned int* global_count)
{
  unsigned int pos = 0;
  if (!reserve_checkpoint_task_slot(lane_id, checkpoint_tail, checkpoint_cap, abort, pos)) return false;

  uint8_t* childLabels = checkpoint_labels + (size_t)pos * MAX_BLK_SIZE;
  uint16_t* childNeiInG = checkpoint_neiInG + (size_t)pos * MAX_BLK_SIZE;
  uint16_t* childNeiInP = checkpoint_neiInP + (size_t)pos * MAX_BLK_SIZE;

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

  if (lane_id == 0) childLabels[pivot] = X;
  __syncwarp();

  subG(lane_id, pivot, childNeiInG, n, neighborsBase, offsetsBase, degreeBase);
  __syncwarp();

  if (lane_id == 0)
  {
    Task &nt = checkpoint_tasks[pos];
    nt.idx = task_idx;
    nt.PlexSz = PlexSz;
    nt.CandSz = CandSz - 1;
    nt.ExclSz = ExclSz + 1;
    nt.labels = childLabels;
    nt.neiInG = childNeiInG;
    nt.neiInP = childNeiInP;
    atomicAdd(&global_count[0], 1);
  }

  return true;
}

__device__ bool enqueue_tiny_child(int lane_id, const TinyTask& source, unsigned int pivot, unsigned int branch_decision, unsigned int PlexSz, unsigned int CandSz, unsigned int ExclSz, TinyTask* tasks, TinyTask* global_tasks, unsigned int* tailPtr, unsigned int* global_tail, BranchLog* Delta, unsigned int* delta_tail, int* abort, unsigned int* global_count)
{
  TinyTask child = source;
  unsigned int childLogStart = INVALID_LOG;
  unsigned int childLogLen = 0;
  if (!append_delta_decision(lane_id, source, pivot, branch_decision, Delta, delta_tail, abort, childLogStart, childLogLen)) return false;

  child.log_start = childLogStart;
  child.log_len = childLogLen;
  child.plex_sz = PlexSz;
  child.cand_sz = CandSz;
  child.excl_sz = ExclSz;

  return enqueue_tiny_record(lane_id, child, PlexSz, CandSz, ExclSz, tasks, global_tasks, tailPtr, global_tail, abort, global_count);
}

__device__ bool emit_exclude_and_continue_include(int lane_id, TinyTask& source, int task_idx, int pivot, int k, int lb, unsigned int n, TinyTask* tasks, TinyTask* global_tasks, unsigned int* tailPtr, unsigned int* global_tail, unsigned int* plex, unsigned int* cand, unsigned int* excl, uint16_t* neiInG, uint16_t* neiInP, unsigned int& PlexSz, unsigned int& CandSz, unsigned int& ExclSz, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeBase, uint8_t* commonMtx, uint32_t* adjList, BranchLog* Delta, unsigned int* delta_tail, Task* checkpoint_tasks, uint8_t* checkpoint_labels, uint16_t* checkpoint_neiInG, uint16_t* checkpoint_neiInP, unsigned int* checkpoint_tail, unsigned int checkpoint_cap, int* abort, unsigned int* global_count)
{
  const unsigned int childLogLen = source.log_len + 1u;
  if (childLogLen >= CHECKPOINT_LOG_LIMIT)
  {
    if (!enqueue_checkpoint_exclude_task(lane_id, task_idx, n, pivot, plex, PlexSz, cand, CandSz, excl, ExclSz, neiInG, neiInP, neighborsBase, offsetsBase, degreeBase, checkpoint_tasks, checkpoint_labels, checkpoint_neiInG, checkpoint_neiInP, checkpoint_tail, checkpoint_cap, abort, global_count) && !enqueue_tiny_child(lane_id, source, pivot, 1u, PlexSz, CandSz - 1, ExclSz + 1, tasks, global_tasks, tailPtr, global_tail, Delta, delta_tail, abort, global_count)) return false;
  }
  else if (!enqueue_tiny_child(lane_id, source, pivot, 1u, PlexSz, CandSz - 1, ExclSz + 1, tasks, global_tasks, tailPtr, global_tail, Delta, delta_tail, abort, global_count)) return false;

  if (abort[0]) return false;

  bool keep_include = apply_include_branch(lane_id, k, lb, n, neighborsBase, offsetsBase, degreeBase, commonMtx, plex, PlexSz, cand, CandSz, excl, ExclSz, neiInP, neiInG, pivot, adjList);
  if (!keep_include) return false;

  if (childLogLen >= CHECKPOINT_LOG_LIMIT)
  {
    if (enqueue_checkpoint_task(lane_id, task_idx, n, plex, PlexSz, cand, CandSz, excl, ExclSz, neiInG, neiInP, checkpoint_tasks, checkpoint_labels, checkpoint_neiInG, checkpoint_neiInP, checkpoint_tail, checkpoint_cap, abort, global_count)) return false;
    if (!enqueue_tiny_child(lane_id, source, pivot, 0u, PlexSz, CandSz, ExclSz, tasks, global_tasks, tailPtr, global_tail, Delta, delta_tail, abort, global_count)) return false;
    return false;
  }

  unsigned int includeLogStart = INVALID_LOG;
  unsigned int includeLogLen = 0;
  if (!append_delta_decision(lane_id, source, pivot, 0u, Delta, delta_tail, abort, includeLogStart, includeLogLen)) return false;

  source.log_start = includeLogStart;
  source.log_len = includeLogLen;
  source.plex_sz = PlexSz;
  source.cand_sz = CandSz;
  source.excl_sz = ExclSz;
  return true;
}

// __device__ void branchInCand2(int warp_id, int lane_id, int minIndex, int idx, int k, int lb, unsigned int PlexSz, unsigned int CandSz, unsigned int ExclSz, unsigned int* local_n, Task* tasks, Task* global_tasks, unsigned int* tailPtr, unsigned int* global_tail, unsigned int* plex, unsigned int* cand, unsigned int* excl, uint16_t* neiInG, uint16_t* neiInP, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeBase,uint8_t* d_all_labels, uint16_t* d_all_neiInG, uint16_t* d_all_neiInP, uint8_t* global_labels, uint16_t* global_neiInG, uint16_t* global_neiInP, uint8_t* commonMtx, unsigned long long* cycles, uint32_t* adjList, uint16_t* local_sat, int* abort, unsigned int* global_count)
// {
//   enqueue_exclude_child(lane_id, idx, local_n, plex, PlexSz, cand, CandSz, excl, ExclSz, minIndex, tasks, global_tasks, tailPtr, global_tail, d_all_labels, d_all_neiInG, d_all_neiInP, global_labels, global_neiInG, global_neiInP, neiInG, neiInP, neighborsBase, offsetsBase, degreeBase, abort, global_count);  
//   if (abort[0]) return;
//   enqueue_include_child(lane_id, idx, k, lb, local_n, neighborsBase, offsetsBase, degreeBase, commonMtx, plex, PlexSz, cand, CandSz, excl, ExclSz, neiInP, neiInG, minIndex, tasks, global_tasks, tailPtr, global_tail, d_all_labels, d_all_neiInG, d_all_neiInP, global_labels, global_neiInG, global_neiInP, cycles, adjList, local_sat, abort, global_count);
// }

__device__ void branchInCand2(int lane_id, const TinyTask& source, int minIndex, int k, int lb, unsigned int n, TinyTask* tasks, TinyTask* global_tasks, unsigned int* tailPtr, unsigned int* global_tail, unsigned int* plex, unsigned int* cand, unsigned int* excl, uint16_t* neiInG, uint16_t* neiInP, unsigned int PlexSz, unsigned int CandSz, unsigned int ExclSz, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeBase, uint8_t* commonMtx, uint32_t* adjList, BranchLog* Delta, unsigned int* delta_tail, int* abort, unsigned int* global_count)
{
  enqueue_tiny_child(lane_id, source, minIndex, 1u, PlexSz, CandSz - 1, ExclSz + 1, tasks, global_tasks, tailPtr, global_tail, Delta, delta_tail, abort, global_count);
  if (abort[0]) return;

  bool keep_include = apply_include_branch(lane_id, k, lb, n, neighborsBase, offsetsBase, degreeBase, commonMtx, plex, PlexSz, cand, CandSz, excl, ExclSz, neiInP, neiInG, minIndex, adjList);
  if (keep_include)
  {
    enqueue_tiny_child(lane_id, source, minIndex, 0u, PlexSz, CandSz, ExclSz, tasks, global_tasks, tailPtr, global_tail, Delta, delta_tail, abort, global_count);
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


__global__ void seedInitialTinyTasks(Task* parent_tasks, TinyTask* tiny_tasks, unsigned int parent_start, unsigned int N)
{
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N) return;

  Task parent = parent_tasks[parent_start + tid];
  TinyTask &tiny = tiny_tasks[tid];
  tiny.parent_task_pos = parent_start + tid;
  tiny.log_start = INVALID_LOG;
  tiny.log_len = 0;
  tiny.plex_sz = parent.PlexSz;
  tiny.cand_sz = parent.CandSz;
  tiny.excl_sz = parent.ExclSz;
}

// original BNB strategy
// __global__ void BNB(int i, P_pointers p, S_pointers s, unsigned int* d_blk, unsigned int* d_left, unsigned int* d_blk_counter, unsigned int* d_left_counter, uint8_t* commonMtx, Task* tasks, Task* outTasks, Task* global_tasks, unsigned int N, unsigned int head, unsigned int* tailPtr, unsigned int* global_tail, uint8_t* d_all_labels, uint16_t* d_all_neiInG, uint16_t* d_all_neiInP, uint8_t* global_labels, uint16_t* global_neiInG, uint16_t* global_neiInP, unsigned int* plex_count, uint16_t* d_sat, uint16_t* d_commons, uint32_t* d_uni, unsigned long long* cycles, uint32_t* d_adj, int* abort, unsigned int *global_count)
// {
  
//   unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
//   unsigned int warp_id = (global_index / 32);
//   unsigned int lane_id = threadIdx.x % 32;

//   if ((warp_id+WARPS*i) >= N)  return;
//   int k = p.k;
//   int q = p.lb;
//   Task t = tasks[warp_id+WARPS*i];
//   uint8_t* labelsBase = t.labels;

//   // if (t.idx != 2878) return;
//   // if (lane_id == 0)
//   // {
//   //   // printf("idx: %d, warp_id: %d\n", t.idx, warp_id);
//   //   // if (t.idx != 0) return;
//   //   printf("Labels: %d\n", labelsBase[0]);
//   //   // for (int j = 0; j < MAX_BLK_SIZE; j++)
//   //   // {
//   //   //   printf("%d ", labelsBase[j]);
//   //   // }
//   //   // printf("\n");
//   // }

//   unsigned int* leftBase = d_left + t.idx * MAX_BLK_SIZE;
//   uint16_t* neiInG = t.neiInG;
//   uint16_t* neiInP = t.neiInP;
//   unsigned int* left_count = d_left_counter + t.idx;
//   unsigned int* local_n = d_blk_counter + t.idx;
//   unsigned int PlexSz = t.PlexSz;
//   unsigned int CandSz = t.CandSz;
//   unsigned int ExclSz = t.ExclSz;
//   size_t capacity = size_t(t.idx) * CAP;
//   uint8_t* commonMtxBase = commonMtx + capacity;
//   // uint8_t* commonMtxBase = commonMtx + warp_id * CAP;

//   unsigned int* degreeBase = s.degree + t.idx * MAX_BLK_SIZE;
//   unsigned int* l_degreeBase = s.l_degree + t.idx * MAX_BLK_SIZE;
//   unsigned int* offsetsBase = s.offsets + t.idx * MAX_BLK_SIZE;
//   unsigned int* neighborsBase = s.neighbors + t.idx * MAX_BLK_SIZE * AVG_DEGREE;
//   unsigned int* l_offsetsBase = s.l_offsets + t.idx * MAX_BLK_SIZE;
//   unsigned int* l_neighborsBase = s.l_neighbors + t.idx * MAX_BLK_SIZE * AVG_LEFT_DEGREE;

//   unsigned int* plex = s.PB + warp_id * MAX_BLK_SIZE;
//   unsigned int* cand = s.CB + warp_id * MAX_BLK_SIZE;
//   unsigned int* excl = s.XB + warp_id * MAX_BLK_SIZE;

//   uint16_t* local_sat = d_sat + warp_id * MAX_BLK_SIZE;
//   uint16_t* local_commons = d_commons + warp_id * MAX_BLK_SIZE;
//   uint32_t* local_uni = d_uni + warp_id * 32;

//   uint32_t* adjList = d_adj + t.idx * ADJSIZE;

//   unsigned int n;
//   if (lane_id == 0) n = local_n[0];
//   n = __shfl_sync(0xFFFFFFFF, n, 0);
  
//   initializePCX(lane_id, labelsBase, n, plex, cand, excl);

//   if (PlexSz + CandSz < q) return;
    
//   if (CandSz == 0)
//   {
//     __syncwarp();
//     if (ExclSz == 0 && PlexSz >= q &&
//         isMaximal_opt(lane_id, k, PlexSz, leftBase, left_count[0], l_neighborsBase, l_offsetsBase, l_degreeBase, neiInP, neighborsBase, offsetsBase, degreeBase, plex, n, local_sat, local_commons, local_uni))
//     {
//       if(lane_id == 0) atomicAdd(&plex_count[0], 1);
//     }
//     return;
//   }
//   __syncwarp();
   
    
//   int minnei_Plex = INT_MAX;
//   int pivot = -1;
//   int minnei_Cand = INT_MAX;

//   for (int i = lane_id; i < PlexSz; i+=32)
//   {
//     const int v = plex[i];
//     if (neiInG[v] < minnei_Plex)
//     {
//       minnei_Plex = neiInG[v];
//       pivot = v;
//     }
//   }

    

//   for (int offset = 16; offset > 0; offset >>= 1)
//   {
//     int otherMin = __shfl_down_sync(0xFFFFFFFF, minnei_Plex, offset);
//     int otherIdx = __shfl_down_sync(0xFFFFFFFF, pivot, offset);
//     if (otherMin < minnei_Plex || (otherMin == minnei_Plex && otherIdx < pivot))
//     {
//       minnei_Plex = otherMin;
//       pivot = otherIdx;
//     }
//   }

//   minnei_Plex = __shfl_sync(0xFFFFFFFF, minnei_Plex, 0);
//   pivot = __shfl_sync(0xFFFFFFFF, pivot, 0);

//   int pivot_plex = pivot;
    
//   if (minnei_Plex + k  < max(q, PlexSz)) return;
    
//   if (minnei_Plex + k < PlexSz + CandSz)
//   {     
//     minnei_Cand = INT_MAX;
//     pivot = -1;
//     for (int i = lane_id; i < CandSz; i+=32)
//     {
//       const int v = cand[i];
//       int check = v * n + pivot_plex;
//       if (!((adjList[check >> 5] >> (check & 31)) & 1u))
//       {
//         if (neiInG[v] < minnei_Cand)
//         {
//           minnei_Cand = neiInG[v];
//           pivot = v;
//         }
//         else if (neiInG[v] == minnei_Cand && neiInP[pivot] > neiInP[v])
//         {
//           pivot = v;
//         }
//       }
//     }
        
//     for (int offset = 16; offset > 0; offset >>= 1)
//     {
//       int otherMin = __shfl_down_sync(0xFFFFFFFF, minnei_Cand, offset);
//       int otherIdx = __shfl_down_sync(0xFFFFFFFF, pivot, offset);
//       if (otherMin < minnei_Cand || (otherMin == minnei_Cand && otherIdx != -1 && neiInP[pivot] > neiInP[otherIdx]))
//       {
//         minnei_Cand = otherMin;
//         pivot = otherIdx;
//       }
//     }

//     minnei_Cand = __shfl_sync(0xFFFFFFFF, minnei_Cand, 0);
//     pivot = __shfl_sync(0xFFFFFFFF, pivot, 0);
         
//     // if (lane_id == 0) printf("Pivot: %d\n", pivot);
//     branchInCand2(warp_id, lane_id, pivot, t.idx, k, q, PlexSz, CandSz, ExclSz, &n, outTasks, global_tasks, tailPtr, global_tail, plex, cand, excl, neiInG, neiInP, neighborsBase, offsetsBase, degreeBase, d_all_labels, d_all_neiInG, d_all_neiInP, global_labels, global_neiInG, global_neiInP, commonMtxBase, cycles, adjList, local_sat, abort, global_count);
//     return;
//   }

   
   
//   int minnei = minnei_Plex;

//   for (int i = lane_id; i < CandSz; i+=32)
//   {
//     const int v = cand[i];
//     if (neiInG[v] < minnei)
//     {
//       minnei = neiInG[v];
//       pivot = v;
//     }
//     else if (neiInG[v] == minnei && neiInP[pivot] > neiInP[v])
//     {
//       pivot = v;
//     }
//   }

//   for(int offset = 16; offset > 0; offset >>= 1)
//   {
//     int otherMin = __shfl_down_sync(0xFFFFFFFF, minnei, offset);
//     int otherIdx = __shfl_down_sync(0xFFFFFFFF, pivot, offset);

//     if (otherMin < minnei || (otherMin == minnei && otherIdx != -1 && neiInP[pivot] > neiInP[otherIdx]))
//     {
//       minnei = otherMin;
//       pivot = otherIdx;
//     }
//   }
//   minnei = __shfl_sync(0xFFFFFFFF, minnei, 0);
//   pivot = __shfl_sync(0xFFFFFFFF, pivot, 0);
//   if (minnei >= (PlexSz + CandSz - k))
//   {
//     if (PlexSz + CandSz < q) return;
//     bool flag = false;
        
//     for (int i = lane_id; i < ExclSz; i+=32)
//     {
//       const int v = excl[i];
//       if (isKplexPC2(v, k, PlexSz+CandSz, PlexSz, CandSz, neiInG, plex, cand, n, neighborsBase, offsetsBase, degreeBase, adjList))
//       {
//         flag = true;
//       }
//     }
        
//     if (__any_sync(0xFFFFFFFF, flag)) return;
         
//     if (isMaximalPC_opt(lane_id, k, PlexSz, CandSz, PlexSz+CandSz, leftBase, left_count[0], l_neighborsBase, l_offsetsBase, l_degreeBase, neiInG, neighborsBase, offsetsBase, degreeBase, plex, cand, n, local_sat, local_commons, local_uni))
//     {
//       if (lane_id == 0) atomicAdd(&plex_count[0], 1);
//     }
        
//     return;
//   }

//   branchInCand2(warp_id, lane_id, pivot, t.idx, k, q, PlexSz, CandSz, ExclSz, &n, outTasks, global_tasks, tailPtr, global_tail, plex, cand, excl, neiInG, neiInP, neighborsBase, offsetsBase, degreeBase, d_all_labels, d_all_neiInG, d_all_neiInP, global_labels, global_neiInG, global_neiInP, commonMtxBase, cycles, adjList, local_sat, abort, global_count);
//   __syncwarp();
// }

// BNB with TinyTask strategy
__global__ void BNB(int i, P_pointers p, S_pointers s, unsigned int* d_blk, unsigned int* d_left, unsigned int* d_blk_counter, unsigned int* d_left_counter, uint8_t* commonMtx, Task* parent_tasks, TinyTask* tasks, TinyTask* outTasks, TinyTask* global_tasks, unsigned int N, unsigned int* tailPtr, unsigned int* global_tail, BranchLog* Delta, unsigned int* delta_tail, unsigned int* replay_stack, Task* checkpoint_tasks, uint8_t* checkpoint_labels, uint16_t* checkpoint_neiInG, uint16_t* checkpoint_neiInP, unsigned int* checkpoint_tail, unsigned int checkpoint_cap, unsigned int* plex_count, uint16_t* d_bnb_neiInG, uint16_t* d_bnb_neiInP, uint16_t* d_sat, uint16_t* d_commons, uint32_t* d_uni, unsigned long long* cycles, uint32_t* d_adj, int* abort, unsigned int* global_count)
{
  
  unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int warp_id = (global_index / 32);
  unsigned int lane_id = threadIdx.x % 32;

  if ((warp_id+WARPS*i) >= N)  return;
  int k = p.k;
  int q = p.lb;
  TinyTask tiny = tasks[warp_id+WARPS*i];
  Task t = parent_tasks[tiny.parent_task_pos];
  uint8_t* labelsBase = t.labels;

  // if (t.idx != 2878) return;
  // if (lane_id == 0)
  // {
  //   // printf("idx: %d, warp_id: %d\n", t.idx, warp_id);
  //   // if (t.idx != 0) return;
  //   printf("Labels: %d\n", labelsBase[0]);
  //   // for (int j = 0; j < MAX_BLK_SIZE; j++)
  //   // {
  //   //   printf("%d ", labelsBase[j]);
  //   // }
  //   // printf("\n");
  // }

  unsigned int* leftBase = d_left + t.idx * MAX_BLK_SIZE;
  uint16_t* neiInG = d_bnb_neiInG + warp_id * MAX_BLK_SIZE;
  uint16_t* neiInP = d_bnb_neiInP + warp_id * MAX_BLK_SIZE;
  unsigned int* left_count = d_left_counter + t.idx;
  unsigned int* local_n = d_blk_counter + t.idx;
  unsigned int PlexSz = t.PlexSz;
  unsigned int CandSz = t.CandSz;
  unsigned int ExclSz = t.ExclSz;
  size_t capacity = size_t(t.idx) * CAP;
  uint8_t* commonMtxBase = commonMtx + capacity;
  // uint8_t* commonMtxBase = commonMtx + warp_id * CAP;

  unsigned int* degreeBase = s.degree + t.idx * MAX_BLK_SIZE;
  unsigned int* l_degreeBase = s.l_degree + t.idx * MAX_BLK_SIZE;
  unsigned int* offsetsBase = s.offsets + t.idx * MAX_BLK_SIZE;
  unsigned int* neighborsBase = s.neighbors + t.idx * MAX_BLK_SIZE * AVG_DEGREE;
  unsigned int* l_offsetsBase = s.l_offsets + t.idx * MAX_BLK_SIZE;
  unsigned int* l_neighborsBase = s.l_neighbors + t.idx * MAX_BLK_SIZE * AVG_LEFT_DEGREE;

  unsigned int* plex = s.PB + warp_id * MAX_BLK_SIZE;
  unsigned int* cand = s.CB + warp_id * MAX_BLK_SIZE;
  unsigned int* excl = s.XB + warp_id * MAX_BLK_SIZE;

  uint16_t* local_sat = d_sat + warp_id * MAX_BLK_SIZE;
  uint16_t* local_commons = d_commons + warp_id * MAX_BLK_SIZE;
  uint32_t* local_uni = d_uni + warp_id * 32;

  uint32_t* adjList = d_adj + t.idx * ADJSIZE;

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

  if (tiny.log_len > REPLAY_STACK_CAP)
  {
    if (lane_id == 0) abort[0] = 3;
    __syncwarp();
    return;
  }

  unsigned int* replay = replay_stack + warp_id * REPLAY_STACK_CAP;

  if (lane_id == 0)
  {
    unsigned int entry_pos = tiny.log_start;
    for (unsigned int log_pos = 0; log_pos < tiny.log_len; log_pos++)
    {
      if (entry_pos == INVALID_LOG || entry_pos >= DELTA_CAP)
      {
        abort[0] = 4;
        break;
      }
      replay[tiny.log_len - 1u - log_pos] = entry_pos;
      entry_pos = Delta[entry_pos].prev;
    }
  }
  __syncwarp();
  if (abort[0]) return;

  for (unsigned int log_pos = 0; log_pos < tiny.log_len; log_pos++)
  {
    unsigned int replay_pivot = 0;
    unsigned int replay_decision = 0;
    if (lane_id == 0)
    {
      BranchLog entry = Delta[replay[log_pos]];
      replay_pivot = entry.pivot;
      replay_decision = entry.branch_decision;
    }
    replay_pivot = __shfl_sync(0xFFFFFFFFu, replay_pivot, 0);
    replay_decision = __shfl_sync(0xFFFFFFFFu, replay_decision, 0);
    if (replay_decision == 1u)
    {
      apply_exclude_branch(lane_id, n, cand, CandSz, excl, ExclSz, neiInG, replay_pivot, neighborsBase, offsetsBase, degreeBase);
    }
    else
    {
      bool keep = apply_include_branch(lane_id, k, q, n, neighborsBase, offsetsBase, degreeBase, commonMtxBase, plex, PlexSz, cand, CandSz, excl, ExclSz, neiInP, neiInG, replay_pivot, adjList);
      if (!keep) return;
    }
  }

  TinyTask current = tiny;
  current.plex_sz = PlexSz;
  current.cand_sz = CandSz;
  current.excl_sz = ExclSz;

  for (int local_step = 0; local_step < LOCAL_TINY_BNB_STEPS; local_step++)
  {
    if (PlexSz + CandSz < q) return;
      
    if (CandSz == 0)
    {
      __syncwarp();
      if (ExclSz == 0 && PlexSz >= q &&
          isMaximal_opt(lane_id, k, PlexSz, leftBase, left_count[0], l_neighborsBase, l_offsetsBase, l_degreeBase, neiInP, neighborsBase, offsetsBase, degreeBase, plex, n, local_sat, local_commons, local_uni))
      {
        if(lane_id == 0) atomicAdd(&plex_count[0], 1);
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
        int check = v * n + pivot_plex;
        if (!((adjList[check >> 5] >> (check & 31)) & 1u))
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
          
      // if (lane_id == 0) printf("Pivot: %d\n", pivot);
      // branchInCand2(lane_id, tiny, pivot, k, q, n,
      //           outTasks, global_tasks, tailPtr, global_tail,
      //           plex, cand, excl, neiInG, neiInP,
      //           PlexSz, CandSz, ExclSz,
      //           neighborsBase, offsetsBase, degreeBase,
      //           commonMtxBase, adjList,
      //           Delta, delta_tail, abort, global_count);
      if (!emit_exclude_and_continue_include(lane_id, current, t.idx, pivot, k, q, n,
                outTasks, global_tasks, tailPtr, global_tail,
                plex, cand, excl, neiInG, neiInP,
                PlexSz, CandSz, ExclSz,
                neighborsBase, offsetsBase, degreeBase,
                commonMtxBase, adjList,
                Delta, delta_tail,
                checkpoint_tasks, checkpoint_labels, checkpoint_neiInG, checkpoint_neiInP, checkpoint_tail, checkpoint_cap,
                abort, global_count)) return;
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
          
      if (isMaximalPC_opt(lane_id, k, PlexSz, CandSz, PlexSz+CandSz, leftBase, left_count[0], l_neighborsBase, l_offsetsBase, l_degreeBase, neiInG, neighborsBase, offsetsBase, degreeBase, plex, cand, n, local_sat, local_commons, local_uni))
      {
        if (lane_id == 0) atomicAdd(&plex_count[0], 1);
      }
          
      return;
    }

    // branchInCand2(lane_id, tiny, pivot, k, q, n,
    //             outTasks, global_tasks, tailPtr, global_tail,
    //             plex, cand, excl, neiInG, neiInP,
    //             PlexSz, CandSz, ExclSz,
    //             neighborsBase, offsetsBase, degreeBase,
    //             commonMtxBase, adjList,
    //             Delta, delta_tail, abort, global_count);
    if (!emit_exclude_and_continue_include(lane_id, current, t.idx, pivot, k, q, n,
              outTasks, global_tasks, tailPtr, global_tail,
              plex, cand, excl, neiInG, neiInP,
              PlexSz, CandSz, ExclSz,
              neighborsBase, offsetsBase, degreeBase,
              commonMtxBase, adjList,
              Delta, delta_tail,
              checkpoint_tasks, checkpoint_labels, checkpoint_neiInG, checkpoint_neiInP, checkpoint_tail, checkpoint_cap,
              abort, global_count)) return;
  }

  if (PlexSz + CandSz < q) return;
  current.plex_sz = PlexSz;
  current.cand_sz = CandSz;
  current.excl_sz = ExclSz;
  if (current.log_len >= CHECKPOINT_LOG_LIMIT)
  {
    if (!enqueue_checkpoint_task(lane_id, t.idx, n, plex, PlexSz, cand, CandSz, excl, ExclSz, neiInG, neiInP, checkpoint_tasks, checkpoint_labels, checkpoint_neiInG, checkpoint_neiInP, checkpoint_tail, checkpoint_cap, abort, global_count) && !enqueue_tiny_record(lane_id, current, PlexSz, CandSz, ExclSz, outTasks, global_tasks, tailPtr, global_tail, abort, global_count)) return;
  }
  else if (!enqueue_tiny_record(lane_id, current, PlexSz, CandSz, ExclSz, outTasks, global_tasks, tailPtr, global_tail, abort, global_count)) return;
  __syncwarp();
}

// BNB with pivot logic: minimum neiInG/neiInP among cand
// __global__ void BNB(
//     int i,
//     P_pointers p,
//     S_pointers s,
//     unsigned int* d_blk,
//     unsigned int* d_left,
//     unsigned int* d_blk_counter,
//     unsigned int* d_left_counter,
//     uint8_t* commonMtx,
//     Task* tasks,
//     Task* outTasks,
//     Task* global_tasks,
//     unsigned int N,
//     unsigned int head,
//     unsigned int* tailPtr,
//     unsigned int* global_tail,
//     uint8_t* d_all_labels,
//     uint16_t* d_all_neiInG,
//     uint16_t* d_all_neiInP,
//     uint8_t* global_labels,
//     uint16_t* global_neiInG,
//     uint16_t* global_neiInP,
//     unsigned int* plex_count,
//     uint16_t* d_sat,
//     uint16_t* d_commons,
//     uint32_t* d_uni,
//     unsigned long long* cycles,
//     uint32_t* d_adj,
//     int* abort,
//     unsigned int* global_count)
// {
//     unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
//     unsigned int warp_id = global_index / 32;
//     unsigned int lane_id = threadIdx.x % 32;

//     if ((warp_id + WARPS * i) >= N) return;

//     int k = p.k;
//     int q = p.lb;

//     Task t = tasks[warp_id + WARPS * i];
//     uint8_t* labelsBase = t.labels;

//     unsigned int* leftBase = d_left + t.idx * MAX_BLK_SIZE;
//     uint16_t* neiInG = t.neiInG;
//     uint16_t* neiInP = t.neiInP;

//     unsigned int* left_count = d_left_counter + t.idx;
//     unsigned int* local_n = d_blk_counter + t.idx;

//     unsigned int PlexSz = t.PlexSz;
//     unsigned int CandSz = t.CandSz;
//     unsigned int ExclSz = t.ExclSz;

//     size_t capacity = size_t(t.idx) * CAP;
//     uint8_t* commonMtxBase = commonMtx + capacity;

//     unsigned int* degreeBase = s.degree + t.idx * MAX_BLK_SIZE;
//     unsigned int* l_degreeBase = s.l_degree + t.idx * MAX_BLK_SIZE;

//     unsigned int* offsetsBase = s.offsets + t.idx * MAX_BLK_SIZE;
//     unsigned int* neighborsBase =
//         s.neighbors + t.idx * MAX_BLK_SIZE * AVG_DEGREE;

//     unsigned int* l_offsetsBase = s.l_offsets + t.idx * MAX_BLK_SIZE;
//     unsigned int* l_neighborsBase =
//         s.l_neighbors + t.idx * MAX_BLK_SIZE * AVG_LEFT_DEGREE;

//     unsigned int* plex = s.PB + warp_id * MAX_BLK_SIZE;
//     unsigned int* cand = s.CB + warp_id * MAX_BLK_SIZE;
//     unsigned int* excl = s.XB + warp_id * MAX_BLK_SIZE;

//     uint16_t* local_sat = d_sat + warp_id * MAX_BLK_SIZE;
//     uint16_t* local_commons = d_commons + warp_id * MAX_BLK_SIZE;
//     uint32_t* local_uni = d_uni + warp_id * 32;

//     uint32_t* adjList = d_adj + t.idx * ADJSIZE;

//     unsigned int n;
//     if (lane_id == 0) n = local_n[0];
//     n = __shfl_sync(0xFFFFFFFF, n, 0);

//     initializePCX(lane_id, labelsBase, n, plex, cand, excl);

//     if (PlexSz + CandSz < q) return;

//     if (CandSz == 0)
//     {
//         __syncwarp();

//         if (ExclSz == 0 &&
//             PlexSz >= q &&
//             isMaximal_opt(
//                 lane_id,
//                 k,
//                 PlexSz,
//                 leftBase,
//                 left_count[0],
//                 l_neighborsBase,
//                 l_offsetsBase,
//                 l_degreeBase,
//                 neiInP,
//                 neighborsBase,
//                 offsetsBase,
//                 degreeBase,
//                 plex,
//                 n,
//                 local_sat,
//                 local_commons,
//                 local_uni))
//         {
//             if (lane_id == 0)
//             {
//                 atomicAdd(&plex_count[0], 1);
//             }
//         }

//         return;
//     }

//     __syncwarp();

//     /*
//      * Keep the original P-based degree upper-bound pruning.
//      *
//      * minnei_Plex + k is an upper bound on the size of any k-plex
//      * containing the current plex P.
//      */
//     int minnei_Plex = INT_MAX;
//     int minnei_Plex_v = -1;

//     for (int j = lane_id; j < PlexSz; j += 32)
//     {
//         const int v = plex[j];

//         if (neiInG[v] < minnei_Plex ||
//             (neiInG[v] == minnei_Plex &&
//              (minnei_Plex_v == -1 || v < minnei_Plex_v)))
//         {
//             minnei_Plex = neiInG[v];
//             minnei_Plex_v = v;
//         }
//     }

//     for (int offset = 16; offset > 0; offset >>= 1)
//     {
//         int otherMin = __shfl_down_sync(0xFFFFFFFF, minnei_Plex, offset);
//         int otherIdx = __shfl_down_sync(0xFFFFFFFF, minnei_Plex_v, offset);

//         if (otherMin < minnei_Plex ||
//             (otherMin == minnei_Plex &&
//              otherIdx != -1 &&
//              (minnei_Plex_v == -1 || otherIdx < minnei_Plex_v)))
//         {
//             minnei_Plex = otherMin;
//             minnei_Plex_v = otherIdx;
//         }
//     }

//     minnei_Plex = __shfl_sync(0xFFFFFFFF, minnei_Plex, 0);
//     minnei_Plex_v = __shfl_sync(0xFFFFFFFF, minnei_Plex_v, 0);

//     if (minnei_Plex + k < max(q, (int)PlexSz)) return;

//     /*
//      * New pivot logic:
//      *
//      * Always choose the branching pivot from C,
//      * with minimum degree in P union C, represented here by neiInG[v].
//      *
//      * Tie-break: smaller vertex id, just for deterministic behavior.
//      */
//     // int minnei_Cand = INT_MAX;
//     // int pivot = -1;

//     // for (int j = lane_id; j < CandSz; j += 32)
//     // {
//     //     const int v = cand[j];

//     //     if (neiInG[v] < minnei_Cand ||
//     //         (neiInG[v] == minnei_Cand &&
//     //          (pivot == -1 || v < pivot)))
//     //     {
//     //         minnei_Cand = neiInG[v];
//     //         pivot = v;
//     //     }
//     // }

//     // for (int offset = 16; offset > 0; offset >>= 1)
//     // {
//     //     int otherMin = __shfl_down_sync(0xFFFFFFFF, minnei_Cand, offset);
//     //     int otherIdx = __shfl_down_sync(0xFFFFFFFF, pivot, offset);

//     //     if (otherMin < minnei_Cand ||
//     //         (otherMin == minnei_Cand &&
//     //          otherIdx != -1 &&
//     //          (pivot == -1 || otherIdx < pivot)))
//     //     {
//     //         minnei_Cand = otherMin;
//     //         pivot = otherIdx;
//     //     }
//     // }

//     // minnei_Cand = __shfl_sync(0xFFFFFFFF, minnei_Cand, 0);
//     // pivot = __shfl_sync(0xFFFFFFFF, pivot, 0);

//     int minNeiInP_Cand = INT_MAX;
//     int minnei_Cand = INT_MAX;
//     int pivot = -1;

//     int minnei_Cand_All = INT_MAX;

//     for (int j = lane_id; j < CandSz; j += 32)
//     {
//         const int v = cand[j];
//         const int candNeiInP = neiInP[v];
//         const int candNeiInG = neiInG[v];

//         if (candNeiInG < minnei_Cand_All)
//         {
//           minnei_Cand_All = candNeiInG;
//         }

//         if (candNeiInP < minNeiInP_Cand ||
//             (candNeiInP == minNeiInP_Cand &&
//              candNeiInG < minnei_Cand) ||
//             (candNeiInP == minNeiInP_Cand &&
//              candNeiInG == minnei_Cand &&
//              (pivot == -1 || v < pivot)))
//         {
//             minNeiInP_Cand = candNeiInP;
//             minnei_Cand = candNeiInG;
//             pivot = v;
//         }
//     }

//     for (int offset = 16; offset > 0; offset >>= 1)
//     {
//         int otherNeiInP = __shfl_down_sync(0xFFFFFFFF, minNeiInP_Cand, offset);
//         int otherNeiInG = __shfl_down_sync(0xFFFFFFFF, minnei_Cand, offset);
//         int otherIdx = __shfl_down_sync(0xFFFFFFFF, pivot, offset);

//         int otherMinneiCandAll = __shfl_down_sync(0xFFFFFFFF, minnei_Cand_All, offset);

//         if (otherMinneiCandAll < minnei_Cand_All)
//         {
//           minnei_Cand_All = otherMinneiCandAll;
//         }

//         if (otherNeiInP < minNeiInP_Cand ||
//             (otherNeiInP == minNeiInP_Cand &&
//              otherNeiInG < minnei_Cand) ||
//             (otherNeiInP == minNeiInP_Cand &&
//              otherNeiInG == minnei_Cand &&
//              otherIdx != -1 &&
//              (pivot == -1 || otherIdx < pivot)))
//         {
//             minNeiInP_Cand = otherNeiInP;
//             minnei_Cand = otherNeiInG;
//             pivot = otherIdx;
//         }
//     }

//     minNeiInP_Cand = __shfl_sync(0xFFFFFFFF, minNeiInP_Cand, 0);
//     minnei_Cand = __shfl_sync(0xFFFFFFFF, minnei_Cand, 0);
//     pivot = __shfl_sync(0xFFFFFFFF, pivot, 0);
//     minnei_Cand_All = __shfl_sync(0xFFFFFFFF, minnei_Cand_All, 0);

//     if (pivot == -1) return;

//     /*
//      * Compute min degree over P union C for the take-all check.
//      *
//      * P union C is a k-plex iff every vertex in P union C has degree
//      * at least |P| + |C| - k.
//      */
//     int minnei_All = minnei_Plex;

//     if (minnei_Cand_All < minnei_All)
//     {
//         minnei_All = minnei_Cand_All;
//     }

//     minnei_All = __shfl_sync(0xFFFFFFFF, minnei_All, 0);

//     if (minnei_All >= ((int)PlexSz + (int)CandSz - k))
//     {
//         if (PlexSz + CandSz < q) return;

//         bool flag = false;

//         for (int j = lane_id; j < ExclSz; j += 32)
//         {
//             const int v = excl[j];

//             if (isKplexPC2(
//                     v,
//                     k,
//                     PlexSz + CandSz,
//                     PlexSz,
//                     CandSz,
//                     neiInG,
//                     plex,
//                     cand,
//                     n,
//                     neighborsBase,
//                     offsetsBase,
//                     degreeBase,
//                     adjList))
//             {
//                 flag = true;
//             }
//         }

//         if (__any_sync(0xFFFFFFFF, flag)) return;

//         if (isMaximalPC_opt(
//                 lane_id,
//                 k,
//                 PlexSz,
//                 CandSz,
//                 PlexSz + CandSz,
//                 leftBase,
//                 left_count[0],
//                 l_neighborsBase,
//                 l_offsetsBase,
//                 l_degreeBase,
//                 neiInG,
//                 neighborsBase,
//                 offsetsBase,
//                 degreeBase,
//                 plex,
//                 cand,
//                 n,
//                 local_sat,
//                 local_commons,
//                 local_uni))
//         {
//             if (lane_id == 0)
//             {
//                 atomicAdd(&plex_count[0], 1);
//             }
//         }

//         return;
//     }

//     /*
//      * Branch on the candidate pivot chosen by:
//      *
//      *   argmin_{v in C} neiInG[v]
//      */
//     branchInCand2(
//         warp_id,
//         lane_id,
//         pivot,
//         t.idx,
//         k,
//         q,
//         PlexSz,
//         CandSz,
//         ExclSz,
//         &n,
//         outTasks,
//         global_tasks,
//         tailPtr,
//         global_tail,
//         plex,
//         cand,
//         excl,
//         neiInG,
//         neiInP,
//         neighborsBase,
//         offsetsBase,
//         degreeBase,
//         d_all_labels,
//         d_all_neiInG,
//         d_all_neiInP,
//         global_labels,
//         global_neiInG,
//         global_neiInP,
//         commonMtxBase,
//         cycles,
//         adjList,
//         local_sat,
//         abort,
//         global_count);

//     __syncwarp();
// }

//MurmurHash3 64-bit hashing.
// __device__ __forceinline__
// uint64_t mix64_debug(uint64_t x)
// {
//     x ^= x >> 33;
//     x *= 0xff51afd7ed558ccdULL;
//     x ^= x >> 33;
//     x *= 0xc4ceb9fe1a85ec53ULL;
//     x ^= x >> 33;
//     return x;
// }

// __device__ __forceinline__
// uint64_t hash_state_debug(uint8_t* labels,
//                           uint16_t* neiInP,
//                           uint16_t* neiInG,
//                           unsigned int n)
// {
//     const unsigned int lane_id = threadIdx.x & 31;

//     uint64_t h = 1469598103934665603ULL;

//     for (unsigned int i = lane_id; i < n; i += 32) {
//         uint64_t x = 0;

//         x ^= ((uint64_t)labels[i] & 0xffULL);
//         x ^= (((uint64_t)neiInP[i]) << 8);
//         x ^= (((uint64_t)neiInG[i]) << 24);
//         x ^= (((uint64_t)i) << 40);

//         h ^= mix64_debug(x + 0x9e3779b97f4a7c15ULL);
//         h *= 1099511628211ULL;
//     }

//     // Warp reduce XOR.
//     for (int offset = 16; offset > 0; offset >>= 1) {
//         h ^= __shfl_down_sync(0xffffffff, h, offset);
//     }

//     return h;
// }

// __device__ __forceinline__
// void push_undo_label_count(uint16_t v,
//                            uint8_t* labels,
//                            uint16_t* neiInP,
//                            uint16_t* neiInG,
//                            UndoRec* undo,
//                            unsigned int* undo_top)
// {
//     uint16_t pos = (*undo_top)++;

//     if (pos < LOCAL_UNDO_CAP) {
//         undo[pos] = UndoRec(v, labels[v], neiInP[v], neiInG[v]);
//     }
// }

// __device__ __forceinline__
// void undo_to_mark(unsigned int mark,
//                   uint8_t* labels,
//                   uint16_t* neiInP,
//                   uint16_t* neiInG,
//                   UndoRec* undo,
//                   unsigned int* undo_top)
// {
//     while (*undo_top > mark) {
//         UndoRec u = undo[--(*undo_top)];
//         labels[u.v] = u.old_label;
//         neiInP[u.v] = u.old_neiInP;
//         neiInG[u.v] = u.old_neiInG;
//     }
// }

// __device__ __forceinline__
// void restore_to_size_mark(const LocalSizeMark& mark,
//                           uint8_t* labels,
//                           uint16_t* neiInP,
//                           uint16_t* neiInG,
//                           UndoRec* undo,
//                           unsigned int* undo_top,
//                           unsigned int* plex_sz,
//                           unsigned int* cand_sz,
//                           unsigned int* excl_sz)
// {
//     int lane_id = threadIdx.x & 31;

//     if (lane_id == 0) {
//         unsigned int guard = 0;

//         while (*undo_top > mark.undo_top && guard < LOCAL_UNDO_CAP) {
//             UndoRec u = undo[--(*undo_top)];
//             labels[u.v] = u.old_label;
//             neiInP[u.v] = u.old_neiInP;
//             neiInG[u.v] = u.old_neiInG;
//             guard++;
//         }

//         *plex_sz = mark.plex_sz;
//         *cand_sz = mark.cand_sz;
//         *excl_sz = mark.excl_sz;
//     }

//     __syncwarp();
// }

// __device__ __forceinline__
// int select_first_candidate(uint8_t* labels, unsigned int n)
// {
//     for (unsigned int v = 0; v < n; ++v) {
//         if (labels[v] == LBL_C) return (int)v;
//     }
//     return -1;
// }

// __device__
// void local_include_vertex_debug(uint16_t v,
//                                 uint8_t* labels,
//                                 uint16_t* neiInP,
//                                 uint16_t* neiInG,
//                                 unsigned int* offsetsBase,
//                                 unsigned int* neighborsBase,
//                                 unsigned int* degreeBase,
//                                 UndoRec* undo,
//                                 unsigned int* undo_top,
//                                 unsigned int* plex_sz,
//                                 unsigned int* cand_sz)
// {
//     int lane_id = threadIdx.x & 31;

//     if (lane_id == 0) {
//         push_undo_label_count(v, labels, neiInP, neiInG, undo, undo_top);

//         labels[v] = P;
//         (*plex_sz)++;
//         (*cand_sz)--;
//     }

//     __syncwarp();

//     // v moves from C to P.
//     // For each neighbor u of v, neiInP[u] increases by 1.
//     unsigned int deg = degreeBase[v];
//     unsigned int off = offsetsBase[v];

//     for (unsigned int i = lane_id; i < deg; i += 32) {
//         uint16_t u = (uint16_t)neighborsBase[off + i];

//         unsigned int pos = atomicAdd(undo_top, 1);

//         if (pos < LOCAL_UNDO_CAP) {
//             undo[pos] = UndoRec(u, labels[u], neiInP[u], neiInG[u]);
//         }

//         neiInP[u]++;
//     }

//     __syncwarp();
// }


// __device__
// void local_exclude_vertex_debug(uint16_t v,
//                                 uint8_t* labels,
//                                 uint16_t* neiInP,
//                                 uint16_t* neiInG,
//                                 unsigned int* offsetsBase,
//                                 unsigned int* neighborsBase,
//                                 unsigned int* degreeBase,
//                                 UndoRec* undo,
//                                 unsigned int* undo_top,
//                                 unsigned int* cand_sz,
//                                 unsigned int* excl_sz)
// {
//     int lane_id = threadIdx.x & 31;

//     if (lane_id == 0) {
//         push_undo_label_count(v, labels, neiInP, neiInG, undo, undo_top);

//         labels[v] = X;
//         (*cand_sz)--;
//         (*excl_sz)++;
//     }

//     __syncwarp();

//     // v moves from C to X.
//     // For each neighbor u of v, neiInG[u] decreases by 1 because v left P ∪ C.
//     unsigned int deg = degreeBase[v];
//     unsigned int off = offsetsBase[v];

//     for (unsigned int i = lane_id; i < deg; i += 32) {
//         uint16_t u = (uint16_t)neighborsBase[off + i];

//         unsigned int pos = atomicAdd(undo_top, 1);

//         if (pos < LOCAL_UNDO_CAP) {
//             undo[pos] = UndoRec(u, labels[u], neiInP[u], neiInG[u]);
//         }

//         if (neiInG[u] > 0) {
//             neiInG[u]--;
//         }
//     }

//     __syncwarp();
// }

// __device__ __forceinline__
// void spill_tiny_task_debug(int idx,
//                            unsigned int parent_task_pos,
//                            unsigned int plex_sz,
//                            unsigned int cand_sz,
//                            unsigned int excl_sz,
//                            unsigned int depth,
//                            TinyTask* out_tiny_tasks,
//                            unsigned int* out_tiny_tail,
//                            unsigned int tiny_cap,
//                            Delta* out_delta_log,
//                            unsigned int* out_delta_tail,
//                            unsigned int delta_cap,
//                            const uint16_t* path_vertices,
//                            const uint8_t* path_decisions,
//                            uint8_t* labels,
//                            uint16_t* neiInP,
//                            uint16_t* neiInG,
//                            unsigned int hash_n,
//                            unsigned int* debug_spilled)
// {
//     const unsigned int lane_id = threadIdx.x & 31;

//     uint64_t state_hash = hash_state_debug(labels, neiInP, neiInG, hash_n);

//     if (lane_id == 0) {
//         bool log_ok = true;
//         unsigned int log_off = 0;

//         if (depth > 0) {
//             log_off = atomicAdd(out_delta_tail, depth);
//             log_ok = (log_off + depth <= delta_cap);
//         }

//         if (log_ok) {
//             unsigned int pos = atomicAdd(out_tiny_tail, 1);

//             if (pos < tiny_cap) {
//                 for (unsigned int i = 0; i < depth; ++i) {
//                     const uint16_t v = path_vertices[i];

//                     if (path_decisions[i] == 1) {
//                         // include branch: C -> P
//                         out_delta_log[log_off + i] = Delta(v, C, P, 0, 0);
//                     } else {
//                         // exclude branch: C -> X
//                         out_delta_log[log_off + i] = Delta(v, C, X, 0, 0);
//                     }
//                 }
                
//                 out_tiny_tasks[pos] = TinyTask(
//                     idx,
//                     (uint16_t)plex_sz,
//                     (uint16_t)cand_sz,
//                     (uint16_t)excl_sz,
//                     log_off,
//                     (uint16_t)depth,
//                     (uint16_t)depth,
//                     parent_task_pos,
//                     state_hash
//                 );
//             }
//         }

//         atomicAdd(debug_spilled, 1);
//     }

//     __syncwarp();
// }

// __global__
// void BNB_localDFS_debug(S_pointers s,
//                         Task* in_tasks,
//                         unsigned int num_tasks,
//                         unsigned int task_head,
//                         TinyTask* out_tiny_tasks,
//                         unsigned int* out_tiny_tail,
//                         unsigned int tiny_cap,
//                         Delta* out_delta_log,
//                         unsigned int* out_delta_tail,
//                         unsigned int delta_cap,
//                         unsigned int* debug_expanded,
//                         unsigned int* debug_spilled)
// {  
//     unsigned int lane_id = threadIdx.x & 31;

//     unsigned int task_id = blockIdx.x;

//     if (task_id >= num_tasks) return;

//     Task task = in_tasks[task_head + task_id];

//     unsigned int* offsetsBase   = s.offsets   + task.idx * MAX_BLK_SIZE;
//     unsigned int* neighborsBase = s.neighbors + task.idx * MAX_BLK_SIZE * AVG_DEGREE;
//     unsigned int* degreeBase    = s.degree    + task.idx * MAX_BLK_SIZE;

//     unsigned int n = task.PlexSz + task.CandSz + task.ExclSz;

//     if (n > MAX_BLK_SIZE) return;

//     __shared__ uint8_t  sh_labels[MAX_BLK_SIZE];
//     __shared__ uint16_t sh_neiInP[MAX_BLK_SIZE];
//     __shared__ uint16_t sh_neiInG[MAX_BLK_SIZE];

//     __shared__ UndoRec sh_undo[LOCAL_UNDO_CAP];

//     __shared__ unsigned int sh_undo_top;

//     __shared__ unsigned int sh_plex_sz;
//     __shared__ unsigned int sh_cand_sz;
//     __shared__ unsigned int sh_excl_sz;

//     uint8_t* labels = sh_labels;
//     uint16_t* neiInP = sh_neiInP;
//     uint16_t* neiInG = sh_neiInG;
//     UndoRec* undo = sh_undo;

//     if (lane_id == 0) {
//         sh_undo_top = 0;
//         sh_plex_sz = task.PlexSz;
//         sh_cand_sz = task.CandSz;
//         sh_excl_sz = task.ExclSz;
//     }

//     __syncwarp();

//     // Copy full task state into shared memory once.
//     for (unsigned int i = lane_id; i < n; i += 32) {
//         labels[i] = task.labels[i];
//         neiInP[i] = task.neiInP[i];
//         neiInG[i] = task.neiInG[i];
//     }

//     __syncwarp();

//     uint16_t local_pivots[LOCAL_BNB_DEPTH];
//     uint8_t local_branch_state[LOCAL_BNB_DEPTH];
//     LocalSizeMark marks[LOCAL_BNB_DEPTH];

//     uint16_t path_vertices[LOCAL_BNB_DEPTH];
//     uint8_t path_decisions[LOCAL_BNB_DEPTH];

//     for (int d = 0; d < LOCAL_BNB_DEPTH; ++d) {
//       local_pivots[d] = 0;
//       local_branch_state[d] = 0;
//       marks[d] = LocalSizeMark(0, 0, 0, 0);

//       path_vertices[d] = 0;
//       path_decisions[d] = 0;
//   }

//     int depth = 0;
//     int iter_guard = 0;
//     const int ITER_GUARD_LIMIT = 4096;

//     while (depth >= 0 && iter_guard < ITER_GUARD_LIMIT) {
//         iter_guard++;
//         // If we reached local depth limit, count this as a spill point.
//         if (depth >= LOCAL_BNB_DEPTH) {
//           spill_tiny_task_debug(task.idx,
//                                 task_head+task_id,
//                                 sh_plex_sz,
//                                 sh_cand_sz,
//                                 sh_excl_sz,
//                                 (unsigned int)depth,
//                                 out_tiny_tasks,
//                                 out_tiny_tail,
//                                 tiny_cap,
//                                 out_delta_log,
//                                 out_delta_tail,
//                                 delta_cap,
//                                 path_vertices,
//                                 path_decisions,
//                                 sh_labels,
//                                 sh_neiInP,
//                                 sh_neiInG,
//                                 MAX_BLK_SIZE,
//                                 debug_spilled);

//             depth--;
//             continue;
//         }

//         // Terminal local node.
//         if (sh_cand_sz == 0) {
//             if (lane_id == 0) {
//                 atomicAdd(debug_expanded, 1);
//             }

//             depth--;
//             continue;
//         }

//         // First visit to this depth: choose pivot and explore INCLUDE.
//         if (local_branch_state[depth] == 0) {
//             int pivot = -1;

//             if (lane_id == 0) {
//                 pivot = select_first_candidate(labels, n);
//             }

//             pivot = __shfl_sync(0xFFFFFFFF, pivot, 0);

//             if (pivot < 0) {
//                 if (lane_id == 0) {
//                     atomicAdd(debug_expanded, 1);
//                 }

//                 depth--;
//                 continue;
//             }

//             local_pivots[depth] = (uint16_t)pivot;

//             marks[depth] = LocalSizeMark(sh_undo_top,
//                                         sh_plex_sz,
//                                         sh_cand_sz,
//                                         sh_excl_sz);

//             local_branch_state[depth] = 1;

//             __syncwarp();

            
//             path_vertices[depth] = (uint16_t)local_pivots[depth];
//             path_decisions[depth] = 1;

//             local_include_vertex_debug((uint16_t)pivot,
//                                       labels,
//                                       neiInP,
//                                       neiInG,
//                                       offsetsBase,
//                                       neighborsBase,
//                                       degreeBase,
//                                       undo,
//                                       &sh_undo_top,
//                                       &sh_plex_sz,
//                                       &sh_cand_sz);

//             depth++;
//             continue;
//         }

//         // INCLUDE child has returned. Restore parent state and explore EXCLUDE.
//         if (local_branch_state[depth] == 1) {
//             restore_to_size_mark(marks[depth],
//                                 labels,
//                                 neiInP,
//                                 neiInG,
//                                 undo,
//                                 &sh_undo_top,
//                                 &sh_plex_sz,
//                                 &sh_cand_sz,
//                                 &sh_excl_sz);

//             uint16_t pivot = local_pivots[depth];

//             local_branch_state[depth] = 2;

//             __syncwarp();

//             path_vertices[depth] = (uint16_t)local_pivots[depth];
//             path_decisions[depth] = 2;

//             local_exclude_vertex_debug(pivot,
//                                       labels,
//                                       neiInP,
//                                       neiInG,
//                                       offsetsBase,
//                                       neighborsBase,
//                                       degreeBase,
//                                       undo,
//                                       &sh_undo_top,
//                                       &sh_cand_sz,
//                                       &sh_excl_sz);

//             depth++;
//             continue;
//         }

//         // Both INCLUDE and EXCLUDE children have returned.
//         // Restore parent and backtrack.
//         if (local_branch_state[depth] == 2) {
//             restore_to_size_mark(marks[depth],
//                                 labels,
//                                 neiInP,
//                                 neiInG,
//                                 undo,
//                                 &sh_undo_top,
//                                 &sh_plex_sz,
//                                 &sh_cand_sz,
//                                 &sh_excl_sz);

//             local_branch_state[depth] = 0;

//             __syncwarp();

//             depth--;
//             continue;
//         }
//     }
//     if (lane_id == 0 && iter_guard >= ITER_GUARD_LIMIT) {
//       atomicAdd(debug_spilled, 1);
//     }
// }

// __device__ __forceinline__
// void count_labels_debug(unsigned int lane_id,
//                         uint8_t* labels,
//                         unsigned int n,
//                         unsigned int* p_count,
//                         unsigned int* c_count,
//                         unsigned int* x_count)
// {
//     __shared__ unsigned int sh_p;
//     __shared__ unsigned int sh_c;
//     __shared__ unsigned int sh_x;

//     if (lane_id == 0) {
//         sh_p = 0;
//         sh_c = 0;
//         sh_x = 0;
//     }

//     __syncwarp();

//     unsigned int local_p = 0;
//     unsigned int local_c = 0;
//     unsigned int local_x = 0;

//     for (unsigned int i = lane_id; i < n; i += 32) {
//         if (labels[i] == P) {
//             local_p++;
//         } else if (labels[i] == C) {
//             local_c++;
//         } else if (labels[i] == X) {
//             local_x++;
//         }
//     }

//     if (local_p) atomicAdd(&sh_p, local_p);
//     if (local_c) atomicAdd(&sh_c, local_c);
//     if (local_x) atomicAdd(&sh_x, local_x);

//     __syncwarp();

//     if (lane_id == 0) {
//         *p_count = sh_p;
//         *c_count = sh_c;
//         *x_count = sh_x;
//     }

//     __syncwarp();
// }

// __global__
// void resumeTinyTasks_debug(S_pointers s,
//                            Task* base_tasks,
//                            TinyTask* tiny_tasks,
//                            unsigned int num_tiny_tasks,
//                            Delta* delta_log,
//                            unsigned int* debug_checked,
//                            unsigned int* debug_errors)
// {
//     unsigned int tiny_id = blockIdx.x;
//     unsigned int lane_id = threadIdx.x & 31;

//     if (tiny_id >= num_tiny_tasks) return;

//     TinyTask tt = tiny_tasks[tiny_id];

//     // Exact parent full Task from which this TinyTask was spilled.
//     Task base = base_tasks[tt.parent_task_pos];

//     __shared__ uint8_t sh_labels[MAX_BLK_SIZE];
//     __shared__ uint16_t sh_neiInP[MAX_BLK_SIZE];
//     __shared__ uint16_t sh_neiInG[MAX_BLK_SIZE];

//     __shared__ UndoRec sh_undo[LOCAL_UNDO_CAP];
//     __shared__ unsigned int sh_undo_top;

//     __shared__ unsigned int sh_plex_sz;
//     __shared__ unsigned int sh_cand_sz;
//     __shared__ unsigned int sh_excl_sz;

//     unsigned int* offsetsBase =
//         s.offsets + base.idx * MAX_BLK_SIZE;

//     unsigned int* degreeBase =
//         s.degree + base.idx * MAX_BLK_SIZE;

//     unsigned int* neighborsBase =
//         s.neighbors + base.idx * MAX_BLK_SIZE * AVG_DEGREE;

//     // Copy parent full Task state into shared memory.
//     for (unsigned int i = lane_id; i < MAX_BLK_SIZE; i += 32) {
//         sh_labels[i] = base.labels[i];
//         sh_neiInP[i] = base.neiInP[i];
//         sh_neiInG[i] = base.neiInG[i];
//     }

//     if (lane_id == 0) {
//         sh_undo_top = 0;
//         sh_plex_sz = base.PlexSz;
//         sh_cand_sz = base.CandSz;
//         sh_excl_sz = base.ExclSz;
//     }

//     __syncwarp();

//     // Replay the compact Delta decision path.
//     for (unsigned int i = 0; i < tt.log_len; ++i) {
//         Delta d = delta_log[tt.log_off + i];
//         uint16_t v = d.v;

//         if (d.old_label != C) {
//             if (lane_id == 0) {
//                 atomicAdd(debug_errors, 1);
//                 printf("resume error: tiny=%u delta=%u old_label=%u expected C\n",
//                        tiny_id, i, d.old_label);
//             }
//             return;
//         }

//         if (d.new_label == P) {
//             local_include_vertex_debug(
//                 v,
//                 sh_labels,
//                 sh_neiInP,
//                 sh_neiInG,
//                 offsetsBase,
//                 neighborsBase,
//                 degreeBase,
//                 sh_undo,
//                 &sh_undo_top,
//                 &sh_plex_sz,
//                 &sh_cand_sz
//             );
//         } else if (d.new_label == X) {
//             local_exclude_vertex_debug(
//                 v,
//                 sh_labels,
//                 sh_neiInP,
//                 sh_neiInG,
//                 offsetsBase,
//                 neighborsBase,
//                 degreeBase,
//                 sh_undo,
//                 &sh_undo_top,
//                 &sh_cand_sz,
//                 &sh_excl_sz
//             );
//         } else {
//             if (lane_id == 0) {
//                 atomicAdd(debug_errors, 1);
//                 printf("resume error: tiny=%u delta=%u invalid new_label=%u\n",
//                        tiny_id, i, d.new_label);
//             }
//             return;
//         }

//         __syncwarp();
//     }

//     uint64_t replay_hash = hash_state_debug(sh_labels, sh_neiInP, sh_neiInG, MAX_BLK_SIZE);

//     // Check final reconstructed sizes against TinyTask metadata.
//     if (lane_id == 0) {
//         bool ok = true;
//         bool size_ok = true;
//         bool hash_ok = true;

//         if (sh_plex_sz != tt.plex_sz) size_ok = false;
//         if (sh_cand_sz != tt.cand_sz) size_ok = false;
//         if (sh_excl_sz != tt.excl_sz) size_ok = false;

//         if (replay_hash != tt.state_hash) hash_ok=false;

//         ok = size_ok && hash_ok;

//         if (!ok) {
//             atomicAdd(debug_errors, 1);

//             if (tiny_id < 8) {
//                 printf("resume size mismatch tiny=%u parent=%u idx=%d "
//                        "size_ok=%u, hash_ok=%u "
//                        "got(P=%u C=%u X=%u) expected(P=%u C=%u X=%u) "
//                        "log_off=%u log_len=%u depth=%u\n",
//                        tiny_id,
//                        tt.parent_task_pos,
//                        tt.idx,
//                        (unsigned int)size_ok,
//                        (unsigned int)hash_ok,
//                        sh_plex_sz,
//                        sh_cand_sz,
//                        sh_excl_sz,
//                        tt.plex_sz,
//                        tt.cand_sz,
//                        tt.excl_sz,
//                        tt.log_off,
//                        tt.log_len,
//                        tt.depth);
//             }
//         } else {
//             if (tiny_id < 4) {
//                 printf("resume ok tiny=%u parent=%u idx=%d P=%u C=%u X=%u log_len=%u hash=%llu\n",
//                        tiny_id,
//                        tt.parent_task_pos,
//                        tt.idx,
//                        sh_plex_sz,
//                        sh_cand_sz,
//                        sh_excl_sz,
//                        tt.log_len,
//                        (unsigned long long)replay_hash);
//             }
//         }

//         atomicAdd(debug_checked, 1);
//     }
// }

// __device__ __forceinline__
// void spill_tiny_task_append_debug(const TinyTask& parent_tt,
//                                   unsigned int plex_sz,
//                                   unsigned int cand_sz,
//                                   unsigned int excl_sz,
//                                   unsigned int new_depth,
//                                   TinyTask* out_tiny_tasks,
//                                   unsigned int* out_tiny_tail,
//                                   unsigned int tiny_cap,
//                                   Delta* in_delta_log,
//                                   Delta* out_delta_log,
//                                   unsigned int* out_delta_tail,
//                                   unsigned int delta_cap,
//                                   const uint16_t* path_vertices,
//                                   const uint8_t* path_decisions,
//                                   uint8_t* labels,
//                                   uint16_t* neiInP,
//                                   uint16_t* neiInG,
//                                   unsigned int hash_n,
//                                   unsigned int* debug_spilled,
//                                   unsigned int* debug_overflow)
// {
//     const unsigned int lane_id = threadIdx.x & 31;

//     const unsigned int old_len = parent_tt.log_len;
//     const unsigned int total_len = old_len + new_depth;

//     uint64_t state_hash = hash_state_debug(labels, neiInP, neiInG, hash_n);
//     state_hash = __shfl_sync(0xFFFFFFFF, state_hash, 0);

//     if (lane_id == 0) {
//         if (total_len > 65535u){
//           unsigned int ov = atomicAdd(debug_overflow, 1);

//           if (ov < 8)
//           {
//             printf("Spill Overflow: total_len too large parent=%u idx=%u old_len=%u new_depth=%u total_len=%u\n", parent_tt.parent_task_pos, parent_tt.idx, old_len, new_depth, total_len);
//           }
//           return;
//         }

//         // unsigned int pos = atomicAdd(out_tiny_tail, 1);

//         // if (pos >= tiny_cap)
//         // {
//         //   atomicAdd(debug_overflow, 1);
//         //   if (pos == tiny_cap){
//         //   printf("Spill overflow: TinyTask cap reached tiny_cap=%u requested_pos=%u parent=%u idx=%d old_len=%u new_depth=%u\n", tiny_cap, pos, parent_tt.parent_task_pos, parent_tt.idx, old_len, new_depth);
//         //   }
//         //   return;
//         // }

//         unsigned int log_off = 0;

//         if (total_len > 0)
//         {
//           log_off = atomicAdd(out_delta_tail, total_len);

//           if (log_off + total_len > delta_cap)
//           {
//             unsigned int ov = atomicAdd(debug_overflow, 1);

//             if (ov < 8)
//             {
//               printf("Spill Overflow: Delta Cap reached delta_cap=%u, log_off=%u, total_len=%u, parent=%u, idx=%d\n", delta_cap, log_off, total_len, parent_tt.parent_task_pos, parent_tt.idx);
//             }
//             return;
//           }
//         }

//         unsigned int pos = atomicAdd(out_tiny_tail, 1);

//         if (pos >= tiny_cap)
//         {
//           unsigned int ov = atomicAdd(debug_overflow, 1);
//           if (ov < 8){
//           printf("Spill overflow: TinyTask cap reached tiny_cap=%u requested_pos=%u parent=%u idx=%d old_len=%u new_depth=%u\n", tiny_cap, pos, parent_tt.parent_task_pos, parent_tt.idx, old_len, new_depth);
//           }
//           return;
//         }

//         for (unsigned int i = 0; i < old_len; i++)
//         {
//           out_delta_log[log_off + i] = in_delta_log[parent_tt.log_off + i];
//         }

//         for (unsigned int i = 0; i < new_depth; i++)
//         {
//           const uint16_t v = path_vertices[i];

//           if (path_decisions[i] == 1)
//           {
//             out_delta_log[log_off + old_len + i] = Delta(v, C, P, 0, 0);
//           }
//           else {
//             out_delta_log[log_off + old_len + i] = Delta(v, C, X, 0, 0);
//           }
//         }

//         out_tiny_tasks[pos] = TinyTask(parent_tt.idx, (uint16_t)plex_sz, (uint16_t)cand_sz, (uint16_t)excl_sz, log_off, (uint16_t)total_len, (uint16_t)total_len, parent_tt.parent_task_pos, state_hash);

//         atomicAdd(debug_spilled, 1);
//     }

//     __syncwarp();
// }

// __device__ __forceinline__
// bool replay_delta_path_debug(unsigned int tiny_id,
//                              const TinyTask& tt,
//                              Delta* delta_log,
//                              uint8_t* labels,
//                              uint16_t* neiInP,
//                              uint16_t* neiInG,
//                              UndoRec* undo,
//                              unsigned int* undo_top,
//                              unsigned int* plex_sz,
//                              unsigned int* cand_sz,
//                              unsigned int* excl_sz,
//                              unsigned int* offsetsBase,
//                              unsigned int* neighborsBase,
//                              unsigned int* degreeBase,
//                              unsigned int* debug_errors)
// {
//     const unsigned int lane_id = threadIdx.x & 31;

//     for (unsigned int i = 0; i < tt.log_len; ++i) {
//         Delta d = delta_log[tt.log_off + i];

//         if (d.old_label != C) {
//             if (lane_id == 0) {
//                 atomicAdd(debug_errors, 1);
//                 printf("replay error: tiny=%u delta=%u old_label=%u expected C\n",
//                        tiny_id, i, d.old_label);
//             }
//             return false;
//         }

//         if (d.new_label == P) {
//             local_include_vertex_debug(
//                 d.v,
//                 labels,
//                 neiInP,
//                 neiInG,
//                 offsetsBase,
//                 neighborsBase,
//                 degreeBase,
//                 undo,
//                 undo_top,
//                 plex_sz,
//                 cand_sz
//             );
//         } else if (d.new_label == X) {
//             local_exclude_vertex_debug(
//                 d.v,
//                 labels,
//                 neiInP,
//                 neiInG,
//                 offsetsBase,
//                 neighborsBase,
//                 degreeBase,
//                 undo,
//                 undo_top,
//                 cand_sz,
//                 excl_sz
//             );
//         } else {
//             if (lane_id == 0) {
//                 atomicAdd(debug_errors, 1);
//                 printf("replay error: tiny=%u delta=%u invalid new_label=%u\n",
//                        tiny_id, i, d.new_label);
//             }
//             return false;
//         }

//         __syncwarp();
//     }

//     return true;
// }

// __device__ __forceinline__
// bool local_undo_has_capacity_for_vertex(
//     uint16_t v,
//     unsigned int* degreeBase,
//     unsigned int undo_top
// ) {
//     // One undo for the vertex itself plus one per neighbor update.
//     unsigned int need = 1u + degreeBase[v];
//     return undo_top + need <= LOCAL_UNDO_CAP;
// }

// __global__
// void resumeTinyTasks_continue_debug(S_pointers s,
//                                     Task* base_tasks,
//                                     TinyTask* in_tiny_tasks,
//                                     unsigned int num_tiny_tasks,
//                                     Delta* in_delta_log,
//                                     TinyTask* out_tiny_tasks,
//                                     unsigned int* out_tiny_tail,
//                                     unsigned int tiny_cap,
//                                     Delta* out_delta_log,
//                                     unsigned int* out_delta_tail,
//                                     unsigned int delta_cap,
//                                     unsigned int* debug_checked,
//                                     unsigned int* debug_errors,
//                                     unsigned int* debug_spilled,
//                                     unsigned int* debug_overflow)
// {
//     unsigned int tiny_id = blockIdx.x;
//     unsigned int lane_id = threadIdx.x & 31;

//     if (tiny_id >= num_tiny_tasks) return;

//     TinyTask tt = in_tiny_tasks[tiny_id];
//     Task base = base_tasks[tt.parent_task_pos];

//     unsigned int n = base.PlexSz + base.CandSz + base.ExclSz;

//     __shared__ uint8_t sh_labels[MAX_BLK_SIZE];
//     __shared__ uint16_t sh_neiInP[MAX_BLK_SIZE];
//     __shared__ uint16_t sh_neiInG[MAX_BLK_SIZE];

//     __shared__ UndoRec sh_undo[LOCAL_UNDO_CAP];
//     __shared__ unsigned int sh_undo_top;

//     __shared__ unsigned int sh_plex_sz;
//     __shared__ unsigned int sh_cand_sz;
//     __shared__ unsigned int sh_excl_sz;

//     unsigned int* offsetsBase =
//         s.offsets + base.idx * MAX_BLK_SIZE;

//     unsigned int* degreeBase =
//         s.degree + base.idx * MAX_BLK_SIZE;

//     unsigned int* neighborsBase =
//         s.neighbors + base.idx * MAX_BLK_SIZE * AVG_DEGREE;

//     // Copy original full parent state.
//     for (unsigned int i = lane_id; i < MAX_BLK_SIZE; i += 32) {
//         sh_labels[i] = base.labels[i];
//         sh_neiInP[i] = base.neiInP[i];
//         sh_neiInG[i] = base.neiInG[i];
//     }

//     if (lane_id == 0) {
//         sh_undo_top = 0;
//         sh_plex_sz = base.PlexSz;
//         sh_cand_sz = base.CandSz;
//         sh_excl_sz = base.ExclSz;
//     }

//     __syncwarp();

//     // Replay parent TinyTask path.
//     bool replay_ok = replay_delta_path_debug(
//         tiny_id,
//         tt,
//         in_delta_log,
//         sh_labels,
//         sh_neiInP,
//         sh_neiInG,
//         sh_undo,
//         &sh_undo_top,
//         &sh_plex_sz,
//         &sh_cand_sz,
//         &sh_excl_sz,
//         offsetsBase,
//         neighborsBase,
//         degreeBase,
//         debug_errors
//     );

//     if (!replay_ok) return;

//     __syncwarp();

//     // Verify replay reached the exact parent TinyTask state.
//     uint64_t replay_hash =
//         hash_state_debug(sh_labels, sh_neiInP, sh_neiInG, MAX_BLK_SIZE);

//     replay_hash = __shfl_sync(0xFFFFFFFF, replay_hash, 0);

//     if (lane_id == 0) {
//         bool ok = true;

//         if (sh_plex_sz != tt.plex_sz) ok = false;
//         if (sh_cand_sz != tt.cand_sz) ok = false;
//         if (sh_excl_sz != tt.excl_sz) ok = false;
//         if (replay_hash != tt.state_hash) ok = false;

//         if (!ok) {
//             atomicAdd(debug_errors, 1);

//             if (tiny_id < 16) {
//                 printf("continue replay mismatch tiny=%u parent=%u idx=%d "
//                        "got(P=%u C=%u X=%u hash=%llu) "
//                        "expected(P=%u C=%u X=%u hash=%llu) "
//                        "log_off=%u log_len=%u\n",
//                        tiny_id,
//                        tt.parent_task_pos,
//                        tt.idx,
//                        sh_plex_sz,
//                        sh_cand_sz,
//                        sh_excl_sz,
//                        (unsigned long long)replay_hash,
//                        tt.plex_sz,
//                        tt.cand_sz,
//                        tt.excl_sz,
//                        (unsigned long long)tt.state_hash,
//                        tt.log_off,
//                        tt.log_len);
//             }
//         }

//         atomicAdd(debug_checked, 1);
//     }

//     __syncwarp();

//     // If replay was wrong, do not continue this TinyTask.
//     bool replay_state_ok = (replay_hash == tt.state_hash && sh_plex_sz == tt.plex_sz && sh_cand_sz == tt.cand_sz && sh_excl_sz == tt.excl_sz);
//     if (!replay_state_ok) return;

//     // ---------------------------------------------------------------------
//     // Continue bounded local DFS from the replayed TinyTask state.
//     // ---------------------------------------------------------------------

//     uint16_t local_pivots[LOCAL_BNB_DEPTH];
//     uint8_t local_branch_state[LOCAL_BNB_DEPTH];
//     LocalSizeMark marks[LOCAL_BNB_DEPTH];

//     uint16_t path_vertices[LOCAL_BNB_DEPTH];
//     uint8_t path_decisions[LOCAL_BNB_DEPTH];

//     for (int d = 0; d < LOCAL_BNB_DEPTH; ++d) {
//         local_pivots[d] = 0;
//         local_branch_state[d] = 0;
//         marks[d] = LocalSizeMark(0, 0, 0, 0);

//         path_vertices[d] = 0;
//         path_decisions[d] = 0;
//     }

//     int depth = 0;
//     int iter_guard = 0;
//     const int ITER_GUARD_LIMIT = 16384;

//     while (depth >= 0 && iter_guard < ITER_GUARD_LIMIT) {
//         iter_guard++;

//         unsigned int overflow_seen = 0;

//         if (lane_id == 0) overflow_seen = *debug_overflow;

//         overflow_seen = __shfl_sync(0xFFFFFFFF, overflow_seen, 0);

//         if (overflow_seen != 0) break;

//         // Reached another local cutoff: spill TinyTask B.
//         if (depth >= LOCAL_BNB_DEPTH) {
//             spill_tiny_task_append_debug(
//                 tt,
//                 sh_plex_sz,
//                 sh_cand_sz,
//                 sh_excl_sz,
//                 (unsigned int)depth,
//                 out_tiny_tasks,
//                 out_tiny_tail,
//                 tiny_cap,
//                 in_delta_log,
//                 out_delta_log,
//                 out_delta_tail,
//                 delta_cap,
//                 path_vertices,
//                 path_decisions,
//                 sh_labels,
//                 sh_neiInP,
//                 sh_neiInG,
//                 MAX_BLK_SIZE,
//                 debug_spilled,
//                 debug_overflow
//             );

//             __syncwarp();

//             depth--;
//             continue;
//         }

//         if (sh_cand_sz == 0){
//           depth--;
//           continue;
//         }

//         // Need to choose pivot at this depth.
//         if (local_branch_state[depth] == 0) {
//             int pivot = -1;

//             unsigned int mark_undo = 0;
//             unsigned int mark_plex = 0;
//             unsigned int mark_cand = 0;
//             unsigned int mark_excl = 0;

//             // Simple debug pivot: first C vertex.
//             if (lane_id == 0) {
//                 pivot = select_first_candidate(sh_labels, n);

//                 mark_undo = sh_undo_top;
//                 mark_plex = sh_plex_sz;
//                 mark_cand = sh_cand_sz;
//                 mark_excl = sh_excl_sz;
//             }

//             pivot = __shfl_sync(0xffffffff, pivot, 0);
//             mark_undo = __shfl_sync(0xFFFFFFFF, mark_undo, 0);
//             mark_plex = __shfl_sync(0xFFFFFFFF, mark_plex, 0);
//             mark_cand = __shfl_sync(0xFFFFFFFF, mark_cand, 0);
//             mark_excl = __shfl_sync(0xFFFFFFFF, mark_excl, 0);

//             if (pivot < 0) {
//               depth--;
//               continue;
//             }

//             local_pivots[depth] = (uint16_t)pivot;
//             marks[depth] = LocalSizeMark(mark_undo, mark_plex, mark_cand, mark_excl);

//             local_branch_state[depth] = 1;

//             path_vertices[depth] = (uint16_t)pivot;
//             path_decisions[depth] = 1;

//             bool undo_ok = true;

//             if (lane_id == 0) {
//                 undo_ok = local_undo_has_capacity_for_vertex(
//                     (uint16_t)pivot,
//                     degreeBase,
//                     sh_undo_top
//                 );

//                 if (!undo_ok) {
//                     unsigned int ov = atomicAdd(debug_overflow, 1);

//                     if (ov < 8) {
//                         printf("Undo overflow before include: tiny=%u parent=%u idx=%d "
//                               "pivot=%u undo_top=%u degree=%u cap=%u log_len=%u depth=%d\n",
//                               tiny_id,
//                               tt.parent_task_pos,
//                               tt.idx,
//                               (unsigned int)pivot,
//                               sh_undo_top,
//                               degreeBase[pivot],
//                               LOCAL_UNDO_CAP,
//                               tt.log_len,
//                               depth);
//                     }
//                 }
//             }

//             undo_ok = __shfl_sync(0xffffffff, undo_ok, 0);

//             if (!undo_ok) {
//                 break;
//             }

//             __syncwarp();

//             local_include_vertex_debug((uint16_t)pivot, sh_labels, sh_neiInP, sh_neiInG, offsetsBase, neighborsBase, degreeBase, sh_undo, &sh_undo_top, &sh_plex_sz, &sh_cand_sz);

//             __syncwarp();

//             depth++;
//             continue;
//         }

//         if (local_branch_state[depth] == 1) {
//             // Restore to mark before trying exclude branch.
//             restore_to_size_mark(
//               marks[depth],
//               sh_labels,
//               sh_neiInP,
//               sh_neiInG,
//               sh_undo,
//               &sh_undo_top,
//               &sh_plex_sz,
//               &sh_cand_sz,
//               &sh_excl_sz
//           );

//             __syncwarp();

//             uint16_t pivot = local_pivots[depth];
//             local_branch_state[depth] = 2;

//             // Exclude branch: C -> X
//             path_vertices[depth] = pivot;
//             path_decisions[depth] = 2;

//             bool undo_ok = true;

//             if (lane_id == 0) {
//                 undo_ok = local_undo_has_capacity_for_vertex(
//                     (uint16_t)pivot,
//                     degreeBase,
//                     sh_undo_top
//                 );

//                 if (!undo_ok) {
//                     unsigned int ov = atomicAdd(debug_overflow, 1);

//                     if (ov < 8) {
//                         printf("Undo overflow before include: tiny=%u parent=%u idx=%d "
//                               "pivot=%u undo_top=%u degree=%u cap=%u log_len=%u depth=%d\n",
//                               tiny_id,
//                               tt.parent_task_pos,
//                               tt.idx,
//                               (unsigned int)pivot,
//                               sh_undo_top,
//                               degreeBase[pivot],
//                               LOCAL_UNDO_CAP,
//                               tt.log_len,
//                               depth);
//                     }
//                 }
//             }

//             undo_ok = __shfl_sync(0xffffffff, undo_ok, 0);

//             if (!undo_ok) {
//                 break;
//             }

//             __syncwarp();

//             local_exclude_vertex_debug(
//                 pivot,
//                 sh_labels,
//                 sh_neiInP,
//                 sh_neiInG,
//                 offsetsBase,
//                 neighborsBase,
//                 degreeBase,
//                 sh_undo,
//                 &sh_undo_top,
//                 &sh_cand_sz,
//                 &sh_excl_sz
//             );

//             __syncwarp();
//             depth++;
//             continue;
//         }

//         // Both branches done at this depth. Backtrack.
//         if (local_branch_state[depth] == 2){
//         restore_to_size_mark(
//           marks[depth],
//           sh_labels,
//           sh_neiInP,
//           sh_neiInG,
//           sh_undo,
//           &sh_undo_top,
//           &sh_plex_sz,
//           &sh_cand_sz,
//           &sh_excl_sz
//       );
//         local_branch_state[depth] = 0;
//         __syncwarp();

//         depth--;
//         continue;
//     }
//     }

//     if (lane_id == 0 && iter_guard >= ITER_GUARD_LIMIT){
//       atomicAdd(debug_errors, 1);
//       if (tiny_id < 16){
//         printf("Continue DFS iter_guard hit tiny=%u parent=%u idx=%d depth=%d log_len=%u\n", tiny_id, tt.parent_task_pos, tt.idx, depth, tt.log_len);
//       }
//     }
// }


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
  unsigned int* neighborsBase = s.neighbors + warp_id * MAX_BLK_SIZE * AVG_DEGREE;
  unsigned int* degreeHop = s.degreeHop + warp_id * MAX_BLK_SIZE;

  // uint8_t* commonMtxBase = commonMtx + warp_id * CAP;
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
      const int common = commonEle(i, j, neighborsBase, offsetsBase, degreeHop);
      if (commonMtxBase[i*n+j])
      {
        if(common>thresCC1)commonMtxBase[i*n+j]=LINK2MORE;
        else if(common==thresCC1)commonMtxBase[i*n+j]=LINK2EQUAL;
        else commonMtxBase[i*n+j]=LINK2LESS;
      }
      else{
        if(common>thresCC2)commonMtxBase[i*n+j]=UNLINK2MORE;
        else if(common==thresCC2)commonMtxBase[i*n+j]=UNLINK2EQUAL;
        else commonMtxBase[i*n+j]=UNLINK2LESS;
      }
      commonMtxBase[j*n+i] = commonMtxBase[i*n+j];
    }
  }
  for (int i = hop+lane_id; i < n;i+=32)
  {
    for (int j = 0; j < hop; j++)
    {
      const int common = commonEle(i, j, neighborsBase, offsetsBase, degreeHop);
      if (commonMtxBase[i*n+j])
      {
        if(common>thresPC1)commonMtxBase[i*n+j]=LINK2MORE;
        else if(common==thresPC1)commonMtxBase[i*n+j]=LINK2EQUAL;
        else commonMtxBase[i*n+j]=LINK2LESS;
      }
      else{
        if(common>thresPC2)commonMtxBase[i*n+j]=UNLINK2MORE;
        else if(common==thresPC2)commonMtxBase[i*n+j]=UNLINK2EQUAL;
        else commonMtxBase[i*n+j]=UNLINK2LESS;
      }
      commonMtxBase[j*n+i]=commonMtxBase[i*n+j];
    }
    if (k==2) continue;
    for (int j = hop; j<i; j++)
    {
      const int common = commonEle(i, j, neighborsBase, offsetsBase, degreeHop);
      if (commonMtxBase[i*n+j])
      {
        if(common>thresPP1)commonMtxBase[i*n+j]=LINK2MORE;
        else if(common==thresPP1)commonMtxBase[i*n+j]=LINK2EQUAL;
        else commonMtxBase[i*n+j]=LINK2LESS;
      }
      else{
        if(common>thresPP2)commonMtxBase[i*n+j]=UNLINK2MORE;
        else if(common==thresPP2)commonMtxBase[i*n+j]=UNLINK2EQUAL;
        else commonMtxBase[i*n+j]=UNLINK2LESS;
      }
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
  unsigned int local_warp_id = threadIdx.x >> 5; 

  if ((warp_id+WARPS*idx) >= (g.n-p.lb+2)) return;

  // if (warp_id >= 10) return;

  int k = p.k;
  int q = p.lb;
  float thres = p.thres;

  unsigned int* counterBase = d_blk_counter + warp_id;

  if (counterBase[0] < q) return;
  

  unsigned int* degreeBase = s.degree + warp_id * MAX_BLK_SIZE;
  unsigned int* offsetsBase = s.offsets + warp_id * MAX_BLK_SIZE;
  unsigned int* neighborsBase = s.neighbors + warp_id * MAX_BLK_SIZE * AVG_DEGREE;
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

  // uint8_t* commonMtxBase = commonMtx + warp_id * CAP;
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
    for (int i = lane_id; i < degreeBase[0]; i+=32)
    {
      const int nei = neighborsBase[offsetsBase[0]+i];
      neiInPBase[nei]++;
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
      if (lane_id == 0) printf("capacity crossed: %d\n", sz);
      break;
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

          
          __syncwarp();
          if (lane_id == 0)
          {
            Task &nt = t.d_tasks_A[pos];
            nt.idx = warp_id;
            nt.PlexSz = PlexSz[0];
            nt.CandSz = Cand1Sz[0];
            nt.ExclSz = ExclSz[0];
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

__global__ void kSearch2(int idx, P_pointers p, S_pointers s, G_pointers g, T_pointers t, unsigned int* d_blk_counter, unsigned int* d_res, unsigned int* d_br, unsigned int* d_state, unsigned int* d_len, unsigned int* d_sz, uint16_t* neiInG, uint16_t* neiInP, unsigned int* plex_count, uint8_t* commonMtx, unsigned int* recCand1, unsigned int* recCand2, unsigned int* d_v2delete, uint32_t* d_adj, unsigned long long* cycles, int* abort_flag, int* d_abort)
{
  unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int warp_id = (global_index / 32);
  unsigned int lane_id = threadIdx.x % 32;
  unsigned int local_warp_id = threadIdx.x >> 5;

  // if (warp_id < 10 || warp_id >= 15) return;
  // if (warp_id != 13) return;

  if ((warp_id+WARPS*idx) >= (g.n-p.lb+2)) return;

  // if (warp_id+WARPS*idx != 4111) return;

  int k = p.k;
  int q = p.lb;

  unsigned int* counterBase = d_blk_counter + warp_id;

  if (counterBase[0] < q) return;
  

  unsigned int* degreeBase = s.degree + warp_id * MAX_BLK_SIZE;
  unsigned int* offsetsBase = s.offsets + warp_id * MAX_BLK_SIZE;
  unsigned int* neighborsBase = s.neighbors + warp_id * MAX_BLK_SIZE * AVG_DEGREE;
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

  // uint8_t* commonMtxBase = commonMtx + warp_id * CAP;
  size_t capacity = size_t(warp_id) * CAP;
  uint8_t* commonMtxBase = commonMtx + capacity;

  uint32_t *adjList = d_adj + ADJSIZE * warp_id;

  int n;
  if (lane_id == 0)
  {
    n = counterBase[0];
  }
  n = __shfl_sync(0xFFFFFFFF, n, 0);

  __shared__ unsigned int sh_PlexSz[WARPS_EACH_BLK];
  __shared__ unsigned int sh_Cand1Sz[WARPS_EACH_BLK];
  __shared__ unsigned int sh_Cand2Sz[WARPS_EACH_BLK];
  __shared__ unsigned int sh_ExclSz[WARPS_EACH_BLK];

  if (lane_id == 0) sh_PlexSz[local_warp_id] = PlexSz[0]; 
  if (lane_id == 0) sh_Cand1Sz[local_warp_id] = Cand1Sz[0];
  if (lane_id == 0) sh_Cand2Sz[local_warp_id] = Cand2Sz[0];
  if (lane_id == 0) sh_ExclSz[local_warp_id] = ExclSz[0];

  

  // int flag = abort_flag[0];
  // if(lane_id == 0) abort_flag[0] = 0;
  int sz;

  int flag;
  if (lane_id == 0) flag = abort_flag[0];
  flag = __shfl_sync(0xFFFFFFFF, flag, 0);
  // __syncthreads();
  // if (lane_id == 0) d_abort[0] = 0;
  // if (lane_id == 0) printf("flag: %d\n", flag);
  // if (warp_id != 24) return;
  if (!flag)
  {
    for (int i = lane_id; i < n; i+=32)
    {
      neiInGBase[i] = degreeHop[i];
      neiInPBase[i] = 0;
    }
    
    
    __syncwarp();
    for (int i = lane_id; i < degreeBase[0]; i+=32)
    {
      const int nei = neighborsBase[offsetsBase[0]+i];
      neiInPBase[nei]++;
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
    if (sz == 0) return;
    // if (lane_id == 0) printf("%d is entering with sz: %d, state:%d, res: %d, br:%d, plexsz: %d, cand1sz: %d, cand2sz: %d, exclsz: %d, len: %d\n", warp_id, sz, state[sz-1], res[sz-1], br[sz-1],PlexSz[0], Cand1Sz[0], Cand2Sz[0], ExclSz[0], length[0]);
  }
  int u, found_idx;
  
  while(sz)
  {
    if (sz >= MAX_DEPTH)
    {
      if (lane_id == 0) printf("capacity crossed: %d\n", sz);
      break;
    }
    switch(state[sz-1])
    {
      //reserve my slot, slot -> []
      case 0:
        // if (lane_id == 0) printf("state: %d, size: %d, plexsz: %d, cand1sz: %d, cand2sz: %d, exclsz: %d\n", state[sz-1], sz, sh_PlexSz[local_warp_id], sh_Cand1Sz[local_warp_id], sh_Cand2Sz[local_warp_id], sh_ExclSz[local_warp_id]);
        if (sh_Cand2Sz[local_warp_id] == 0)
        {
          if (!flag)
          {
          if (sh_PlexSz[local_warp_id] + sh_Cand1Sz[local_warp_id] < q)
          {
            sz--;
            continue;
          }
          
          bool cond = !upperBoundK2(lane_id, k, q, plex, neiInGBase, sh_PlexSz[local_warp_id]);
          
          if (sh_PlexSz[local_warp_id] > 1 && cond)
          {
            sz--;
            continue;
          }
          
          int pos;
          if (lane_id == 0)
          {
            // printf("Appending a new task, warp_id: %d\n", warp_id);
            unsigned int* tail = t.d_tail_A;
            pos = atomicAdd(&tail[0], 1u);
            // printf("position: %d, warp_id: %d\n", pos, warp_id);
          }
          pos = __shfl_sync(0xFFFFFFFF, pos, 0);
          // if (lane_id == 0) printf("pos: %d, max_cap: %d\n", pos, MAX_CAP);
          

          uint8_t* newLabels = t.d_all_labels_A + pos * MAX_BLK_SIZE;
          uint16_t* newNeiInG = t.d_all_neiInG_A + pos * MAX_BLK_SIZE;
          uint16_t* newNeiInP = t.d_all_neiInP_A + pos * MAX_BLK_SIZE;
          for (int i = lane_id; i < n; i+=32)
          {
            newLabels[i] = U;
            newNeiInG[i] = neiInGBase[i];
            newNeiInP[i] = neiInPBase[i];
          }
          for (int i = lane_id; i < sh_PlexSz[local_warp_id]; i+=32)
          {
            const int v = plex[i];
            newLabels[v] = P;
          }
          for (int i = lane_id; i < sh_Cand1Sz[local_warp_id]; i+=32)
          {
            const int v = cand1[i];
            newLabels[v] = C;
          }
          for (int i = lane_id; i < sh_Cand2Sz[local_warp_id]; i+=32)
          {
            const int v = cand2[i];
            newLabels[v] = H;
          }
          for (int i = lane_id; i < sh_ExclSz[local_warp_id]; i+=32)
          {
            const int v = excl[i];
            newLabels[v] = X;
          }

          
          __syncwarp();
          if (lane_id == 0)
          {
            Task &nt = t.d_tasks_A[pos];
            nt.idx = warp_id;
            nt.PlexSz = sh_PlexSz[local_warp_id];
            nt.CandSz = sh_Cand1Sz[local_warp_id];
            nt.ExclSz = sh_ExclSz[local_warp_id];
            nt.labels = newLabels;
            nt.neiInG = newNeiInG;
            nt.neiInP = newNeiInP;
          
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
        if (pos+1 > (MAX_CAP/4)-WARPS)
        {
          // if(lane_id == 0) printf("Maximum Capacity Reached in kSearch\n");
          // atomicExch(abort_flag, 1);
          if (lane_id == 0) d_abort[0] = 1;
          size[0] = sz;
          return;
        }
      }
          flag = 0;
          sz--;
          continue;
        }
        
        if (lane_id == 0)
        {
          u = cand2[sh_Cand2Sz[local_warp_id]-1];
          excl[sh_ExclSz[local_warp_id]++] = u;
          --sh_Cand2Sz[local_warp_id];
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
      // if (lane_id == 0) printf("state: %d, size: %d, plexsz: %d, cand1sz: %d, cand2sz: %d, exclsz: %d\n", state[sz-1], sz, sh_PlexSz[local_warp_id], sh_Cand1Sz[local_warp_id], sh_Cand2Sz[local_warp_id], sh_ExclSz[local_warp_id]);      
      // return;  
      if(lane_id == 0)
        {
          state[sz-1] = 2;

          u = v2delete[sz-1];
          cand2[sh_Cand2Sz[local_warp_id]++] = u;
        }
        u = v2delete[sz-1];
        found_idx = -1;
        for (int base = 0; base < sh_ExclSz[local_warp_id]; base += 32)
        {
          int idx = base + lane_id;

          bool match = (idx < sh_ExclSz[local_warp_id]) && (excl[idx] == u);
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
          int last = --sh_ExclSz[local_warp_id];
          int temp = excl[last];
          excl[last] = excl[found_idx];
          excl[found_idx] = temp;
        }
        __syncwarp();  
        
        continue;
      case 2:
      // if (lane_id == 0) printf("state: %d, size: %d, plexsz: %d, cand1sz: %d, cand2sz: %d, exclsz: %d\n", state[sz-1], sz, sh_PlexSz[local_warp_id], sh_Cand1Sz[local_warp_id], sh_Cand2Sz[local_warp_id], sh_ExclSz[local_warp_id]);
        if (br[sz-1] < res[sz-1])
        {
          if (lane_id == 0)
          {
            u = cand2[--sh_Cand2Sz[local_warp_id]];
            plex[sh_PlexSz[local_warp_id]++] = u;
          }
          __syncwarp();
          unsigned int node = plex[sh_PlexSz[local_warp_id]-1];
          for (int i = lane_id; i < degreeBase[node]; i+=32)
          {
            const int nei = neighborsBase[offsetsBase[node]+i];
            neiInPBase[nei]++;
          }
          
          addG(lane_id, node, neiInGBase, n, neighborsBase, offsetsBase, degreeBase);
          // printf("lane_id: %d, node: %d\n", lane_id, node);
          updateCand13(lane_id, cand1, commonMtxBase, recCand1Base, neiInGBase, sz, n, node, &sh_Cand1Sz[local_warp_id], neighborsBase, degreeBase, offsetsBase);

          const uint8_t* __restrict__ row = commonMtxBase + (size_t) node * n;
          updateCand23(lane_id, cand2, row, recCand2Base, sz, &sh_Cand2Sz[local_warp_id]);           

          if (sh_Cand2Sz[local_warp_id])
          {
            if (lane_id == 0)
            {
              u = cand2[--sh_Cand2Sz[local_warp_id]];
              excl[sh_ExclSz[local_warp_id]++] = u;
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
      // if (lane_id == 0) printf("state: %d, size: %d, plexsz: %d, cand1sz: %d, cand2sz: %d, exclsz: %d\n", state[sz-1], sz, sh_PlexSz[local_warp_id], sh_Cand1Sz[local_warp_id], sh_Cand2Sz[local_warp_id], sh_ExclSz[local_warp_id]);
        if (lane_id == 0)
        {
          br[sz-1]++;
          state[sz-1] = 2;

          u = v2delete[sz-1];
          cand2[sh_Cand2Sz[local_warp_id]++] = u;
        }
        u = v2delete[sz-1];
        found_idx = -1;
        for (int base = 0; base < sh_ExclSz[local_warp_id]; base += 32)
        {
          int idx = base + lane_id;

          bool match = (idx < sh_ExclSz[local_warp_id]) && (excl[idx] == u);
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
          int last = --sh_ExclSz[local_warp_id];
          int temp = excl[last];
          excl[last] = excl[found_idx];
          excl[found_idx] = temp;
        }
        __syncwarp();
        continue;
      case 4:
      // if (lane_id == 0) printf("state: %d, size: %d, plexsz: %d, cand1sz: %d, cand2sz: %d, exclsz: %d, br: %d, res: %d\n", state[sz-1], sz, sh_PlexSz[local_warp_id], sh_Cand1Sz[local_warp_id], sh_Cand2Sz[local_warp_id], sh_ExclSz[local_warp_id], br[sz-1], res[sz-1]); 
      if (br[sz-1] == res[sz-1])
      {
        // if (lane_id == 0) printf("abort flag: %d\n", flag);
        if (!flag)
        { 
          if (lane_id == 0)
          {
            u = cand2[--sh_Cand2Sz[local_warp_id]];
            plex[sh_PlexSz[local_warp_id]++] = u;
          }
          __syncwarp();
          unsigned int node = plex[sh_PlexSz[local_warp_id]-1];
          for (int i = lane_id; i < degreeBase[node]; i+=32)
          {
            const int nei = neighborsBase[offsetsBase[node]+i];
            neiInPBase[nei]++;
          }
          addG(lane_id, node, neiInGBase, n, neighborsBase, offsetsBase, degreeBase);
          updateCand13(lane_id, cand1, commonMtxBase, recCand1Base, neiInGBase, sz, n, node, &sh_Cand1Sz[local_warp_id], neighborsBase, degreeBase, offsetsBase);
          if (sh_PlexSz[local_warp_id] + sh_Cand1Sz[local_warp_id] < q)
          {
            if (lane_id == 0)
            {
              state[sz-1] = 5;
            }
            __syncwarp();
            continue;
          }
          if (sh_PlexSz[local_warp_id] > 1 && !upperBoundK2(lane_id, k, q, plex, neiInGBase, sh_PlexSz[local_warp_id]))
          {
            if (lane_id == 0)
            {
              state[sz-1] = 5;
            }
            __syncwarp();
            continue;
          }
          

          int len = 0;
          for (int i = 0; i < sh_Cand1Sz[local_warp_id]; i++)
          {
            const int v = cand1[i];
            if (!isKplex2(lane_id, v, k, sh_PlexSz[local_warp_id], neiInPBase, plex, n, neighborsBase, offsetsBase, degreeBase, adjList))
            {
              if (lane_id == 0)
              {
                int temp = cand1[sh_Cand1Sz[local_warp_id]-1];
                cand1[sh_Cand1Sz[local_warp_id]-1] = cand1[i];
                cand1[i] = temp;
                sh_Cand1Sz[local_warp_id]--;
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
            // printf("Appending a new task, warp_id: %d\n", warp_id);
            unsigned int* tail = t.d_tail_A;
            pos = atomicAdd(&tail[0], 1u);
            // printf("position: %d, warp_id: %d\n", pos, warp_id);
            state[sz-1] = 5;
          }
          pos = __shfl_sync(0xFFFFFFFF, pos, 0);
          // if (lane_id == 0) printf("pos: %d, max_cap: %d\n", pos, MAX_CAP);
          
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
          for (int i = lane_id; i < sh_PlexSz[local_warp_id]; i+=32)
          {
            const int v = plex[i];
            newLabels[v] = P;
          }
          for (int i = lane_id; i < sh_Cand1Sz[local_warp_id]; i+=32)
          {
            const int v = cand1[i];
            newLabels[v] = C;
          }
          for (int i = lane_id; i < sh_Cand2Sz[local_warp_id]; i+=32)
          {
            const int v = cand2[i];
            newLabels[v] = H;
          }
          for (int i = lane_id; i < sh_ExclSz[local_warp_id]; i+=32)
          {
            const int v = excl[i];
            newLabels[v] = X;
          }
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
            nt.PlexSz = sh_PlexSz[local_warp_id];
            nt.CandSz = sh_Cand1Sz[local_warp_id];
            nt.ExclSz = sh_ExclSz[local_warp_id];
            nt.labels = newLabels;
            nt.neiInG = newNeiInG;
            nt.neiInP = newNeiInP;
          }
          __syncwarp();

          if (pos+1 > (MAX_CAP/64)-WARPS)
          {
            // if(lane_id == 0) printf("Maximum Capacity Reached in kSearch\n");
            // atomicExch(abort_flag, 1);
            if (lane_id == 0) d_abort[0] = 1;
            size[0] = sz;
            state[sz-1] = 4;
            return;
          }
        }
        
        flag = 0;
        // if (lane_id == 0) printf("flag: %d, length: %d\n", flag, length[0]);
          for (int i = 0; i < length[0]; i++)
          {
            if (lane_id == 0) sh_Cand1Sz[local_warp_id]++;
            __syncwarp();
            const int v = cand1[sh_Cand1Sz[local_warp_id]-1];
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
      // if (lane_id == 0) printf("state: %d, size: %d, plexsz: %d, cand1sz: %d, cand2sz: %d, exclsz: %d\n", state[sz-1], sz, sh_PlexSz[local_warp_id], sh_Cand1Sz[local_warp_id], sh_Cand2Sz[local_warp_id], sh_ExclSz[local_warp_id]);
      for (int i = br[sz-1]; i >= 1; i--)
        {
          unsigned int node = plex[sh_PlexSz[local_warp_id]-1];
          if (lane_id == 0)
          {
            cand2[sh_Cand2Sz[local_warp_id]++] = plex[--sh_PlexSz[local_warp_id]];
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
        
        recoverCand12(lane_id, cand1, recCand1Base, neiInGBase, sz, n, &sh_Cand1Sz[local_warp_id], neighborsBase, degreeBase, offsetsBase);
        
        recoverCand23(lane_id, n, cand2, recCand2Base, sz, &sh_Cand2Sz[local_warp_id]);
        __syncwarp();
        
        sz--;
        continue;
    }
  }
}

__global__ void kSearch3(int idx, P_pointers p, S_pointers s, G_pointers g, T_pointers t, unsigned int* d_left,  unsigned int* d_blk_counter, unsigned int* d_left_counter, unsigned int* d_res, unsigned int* d_br, unsigned int* d_state, unsigned int* d_len, unsigned int* d_sz, uint16_t* neiInG, uint16_t* neiInP, unsigned int* plex_count, uint8_t* commonMtx, unsigned int* recCand1, unsigned int* recCand2, unsigned int* recExcl, unsigned int* recCand, unsigned int* d_v2delete, uint32_t* d_adj, uint16_t* d_sat, uint16_t* d_commons, uint32_t* d_uni, unsigned int *global_count)
{
  unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int warp_id = (global_index / 32);
  unsigned int lane_id = threadIdx.x % 32;
  unsigned int local_warp_id = threadIdx.x >> 5; 

  // if (warp_id != 10) return;

  if ((warp_id+WARPS*idx) >= (g.n-p.lb+2)) 
  {
    if (lane_id == 0) atomicAdd(&global_count[0], 1);
    return;
  }

  // if ((warp_id+WARPS*idx) == 4233 || (warp_id+WARPS*idx) == 4228 || (warp_id+WARPS*idx) == 4229 || (warp_id+WARPS*idx) ==  4234 || (warp_id+WARPS*idx) == 4242)
  // {
  //   if (lane_id == 0) atomicAdd(&global_count[0], 1);
  //   return;
  // }

  // if ((warp_id+WARPS*idx) == 4243 || (warp_id+WARPS*idx) == 4255 || (warp_id+WARPS*idx) == 4190 || (warp_id+WARPS*idx) == 4189 || (warp_id+WARPS*idx) == 4193)
  // {
  //   if (lane_id == 0) atomicAdd(&global_count[0], 1);
  //   return;
  // }

  // if ((warp_id+WARPS*idx) == 4172 || (warp_id+WARPS*idx) == 4153 || (warp_id+WARPS*idx) == 4237 || (warp_id+WARPS*idx) == 4250 || (warp_id+WARPS*idx) == 4244)
  // {
  //   if (lane_id == 0) atomicAdd(&global_count[0], 1);
  //   return;
  // }

  // if ((warp_id+WARPS*idx) == 4254 || (warp_id+WARPS*idx) == 4248 || (warp_id+WARPS*idx) == 4258 || (warp_id+WARPS*idx) == 4240 || (warp_id+WARPS*idx) == 4253)
  // {
  //   if (lane_id == 0) atomicAdd(&global_count[0], 1);
  //   return;
  // }

  // if (warp_id+WARPS*idx != 4111) return;

  int k = p.k;
  int q = p.lb;

  unsigned int* counterBase = d_blk_counter + warp_id;

  if (counterBase[0] < q) 
  {
    if (lane_id == 0) atomicAdd(&global_count[0], 1);
    return;
  }

  unsigned int* degreeBase = s.degree + warp_id * MAX_BLK_SIZE;
  unsigned int* offsetsBase = s.offsets + warp_id * MAX_BLK_SIZE;
  unsigned int* neighborsBase = s.neighbors + warp_id * MAX_BLK_SIZE * AVG_DEGREE;
  unsigned int* degreeHop = s.degreeHop + warp_id * MAX_BLK_SIZE;

  unsigned int* leftBase = d_left + warp_id * MAX_BLK_SIZE;
  unsigned int* left_count = d_left_counter + warp_id;
  unsigned int* l_degreeBase = s.l_degree + warp_id * MAX_BLK_SIZE;
  unsigned int* l_offsetsBase = s.l_offsets + warp_id * MAX_BLK_SIZE;
  unsigned int* l_neighborsBase = s.l_neighbors + warp_id * MAX_BLK_SIZE * AVG_LEFT_DEGREE;

  uint16_t* local_sat = d_sat + warp_id * MAX_BLK_SIZE;
  uint16_t* local_commons = d_commons + warp_id * MAX_BLK_SIZE;
  uint32_t* local_uni = d_uni + warp_id * 32;

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
  unsigned int* recExclBase = recExcl + warp_id * MAX_BLK_SIZE;
  unsigned int* recCandBase = recCand + warp_id * MAX_BLK_SIZE;

  // t.d_tail_A[0] = 0;

  // uint8_t* commonMtxBase = commonMtx + warp_id * CAP;
  size_t capacity = size_t(warp_id) * CAP;
  uint8_t* commonMtxBase = commonMtx + capacity;

  uint32_t *adjList = d_adj + ADJSIZE * warp_id;

  int n;
  if (lane_id == 0)
  {
    n = counterBase[0];
  }
  n = __shfl_sync(0xFFFFFFFF, n, 0);

  // __shared__ unsigned int sh_PlexSz[WARPS_EACH_BLK];
  // __shared__ unsigned int sh_Cand1Sz[WARPS_EACH_BLK];
  // __shared__ unsigned int sh_Cand2Sz[WARPS_EACH_BLK];
  // __shared__ unsigned int sh_ExclSz[WARPS_EACH_BLK];

  // if (lane_id == 0) sh_PlexSz[local_warp_id] = PlexSz[0]; 
  // if (lane_id == 0) sh_Cand1Sz[local_warp_id] = Cand1Sz[0];
  // if (lane_id == 0) sh_Cand2Sz[local_warp_id] = Cand2Sz[0];
  // if (lane_id == 0) sh_ExclSz[local_warp_id] = ExclSz[0];

  for (int i = lane_id; i < n; i+=32)
  {
    neiInGBase[i] = degreeHop[i];
    neiInPBase[i] = 0;
  }
  
  
  __syncwarp();
  for (int i = lane_id; i < degreeBase[0]; i+=32)
  {
    const int nei = neighborsBase[offsetsBase[0]+i];
    neiInPBase[nei]++;
  }

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
  // int flag = abort_flag[0];
  int sz;
  // if (!abort_flag[0])
  // {
    sz = 0;
    res[sz] = k - 1;
    br[sz] = 1;
    state[sz] = 0;
    sz++;
  // }
  // else 
  // {
  //   sz = size[0];
  // }
  
  int u, found_idx;
  int count = 0;
  while(sz)
  {
    if (sz >= MAX_DEPTH)
    {
      if (lane_id == 0) printf("capacity crossed: %d\n", sz);
      break;
    }
    // count = count + 1;
    // if (count > 200) return;
    switch(state[sz-1])
    {
      //reserve my slot, slot -> []
      case 0:
        if (lane_id == 0) 
        {
        // printf("state: %d, size: %d, plexsz: %d, cand1sz: %d, cand2sz: %d, exclsz: %d, maximal k-plexes: %d\n", state[sz-1], sz, PlexSz[0], Cand1Sz[0], Cand2Sz[0], ExclSz[0], plex_count[0]);
        // printf("plex: ");
        // for (int i = 0 ; i < PlexSz[0]; i++)
        // {
        //   printf("%d ", plex[i]);
        // }
        // printf("\n");
        // printf("cand1: ");
        // for (int i = 0 ; i < Cand1Sz[0]; i++)
        // {
        //   printf("%d ", cand1[i]);
        // }
        // printf("\n");
        // printf("cand2: ");
        // for (int i = 0 ; i < Cand2Sz[0]; i++)
        // {
        //   printf("%d ", cand2[i]);
        // }
        // printf("\n");
        // printf("excl: ");
        // for (int i = 0 ; i < ExclSz[0]; i++)
        // {
        //   printf("%d ", excl[i]);
        // }
        // printf("\n");
        // printf("\n");
        }
        if (Cand2Sz[0] == 0)
        {
          // if (!flag)
          // {
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
          
          // int pos;
          // if (lane_id == 0)
          // {
          //   unsigned int* tail = t.d_tail_A;
          //   pos = atomicAdd(&tail[0], 1u);
          // }
          // pos = __shfl_sync(0xFFFFFFFF, pos, 0);
          // // if (lane_id == 0) printf("pos: %d, max_cap: %d\n", pos, MAX_CAP);
          

        //   uint8_t* newLabels = t.d_all_labels_A + pos * MAX_BLK_SIZE;
        //   uint16_t* newNeiInG = t.d_all_neiInG_A + pos * MAX_BLK_SIZE;
        //   uint16_t* newNeiInP = t.d_all_neiInP_A + pos * MAX_BLK_SIZE;
        //   for (int i = lane_id; i < n; i+=32)
        //   {
        //     newLabels[i] = U;
        //     newNeiInG[i] = neiInGBase[i];
        //     newNeiInP[i] = neiInPBase[i];
        //   }
        //   for (int i = lane_id; i < PlexSz[0]; i+=32)
        //   {
        //     const int v = plex[i];
        //     newLabels[v] = P;
        //   }
        //   for (int i = lane_id; i < Cand1Sz[0]; i+=32)
        //   {
        //     const int v = cand1[i];
        //     newLabels[v] = C;
        //   }
        //   for (int i = lane_id; i < Cand2Sz[0]; i+=32)
        //   {
        //     const int v = cand2[i];
        //     newLabels[v] = H;
        //   }
        //   for (int i = lane_id; i < ExclSz[0]; i+=32)
        //   {
        //     const int v = excl[i];
        //     newLabels[v] = X;
        //   }

          
        //   __syncwarp();
        //   if (lane_id == 0)
        //   {
        //     Task &nt = t.d_tasks_A[pos];
        //     nt.idx = warp_id;
        //     nt.PlexSz = PlexSz[0];
        //     nt.CandSz = Cand1Sz[0];
        //     nt.ExclSz = ExclSz[0];
        //     nt.labels = newLabels;
        //     nt.neiInG = newNeiInG;
        //     nt.neiInP = newNeiInP;
          
        //   // __syncwarp();
        //   // if (warp_id == 0)
        //   // {
        //   //   // printf("idx: %d\n", t.idx);
        //   //   // if (t.idx == 0) return;
        //   //   // printf("Labels1: %d\n", newLabels[0]);
        //   //   for (int j = 0; j < MAX_BLK_SIZE; j++)
        //   //   {
        //   //     printf("%d ", nt.labels[j]);
        //   //   }
        //   //   printf("\n");
        //   // }
        // }
        // __syncwarp();
        // if (pos+1 > (MAX_CAP/4)-WARPS)
        // {
        //   // if(lane_id == 0) printf("Maximum Capacity Reached in kSearch\n");
        //   // atomicExch(abort_flag, 1);
        //   if (lane_id == 0) abort_flag[0] = 1;
        //   size[0] = sz;
        //   //  if (lane_id == 0)
        //   // {
        //   //   printf("State0: ");
        //   //   for (int i = 0; i < sz; i++)
        //   //   {
        //   //     printf("%d ", state[i]);
        //   //   }
        //   //   printf("\n");
        //   // }
        //   state[sz-1] = 0;
        //   // if (warp_id == 13 && lane_id == 0) printf("%d is returning with sz: %d, state:%d, res: %d, br:%d, plexsz: %d, cand1sz: %d, cand2sz: %d, exclsz: %d\n", warp_id, sz, state[sz-1], res[sz-1], br[sz-1],PlexSz[0], Cand1Sz[0], Cand2Sz[0], ExclSz[0]);
        //   return;
        // }
      // }
          // if (lane_id == 0) abort_flag[0] = 0;
          // flag = 0;
          if (lane_id == 0) state[sz-1] = 7;
          __syncwarp();
          // sz--;
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
        if (lane_id == 0) 
        {
        // printf("state: %d, size: %d, plexsz: %d, cand1sz: %d, cand2sz: %d, exclsz: %d, maximal k-plexes: %d\n", state[sz-1], sz, PlexSz[0], Cand1Sz[0], Cand2Sz[0], ExclSz[0], plex_count[0]);
        // printf("plex: ");
        // for (int i = 0 ; i < PlexSz[0]; i++)
        // {
        //   printf("%d ", plex[i]);
        // }
        // printf("\n");
        // printf("cand1: ");
        // for (int i = 0 ; i < Cand1Sz[0]; i++)
        // {
        //   printf("%d ", cand1[i]);
        // }
        // printf("\n");
        // printf("cand2: ");
        // for (int i = 0 ; i < Cand2Sz[0]; i++)
        // {
        //   printf("%d ", cand2[i]);
        // }
        // printf("\n");
        // printf("excl: ");
        // for (int i = 0 ; i < ExclSz[0]; i++)
        // {
        //   printf("%d ", excl[i]);
        // }
        // printf("\n");
        // printf("\n");
        }
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
      if (lane_id == 0) 
        {
        // printf("state: %d, size: %d, plexsz: %d, cand1sz: %d, cand2sz: %d, exclsz: %d, maximal k-plexes: %d\n", state[sz-1], sz, PlexSz[0], Cand1Sz[0], Cand2Sz[0], ExclSz[0], plex_count[0]);
        // printf("plex: ");
        // for (int i = 0 ; i < PlexSz[0]; i++)
        // {
        //   printf("%d ", plex[i]);
        // }
        // printf("\n");
        // printf("cand1: ");
        // for (int i = 0 ; i < Cand1Sz[0]; i++)
        // {
        //   printf("%d ", cand1[i]);
        // }
        // printf("\n");
        // printf("cand2: ");
        // for (int i = 0 ; i < Cand2Sz[0]; i++)
        // {
        //   printf("%d ", cand2[i]);
        // }
        // printf("\n");
        // printf("excl: ");
        // for (int i = 0 ; i < ExclSz[0]; i++)
        // {
        //   printf("%d ", excl[i]);
        // }
        // printf("\n");
        // printf("\n");
        }
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
      if (lane_id == 0) 
        {
        // printf("state: %d, size: %d, plexsz: %d, cand1sz: %d, cand2sz: %d, exclsz: %d, maximal k-plexes: %d\n", state[sz-1], sz, PlexSz[0], Cand1Sz[0], Cand2Sz[0], ExclSz[0], plex_count[0]);
        // printf("plex: ");
        // for (int i = 0 ; i < PlexSz[0]; i++)
        // {
        //   printf("%d ", plex[i]);
        // }
        // printf("\n");
        // printf("cand1: ");
        // for (int i = 0 ; i < Cand1Sz[0]; i++)
        // {
        //   printf("%d ", cand1[i]);
        // }
        // printf("\n");
        // printf("cand2: ");
        // for (int i = 0 ; i < Cand2Sz[0]; i++)
        // {
        //   printf("%d ", cand2[i]);
        // }
        // printf("\n");
        // printf("excl: ");
        // for (int i = 0 ; i < ExclSz[0]; i++)
        // {
        //   printf("%d ", excl[i]);
        // }
        // printf("\n");
        // printf("\n");
        }
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
      if (lane_id == 0) 
        {
        // printf("state: %d, size: %d, plexsz: %d, cand1sz: %d, cand2sz: %d, exclsz: %d, maximal k-plexes: %d\n", state[sz-1], sz, PlexSz[0], Cand1Sz[0], Cand2Sz[0], ExclSz[0], plex_count[0]);
        // printf("plex: ");
        // for (int i = 0 ; i < PlexSz[0]; i++)
        // {
        //   printf("%d ", plex[i]);
        // }
        // printf("\n");
        // printf("cand1: ");
        // for (int i = 0 ; i < Cand1Sz[0]; i++)
        // {
        //   printf("%d ", cand1[i]);
        // }
        // printf("\n");
        // printf("cand2: ");
        // for (int i = 0 ; i < Cand2Sz[0]; i++)
        // {
        //   printf("%d ", cand2[i]);
        // }
        // printf("\n");
        // printf("excl: ");
        // for (int i = 0 ; i < ExclSz[0]; i++)
        // {
        //   printf("%d ", excl[i]);
        // }
        // printf("\n");
        // printf("\n");
        }
      if (br[sz-1] == res[sz-1])
      {
      //   if (!flag)
      // { 
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
          
          // int pos;
          // if (lane_id == 0)
          // {
          //   unsigned int* tail = t.d_tail_A;
          //   pos = atomicAdd(&tail[0], 1u);
          //   state[sz-1] = 5;
          // }
          // pos = __shfl_sync(0xFFFFFFFF, pos, 0);
          // // if (lane_id == 0) printf("pos: %d, max_cap: %d\n", pos, MAX_CAP);
          
          // size_t baseOff = (size_t)pos * (size_t)MAX_BLK_SIZE;
          // uint8_t* newLabels = t.d_all_labels_A + baseOff;
          // uint16_t* newNeiInG = t.d_all_neiInG_A + baseOff;
          // uint16_t* newNeiInP = t.d_all_neiInP_A + baseOff;
          // for (int i = lane_id; i < n; i+=32)
          // {
          //   newLabels[i] = U;
          //   newNeiInG[i] = neiInGBase[i];
          //   int temp = neiInPBase[i];
          //   newNeiInP[i] = temp;
          // }
          // for (int i = lane_id; i < PlexSz[0]; i+=32)
          // {
          //   const int v = plex[i];
          //   newLabels[v] = P;
          // }
          // for (int i = lane_id; i < Cand1Sz[0]; i+=32)
          // {
          //   const int v = cand1[i];
          //   newLabels[v] = C;
          // }
          // for (int i = lane_id; i < Cand2Sz[0]; i+=32)
          // {
          //   const int v = cand2[i];
          //   newLabels[v] = H;
          // }
          // for (int i = lane_id; i < ExclSz[0]; i+=32)
          // {
          //   const int v = excl[i];
          //   newLabels[v] = X;
          // }
          // __syncwarp();
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
          // if (lane_id == 0)
          // {
          //   Task &nt = t.d_tasks_A[pos];
          //   nt.idx = warp_id;
          //   nt.PlexSz = PlexSz[0];
          //   nt.CandSz = Cand1Sz[0];
          //   nt.ExclSz = ExclSz[0];
          //   nt.labels = newLabels;
          //   nt.neiInG = newNeiInG;
          //   nt.neiInP = newNeiInP;
          // }
          // __syncwarp();

          // if (pos+1 > (MAX_CAP/4)-WARPS)
          // {
          //   // if(lane_id == 0) printf("Maximum Capacity Reached in kSearch\n");
          //   // atomicExch(abort_flag, 1);
          //   if (lane_id == 0) abort_flag[0] = 1;
          //   size[0] = sz;
          //   state[sz-1] = 4;
          //   // if (warp_id == 13 && lane_id == 0) printf("%d is returning with sz: %d, state:%d, res: %d, br:%d, plexsz: %d, cand1sz: %d, cand2sz: %d, exclsz: %d, len: %d\n", warp_id, sz, state[sz-1], res[sz-1], br[sz-1],PlexSz[0], Cand1Sz[0], Cand2Sz[0], ExclSz[0], length[0]);
          //   return;
          // }
        // }
        // if (lane_id == 0) abort_flag[0] = 0;
        // flag = 0;
        if (lane_id == 0)
        {
          state[sz] = 7;
          state[sz-1] = 6;
        }
        __syncwarp();
        sz++;
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
      if (lane_id == 0) 
        {
        // printf("state: %d, size: %d, plexsz: %d, cand1sz: %d, cand2sz: %d, exclsz: %d, maximal k-plexes: %d\n", state[sz-1], sz, PlexSz[0], Cand1Sz[0], Cand2Sz[0], ExclSz[0], plex_count[0]);
        // printf("plex: ");
        // for (int i = 0 ; i < PlexSz[0]; i++)
        // {
        //   printf("%d ", plex[i]);
        // }
        // printf("\n");
        // printf("cand1: ");
        // for (int i = 0 ; i < Cand1Sz[0]; i++)
        // {
        //   printf("%d ", cand1[i]);
        // }
        // printf("\n");
        // printf("cand2: ");
        // for (int i = 0 ; i < Cand2Sz[0]; i++)
        // {
        //   printf("%d ", cand2[i]);
        // }
        // printf("\n");
        // printf("excl: ");
        // for (int i = 0 ; i < ExclSz[0]; i++)
        // {
        //   printf("%d ", excl[i]);
        // }
        // printf("\n");
        // printf("\n");
        }
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
      
      case 6:
      if (lane_id == 0) 
        {
        // printf("state: %d, size: %d, plexsz: %d, cand1sz: %d, cand2sz: %d, exclsz: %d, maximal k-plexes: %d\n", state[sz-1], sz, PlexSz[0], Cand1Sz[0], Cand2Sz[0], ExclSz[0], plex_count[0]);
        // printf("plex: ");
        // for (int i = 0 ; i < PlexSz[0]; i++)
        // {
        //   printf("%d ", plex[i]);
        // }
        // printf("\n");
        // printf("cand1: ");
        // for (int i = 0 ; i < Cand1Sz[0]; i++)
        // {
        //   printf("%d ", cand1[i]);
        // }
        // printf("\n");
        // printf("cand2: ");
        // for (int i = 0 ; i < Cand2Sz[0]; i++)
        // {
        //   printf("%d ", cand2[i]);
        // }
        // printf("\n");
        // printf("excl: ");
        // for (int i = 0 ; i < ExclSz[0]; i++)
        // {
        //   printf("%d ", excl[i]);
        // }
        // printf("\n");
        // printf("\n");
        }
        for (int i = 0; i < length[0]; i++)
          {
            if (lane_id == 0) Cand1Sz[0]++;
            __syncwarp();
            const int v = cand1[Cand1Sz[0]-1];
            addG(lane_id, v, neiInGBase, n, neighborsBase, offsetsBase, degreeBase);
          }
          if (lane_id == 0) state[sz-1] = 5;
          __syncwarp();
          continue;
      case 7:
      if (lane_id == 0) 
        {
        // printf("state: %d, size: %d, plexsz: %d, cand1sz: %d, cand2sz: %d, exclsz: %d, maximal k-plexes: %d\n", state[sz-1], sz, PlexSz[0], Cand1Sz[0], Cand2Sz[0], ExclSz[0], plex_count[0]);
        // printf("plex: ");
        // for (int i = 0 ; i < PlexSz[0]; i++)
        // {
        //   printf("%d ", plex[i]);
        // }
        // printf("\n");
        // printf("cand1: ");
        // for (int i = 0 ; i < Cand1Sz[0]; i++)
        // {
        //   printf("%d ", cand1[i]);
        // }
        // printf("\n");
        // printf("cand2: ");
        // for (int i = 0 ; i < Cand2Sz[0]; i++)
        // {
        //   printf("%d ", cand2[i]);
        // }
        // printf("\n");
        // printf("excl: ");
        // for (int i = 0 ; i < ExclSz[0]; i++)
        // {
        //   printf("%d ", excl[i]);
        // }
        // printf("\n");
        // printf("\n");
        }
          // if (lane_id == 0) printf("Hello from list branch\n");
          if (PlexSz[0] + Cand1Sz[0] < q)
          {
            sz--;
            continue;
          }
          if (Cand1Sz[0] == 0)
          {
            if (ExclSz[0] == 0 &&
                PlexSz[0] >= q &&
                isMaximal_opt(lane_id, k, PlexSz[0], leftBase, left_count[0], l_neighborsBase, l_offsetsBase, l_degreeBase, neiInPBase, neighborsBase, offsetsBase, degreeBase, plex, n, local_sat, local_commons, local_uni))
                {
                  if (lane_id == 0) atomicAdd(&plex_count[0], 1);
                }
                sz--;
                continue;
          }
          __syncwarp();

          int minnei_Plex = INT_MAX;
          int pivot = -1;
          int minnei_Cand = INT_MAX;

          for(int i = lane_id; i < PlexSz[0]; i+=32)
          {
            const int v = plex[i];
            if (neiInGBase[v] < minnei_Plex)
            {
              minnei_Plex = neiInGBase[v];
              pivot = v;
            }
          }

          for(int offset = 16; offset > 0; offset >>= 1)
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
          
          if (minnei_Plex + k < max(q, PlexSz[0]))
          {
            sz--;
            continue;
          }
          if (minnei_Plex + k < PlexSz[0] + Cand1Sz[0])
          {
            minnei_Cand = INT_MAX;
            pivot = -1;
            
            for (int i = lane_id; i < Cand1Sz[0]; i+=32)
            {
              const int v = cand1[i];
              int check = v * n + pivot_plex;
              if (!((adjList[check >> 5] >> (check & 31)) &1u ))
              {
                if (neiInGBase[v] < minnei_Cand)
                {
                  minnei_Cand = neiInGBase[v];
                  pivot = v;
                }
                else if (neiInGBase[v] == minnei_Cand && neiInPBase[pivot] > neiInPBase[v])
                {
                  pivot = v;
                }
              }
            }
            
            for (int offset = 16; offset > 0; offset >>= 1)
            {
              int otherMin = __shfl_down_sync(0xFFFFFFFF, minnei_Cand, offset);
              int otherIdx = __shfl_down_sync(0xFFFFFFFF, pivot, offset);
              if (otherMin < minnei_Cand || (otherMin == minnei_Cand && otherIdx != -1 && neiInPBase[pivot] > neiInPBase[otherIdx]))
              {
                minnei_Cand = otherMin;
                pivot = otherIdx;
              }
            }
            minnei_Cand = __shfl_sync(0xFFFFFFFF, minnei_Cand, 0);
            pivot = __shfl_sync(0xFFFFFFFF, pivot, 0);
            // if (lane_id == 0) printf("pivot: %d\n", pivot);
            // if (lane_id == 0) printf("pivot plex: %d\n", pivot_plex);
            if (pivot == -1)
            {
              pivot = cand1[Cand1Sz[0]-1];
            }
            if (lane_id == 0) state[sz-1] = 8;
            __syncwarp();
            continue; 
          }
          int minnei = minnei_Plex;

          for (int i = lane_id; i < Cand1Sz[0]; i+=32)
          {
            const int v = cand1[i];
            if (neiInGBase[v] < minnei)
            {
              minnei = neiInGBase[v];
              pivot = v;
            }
            else if (neiInGBase[v] == minnei && neiInPBase[pivot] > neiInPBase[v])
            {
              pivot = v;
            }
          }

          for (int offset = 16; offset > 0; offset >>= 1)
          {
            int otherMin = __shfl_down_sync(0xFFFFFFFF, minnei, offset);
            int otherIdx = __shfl_down_sync(0xFFFFFFFF, pivot, offset);
            if (otherMin < minnei || (otherMin == minnei && otherIdx != -1 && neiInPBase[pivot] > neiInPBase[otherIdx]))
            {
              minnei = otherMin;
              pivot = otherIdx;
            }
          }
          minnei = __shfl_sync(0xFFFFFFFF, minnei, 0);
          pivot = __shfl_sync(0xFFFFFFFF, pivot, 0);
          
          if(minnei >= (PlexSz[0] + Cand1Sz[0] - k))
          {
            if (PlexSz[0] + Cand1Sz[0] < q)
            {
              sz--;
              continue;
            }
            bool flag = false;

            for (int i = lane_id; i < ExclSz[0]; i+=32)
            {
              const int v = excl[i];
              if (isKplexPC2(v, k, PlexSz[0]+Cand1Sz[0], PlexSz[0], Cand1Sz[0], neiInGBase, plex, cand1, n, neighborsBase, offsetsBase, degreeBase, adjList))
              {
                flag = true;
              }
            }

            if (__any_sync(0xFFFFFFFF, flag))
            {
              sz--;
              continue;
            }

            if (isMaximalPC_opt(lane_id, k, PlexSz[0], Cand1Sz[0], PlexSz[0] + Cand1Sz[0], leftBase, left_count[0], l_neighborsBase, l_offsetsBase, l_degreeBase, neiInGBase, neighborsBase, offsetsBase, degreeBase, plex, cand1, n, local_sat, local_commons, local_uni))
            {
              if (lane_id == 0) atomicAdd(&plex_count[0], 1);
            }
            sz--;
            continue;
          }
          if (lane_id == 0) state[sz-1] = 8;
          __syncwarp();
          continue;
      case 8:
      if (lane_id == 0) 
        {
        // printf("state: %d, size: %d, plexsz: %d, cand1sz: %d, cand2sz: %d, exclsz: %d\n", state[sz-1], sz, PlexSz[0], Cand1Sz[0], Cand2Sz[0], ExclSz[0]);
        // printf("plex: ");
        // for (int i = 0 ; i < PlexSz[0]; i++)
        // {
        //   printf("%d ", plex[i]);
        // }
        // printf("\n");
        // printf("cand1: ");
        // for (int i = 0 ; i < Cand1Sz[0]; i++)
        // {
        //   printf("%d ", cand1[i]);
        // }
        // printf("\n");
        // printf("cand2: ");
        // for (int i = 0 ; i < Cand2Sz[0]; i++)
        // {
        //   printf("%d ", cand2[i]);
        // }
        // printf("\n");
        // printf("excl: ");
        // for (int i = 0 ; i < ExclSz[0]; i++)
        // {
        //   printf("%d ", excl[i]);
        // }
        // printf("\n");
        // printf("\n");
        }
          if (lane_id == 0)
          {
            excl[ExclSz[0]++] = pivot;
            for (int i = 0; i < Cand1Sz[0]; i++)
            {
              if (cand1[i] == pivot)
              {
                int temp = cand1[i];
                cand1[i] = cand1[Cand1Sz[0]-1];
                cand1[Cand1Sz[0]-1] = temp;
                Cand1Sz[0]--;
                break;
              }
            }
            res[sz-1] = pivot;
          }
          __syncwarp();
          subG(lane_id, pivot, neiInGBase, n, neighborsBase, offsetsBase, degreeBase);

          // int read = 0;
          // int write = 0;
          // int size = Cand1Sz[0];

          // while (read < size)
          // {
          //   const int take = min(32, size - read);
          //   const bool active = (lane_id < take);

          //   unsigned int v = 0;
          //   if (active) v = cand1[read+lane_id];
            
          //   const bool is_pivot = active && (v == pivot);
          //   const bool keep = active && !is_pivot;

          //   unsigned int mask = __activemask();
          //   unsigned int keepmask = __ballot_sync(mask, keep);

          //   const int keep_rank = __popc(keepmask & ((1u << lane_id) - 1));
          //   const int num_keep = __popc(keepmask);

          //   if (active && keep) cand1[write+keep_rank] = v;
            
          //   if (lane_id == 0)
          //   {
          //     read += take;
          //     write += num_keep;
          //   }
          //   read = __shfl_sync(mask, read, 0);
          //   write = __shfl_sync(mask, write, 0);
          // }
          // if (lane_id == 0) Cand1Sz[0] = write;
          // __syncwarp();
          // subG(lane_id, pivot, neiInGBase, n, neighborsBase, offsetsBase, degreeBase);
          if (lane_id == 0)
          {
            state[sz-1] = 9;
            state[sz] = 7;
          }
          __syncwarp();
          sz++;
          continue;
      case 9:
      if (lane_id == 0) 
        {
        // printf("state: %d, size: %d, plexsz: %d, cand1sz: %d, cand2sz: %d, exclsz: %d, maximal k-plexes: %d\n", state[sz-1], sz, PlexSz[0], Cand1Sz[0], Cand2Sz[0], ExclSz[0], plex_count[0]);
        // printf("plex: ");
        // for (int i = 0 ; i < PlexSz[0]; i++)
        // {
        //   printf("%d ", plex[i]);
        // }
        // printf("\n");
        // printf("cand1: ");
        // for (int i = 0 ; i < Cand1Sz[0]; i++)
        // {
        //   printf("%d ", cand1[i]);
        // }
        // printf("\n");
        // printf("cand2: ");
        // for (int i = 0 ; i < Cand2Sz[0]; i++)
        // {
        //   printf("%d ", cand2[i]);
        // }
        // printf("\n");
        // printf("excl: ");
        // for (int i = 0 ; i < ExclSz[0]; i++)
        // {
        //   printf("%d ", excl[i]);
        // }
        // printf("\n");
        // printf("\n");
        }
          if (lane_id == 0)
          {
          cand1[Cand1Sz[0]++] = res[sz-1];
          // excl[--ExclSz[0]];
          }
          __syncwarp();
          pivot = cand1[Cand1Sz[0]-1];
          for (int i = 0; i < ExclSz[0]; i++)
            {
              if (excl[i] == pivot)
              {
                int temp = excl[i];
                excl[i] = excl[ExclSz[0]-1];
                excl[ExclSz[0]-1] = temp;
                ExclSz[0]--;
                break;
              }
            }
          addG(lane_id, pivot, neiInGBase, n, neighborsBase, offsetsBase, degreeBase);

          if (lane_id == 0)
          {
            plex[PlexSz[0]++] = pivot;
            for (int i = 0; i < Cand1Sz[0]; i++)
            {
              if (cand1[i] == pivot)
              {
                int temp = cand1[i];
                cand1[i] = cand1[Cand1Sz[0]-1];
                cand1[Cand1Sz[0]-1] = temp;
                Cand1Sz[0]--;
                break;
              }
            }
          }
          __syncwarp();
          for (int j = lane_id; j < degreeBase[pivot]; j+=32)
          {
            const int nei = neighborsBase[offsetsBase[pivot]+j];
            neiInPBase[nei]++;
          }
          __syncwarp();

          const uint8_t* row = commonMtxBase + (size_t) pivot * n;

          int read  = 0;
          int write = 0;
          int size = Cand1Sz[0];
          // int total_removed = 0;

          while (read < size)
          {
            const int take = min(32, size - read);
            const bool active = (lane_id < take);
            //const int idx = i + lane_id;

            unsigned int v = 0;
            if (active) v = cand1[read+lane_id];

            const bool keep = active && (isKplex3(v, k, PlexSz[0], neiInPBase, plex, n, neighborsBase, offsetsBase, degreeBase, adjList)) && !(row[v] < UNLINK2EQUAL);

            const unsigned activemask = __ballot_sync(0xFFFFFFFF, active);
            unsigned keepmask = __ballot_sync(0xFFFFFFFF, keep);
            unsigned dropmask = activemask ^ keepmask;

            const int keep_rank = __popc(keepmask & ((1u << lane_id) - 1));
            // const int drop_rank = __popc(dropmask & ((1u << lane_id) - 1));
            const int num_keep  = __popc(keepmask);

            if (active && keep)
            {
              cand1[write + keep_rank] = v;
            }

            while (dropmask)
            {
              const int leader = __ffs(dropmask) - 1;
              const unsigned vdrop = __shfl_sync(0xFFFFFFFF, v, leader);
              recCandBase[vdrop] = (unsigned) (sz-1);
              subG(lane_id, vdrop, neiInGBase, n, neighborsBase, offsetsBase, degreeBase);
              dropmask &= (dropmask - 1);
            }

            if (lane_id == 0)
            {
              read += take;
              write += num_keep;
              // size = newR;
              // total_removed += num_drop;
            }
            read  = __shfl_sync(0xFFFFFFFF, read, 0);
            write = __shfl_sync(0xFFFFFFFF, write, 0);
            // size = __shfl_sync(0xFFFFFFFF, size, 0);
          }
            if (lane_id == 0) Cand1Sz[0] = write;
          __syncwarp();
          // if (lane_id == 0) v2delete[sz-1] = total_removed;

          if (upperBound2(lane_id, k, q, plex, neiInGBase, PlexSz[0]))
          {
            read  = 0;
            write = 0;
            size = ExclSz[0];
            // int total_removed = 0;

            while (read < size)
            {
              const int take = min(32, size - read);
              const bool active = (lane_id < take);
              //const int idx = i + lane_id;

              unsigned int v = 0;
              if (active) v = excl[read+lane_id];

              const bool keep = active && (isKplex3(v, k, PlexSz[0], neiInPBase, plex, n, neighborsBase, offsetsBase, degreeBase, adjList)) && !(row[v] < UNLINK2MORE);

              const unsigned activemask = __ballot_sync(0xFFFFFFFF, active);
              unsigned keepmask = __ballot_sync(0xFFFFFFFF, keep);
              unsigned dropmask = activemask ^ keepmask;

              const int keep_rank = __popc(keepmask & ((1u << lane_id) - 1));
              // const int drop_rank = __popc(dropmask & ((1u << lane_id) - 1));
              const int num_keep  = __popc(keepmask);

              if (active && keep)
              {
                excl[write + keep_rank] = v;
              }

              while (dropmask)
              {
                const int leader = __ffs(dropmask) - 1;
                const unsigned vdrop = __shfl_sync(0xFFFFFFFF, v, leader);
                recExclBase[vdrop] = (unsigned) (sz-1);
                dropmask &= (dropmask - 1);
              }

              if (lane_id == 0)
              {
                read += take;
                write += num_keep;
                // size = newR;
                // total_removed += num_drop;
              }
              read  = __shfl_sync(0xFFFFFFFF, read, 0);
              write = __shfl_sync(0xFFFFFFFF, write, 0);
              // size = __shfl_sync(0xFFFFFFFF, size, 0);
            }
              if (lane_id == 0) ExclSz[0] = write;
            __syncwarp();

            if (lane_id == 0)
            {
              state[sz-1] = 10;
              state[sz] = 7;
            }
            __syncwarp();
            sz++;
            continue;
          }
          // if (lane_id == 0) res[sz-1] = 0;
          if (lane_id == 0) state[sz-1] = 10;
          __syncwarp();
          continue;
      case 10:
      if (lane_id == 0) 
        {
        // printf("state: %d, size: %d, plexsz: %d, cand1sz: %d, cand2sz: %d, exclsz: %d, maximal k-plexes: %d\n", state[sz-1], sz, PlexSz[0], Cand1Sz[0], Cand2Sz[0], ExclSz[0], plex_count[0]);
        // printf("plex: ");
        // for (int i = 0 ; i < PlexSz[0]; i++)
        // {
        //   printf("%d ", plex[i]);
        // }
        // printf("\n");
        // printf("cand1: ");
        // for (int i = 0 ; i < Cand1Sz[0]; i++)
        // {
        //   printf("%d ", cand1[i]);
        // }
        // printf("\n");
        // printf("cand2: ");
        // for (int i = 0 ; i < Cand2Sz[0]; i++)
        // {
        //   printf("%d ", cand2[i]);
        // }
        // printf("\n");
        // printf("excl: ");
        // for (int i = 0 ; i < ExclSz[0]; i++)
        // {
        //   printf("%d ", excl[i]);
        // }
        // printf("\n");
        // printf("\n");
        }
          recoverCand23(lane_id, n, excl, recExclBase, sz, &ExclSz[0]);
          recoverCand12(lane_id, cand1, recCandBase, neiInGBase, sz, n, &Cand1Sz[0], neighborsBase, degreeBase, offsetsBase);

          if (lane_id == 0) cand1[Cand1Sz[0]++] = plex[--PlexSz[0]];
          __syncwarp();
          pivot = cand1[Cand1Sz[0]-1];
          for (int j = lane_id; j < degreeBase[pivot]; j+=32)
          {
            const int nei = neighborsBase[offsetsBase[pivot]+j];
            neiInPBase[nei]--;
          }
          __syncwarp();
          // recoverCand12(lane_id, cand1, recCand1Base, neiInGBase, sz, n, &Cand1Sz[0], neighborsBase, degreeBase, offsetsBase);
          
          sz--;
          continue;
    }
  }
  // if (lane_id == 0) atomicAdd(&global_count[0], 1);
  // if (lane_id == 0) printf("%d warps completed out of 4544 total warps with warp_id: %d\n", global_count[0], warp_id+WARPS*idx);
}

__global__ void BNB2(int i, P_pointers p, S_pointers s, unsigned int* d_blk, unsigned int* d_left, unsigned int* d_blk_counter, unsigned int* d_left_counter, uint8_t* commonMtx, Task* tasks, Task* outTasks, Task* global_tasks, unsigned int N, unsigned int head, unsigned int* tailPtr, unsigned int* global_tail, uint8_t* d_all_labels, uint16_t* d_all_neiInG, uint16_t* d_all_neiInP, uint8_t* global_labels, uint16_t* global_neiInG, uint16_t* global_neiInP, unsigned int* plex_count, uint16_t* d_sat, uint16_t* d_commons, uint32_t* d_uni, unsigned long long* cycles, uint32_t* d_adj, int* d_abort, unsigned int* d_state, unsigned int* d_res, unsigned int* recExcl, unsigned int* recCand)
{
  unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int warp_id = (global_index / 32);
  unsigned int lane_id = threadIdx.x % 32;

  //if (lane_id == 0) printf("N: %d, warp_id: %d, i: %d\n", N, warp_id, i);
  if ((warp_id+WARPS*i) >= N)  return;
  // //if (lane_id == 0) printf("Hi\n");
  int k = p.k;
  int q = p.lb;
  Task t = tasks[warp_id+WARPS*i];
  uint8_t* labelsBase = t.labels;
  unsigned int* blkBase = d_blk + t.idx * MAX_BLK_SIZE;
  unsigned int* leftBase = d_left + t.idx * MAX_BLK_SIZE;
  uint16_t* neiInG = t.neiInG;
  uint16_t* neiInP = t.neiInP;
  unsigned int* left_count = d_left_counter + t.idx;
  unsigned int* local_n = d_blk_counter + t.idx;
  unsigned int PlexSz = t.PlexSz;
  unsigned int CandSz = t.CandSz;
  unsigned int ExclSz = t.ExclSz;
  size_t capacity = size_t(t.idx) * CAP;
  uint8_t* commonMtxBase = commonMtx + capacity;

  unsigned int* degreeBase = s.degree + t.idx * MAX_BLK_SIZE;
  unsigned int* l_degreeBase = s.l_degree + t.idx * MAX_BLK_SIZE;
  unsigned int* offsetsBase = s.offsets + t.idx * MAX_BLK_SIZE;
  unsigned int* neighborsBase = s.neighbors + t.idx * MAX_BLK_SIZE * AVG_DEGREE;
  unsigned int* l_offsetsBase = s.l_offsets + t.idx * MAX_BLK_SIZE;
  unsigned int* l_neighborsBase = s.l_neighbors + t.idx * MAX_BLK_SIZE * AVG_LEFT_DEGREE;

  unsigned int* plex = s.P + warp_id * MAX_BLK_SIZE;
  unsigned int* cand = s.C + warp_id * MAX_BLK_SIZE;
  unsigned int* excl = s.X + warp_id * MAX_BLK_SIZE;

  uint16_t* local_sat = d_sat + warp_id * MAX_BLK_SIZE;
  uint16_t* local_commons = d_commons + warp_id * MAX_BLK_SIZE;
  uint32_t* local_uni = d_uni + warp_id * 32;

  uint32_t* adjList = d_adj + t.idx * ADJSIZE;

  unsigned int* state = d_state + warp_id * MAX_DEPTH;
  unsigned int* res = d_res + warp_id * MAX_DEPTH;

  unsigned int* recCandBase = recCand + warp_id * MAX_BLK_SIZE;
  unsigned int* recExclBase = recExcl + warp_id * MAX_BLK_SIZE;

  unsigned int n;
  if (lane_id == 0) n = local_n[0];
  n = __shfl_sync(0xFFFFFFFF, n, 0);

  initializePCX(lane_id, labelsBase, n, plex, cand, excl);

  int sz = 0;
  if (lane_id == 0) state[sz] = 0;
  sz++;


  while(sz)
  {
    if (sz >= MAX_DEPTH)
    {
      if (lane_id == 0) printf("capacity crossed: %d\n", sz);
      break;
    }

    switch(state[sz-1])
    {
      case 0:
        // if (lane_id == 0) printf("Hello to case 1 from warp: %d with task: %d\n", warp_id, t.idx);
        if (PlexSz + CandSz < q)
        {
          sz--;
          continue;
        }
        if (CandSz == 0)
        {
          if (ExclSz == 0 &&
              PlexSz >= q &&
              isMaximal_opt(lane_id, k, PlexSz, leftBase, left_count[0], l_neighborsBase, l_offsetsBase, l_degreeBase, neiInP, neighborsBase, offsetsBase, degreeBase, plex, n, local_sat, local_commons, local_uni))
              {
                if (lane_id == 0) atomicAdd(&plex_count[0], 1);
              }
              sz--;
              continue;
        }
        __syncwarp();

        int minnei_Plex = INT_MAX;
        int pivot = -1;
        int minnei_Cand = INT_MAX;

        for(int i = lane_id; i < PlexSz; i+=32)
        {
          const int v = plex[i];
          if (neiInG[v] < minnei_Plex)
          {
            minnei_Plex = neiInG[v];
            pivot = v;
          }
        }

        for(int offset = 16; offset > 0; offset >>= 1)
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
          
        if (minnei_Plex + k < max(q, PlexSz))
        {
          sz--;
          continue;
        }
        if (minnei_Plex + k < PlexSz + CandSz)
        {
          minnei_Cand = INT_MAX;
          pivot = -1;
            
          for (int i = lane_id; i < CandSz; i+=32)
          {
            const int v = cand[i];
            int check = v * n + pivot_plex;
            if (!((adjList[check >> 5] >> (check & 31)) &1u ))
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
          // if (lane_id == 0) printf("pivot: %d\n", pivot);
          // if (lane_id == 0) printf("pivot plex: %d\n", pivot_plex);
          if (pivot == -1)
          {
            pivot = cand[CandSz-1];
          }
          if (lane_id == 0) state[sz-1] = 1;
          __syncwarp();
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

        for (int offset = 16; offset > 0; offset >>= 1)
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
          
        if(minnei >= (PlexSz + CandSz - k))
        {
          if (PlexSz + CandSz < q)
          {
            sz--;
            continue;
          }
          bool flag = false;

          for (int i = lane_id; i < ExclSz; i+=32)
          {
            const int v = excl[i];
            if (isKplexPC2(v, k, PlexSz+CandSz, PlexSz, CandSz, neiInG, plex, cand, n, neighborsBase, offsetsBase, degreeBase, adjList))
            {
              flag = true;
            }
          }

          if (__any_sync(0xFFFFFFFF, flag))
          {
            sz--;
            continue;
          }

          if (isMaximalPC_opt(lane_id, k, PlexSz, CandSz, PlexSz + CandSz, leftBase, left_count[0], l_neighborsBase, l_offsetsBase, l_degreeBase, neiInG, neighborsBase, offsetsBase, degreeBase, plex, cand, n, local_sat, local_commons, local_uni))
          {
            if (lane_id == 0) atomicAdd(&plex_count[0], 1);
          }
          sz--;
          continue;
        }
        if (lane_id == 0) state[sz-1] = 1;
        __syncwarp();
        continue;
      
      case 1:
        // if (lane_id == 0) printf("Hello to case 1 from warp: %d with task: %d\n", warp_id, t.idx);
        // if (lane_id == 0) printf("Case - 0\n");
        ExclSz++;
        if (lane_id == 0)
        {
          excl[ExclSz-1] = pivot;
          for (int i = 0; i < CandSz; i++)
          {
            if (cand[i] == pivot)
            {
              int temp = cand[i];
              cand[i] = cand[CandSz-1];
              cand[CandSz-1] = temp;
              break;
            }
          }
          res[sz-1] = pivot;
        }
        CandSz--;
        __syncwarp();
        subG(lane_id, pivot, neiInG, n, neighborsBase, offsetsBase, degreeBase);

          
        if (lane_id == 0)
        {
          state[sz-1] = 2;
          state[sz] = 0;
        }
        __syncwarp();
        sz++;
        continue;
      
      case 2:
        // if (lane_id == 0) printf("Hello to case 2 from warp: %d with task: %d\n", warp_id, t.idx);
        CandSz++;
        if (lane_id == 0)
        {
        cand[CandSz-1] = res[sz-1];
        // excl[--ExclSz[0]];
        }
        __syncwarp();
        pivot = cand[CandSz-1];
        for (int i = 0; i < ExclSz; i++)
        {
          if (excl[i] == pivot)
          {
            if (lane_id == 0)
            {
              int temp = excl[i];
              excl[i] = excl[ExclSz-1];
              excl[ExclSz-1] = temp;
            }
            ExclSz--;
            break;
          }
        }
        addG(lane_id, pivot, neiInG, n, neighborsBase, offsetsBase, degreeBase);

        PlexSz++;
        if (lane_id == 0)
        {
          plex[PlexSz-1] = pivot;
        }
        for (int i = 0; i < CandSz; i++)
        {
          if (cand[i] == pivot)
          {
            if (lane_id == 0)
            {
              int temp = cand[i];
              cand[i] = cand[CandSz-1];
              cand[CandSz-1] = temp;
            }
            CandSz--;
            break;
          }
        }
        __syncwarp();
        for (int j = lane_id; j < degreeBase[pivot]; j+=32)
        {
          const int nei = neighborsBase[offsetsBase[pivot]+j];
          neiInP[nei]++;
        }
        __syncwarp();

        const uint8_t* row = commonMtxBase + (size_t) pivot * n;

        int read  = 0;
        int write = 0;
        int size = CandSz;
        // int total_removed = 0;

        while (read < size)
        {
          const int take = min(32, size - read);
          const bool active = (lane_id < take);
          //const int idx = i + lane_id;

          unsigned int v = 0;
          if (active) v = cand[read+lane_id];

          const bool keep = active && (isKplex3(v, k, PlexSz, neiInP, plex, n, neighborsBase, offsetsBase, degreeBase, adjList)) && !(row[v] < UNLINK2EQUAL);

          const unsigned activemask = __ballot_sync(0xFFFFFFFF, active);
          unsigned keepmask = __ballot_sync(0xFFFFFFFF, keep);
          unsigned dropmask = activemask ^ keepmask;

          const int keep_rank = __popc(keepmask & ((1u << lane_id) - 1));
          // const int drop_rank = __popc(dropmask & ((1u << lane_id) - 1));
          const int num_keep  = __popc(keepmask);

          if (active && keep)
          {
            cand[write + keep_rank] = v;
          }

          while (dropmask)
          {
            const int leader = __ffs(dropmask) - 1;
            const unsigned vdrop = __shfl_sync(0xFFFFFFFF, v, leader);
            recCandBase[vdrop] = (unsigned) (sz-1);
            subG(lane_id, vdrop, neiInG, n, neighborsBase, offsetsBase, degreeBase);
            dropmask &= (dropmask - 1);
          }

          if (lane_id == 0)
          {
            read += take;
            write += num_keep;
            // size = newR;
            // total_removed += num_drop;
          }
          read  = __shfl_sync(0xFFFFFFFF, read, 0);
          write = __shfl_sync(0xFFFFFFFF, write, 0);
          // size = __shfl_sync(0xFFFFFFFF, size, 0);
        }
        CandSz = write;
        __syncwarp();
        // if (lane_id == 0) v2delete[sz-1] = total_removed;

        if (upperBound2(lane_id, k, q, plex, neiInG, PlexSz))
        {
          read  = 0;
          write = 0;
          size = ExclSz;
          // int total_removed = 0;

          while (read < size)
          {
            const int take = min(32, size - read);
            const bool active = (lane_id < take);
            //const int idx = i + lane_id;

            unsigned int v = 0;
            if (active) v = excl[read+lane_id];

            const bool keep = active && (isKplex3(v, k, PlexSz, neiInP, plex, n, neighborsBase, offsetsBase, degreeBase, adjList)) && !(row[v] < UNLINK2MORE);

            const unsigned activemask = __ballot_sync(0xFFFFFFFF, active);
            unsigned keepmask = __ballot_sync(0xFFFFFFFF, keep);
            unsigned dropmask = activemask ^ keepmask;

            const int keep_rank = __popc(keepmask & ((1u << lane_id) - 1));
            // const int drop_rank = __popc(dropmask & ((1u << lane_id) - 1));
            const int num_keep  = __popc(keepmask);

            if (active && keep)
            {
              excl[write + keep_rank] = v;
            }

            while (dropmask)
            {
              const int leader = __ffs(dropmask) - 1;
              const unsigned vdrop = __shfl_sync(0xFFFFFFFF, v, leader);
              recExclBase[vdrop] = (unsigned) (sz-1);
              dropmask &= (dropmask - 1);
            }

            if (lane_id == 0)
            {
              read += take;
              write += num_keep;
              // size = newR;
              // total_removed += num_drop;
            }
            read  = __shfl_sync(0xFFFFFFFF, read, 0);
            write = __shfl_sync(0xFFFFFFFF, write, 0);
              // size = __shfl_sync(0xFFFFFFFF, size, 0);
          }
          ExclSz = write;
          __syncwarp();

          if (lane_id == 0)
          {
            state[sz-1] = 3;
            state[sz] = 0;
          }
          __syncwarp();
          sz++;
          continue;
        }
        // if (lane_id == 0) res[sz-1] = 0;
        if (lane_id == 0) state[sz-1] = 3;
        __syncwarp();
        continue;

      case 3:
        // if (lane_id == 0) printf("Hello to case 3 from warp: %d with task: %d\n", warp_id, t.idx);
        recoverCand23(lane_id, n, excl, recExclBase, sz, &ExclSz);
        ExclSz = __shfl_sync(0xFFFFFFFF, ExclSz, 0);
        recoverCand12(lane_id, cand, recCandBase, neiInG, sz, n, &CandSz, neighborsBase, degreeBase, offsetsBase);
        CandSz = __shfl_sync(0xFFFFFFFF, CandSz, 0);
        CandSz++;
        PlexSz--;
        if (lane_id == 0) cand[CandSz-1] = plex[PlexSz];
        __syncwarp();
        pivot = cand[CandSz-1];
        for (int j = lane_id; j < degreeBase[pivot]; j+=32)
        {
          const int nei = neighborsBase[offsetsBase[pivot]+j];
          neiInP[nei]--;
        }
        __syncwarp();
        sz--;
        continue;
    }
  }
}

__global__ void BNB3(int i, P_pointers p, S_pointers s, unsigned int* d_blk, unsigned int* d_left, unsigned int* d_blk_counter, unsigned int* d_left_counter, uint8_t* commonMtx, Task* tasks, Task* outTasks, Task* global_tasks, unsigned int N, unsigned int head, unsigned int* tailPtr, unsigned int* global_tail, uint8_t* d_all_labels, uint16_t* d_all_neiInG, uint16_t* d_all_neiInP, uint8_t* global_labels, uint16_t* global_neiInG, uint16_t* global_neiInP, unsigned int* plex_count, uint16_t* d_sat, uint16_t* d_commons, uint32_t* d_uni, unsigned long long* cycles, uint32_t* d_adj, int* d_abort, unsigned int* d_state, unsigned int* d_res, unsigned int* recExcl, unsigned int* recCand)
{
  unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int warp_id = (global_index / 32);
  unsigned int lane_id = threadIdx.x % 32;

  //if (lane_id == 0) printf("N: %d, warp_id: %d, i: %d\n", N, warp_id, i);
  if ((warp_id+WARPS*i) >= N)  return;
  // //if (lane_id == 0) printf("Hi\n");
  int k = p.k;
  int q = p.lb;
  Task t = tasks[warp_id+WARPS*i];
  if (t.idx == -1) return;
  uint8_t* labelsBase = t.labels;
  unsigned int* blkBase = d_blk + t.idx * MAX_BLK_SIZE;
  unsigned int* leftBase = d_left + t.idx * MAX_BLK_SIZE;
  uint16_t* neiInG = t.neiInG;
  uint16_t* neiInP = t.neiInP;
  unsigned int* left_count = d_left_counter + t.idx;
  unsigned int* local_n = d_blk_counter + t.idx;
  unsigned int PlexSz = t.PlexSz;
  unsigned int CandSz = t.CandSz;
  unsigned int ExclSz = t.ExclSz;
  size_t capacity = size_t(t.idx) * CAP;
  uint8_t* commonMtxBase = commonMtx + capacity;

  unsigned int* degreeBase = s.degree + t.idx * MAX_BLK_SIZE;
  unsigned int* l_degreeBase = s.l_degree + t.idx * MAX_BLK_SIZE;
  unsigned int* offsetsBase = s.offsets + t.idx * MAX_BLK_SIZE;
  unsigned int* neighborsBase = s.neighbors + t.idx * MAX_BLK_SIZE * AVG_DEGREE;
  unsigned int* l_offsetsBase = s.l_offsets + t.idx * MAX_BLK_SIZE;
  unsigned int* l_neighborsBase = s.l_neighbors + t.idx * MAX_BLK_SIZE * AVG_LEFT_DEGREE;

  unsigned int* plex = s.P + warp_id * MAX_BLK_SIZE;
  unsigned int* cand = s.C + warp_id * MAX_BLK_SIZE;
  unsigned int* excl = s.X + warp_id * MAX_BLK_SIZE;

  uint16_t* local_sat = d_sat + warp_id * MAX_BLK_SIZE;
  uint16_t* local_commons = d_commons + warp_id * MAX_BLK_SIZE;
  uint32_t* local_uni = d_uni + warp_id * 32;

  uint32_t* adjList = d_adj + t.idx * ADJSIZE;

  unsigned int* state = d_state + warp_id * MAX_DEPTH;
  unsigned int* res = d_res + warp_id * MAX_DEPTH;

  unsigned int* recCandBase = recCand + warp_id * MAX_BLK_SIZE;
  unsigned int* recExclBase = recExcl + warp_id * MAX_BLK_SIZE;

  unsigned int n;
  if (lane_id == 0) n = local_n[0];
  n = __shfl_sync(0xFFFFFFFF, n, 0);

  initializePCX(lane_id, labelsBase, n, plex, cand, excl);

  // if (lane_id == 0) printf("tail: %d\n", tailPtr[0]);

  int sz = 0;
  state[sz] = 0;
  sz++;

  uint8_t* labels = d_all_labels;
  uint16_t* all_neiInG = d_all_neiInG;
  uint16_t* all_neiInP = d_all_neiInP;
  Task* localTasks = outTasks;


  while(sz)
  {
    if (sz >= MAX_DEPTH)
    {
      if (lane_id == 0) printf("capacity crossed: %d\n", sz);
      break;
    }

    switch(state[sz-1])
    {
      case 0:
        // if (lane_id == 0) printf("Hello to case 1 from warp: %d with task: %d\n", warp_id, t.idx);
        if (PlexSz + CandSz < q)
        {
          sz--;
          continue;
        }
        if (CandSz == 0)
        {
          if (ExclSz == 0 &&
              PlexSz >= q &&
              isMaximal_opt(lane_id, k, PlexSz, leftBase, left_count[0], l_neighborsBase, l_offsetsBase, l_degreeBase, neiInP, neighborsBase, offsetsBase, degreeBase, plex, n, local_sat, local_commons, local_uni))
              {
                if (lane_id == 0) atomicAdd(&plex_count[0], 1);
              }
              sz--;
              continue;
        }
        __syncwarp();

        int minnei_Plex = INT_MAX;
        int pivot = -1;
        int minnei_Cand = INT_MAX;

        for(int i = lane_id; i < PlexSz; i+=32)
        {
          const int v = plex[i];
          if (neiInG[v] < minnei_Plex)
          {
            minnei_Plex = neiInG[v];
            pivot = v;
          }
        }

        for(int offset = 16; offset > 0; offset >>= 1)
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
          
        if (minnei_Plex + k < max(q, PlexSz))
        {
          sz--;
          continue;
        }
        if (minnei_Plex + k < PlexSz + CandSz)
        {
          minnei_Cand = INT_MAX;
          pivot = -1;
            
          for (int i = lane_id; i < CandSz; i+=32)
          {
            const int v = cand[i];
            int check = v * n + pivot_plex;
            if (!((adjList[check >> 5] >> (check & 31)) &1u ))
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
          // if (lane_id == 0) printf("pivot: %d\n", pivot);
          // if (lane_id == 0) printf("pivot plex: %d\n", pivot_plex);
          if (pivot == -1)
          {
            pivot = cand[CandSz-1];
          }
          if (lane_id == 0) state[sz-1] = 1;
          __syncwarp();
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

        for (int offset = 16; offset > 0; offset >>= 1)
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
          
        if(minnei >= (PlexSz + CandSz - k))
        {
          if (PlexSz + CandSz < q)
          {
            sz--;
            continue;
          }
          bool flag = false;

          for (int i = lane_id; i < ExclSz; i+=32)
          {
            const int v = excl[i];
            if (isKplexPC2(v, k, PlexSz+CandSz, PlexSz, CandSz, neiInG, plex, cand, n, neighborsBase, offsetsBase, degreeBase, adjList))
            {
              flag = true;
            }
          }

          if (__any_sync(0xFFFFFFFF, flag))
          {
            sz--;
            continue;
          }

          if (isMaximalPC_opt(lane_id, k, PlexSz, CandSz, PlexSz + CandSz, leftBase, left_count[0], l_neighborsBase, l_offsetsBase, l_degreeBase, neiInG, neighborsBase, offsetsBase, degreeBase, plex, cand, n, local_sat, local_commons, local_uni))
          {
            if (lane_id == 0) atomicAdd(&plex_count[0], 1);
          }
          sz--;
          continue;
        }
        if (lane_id == 0) state[sz-1] = 1;
        __syncwarp();
        continue;
      
      case 1:
        // if (lane_id == 0) printf("Hello to case 1 from warp: %d with task: %d\n", warp_id, t.idx);
        // if (lane_id == 0) printf("Case - 0\n");
        ExclSz++;
        if (lane_id == 0)
        {
          excl[ExclSz-1] = pivot;
          for (int i = 0; i < CandSz; i++)
          {
            if (cand[i] == pivot)
            {
              int temp = cand[i];
              cand[i] = cand[CandSz-1];
              cand[CandSz-1] = temp;
              break;
            }
          }
          res[sz-1] = pivot;
        }
        CandSz--;
        __syncwarp();
        subG(lane_id, pivot, neiInG, n, neighborsBase, offsetsBase, degreeBase);

          
        if (lane_id == 0)
        {
          state[sz-1] = 2;
          state[sz] = 0;
        }
        __syncwarp();
        sz++;
        continue;
      
      case 2:
        // if (lane_id == 0) printf("Hello to case 2 from warp: %d with task: %d\n", warp_id, t.idx);
        CandSz++;
        if (lane_id == 0)
        {
        cand[CandSz-1] = res[sz-1];
        // excl[--ExclSz[0]];
        }
        __syncwarp();
        pivot = cand[CandSz-1];
        for (int i = 0; i < ExclSz; i++)
        {
          if (excl[i] == pivot)
          {
            if (lane_id == 0)
            {
              int temp = excl[i];
              excl[i] = excl[ExclSz-1];
              excl[ExclSz-1] = temp;
            }
            ExclSz--;
            break;
          }
        }
        addG(lane_id, pivot, neiInG, n, neighborsBase, offsetsBase, degreeBase);

        PlexSz++;
        if (lane_id == 0)
        {
          plex[PlexSz-1] = pivot;
        }
        for (int i = 0; i < CandSz; i++)
        {
          if (cand[i] == pivot)
          {
            if (lane_id == 0)
            {
              int temp = cand[i];
              cand[i] = cand[CandSz-1];
              cand[CandSz-1] = temp;
            }
            CandSz--;
            break;
          }
        }
        __syncwarp();
        for (int j = lane_id; j < degreeBase[pivot]; j+=32)
        {
          const int nei = neighborsBase[offsetsBase[pivot]+j];
          neiInP[nei]++;
        }
        __syncwarp();

        const uint8_t* row = commonMtxBase + (size_t) pivot * n;

        int read  = 0;
        int write = 0;
        int size = CandSz;
        // int total_removed = 0;

        while (read < size)
        {
          const int take = min(32, size - read);
          const bool active = (lane_id < take);
          //const int idx = i + lane_id;

          unsigned int v = 0;
          if (active) v = cand[read+lane_id];

          const bool keep = active && (isKplex3(v, k, PlexSz, neiInP, plex, n, neighborsBase, offsetsBase, degreeBase, adjList)) && !(row[v] < UNLINK2EQUAL);

          const unsigned activemask = __ballot_sync(0xFFFFFFFF, active);
          unsigned keepmask = __ballot_sync(0xFFFFFFFF, keep);
          unsigned dropmask = activemask ^ keepmask;

          const int keep_rank = __popc(keepmask & ((1u << lane_id) - 1));
          // const int drop_rank = __popc(dropmask & ((1u << lane_id) - 1));
          const int num_keep  = __popc(keepmask);

          if (active && keep)
          {
            cand[write + keep_rank] = v;
          }

          while (dropmask)
          {
            const int leader = __ffs(dropmask) - 1;
            const unsigned vdrop = __shfl_sync(0xFFFFFFFF, v, leader);
            recCandBase[vdrop] = (unsigned) (sz-1);
            subG(lane_id, vdrop, neiInG, n, neighborsBase, offsetsBase, degreeBase);
            dropmask &= (dropmask - 1);
          }

          if (lane_id == 0)
          {
            read += take;
            write += num_keep;
            // size = newR;
            // total_removed += num_drop;
          }
          read  = __shfl_sync(0xFFFFFFFF, read, 0);
          write = __shfl_sync(0xFFFFFFFF, write, 0);
          // size = __shfl_sync(0xFFFFFFFF, size, 0);
        }
        CandSz = write;
        __syncwarp();
        // if (lane_id == 0) v2delete[sz-1] = total_removed;

        if (upperBound2(lane_id, k, q, plex, neiInG, PlexSz))
        {
          read  = 0;
          write = 0;
          size = ExclSz;
          // int total_removed = 0;

          while (read < size)
          {
            const int take = min(32, size - read);
            const bool active = (lane_id < take);
            //const int idx = i + lane_id;

            unsigned int v = 0;
            if (active) v = excl[read+lane_id];

            const bool keep = active && (isKplex3(v, k, PlexSz, neiInP, plex, n, neighborsBase, offsetsBase, degreeBase, adjList)) && !(row[v] < UNLINK2MORE);

            const unsigned activemask = __ballot_sync(0xFFFFFFFF, active);
            unsigned keepmask = __ballot_sync(0xFFFFFFFF, keep);
            unsigned dropmask = activemask ^ keepmask;

            const int keep_rank = __popc(keepmask & ((1u << lane_id) - 1));
            // const int drop_rank = __popc(dropmask & ((1u << lane_id) - 1));
            const int num_keep  = __popc(keepmask);

            if (active && keep)
            {
              excl[write + keep_rank] = v;
            }

            while (dropmask)
            {
              const int leader = __ffs(dropmask) - 1;
              const unsigned vdrop = __shfl_sync(0xFFFFFFFF, v, leader);
              recExclBase[vdrop] = (unsigned) (sz-1);
              dropmask &= (dropmask - 1);
            }

            if (lane_id == 0)
            {
              read += take;
              write += num_keep;
              // size = newR;
              // total_removed += num_drop;
            }
            read  = __shfl_sync(0xFFFFFFFF, read, 0);
            write = __shfl_sync(0xFFFFFFFF, write, 0);
              // size = __shfl_sync(0xFFFFFFFF, size, 0);
          }
          ExclSz = write;
          __syncwarp();

          unsigned int pos;
          // if (lane_id == 0) printf("tail: %d\n", tailPtr[0]);
          if (lane_id == 0) pos = atomicAdd(&tailPtr[0], 1u);
          pos = __shfl_sync(0xFFFFFFFF, pos, 0);
          // bool use_global = (pos + 1 > (SMALL_CAP)-WARPS);

          if (pos + 1 > SMALL_CAP) 
          {
            if (lane_id == 0)
            {
              atomicSub(&tailPtr[0], 1);
              pos = atomicAdd(&global_tail[0], 1u);
              // printf("position: %d, idx: %d, include\n", pos, idx);
            }
            pos = __shfl_sync(0xFFFFFFFF, pos, 0);
            if (pos+1 >= MAX_CAP/2-WARPS)
            {
              printf("Maximum Capacity Reached\n");
              return;
            }
            labels = global_labels;
            all_neiInG = global_neiInG;
            all_neiInP = global_neiInP;
            localTasks = global_tasks;
          }
          uint8_t* childLabels = labels + pos*MAX_BLK_SIZE;
          uint16_t* childNeiInG = all_neiInG + pos*MAX_BLK_SIZE;
          uint16_t* childNeiInP = all_neiInP + pos*MAX_BLK_SIZE;

          for (int j = lane_id; j < local_n[0]; j+=32)
          {
            childLabels[j] = U;
            childNeiInG[j] = neiInG[j];
            childNeiInP[j] = neiInP[j];
          }
          for (int j = lane_id; j < PlexSz; j+=32)
          {
            unsigned int v = plex[j];
            childLabels[v] = P;
          }
          for (int j = lane_id; j < CandSz; j+=32)
          {
            unsigned int v = cand[j];
            childLabels[v] = C;
          }
          for (int j = lane_id; j < ExclSz; j+=32)
          {
            unsigned int v = excl[j];
            childLabels[v] = X;
          }

          __syncwarp();
          if (lane_id == 0)
          {
            Task &nt = localTasks[pos];
            nt.idx = t.idx;
            nt.PlexSz = PlexSz;
            nt.CandSz = CandSz;
            nt.ExclSz = ExclSz;
            nt.labels = childLabels;
            nt.neiInG = childNeiInG;
            nt.neiInP = childNeiInP;
          }
          // if (lane_id == 0) state[sz-1] = 3;
          // __syncwarp();
          // continue;
        }
        
        __syncwarp();
        if (lane_id == 0) state[sz-1] = 3;
        continue;
      case 3:
        // if (lane_id == 0) printf("Hello to case 3 from warp: %d with task: %d\n", warp_id, t.idx);
        recoverCand23(lane_id, n, excl, recExclBase, sz, &ExclSz);
        ExclSz = __shfl_sync(0xFFFFFFFF, ExclSz, 0);
        recoverCand12(lane_id, cand, recCandBase, neiInG, sz, n, &CandSz, neighborsBase, degreeBase, offsetsBase);
        CandSz = __shfl_sync(0xFFFFFFFF, CandSz, 0);
        CandSz++;
        PlexSz--;
        if (lane_id == 0) cand[CandSz-1] = plex[PlexSz];
        __syncwarp();
        pivot = cand[CandSz-1];
        for (int j = lane_id; j < degreeBase[pivot]; j+=32)
        {
          const int nei = neighborsBase[offsetsBase[pivot]+j];
          neiInP[nei]--;
        }
        __syncwarp();
        sz--;
        continue;
    }
  }
}