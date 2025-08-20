// #include "../inc/device_funcs.h"
#include "device_funcs.h"

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

__global__ void decompose(int i, P_pointers p, G_pointers g, D_pointers d, unsigned int *d_blk, unsigned int *d_blk_counter, unsigned int *d_left, unsigned int *d_left_counter, uint8_t *visited, unsigned int *global_count, unsigned int *left_count, unsigned int *validblk, unsigned int* d_hopSz)
{
    unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int warp_id = (global_index / 32);
    unsigned int lane_id = threadIdx.x % 32;

//     //__shared__ unsigned int sh_buf[32 * 112];
    // if (warp_id + WARPS * i != 1) return;
    if (warp_id != 63) return;
    long cond = long(g.n) - long(p.lb) + 2;
    if ((warp_id+WARPS*i) >= cond) return;
//     // if ((warp_id+WARPS*i) >= (g.n)) return;
//     //printf("Currently in thread %d of warp %d\n", lane_id, warp_id);

//     //unsigned int *neibors = sh_buf + (threadIdx.x / 32) * 112;
//     //int degreeCount = 0;
    int vstart = d.dseq[warp_id+WARPS*i];
    int idx;
    unsigned int* blkBase = d_blk + warp_id * MAX_BLK_SIZE; // d_blk = [2048*1280]
    unsigned int* counterBase = d_blk_counter + warp_id;

    unsigned int* leftBase = d_left + warp_id * MAX_BLK_SIZE;
    unsigned int* left_counter = d_left_counter + warp_id;
    unsigned int* hopSz = d_hopSz + warp_id;

    
    uint8_t *visitedBase = visited + warp_id * g.n;
    //unsigned int *countBase = count + warp_id * g.n;
    int lb_2k = p.lb - 2 * p.k; // q - 2*k
    if (lane_id == 0)
    {
      idx = atomicAdd(&counterBase[0], 1);
      blkBase[idx] = vstart; // seed vertex 
      visitedBase[vstart] = 1;
      //printf("Block Size is %d in warp %d\n", counterBase[0], warp_id);
    }
//     // if (lane_id == 0)
//     // {
//     //   printf("degree in %d: %d\n", warp_id, g.offsets[vstart]);
//     // }
     __syncwarp();
    int start = 0;
    while(true)
    {
      bool localOverFlow = false;
      int overFlowIdx = INT_MAX;

    for (int i = lane_id+start; i < g.degree[vstart]; i+=32)
    {
      if(localOverFlow) break;
      int nei = g.neighbors[g.offsets[vstart] + i];
      //if (warp_id == 440) printf("%d ", nei);
        if (d.dpos[vstart] < d.dpos[nei])
        {
          idx = atomicAdd(&counterBase[0], 1);
          if (idx < MAX_BLK_SIZE) 
          {
            blkBase[idx] = nei;
            visitedBase[nei] = 1;
          }
          else{
            atomicAdd(&counterBase[0], -1);
            localOverFlow = true;
            overFlowIdx = i;
            if (lane_id == 0) printf("Above the capacity for blk in %d\n", warp_id);
          }
          
        }
        else
        {
          idx = atomicAdd(&left_counter[0], 1);
          if (idx < MAX_BLK_SIZE) 
          {
            leftBase[idx] = nei;
            visitedBase[nei] = 1;
          }
          else
          {
            atomicAdd(&left_counter[0], -1);
            localOverFlow = true;
            overFlowIdx = i;
            if (lane_id == 0) printf("Above the capacity for left in %d\n", warp_id);
          }
        }

        localOverFlow = __any_sync(__activemask(), localOverFlow);
    }
    __syncwarp(0xFFFFFFFF);

    int otherMin = overFlowIdx;
    for(int offset = 16; offset > 0; offset >>= 1)
  {
    otherMin = min(otherMin, __shfl_down_sync(0xFFFFFFFF, otherMin, offset));
  }
  

  start = __shfl_sync(0xFFFFFFFF, otherMin, 0);

  size_t sz;
    do{
      sz = counterBase[0];
      if (sz - 1 < p.bd) goto CLEAN;
      for (int i = lane_id+1; i < counterBase[0]; i+=32)
      {
        size_t cnt = 0;
        unsigned int u = blkBase[i];
        for (int j = 1; j < counterBase[0]; j++)
        {
          unsigned int v = blkBase[j];
          if (binary_search(v, g.neighbors+g.offsets[u], g.degree[u])) cnt++;
        }
        if (cnt < lb_2k)
        {
          visitedBase[u] = 0;
        }
      }
      if (lane_id == 0)
      {
      int writeIdx = 1;
      int oldCount = counterBase[0];
      for (int i = 1; i < oldCount; i++)
      {
        unsigned int u = blkBase[i];
        if (visitedBase[u])
        {
          blkBase[writeIdx++] = u;
        }
      }
      counterBase[0] = writeIdx;
      }
  __syncwarp();
  } while (counterBase[0] < sz);
  
  if (lane_id == 0) 
  {
    hopSz[0] = counterBase[0];
  }

  __syncwarp();

  
    for (int i = lane_id; i < left_counter[0]; i+=32)
    {
      size_t cnt = 0;
      unsigned int u = leftBase[i];
      for (int j = 1; j < counterBase[0];j++)
      {
        unsigned int v = blkBase[j];
        if (binary_search(v, g.neighbors+g.offsets[u], g.degree[u])) cnt++;
      }
      if (cnt < lb_2k + 1) visitedBase[u] = 0;
    }
    __syncwarp();
  if (lane_id == 0)
  {
    int writeIdx = 0;
    int oldCount = left_counter[0];
    for (int i = 0; i < oldCount; i++)
    {
      unsigned int u = leftBase[i];
      if (visitedBase[u])
      {
        leftBase[writeIdx++] = u;
      }
    }
    left_counter[0] = writeIdx;
  }


  __syncwarp();
  if (!localOverFlow) break;
}


  // // try to run this loop, set mark to true, prune all those neighbors
  //   for (int i = lane_id; i < g.degree[vstart]; i+=32)
  //   {
  //     int nei = g.neighbors[g.offsets[vstart] + i];
  //     if (d.dpos[vstart] < d.dpos[nei]){
  //       for (int j = 0; j < g.degree[nei]; j++)
  //       {
  //           int twoHop = g.neighbors[g.offsets[nei]+j];
  //           if (!visitedBase[twoHop]) // blk -> direct neighbors, log O(degree(vstart)) time, normalizer into 1 - 900 range, denormalize it 
  //            {
  //              atomicAdd(&countBase[twoHop], 1); 
  //             //countBase[twoHop]++;
  //           }
  //       }
  //     }
  //   }

  //   __syncwarp(); //count = [3 4 5]

  //   for (int i = lane_id; i < g.degree[vstart]; i+=32)
  //   {
  //     int nei = g.neighbors[g.offsets[vstart] + i];
  //     if (d.dpos[vstart] < d.dpos[nei])
  //     {
  //       for (int j = 0; j < g.degree[nei]; j++)
  //       {
  //         int twoHop = g.neighbors[g.offsets[nei]+j];
  //         int old = atomicCAS(&visitedBase[twoHop], 0, 1);
  //         if (old == 0)
  //         {
  //           //visitedBase[twoHop] = 1;
  //           if (d.dpos[vstart] < d.dpos[twoHop])
  //           {
  //             if (countBase[twoHop] >= lb_2k + 2)
  //             {
  //               idx = atomicAdd(&counterBase[0], 1);
  //               blkBase[idx] = twoHop;
  //             }
  //           }
  //           else
  //           {
  //             if (countBase[twoHop] >= lb_2k + 3)
  //             {
  //               idx = atomicAdd(&left_counter[0], 1);
  //               leftBase[idx] = twoHop;
  //             }
  //           }
            
  //       }
  //       }
  //   }
  //   }

      for (int i = 0; i < g.degree[vstart]; i++)
      {
        int nei = g.neighbors[g.offsets[vstart] + i];
        if (d.dpos[vstart] < d.dpos[nei])
        {
          for (int j = lane_id; j < g.degree[nei]; j+=32)
          {
            int twoHop = g.neighbors[g.offsets[nei]+j];
            if (visitedBase[twoHop]) continue;
            int cnt = 0;
            for (int k = 1; k < hopSz[0]; k++)
            {
              unsigned int v = blkBase[k];
              if (basic_search(v, g.neighbors+g.offsets[twoHop], g.degree[twoHop]))
              {
                if (++cnt >= lb_2k + 3)
                {
                  break;
                }
              }
            }
            //int old = atomicCAS(&visitedBase[twoHop], 0, 1);
            if (visitedBase[twoHop] == 0)
            {
              visitedBase[twoHop] = 1;
              if (d.dpos[vstart] < d.dpos[twoHop])
              {
                if (cnt >= lb_2k + 2)
                {
                  idx = atomicAdd(&counterBase[0], 1);
                  blkBase[idx] = twoHop;
                }
              }
              else
              {
                if (cnt >= lb_2k + 3)
                {
                  idx = atomicAdd(&left_counter[0], 1);
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
      if (counterBase[0] >= p.lb){
        //printf("Number of direct neighbors + node itself in warp %d is %d\n", warp_id, hopSz[0]);
        atomicAdd(&validblk[0], 1);
        atomicAdd(&global_count[0], counterBase[0]);
        atomicAdd(&left_count[0], left_counter[0]);
      }
    }
    if (lane_id == 0)
    {
      if (counterBase[0] > MAX_BLK_SIZE)
      {
        printf("Block Size is greater than the constant\n");
      }
      else if (left_counter[0] > MAX_BLK_SIZE)
      {
        printf("Left Size is greater than constant\n");
      }
    }
    CLEAN:
    __syncwarp();
    for (int i = lane_id; i < g.n; i += 32)
    {
      visitedBase[i] = 0;
      //countBase[i] = 0;
    }


//     // if (lane_id == 0)
//     // {
//     //   //printf("Total number of nodes in blk in warp %d is %d\n", warp_id, counterBase[0]);
//     //   counterBase[0] = 0;
//     //   left_counter[0] = 0;
//     // }

  // __syncwarp();
}

__global__ void calculateDegrees(int i , P_pointers p, G_pointers g, S_pointers s, unsigned int *d_blk, unsigned int *d_blk_counter, unsigned int *d_left, unsigned int *d_left_counter, unsigned int *global_count, unsigned int *left_count)
{
  unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int warp_id = (global_index / 32);
  unsigned int lane_id = threadIdx.x % 32;
  // if ((warp_id+WARPS*i) >= g.n)  return;
  if ((warp_id+WARPS*i) >= (g.n-p.lb+2)) return;
  unsigned int* blkBase = d_blk + warp_id * MAX_BLK_SIZE;
  unsigned int* counterBase = d_blk_counter + warp_id;

  unsigned int* leftBase = d_left + warp_id * MAX_BLK_SIZE;
  unsigned int* left_counter = d_left_counter + warp_id;

  if (counterBase[0] < p.lb) return;
  unsigned int* local_n = s.n + warp_id;

  unsigned int* degreeBase = s.degree + warp_id * (MAX_BLK_SIZE);
  unsigned int* l_degreeBase = s.l_degree + warp_id * (MAX_BLK_SIZE);

  if (lane_id == 0)
  {
    local_n[0] = counterBase[0];
  }
  
  unsigned int lane_sum_ne = 0;
  unsigned int lane_sum_neLeft = 0;

  for (int idx = lane_id; idx < counterBase[0]; idx+=32)
  {
    unsigned int origin = blkBase[idx];
    int ne = 0, neLeft = 0;
    for (int j = 0; j < g.degree[origin];j++)
    {
      unsigned int nei = g.neighbors[g.offsets[origin]+j];
      if (basic_search(nei, blkBase, counterBase[0]))
      {
        ne++;
      }
      else if (basic_search(nei, leftBase, left_counter[0]))
      {
        neLeft++;
      }
    }

    degreeBase[idx] = ne;
    l_degreeBase[idx] = neLeft;

    lane_sum_ne += ne;
    lane_sum_neLeft += neLeft;
  }

  unsigned int warp_sum_ne = lane_sum_ne;
  unsigned int warp_sum_neLeft = lane_sum_neLeft;

  for(int j=16;j>0;j=j/2){
    (warp_sum_ne) += __shfl_down_sync(0xFFFFFFFF,(warp_sum_ne),j);
    (warp_sum_neLeft) += __shfl_down_sync(0xFFFFFFFF,(warp_sum_neLeft),j);
  }

  // if (lane_id == 0)
  // {
  //   //printf("Total degrees: %d in warp %d\n", ne, warp_id);
  //   atomicAdd(&global_count[0], warp_sum_ne);
  //   atomicAdd(&left_count[0], warp_sum_neLeft);
  // }
}

__global__ void fillNeighbors(int i, S_pointers s, P_pointers p, G_pointers g, unsigned int *d_blk, unsigned int *d_blk_counter, unsigned int *d_left, unsigned int *d_left_counter, unsigned int *d_hopSz, uint8_t* commonMtx, uint32_t* d_adj)
{
  unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int warp_id = (global_index / 32);
  unsigned int lane_id = threadIdx.x % 32;

  // if ((warp_id+WARPS*i) >= g.n)  return;
  if ((warp_id+WARPS*i) >= (g.n-p.lb+2)) return;
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
  bool* proper = s.proper + warp_id;
  uint8_t* commonMtxBase = commonMtx + warp_id * CAP;

  uint32_t* adjList = d_adj + warp_id * ADJSIZE;
  //-----------------BNB-----------------
  // unsigned int* Cand1Sz = s.C1Size + warp_id;
  // unsigned int* Cand2Sz = s.C2Size + warp_id;
  //-------------------BNB----------------
  //--------------------BK-----------------
  unsigned int* CandSz = s.CSize + warp_id;
  unsigned int* Cand2Sz = s.C2Size + warp_id;
  //------------------BK-----------------
  unsigned int* ExclSz = s.XSize + warp_id;
  uint8_t* labelsBase = s.labels + warp_id * (MAX_BLK_SIZE);

  unsigned int* plex = s.P + warp_id * MAX_BLK_SIZE;
  unsigned int* cand1 = s.C + warp_id * MAX_BLK_SIZE;
  unsigned int* cand2 = s.C2 + warp_id * MAX_BLK_SIZE;
  unsigned int* excl = s.X + warp_id * MAX_BLK_SIZE;

  uint32_t* Pset = s.Pset + warp_id * 32;
  uint32_t* Cset = s.Cset + warp_id * 32;
  uint32_t* C2set = s.C2set + warp_id * 32;
  uint32_t* Xset = s.Xset + warp_id * 32;

  int edgesHop = 0;
  int edges = 0;
  for (int idx = lane_id; idx < counterBase[0]; idx+=32)
  {
    unsigned int origin = blkBase[idx];
    int cnt = 0;
    unsigned int offset = offsetsBase[idx];
    unsigned int l_offset = l_offsetsBase[idx];
    for (int j = 0; j < counterBase[0]; j++)
    {
      unsigned int nei = blkBase[j];
      if (binary_search(nei, g.neighbors+g.offsets[origin], g.degree[origin]))
      {
        int v = idx*counterBase[0]+j;
        int v2 = j*counterBase[0] + idx;
        atomicOr(&adjList[v >> 5], 1u << (v & 31));
        atomicOr(&adjList[v2 >> 5], 1u << (v2 & 31));
        commonMtxBase[v] = 1;
        neighborsBase[offset+cnt] = j;
        cnt++;
      }
      if (j == hopSz[0]-1) degreeHop[idx] = cnt;
    }

    cnt = 0;
    for (int j = 0; j < left_counter[0]; j++)
    {
      unsigned int nei = leftBase[j];
      if (binary_search(nei, g.neighbors+g.offsets[origin], g.degree[origin]))
      {
        l_neighborsBase[l_offset+cnt] = j;
        cnt++;
      }
    }
    if (idx < hopSz[0]) edgesHop += degreeHop[idx];
    edges += degreeBase[idx];
  }

  __syncwarp();

  for (int off = 16; off > 0; off >>= 1)
  {
    edges += __shfl_down_sync(0xFFFFFFFF, edges, off);
    edgesHop += __shfl_down_sync(0xFFFFFFFF, edgesHop, off);
  }

  if (lane_id == 0)
  {
    proper[0] = (double(edgesHop)/(hopSz[0]*(hopSz[0]-1))) < 0.95 && (double(edges)/(counterBase[0]*(counterBase[0]-1))) < 0.9;
  }
  //--------------------------BNB---------------

  for (int i = lane_id+1; i < hopSz[0]; i += 32)
  {
    cand1[i-1] = i;
    atomicOr(&Cset[i >> 5], 1u << (i & 31));
  }
  for (int i = lane_id+hopSz[0]; i < local_n[0]; i+=32)
  {
    cand2[i-hopSz[0]] = i;
    atomicOr(&C2set[i >> 5], 1u << (i & 31));
  }
  if (lane_id == 0) plex[0] = 0;
  atomicOr(&Pset[0 >> 5], 1u << (0 & 31));
  //----------------------BNB-----------------
  //--------------------BK-------------------
  // for (int i = lane_id; i < local_n[0]; i+=32)
  // {
  //   labelsBase[i] = C;
  // }
  //--------------------BK---------------------

  //----------------BNB---------------------
  //labelsBase[0] = P;
  local_m[0] = offsetsBase[local_n[0]];
  //PlexSz[0] = 1;
  // Cand1Sz[0] = hopSz[0] - 1;
  // Cand2Sz[0] = local_n[0] - hopSz[0];
  //---------------BNB--------------------
  //--------------BK-------------------
  CandSz[0] = hopSz[0]-1;
  Cand2Sz[0] = local_n[0] - hopSz[0];
  ExclSz[0] = 0;
  PlexSz[0] = 1;

  if (warp_id == 63 && lane_id == 0)
  {
    printf("Blks: ");
    for (int i = 0; i < counterBase[0]; i++)
    {
      printf("%d ", blkBase[i]);
    }
    printf("\n");
    printf("Lefts: ");
    for (int i = 0; i < left_counter[0]; i++)
    {
      printf("%d ", leftBase[i]);
    }
    printf("\n");
    printf("Degree of Node %d: ", blkBase[0]);
    for (int i = 0; i < counterBase[0]; i++)
    {
      printf("%d ", degreeBase[i]);
    }
    printf("\n");

    printf("Offsets Array: ");
    for (int i = 0; i < counterBase[0]+1; i++)
    {
      printf("%d ", offsetsBase[i]);
    }
    printf("\n");

    printf("Left Degree of Node %d: ", blkBase[0]);
    for (int i = 0; i < counterBase[0]; i++)
    {
      printf("%d ", l_degreeBase[i]);
    }
    printf("\n");

    printf("Left Offsets Array: ");
    for (int i = 0; i < counterBase[0]+1; i++)
    {
      printf("%d ", l_offsetsBase[i]);
    }
    printf("\n");

    printf("Degree Hop Array: ");
    for (int i = 0; i < counterBase[0]; i++)
    {
      printf("%d ", degreeHop[i]);
    }
    printf("\n");

    // printf("neiInG Array: ");
    // for (int i = 0; i < counterBase[0]; i++)
    // {
    //   printf("%d ", neiInGBase[i]);
    // }
    // printf("\n");

    printf("Neighors Array: ");
    for (int i = 0; i < offsetsBase[counterBase[0]]; i++)
    {
      printf("%d ", neighborsBase[i]);
    }
    printf("\n");

    printf("Left Neighors Array: ");
    for (int i = 0; i < l_offsetsBase[counterBase[0]]; i++)
    {
      printf("%d ", l_neighborsBase[i]);
    }
    printf("\n");

    // //---------------BNB----------------
    // //printf("n = %d, m = %d, Plex Size = %d, Cand1 Size = %d, Cand2 Size = %d, Excl Size = %d\n", local_n[0], local_m[0], PlexSz[0], Cand1Sz[0], Cand2Sz[0], ExclSz[0]);
    // //-----------------BK-----------
    printf("n = %d, m = %d, Plex Size = %d, Cand Size = %d, Cand2 Sz = %d, Excl Size = %d, hopSz: %d, leftCounter: %d\n", local_n[0], local_m[0], PlexSz[0], CandSz[0], Cand2Sz[0], ExclSz[0], hopSz[0], left_counter[0]);
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

// __device__ bool upperBoundK(P_pointers p, unsigned int lane_id, unsigned int n, uint8_t* labelsBase, unsigned int* blkBase, unsigned int* neiInG)
// {
//   int cnt = 1;
//   for (int t = lane_id; t < n; t+= 32)
//   {
//     if (labelsBase[t] == P)
//     {
//       if (neiInG[t] + p.k < p.lb) cnt = 0;
//     }
//   }

//   bool anyZero = __any_sync(0xFFFFFFFF, cnt == 0);
//   return !anyZero;

// }

// __device__ void listBranch(P_pointers p, unsigned int lane_id, unsigned int n, unsigned int PlexSz, unsigned int Cand1Sz, unsigned int ExclSz, unsigned int* labelsBase, unsigned int *blkBase, unsigned int* neiInG, unsigned int* neiInP)
// {
//   if (PlexSz + Cand1Sz < p.lb)
//   {
//    return; 
//   }
//   if (Cand1Sz == 0)
//   {
//     // if (ExclSz == 0 && isMaximal())
//     // {
//     //   emitPlex();
//     // }
//   }
//   return;
// }

// __device__ int cand2BackToExcl(unsigned int lane_id, uint8_t* labelsBase, unsigned int n, unsigned int* Cand2Sz, unsigned int* ExclSz)
// {
//   int idx = 0;
//   if (lane_id == 0)
//   {
//     idx = n - 1;
//     unsigned int v = labelsBase[idx];
//     while (v != C2)
//     {
//       idx--;
//       v = labelsBase[idx];
//     }
//     labelsBase[idx] = X;
//     Cand2Sz[0]--;
//     ExclSz[0]++;
//   }

//   idx = __shfl_sync(0xFFFFFFFF, idx, 0);
//   return idx;
// }

// __device__ void exclToCand2(unsigned int lane_id, int v, uint8_t* labelsBase, unsigned int n, unsigned int* Cand2Sz, unsigned int* ExclSz)
// {
//   if (lane_id == 0)
//   {
//     labelsBase[v] = C2;
//     Cand2Sz[0]++;
//     ExclSz[0]--;
//   }
//   __syncwarp();
// }

// __device__ int cand2BackToPlex(unsigned int lane_id, uint8_t* labelsBase, unsigned int n, unsigned int* Cand2Sz, unsigned int* PlexSz)
// {
//   int idx = 0;
//   if (lane_id == 0)
//   {
//     idx = n - 1;
//     unsigned int v = labelsBase[idx];
//     while (v != C2)
//     {
//       idx--;
//       v = labelsBase[idx];
//     }
//     labelsBase[idx] = P;
//     Cand2Sz[0]--;
//     PlexSz[0]++;
//   }

//   idx = __shfl_sync(0xFFFFFFFF, idx, 0);
//   return idx;
// }

// __device__ void plexToCand2(unsigned int lane_id, int v, uint8_t* labelsBase, unsigned int n, unsigned int* Cand2Sz, unsigned int* PlexSz)
// {
//   if (lane_id == 0)
//   {
//     labelsBase[v] = C2;
//     Cand2Sz[0]++;
//     PlexSz[0]--;
//   }
// }

// __device__ void search(int res, unsigned int warp_id, unsigned int lane_id, S_pointers s, P_pointers p, unsigned int *d_blk, unsigned int *d_left, unsigned int *neiInG, unsigned int *neiInP, unsigned int *d_hopSz)
// {
//   if (warp_id == 0 and lane_id == 0) printf("Search Start\n");
//   unsigned int* blkBase = d_blk + warp_id * MAX_BLK_SIZE;
//   //unsigned int* leftBase = d_left + warp_id * MAX_BLK_SIZE;

//   //unsigned int* offsetsBase = s.offsets + warp_id * (MAX_BLK_SIZE);
//   //unsigned int* l_offsetsBase = s.l_offsets + warp_id * (MAX_BLK_SIZE);

//   //unsigned int* degreeBase = s.degree + warp_id * (MAX_BLK_SIZE);
//   //unsigned int* l_degreeBase = s.l_degree + warp_id * (MAX_BLK_SIZE);
//   //unsigned int* degreeHop = s.degreeHop + warp_id * (MAX_BLK_SIZE);
//   unsigned int* neiInGBase = neiInG + warp_id * (MAX_BLK_SIZE);

//   //unsigned int* neighborsBase = s.neighbors + warp_id * (MAX_BLK_SIZE) * (AVG_DEGREE);
//   //unsigned int* l_neighborsBase = s.l_neighbors + warp_id * (MAX_BLK_SIZE) * (AVG_LEFT_DEGREE);

//   unsigned int* local_n = s.n + warp_id;
//   //unsigned int* local_m = s.m + warp_id;
//   unsigned int* PlexSz = s.PSize + warp_id;
//   unsigned int* Cand1Sz = s.C1Size + warp_id;
//   unsigned int* Cand2Sz = s.C2Size + warp_id;
//   unsigned int* ExclSz = s.XSize + warp_id;
//   //unsigned int* hopSz = d_hopSz + warp_id;
//   uint8_t* labelsBase = s.labels + warp_id * (MAX_BLK_SIZE);

//   if (Cand2Sz[0] == 0)
//   {
//     if (PlexSz[0] + Cand1Sz[0] < p.lb)
//     {
//       return;
//     }
//     if (PlexSz[0] > 1 && !upperBoundK(p, lane_id, local_n[0], labelsBase, blkBase, neiInGBase))
//     {
//       return;
//     }
//     //listBranch(p, lane_id, local_n[0], PlexSz[0], Cand1Sz[0], ExclSz[0], labelsBase, blkBase, neiInG, neiInP);
//     return;
//   }
//   int v2delete = cand2BackToExcl(lane_id, labelsBase, local_n[0], Cand2Sz, ExclSz);
//   search(res, warp_id, lane_id, s, p, d_blk, d_left, neiInG, neiInP, d_hopSz);
//   exclToCand2(lane_id, v2delete, labelsBase, local_n[0], Cand2Sz, ExclSz);
//   int br = 1;
//   int v2add[15];
//   int cnt = 0;
//   for (; br < res; br++)
//   {
//     v2add[cnt++] = cand2BackToPlex(lane_id, labelsBase, local_n[0], Cand2Sz, PlexSz);
//     if (Cand2Sz[0])
//     {
//       v2delete = cand2BackToExcl(lane_id, labelsBase, local_n[0], Cand2Sz, ExclSz);
//       search(res-br, warp_id, lane_id, s, p, d_blk, d_left, neiInG, neiInP, d_hopSz);
//       exclToCand2(lane_id, v2delete, labelsBase, local_n[0], Cand2Sz, ExclSz);
//     }
//     else
//     {
//       search(res-br, warp_id, lane_id, s, p, d_blk, d_left, neiInG, neiInP, d_hopSz);
//       break;
//     }
//   }
//   if (br == res)
//   {
//     v2add[cnt++] = cand2BackToPlex(lane_id, labelsBase, local_n[0], Cand2Sz, PlexSz);
//     if (PlexSz[0] + Cand1Sz[0] < p.lb)
//     {
//       goto restore;
//     }
//     if (PlexSz[0] > 1 && !upperBoundK(p, lane_id, local_n[0], labelsBase, blkBase, neiInGBase))
//     {
//       goto restore;
//     }
//     //listBranch(p, lane_id, local_n[0], PlexSz[0], Cand1Sz[0], ExclSz[0], labelsBase, blkBase, neiInG, neiInP);
//   }
//   restore:
//   for (int i = lane_id; i < cnt; i+=32)
//   {
//     plexToCand2(lane_id, v2add[i], labelsBase, local_n[0], Cand2Sz, PlexSz);
//   }
//   __syncwarp();

// }



__device__ void push(unsigned int lane_id, unsigned int k, int* cur_id, int * level, unsigned int n, unsigned int left_count, unsigned int* neighborsBase, unsigned int* l_neighborsBase, 
                     unsigned int * offsetsBase, unsigned int* l_offsetsBase, unsigned int* degreeBase, unsigned int* l_degreeBase,
                     uint8_t* labelsBase, unsigned int* nonNeighBase, unsigned int* nonNeighLeftBase, unsigned int* depthBase,
                     unsigned int* stackBase, unsigned int* leftSz, unsigned int* PlexSz, unsigned int* CandSz,
                     unsigned int* ExclSz)
{
  int local_flag = 0;
  if (lane_id == 0)
  {
  for (int i = 0; i < PlexSz[0]; i++)
  {
    unsigned int vtx = stackBase[i];
    if ((nonNeighBase[vtx]+1 > (k-1)) && (!basic_search(vtx, neighborsBase + offsetsBase[cur_id[0]], degreeBase[cur_id[0]])))
    {
      depthBase[cur_id[0]]--;
      CandSz[0]--;
      cur_id[0] = stackBase[level[0] - 1];
      local_flag = 1;
      break;
    }
  }
}
  __syncwarp();
  int bcur = __shfl_sync(0xFFFFFFFF, cur_id[0], 0);
  cur_id[0] = bcur;
  __syncwarp();

  int warp_any = __any_sync(0xFFFFFFFF, local_flag);
  if (warp_any)
  {
    return;
  }

  if (lane_id == 0)
  {
  labelsBase[cur_id[0]] = P;
  PlexSz[0]++;
  CandSz[0]--;
  stackBase[level[0]] = cur_id[0];
  }

  __syncwarp();

  int local_cand = 0;
  int local_excl = 0;
  //if (lane_id == 0){
  for (int i = lane_id; i < n; i+=32)
  {
    if (i == cur_id[0]) continue;

    if (!basic_search(i, neighborsBase + offsetsBase[cur_id[0]], degreeBase[cur_id[0]]))
      nonNeighBase[i]++;
    
    if (labelsBase[i] == C && depthBase[i] == level[0])
    {
      if (nonNeighBase[i] <= k - 1)
        depthBase[i]++;
      else
        local_cand++;

    }

    else if (labelsBase[i] == X && depthBase[i] == level[0])
    {
      if (nonNeighBase[i] <= k - 1)
        depthBase[i]++;
      else
        local_excl++;
    }
  }
//}

  int sum_cand = local_cand;
  int sum_excl = local_excl;

  for (int offset = 16; offset > 0; offset >>= 1)
  {
    sum_cand += __shfl_down_sync(0xFFFFFFFF, sum_cand, offset);
    sum_excl += __shfl_down_sync(0xFFFFFFFF, sum_excl, offset);
  }

  if (lane_id == 0)
  {
    CandSz[0] -= sum_cand;
    ExclSz[0] -= sum_excl;
  }

  __syncwarp();
  level[0]++;
}

__device__ void pop(unsigned int lane_id, unsigned int k, int* cur_id, int * level, unsigned int n, unsigned int left_count, unsigned int* neighborsBase, unsigned int* l_neighborsBase, 
                     unsigned int * offsetsBase, unsigned int* l_offsetsBase, unsigned int* degreeBase, unsigned int* l_degreeBase,
                     uint8_t* labelsBase, unsigned int* nonNeighBase, unsigned int* nonNeighLeftBase, unsigned int* depthBase,
                     unsigned int* stackBase, unsigned int* leftSz, unsigned int* PlexSz, unsigned int* CandSz,
                     unsigned int* ExclSz)
{
  int temp = 0;
  int local_excl = 0;
  int sum_excl = 0;
  for (int i = 0; i < PlexSz[0]; i++)
  {
    local_excl = 0;
    unsigned int vtx = stackBase[i];
    if (nonNeighBase[vtx] == (k-1))
    {
      //if (lane_id == 0){
      for (int j = lane_id; j < n; j+=32)
      {
        if (labelsBase[j] == X && depthBase[j] == level[0])
        {
          if (!basic_search(j, neighborsBase + offsetsBase[vtx], degreeBase[vtx]))
            {
              //atomicAdd(&ExclSz[0], -1);
              local_excl++;
              depthBase[j]--;
              temp = 1;
            }
        }
      }
    //}
      sum_excl = local_excl;
      for (int offset = 16; offset > 0; offset >>= 1)
      {
        sum_excl += __shfl_down_sync(0xFFFFFFFF, sum_excl, offset);
      }
      if (lane_id == 0) ExclSz[0] -= sum_excl;
      int warp_any = __any_sync(0xFFFFFFFF, temp);
      if (warp_any)
      {
        return;
      }
    }
  }
  __syncwarp();
  int local_cand = 0;
  local_excl = 0;
  //if (lane_id == 0){
  for (int i = lane_id; i < n; i+=32)
  {
    if (i == cur_id[0]) continue;

    if (depthBase[i] == (level[0] - 1))
    {
      if (labelsBase[i] == C) local_cand++;
        //atomicAdd(&CandSz[0], 1);
      if (labelsBase[i] == X) local_excl++;
        //atomicAdd(&ExclSz[0], 1);
    }
  }
  //}
  int sum_cand = local_cand;
  sum_excl = local_excl;

  for (int offset = 16; offset > 0; offset >>= 1)
  {
    sum_cand += __shfl_down_sync(0xFFFFFFFF, sum_cand, offset);
    sum_excl += __shfl_down_sync(0xFFFFFFFF, sum_excl, offset);
  }
  if (lane_id == 0)
  {
    CandSz[0] += sum_cand;
    ExclSz[0] += sum_excl;
  }

  //if (lane_id == 0) printf("CandSz: %d\n", CandSz[0]);

  __syncwarp();

  // if (lane_id == 0)
  // {
    for (int i = lane_id; i < n; i+=32)
    {
    if (i == cur_id[0]) continue;
    if (depthBase[i] == level[0])
      depthBase[i]--;
    
    if (!basic_search(i, neighborsBase + offsetsBase[cur_id[0]], degreeBase[cur_id[0]]))
      nonNeighBase[i]--;
    }
  // }
  __syncwarp();

  if (lane_id == 0)
  {
  labelsBase[cur_id[0]] = X;
  PlexSz[0]--;
  ExclSz[0]++;
  }
  __syncwarp();

  //printf("Nodes coming back: ");
  local_cand = 0;
  local_excl = 0;
  //if (lane_id == 0){
  for (int i = lane_id + cur_id[0] + 1; i < n; i+=32)
  {
    if (labelsBase[i] == X)
    {
      labelsBase[i] = C;
      // atomicAdd(&CandSz[0], 1);
      // atomicAdd(&ExclSz[0], -1);
      local_cand++;
      local_excl++;
      //printf("%d ", i);
    }
  }
//}

  sum_cand = local_cand;
  sum_excl = local_excl;

  for (int offset = 16; offset > 0; offset >>= 1)
  {
    sum_cand += __shfl_down_sync(0xFFFFFFFF, sum_cand, offset);
    sum_excl += __shfl_down_sync(0xFFFFFFFF, sum_excl, offset);
  }

  if (lane_id == 0)
  {
    CandSz[0] += sum_cand;
    ExclSz[0] -= sum_excl;
  }
  level[0]--;
  if (level[0] > 0) 
    cur_id[0] = stackBase[level[0]-1];

  __syncwarp();
}

__device__ int findID(int cur_id, unsigned int level, unsigned int n, uint8_t* labelsBase, unsigned int* depthBase)
{
  int next_id = -1;
  for (int i = cur_id + 1; i < n; i++)
  {
    if ((labelsBase[i] == C) && (depthBase[i] == level))
    {
      next_id = i;
      break;
    }
  }
  return next_id;
}

__device__ bool isMaximal(unsigned int lane_id, int k, unsigned int* left, unsigned int left_count, unsigned int* l_neighborsBase, unsigned int* l_offsetsBase, unsigned int* l_degreeBase, unsigned int* stack, unsigned int PlexSz, unsigned int* nonNeigh, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeBase)
{
  //printf("Checking Maximality\n");
  for (int i = 0; i < left_count; i++)
  {
    unsigned int u = i;
    int count = 0;
    for (int j = lane_id; j < PlexSz; j+=32)
    {
      unsigned int v = stack[j];
      if (!binary_search(u, l_neighborsBase + l_offsetsBase[v], l_degreeBase[v]))
        count++;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
    {
      count += __shfl_down_sync(0xFFFFFFFF, count, offset);
    }
    count = __shfl_sync(0xFFFFFFFF, count, /*srcLane=*/0);
    if (count > (k-1)) continue;
    bool local_invalid = false;
    for (int j = lane_id; j < PlexSz; j+=32)
    {
      unsigned int v = stack[j];
      bool bad = (nonNeigh[v] >= (k-1)) && (!binary_search(u, l_neighborsBase + l_offsetsBase[v], l_degreeBase[v]));

      local_invalid |= bad;
    }

    unsigned int mask = __activemask();
    bool warp_invalid = __any_sync(mask, local_invalid);
    bool validExtension = !warp_invalid;
    if (validExtension)
    {
      return false;
    }
  }
return true;
}

__global__ void BKIterative(int i, P_pointers p, S_pointers s, G_pointers g, unsigned int *d_blk, unsigned int *d_left,
                            unsigned int *d_left_counter, unsigned int* plex_count, unsigned int* nonNeigh,
                            unsigned int* nonNeighLeft, unsigned int* depth, unsigned int* stack, unsigned int *global_count)
{
  unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int warp_id = (global_index / 32);
  unsigned int lane_id = threadIdx.x % 32;

  int k = p.k;
  int q = p.lb;

  unsigned int* blkBase = d_blk + warp_id * MAX_BLK_SIZE;
  unsigned int* leftBase = d_left + warp_id * MAX_BLK_SIZE;
  unsigned int* left_counter = d_left_counter + warp_id;
  unsigned int leftSz = left_counter[0];

  unsigned int* neighborsBase = s.neighbors + warp_id * (MAX_BLK_SIZE) * (AVG_DEGREE);
  unsigned int* l_neighborsBase = s.l_neighbors + warp_id * (MAX_BLK_SIZE) * (AVG_LEFT_DEGREE);

  unsigned int* offsetsBase = s.offsets + warp_id * (MAX_BLK_SIZE);
  unsigned int* l_offsetsBase = s.l_offsets + warp_id * (MAX_BLK_SIZE);

  unsigned int* degreeBase = s.degree + warp_id * (MAX_BLK_SIZE);
  unsigned int* l_degreeBase = s.l_degree + warp_id * (MAX_BLK_SIZE);

  unsigned int* local_n = s.n + warp_id;
  unsigned int* PlexSz = s.PSize + warp_id;
  unsigned int* CandSz = s.CSize + warp_id;
  unsigned int* ExclSz = s.XSize + warp_id;
  uint8_t* labelsBase = s.labels + warp_id * (MAX_BLK_SIZE);

  unsigned int* nonNeighBase = nonNeigh + warp_id * MAX_BLK_SIZE;
  unsigned int* nonNeighLeftBase = nonNeighLeft + warp_id * MAX_BLK_SIZE;
  unsigned int* depthBase = depth + warp_id * MAX_BLK_SIZE;
  unsigned int* stackBase = stack + warp_id * MAX_BLK_SIZE;

  if ((warp_id+WARPS*i) >= g.n)  return;

  if (local_n[0] < q) return;
  if (warp_id == 51)
  {
    //if (lane_id == 0) printf("Left Count: %d\n", left_counter[0]);
    int level = 0;
    int cur_id = 0;
    //int depth = 0;
    push(lane_id, k, &cur_id, &level, local_n[0], left_counter[0], neighborsBase, l_neighborsBase, offsetsBase, l_offsetsBase, degreeBase, l_degreeBase, labelsBase,
         nonNeighBase, nonNeighLeftBase, depthBase, stackBase, &leftSz, PlexSz, CandSz, ExclSz);   
    while (level != 0)
    {
    //printf("Level: %d\n", level);
    //   if (lane_id == 0)
    //   {
    //   printf("\ncur_id = %d, Level %d PlexSz: %d CandSz %d ExclSz %d LeftSz %d\n", cur_id, level, PlexSz[0], CandSz[0], ExclSz[0], leftSz);
    //   printf("Labels Array: ");
    //   for (int i = 0; i < local_n[0]; i++)
    //   {
    //     printf("%d ", labelsBase[i]);
    //   }
    //   printf("\n");
    //   printf("depth Array: ");
    //   for (int i = 0; i < local_n[0]; i++)
    //   {
    //     printf("%d ", depthBase[i]);
    //   }
    //   printf("\n");
    //   printf("NonNeigh Array: ");
    //   for (int i = 0; i < local_n[0]; i++)
    //   {
    //     printf("%d ", nonNeighBase[i]);
    //   }
    //   printf("\n");
    //   printf("NonNeigh Left Array: ");
    //   for (int i = 0; i < left_counter[0]; i++)
    //   {
    //     printf("%d ", nonNeighLeftBase[i]);
    //   }
    //   printf("\n");
    // }
      //if (depth == 1000000) break;
      if (PlexSz[0] + CandSz[0] < q)
      {
        //if (lane_id == 0) printf("Pruned due to less size than q\n");
        pop(lane_id, k, &cur_id, &level, local_n[0], left_counter[0], neighborsBase, l_neighborsBase, offsetsBase, l_offsetsBase, degreeBase, l_degreeBase, labelsBase,
         nonNeighBase, nonNeighLeftBase, depthBase, stackBase, &leftSz, PlexSz, CandSz, ExclSz);
      }
      else
      {
        if ((CandSz[0] == 0) && (ExclSz[0] == 0) && isMaximal(lane_id, k, leftBase, left_counter[0], l_neighborsBase, l_offsetsBase, l_degreeBase, stackBase, PlexSz[0], nonNeighBase, neighborsBase, offsetsBase, degreeBase))
        {
          if (lane_id == 0)
          {
            atomicAdd(&plex_count[0], 1);
          }
          pop(lane_id, k, &cur_id, &level, local_n[0], left_counter[0], neighborsBase, l_neighborsBase, offsetsBase, l_offsetsBase, degreeBase, l_degreeBase, labelsBase,
         nonNeighBase, nonNeighLeftBase, depthBase, stackBase, &leftSz, PlexSz, CandSz, ExclSz);
        }
        else if (CandSz[0] > 0)
        {
          //if (lane_id == 0) printf("Pushing another node to Plex\n");
          cur_id = findID(cur_id, level, local_n[0], labelsBase, depthBase);
          if (lane_id == 0)
          {
            if (cur_id == -1)
            {
              printf("Invalid cur_id\n");
              break;
            }
          }
          push(lane_id, k, &cur_id, &level, local_n[0], left_counter[0], neighborsBase, l_neighborsBase, offsetsBase, l_offsetsBase, degreeBase, l_degreeBase, labelsBase,
               nonNeighBase, nonNeighLeftBase, depthBase, stackBase, &leftSz, PlexSz, CandSz, ExclSz);
        }
        else
        {
          //if (lane_id == 0) printf("Popping node from Plex\n");
          pop(lane_id, k, &cur_id, &level, local_n[0], left_counter[0], neighborsBase, l_neighborsBase, offsetsBase, l_offsetsBase, degreeBase, l_degreeBase, labelsBase,
         nonNeighBase, nonNeighLeftBase, depthBase, stackBase, &leftSz, PlexSz, CandSz, ExclSz);
        }
      }
      //depth++;
    }
    if (lane_id == 0)
    {
    atomicAdd(&global_count[0], 1);
    printf("%d warp is finished with Maximal %d-Plexes: %d.\n", warp_id, k, plex_count[0]);
    labelsBase[0] = C;
    ExclSz[0]--;
    CandSz[0]++;
    }
  }
}

__device__ void push2(G_pointers g, unsigned int* blk, int lane_id, unsigned int k, int* cur_id, int * level, unsigned int n, unsigned int left_count,  
                     uint8_t* labelsBase, unsigned int* nonNeighBase, unsigned int* depthBase,
                     unsigned int* stackBase, unsigned int leftSz, unsigned int* PlexSz, unsigned int* CandSz,
                     unsigned int* ExclSz)
{
  int local_flag = 0;
  if (lane_id == 0)
  {
  for (int vtx = 0; vtx < n; vtx++)
  {
    if (labelsBase[vtx] == P)
    {
    //unsigned int vtx = stackBase[i];
    if ((nonNeighBase[vtx]+1 > (k-1)) && (!basic_search(blk[vtx], g.neighbors + g.offsets[blk[cur_id[0]]], g.degree[blk[cur_id[0]]])))
    {
      depthBase[cur_id[0]]--;
      CandSz[0]--;
      cur_id[0] = stackBase[level[0] - 1];
      local_flag = 1;
      break;
    }
  }
  }
}
  __syncwarp();
  int bcur = __shfl_sync(0xFFFFFFFF, cur_id[0], 0);
  cur_id[0] = bcur;
  __syncwarp();

  int warp_any = __any_sync(0xFFFFFFFF, local_flag);
  if (warp_any)
  {
    return;
  }

  if (lane_id == 0)
  {
  labelsBase[cur_id[0]] = P;
  PlexSz[0]++;
  CandSz[0]--;
  stackBase[level[0]] = cur_id[0];
  }

  __syncwarp();

  int local_cand = 0;
  int local_excl = 0;
  //if (lane_id == 0){
  for (int i = lane_id; i < n; i+=32)
  {
    if (i == cur_id[0]) continue;

    if (!basic_search(blk[i], g.neighbors + g.offsets[blk[cur_id[0]]], g.degree[blk[cur_id[0]]]))
      nonNeighBase[i]++;
    
    if (labelsBase[i] == C && depthBase[i] == level[0])
    {
      if (nonNeighBase[i] <= k - 1)
        depthBase[i]++;
      else
        local_cand++;

    }

    else if (labelsBase[i] == X && depthBase[i] == level[0])
    {
      if (nonNeighBase[i] <= k - 1)
        depthBase[i]++;
      else
        local_excl++;
    }
  }
//}

  int sum_cand = local_cand;
  int sum_excl = local_excl;

  for (int offset = 16; offset > 0; offset >>= 1)
  {
    sum_cand += __shfl_down_sync(0xFFFFFFFF, sum_cand, offset);
    sum_excl += __shfl_down_sync(0xFFFFFFFF, sum_excl, offset);
  }

  if (lane_id == 0)
  {
    CandSz[0] -= sum_cand;
    ExclSz[0] -= sum_excl;
  }

  __syncwarp();
  level[0]++;
}

__device__ void pop2(G_pointers g, unsigned int* blk, int lane_id, unsigned int k, int* cur_id, int * level, unsigned int n, unsigned int left_count,
                     uint8_t* labelsBase, unsigned int* nonNeighBase, unsigned int* depthBase,
                     unsigned int* stackBase, unsigned int leftSz, unsigned int* PlexSz, unsigned int* CandSz,
                     unsigned int* ExclSz)
{
  int temp = 0;
  int local_excl = 0;
  int sum_excl = 0;
  for (int vtx = 0; vtx < n; vtx++)
  {
    if (labelsBase[vtx] == P)
    {
    local_excl = 0;
    //unsigned int vtx = stackBase[i];
    if (nonNeighBase[vtx] == (k-1))
    {
      //if (lane_id == 0){
      for (int j = lane_id; j < n; j+=32)
      {
        if (labelsBase[j] == X && depthBase[j] == level[0])
        {
          if (!basic_search(blk[j], g.neighbors + g.offsets[blk[vtx]], g.degree[blk[vtx]]))
            {
              //atomicAdd(&ExclSz[0], -1);
              local_excl++;
              depthBase[j]--;
              temp = 1;
            }
        }
      }
    //}
      sum_excl = local_excl;
      for (int offset = 16; offset > 0; offset >>= 1)
      {
        sum_excl += __shfl_down_sync(0xFFFFFFFF, sum_excl, offset);
      }
      if (lane_id == 0) ExclSz[0] -= sum_excl;
      int warp_any = __any_sync(0xFFFFFFFF, temp);
      if (warp_any)
      {
        return;
      }
    }
  }
  }
  __syncwarp();
  int local_cand = 0;
  local_excl = 0;
  //if (lane_id == 0){
  for (int i = lane_id; i < n; i+=32)
  {
    if (i == cur_id[0]) continue;

    if (depthBase[i] == (level[0] - 1))
    {
      if (labelsBase[i] == C) local_cand++;
        //atomicAdd(&CandSz[0], 1);
      if (labelsBase[i] == X) local_excl++;
        //atomicAdd(&ExclSz[0], 1);
    }
  }
  //}
  int sum_cand = local_cand;
  sum_excl = local_excl;

  for (int offset = 16; offset > 0; offset >>= 1)
  {
    sum_cand += __shfl_down_sync(0xFFFFFFFF, sum_cand, offset);
    sum_excl += __shfl_down_sync(0xFFFFFFFF, sum_excl, offset);
  }
  if (lane_id == 0)
  {
    CandSz[0] += sum_cand;
    ExclSz[0] += sum_excl;
  }

  //if (lane_id == 0) printf("CandSz: %d\n", CandSz[0]);

  __syncwarp();

  // if (lane_id == 0)
  // {
    for (int i = lane_id; i < n; i+=32)
    {
    if (i == cur_id[0]) continue;
    if (depthBase[i] == level[0])
      depthBase[i]--;
    
    if (!basic_search(blk[i], g.neighbors + g.offsets[blk[cur_id[0]]], g.degree[blk[cur_id[0]]]))
      nonNeighBase[i]--;
    }
  // }
  __syncwarp();

  if (lane_id == 0)
  {
  labelsBase[cur_id[0]] = X;
  PlexSz[0]--;
  ExclSz[0]++;
  }
  __syncwarp();

  //printf("Nodes coming back: ");
  local_cand = 0;
  local_excl = 0;
  //if (lane_id == 0){
  for (int i = lane_id + cur_id[0] + 1; i < n; i+=32)
  {
    if (labelsBase[i] == X)
    {
      labelsBase[i] = C;
      // atomicAdd(&CandSz[0], 1);
      // atomicAdd(&ExclSz[0], -1);
      local_cand++;
      local_excl++;
      //printf("%d ", i);
    }
  }
//}

  sum_cand = local_cand;
  sum_excl = local_excl;

  for (int offset = 16; offset > 0; offset >>= 1)
  {
    sum_cand += __shfl_down_sync(0xFFFFFFFF, sum_cand, offset);
    sum_excl += __shfl_down_sync(0xFFFFFFFF, sum_excl, offset);
  }

  if (lane_id == 0)
  {
    CandSz[0] += sum_cand;
    ExclSz[0] -= sum_excl;
  }
  level[0]--;
  if (level[0] > 0) 
    cur_id[0] = stackBase[level[0]-1];

  __syncwarp();
}

__device__ bool isNeighbor(int u, int v, G_pointers g)
{
  int begin = g.offsets[u];
  int end = g.offsets[u+1];
  for (int i = begin; i < end; i++)
  {
    if (g.neighbors[i] == v) return true;
  }
  return false;
}

__device__ void update_missing_add(int v, unsigned int* blk, unsigned int* missing, G_pointers g, unsigned int n)
{
  for (int i = 0; i < n; i++)
  {
    if (i != v)
    {
      if (!isNeighbor(blk[i], blk[v], g)) missing[i]++;
    }
  }
}

__device__ void update_missing_remove(int v, unsigned int* blk, unsigned int* missing, uint8_t * labelsBase, uint8_t label, G_pointers g, unsigned int n)
{
  for (int i = 0; i < n; i++)
  {
    if (labelsBase[i] == label)
    {
      if (!isNeighbor(blk[i], blk[v], g)) missing[i]--;
    }
  }
}

__device__ bool isKplex(int lane_id, int v, int k, unsigned int PlexSz, unsigned int* neiInP, uint8_t* labelsBase, unsigned int n, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeBase)
{
  unsigned mask = __activemask();
  if (neiInP[v] + k <  (PlexSz+1))
  {
    return false;
  }
  bool localPass = true;
  for (int i = lane_id; i < n; i+=32)
  {
    if (labelsBase[i] == P)
    {
      if (neiInP[i] + k == PlexSz && !binary_search(i, neighborsBase + offsetsBase[v], degreeBase[v]))
        localPass = false;
    }
  }
  if (__any_sync(mask, !localPass)) return false;
  return true;
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


__device__ bool isKplexPC(int v, int k, unsigned int PlexSz, unsigned int* missing, uint8_t* labelsBase, unsigned int n, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeBase)
{
  if (missing[v] + k < (PlexSz + 1))
  {
    return false;
  }
  for (int i = 0; i < n; i++)
  {
    if (labelsBase[i] == P || labelsBase[i] == C)
    {
      if (missing[i] + k == (PlexSz) && !binary_search(i, neighborsBase + offsetsBase[v], degreeBase[v]))
        return false;
    }
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
    // if (i == j) continue;

    // if (basic_search(i, neighborsBase + offsetsBase[j], degreeBase[j]))
    //   neiInG[i]--;
    int nei = neighborsBase[offsetsBase[j]+i];
    neiInG[nei]--;
  }
}

__device__ void addG(int lane_id, int j, uint16_t* neiInG, unsigned int n, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeBase)
{
  for (int i = lane_id; i < degreeBase[j]; i+=32)
  {
    // if (i == j) continue;

    // if (basic_search(i, neighborsBase + offsetsBase[j], degreeBase[j]))
    //   neiInG[i]--;
    int nei = neighborsBase[offsetsBase[j]+i];
    neiInG[nei]++;
  }
  __syncwarp();
}

__device__ bool isMaximal2(unsigned int lane_id, int k, unsigned int PlexSz, unsigned int* left, unsigned int left_count, unsigned int* l_neighborsBase, unsigned int* l_offsetsBase, unsigned int* l_degreeBase, unsigned int* nonNeigh, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeBase, uint8_t* labels, unsigned int n)
{
  // if (lane_id == 0)
  // {
  //printf("Checking Maximality\n");
  for (int i = 0; i < left_count; i++)
  {
    unsigned int u = i;
    int count = 0;
    for (int j = lane_id; j < n; j+=32)
    {
      if (labels[j] == P)
      {
        if (!binary_search(u, l_neighborsBase + l_offsetsBase[j], l_degreeBase[j]))
          count++;
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1)
    {
      count += __shfl_down_sync(0xFFFFFFFF, count, offset);
    }
    count = __shfl_sync(0xFFFFFFFF, count, /*srcLane=*/0);
    if (count > (k-1)) continue;
    bool local_invalid = false;
    for (int j = lane_id; j < n; j+=32)
    {
      if (labels[j] == P)
      {
        bool bad = (nonNeigh[j] + k < PlexSz + 1) && (!binary_search(u, l_neighborsBase + l_offsetsBase[j], l_degreeBase[j]));

        local_invalid |= bad;
      }
    }

    unsigned int mask = __activemask();
    bool warp_invalid = __any_sync(mask, local_invalid);
    bool validExtension = !warp_invalid;
    if (validExtension)
    {
      return false;
    }
  }
return true;
  // }
}

__device__ bool isMaximal22(unsigned int lane_id, int k, unsigned int PlexSz, unsigned int* left, unsigned int left_count, unsigned int* l_neighborsBase, unsigned int* l_offsetsBase, unsigned int* l_degreeBase, uint16_t* nonNeigh, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeBase, unsigned int* plex, unsigned int n)
{
  // if (lane_id == 0)
  // {
  //printf("Checking Maximality\n");

  for (int i = 0; i < left_count; i++)
  {
    unsigned int u = i;

    int count = 0;
    for (int j = lane_id; j < PlexSz; j+=32)
    {
      const int v = plex[j];
      if (!binary_search(u, l_neighborsBase + l_offsetsBase[v], l_degreeBase[v]))
        count++;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
    {
      count += __shfl_down_sync(0xFFFFFFFF, count, offset);
    }
    count = __shfl_sync(0xFFFFFFFF, count, /*srcLane=*/0);
    if (count > (k-1)) continue;
    bool local_invalid = false;
    for (int j = lane_id; j < PlexSz; j+=32)
    {
      const int v = plex[j];
      bool bad = (nonNeigh[v] + k < PlexSz + 1) && (!binary_search(u, l_neighborsBase + l_offsetsBase[v], l_degreeBase[v]));

      local_invalid |= bad;
    }

    unsigned mask = __activemask();
    bool warp_invalid = __any_sync(mask, local_invalid);
    bool validExtension = !warp_invalid;
    if (validExtension)
    {
      return false;
    }
  }
return true;
  // }
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
  
  // bool max = true;
  // if (lane_id == 0)
  // {
  //   int nsat = 0;
  //   for (int i = 0; i < PlexSz; i++)
  //   {
  //     uint16_t v = plex[i];
  //     if (nonNeigh[v] + k < PlexSz + 1)
  //     {
  //       local_sat[nsat++] = v;
  //     }
  //   }
  //   if(nsat)
  //   {
  //     uint16_t v0 = local_sat[0];
  //     int start = l_offsetsBase[v0];
  //     int end = l_offsetsBase[v0+1];
  //     int szcom = 0;

  //     for (int i = start; i < end; i++)
  //     {
  //       local_commons[szcom++] = l_neighborsBase[i];
  //     }
  //     for (int i = 1; i < nsat; i++)
  //     {
  //       uint16_t v = local_sat[i];
  //       szcom = intersection(local_commons, l_neighborsBase+l_offsetsBase[v], szcom, l_degreeBase[v], local_commons);
  //     }
  //     if (nsat == 1)
  //     {
  //       for (int i = 0; i < k; i++)
  //       {
  //         unsigned int u = plex[i];
  //         int start = l_offsetsBase[u];
  //         int end = l_offsetsBase[u+1];
  //         for (int j = start; j < end; j++)
  //         {
  //           unsigned int v = l_neighborsBase[j];
  //           int wordIdx = v / 32;
  //           int bitIdx = v % 32;
  //           uni[wordIdx] |= (1u << bitIdx);
  //         }
  //       }
  //       int write = 0;
  //       for (int i = 0; i < szcom; i++)
  //       {
  //         uint16_t v = local_commons[i];
  //         if (uni[v >> 5] >> (v & 31) & 1)
  //         {
  //           local_commons[write++] = local_commons[i];
  //         }
  //       }
  //       szcom = write;
  //     }

  //     for (int i = 0; i < szcom; i++)
  //     {
  //       uint16_t u = local_commons[i];
  //       if (canGlobalAdd(u, k, PlexSz, plex, l_neighborsBase, l_offsetsBase, l_degreeBase))
  //       {
  //         max = false;
  //         break;
  //       }
  //     }
  //     for (int i = 0; i < 32; i++)
  //     {
  //       uni[i] = 0u;
  //     }
  //   }
  //   else
  //   {
  //     int szcom = 0;
  //     for (int i = 0; i < k; i++)
  //     {
  //       unsigned int u = plex[i];
  //       int start = l_offsetsBase[u];
  //       int end = l_offsetsBase[u+1];
  //       for (int j = start; j < end; j++)
  //       {
  //         unsigned int v = l_neighborsBase[j];
  //         if(uni[v >> 5] >> (v & 31) & 1) continue;
  //         local_commons[szcom++] = v;
  //         int wordIdx = v / 32;
  //         int bitIdx = v % 32;
  //         uni[wordIdx] |= (1u << bitIdx);
  //       }
  //     }

  //     for (int i = 0; i < szcom; i++)
  //     {
  //       uint16_t u = local_commons[i];
  //       if (canGlobalAdd(u, k, PlexSz, plex, l_neighborsBase, l_offsetsBase, l_degreeBase))
  //       {
  //         max = false;
  //         break;
  //       }
  //     }
  //     for (int i = 0; i < 32; i++)
  //     {
  //       uni[i] = 0u;
  //     }
  //   }
  // }
  // max = __shfl_sync(0xFFFFFFFF, max, 0);

  // return max;

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
__device__ bool isMaximalPC(unsigned int lane_id, int k, unsigned int PlexSz, unsigned int* left, unsigned int left_count, unsigned int* l_neighborsBase, unsigned int* l_offsetsBase, unsigned int* l_degreeBase, unsigned int* nonNeigh, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeBase, uint8_t* labels, unsigned int n)
{
  //printf("Checking Maximality\n");
  for (int i = 0; i < left_count; i++)
  {
    unsigned int u = i;
    int count = 0;
    for (int j = lane_id; j < n; j+=32)
    {
      if (labels[j] == P || labels[j] == C)
      {
        if (!binary_search(u, l_neighborsBase + l_offsetsBase[j], l_degreeBase[j]))
          count++;
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1)
    {
      count += __shfl_down_sync(0xFFFFFFFF, count, offset);
    }
    count = __shfl_sync(0xFFFFFFFF, count, /*srcLane=*/0);
    if (count > (k-1)) continue;
    bool local_invalid = false;
    for (int j = lane_id; j < n; j+=32)
    {
      if (labels[j] == P || labels[j] == C)
      {
        bool bad = (nonNeigh[j] + k < PlexSz + 1) && (!binary_search(u, l_neighborsBase + l_offsetsBase[j], l_degreeBase[j]));

        local_invalid |= bad;
      }
    }

    unsigned int mask = __activemask();
    bool warp_invalid = __any_sync(mask, local_invalid);
    bool validExtension = !warp_invalid;
    if (validExtension)
    {
      return false;
    }
  }
return true;
}

__device__ bool isMaximalPC2(unsigned int lane_id, int k, unsigned int totalSz, unsigned int PlexSz, unsigned int CandSz, unsigned int* left, unsigned int left_count, unsigned int* l_neighborsBase, unsigned int* l_offsetsBase, unsigned int* l_degreeBase, uint16_t* nonNeigh, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeBase, unsigned int* plex, unsigned int* cand, unsigned int n)
{
  //printf("Checking Maximality\n");
  for (int i = 0; i < left_count; i++)
  {
    unsigned int u = i;
    int count = 0;
    // for (int j = lane_id; j < n; j+=32)
    // {
    //   if (labels[j] == P || labels[j] == C)
    //   {
    //     if (!binary_search(u, l_neighborsBase + l_offsetsBase[j], l_degreeBase[j]))
    //       count++;
    //   }
    // }
    for (int j = lane_id; j < PlexSz; j+=32)
    {
        const int v = plex[j];
        if (!binary_search(u, l_neighborsBase + l_offsetsBase[v], l_degreeBase[v]))
          count++;
    }
    for (int j = lane_id; j < CandSz; j+=32)
    {
        const int v = cand[j];
        if (!binary_search(u, l_neighborsBase + l_offsetsBase[v], l_degreeBase[v]))
          count++;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
    {
      count += __shfl_down_sync(0xFFFFFFFF, count, offset);
    }
    count = __shfl_sync(0xFFFFFFFF, count, /*srcLane=*/0);
    if (count > (k-1)) continue;
    bool local_invalid = false;
    for (int j = lane_id; j < PlexSz; j+=32)
    {
        const int v = plex[j];
        bool bad = (nonNeigh[v] + k < totalSz + 1) && (!binary_search(u, l_neighborsBase + l_offsetsBase[v], l_degreeBase[v]));

        local_invalid |= bad;
    }
    for (int j = lane_id; j < CandSz; j+=32)
    {
        const int v = cand[j];
        bool bad = (nonNeigh[v] + k < totalSz + 1) && (!binary_search(u, l_neighborsBase + l_offsetsBase[v], l_degreeBase[v]));

        local_invalid |= bad;
    }

    //unsigned int mask = __activemask();
    bool warp_invalid = __any_sync(0xFFFFFFFF, local_invalid);
    bool validExtension = !warp_invalid;
    if (validExtension)
    {
      return false;
    }
  }
return true;
}

__device__ bool isMaximalPC_opt(unsigned int lane_id, int k, unsigned int PlexSz, unsigned int CandSz, unsigned int totalSz, unsigned int* left, unsigned int left_count, unsigned int* l_neighborsBase, unsigned int* l_offsetsBase, unsigned int* l_degreeBase, uint16_t* nonNeigh, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeBase, unsigned int* plex, unsigned int* cand,  unsigned int n, uint16_t* local_sat, uint16_t* local_commons, uint32_t* uni)
{
  const unsigned FULL_MASK = 0xFFFFFFFFu;
  auto warp_or = [&](unsigned v){
    for (int ofs = 16; ofs > 0; ofs >>= 1)
    {
      v |= __shfl_xor_sync(FULL_MASK, v, ofs);
    }
    return v;
  };

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


__device__ bool upperBoundK(int lane_id, int n, int k, int lb, uint8_t* labels, unsigned int* neiInG, unsigned int* nonAdjInP, unsigned int PlexSz, unsigned int* neiInP, bool* proper, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeBase)
{
  bool ok = true;
  for (int i = lane_id; i < n; i+=32)
  {
    if (labels[i] == P)
    {
      if (neiInG[i] + k < lb)
      {
        ok = false;
        break;
      }
      nonAdjInP[i] = PlexSz - neiInP[i];
    }
  }
  bool any_fail = __any_sync(0xFFFFFFFF, !ok);
  if (any_fail)
  {
    return false;
  }
  // return true;
  if (!proper[0]) return true;
  
    ok = false;
    int cnt = PlexSz;
    for (int i = 0; i < n; i++)
    {
      if (labels[i] == C)
      {
        int max_noadj = -1;
        int max_index = 0;
        for (int j = lane_id; j < n; j+=32)
        {
          if (labels[j] == P)
          {
            if (!binary_search(i, neighborsBase + offsetsBase[j], degreeBase[j]) && nonAdjInP[j] > max_noadj)
            {
              max_noadj = nonAdjInP[j];
              max_index = j;
            }
          }
        }

        int val = max_noadj;
        int idx = max_index;
        for (int offset = 16; offset > 0; offset <<= 1)
        {
          int other_val = __shfl_down_sync(0xFFFFFFFF, val, offset);
          int other_idx = __shfl_down_sync(0xFFFFFFFF, idx, offset);
          if (other_val > val)
          {
            val = other_val;
            idx = other_idx;
          }
        }

        if(lane_id == 0)
        {
          max_noadj = val;
          max_index = idx;
        }

        max_noadj = __shfl_sync(0xFFFFFFFF, max_noadj, 0);
        max_index = __shfl_sync(0xFFFFFFFF, max_index, 0);

        if (max_noadj < k)
        {
          cnt++;
          nonAdjInP[max_index]++;
        }
        if (cnt >= lb)
        {
          return true;
        }
      }
    }
    return cnt >= lb;
  }

  __device__ bool upperBoundK2(int lane_id, int n, int k, int lb, unsigned int* plex, unsigned int* cand1, uint16_t* neiInG, unsigned int* nonAdjInP, unsigned int PlexSz, unsigned int Cand1Sz, uint16_t* neiInP, bool* proper, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeBase)
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
      nonAdjInP[u] = PlexSz - neiInP[u];
  }
  bool any_fail = __any_sync(0xFFFFFFFF, !ok);
  if (any_fail)
  {
    return false;
  }
  return true;
  // if (!proper[0]) return true;
  
  //   ok = false;
  //   int cnt = PlexSz;
  //   for (int i = 0; i < Cand1Sz; i++)
  //   {
  //       const int u = cand1[i];
  //       int max_noadj = -1;
  //       int max_index = 0;
  //       for (int j = lane_id; j < PlexSz; j+=32)
  //       {
  //           const int v = plex[j];
  //           if (!binary_search(u, neighborsBase + offsetsBase[v], degreeBase[v]) && nonAdjInP[v] > max_noadj)
  //           {
  //             max_noadj = nonAdjInP[v];
  //             max_index = v;
  //           }
  //       }

  //       int val = max_noadj;
  //       int idx = max_index;
  //       for (int offset = 16; offset > 0; offset <<= 1)
  //       {
  //         int other_val = __shfl_down_sync(0xFFFFFFFF, val, offset);
  //         int other_idx = __shfl_down_sync(0xFFFFFFFF, idx, offset);
  //         if (other_val > val)
  //         {
  //           val = other_val;
  //           idx = other_idx;
  //         }
  //       }

  //       if(lane_id == 0)
  //       {
  //         max_noadj = val;
  //         max_index = idx;
  //       }

  //       max_noadj = __shfl_sync(0xFFFFFFFF, max_noadj, 0);
  //       max_index = __shfl_sync(0xFFFFFFFF, max_index, 0);

  //       if (max_noadj < k)
  //       {
  //         cnt++;
  //         nonAdjInP[max_index]++;
  //       }
  //       if (cnt >= lb)
  //       {
  //         return true;
  //       }
  //   }
  //   return cnt >= lb;
  }

  __device__ bool upperBound(int lane_id, int n, int k, int lb, unsigned int pivot, uint8_t* labels, unsigned int* neiInG, unsigned int* nonAdjInP, unsigned int PlexSz, unsigned int* neiInP, bool* proper, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeBase)
{
  bool ok = true;
  for (int i = lane_id; i < n; i+=32)
  {
    if (labels[i] == P)
    {
      if (neiInG[i] + k < lb)
      {
        ok = false;
        break;
      }
      nonAdjInP[i] = PlexSz - neiInP[i];
    }
  }
  bool any_fail = __any_sync(0xFFFFFFFF, !ok);
  if (any_fail)
  {
    return false;
  }
  // return true;
  if (!proper[0]) return true;
  
    ok = false;
    int cnt = k+neiInP[pivot];
    for (int i = 0; i < n; i++)
    {
      if (labels[i] == C)
      {
        int max_noadj = -1;
        int max_index = 0;
        if (binary_search(pivot, neighborsBase + offsetsBase[i], degreeBase[i]))
        {
        for (int j = lane_id; j < n; j+=32)
        {
          if (labels[j] == P)
          {
            if (!binary_search(i, neighborsBase + offsetsBase[j], degreeBase[j]) && nonAdjInP[j] > max_noadj)
            {
              max_noadj = nonAdjInP[j];
              max_index = j;
            }
          }
        }

        int val = max_noadj;
        int idx = max_index;
        for (int offset = 16; offset > 0; offset <<= 1)
        {
          int other_val = __shfl_down_sync(0xFFFFFFFF, val, offset);
          int other_idx = __shfl_down_sync(0xFFFFFFFF, idx, offset);
          if (other_val > val)
          {
            val = other_val;
            idx = other_idx;
          }
        }

        if(lane_id == 0)
        {
          max_noadj = val;
          max_index = idx;
        }

        max_noadj = __shfl_sync(0xFFFFFFFF, max_noadj, 0);
        max_index = __shfl_sync(0xFFFFFFFF, max_index, 0);

        if (max_noadj < k)
        {
          cnt++;
          nonAdjInP[max_index]++;
        }
        if (cnt >= lb)
        {
          return true;
        }
      }
    }
  }
    return cnt >= lb;
  }

  __device__ bool upperBound2(int lane_id, int n, int k, int lb, unsigned int pivot, unsigned int* plex, unsigned int* cand, uint16_t* neiInG, unsigned int* nonAdjInP, unsigned int PlexSz, unsigned int CandSz, uint16_t* neiInP, bool* proper, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeBase)
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
      nonAdjInP[v] = PlexSz - neiInP[v];
  }
  bool any_fail = __any_sync(0xFFFFFFFF, !ok);
  if (any_fail)
  {
    return false;
  }
  return true;
  //  if (!proper[0]) return true;
  
  //   ok = false;
  //   int cnt = k+neiInP[pivot];
  //   for (int i = 0; i < CandSz; i++)
  //   {
  //       unsigned int v = cand[i];
  //       int max_noadj = -1;
  //       int max_index = 0;
  //       if (binary_search(pivot, neighborsBase + offsetsBase[v], degreeBase[v]))
  //       {
  //       for (int j = lane_id; j < PlexSz; j+=32)
  //       {
  //           unsigned int u = plex[j];
  //           if (!binary_search(v, neighborsBase + offsetsBase[u], degreeBase[u]) && nonAdjInP[u] > max_noadj)
  //           {
  //             max_noadj = nonAdjInP[u];
  //             max_index = u;
  //           }
  //       }

  //       int val = max_noadj;
  //       int idx = max_index;
  //       for (int offset = 16; offset > 0; offset <<= 1)
  //       {
  //         int other_val = __shfl_down_sync(0xFFFFFFFF, val, offset);
  //         int other_idx = __shfl_down_sync(0xFFFFFFFF, idx, offset);
  //         if (other_val > val)
  //         {
  //           val = other_val;
  //           idx = other_idx;
  //         }
  //       }

  //       if(lane_id == 0)
  //       {
  //         max_noadj = val;
  //         max_index = idx;
  //       }

  //       max_noadj = __shfl_sync(0xFFFFFFFF, max_noadj, 0);
  //       max_index = __shfl_sync(0xFFFFFFFF, max_index, 0);

  //       if (max_noadj < k)
  //       {
  //         cnt++;
  //         nonAdjInP[max_index]++;
  //       }
  //       if (cnt >= lb)
  //       {
  //         return true;
  //       }
  //     }
  // }
  //   return cnt >= lb;
  }

  __device__ void warp_memcpy_u16_16B(uint16_t* __restrict__ dst, const uint16_t* __restrict__ src, int nElts)
  {
    int nVec = (nElts * sizeof(16)) >> 4;
    auto* __restrict__ d4 = reinterpret_cast<int4*>(dst);
    auto const* __restrict__ s4 = reinterpret_cast<const int4*>(src);
    int lane = threadIdx.x & 31;
    for (int i = lane; i < nVec; i+=32) d4[i] = s4[i];
    __syncwarp();
  }

__device__ void updateCand1(int lane_id, uint8_t* labelsBase, uint8_t* commonMtx, unsigned int* recCand1Base, uint16_t* neiInGBase, bool* proper, int sz, int n, int v2add, unsigned int* Cand1Sz, unsigned int* neighborsBase, unsigned int* degreeBase, unsigned int* offsetsBase)
{
  for (int i = 0; i < n; i++)
  {
    if (labelsBase[i] == C)
    {
      if (proper[0] && UNLINK2EQUAL > commonMtx[v2add * n + i])
      {
        if (lane_id == 0)
        {
        labelsBase[i] = V;
        Cand1Sz[0]--;
        recCand1Base[i] = sz;
        }
        subG(lane_id, i, neiInGBase, n, neighborsBase, offsetsBase, degreeBase);
      }
    }
  }
  __syncwarp();
}

__device__ void updateCand12(int lane_id, unsigned int* cand1, uint8_t* commonMtx, unsigned int* recCand1Base, uint16_t* neiInGBase, bool* proper, int sz, int n, int v2add, unsigned int* Cand1Sz, unsigned int* neighborsBase, unsigned int* degreeBase, unsigned int* offsetsBase)
{
  int len = 0;
  for (int i = 0; i < Cand1Sz[0]; i++)
  {
      unsigned int v = cand1[i];
      if (/*proper[0] && */UNLINK2EQUAL > commonMtx[v2add * n + v])
      {
        if (lane_id == 0)
        {
        int temp = cand1[Cand1Sz[0]-1];
        cand1[Cand1Sz[0]-1] = v;
        cand1[i] = temp;
        Cand1Sz[0]--;
        len++;
        }
        subG(lane_id, v, neiInGBase, n, neighborsBase, offsetsBase, degreeBase);
        i--;
      }
  }
  __syncwarp();
  if (lane_id == 0)
  {
    recCand1Base[sz-1] += len;
  }
  __syncwarp();
}

__device__ void updateCand2(uint8_t* labelsBase, uint8_t* commonMtx, unsigned int* recCand2Base, unsigned int* neiInGBase, bool* proper, int sz, int n, int v2add, unsigned int* Cand2Sz, unsigned int* neighborsBase, unsigned int* degreeBase, unsigned int* offsetsBase)
{
  for (int i = 0; i < n; i++)
  {
    if (labelsBase[i] == H)
    {
      if (proper[0] && UNLINK2EQUAL > commonMtx[v2add * n + i])
      {
        labelsBase[i] = J;
        Cand2Sz[0]--;
        recCand2Base[i] = sz;
      }
    }
  }
}

__device__ void updateCand22(unsigned int* cand2, uint8_t* commonMtx, unsigned int* recCand2Base, bool* proper, int sz, int n, int v2add, unsigned int* Cand2Sz)
{
  for (int i = 0; i < Cand2Sz[0]; i++)
  {
      unsigned int v = cand2[i];
      if (/*proper[0] && */UNLINK2EQUAL > commonMtx[v2add * n + v])
      {
        int temp = cand2[Cand2Sz[0]-1];
        cand2[Cand2Sz[0]-1] = v;
        cand2[i] = temp;
        Cand2Sz[0]--;
        recCand2Base[v] = sz-1;
        i--;
      }
  }
}

__device__ void updateExcl(uint8_t* labelsBase, uint8_t* commonMtx, unsigned int* recExclBase, unsigned int* neiInGBase, bool* proper, int sz, int n, int v2add, unsigned int* ExclSz, unsigned int* neighborsBase, unsigned int* degreeBase, unsigned int* offsetsBase)
{
  for (int i = 0; i < n; i++)
  {
    if (labelsBase[i] == X)
    {
      if (proper[0] && UNLINK2MORE > commonMtx[v2add * n + i])
      {
        labelsBase[i] = K;
        ExclSz[0]--;
        recExclBase[i] = sz;
      }
    }
  }
}

__device__ void updateExcl2(unsigned int* excl, uint8_t* commonMtx, unsigned int* recExclBase, bool* proper, int sz, int n, int v2add, unsigned int* ExclSz)
{
  for (int i = 0; i < ExclSz[0]; i++)
  {
      unsigned int v = excl[i];
      if (/*proper[0] && */UNLINK2MORE > commonMtx[v2add * n + v])
      {
        int temp = excl[ExclSz[0]-1];
        excl[ExclSz[0]-1] = v;
        excl[i] = temp;
        ExclSz[0]--;
        recExclBase[v] = sz-1;
        i--;
      }
  }
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

__device__ void updateExcl2_fast(int lane_id, unsigned int* excl, uint8_t* commonMtx, unsigned int* recExclBase, bool* proper, int sz, int n, int v2add, unsigned int* ExclSz)
{
  unsigned int write = 0;
  for (int i = 0; i < ExclSz[0]; i++)
  {
    const unsigned int v = excl[i];
    if (proper[0] && UNLINK2MORE > commonMtx[v2add * n + v])
    {
      recExclBase[v] = sz-1; 
    }
    else
    {
      // if (i != write) excl[write] = v;
      // write++;
      excl[write++] = v;
    }
  }
  ExclSz[0] = write;
}

__device__ void recoverCand1(int lane_id, uint8_t* labelsBase, uint8_t* commonMtx, unsigned int* recCand1Base, uint16_t* neiInGBase, bool* proper, int sz, int n, int v2add, unsigned int* Cand1Sz, unsigned int* neighborsBase, unsigned int* degreeBase, unsigned int* offsetsBase)
{
  for (int i = 0; i < n; i++)
  {
    if (proper[0] && labelsBase[i] == V && recCand1Base[i] == sz)
    {
      if (lane_id == 0)
      {
      labelsBase[i] = C;
      Cand1Sz[0]++;
      recCand1Base[i] = 0;
      }
      
      addG(lane_id, i, neiInGBase, n, neighborsBase, offsetsBase, degreeBase);
    }
  }
  __syncwarp();
}

__device__ void recoverCand12(int lane_id, unsigned int* cand1, unsigned int* recCand1Base, uint16_t* neiInGBase, int sz, int n, unsigned int* Cand1Sz, unsigned int* neighborsBase, unsigned int* degreeBase, unsigned int* offsetsBase)
{
  for (int i = 0; i < recCand1Base[sz-1]; i++)
  {
      if (lane_id == 0)
      {
      Cand1Sz[0]++;
      }
      __syncwarp();
      addG(lane_id, cand1[Cand1Sz[0]-1], neiInGBase, n, neighborsBase, offsetsBase, degreeBase);
  }
  recCand1Base[sz-1] = 0;
  __syncwarp();
}

__device__ void recoverCand2(uint8_t* labelsBase, uint8_t* commonMtx, unsigned int* recCand2Base, unsigned int* neiInGBase, bool* proper, int sz, int n, int v2add, unsigned int* Cand2Sz, unsigned int* neighborsBase, unsigned int* degreeBase, unsigned int* offsetsBase)
{
  for (int i = 0; i < n; i++)
  {
    if (proper[0] && labelsBase[i] == J && recCand2Base[i] == sz)
    {
      labelsBase[i] = H;
      Cand2Sz[0]++;
      recCand2Base[i] = 0;
    }
  }
}
__device__ void recoverCand22(int n, unsigned int* cand2, unsigned int* recCand2Base, int sz, unsigned int* Cand2Sz)
{
  for (int i = 0; i < n; i++)
  {
    if (recCand2Base[i] == sz-1)
    {
      cand2[Cand2Sz[0]++] = i;
      recCand2Base[i] = 0;
    }
  }
}

__device__ void recoverExcl2(int n, unsigned int* excl, unsigned int* recExclBase, int sz, unsigned int* ExclSz)
{
  for (int i = 0; i < n; i++)
  {
    if (recExclBase[i] == sz-1)
    {
      excl[ExclSz[0]++] = i;
      recExclBase[i] = 0;
    }
  }
}


__device__ void recoverExcl(uint8_t* labelsBase, uint8_t* commonMtx, unsigned int* recExclBase, unsigned int* neiInGBase, bool* proper, int sz, int n, int v2add, unsigned int* ExclSz, unsigned int* neighborsBase, unsigned int* degreeBase, unsigned int* offsetsBase)
{
  for (int i = 0; i < n; i++)
  {
    if (proper[0] && labelsBase[i] == K && recExclBase[i] == sz)
    {
      labelsBase[i] = X;
      ExclSz[0]++;
      recExclBase[i] = 0;
    }
  }
}

__device__ void enqueue_exclude_child(int lane_id, int idx, unsigned int* local_n, unsigned int* plex, unsigned int PlexSz, unsigned int* cand, unsigned int CandSz, unsigned int* excl, unsigned int ExclSz, int minIndex, Task* tasks, Task* global_tasks, unsigned int* tailPtr, unsigned int* global_tail, uint8_t* d_all_labels, uint16_t* d_all_neiInG, uint16_t* d_all_neiInP, uint8_t* global_labels, uint16_t* global_neiInG, uint16_t* global_neiInP, uint16_t* neiInG, uint16_t* neiInP, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeBase)
{
    uint8_t* labels = d_all_labels;
    uint16_t* all_neiInG = d_all_neiInG;
    uint16_t* all_neiInP = d_all_neiInP;
    Task* localTasks = tasks;
    unsigned int pos2;
    if (lane_id == 0) pos2 = atomicAdd(&tailPtr[0], 1u);
    pos2 = __shfl_sync(0xFFFFFFFF, pos2, 0);
    if (pos2 + 1> SMALL_CAP) 
    {
      // if (lane_id == 0) printf("Maximum Capacity Reached.\n");
      // //tailPtr[0] = 0;
      // return;
      if (lane_id == 0)
      {
      atomicAdd(&tailPtr[0], -1);
      pos2 = atomicAdd(&global_tail[0], 1u);
      }
      pos2 = __shfl_sync(0xFFFFFFFF, pos2, 0);
      if (pos2 + 1 >= MAX_CAP)
      {
        printf("Maximum Capacity Reached.\n");
        return;
      }
      labels = global_labels;
      all_neiInG = global_neiInG;
      all_neiInP = global_neiInP;
      localTasks = global_tasks;
    }
    uint8_t* childLabels2 = labels + pos2*MAX_BLK_SIZE;
    uint16_t* childNeiInG2 = all_neiInG + pos2*MAX_BLK_SIZE;
    uint16_t* childNeiInP2 = all_neiInP + pos2*MAX_BLK_SIZE;

    for (int j = lane_id; j < local_n[0]; j+=32)
    {
      childLabels2[j] = U;
      //childNeiInG2[j] = neiInG[j];
      //childNeiInP2[j] = neiInP[j];
    }
    warp_memcpy_u16_16B(childNeiInG2, neiInG, local_n[0]);
    warp_memcpy_u16_16B(childNeiInP2, neiInP, local_n[0]);

    for (int j = lane_id; j < PlexSz; j+=32)
    {
      unsigned int v = plex[j];
      childLabels2[v] = P;
    }
    for (int j = lane_id; j < CandSz; j+=32)
    {
      unsigned int v = cand[j];
      childLabels2[v] = C;
    }
    for (int j = lane_id; j < ExclSz; j+=32)
    {
      unsigned int v = excl[j];
      childLabels2[v] = X;
    }
    subG(lane_id, minIndex, childNeiInG2, local_n[0], neighborsBase, offsetsBase, degreeBase);
    if (lane_id == 0)
    {
    Task &nt2 = localTasks[pos2];
    nt2.idx = idx;
    childLabels2[minIndex] = X;
    nt2.PlexSz = PlexSz;
    nt2.CandSz = CandSz - 1;
    nt2.ExclSz = ExclSz + 1;
    nt2.labels = childLabels2;
    nt2.neiInG = childNeiInG2;
    nt2.neiInP = childNeiInP2;
    }
}


__device__ void enqueue_include_child(int lane_id, int idx, int k, int lb, unsigned int* local_n, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeBase, uint8_t* commonMtx, bool* proper, unsigned int* plex, unsigned int& PlexSz, unsigned int* cand, unsigned int& CandSz, unsigned int* excl, unsigned int& ExclSz, uint16_t* neiInP, uint16_t* neiInG, int minIndex, unsigned int* nonAdjInP, Task* tasks, Task* global_tasks, unsigned int* tailPtr, unsigned int* global_tail, uint8_t* d_all_labels, uint16_t* d_all_neiInG, uint16_t* d_all_neiInP, uint8_t* global_labels, uint16_t* global_neiInG, uint16_t* global_neiInP, unsigned long long* cycles, uint32_t* adjList, uint16_t* local_sat)
{

    uint8_t* labels = d_all_labels;
    uint16_t* all_neiInG = d_all_neiInG;
    uint16_t* all_neiInP = d_all_neiInP;
    Task* localTasks = tasks;
    
    if (lane_id == 0)
    {
      plex[PlexSz++] = minIndex;
      for (int i = 0; i < CandSz; i++)
      {
        if(cand[i] == minIndex)
        {
          int temp = cand[i];
          cand[i] = cand[CandSz-1];
          cand[CandSz-1] = temp;
          CandSz--;
          break;
        }
      }
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
     
    
    
    const uint8_t* __restrict__ commonRow = &commonMtx[minIndex * local_n[0]];
    for (int j = 0; j < CandSz; j++)
    {
        unsigned int v = cand[j];
        if ((/*proper[0] && */UNLINK2EQUAL>*(commonRow + v)) || !isKplex2(lane_id, v, k, PlexSz, neiInP, plex, local_n[0], neighborsBase, offsetsBase, degreeBase, adjList))
        {
          if (lane_id == 0)
          {
            int temp = cand[j];
            cand[j] = cand[CandSz-1];
            cand[CandSz-1] = temp;
          }
          CandSz--;
          j--;
          subG(lane_id, v, neiInG, local_n[0], neighborsBase, offsetsBase, degreeBase);
        }
    }

    __syncwarp();
   
    
    
    
    bool ub = upperBound2(lane_id, local_n[0], k, lb, minIndex, plex, cand, neiInG, nonAdjInP, PlexSz, CandSz, neiInP, proper, neighborsBase, offsetsBase, degreeBase);
    
    if (ub)
    {
      
      for (int j = 0; j < ExclSz; j++)
      {
          const unsigned int v = excl[j];
          if ((/*proper[0] && */UNLINK2MORE>*(commonRow + v)) || !isKplex2(lane_id, v, k, PlexSz, neiInP, plex, local_n[0], neighborsBase, offsetsBase, degreeBase, adjList))
          {
            if (lane_id == 0)
            {
              int temp = excl[j];
              excl[j] = excl[ExclSz-1];
              excl[ExclSz-1] = temp;
            }
            __syncwarp();
            ExclSz--;
            j--;
          }
      }

      __syncwarp();
      
      
      // __syncthreads();
      // unsigned long long t0 = clock64();
      unsigned int pos;
      if (lane_id == 0) pos = atomicAdd(&tailPtr[0], 1u);
      pos = __shfl_sync(0xFFFFFFFF, pos, 0);
      bool use_global = (pos + 1 > SMALL_CAP);

       if (use_global) 
        {
          //if (lane_id == 0) printf("Maximum Capacity Reached.\n");
          //return;
          if (lane_id == 0)
          {
          atomicAdd(&tailPtr[0], -1);
          pos = atomicAdd(&global_tail[0], 1u);
          }
          pos = __shfl_sync(0xFFFFFFFF, pos, 0);
          if (pos + 1 >= MAX_CAP)
          {
            printf("Maximum Capacity Reached.\n");
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
          // childNeiInG[j] = neiInG[j];
          // childNeiInP[j] = neiInP[j];
        }
        warp_memcpy_u16_16B(childNeiInG, neiInG, local_n[0]);
        warp_memcpy_u16_16B(childNeiInP, neiInP, local_n[0]);
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
        //printf("Pushing a task 1\n");
        Task &nt = localTasks[pos];
        nt.idx = idx;

        __syncwarp();
        if (lane_id == 0)
        {
          nt.PlexSz = PlexSz;
          nt.CandSz = CandSz;
          nt.ExclSz = ExclSz;
          nt.labels = childLabels;
          nt.neiInG = childNeiInG;
          nt.neiInP = childNeiInP;
        }
        

    //ExclSz += len2;
  }
    // int oldlen = CandSz;
    // CandSz += len;
    // for (int j = oldlen; j < CandSz; j++)
    // {
    //     const unsigned int v = cand[j];
    //     // if ((proper[0] && UNLINK2EQUAL>commonMtx[minIndex*local_n[0] + v]) || !isKplex2(lane_id, v, k, PlexSz, neiInP, plex, local_n[0], neighborsBase, offsetsBase, degreeBase))
    //     // {
    //       addG(lane_id, v, neiInG, local_n[0], neighborsBase, offsetsBase, degreeBase);
    //     // }
    // }
    // __syncwarp();
    // for (int j = lane_id; j < degreeBase[minIndex]; j+=32)
    // {
    //   const int nei = neighborsBase[offsetsBase[minIndex]+j];
    //   neiInP[nei]--;
    // }
    // if (lane_id == 0)
    // {
    //   PlexSz--;
    //   cand[CandSz++] = minIndex;
    // }
    // PlexSz = __shfl_sync(0xFFFFFFFF, PlexSz, 0);
    // CandSz = __shfl_sync(0xFFFFFFFF, CandSz, 0);
    __syncwarp();
    //return true;
}

__device__ void branchInCand2(int warp_id, int lane_id, int minIndex, int idx, int k, int lb, unsigned int PlexSz, unsigned int CandSz, unsigned int ExclSz, unsigned int* local_n, Task* tasks, Task* global_tasks, unsigned int* tailPtr, unsigned int* global_tail, unsigned int* plex, unsigned int* cand, unsigned int* excl, uint16_t* neiInG, uint16_t* neiInP, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeBase,uint8_t* d_all_labels, uint16_t* d_all_neiInG, uint16_t* d_all_neiInP, uint8_t* global_labels, uint16_t* global_neiInG, uint16_t* global_neiInP, uint8_t* commonMtx, bool* proper, unsigned int* nonAdjInP, unsigned long long* cycles, uint32_t* adjList, uint16_t* local_sat)
{
      
      enqueue_exclude_child(lane_id, idx, local_n, plex, PlexSz, cand, CandSz, excl, ExclSz, minIndex, tasks, global_tasks, tailPtr, global_tail, d_all_labels, d_all_neiInG, d_all_neiInP, global_labels, global_neiInG, global_neiInP, neiInG, neiInP, neighborsBase, offsetsBase, degreeBase);
      
      //__syncwarp();
      
      enqueue_include_child(lane_id, idx, k, lb, local_n, neighborsBase, offsetsBase, degreeBase, commonMtx, proper, plex, PlexSz, cand, CandSz, excl, ExclSz, neiInP, neiInG, minIndex, nonAdjInP, tasks, global_tasks, tailPtr, global_tail, d_all_labels, d_all_neiInG, d_all_neiInP, global_labels, global_neiInG, global_neiInP, cycles, adjList, local_sat);
      //printf("Pushing a task 2\n");
}

__device__ void initializePCX(int lane_id, const uint8_t* __restrict__ labelsBase, unsigned int n, unsigned int* __restrict__ plex, unsigned int* __restrict__ cand, unsigned int* __restrict__ excl)
{
  const unsigned FULL = 0xFFFFFFFFu;

  int p_base = 0, c_base = 0, x_base = 0;

  for (int i0 = 0; i0 < n; i0 += 32)
  {
    int i = i0 + lane_id;
    uint8_t lbl = (i < n) ? labelsBase[i] : 0xFF;

    unsigned p_mask = __ballot_sync(FULL, lbl == P);
    unsigned c_mask = __ballot_sync(FULL, lbl == C);
    unsigned x_mask = __ballot_sync(FULL, lbl == X);

    int p_rank = __popc(p_mask & ((1u << lane_id) - 1));
    int c_rank = __popc(c_mask & ((1u << lane_id) - 1));
    int x_rank = __popc(x_mask & ((1u << lane_id) - 1));

    int p_wbase = __shfl_sync(FULL, p_base, 0);
    int c_wbase = __shfl_sync(FULL, c_base, 0);
    int x_wbase = __shfl_sync(FULL, x_base, 0);

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

__global__ void BNB(int i, P_pointers p, S_pointers s, unsigned int* d_blk, unsigned int* d_left, unsigned int* d_blk_counter, unsigned int* d_left_counter, uint8_t* commonMtx, Task* tasks, Task* outTasks, Task* global_tasks, unsigned int N, unsigned int head, unsigned int* tailPtr, unsigned int* global_tail, uint8_t* d_all_labels, uint16_t* d_all_neiInG, uint16_t* d_all_neiInP, uint8_t* global_labels, uint16_t* global_neiInG, uint16_t* global_neiInP, unsigned int* plex_count, unsigned int* nonAdjInP, uint16_t* d_sat, uint16_t* d_commons, uint32_t* d_uni, unsigned long long* cycles, uint32_t* d_adj)
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
  unsigned int* nonAdjInPBase = nonAdjInP + warp_id * MAX_BLK_SIZE;
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
  
  
  // if (lane_id == 0)
  // {
  //   int j = 0, k = 0, l = 0;
  //   for (int i = 0; i < local_n[0]; i++)
  //   {
  //     if (labelsBase[i] == P) plex[j++] = i;
  //     else if (labelsBase[i] == C) cand[k++] = i;
  //     else if (labelsBase[i] == X) excl[l++] = i;
  //   }
  // }
  // __syncwarp();
  
  
  initializePCX(lane_id, labelsBase, local_n[0], plex, cand, excl);
  
 
  bool* proper = s.proper + t.idx;

  // __shared__ uint16_t uni[WARPS_EACH_BLK*32];
  // __shared__ unsigned int  commons[WARPS_EACH_BLK];
  // __shared__ uint32_t uni[WARPS_EACH_BLK * 32];

  // uint16_t* local_sat = sat[warp_id*512];
  // unsigned int * local_commons = commons + warp_id;
  // uint16_t* local_uni2 = uni[warp_id*32];

  // if (lane_id == 0) local_uni2[0] = 0;
  // // if (lane_id == 0 ) local_commons[0] = 0;

  // if (lane_id == 0) printf("local_uni: %d\n", local_uni2[0]);
  // if (lane_id == 0) printf("local_commons: %d\n", local_commons[0]);

  if (PlexSz + CandSz < q) return;

    // int P = PlexSz;
    // size_t shm_bytes = 5 * sizeof(int);
    // __shared__ unsigned int[]
    // extern __shared__ int shmem[];

    // uint16_t* myWarpMem = shmem + warp_id * shm_bytes;
    // myWarpMem[0] = 0;
    //isMaximalPC_opt(lane_id, k, PlexSz, CandSz, PlexSz+CandSz, leftBase, left_count[0], l_neighborsBase, l_offsetsBase, l_degreeBase, neiInP, neighborsBase, offsetsBase, degreeBase, plex, cand, local_n[0], local_sat, local_commons, local_uni);
    //bool isMax = isMaximal_opt(lane_id, k, PlexSz, leftBase, left_count[0], l_neighborsBase, l_offsetsBase, l_degreeBase, neiInP, neighborsBase, offsetsBase, degreeBase, plex, local_n[0], local_sat, local_commons, local_uni);
    
    

    if (CandSz == 0)
    {
       __syncwarp();
      if (ExclSz == 0 && PlexSz >= q &&
         isMaximal_opt(lane_id, k, PlexSz, leftBase, left_count[0], l_neighborsBase, l_offsetsBase, l_degreeBase, neiInP, neighborsBase, offsetsBase, degreeBase, plex, local_n[0], local_sat, local_commons, local_uni)
         /*isMaximal22(lane_id, k, PlexSz, leftBase, left_count[0], l_neighborsBase, l_offsetsBase, l_degreeBase, neiInP, neighborsBase, offsetsBase, degreeBase, plex, local_n[0])
         /*isMaximal2(lane_id, k, PlexSz, leftBase, left_count[0], l_neighborsBase, l_offsetsBase, l_degreeBase, neiInP, neighborsBase, offsetsBase, degreeBase, labelsBase, local_n[0])*/)
      {
        if(lane_id == 0) atomicAdd(&plex_count[0], 1);
      }
      return;
    }
    __syncwarp();
   
    
    int minnei_Plex = INT_MAX;
    int pivot = -1;
    int minnei_Cand = INT_MAX;

    // for (int i = lane_id; i < local_n[0]; i+=32)
    // {
    //     if (labelsBase[i] == P)
    //     {
    //         if (neiInG[i] < minnei_Plex)
    //         {
    //             minnei_Plex = neiInG[i];
    //             pivot = i;
    //         }
    //     }
    // }

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

    //if (lane_id == 0) printf("minnei_Plex: %d, pivot: %d\n", minnei_Plex, pivot);

    int pivot_plex = pivot;
    
    if (minnei_Plex + k  < max(q, PlexSz)) return;
    
    if (minnei_Plex + k < PlexSz + CandSz)
    {
        // printf("Going to select pivot again from N(C)'\n");
        
        minnei_Cand = INT_MAX;
        pivot = -1;
        // for (int i = lane_id; i < local_n[0]; i+=32)
        // {
        //     if (labelsBase[i] == C)
        //     {
        //         if (!binary_search(i, neighborsBase + offsetsBase[pivot_plex], degreeBase[pivot_plex]))
        //         {
        //             if (neiInG[i] < minnei_Cand)
        //             {
        //                 minnei_Cand = neiInG[i];
        //                 pivot = i;
        //             }
        //             else if (neiInG[i] == minnei_Cand && neiInP[pivot] > neiInP[i])
        //             {
        //                 pivot = i;
        //             }
        //         }
        //     }
        // }

        for (int i = lane_id; i < CandSz; i+=32)
        {
                const int v = cand[i];
                int check = v * local_n[0] + pivot_plex;
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
         
  //       // printf("Creating new branches\n");
        // branchInCand(warp_id, lane_id, pivot, t.idx, k, q, PlexSz, CandSz, ExclSz, local_n, outTasks, global_tasks, tailPtr, global_tail, labelsBase, neiInG, neiInP, neighborsBase, offsetsBase, degreeBase, d_all_labels, d_all_neiInG, d_all_neiInP, global_labels, global_neiInG, global_neiInP, commonMtxBase, proper, nonAdjInPBase);

        // __syncthreads();
        // unsigned long long t0 = read_global_timer();
        
        branchInCand2(warp_id, lane_id, pivot, t.idx, k, q, PlexSz, CandSz, ExclSz, local_n, outTasks, global_tasks, tailPtr, global_tail, plex, cand, excl, neiInG, neiInP, neighborsBase, offsetsBase, degreeBase, d_all_labels, d_all_neiInG, d_all_neiInP, global_labels, global_neiInG, global_neiInP, commonMtxBase, proper, nonAdjInPBase, cycles, adjList, local_sat);
        // unsigned long long t1 = read_global_timer();
        // if (lane_id == 0)
        // {
        //   atomicAdd(&time[0], t1 - t0);
        // }
        // __syncthreads();
        return;
    }

   
   
    int minnei = minnei_Plex;
    // for (int i = lane_id; i < local_n[0]; i+=32)
    // {
    //     if (labelsBase[i] == C)
    //     {
    //         if (neiInG[i] < minnei)
    //         {
    //             minnei = neiInG[i];
    //             pivot = i;
    //         }
    //         else if (neiInG[i] == minnei && neiInP[pivot] > neiInP[i])
    //         {
    //             pivot = i;
    //         }
    //     }
    // }

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
  // if (lane_id == 0) printf("0");
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
      
        // printf("Plex Found 2\n");
        if (PlexSz + CandSz < q) return;
        bool flag = false;
        // for (int i = lane_id; i < local_n[0]; i+=32)
        // {
        //     if (labelsBase[i] == X)
        //     {
        //         if (isKplexPC(i, k, PlexSz+CandSz, neiInG, labelsBase, local_n[0], neighborsBase, offsetsBase, degreeBase))
        //         {
        //             flag = true;
        //         }
        //     }
        // }
        
        for (int i = lane_id; i < ExclSz; i+=32)
        {
                const int v = excl[i];
                if (isKplexPC2(v, k, PlexSz+CandSz, PlexSz, CandSz, neiInG, plex, cand, local_n[0], neighborsBase, offsetsBase, degreeBase, adjList))
                {
                    flag = true;
                }
        }
        
        if (__any_sync(0xFFFFFFFF, flag)) return;
        // if (isMaximalPC_opt(lane_id, k, PlexSz, CandSz, PlexSz+CandSz, leftBase, left_count[0], l_neighborsBase, l_offsetsBase, l_degreeBase, neiInP, neighborsBase, offsetsBase, degreeBase, plex, cand, local_n[0], local_sat, local_commons, local_uni) != isMaximalPC2(lane_id, k, PlexSz+CandSz, PlexSz, CandSz, leftBase, left_count[0], l_neighborsBase, l_offsetsBase, l_degreeBase, neiInG, neighborsBase, offsetsBase, degreeBase, plex, cand, local_n[0]))
        // {
        //   return;
        // }
         
        if (isMaximalPC_opt(lane_id, k, PlexSz, CandSz, PlexSz+CandSz, leftBase, left_count[0], l_neighborsBase, l_offsetsBase, l_degreeBase, neiInG, neighborsBase, offsetsBase, degreeBase, plex, cand, local_n[0], local_sat, local_commons, local_uni)/*isMaximalPC2(lane_id, k, PlexSz+CandSz, PlexSz, CandSz, leftBase, left_count[0], l_neighborsBase, l_offsetsBase, l_degreeBase, neiInG, neighborsBase, offsetsBase, degreeBase, plex, cand, local_n[0])/*isMaximalPC(lane_id, k, PlexSz+CandSz, leftBase, left_count[0], l_neighborsBase, l_offsetsBase, l_degreeBase, neiInG, neighborsBase, offsetsBase, degreeBase, labelsBase, local_n[0])*/)
        {
            // printf("Maximal k-Plex Found 2\n");
            if (lane_id == 0) atomicAdd(&plex_count[0], 1);
        }
        
        return;
    }
    
    // branchInCand(warp_id, lane_id, pivot, t.idx, k, q, PlexSz, CandSz, ExclSz, local_n, outTasks, global_tasks, tailPtr, global_tail, labelsBase, neiInG, neiInP, neighborsBase, offsetsBase, degreeBase, d_all_labels, d_all_neiInG, d_all_neiInP, global_labels, global_neiInG, global_neiInP, commonMtxBase, proper, nonAdjInPBase);
    branchInCand2(warp_id, lane_id, pivot, t.idx, k, q, PlexSz, CandSz, ExclSz, local_n, outTasks, global_tasks, tailPtr, global_tail, plex, cand, excl, neiInG, neiInP, neighborsBase, offsetsBase, degreeBase, d_all_labels, d_all_neiInG, d_all_neiInP, global_labels, global_neiInG, global_neiInP, commonMtxBase, proper, nonAdjInPBase, cycles, adjList, local_sat);
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

__global__ void buildCommonMtx(int idx, P_pointers p, S_pointers s, G_pointers g, uint8_t* commonMtx, unsigned int* d_hopSz)
{
  unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int warp_id = (global_index / 32);
  unsigned int lane_id = threadIdx.x % 32;

  //if (lane_id == 0) printf("N: %d, warp_id: %d, i: %d\n", N, warp_id, i);
  // if ((warp_id+WARPS*idx) >= g.n)  return;
  if ((warp_id+WARPS*idx) >= (g.n-p.lb+2)) return;

  int k = p.k;
  int lb = p.lb;
  bool *proper = s.proper + warp_id;
  //if (!proper[0]) return;
  unsigned int* local_n = s.n + warp_id;
  unsigned int *hopSz = d_hopSz + warp_id;

  unsigned int* degreeBase = s.degree + warp_id * MAX_BLK_SIZE;
  unsigned int* l_degreeBase = s.l_degree + warp_id * MAX_BLK_SIZE;
  unsigned int* offsetsBase = s.offsets + warp_id * MAX_BLK_SIZE;
  unsigned int* neighborsBase = s.neighbors + warp_id * MAX_BLK_SIZE * AVG_DEGREE;
  unsigned int* l_offsetsBase = s.l_offsets + warp_id * MAX_BLK_SIZE;
  unsigned int* l_neighborsBase = s.l_neighbors + warp_id * MAX_BLK_SIZE * AVG_LEFT_DEGREE;
  unsigned int* degreeHop = s.degreeHop + warp_id * MAX_BLK_SIZE;

  uint8_t* commonMtxBase = commonMtx + warp_id * CAP;
    const int thresPP1=lb-k-2*max(k-2,0),thresPP2=lb-k-2*max(k-3,0);
    const int thresPC1=lb-2*k-max(k-2,0),thresPC2=lb-k-1-max(k-2,0)-max(k-3,0);
    const int thresCC1=lb-2*k-(k-1),thresCC2=lb-2*k+2-(k-1);
    
    for (int i = lane_id; i < hopSz[0]; i+=32)
    {
      for (int j = 0; j < i; j++)
      {
        const int common = commonEle(i, j, neighborsBase, offsetsBase, degreeHop);
        if (commonMtxBase[i*local_n[0]+j])
        {
          if(common>thresCC1)commonMtxBase[i*local_n[0]+j]=LINK2MORE;
          else if(common==thresCC1)commonMtxBase[i*local_n[0]+j]=LINK2EQUAL;
          else commonMtxBase[i*local_n[0]+j]=LINK2LESS;
        }
        else{
          if(common>thresCC2)commonMtxBase[i*local_n[0]+j]=UNLINK2MORE;
          else if(common==thresCC2)commonMtxBase[i*local_n[0]+j]=UNLINK2EQUAL;
          else commonMtxBase[i*local_n[0]+j]=UNLINK2LESS;
        }
        commonMtxBase[j*local_n[0]+i] = commonMtxBase[i*local_n[0]+j];
      }
    }
    for (int i = hopSz[0]+lane_id; i < local_n[0];i+=32)
    {
      for (int j = 0; j < hopSz[0]; j++)
      {
        const int common = commonEle(i, j, neighborsBase, offsetsBase, degreeHop);
        if (commonMtxBase[i*local_n[0]+j])
        {
          if(common>thresPC1)commonMtxBase[i*local_n[0]+j]=LINK2MORE;
          else if(common==thresPC1)commonMtxBase[i*local_n[0]+j]=LINK2EQUAL;
          else commonMtxBase[i*local_n[0]+j]=LINK2LESS;
        }
        else{
          if(common>thresPC2)commonMtxBase[i*local_n[0]+j]=UNLINK2MORE;
          else if(common==thresPC2)commonMtxBase[i*local_n[0]+j]=UNLINK2EQUAL;
          else commonMtxBase[i*local_n[0]+j]=UNLINK2LESS;
        }
        commonMtxBase[j*local_n[0]+i]=commonMtxBase[i*local_n[0]+j];
      }
      if (k==2) continue;
      for (int j = hopSz[0]; j<i; j++)
      {
        const int common = commonEle(i, j, neighborsBase, offsetsBase, degreeHop);
        if (commonMtxBase[i*local_n[0]+j])
        {
          if(common>thresPP1)commonMtxBase[i*local_n[0]+j]=LINK2MORE;
          else if(common==thresPP1)commonMtxBase[i*local_n[0]+j]=LINK2EQUAL;
          else commonMtxBase[i*local_n[0]+j]=LINK2LESS;
        }
        else{
          if(common>thresPP2)commonMtxBase[i*local_n[0]+j]=UNLINK2MORE;
          else if(common==thresPP2)commonMtxBase[i*local_n[0]+j]=UNLINK2EQUAL;
          else commonMtxBase[i*local_n[0]+j]=UNLINK2LESS;
        }
        commonMtxBase[j*local_n[0]+i]=commonMtxBase[i*local_n[0]+j];
      }
    }

    // if (warp_id == 2230 && lane_id == 0)
    // {
    //   // printf("Hop Size: %d\n", hopSz[0]);
    //   printf("Common Matrix: \n");
    //   for (int a = 0; a < local_n[0]; a++)
    //     {
    //       for (int b = 0; b < local_n[0]; b++)
    //         {
    //           printf("%d ", commonMtxBase[a*local_n[0]+b]);
    //         }
    //       printf("\n\n");
    //     }
    //   //printf("common mtx: %d\n", commonMtxBase[1041]);
    // }
}

__global__ void kSearch(int idx, P_pointers p, S_pointers s, G_pointers g, T_pointers t, unsigned int* d_blk, unsigned int* d_blk_counter, unsigned int* d_left, unsigned int* d_left_counter, unsigned int* d_res, unsigned int* d_br, unsigned int* d_state, unsigned int* d_hopSz, uint16_t* neiInG, uint16_t* neiInP, unsigned int* plex_count, uint8_t* commonMtx, unsigned int* recCand1, unsigned int* recCand2, unsigned int* recExcl, unsigned int* nonAdjInP, unsigned int* d_v2delete, uint32_t* d_adj, unsigned long long* cycles)
{
  unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int warp_id = (global_index / 32);
  unsigned int lane_id = threadIdx.x % 32;

  //if (lane_id == 0) printf("N: %d, warp_id: %d, i: %d\n", N, warp_id, i);
  //if (warp_id + WARPS*idx < 1366 || warp_id + WARPS*idx >= 2000) return;
  //if (warp_id + WARPS*idx != 0) return;
  // if ((warp_id+WARPS*idx) >= g.n)  return;
  if ((warp_id+WARPS*idx) >= (g.n-p.lb+2)) return;
  // //if (lane_id == 0) printf("Hi\n");
  int k = p.k;
  int q = p.lb;

  unsigned int* counterBase = d_blk_counter + warp_id;

  

  if (counterBase[0] < q) return;
  

  unsigned int* degreeBase = s.degree + warp_id * MAX_BLK_SIZE;
  unsigned int* l_degreeBase = s.l_degree + warp_id * MAX_BLK_SIZE;
  unsigned int* offsetsBase = s.offsets + warp_id * MAX_BLK_SIZE;
  unsigned int* neighborsBase = s.neighbors + warp_id * MAX_BLK_SIZE * AVG_DEGREE;
  unsigned int* l_offsetsBase = s.l_offsets + warp_id * MAX_BLK_SIZE;
  unsigned int* l_neighborsBase = s.l_neighbors + warp_id * MAX_BLK_SIZE * AVG_LEFT_DEGREE;
  unsigned int* degreeHop = s.degreeHop + warp_id * MAX_BLK_SIZE;
  bool* proper = s.proper + warp_id;
  uint8_t* labelsBase = s.labels + warp_id * MAX_BLK_SIZE;

  unsigned int* plex = s.P + warp_id * MAX_BLK_SIZE;
  unsigned int* cand1 = s.C + warp_id * MAX_BLK_SIZE;
  unsigned int* cand2 = s.C2 + warp_id * MAX_BLK_SIZE;
  unsigned int* excl = s.X + warp_id * MAX_BLK_SIZE;

  unsigned int* PlexSz = s.PSize + warp_id;
  unsigned int* Cand1Sz = s.CSize + warp_id;
  unsigned int* Cand2Sz = s.C2Size + warp_id;
  unsigned int* ExclSz = s.XSize + warp_id;

  uint32_t* Pset = s.Pset + warp_id;
  uint32_t* Cset = s.Cset + warp_id;
  uint32_t* C2set = s.C2set + warp_id;
  uint32_t* Xset = s.Xset + warp_id;

  unsigned int* blkBase = d_blk + warp_id * MAX_BLK_SIZE;

  unsigned int* leftBase = d_left + warp_id * MAX_BLK_SIZE;
  unsigned int* left_counter = d_left_counter + warp_id;

  unsigned int* hopSz = d_hopSz + warp_id;
  
  unsigned int* res = d_res + warp_id * MAX_DEPTH;
  unsigned int* br = d_br + warp_id * MAX_DEPTH;
  unsigned int* state = d_state + warp_id * MAX_DEPTH;
  unsigned int* v2delete = d_v2delete + warp_id * MAX_DEPTH;

  uint16_t* neiInGBase = neiInG + warp_id * MAX_BLK_SIZE;
  uint16_t* neiInPBase = neiInP + warp_id * MAX_BLK_SIZE;
  unsigned int* nonAdjInPBase = nonAdjInP + warp_id * MAX_BLK_SIZE;

  unsigned int* recCand1Base = recCand1 + warp_id * MAX_BLK_SIZE;
  unsigned int* recCand2Base = recCand2 + warp_id * MAX_BLK_SIZE;
  unsigned int* recExclBase = recExcl + warp_id * MAX_BLK_SIZE;
  size_t capacity = size_t(warp_id) * CAP;
  uint8_t* commonMtxBase = commonMtx + capacity;

  uint32_t *adjList = d_adj + ADJSIZE * warp_id;

  
  

  int sz = 0;

  for (int i = lane_id; i < counterBase[0]; i+=32)
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
  
  res[sz] = k - 1;
  br[sz] = 1;
  state[sz] = 0;
  sz++;
  

  
  while(sz)
  {
    
      //printf("sz: %d, res: %d, state: %d, br: %d, v2delete: %d, vectorSz: %d, PlexSz: %d, Cand1Sz: %d, Cand2Sz: %d, ExclSz: %d\n", sz, res[sz-1], state[sz-1], br[sz-1], v2delete[sz-1], vectorSz, PlexSz[0], Cand1Sz[0], Cand2Sz[0], ExclSz[0]);
      if (sz >= MAX_DEPTH)
      {
        if (lane_id == 0) printf("capacity crossed: %d\n", sz);
        break;
      }
    switch(state[sz-1])
    {
      case 0:
        // if(warp_id == 5 and lane_id == 0) printf("Case 0\n");
        if (Cand2Sz[0] == 0)
        {
          if (PlexSz[0] + Cand1Sz[0] < q)
          {
            sz--;
            continue;
          }
          
          bool cond = !upperBoundK2(lane_id, counterBase[0], k, q, plex, cand1, neiInGBase, nonAdjInPBase, PlexSz[0], Cand1Sz[0], neiInPBase, proper, neighborsBase, offsetsBase, degreeBase);
          
          if (PlexSz[0] > 1 && cond)
          {
            sz--;
            continue;
          }
          // if (warp_id == 13 && lane_id == 0)
          // {
          //   printf("Upperbound with labels: %d, Upperbound: %d\n", !upperBoundK(lane_id, counterBase[0], k, q, labelsBase, neiInGBase, nonAdjInPBase, PlexSz[0], neiInPBase, proper, neighborsBase, offsetsBase, degreeBase), !upperBoundK2(lane_id, counterBase[0], k, q, plex, cand1, neiInGBase, nonAdjInPBase, PlexSz[0], Cand1Sz[0], neiInPBase, proper, neighborsBase, offsetsBase, degreeBase));
          // }
          
          int pos;
          if (lane_id == 0)
          {
            //printf("Emplacing a Task 1\n");
            unsigned int* tail = t.d_tail_A;
            pos = atomicAdd(&tail[0], 1u);
          }
          pos = __shfl_sync(0xFFFFFFFF, pos, 0);
          if (pos+1 > MAX_CAP)
          {
            if(lane_id == 0) printf("Maximum Capacity Reached in kSearch\n");
            return;
          }
          uint8_t* newLabels = t.d_all_labels_A + pos * MAX_BLK_SIZE;
          uint16_t* newNeiInG = t.d_all_neiInG_A + pos * MAX_BLK_SIZE;
          uint16_t* newNeiInP = t.d_all_neiInP_A + pos * MAX_BLK_SIZE;
          for (int i = lane_id; i < counterBase[0]; i+=32)
          {
            newLabels[i] = U;
            // newNeiInG[i] = neiInGBase[i];
            // newNeiInP[i] = neiInPBase[i];
          }
          warp_memcpy_u16_16B(newNeiInG, neiInGBase, counterBase[0]);
          warp_memcpy_u16_16B(newNeiInP, neiInPBase, counterBase[0]);
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
          //   if (warp_id == 13 && lane_id == 0){
          //   printf("Task No. %d\n", pos);
          //   printf("idx: %d, PlexSz: %d, CandSz: %d, ExclSz: %d\n", nt.idx, nt.PlexSz, nt.CandSz, nt.ExclSz);
          //   printf("labels: \n");
          //   for (int i = 0; i < counterBase[0]; i++)
          //   {
          //     printf("%d ", nt.labels[i]);
          //   }
          //   printf("\n");
          //   printf("neiInG: \n");
          //   for (int i = 0; i < counterBase[0]; i++)
          //   {
          //     printf("%d ", nt.neiInG[i]);
          //   }
          //   printf("\n");
          //   printf("neiInP: \n");
          //   for (int i = 0; i < counterBase[0]; i++)
          //   {
          //     printf("%d ", nt.neiInP[i]);
          //   }
          //   printf("\n");
          }
          __syncwarp();
          
          sz--;
          continue;
        }
       
        if (lane_id == 0)
        {
          // for (int i = counterBase[0]-1; i >= 0; i--)
          // {
          //   if (labelsBase[i] == H)
          //   {
          //     labelsBase[i] = X;
          //     ExclSz[0]++;
          //     Cand2Sz[0]--;
          //     v2delete[sz-1] = i;
          //     break;
          //   }
          // }
          unsigned int u = cand2[Cand2Sz[0]-1];
          C2set[u >> 5] &= ~(1u << (u & 31));
          excl[ExclSz[0]++] = cand2[--Cand2Sz[0]];
          v2delete[sz-1] = u;
          Xset[u >> 5] |=  (1u << (u & 31));

          res[sz] = res[sz-1];
          br[sz] = 1;
          state[sz] = 0;

          state[sz-1] = 1;
        }
        sz++;
        __syncwarp();
        
        continue;

      case 1:
        // if(warp_id == 5 and lane_id == 0) printf("Case 1\n");
        
        if(lane_id == 0)
        {
          // labelsBase[v2delete[sz-1]] = H;
          // ExclSz[0]--;
          // Cand2Sz[0]++;
          state[sz-1] = 2;

          unsigned int u = v2delete[sz-1];
          cand2[Cand2Sz[0]++] = u;
          C2set[u >> 5] |=  (1u << (u & 31));
          Xset[u >> 5] &= ~(1u << (u & 31));
          for (int i = 0; i < ExclSz[0]; i++)
          {
            if(excl[i] == v2delete[sz-1])
            {
              int temp = excl[i];
              excl[i] = excl[ExclSz[0]-1];
              excl[ExclSz[0]-1] = temp;
              ExclSz[0]--;
              break;
            }
          }
          //ExclSz[0]--;
        }
        __syncwarp();
        
        continue;
      case 2:
        // if(warp_id == 5 and lane_id == 0) printf("Case 2\n");
        if (br[sz-1] < res[sz-1])
        {
          if (lane_id == 0)
          {
            // for (int i = counterBase[0]-1; i >= 0; i--)
            // {
            //   if (labelsBase[i] == H)
            //   {
            //     labelsBase[i] = P;
            //     PlexSz[0]++;
            //     Cand2Sz[0]--;
            //     v2adds[vectorSz] = i;
            //     break;
            //   } 
            // }

            unsigned int u = cand2[--Cand2Sz[0]];
            Pset[u >> 5] |=  (1u << (u & 31));
            C2set[u >> 5] &= ~(1u << (u & 31));
            plex[PlexSz[0]++] = u;
            //v2adds[vectorSz] = plex[PlexSz[0]-1];
          }
          __syncwarp();
          unsigned int node = plex[PlexSz[0]-1];
          for (int i = lane_id; i < degreeBase[node]; i+=32)
          {
            const int nei = neighborsBase[offsetsBase[node]+i];
            neiInPBase[nei]++;
          }
          
          addG(lane_id, node, neiInGBase, counterBase[0], neighborsBase, offsetsBase, degreeBase);
          
          
           
          updateCand12(lane_id, cand1, commonMtxBase, recCand1Base, neiInGBase, proper, sz, counterBase[0], node, Cand1Sz, neighborsBase, degreeBase, offsetsBase);
          
          // if (lane_id == 0) updateCand22(cand2, commonMtxBase, recCand2Base, proper, sz, counterBase[0], node, Cand2Sz);
          //if (lane_id == 0) updateExcl2(excl, commonMtxBase, recExclBase, proper, sz, counterBase[0], node, ExclSz);
          
          const uint8_t* __restrict__ row = commonMtxBase + (size_t) node * counterBase[0];
          updateCand23(lane_id, cand2, row, recCand2Base, sz, Cand2Sz);
          
          // __syncwarp();
          
          
           
          //vectorSz++;
          if (Cand2Sz[0])
          {
            if (lane_id == 0)
            {
              // for (int idx = counterBase[0]-1; idx >= 0; idx--)
              // {
              //   if (labelsBase[idx] == H)
              //   {
              //     labelsBase[idx] = X;
              //     ExclSz[0]++;
              //     Cand2Sz[0]--;
              //     v2delete[sz-1] = idx;
              //     break;
              //   }
              // }
              unsigned int u = cand2[--Cand2Sz[0]];
              excl[ExclSz[0]++] = u;
              v2delete[sz-1] = u;
              C2set[u >> 5] &= ~(1u << (u & 31));
              Xset[u >> 5] |=  (1u << (u & 31));

              state[sz-1] = 3;
              //sz++;
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
        // if(warp_id == 5 and lane_id == 0) printf("Case 3\n");
        if (lane_id == 0)
        {
          // labelsBase[v2delete[sz-1]] = H;
          // ExclSz[0]--;
          // Cand2Sz[0]++;
          br[sz-1]++;
          state[sz-1] = 2;

          unsigned int u = v2delete[sz-1];
          cand2[Cand2Sz[0]++] = u;
          C2set[u >> 5] |=  (1u << (u & 31));
          Xset[u >> 5] &= ~(1u << (u & 31));
          for (int i = 0; i < ExclSz[0]; i++)
          {
            if(excl[i] == u)
            {
              int temp = excl[i];
              excl[i] = excl[ExclSz[0]-1];
              excl[ExclSz[0]-1] = temp;
              ExclSz[0]--;
              break;
            }
          }
        }
        __syncwarp();
        continue;
      case 4:
      // if(warp_id == 5 and lane_id == 0) printf("Case 4\n");
      if (br[sz-1] == res[sz-1])
      {
        if (lane_id == 0)
        {
          // for (int i = counterBase[0]-1; i>=0; i--)
          // {
          //   if (labelsBase[i] == H)
          //   {
          //     labelsBase[i] = P;
          //     PlexSz[0]++;
          //     Cand2Sz[0]--;
          //     v2adds[vectorSz] = i;
          //     break;
          //   }
          // }
          unsigned int u = cand2[--Cand2Sz[0]];
          Pset[u >> 5] |=  (1u << (u & 31));
          C2set[u >> 5] &= ~(1u << (u & 31));
          plex[PlexSz[0]++] = u;
          //v2adds[vectorSz] = plex[PlexSz[0]-1];
        }
        __syncwarp();
        unsigned int node = plex[PlexSz[0]-1];
        for (int i = lane_id; i < degreeBase[node]; i+=32)
        {
          const int nei = neighborsBase[offsetsBase[node]+i];
          neiInPBase[nei]++;
        }
        addG(lane_id, node, neiInGBase, counterBase[0], neighborsBase, offsetsBase, degreeBase);

        
        updateCand12(lane_id, cand1, commonMtxBase, recCand1Base, neiInGBase, proper, sz, counterBase[0], node, Cand1Sz, neighborsBase, degreeBase, offsetsBase);
        //updateCand1(lane_id, labelsBase, commonMtxBase, recCand1Base, neiInGBase, proper, sz-1, counterBase[0], v2adds[vectorSz], Cand1Sz, neighborsBase, degreeBase, offsetsBase);
        
        // for (int i = 0; i < counterBase[0]; i++)
        // {
        //   if (labelsBase[i] == C)
        //   {
        //     if (!isKplex(lane_id, i, k, PlexSz[0], neiInPBase, labelsBase, counterBase[0], neighborsBase, offsetsBase, degreeBase))
        //     {
        //       if (lane_id == 0)
        //       {
        //         labelsBase[i] = U;
        //         Cand1Sz[0]--;
        //       }
        //       __syncwarp();
        //       subG(lane_id, i, neiInGBase, counterBase[0], neighborsBase, offsetsBase, degreeBase);
        //     }
        //   }
        // }
        
        int len = 0;
        for (int i = 0; i < Cand1Sz[0]; i++)
        {
          const int v = cand1[i];
          if (!isKplex2(lane_id, v, k, PlexSz[0], neiInPBase, plex, counterBase[0], neighborsBase, offsetsBase, degreeBase, adjList))
          {
            if (lane_id == 0)
            {
              int temp = cand1[Cand1Sz[0]-1];
              cand1[Cand1Sz[0]-1] = cand1[i];
              cand1[i] = temp;
              Cand1Sz[0]--;
              len++;
              Cset[v >> 5] &= ~(1u << (v & 31));
            }
            __syncwarp();
            subG(lane_id, v, neiInGBase, counterBase[0], neighborsBase, offsetsBase, degreeBase);
            i--;
          }
        }
        len = __shfl_sync(0xFFFFFFFF, len, 0);
       
        //vectorSz++;
  //       if (warp_id == 13 && lane_id == 0)
  // {
  //   printf("n = %d, Plex Size = %d, Cand Size = %d, Cand2 Sz = %d, Excl Size = %d, hopSz: %d, leftCounter: %d\n", counterBase[0], PlexSz[0], Cand1Sz[0], Cand2Sz[0], ExclSz[0], hopSz[0], left_counter[0]);
  //   printf("Labels Array: ");
  //   for (int i = 0; i < counterBase[0]; i++)
  //   {
  //     printf("%d ", labelsBase[i]);
  //   }
  //   printf("\n");

  //   printf("Cand1: ");
  //   for (int i = 0; i < Cand1Sz[0]; i++)
  //   {
  //     printf("%d ", cand1[i]);
  //   }
  //   printf("\n");

  //   printf("Cand2: ");
  //   for (int i = 0; i < Cand2Sz[0]; i++)
  //   {
  //     printf("%d ", cand2[i]);
  //   }
  //   printf("\n");

  //   printf("Plex: ");
  //   for (int i = 0; i < PlexSz[0]; i++)
  //   {
  //     printf("%d ", plex[i]);
  //   }
  //   printf("\n");

  //   printf("Excl: ");
  //   for (int i = 0; i < ExclSz[0]; i++)
  //   {
  //     printf("%d ", excl[i]);
  //   }
  //   printf("\n");
  // }
  // return;
        if (PlexSz[0] + Cand1Sz[0] < q)
        {
          if (lane_id == 0)
          {
            state[sz-1] = 5;
          }
          __syncwarp();
          // for (int i = 0; i < counterBase[0]; i++)
          // {
          //   if (labelsBase[i] == U)
          //   {
          //       if (lane_id == 0)
          //       {
          //         labelsBase[i] = C;
          //         Cand1Sz[0]++;
          //       }
          //       __syncwarp();
          //       addG(lane_id, i, neiInGBase, counterBase[0], neighborsBase, offsetsBase, degreeBase);
          //   }
          // }
          
          for (int i = 0; i < len; i++)
          {
            if (lane_id == 0) Cand1Sz[0]++;
            __syncwarp();
            const int v = cand1[Cand1Sz[0]-1];
            if (lane_id == 0) Cset[v >> 5] |=  (1u << (v & 31));
            addG(lane_id, v, neiInGBase, counterBase[0], neighborsBase, offsetsBase, degreeBase);
          }
          
          continue;
        }
        if (PlexSz[0] > 1 && !upperBoundK2(lane_id, counterBase[0], k, q, plex, cand1, neiInGBase, nonAdjInPBase, PlexSz[0], Cand1Sz[0], neiInPBase, proper, neighborsBase, offsetsBase, degreeBase)/*!upperBoundK(lane_id, counterBase[0], k, q, labelsBase, neiInGBase, nonAdjInPBase, PlexSz[0], neiInPBase, proper, neighborsBase, offsetsBase, degreeBase)*/)
        {
          if (lane_id == 0)
          {
            state[sz-1] = 5;
          }
          __syncwarp();
          // for (int i = 0; i < counterBase[0]; i++)
          // {
          //   if (labelsBase[i] == U)
          //   {
          //       if (lane_id == 0)
          //       {
          //         labelsBase[i] = C;
          //         Cand1Sz[0]++;
          //       }
          //       __syncwarp();
          //       addG(lane_id, i, neiInGBase, counterBase[0], neighborsBase, offsetsBase, degreeBase);
          //   }
          // }
          
          for (int i = 0; i < len; i++)
          {
            if (lane_id == 0) Cand1Sz[0]++;
            __syncwarp();
            const int v = cand1[Cand1Sz[0]-1];
            if (lane_id == 0) Cset[v >> 5] |=  (1u << (v & 31));
            addG(lane_id, v, neiInGBase, counterBase[0], neighborsBase, offsetsBase, degreeBase);
          }
          continue;
        }
        
        int pos;
        if (lane_id == 0)
        {
          //printf("Emplacing a Task 2\n");
          unsigned int* tail = t.d_tail_A;
          pos = atomicAdd(&tail[0], 1u);
          state[sz-1] = 5;
        }
        pos = __shfl_sync(0xFFFFFFFF, pos, 0);
        if (pos+1 > MAX_CAP)
        {
          if(lane_id == 0) printf("Maximum Capacity Reached in kSearch\n");
          return;
        }
        uint8_t* newLabels = t.d_all_labels_A + pos * MAX_BLK_SIZE;
        uint16_t* newNeiInG = t.d_all_neiInG_A + pos * MAX_BLK_SIZE;
        uint16_t* newNeiInP = t.d_all_neiInP_A + pos * MAX_BLK_SIZE;
        for (int i = lane_id; i < counterBase[0]; i+=32)
        {
          newLabels[i] = U;
          // newNeiInG[i] = neiInGBase[i];
          // newNeiInP[i] = neiInPBase[i];
        }
        warp_memcpy_u16_16B(newNeiInG, neiInGBase, counterBase[0]);
        warp_memcpy_u16_16B(newNeiInP, neiInPBase, counterBase[0]);
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
          // if (warp_id == 13 && lane_id == 0){
          // printf("Task No. %d\n", pos);
          //   printf("idx: %d, PlexSz: %d, CandSz: %d, ExclSz: %d\n", nt.idx, nt.PlexSz, nt.CandSz, nt.ExclSz);
          //   printf("labels: \n");
          //   for (int i = 0; i < counterBase[0]; i++)
          //   {
          //     printf("%d ", nt.labels[i]);
          //   }
          //   printf("\n");
          //   printf("neiInG: \n");
          //   for (int i = 0; i < counterBase[0]; i++)
          //   {
          //     printf("%d ", nt.neiInG[i]);
          //   }
          //   printf("\n");
          //   printf("neiInP: \n");
          //   for (int i = 0; i < counterBase[0]; i++)
          //   {
          //     printf("%d ", nt.neiInP[i]);
          //   }
          //   printf("\n");
          // }
        }
         __syncwarp();
         
        //  for (int i = 0; i < counterBase[0]; i++)
        // {
        //   if (labelsBase[i] == U)
        //   {
        //       if (lane_id == 0)
        //       {
        //         labelsBase[i] = C;
        //         Cand1Sz[0]++;
        //       }
        //       __syncwarp();
        //       addG(lane_id, i, neiInGBase, counterBase[0], neighborsBase, offsetsBase, degreeBase);
        //   }
        // }
        
        for (int i = 0; i < len; i++)
          {
            if (lane_id == 0) Cand1Sz[0]++;
            __syncwarp();
            const int v = cand1[Cand1Sz[0]-1];
            if (lane_id == 0) Cset[v >> 5] |=  (1u << (v & 31));
            addG(lane_id, v, neiInGBase, counterBase[0], neighborsBase, offsetsBase, degreeBase);
          }
          
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
      //if(warp_id == 5 and lane_id == 0) printf("Case 5\n");
      for (int i = br[sz-1]; i >= 1; i--)
      {
        
        unsigned int node = plex[PlexSz[0]-1];
        if (lane_id == 0)
        {
          // labelsBase[v2adds[vectorSz-1]] = H;
          // PlexSz[0]--;
          // Cand2Sz[0]++;
          // cand2[Cand2Sz[0]++] = v2adds[vectorSz-1];
          // PlexSz[0]--;
          cand2[Cand2Sz[0]++] = plex[--PlexSz[0]];
          C2set[node >> 5] |=  (1u << (node & 31));
          Pset[node >> 5] &= ~(1u << (node & 31));
        }
        __syncwarp();
        for (int j = lane_id; j < degreeBase[node]; j+=32)
        {
          const int nei = neighborsBase[offsetsBase[node]+j];
          neiInPBase[nei]--;
        }
        subG(lane_id, node, neiInGBase, counterBase[0], neighborsBase, offsetsBase, degreeBase);
        // recoverCand1(lane_id, labelsBase, commonMtxBase, recCand1Base, neiInGBase, proper, sz-1, counterBase[0], v2adds[vectorSz-1], Cand1Sz, neighborsBase, degreeBase, offsetsBase);
        // if (lane_id == 0) recoverCand2(labelsBase, commonMtxBase, recCand2Base, neiInGBase, proper, sz-1, counterBase[0], v2adds[vectorSz-1], Cand2Sz, neighborsBase, degreeBase, offsetsBase);
        // if (lane_id == 0) recoverExcl(labelsBase, commonMtxBase, recExclBase, neiInGBase, proper, sz-1, counterBase[0], v2adds[vectorSz-1], ExclSz, neighborsBase, degreeBase, offsetsBase);
        __syncwarp();
        //vectorSz--;
      }
      
      recoverCand12(lane_id, cand1, recCand1Base, neiInGBase, sz, counterBase[0], Cand1Sz, neighborsBase, degreeBase, offsetsBase);
      
      //     __syncwarp();
      // unsigned long long t0 = clock64();
      if (lane_id == 0) recoverCand22(counterBase[0], cand2, recCand2Base, sz, Cand2Sz);
      // if (lane_id == 0) recoverExcl2(counterBase[0], excl, recExclBase, sz, ExclSz);
      __syncwarp();
      // __syncwarp();
      //     unsigned long long t1 = clock64();
      //     if (lane_id == 0)
      //     {
      //       cycles[0] += (t1-t0);
      //     }
      sz--;
      continue;
    }
  }
  if (lane_id == 0)
  {
    printf("Warp: %d with blkSz: %d\n", warp_id, counterBase[0]);
  }
}