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

__global__ void decompose(int i, P_pointers p, G_pointers g, D_pointers d, unsigned int *d_blk, unsigned int *d_blk_counter, unsigned int *d_left, unsigned int *d_left_counter, unsigned int *visited, unsigned int *count, unsigned int *global_count, unsigned int *left_count, unsigned int *validblk, unsigned int* d_hopSz)
{
    unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int warp_id = (global_index / 32);
    unsigned int lane_id = threadIdx.x % 32;

    //__shared__ unsigned int sh_buf[32 * 112];

    if ((warp_id+WARPS*i) >= g.n) return;
    //printf("Currently in thread %d of warp %d\n", lane_id, warp_id);

    //unsigned int *neibors = sh_buf + (threadIdx.x / 32) * 112;
    //int degreeCount = 0;
    int vstart = d.dseq[warp_id+WARPS*i];
    int idx;
    unsigned int* blkBase = d_blk + warp_id * MAX_BLK_SIZE; // d_blk = [2048*1280]
    unsigned int* counterBase = d_blk_counter + warp_id;

    unsigned int* leftBase = d_left + warp_id * MAX_BLK_SIZE;
    unsigned int* left_counter = d_left_counter + warp_id;
    unsigned int* hopSz = d_hopSz + warp_id;

    
    unsigned int *visitedBase = visited + warp_id * g.n;
    unsigned int *countBase = count + warp_id * g.n;
    int lb_2k = p.lb - 2 * p.k; // q - 2*k
    if (lane_id == 0)
    {
      idx = atomicAdd(&counterBase[0], 1);
      blkBase[idx] = vstart; // seed vertex 
      visitedBase[vstart] = 1;
      //printf("Block Size is %d in warp %d\n", counterBase[0], warp_id);
    }
    // if (lane_id == 0)
    // {
    //   printf("degree in %d: %d\n", warp_id, g.offsets[vstart]);
    // }
     __syncwarp();
    int start = 0;
    while(true)
    {
      bool localOverFlow = false;
      bool warpOverFlow = false;
      int overFlowIdx = INT_MAX;

    for (int i = lane_id+start; i < g.degree[vstart]; i+=32)
    {
      if(warpOverFlow) break;
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
            // if (lane_id == 0) printf("Above the capacity\n");
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
            // if (lane_id == 0) printf("Above the capacity\n");
          }
        }

        warpOverFlow = __any_sync(__activemask(), localOverFlow);
    }
    __syncwarp(0xFFFFFFFF);

    for(int offset = 16; offset > 0; offset >>= 1)
  {
    int otherMin = __shfl_down_sync(0xFFFFFFFF, overFlowIdx, offset);

    if (otherMin < overFlowIdx)
    {
      overFlowIdx = otherMin;
    }
  }
  

  overFlowIdx = __shfl_sync(0xFFFFFFFF, overFlowIdx, 0);
  start = overFlowIdx;

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
          if (basic_search(v, g.neighbors+g.offsets[u], g.degree[u])) cnt++;
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
        if (basic_search(v, g.neighbors+g.offsets[u], g.degree[u])) cnt++;
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
  if (!warpOverFlow) break;
}


  // try to run this loop, set mark to true, prune all those neighbors
    for (int i = lane_id; i < g.degree[vstart]; i+=32)
    {
      int nei = g.neighbors[g.offsets[vstart] + i];
      if (d.dpos[vstart] < d.dpos[nei]){
        for (int j = 0; j < g.degree[nei]; j++)
        {
            int twoHop = g.neighbors[g.offsets[nei]+j];
            if (!visitedBase[twoHop]) // blk -> direct neighbors, log O(degree(vstart)) time, normalizer into 1 - 900 range, denormalize it 
             {
              atomicAdd(&countBase[twoHop], 1); 
            }
        }
    }
    }

    __syncwarp(); //count = [3 4 5]

    for (int i = lane_id; i < g.degree[vstart]; i+=32)
    {
      int nei = g.neighbors[g.offsets[vstart] + i];
      if (d.dpos[vstart] < d.dpos[nei])
      {
        for (int j = 0; j < g.degree[nei]; j++)
        {
          int twoHop = g.neighbors[g.offsets[nei]+j];
          int old = atomicCAS(&visitedBase[twoHop], 0, 1);
          if (old == 0)
          {
            if (d.dpos[vstart] < d.dpos[twoHop])
            {
              if (countBase[twoHop] >= lb_2k + 2)
              {
                idx = atomicAdd(&counterBase[0], 1);
                blkBase[idx] = twoHop;
              }
            }
            else
            {
              if (countBase[twoHop] >= lb_2k + 3)
              {
                idx = atomicAdd(&left_counter[0], 1);
                leftBase[idx] = twoHop;
              }
            }
            
        }
        }
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
      countBase[i] = 0;
    }


    // if (lane_id == 0)
    // {
    //   //printf("Total number of nodes in blk in warp %d is %d\n", warp_id, counterBase[0]);
    //   counterBase[0] = 0;
    //   left_counter[0] = 0;
    // }

  __syncwarp();
}

__global__ void calculateDegrees(int i , P_pointers p, G_pointers g, S_pointers s, unsigned int *d_blk, unsigned int *d_blk_counter, unsigned int *d_left, unsigned int *d_left_counter, unsigned int *global_count, unsigned int *left_count)
{
  unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int warp_id = (global_index / 32);
  unsigned int lane_id = threadIdx.x % 32;
  if ((warp_id+WARPS*i) >= g.n)  return;

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

__global__ void fillNeighbors(int i, S_pointers s, G_pointers g, unsigned int *d_blk, unsigned int *d_blk_counter, unsigned int *d_left, unsigned int *d_left_counter, unsigned int *d_hopSz, uint8_t* commonMtx)
{
  unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int warp_id = (global_index / 32);
  unsigned int lane_id = threadIdx.x % 32;

  if ((warp_id+WARPS*i) >= g.n)  return;

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
  //-----------------BNB-----------------
  // unsigned int* Cand1Sz = s.C1Size + warp_id;
  // unsigned int* Cand2Sz = s.C2Size + warp_id;
  //-------------------BNB----------------
  //--------------------BK-----------------
  unsigned int* CandSz = s.CSize + warp_id;
  //------------------BK-----------------
  unsigned int* ExclSz = s.XSize + warp_id;
  uint8_t* labelsBase = s.labels + warp_id * (MAX_BLK_SIZE);
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
      if (basic_search(nei, g.neighbors+g.offsets[origin], g.degree[origin]))
      {
        commonMtxBase[idx*counterBase[0]+j] = 1;
        neighborsBase[offset+cnt] = j;
        cnt++;
      }
      if (j == hopSz[0]-1) degreeHop[idx] = cnt;
    }

    cnt = 0;
    for (int j = 0; j < left_counter[0]; j++)
    {
      unsigned int nei = leftBase[j];
      if (basic_search(nei, g.neighbors+g.offsets[origin], g.degree[origin]))
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
  // for (int i = lane_id; i < hopSz[0]; i+=32)
  // {
  //   labelsBase[i] = C1;
  // }

  // for (int i = lane_id+hopSz[0]; i < local_n[0]; i+=32)
  // {
  //   labelsBase[i] = C2;
  // }
  //----------------------BNB-----------------
  //--------------------BK-------------------
  for (int i = lane_id; i < local_n[0]; i+=32)
  {
    labelsBase[i] = C;
  }
  //--------------------BK---------------------

  //----------------BNB---------------------
  //labelsBase[0] = P;
  local_m[0] = offsetsBase[local_n[0]];
  //PlexSz[0] = 1;
  // Cand1Sz[0] = hopSz[0] - 1;
  // Cand2Sz[0] = local_n[0] - hopSz[0];
  //---------------BNB--------------------
  //--------------BK-------------------
  CandSz[0] = local_n[0];
  ExclSz[0] = 0;
  PlexSz[0] = 0;

  if (warp_id == 439 && lane_id == 0)
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
    // printf("n = %d, m = %d, Plex Size = %d, Cand Size = %d, Excl Size = %d, hopSz: %d, leftCounter: %d\n", local_n[0], local_m[0], PlexSz[0], CandSz[0], ExclSz[0], hopSz[0], left_counter[0]);
    // printf("Labels Array: ");
    // for (int i = 0; i < local_n[0]; i++)
    // {
    //   printf("%d ", labelsBase[i]);
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
      if (!basic_search(u, l_neighborsBase + l_offsetsBase[v], l_degreeBase[v]))
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
      bool bad = (nonNeigh[v] >= (k-1)) && (!basic_search(u, l_neighborsBase + l_offsetsBase[v], l_degreeBase[v]));

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
      if (neiInP[i] + k == PlexSz && !basic_search(i, neighborsBase + offsetsBase[v], degreeBase[v]))
        localPass = false;
    }
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
      if (missing[i] + k == (PlexSz) && !basic_search(i, neighborsBase + offsetsBase[v], degreeBase[v]))
        return false;
    }
  }
  return true;
}

__device__ void subG(int lane_id, int j, unsigned int* neiInG, unsigned int n, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeBase)
{
  for (int i = lane_id; i < degreeBase[j]; i+=32)
  {
    // if (i == j) continue;

    // if (basic_search(i, neighborsBase + offsetsBase[j], degreeBase[j]))
    //   neiInG[i]--;
    int nei = neighborsBase[offsetsBase[j]+i];
    neiInG[nei]--;
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
        if (!basic_search(u, l_neighborsBase + l_offsetsBase[j], l_degreeBase[j]))
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
        bool bad = (nonNeigh[j] + k < PlexSz + 1) && (!basic_search(u, l_neighborsBase + l_offsetsBase[j], l_degreeBase[j]));

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
        if (!basic_search(u, l_neighborsBase + l_offsetsBase[j], l_degreeBase[j]))
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
        bool bad = (nonNeigh[j] + k < PlexSz + 1) && (!basic_search(u, l_neighborsBase + l_offsetsBase[j], l_degreeBase[j]));

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



__device__ bool upperBound(int lane_id, int n, int k, int lb, uint8_t* labels, unsigned int* neiInG)
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
    }
  }
  bool any_fail = __any_sync(0xFFFFFFFF, !ok);
  return !any_fail;
}

__device__ void branchInCand(int lane_id, int minIndex, int idx, int k, int lb, unsigned int PlexSz, unsigned int CandSz, unsigned int ExclSz, unsigned int* local_n, Task* tasks, Task* global_tasks, unsigned int* tailPtr, unsigned int* global_tail, uint8_t* labelsBase, unsigned int* neiInG, unsigned int* neiInP, unsigned int* neighborsBase, unsigned int* offsetsBase, unsigned int* degreeBase,uint8_t* d_all_labels, unsigned int* d_all_neiInG, unsigned int* d_all_neiInP, uint8_t* global_labels, unsigned int* global_neiInG, unsigned int* global_neiInP, uint8_t* commonMtx)
{
    uint8_t* labels = d_all_labels;
    unsigned int* all_neiInG = d_all_neiInG;
    unsigned int* all_neiInP = d_all_neiInP;
    Task* localTasks = tasks;

    int recCand1 = 0;
    bool ub;
    ub = upperBound(lane_id, local_n[0], k, lb, labelsBase, neiInG);
    // ub = __shfl_sync(0xFFFFFFFF, ub, 0);
    if (ub)
    {
    unsigned int pos;
    if (lane_id == 0) pos = atomicAdd(&tailPtr[0], 1u);
    pos = __shfl_sync(0xFFFFFFFF, pos, 0);
    if (pos + 1 > SMALL_CAP) 
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
    unsigned int* childNeiInG = all_neiInG + pos*MAX_BLK_SIZE;
    unsigned int* childNeiInP = all_neiInP + pos*MAX_BLK_SIZE;


    for (int j = lane_id; j < local_n[0]; j+=32)
    {
      childLabels[j] = labelsBase[j];
      childNeiInG[j] = neiInG[j];
      childNeiInP[j] = neiInP[j];
    }
    //printf("Pushing a task 1\n");
    Task &nt = localTasks[pos];
    nt.idx = idx;

    childLabels[minIndex] = P;
    unsigned int childPlexSz = PlexSz + 1;
    unsigned int childCandSz = CandSz - 1;
    unsigned int childExclSz = ExclSz;

    //updateCand1Fake(recCand1, commonMtx, labelsBase, proper, );


    for (int j = lane_id; j < degreeBase[minIndex]; j+=32)
    {
      const int nei = neighborsBase[offsetsBase[minIndex]+j];
      childNeiInP[nei]++;
    }

    // int local_cand = 0;
    // int local_excl = 0;
    for (int j = 0; j < local_n[0]; j++)
    {
      if (childLabels[j] == C)
      {
        if (!isKplex(lane_id, j, k, childPlexSz, childNeiInP, childLabels, local_n[0], neighborsBase, offsetsBase, degreeBase))
        {
          childCandSz--;
          //local_cand++;
          childLabels[j] = U;
          subG(lane_id, j, childNeiInG, local_n[0], neighborsBase, offsetsBase, degreeBase);
        }
      }
      else if (childLabels[j] == X)
      {
        if (!isKplex(lane_id, j, k, childPlexSz, childNeiInP, childLabels, local_n[0], neighborsBase, offsetsBase, degreeBase))
        {
          childExclSz--;
          //local_excl++;
          childLabels[j] = V;
        }
      }
    }
    __syncwarp();

  if (lane_id == 0)
  {
    nt.PlexSz = childPlexSz;
    nt.CandSz = childCandSz;
    nt.ExclSz = childExclSz;
    nt.labels = childLabels;
    nt.neiInG = childNeiInG;
    nt.neiInP = childNeiInP;
  }
}

    //printf("Pushing a task 2\n");
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
    unsigned int* childNeiInG2 = all_neiInG + pos2*MAX_BLK_SIZE;
    unsigned int* childNeiInP2 = all_neiInP + pos2*MAX_BLK_SIZE;

    for (int j = lane_id; j < local_n[0]; j+=32)
    {
      childLabels2[j] = labelsBase[j];
      childNeiInG2[j] = neiInG[j];
      childNeiInP2[j] = neiInP[j];
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

__global__ void BNB(int i, P_pointers p, S_pointers s, unsigned int* d_blk, unsigned int* d_left, unsigned int* d_blk_counter, unsigned int* d_left_counter, uint8_t* commonMtx, Task* tasks, Task* outTasks, Task* global_tasks, unsigned int N, unsigned int head, unsigned int* tailPtr, unsigned int* global_tail, uint8_t* d_all_labels, unsigned int* d_all_neiInG, unsigned int* d_all_neiInP, uint8_t* global_labels, unsigned int* global_neiInG, unsigned int* global_neiInP, unsigned int* plex_count)
{
  unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int warp_id = (global_index / 32);
  unsigned int lane_id = threadIdx.x % 32;

  //if (lane_id == 0) printf("N: %d, warp_id: %d, i: %d\n", N, warp_id, i);
  if ((warp_id+WARPS*i) >= N)  return;
  //if (lane_id == 0) printf("Hi\n");
  int k = p.k;
  int q = p.lb;
  Task t = tasks[warp_id+WARPS*i];
  uint8_t* labelsBase = t.labels;
  unsigned int* blkBase = d_blk + t.idx * MAX_BLK_SIZE;
  unsigned int* leftBase = d_left + t.idx * MAX_BLK_SIZE;
  unsigned int* neiInG = t.neiInG;
  unsigned int* neiInP = t.neiInP;
  unsigned int* left_count = d_left_counter + t.idx;
  unsigned int* local_n = d_blk_counter + t.idx;
  unsigned int PlexSz = t.PlexSz;
  unsigned int CandSz = t.CandSz;
  unsigned int ExclSz = t.ExclSz;

  uint8_t* commonMtxBase = commonMtx + t.idx*CAP;

  unsigned int* degreeBase = s.degree + t.idx * MAX_BLK_SIZE;
  unsigned int* l_degreeBase = s.l_degree + t.idx * MAX_BLK_SIZE;
  unsigned int* offsetsBase = s.offsets + t.idx * MAX_BLK_SIZE;
  unsigned int* neighborsBase = s.neighbors + t.idx * MAX_BLK_SIZE * AVG_DEGREE;
  unsigned int* l_offsetsBase = s.l_offsets + t.idx * MAX_BLK_SIZE;
  unsigned int* l_neighborsBase = s.l_neighbors + t.idx * MAX_BLK_SIZE * AVG_LEFT_DEGREE;

  if (PlexSz + CandSz < q) return;

  
    if (CandSz == 0)
    {
      if (ExclSz == 0 && PlexSz >= q && isMaximal2(lane_id, k, PlexSz, leftBase, left_count[0], l_neighborsBase, l_offsetsBase, l_degreeBase, neiInP, neighborsBase, offsetsBase, degreeBase, labelsBase, local_n[0]))
      {
        if(lane_id == 0) atomicAdd(&plex_count[0], 1);
      }
      return;
    }
    __syncwarp();

    int minnei_Plex = INT_MAX;
    int pivot = -1;
    int minnei_Cand = INT_MAX;

    for (int i = lane_id; i < local_n[0]; i+=32)
    {
        if (labelsBase[i] == P)
        {
            if (neiInG[i] < minnei_Plex)
            {
                minnei_Plex = neiInG[i];
                pivot = i;
            }
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
        // printf("Going to select pivot again from N(C)'\n");
        minnei_Cand = INT_MAX;
        pivot = -1;
        for (int i = lane_id; i < local_n[0]; i+=32)
        {
            if (labelsBase[i] == C)
            {
                if (!basic_search(i, neighborsBase + offsetsBase[pivot_plex], degreeBase[pivot_plex]))
                {
                    if (neiInG[i] < minnei_Cand)
                    {
                        minnei_Cand = neiInG[i];
                        pivot = i;
                    }
                    else if (neiInG[i] == minnei_Cand && neiInP[pivot] > neiInP[i])
                    {
                        pivot = i;
                    }
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
        // printf("Creating new branches\n");
        branchInCand(lane_id, pivot, t.idx, k, q, PlexSz, CandSz, ExclSz, local_n, outTasks, global_tasks, tailPtr, global_tail, labelsBase, neiInG, neiInP, neighborsBase, offsetsBase, degreeBase, d_all_labels, d_all_neiInG, d_all_neiInP, global_labels, global_neiInG, global_neiInP, commonMtxBase);
        return;
    }

    
    int minnei = minnei_Plex;
    for (int i = lane_id; i < local_n[0]; i+=32)
    {
        if (labelsBase[i] == C)
        {
            if (neiInG[i] < minnei)
            {
                minnei = neiInG[i];
                pivot = i;
            }
            else if (neiInG[i] == minnei && neiInP[pivot] > neiInP[i])
            {
                pivot = i;
            }
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
        // printf("Plex Found 2\n");
        if (PlexSz + CandSz < q) return;
        bool flag = false;
        for (int i = lane_id; i < local_n[0]; i+=32)
        {
            if (labelsBase[i] == X)
            {
                if (isKplexPC(i, k, PlexSz+CandSz, neiInG, labelsBase, local_n[0], neighborsBase, offsetsBase, degreeBase))
                {
                    flag = true;
                }
            }
        }
        if (__any_sync(0xFFFFFFFF, flag)) return;
        if (isMaximalPC(lane_id, k, PlexSz+CandSz, leftBase, left_count[0], l_neighborsBase, l_offsetsBase, l_degreeBase, neiInG, neighborsBase, offsetsBase, degreeBase, labelsBase, local_n[0]))
        {
            // printf("Maximal k-Plex Found 2\n");
            if (lane_id == 0) atomicAdd(&plex_count[0], 1);
        }
        return;
    }

    branchInCand(lane_id, pivot, t.idx, k, q, PlexSz, CandSz, ExclSz, local_n, outTasks, global_tasks, tailPtr, global_tail, labelsBase, neiInG, neiInP, neighborsBase, offsetsBase, degreeBase, d_all_labels, d_all_neiInG, d_all_neiInP, global_labels, global_neiInG, global_neiInP, commonMtxBase);
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
  if ((warp_id+WARPS*idx) >= g.n)  return;


  int k = p.k;
  int lb = p.lb;
  bool *proper = s.proper + warp_id;
  if (!proper[0]) return;
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

    if (warp_id == 0 && lane_id == 0)
    {
      // printf("Hop Size: %d\n", hopSz[0]);
      // printf("Common Matrix: \n");
      // for (int a = 0; a < local_n[0]; a++)
      //   {
      //     for (int b = 0; b < local_n[0]; b++)
      //       {
      //         printf("%d ", commonMtxBase[a*local_n[0]+b]);
      //       }
      //     printf("\n\n");
      //   }
    }
}