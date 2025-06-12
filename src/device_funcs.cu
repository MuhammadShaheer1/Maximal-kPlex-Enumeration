#include "../inc/device_funcs.h"
// #include "device_funcs.h"

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
    for (int i = lane_id; i < g.degree[vstart]; i += 32)
    {
      int nei = g.neighbors[g.offsets[vstart] + i];

      //if (!visitedBase[nei]) // position of one-hop 
      //{
        if (d.dpos[vstart] < d.dpos[nei])
        {
          idx = atomicAdd(&counterBase[0], 1);
          blkBase[idx] = nei;
        }
        else
        {
          idx = atomicAdd(&left_counter[0], 1);
          leftBase[idx] = nei;
        }
        visitedBase[nei] = 1;
      //}
    } 

  //   for(int j=16;j>0;j=j/2){
  //     degreeCount += __shfl_down_sync(0xFFFFFFFF,(degreeCount),j);
  // }
  //   degreeCount = __shfl_sync(0xFFFFFFFF,degreeCount,0);
  __syncwarp(0xFFFFFFFF);
  if (lane_id == 0) 
  {
    hopSz[0] = counterBase[0];
  }
  if (counterBase[0] - 1 < p.bd) goto CLEAN; //if no of direct neighbors < q - k, prune it

    // for (int i = lane_id; i < degreeCount; i+=32)
    // {
    //   int cnt = 0;
    //   int u = neibors[i];
    //   unsigned int *lst1 = g.neighbors + g.offsets[u];
    //   int j1 = 0, j2 = 0;
    //   while (j1 < g.degree[u] && j2 < degreeCount)
    //   {
    //     if (lst1[j1] < neibors[j2]) ++j1;
    //     else if (lst1[j1] > neibors[j2]) ++j2;
    //     else cnt++, j1++, j2++;
    //   }

    //   if (cnt >= lb_2k)
    //   {
    //     idx = atomicAdd(&counterBase[0], 1);
    //     blkBase[idx] = u;
    //     visitedBase[u] = 1;
    //   }

    // }

  // if (lane_id == 0)
  // {
  //   printf("No of direct neighbors and left neighbors in warp %d: %d %d\n", warp_id, counterBase[0], left_counter[0]);
  // }

  // two-hop neighbors >= q - 2k + 2

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

    __syncwarp(0xFFFFFFFF); //count = [3 4 5]

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

    __syncwarp(0xFFFFFFFF);

    if (lane_id == 0)
    {
      if (counterBase[0] >= p.lb){
        //printf("Number of direct neighbors + node itself in warp %d is %d\n", warp_id, hopSz[0]);
        atomicAdd(&validblk[0], 1);
        atomicAdd(&global_count[0], counterBase[0]);
        atomicAdd(&left_count[0], left_counter[0]);
      }
    }
    CLEAN:
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

  __syncwarp(0xFFFFFFFF);

    // printf("Inside GPU thread %d k: %d, q: %d, q-k:%d\n", global_index, plex_pointers.k, plex_pointers.lb, plex_pointers.bd);
    // printf("Thread No. %d Number of nodes n = %d with edges m = %d\n", global_index, graph_pointers.n, graph_pointers.m);
    // if (global_index == 2){
    // printf("Offsets: ");
    // for (int i = 0; i < graph_pointers.n+1; i++)
    // {
    //   printf("%d ", graph_pointers.offsets[i]);
    // }
    
    // printf("\nNeighbors: ");
    // for (int i = 0; i < graph_pointers.m; i++)
    // {
    //   printf("%d ", graph_pointers.neighbors[i]);
    // }
    // printf("\nDegree: ");
    // for (int i = 0; i < graph_pointers.n; i++)
    // {
    //   printf("%d ", graph_pointers.degree[i]);
    // }
    

    // printf("\ndseq: ");
    // for (int i =0; i < graph_pointers.n; i++)
    // {
    //     printf("%d ", degen_pointers.dseq[i]);
    // }
    // printf("\ndpos: ");
    // for (int i =0; i < graph_pointers.n; i++)
    // {
    //     printf("%d ", degen_pointers.dpos[i]);
    // }
    // printf("\n");
// }
}

__device__ bool basic_search(unsigned int node, unsigned int *buffer, unsigned int len)
{
  for (int idx = 0; idx < len; idx++)
  {
    if (node == buffer[idx]) return true;
  }
  return false;
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

  if (lane_id == 0)
  {
    //printf("Total degrees: %d in warp %d\n", ne, warp_id);
    atomicAdd(&global_count[0], warp_sum_ne);
    atomicAdd(&left_count[0], warp_sum_neLeft);
  }
}

__global__ void fillNeighbors(int i, S_pointers s, G_pointers g, unsigned int *d_blk, unsigned int *d_blk_counter, unsigned int *d_left, unsigned int *d_left_counter, unsigned int *d_hopSz, unsigned int *neiInG)
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
  unsigned int* neiInGBase = neiInG + warp_id * (MAX_BLK_SIZE);

  unsigned int* leftBase = d_left + warp_id * MAX_BLK_SIZE;
  unsigned int* left_counter = d_left_counter + warp_id;

  unsigned int* hopSz = d_hopSz + warp_id;

  unsigned int* neighborsBase = s.neighbors + warp_id * (MAX_BLK_SIZE) * (AVG_DEGREE);
  unsigned int* l_neighborsBase = s.l_neighbors + warp_id * (MAX_BLK_SIZE) * (AVG_LEFT_DEGREE);

  unsigned int* local_n = s.n + warp_id;
  unsigned int* local_m = s.m + warp_id;
  unsigned int* PlexSz = s.PSize + warp_id;
  //-----------------BNB-----------------
  // unsigned int* Cand1Sz = s.C1Size + warp_id;
  // unsigned int* Cand2Sz = s.C2Size + warp_id;
  //-------------------BNB----------------
  //--------------------BK-----------------
  unsigned int* CandSz = s.CSize + warp_id;
  //------------------BK-----------------
  unsigned int* ExclSz = s.XSize + warp_id;
  uint8_t* labelsBase = s.labels + warp_id * (MAX_BLK_SIZE);

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
  }

  __syncwarp();

  for (int idx = lane_id; idx < counterBase[0]; idx+=32)
  {
    neiInGBase[idx] = degreeHop[idx];
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

  if (warp_id == 9 && lane_id == 0)
  {
    printf("Blks: ");
    for (int i = 0; i < counterBase[0]; i++)
    {
      printf("%d ", blkBase[i]);
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

    // printf("Degree Hop Array: ");
    // for (int i = 0; i < counterBase[0]; i++)
    // {
    //   printf("%d ", degreeHop[i]);
    // }
    // printf("\n");

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
    printf("n = %d, m = %d, Plex Size = %d, Cand Size = %d, Excl Size = %d\n", local_n[0], local_m[0], PlexSz[0], CandSz[0], ExclSz[0]);
    printf("Labels Array: ");
    for (int i = 0; i < local_n[0]; i++)
    {
      printf("%d ", labelsBase[i]);
    }
    printf("\n");
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

__device__ void clearBitSet(uint32_t* bitSet)
{
  memset(bitSet, 0, sizeof(uint32_t) * (MAX_BLK_SIZE >> 5));
}

__device__ void setBit(uint32_t* bitSet, int x)
{
  bitSet[x >> 5] = (unsigned)1 << (x & 31);
}

__device__ int testBit(uint32_t* bitSet, int x)
{
  return bitSet[x >> 5] >> (x & 31) & 1;
}

__device__ void clearBit(uint32_t* bitSet, int x)
{
  bitSet[x >> 5] = (unsigned)0 << (x & 31);
}

__device__ void swapDevice(unsigned int &x, unsigned int &y)
{
  int tmp = x;
  x = y;
  y = tmp;
}

__device__ void bronKerbosch(unsigned int warp_id, unsigned int lane_id, P_pointers p, S_pointers s, unsigned int* d_blk, unsigned int* d_left, unsigned int* plex_count, unsigned int *nonNeigh)
{
  if (warp_id == 0) printf("I'm in BK\n");
  unsigned int* blkBase = d_blk + warp_id * MAX_BLK_SIZE;
  unsigned int* nonNeighBase = nonNeigh + warp_id * MAX_BLK_SIZE;

  unsigned int* neighborsBase = s.neighbors + warp_id * (MAX_BLK_SIZE) * (AVG_DEGREE);
  unsigned int* offsetsBase = s.offsets + warp_id * (MAX_BLK_SIZE);
  unsigned int* degreeBase = s.degree + warp_id * (MAX_BLK_SIZE);

  unsigned int* local_n = s.n + warp_id;
  unsigned int* PlexSz = s.PSize + warp_id;
  unsigned int* CandSz = s.CSize + warp_id;
  unsigned int* ExclSz = s.XSize + warp_id;
  uint8_t* labelsBase = s.labels + warp_id * (MAX_BLK_SIZE);
  int ret = 1;
  // if (lane_id == 0)
  // {
    if (CandSz[0] == 0 && ExclSz[0] == 0)
    {
      atomicAdd(&plex_count[0], 1);
      ret = 0;
    }
  // }
  // ret = __shfl_sync(0xFFFFFFFF, ret, 0);
  
  if (ret == 0)
  {
    return;
  }
  uint32_t bitSet[MAX_BLK_SIZE >> 5];
  clearBitSet(bitSet);

  // if (lane_id == 0)
  // {
    int length_cand = CandSz[0];
    int length_excl = ExclSz[0];
    for (int i = 0; i < length_cand; i++)
    {
      if (nonNeighBase[i] > p.k - 1) continue;

      labelsBase[i] = P;
      for (int j = i+1; j < length_cand; j++)
      {
        //TODO: Binary Search
        if (!basic_search(j, neighborsBase+offsetsBase[i], degreeBase[i]))
        {
          nonNeighBase[j]++;
          setBit(bitSet, j);
        }
        if (nonNeighBase[j] > p.k - 1)
        {
          swapDevice(blkBase[j], blkBase[length_cand--]);
          j--;
        }
      }

    
      for (int j = 0; j < length_excl; j++)
      {
        if (!basic_search(j, neighborsBase+ offsetsBase[i], degreeBase[i]))
        {
          nonNeighBase[j]++;
          setBit(bitSet, j);
        }
        if (nonNeighBase[j] > p.k - 1)
        {
          swapDevice(blkBase[j], blkBase[length_excl--]);
          j--;
        }
      }

      int tempCand = CandSz[0];
      int tempExcl = ExclSz[0];
      CandSz[0] = length_cand;
      ExclSz[0] = length_excl;
      bronKerbosch(warp_id, lane_id, p, s, d_blk, d_left, plex_count, nonNeigh); 
      CandSz[0] = tempCand;
      ExclSz[0] = tempExcl;

      for (int j = 0; j < local_n[0]; j++)
      {
        if (testBit(bitSet, j))
        {
          nonNeighBase[j]--;
          clearBit(bitSet, j);
        }
      }

      labelsBase[i] = X;
      CandSz[0]--;
      ExclSz[0]++;
    }
  // }
  // __syncwarp();
  
}

// __global__ void kSearch(int i, P_pointers p, S_pointers s, G_pointers g, unsigned int *d_blk, unsigned int *d_left, unsigned int *neiInG, unsigned int *neiInP, unsigned int *d_hopSz, unsigned int* plex_count, unsigned int* nonNeigh, unsigned int* depth)
// {
//   unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
//   unsigned int warp_id = (global_index / 32);
//   unsigned int lane_id = threadIdx.x % 32;

//   if ((warp_id+WARPS*i) >= g.n)  return;
//  //----------------------------BNB--------------------------
//  //search(p.k - 1, warp_id, lane_id, s, p, d_blk, d_left, neiInG, neiInP, d_hopSz);
//  //--------------------------BNB----------------------------
//  if (lane_id == 0) bronKerbosch(warp_id, lane_id, p, s, d_blk, d_left, plex_count, nonNeigh, depth);
//  __syncwarp();
// }

__device__ void push(unsigned int lane_id, unsigned int k, int* cur_id, int * level, unsigned int n, unsigned int left_count, unsigned int* neighborsBase, unsigned int* l_neighborsBase, 
                     unsigned int * offsetsBase, unsigned int* l_offsetsBase, unsigned int* degreeBase, unsigned int* l_degreeBase,
                     uint8_t* labelsBase, unsigned int* nonNeighBase, unsigned int* nonNeighLeftBase, unsigned int* depthBase,
                     unsigned int* stackBase, unsigned int* leftSz, unsigned int* PlexSz, unsigned int* CandSz,
                     unsigned int* ExclSz)
{
  // for (int i = 0; i < n; i++)
  // {
  //   if (labelsBase[i] == P && (nonNeighBase[i]+1 > (k-1)) && (!basic_search(i, neighborsBase + offsetsBase[cur_id[0]], degreeBase[cur_id[0]])))
  //   {
  //     //printf("K-plex can't be made\n");
  //     depthBase[cur_id[0]]--;
  //     CandSz[0]--;
  //     cur_id[0] = stackBase[level[0] - 1];
  //     return;
  //   }
  // }
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

  int local_left = 0;
  //if (lane_id == 0){
  for (int i = lane_id; i < left_count; i+=32)
  {
    if (!basic_search(i, l_neighborsBase + l_offsetsBase[cur_id[0]], l_degreeBase[cur_id[0]]))
    {
      nonNeighLeftBase[i]++;

    if (nonNeighLeftBase[i] == k)
    {
      //atomicAdd(&leftSz[0], -1);
      local_left++;
    }
    }
  }
//}

  int sum_left = local_left;
  for (int offset = 16; offset > 0; offset >>= 1)
  {
    sum_left += __shfl_down_sync(0xFFFFFFFF, sum_left, offset);
  }

  leftSz[0] -= sum_left;
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
      if (labelsBase[i] == C) //local_cand++;
        atomicAdd(&CandSz[0], 1);
      if (labelsBase[i] == X) //local_excl++;
        atomicAdd(&ExclSz[0], 1);
    }
  }
  //}
  int sum_cand = local_cand;
  sum_excl = local_excl;

  // for (int offset = 16; offset > 0; offset >>= 1)
  // {
  //   sum_cand += __shfl_down_sync(0xFFFFFFFF, sum_cand, offset);
  //   sum_excl += __shfl_down_sync(0xFFFFFFFF, sum_excl, offset);
  // }
  // if (lane_id == 0)
  // {
  //   CandSz[0] += sum_cand;
  //   ExclSz[0] += sum_excl;
  // }

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
  // int local_cand = 0;
  // local_excl = 0;
  // if (lane_id == 0){
  // for (int i = lane_id; i < n; i++)
  // {
  //   if (i == cur_id[0]) continue;

  //   if (depthBase[i] == level[0] - 1)
  //   {
  //     if (labelsBase[i] == C)
  //       local_cand++;

  //     else if (labelsBase[i] == X)
  //       local_excl++;
  //   }

  //   else if (depthBase[i] == level[0])
  //     depthBase[i]--;
    
  //   if (!basic_search(i, neighborsBase + offsetsBase[cur_id[0]], degreeBase[cur_id[0]]))
  //     nonNeighBase[i]--;
  // }}

  // int sum_cand = local_cand;
  // int sum_excl = local_excl;

  // for (int offset = 16; offset > 0; offset >>= 1)
  // {
  //   sum_cand += __shfl_down_sync(0xFFFFFFFF, sum_cand, offset);
  //   sum_excl += __shfl_down_sync(0xFFFFFFFF, sum_excl, offset);
  // }

  // if (lane_id == 0)
  // {
  //   CandSz[0] += sum_cand;
  //   ExclSz[0] += sum_excl;
  // }

  // __syncwarp();

  int local_left = 0;
  //if (lane_id == 0){
  for (int i = lane_id; i < left_count; i+=32)
  {
    if (!basic_search(i, l_neighborsBase + l_offsetsBase[cur_id[0]], l_degreeBase[cur_id[0]]))
    {
      nonNeighLeftBase[i]--;
      if (nonNeighLeftBase[i] == (k-1))
        local_left++;
    }
  }
//}

  int sum_left = local_left;
  for (int offset = 16; offset > 0; offset >>= 1)
  {
    sum_left += __shfl_down_sync(0xFFFFFFFF, sum_left, offset);
  }

  leftSz[0] += sum_left;

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

  if (warp_id < 10)
  {
    //printf("Left Count: %d\n", left_counter[0]);
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
        if ((CandSz[0] == 0) && (ExclSz[0] == 0) && (leftSz == 0))
        {
          if (lane_id == 0)
          {
          // printf("Maximal-kPlex Found\n");
          // printf("k-Plex: ");
          // for (int i = 0; i < PlexSz[0]; i++)
          // {
          //   printf("%d ", stackBase[i]);
          // }
          // printf("\n");
          atomicAdd(&plex_count[0], 1);
          }
          pop(lane_id, k, &cur_id, &level, local_n[0], left_counter[0], neighborsBase, l_neighborsBase, offsetsBase, l_offsetsBase, degreeBase, l_degreeBase, labelsBase,
         nonNeighBase, nonNeighLeftBase, depthBase, stackBase, &leftSz, PlexSz, CandSz, ExclSz);
        }
        else if (CandSz[0] > 0)
        {
          //if (lane_id == 0) printf("Pushing another node to Plex\n");
          cur_id = findID(cur_id, level, local_n[0], labelsBase, depthBase);
          // if (lane_id == 0)
          // {
          // if (cur_id == -1)
          // {
          //   printf("I'm going to break\n");
          //   break;
          // }
          // }
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
    printf("%d warps are finished with Maximal %d-Plexes: %d.\n", global_count[0], k, plex_count[0]);
    }
  }
}