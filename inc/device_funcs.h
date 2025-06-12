#include "common.h"
#include <cstdio>

__global__ void decompose(int i, P_pointers p, G_pointers g, D_pointers d, unsigned int *d_blk, unsigned int *d_blk_counter, unsigned int *d_left, unsigned int *d_left_counter, unsigned int *visited, unsigned int *count, unsigned int *global_count, unsigned int *left_count, unsigned int *validblk, unsigned int *d_hopSz);


__global__ void calculateDegrees(int i , P_pointers p, G_pointers g, S_pointers s, unsigned int *d_blk, unsigned int *d_blk_counter, unsigned int *d_left, unsigned int *d_left_counter, unsigned int *global_count, unsigned int *left_count);

__device__ bool basic_search(unsigned int node, unsigned int *buffer, unsigned int len);

__global__ void fillNeighbors(int i, S_pointers s, G_pointers g, unsigned int *d_blk, unsigned int *d_blk_counter, unsigned int *d_left, unsigned int *d_left_counter, unsigned int *d_hopSz, unsigned int *neiInG);

// __global__ void kSearch(int i, P_pointers p, S_pointers s, G_pointers g, unsigned int *d_blk, unsigned int *d_left, unsigned int *neiInG, unsigned int *neiInP, unsigned int *d_hopSz, unsigned int *plex_count, unsigned int* nonNeigh, unsigned int* depth);

__global__ void BKIterative(int i, P_pointers p, S_pointers s, G_pointers g, unsigned int *d_blk, unsigned int *d_left, unsigned int *d_left_counter, unsigned int* plex_count, unsigned int* nonNeigh, unsigned int* nonNeighLeft, unsigned int* depth, unsigned int* stack, unsigned int *global_count);

// __device__ void search(int res, unsigned int warp_id, unsigned int lane_id, S_pointers s, P_pointers p, unsigned int *d_blk, unsigned int *d_left, unsigned int *neiInG, unsigned int *neiInP, unsigned int *d_hopSz);

// __device__ bool upperBoundK(P_pointers p, unsigned int lane_id, unsigned int PlexSz, uint8_t* labelsBase, unsigned int* blkBase, unsigned int* neiInG);

// __device__ void listBranch(P_pointers p, unsigned int lane_id, unsigned int n, unsigned int PlexSz, unsigned int Cand1Sz, unsigned int ExclSz, unsigned int* labelsBase, unsigned int *blkBase, unsigned int* neiInG, unsigned int* neiInP);

// __device__ int cand2BackToExcl(unsigned int lane_id, uint8_t* labelsBase, unsigned int n, unsigned int* Cand2Sz, unsigned int* ExclSz);

// __device__ void exclToCand2(unsigned int lane_id, int v, uint8_t* labelsBase, unsigned int n, unsigned int* Cand2Sz, unsigned int* ExclSz);

// __device__ int cand2BackToPlex(unsigned int lane_id, uint8_t* labelsBase, unsigned int n, unsigned int* Cand2Sz, unsigned int* PlexSz);

// __device__ void plexToCand2(unsigned int lane_id, int v, uint8_t* labelsBase, unsigned int n, unsigned int* Cand2Sz, unsigned int* PlexSz);

//__device__ void bronKerbosch(unsigned int warp_id, unsigned int lane_id, P_pointers p, S_pointers s, unsigned int* d_blk, unsigned int* d_left, unsigned int* plex_count, unsigned int* nonNeigh);