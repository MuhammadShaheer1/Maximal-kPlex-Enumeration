#include "common.h"
#include <cstdio>

__global__ void decompose(int i, P_pointers p, G_pointers g, D_pointers d, unsigned int *d_blk, unsigned int *d_blk_counter, unsigned int *d_left, unsigned int *d_left_counter, uint32_t *visited, unsigned int *global_count, unsigned int *left_count, unsigned int *validblk, unsigned int *d_hopSz, unsigned long long* cycles);

__global__ void calculateDegrees(int i , P_pointers p, G_pointers g, S_pointers s, unsigned int *d_blk, unsigned int *d_blk_counter, unsigned int *d_left, unsigned int *d_left_counter, unsigned int *global_count, unsigned int *left_count);

__device__ bool basic_search(unsigned int node, unsigned int *buffer, unsigned int len);

__global__ void fillNeighbors(int i, S_pointers s, P_pointers p, G_pointers g, unsigned int *d_blk, unsigned int *d_blk_counter, unsigned int *d_left, unsigned int *d_left_counter, unsigned int *d_hopSz, uint8_t* commonMtx, uint32_t* d_adj);

__global__ void seedInitialTinyTasks(Task* parent_tasks, TinyTask* tiny_tasks, unsigned int parent_start, unsigned int N);

//original BNB Call
// __global__ void BNB(int i, P_pointers p, S_pointers s, unsigned int* d_blk, unsigned int* d_left, unsigned int* d_blk_counter, unsigned int* d_left_counter, uint8_t* commonMtx, Task* tasks, Task* outTasks, Task* global_tasks, unsigned int N, unsigned int head, unsigned int* tailPtr, unsigned int* global_tail, uint8_t* d_all_labels, uint16_t* d_all_neiInG, uint16_t* d_all_neiInP, uint8_t* global_labels, uint16_t* global_neiInG, uint16_t* global_neiInP, unsigned int* plex_count, uint16_t* d_sat, uint16_t* d_commons, uint32_t* d_uni, unsigned long long* cycles, uint32_t* d_adj, int* d_abort, unsigned int *global_count);

//BNB with TinyTask call
__global__ void BNB(int i, P_pointers p, S_pointers s, unsigned int* d_blk, unsigned int* d_left, unsigned int* d_blk_counter, unsigned int* d_left_counter, uint8_t* commonMtx, Task* parent_tasks, TinyTask* tasks, TinyTask* outTasks, TinyTask* global_tasks, unsigned int N, unsigned int* tailPtr, unsigned int* global_tail, BranchLog* Delta, unsigned int* delta_tail, unsigned int* replay_stack, Task* checkpoint_tasks, uint8_t* checkpoint_labels, uint16_t* checkpoint_neiInG, uint16_t* checkpoint_neiInP, unsigned int* checkpoint_tail, unsigned int* plex_count, uint16_t* d_bnb_neiInG, uint16_t* d_bnb_neiInP, uint16_t* d_sat, uint16_t* d_commons, uint32_t* d_uni, unsigned long long* cycles, uint32_t* d_adj, int* abort, unsigned int* global_count);

__global__ void buildCommonMtx(int i, P_pointers p, S_pointers s, G_pointers g, uint8_t* commonMtx, unsigned int* d_hopSz);

__global__ void kSearch(int idx, P_pointers p, S_pointers s, G_pointers g, T_pointers t, unsigned int* d_blk_counter, unsigned int* d_res, unsigned int* d_br, unsigned int* d_state, unsigned int* d_len, unsigned int* d_sz, uint16_t* neiInG, uint16_t* neiInP, unsigned int* plex_count, uint8_t* commonMtx, unsigned int* recCand1, unsigned int* recCand2, unsigned int* d_v2delete, uint32_t* d_adj, unsigned long long* cycles, int* abort_flag, int* d_abort, unsigned int *global_count);

__global__ void kSearch2(int idx, P_pointers p, S_pointers s, G_pointers g, T_pointers t, unsigned int* d_blk_counter, unsigned int* d_res, unsigned int* d_br, unsigned int* d_state, unsigned int* d_len, unsigned int* d_sz, uint16_t* neiInG, uint16_t* neiInP, unsigned int* plex_count, uint8_t* commonMtx, unsigned int* recCand1, unsigned int* recCand2, unsigned int* d_v2delete, uint32_t* d_adj, unsigned long long* cycles, int* abort_flag, int* d_abort);


__global__ void kSearch3(int idx, P_pointers p, S_pointers s, G_pointers g, T_pointers t, unsigned int* d_left,  unsigned int* d_blk_counter, unsigned int* d_left_counter, unsigned int* d_res, unsigned int* d_br, unsigned int* d_state, unsigned int* d_len, unsigned int* d_sz, uint16_t* neiInG, uint16_t* neiInP, unsigned int* plex_count, uint8_t* commonMtx, unsigned int* recCand1, unsigned int* recCand2, unsigned int* recExcl, unsigned int* recCand, unsigned int* d_v2delete, uint32_t* d_adj, uint16_t* d_sat, uint16_t* d_commons, uint32_t* d_uni, unsigned int *global_count);

__global__ void BNB2(int i, P_pointers p, S_pointers s, unsigned int* d_blk, unsigned int* d_left, unsigned int* d_blk_counter, unsigned int* d_left_counter, uint8_t* commonMtx, Task* tasks, Task* outTasks, Task* global_tasks, unsigned int N, unsigned int head, unsigned int* tailPtr, unsigned int* global_tail, uint8_t* d_all_labels, uint16_t* d_all_neiInG, uint16_t* d_all_neiInP, uint8_t* global_labels, uint16_t* global_neiInG, uint16_t* global_neiInP, unsigned int* plex_count, uint16_t* d_sat, uint16_t* d_commons, uint32_t* d_uni, unsigned long long* cycles, uint32_t* d_adj, int* d_abort, unsigned int* d_state, unsigned int* d_res, unsigned int* recExcl, unsigned int* recCand);

__global__ void BNB3(int i, P_pointers p, S_pointers s, unsigned int* d_blk, unsigned int* d_left, unsigned int* d_blk_counter, unsigned int* d_left_counter, uint8_t* commonMtx, Task* tasks, Task* outTasks, Task* global_tasks, unsigned int N, unsigned int head, unsigned int* tailPtr, unsigned int* global_tail, uint8_t* d_all_labels, uint16_t* d_all_neiInG, uint16_t* d_all_neiInP, uint8_t* global_labels, uint16_t* global_neiInG, uint16_t* global_neiInP, unsigned int* plex_count, uint16_t* d_sat, uint16_t* d_commons, uint32_t* d_uni, unsigned long long* cycles, uint32_t* d_adj, int* d_abort, unsigned int* d_state, unsigned int* d_res, unsigned int* recExcl, unsigned int* recCand);

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
//                         unsigned int* debug_spilled);

// __global__
// void resumeTinyTasks_debug(S_pointers s,
//                            Task* base_tasks,
//                            TinyTask* tiny_tasks,
//                            unsigned int num_tiny_tasks,
//                            Delta* delta_log,
//                            unsigned int* debug_checked,
//                            unsigned int* debug_errors);

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
//                                     unsigned int* debug_overflow);
