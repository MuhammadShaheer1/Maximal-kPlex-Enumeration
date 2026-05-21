#include <cstdint>
#include <numeric>

#define BLK_NUMS 142 
#define BLK_DIM 1024
#define WARPS_EACH_BLK (BLK_DIM/32)
#define WARPS (BLK_NUMS*WARPS_EACH_BLK)
#define MAX_BLK_SIZE 1024 // Maximum size of my neighborhood
#define PCX_MASK_WORDS ((MAX_BLK_SIZE + 31) / 32) 
#define AVG_DEGREE 200
#define AVG_LEFT_DEGREE 200
#define MAX_CAP 2048 * 2048 
#define SMALL_CAP 512 * 512 * 2
// #define TINY_MAX_CAP   MAX_CAP
// #define TINY_SMALL_CAP SMALL_CAP
// #define DELTA_PER_TINY 32

// #define DELTA_MAX_CAP   (TINY_MAX_CAP * DELTA_PER_TINY)
// #define DELTA_SMALL_CAP (TINY_SMALL_CAP * DELTA_PER_TINY)

// #define LOCAL_BNB_DEPTH 3
// #define LOCAL_UNDO_CAP 512
// #define CONTINUE_CHUNK 65536u

// #define LBL_P 0
// #define LBL_C 1
// #define LBL_X 2

#define K_LIMIT 10
#define MAX_DEPTH 1000
#define CAP MAX_BLK_SIZE * MAX_BLK_SIZE
// #define ADJSIZE ((MAX_BLK_SIZE * MAX_BLK_SIZE) / 32)
#define ADJSIZE (MAX_BLK_SIZE * PCX_MASK_WORDS)
#define STAGING_CHUNK 1024*1024

#define TINY_FRONTIER_CAP (MAX_CAP * 4)
#define TINY_OVERFLOW_CAP (MAX_CAP * 16)
#define DELTA_CAP (MAX_CAP * 16)
#define REPLAY_STACK_CAP MAX_BLK_SIZE
#define LOCAL_TINY_BNB_STEPS 2
#define CHECKPOINT_LOG_LIMIT 4
#define CHECKPOINT_TASK_CAP 2 * SMALL_CAP
// #define LOCAL_TASK_BNB_STEPS 2
#define INVALID_LOG 0xFFFFFFFFu

#define LOCAL_DFS_STACK_FRAMES 96
#define LOCAL_DFS_FRAME_BYTES (6 * MAX_BLK_SIZE)
#define LOCAL_DFS_NEIG_OFFSET 0
#define LOCAL_DFS_NEIP_OFFSET (2 * MAX_BLK_SIZE)
#define LOCAL_DFS_VERT_OFFSET (4 * MAX_BLK_SIZE)

using namespace std;

enum : uint8_t{
    //-------------BNB------------
    // P = 0,
    // C1 = 1,
    // C2 = 2,
    // X = 3
    //------------BK------------
    P = 0,
    C = 1,
    X = 2,
    U = 3,
    V = 4,
    H = 5,
    J = 6,
    K = 7
};

enum : uint8_t{        
    UNLINK2LESS=0,
    LINK2LESS=1,
    UNLINK2EQUAL=2,
    LINK2EQUAL=3,
    UNLINK2MORE=4,
    LINK2MORE=5
};

typedef struct P_pointers{
    int k;
    int lb;
    int bd; //q-k
    int local_bnb_steps;
    float thres;
} P_pointers;

typedef struct G_pointers{
    unsigned int n, m;
    unsigned int *offsets;
    unsigned int *neighbors;
    unsigned int *degree;
} G_pointers;

typedef struct D_pointers{
    int *dpos;
    int *dseq;
} D_pointers;

typedef struct S_pointers{
    unsigned int* n;
    unsigned int* m;
    unsigned int *offsets;
    unsigned int *l_offsets;
    unsigned int *neighbors;
    unsigned int *l_neighbors;
    unsigned int *degree;
    unsigned int *l_degree;
    unsigned int *degreeHop;
    unsigned int* P;
    unsigned int* C;
    unsigned int* C2;
    unsigned int* X;
    unsigned int* PSize;
    unsigned int* CSize;
    unsigned int* C2Size;
    unsigned int* XSize;
    unsigned int* PB;
    unsigned int* CB;
    unsigned int* XB;
} S_pointers;

typedef struct H_pointers{
    unsigned int* h_degree;
    unsigned int* h_degree_hop;
    unsigned int* h_offsets;
    unsigned int* h_neighbors;
    unsigned int* h_blk_counter;
    bool *h_proper;
    unsigned int* h_hopSz;
    uint8_t* h_commonMtx;
} H_pointers;

struct State {
    std::vector<int> P;
    std::vector<int> C;
    std::vector<int> X;
    std::vector<int> missing;
    std::vector<int> left;

    // Constructor to copy in all five arrays
    State(std::vector<int> P_,
          std::vector<int> C_,
          std::vector<int> X_,
          std::vector<int> missing_,
          std::vector<int> left_)
        : P(std::move(P_))
        , C(std::move(C_))
        , X(std::move(X_))
        , missing(std::move(missing_))
        , left(std::move(left_))
    {}
};

struct State2 {
    std::vector<int> P;
    std::vector<int> C;
    std::vector<int> X;
    std::vector<int> missing;
    std::vector<int> left;
    unsigned int * blk;

    // Constructor to copy in all five arrays
    State2(std::vector<int> P_,
          std::vector<int> C_,
          std::vector<int> X_,
          std::vector<int> missing_,
          std::vector<int> left_, 
          unsigned int* blk_)
        : P(std::move(P_))
        , C(std::move(C_))
        , X(std::move(X_))
        , missing(std::move(missing_))
        , left(std::move(left_))
        , blk(blk_)
    {}
};

struct Task{
    int idx;
    unsigned int PlexSz;
    unsigned int CandSz;
    unsigned int ExclSz;
    uint8_t* labels; // labels = [P, C ,X ,C, C]
    uint32_t* Pmask;
    uint32_t* Cmask;
    uint32_t* Xmask;
    uint16_t* neiInG;
    uint16_t* neiInP;

    Task() {}

    Task(int idx_,
         unsigned int PlexSz_,
         unsigned int CandSz_,
         unsigned int ExclSz_,
         uint8_t* labels_,
         uint16_t* neiInG_,
         uint16_t* neiInP_,
         uint32_t* Pmask_ = nullptr,
         uint32_t* Cmask_ = nullptr,
         uint32_t* Xmask_ = nullptr)
         : idx(idx_)
         , PlexSz(PlexSz_)
         , CandSz(CandSz_)
         , ExclSz(ExclSz_)
         , labels(labels_)
         , Pmask(Pmask_)
         , Cmask(Cmask_)
         , Xmask(Xmask_)
         , neiInG(neiInG_)
         , neiInP(neiInP_)
    {}
};

struct LocalDfsFrame {
    unsigned int PlexSz;
    unsigned int CandSz;
    unsigned int ExclSz;
    unsigned int depth;
};

// struct TinyTask {
//     int idx;                 // same meaning as Task::idx: root/subgraph id

//     uint16_t plex_sz;        // compact version of PlexSz
//     uint16_t cand_sz;        // compact version of CandSz
//     uint16_t excl_sz;        // compact version of ExclSz

//     uint32_t log_off;        // starting offset in global Delta array
//     uint16_t log_len;        // number of Delta records for this continuation

//     uint16_t depth;          // resume depth inside local BNB search

//     uint32_t parent_task_pos;

//     uint64_t state_hash;

//     __host__ __device__ TinyTask() {}

//     __host__ __device__ TinyTask(int idx_,
//              uint16_t plex_sz_,
//              uint16_t cand_sz_,
//              uint16_t excl_sz_,
//              uint32_t log_off_,
//              uint16_t log_len_,
//              uint16_t depth_,
//              uint32_t parent_task_pos_,
//              uint64_t state_hash_)
//     : idx(idx_)
//     , plex_sz(plex_sz_)
//     , cand_sz(cand_sz_)
//     , excl_sz(excl_sz_)
//     , log_off(log_off_)
//     , log_len(log_len_)
//     , depth(depth_)
//     , parent_task_pos(parent_task_pos_)
//     , state_hash(state_hash_)
//     {}
// };

// struct Delta {
//     uint16_t v;              // local vertex id inside the subgraph/root task

//     uint8_t old_label;       // previous label: P/C/X/...
//     uint8_t new_label;       // new label after this branch decision

//     int16_t neiInP_delta;    // change applied to neiInP[v]
//     int16_t neiInG_delta;    // change applied to neiInG[v]

//     __host__ __device__ Delta() {}

//     __host__ __device__ Delta(uint16_t v_,
//           uint8_t old_label_,
//           uint8_t new_label_,
//           int16_t neiInP_delta_,
//           int16_t neiInG_delta_)
//     : v(v_)
//     , old_label(old_label_)
//     , new_label(new_label_)
//     , neiInP_delta(neiInP_delta_)
//     , neiInG_delta(neiInG_delta_)
//     {}
// };

// struct UndoRec {
//     uint16_t v;
//     uint8_t old_label;
//     uint16_t old_neiInP;
//     uint16_t old_neiInG;

//     __host__ __device__ UndoRec() {}

//     __host__ __device__ UndoRec(uint16_t v_,
//             uint8_t old_label_,
//             uint16_t old_neiInP_,
//             uint16_t old_neiInG_)
//     : v(v_)
//     , old_label(old_label_)
//     , old_neiInP(old_neiInP_)
//     , old_neiInG(old_neiInG_)
//     {}
// };

// struct LocalSizeMark {
//     unsigned int undo_top;
//     unsigned int plex_sz;
//     unsigned int cand_sz;
//     unsigned int excl_sz;

//     __host__ __device__
//     LocalSizeMark() {}

//     __host__ __device__
//     LocalSizeMark(unsigned int undo_top_,
//                   unsigned int plex_sz_,
//                   unsigned int cand_sz_,
//                   unsigned int excl_sz_)
//     : undo_top(undo_top_)
//     , plex_sz(plex_sz_)
//     , cand_sz(cand_sz_)
//     , excl_sz(excl_sz_)
//     {}
// };

struct BranchLog{
    unsigned int pivot;
    unsigned int branch_decision;
    unsigned int prev;
};

struct TinyTask {
    unsigned int parent_task_pos;
    unsigned int log_start;
    unsigned int log_len;

    unsigned int plex_sz;
    unsigned int cand_sz;
    unsigned int excl_sz;
};


struct HostTask{
    int idx;
    unsigned int PlexSz;
    unsigned int CandSz;
    unsigned int ExclSz;

    uint8_t labels[MAX_BLK_SIZE];
    uint16_t neiInG[MAX_BLK_SIZE];
    uint16_t neiInP[MAX_BLK_SIZE];
};

struct HostTaskBuffer{
    HostTask *tasks;
    unsigned int capacity;
    unsigned int size;
};

struct Frame{
    int res;
    int br;
    int state;
    int v2delete;
    vector<int> v2adds;

    Frame() {}

    Frame(int res_,
          int br_,
          int state_)
          : res(res_)
          , br(br_)
          , state(state_)
          {}
};

typedef struct T_pointers{
    Task* d_tasks_A;
    uint8_t* d_all_labels_A;
    uint32_t* d_all_Pmask_A;
    uint32_t* d_all_Cmask_A;
    uint32_t* d_all_Xmask_A;
    uint16_t* d_all_neiInG_A;
    uint16_t* d_all_neiInP_A;
    unsigned int* d_tail_A;

    TinyTask* d_tiny_tasks_A;
    unsigned int* d_tiny_tail_A;

    Task* d_tasks_B;
    uint8_t* d_all_labels_B;
    uint32_t* d_all_Pmask_B;
    uint32_t* d_all_Cmask_B;
    uint32_t* d_all_Xmask_B;
    uint16_t* d_all_neiInG_B;
    uint16_t* d_all_neiInP_B;
    unsigned int* d_tail_B;

    TinyTask* d_tiny_tasks_B;
    unsigned int* d_tiny_tail_B;

    Task* d_tasks_C;
    uint8_t* d_all_labels_C;
    uint32_t* d_all_Pmask_C;
    uint32_t* d_all_Cmask_C;
    uint32_t* d_all_Xmask_C;
    uint16_t* d_all_neiInG_C;
    uint16_t* d_all_neiInP_C;
    unsigned int* d_tail_C;

    TinyTask* d_tiny_tasks_C;
    unsigned int* d_tiny_tail_C;

    BranchLog* Delta;
    unsigned int* d_delta_tail;
    unsigned int* d_replay_stack;

    // TinyTask* d_tiny_tasks_A;
    // TinyTask* d_tiny_tasks_B;
    // TinyTask* d_tiny_tasks_C;

    // unsigned int* d_tiny_tail_A;
    // unsigned int* d_tiny_tail_B;
    // unsigned int* d_tiny_tail_C;

    // Delta* d_delta_log_A;
    // Delta* d_delta_log_B;
    // Delta* d_delta_log_C;

    // unsigned int* d_delta_tail_A;
    // unsigned int* d_delta_tail_B;
    // unsigned int* d_delta_tail_C;
} T_pointers;
