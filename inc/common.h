#include <cstdint>
#include <numeric>

#define BLK_NUMS 142 
#define BLK_DIM 1024
#define WARPS_EACH_BLK (BLK_DIM/32)
#define WARPS (BLK_NUMS*WARPS_EACH_BLK)
#define MAX_BLK_SIZE 1024
#define AVG_DEGREE 200
#define AVG_LEFT_DEGREE 200
#define MAX_CAP 2048 * 2048 
#define SMALL_CAP 512 * 512 * 2
#define K_LIMIT 10
#define MAX_DEPTH 1000
#define CAP MAX_BLK_SIZE * MAX_BLK_SIZE
#define ADJSIZE ((MAX_BLK_SIZE * MAX_BLK_SIZE) / 32)
#define STAGING_CHUNK 4096
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
    uint16_t* neiInG;
    uint16_t* neiInP;

    Task() {}

    Task(int idx_,
         unsigned int PlexSz_,
         unsigned int CandSz_,
         unsigned int ExclSz_,
         uint8_t* labels_,
         uint16_t* neiInG_,
         uint16_t* neiInP_)
         : idx(idx_)
         , PlexSz(PlexSz_)
         , CandSz(CandSz_)
         , ExclSz(ExclSz_)
         , labels(labels_)
         , neiInG(neiInG_)
         , neiInP(neiInP_)
    {}
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
    uint16_t* d_all_neiInG_A;
    uint16_t* d_all_neiInP_A;
    unsigned int* d_tail_A;

    Task* d_tasks_B;
    uint8_t* d_all_labels_B;
    uint16_t* d_all_neiInG_B;
    uint16_t* d_all_neiInP_B;
    unsigned int* d_tail_B;

    Task* d_tasks_C;
    uint8_t* d_all_labels_C;
    uint16_t* d_all_neiInG_C;
    uint16_t* d_all_neiInP_C;
    unsigned int* d_tail_C;
} T_pointers;
