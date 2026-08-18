#include <cstdint>
#include <numeric>

#define BLK_NUMS 142
#define BLK_DIM 1024
#define WARPS_EACH_BLK (BLK_DIM/32)
#define WARPS (BLK_NUMS*WARPS_EACH_BLK)
#define MAX_BLK_SIZE 1024 // Maximum size of my neighborhood
#define MAX_LEFT_SIZE 1024
#define AVG_DEGREE 400
#define MAX_CAP 2048 * 2048
#define SMALL_CAP 512 * 512 * 2

#define MAX_DEPTH 1000
#define CAP MAX_BLK_SIZE * MAX_BLK_SIZE
#define ADJSIZE ((MAX_BLK_SIZE * MAX_BLK_SIZE) / 32)
#define MASK_WORDS ((MAX_BLK_SIZE + 31) / 32)
#define LEFT_MASK_WORDS ((MAX_LEFT_SIZE + 31) / 32)
#define LEFT_ADJ_SIZE (MAX_LEFT_SIZE * MASK_WORDS)
#define LOCAL_LEFT_ADJ_SIZE (MAX_BLK_SIZE * LEFT_MASK_WORDS)
#define MAXIMAL_MASK_WORDS (2 * MASK_WORDS + LEFT_MASK_WORDS)

using namespace std;

enum : uint8_t{
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
    unsigned int *neighbors;
    unsigned int *degree;
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

struct Task{
    int idx;
    unsigned int PlexSz;
    unsigned int CandSz;
    unsigned int ExclSz;
    unsigned int edgePotential;
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
         uint16_t* neiInP_,
         unsigned int edgePotential_ = 0)
         : idx(idx_)
         , PlexSz(PlexSz_)
         , CandSz(CandSz_)
         , ExclSz(ExclSz_)
         , edgePotential(edgePotential_)
         , labels(labels_)
         , neiInG(neiInG_)
         , neiInP(neiInP_)
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
