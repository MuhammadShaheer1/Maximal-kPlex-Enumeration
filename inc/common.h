#include <cstdint>

#define BLK_NUMS 40
#define BLK_DIM 1024
#define WARPS_EACH_BLK (BLK_DIM/32)
#define WARPS (BLK_NUMS*WARPS_EACH_BLK)
#define MAX_BLK_SIZE 1024
#define AVG_DEGREE 50
#define AVG_LEFT_DEGREE 20

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
    X = 2
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
    uint8_t *labels;
    unsigned int* PSize;
    //-------------BNB--------
    // unsigned int* C1Size;
    // unsigned int* C2Size;
    //-------------BNB-------
    unsigned int* CSize;
    unsigned int* XSize;
} S_pointers;




