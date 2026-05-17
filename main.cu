#include <atomic>
#include <cmath>
#include <queue>
#include <cassert>
#include <functional>
#include <chrono>
#include "inc/host_funcs.h"
//#include "kPlexEnum.h"
//#include "device_funcs.h"

void usage(){
    fprintf(stderr,"usage : ./PlexEnum <dataset> -k <k> -q <q> -t <thres> -tr <truss> -ls <local_bnb_steps>\n");
}

int check_inc(int i, int max){
    if(i==max){
        usage();
        exit(1);
    }
    return i+1;
}

int main(int argc, char*argv[])
{
    int i = 2;
    if(argc<6){
        usage();
        exit(1);
    } 
    printf("file: %s\n",argv[1]); 
    while(i<argc){
       if(!strcmp(argv[i],"-k")){
            i = check_inc(i, argc);
            k = atoi(argv[i]);
        }
       else if(!strcmp(argv[i],"-q")){
            i = check_inc(i, argc);
            lb = atoi(argv[i]);
        }
        else if(!strcmp(argv[i],"-t")){
            i = check_inc(i, argc);
            thres = stod(argv[i]);
        }
        else if(!strcmp(argv[i],"-tr")){
            i = check_inc(i, argc);
            truss = atoi(argv[i]);
        }
        else if (!strcmp(argv[i], "-ls"))
        {
            i = check_inc(i, argc);
            local_bnb_steps = atoi(argv[i]);
            if (local_bnb_steps < 1) local_bnb_steps = 1;
        }
        else {
            usage();
            exit(1);
        }
        i++;
    }
    printf("k=%d,q=%d,thres=%f, truss: %d, local_bnb_steps=%d\n",k,lb, thres, truss, local_bnb_steps);
    bd = lb-k; //lb = q
    graph<intT> g(argv[1]);
    //g.printGraph();
    // int n=0; cudaGetDeviceCount(&n);
    // printf("Visible CUDA devices: %d\n", n);
    decomposableSearch(g);
    g.del();
    return 0;
}