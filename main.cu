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
    fprintf(stderr,"usage : ./PlexEnum <dataset> -k <k> -q <q>\n");
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
        else {
            usage();
            exit(1);
        }
        i++;
    }
    printf("k=%d,q=%d\n",k,lb);
    bd = lb-k; //lb = q
    graph<intT> g(argv[1]);
    //g.printGraph();
    // int n=0; cudaGetDeviceCount(&n);
    // printf("Visible CUDA devices: %d\n", n);
    decomposableSearch(g);
    g.del();
    return 0;
}