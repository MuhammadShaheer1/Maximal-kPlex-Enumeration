#ifndef _BENCH_GRAPH_IO
#define _BENCH_GRAPH_IO

#include <iostream>
#include <stdint.h>
#include <cstring>
#include "config.h"
#include "graph.h"
using namespace std;


namespace kPlexEnum{
  void writeBinaryGraph(const char *filepath, const graph<intT> &g){
    using ui = unsigned int;
    std::vector<ui> nodes;
    FILE *f = utils::open_file(filepath, "wb");
    ui tt = sizeof(ui);
    //fwrite(&tt, sizeof(ui), 1, f); //length of ui
    fwrite(&(g.n), sizeof(ui), 1, f);
    fwrite(&(g.m), sizeof(ui), 1, f);
    ui *degree = new ui[g.n];
    for (ui i = 0; i < g.n; i++)
      degree[i] = g.V[i].degree;
    fwrite(degree, sizeof(ui), g.n, f);
    fwrite(g.allocatedInplace + 2 + g.n, sizeof(ui), g.m, f);
    fclose(f);
  }

  //read graph from binary file
  void readBinaryGraph(const char *filepath)
  {
      Timer watch1;
      watch1.start();
      FILE *f = utils::open_file(filepath, "rb");
      if (f == nullptr) {
        printf("Failed to open file: %s\n", filepath);
        return;
    }
      //uintT tt;
      //utils::fread_wall(&tt, sizeof(uintT), 1, f);
      //if (tt != sizeof(uintT))
      //{
      //    printf("sizeof unsigned int is different: file %u, machine %lu\n", tt, sizeof(uintT));
      //}
      uintT n, m;
      printf("Going to n and m\n");
      utils::fread_wall(&n, sizeof(uintT), 1, f); //number of vertices
      printf("Read n\n");
      utils::fread_wall(&m, sizeof(uintT), 1, f); //number of edges
      printf("Read m\n");
      printf("n=%u, m=%u (undirected)\n", n, m);
    //   uintT *in = newA(uintT, n + m + 2);
    //   in[0] = n;
    //   in[1] = m;
      uintT *offsets = newA(uintT, n);
      uintT *neighbors = newA(uintT, m);

      //degree
      printf("Reading degree\n");
      uintT *degree = newA(uintT, n);
      utils::fread_wall(degree, sizeof(uintT), n, f);
      //if (reverse != nullptr) delete[] reverse;
      //reverse = new ui[m];
      printf("Reading neighbors\n");
      uintT sum = 0;
      for (uintT i = 0; i < n; i++)
      {
          offsets[i] = (i == 0) ? 0 : offsets[i - 1] + degree[i - 1];
          utils::fread_wall(neighbors + offsets[i], sizeof(uintT), degree[i], f);
      }

      //vertex<intT> *v = newA(vertex<intT>, n);

      // parallel_for(uintT i = 0; i < n; i++)
      // {
      //     uintT d = degree[i];
      //     uintT o = offset[i];
      //     v[i].degree = d;
      //     v[i].Neighbors = (intT *)edges + o;
      // }
      printf("Closing file\n");
      fclose(f);
      double readtime = watch1.stop();
      printf("Time of read graph:%.3f\n",readtime);
      printf("Number of nodes n = %d with edges m = %d\n", n, m);
      printf("Offsets: ");
      for (int i = 0; i < n; i++)
      {
        printf("%d ", offsets[i]);
      }
      printf("\nNeighbors: ");
      for (int i = 0; i < m; i++)
      {
        printf("%d ", neighbors[i]);
      }
      free(degree);
      //return graph<intT>(v, (intT)n, (uintT)m, (intT *)in);
  }

  void checkGraph(graph<int> &g){
        int ecnt = 0;
        for (int i = 0; i < g.n; i++){
            printf("Nei [%d]:", i);
            for (int j = 0; j < g.V[i].degree; j++){
                int nei = g.V[i].Neighbors[j];
                ecnt++;
                if ( j>0 ) assert(nei > g.V[i].Neighbors[j-1]);
                printf("%d ", nei);
                auto r= find(g.V[nei].Neighbors, g.V[nei].Neighbors + g.V[nei].degree, i);
                assert(r != g.V[nei].Neighbors + g.V[nei].degree);
            } 
            printf("\n");
        }
        assert(ecnt == g.m);
    }
}

#endif // _BENCH_GRAPH_IO
