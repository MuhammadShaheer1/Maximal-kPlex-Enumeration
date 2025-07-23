#ifndef _GRAPH_INCLUDED
#define _GRAPH_INCLUDED

#include <iostream>
#include <algorithm>
#include "utils.h"
#include "config.h"

// template <class intT>
// struct vertex
// {
//   intT *Neighbors;
//   intT *NeighborsLeft;
//   intT degree,degreeHop,degreeLeft;
//   void del() {
//     free(Neighbors);
//     if(NeighborsLeft)free(NeighborsLeft);
//   }
// };

template <class intT>
struct graph
{
  intT n, m;
  uintT *offsets;
  uintT *neighbors;
  uintT *degree;
  uintT *neighborsLeft;
  uintT *degreeLeft;
  uintT *offsetsLeft;
  uintT *degreeHop;
  bool proper;
  intT hopSz;
  uint8_t* commonMtx;
  
  graph (){}
  graph(const char *filepath)
  {
    FILE *f = utils::open_file(filepath, "rb");
    if (f == nullptr) {
      printf("Failed to open file: %s\n", filepath);
      return;
  }

  utils::fread_wall(&n, sizeof(uintT), 1, f); //number of vertices
  utils::fread_wall(&m, sizeof(uintT), 1, f); //number of edges
  printf("n=%u, m=%u (undirected)\n", n, m);

  offsets = newA(uintT, n+1);
  neighbors = newA(uintT, m);

  //degree
  degree = newA(uintT, n);
  utils::fread_wall(degree, sizeof(uintT), n, f);

  // uintT sum = 0;
  for (uintT i = 0; i < n; i++)
  {
      offsets[i] = (i == 0) ? 0 : offsets[i - 1] + degree[i - 1];
      utils::fread_wall(neighbors + offsets[i], sizeof(uintT), degree[i], f);
  }
  offsets[n] = offsets[n-1] + degree[n-1];

  fclose(f);
  }

  void printGraph()
  {
    printf("Number of nodes n = %d with edges m = %d\n", n, m);
    printf("Offsets: ");
    for (int i = 0; i < n+1; i++)
    {
      printf("%d ", offsets[i]);
    }
    
    printf("\nNeighbors: ");
    for (int i = 0; i < m; i++)
    {
      printf("%d ", neighbors[i]);
    }
  }

  void del()
  {
    free(offsets);
    free(neighbors);
    free(degree);
    // free(neighborsLeft);
    // free(degreeLeft);
    // free(degreeHop);
  }

  bool isAdj(int v1, int v2)
  {
    if (degree[v1] < degree[v2])
    {
      return std::binary_search(neighbors + offsets[v1], neighbors + offsets[v1+1], v2);
    }
    else
    {
      return std::binary_search(neighbors + offsets[v2], neighbors + offsets[v2+1], v1);
    }
  }
  
};

// template <class intT>
// struct graph
// {
//   vertex<intT> *V;
//   bool proper;
//   const intT n;
//   intT m;   
//   intT *allocatedInplace;

//   graph():n(0),m(0),V(nullptr),allocatedInplace(nullptr){};
//   graph(vertex<intT> *VV, intT nn, uintT mm)
//       : V(VV), n(nn), m(mm), allocatedInplace(NULL) {}
//   graph(vertex<intT> *VV, intT nn, uintT mm, intT *ai)
//       : V(VV), n(nn), m(mm), allocatedInplace(ai) {}
//   graph(vertex<intT> *VV, intT nn, uintT mm, intT *ai, double _proper)
//       : V(VV), n(nn), m(mm), allocatedInplace(ai), proper(_proper){}
//   graph copy()
//   {
//     vertex<intT> *VN = newA(vertex<intT>, n);
//     intT *_allocatedInplace = newA(intT, n + m + 2);
//     _allocatedInplace[0] = n;
//     _allocatedInplace[1] = m;
//     intT *Edges = _allocatedInplace + n + 2;
//     intT k = 0;
//     for (intT i = 0; i < n; i++) //for each vertex
//     {
//       _allocatedInplace[i + 2] = allocatedInplace[i + 2]; //copy 
//       VN[i] = V[i];
//       VN[i].Neighbors = Edges + k;
//       for (intT j = 0; j < V[i].degree; j++)
//         Edges[k++] = V[i].Neighbors[j];
//     }
//     return graph(VN, n, m, _allocatedInplace);
//   }
//   void del()
//   {
//     if (allocatedInplace == nullptr)
//       for (intT i = 0; i < n; i++)
//         V[i].del();
//     else
//       free(allocatedInplace);
//     free(V);
//   }
//   bool isAdj(intT v1, intT v2) const
//   {
//       if(V[v1].degree>V[v2].degree)
//           std::swap(v1, v2);//找度数更小的去查是否是邻居
//       return std::binary_search(V[v1].Neighbors, V[v1].Neighbors + V[v1].degree, v2);
//   }
//   bool isAdj11(intT v1, intT v2) const
//   {
//       if(V[v1].degreeHop>V[v2].degreeHop)
//           std::swap(v1, v2);
//       return std::binary_search(V[v1].Neighbors, V[v1].Neighbors + V[v1].degreeHop, v2);
//   }
// };

#endif // _GRAPH_INCLUDED
