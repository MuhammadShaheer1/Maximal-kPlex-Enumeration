inline void chkerr(cudaError_t code)
{
    if (code != cudaSuccess)
    {
        std::cout<<cudaGetErrorString(code)<<std::endl;
        exit(-1);
    }
}

template <typename T>
void copy_graph_to_gpu(const graph<T> &peelG, int* dpos, int* dseq, G_pointers &g, D_pointers &d, S_pointers &s)
{
    // Copying original graph to GPU
    chkerr(cudaMalloc(&(g.offsets), (peelG.n + 1) * sizeof(unsigned int)));
    chkerr(cudaMalloc(&(g.neighbors), (peelG.m) * sizeof(unsigned int)));
    chkerr(cudaMalloc(&(g.degree), (peelG.n) * sizeof(unsigned int)));

    chkerr(cudaMemcpy(g.offsets, peelG.offsets, (peelG.n + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(g.neighbors, peelG.neighbors, (peelG.m) * sizeof(unsigned int), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(g.degree, peelG.degree, (peelG.n) * sizeof(unsigned int), cudaMemcpyHostToDevice));

    g.n = peelG.n;
    g.m = peelG.m;

    chkerr(cudaMalloc(&(d.dpos), (peelG.n) * sizeof(int)));
    chkerr(cudaMalloc(&(d.dseq), (peelG.n) * sizeof(int)));

    chkerr(cudaMemcpy(d.dpos, dpos, (peelG.n) * sizeof(int), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(d.dseq, dseq, (peelG.n) * sizeof(int), cudaMemcpyHostToDevice));

    //----------------------------------------------------------------------------------

    //Allocating memory for subgraphs
    chkerr(cudaMalloc(&(s.offsets), (MAX_BLK_SIZE)*WARPS*sizeof(unsigned int)));
    chkerr(cudaMalloc(&(s.l_offsets), (MAX_BLK_SIZE)*WARPS*sizeof(unsigned int)));

    chkerr(cudaMalloc(&(s.degree), (MAX_BLK_SIZE)*WARPS*sizeof(unsigned int)));
    chkerr(cudaMalloc(&(s.l_degree), (MAX_BLK_SIZE)*WARPS*sizeof(unsigned int)));
    chkerr(cudaMalloc(&(s.degreeHop), (MAX_BLK_SIZE)*WARPS*sizeof(unsigned int)));

    chkerr(cudaMalloc(&(s.neighbors), (MAX_BLK_SIZE)*AVG_DEGREE*WARPS*sizeof(unsigned int)));
    chkerr(cudaMalloc(&(s.l_neighbors), (MAX_BLK_SIZE)*AVG_LEFT_DEGREE*WARPS*sizeof(unsigned int)));

    chkerr(cudaMalloc(&(s.P), (MAX_BLK_SIZE)*WARPS*sizeof(unsigned int)));
    chkerr(cudaMalloc(&(s.C), (MAX_BLK_SIZE)*WARPS*sizeof(unsigned int)));
    chkerr(cudaMalloc(&(s.C2), (MAX_BLK_SIZE)*WARPS*sizeof(unsigned int)));
    chkerr(cudaMalloc(&(s.X), (MAX_BLK_SIZE)*WARPS*sizeof(unsigned int)));
    
    chkerr(cudaMalloc(&(s.n), WARPS*sizeof(unsigned int)));
    chkerr(cudaMalloc(&(s.m), WARPS*sizeof(unsigned int)));
    chkerr(cudaMalloc(&(s.PSize), WARPS*sizeof(unsigned int)));
    // --------------------------BNB----------------------
    // chkerr(cudaMalloc(&(s.C1Size), WARPS*sizeof(unsigned int)));
    // chkerr(cudaMalloc(&(s.C2Size), WARPS*sizeof(unsigned int)));
    //-------------------------BNB-------------------------
    chkerr(cudaMalloc(&(s.CSize), WARPS*sizeof(unsigned int)));
    chkerr(cudaMalloc(&(s.C2Size), WARPS*sizeof(unsigned int)));
    chkerr(cudaMalloc(&(s.XSize), WARPS*sizeof(unsigned int)));
}