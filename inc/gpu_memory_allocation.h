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

    chkerr(cudaMalloc(&(s.PB), (MAX_BLK_SIZE)*WARPS*sizeof(unsigned int)));
    chkerr(cudaMalloc(&(s.CB), (MAX_BLK_SIZE)*WARPS*sizeof(unsigned int)));
    chkerr(cudaMalloc(&(s.XB), (MAX_BLK_SIZE)*WARPS*sizeof(unsigned int)));
    
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

inline void allocate_tiny_task_queues(T_pointers &t) {
    // -------------------------------------------------------------------------
    // TinyTask queues
    // A and B are small ping-pong queues.
    // C is the larger overflow/global continuation queue.
    // -------------------------------------------------------------------------
    chkerr(cudaMalloc(&(t.d_tiny_tasks_A),
                      TINY_SMALL_CAP * sizeof(TinyTask)));
    chkerr(cudaMalloc(&(t.d_tiny_tasks_B),
                      TINY_SMALL_CAP * sizeof(TinyTask)));
    chkerr(cudaMalloc(&(t.d_tiny_tasks_C),
                      TINY_MAX_CAP * sizeof(TinyTask)));

    chkerr(cudaMalloc(&(t.d_tiny_tail_A), sizeof(unsigned int)));
    chkerr(cudaMalloc(&(t.d_tiny_tail_B), sizeof(unsigned int)));
    chkerr(cudaMalloc(&(t.d_tiny_tail_C), sizeof(unsigned int)));

    chkerr(cudaMemset(t.d_tiny_tail_A, 0, sizeof(unsigned int)));
    chkerr(cudaMemset(t.d_tiny_tail_B, 0, sizeof(unsigned int)));
    chkerr(cudaMemset(t.d_tiny_tail_C, 0, sizeof(unsigned int)));


    // -------------------------------------------------------------------------
    // Delta logs
    // A/B correspond to small ping-pong queues.
    // C corresponds to the larger overflow/global continuation queue.
    // -------------------------------------------------------------------------
    chkerr(cudaMalloc(&(t.d_delta_log_A),
                      DELTA_SMALL_CAP * sizeof(Delta)));
    chkerr(cudaMalloc(&(t.d_delta_log_B),
                      DELTA_SMALL_CAP * sizeof(Delta)));
    chkerr(cudaMalloc(&(t.d_delta_log_C),
                      DELTA_MAX_CAP * sizeof(Delta)));

    chkerr(cudaMalloc(&(t.d_delta_tail_A), sizeof(unsigned int)));
    chkerr(cudaMalloc(&(t.d_delta_tail_B), sizeof(unsigned int)));
    chkerr(cudaMalloc(&(t.d_delta_tail_C), sizeof(unsigned int)));

    chkerr(cudaMemset(t.d_delta_tail_A, 0, sizeof(unsigned int)));
    chkerr(cudaMemset(t.d_delta_tail_B, 0, sizeof(unsigned int)));
    chkerr(cudaMemset(t.d_delta_tail_C, 0, sizeof(unsigned int)));
}

inline void free_tiny_task_queues(T_pointers &t) {
    cudaFree(t.d_tiny_tasks_A);
    cudaFree(t.d_tiny_tasks_B);
    cudaFree(t.d_tiny_tasks_C);

    cudaFree(t.d_tiny_tail_A);
    cudaFree(t.d_tiny_tail_B);
    cudaFree(t.d_tiny_tail_C);

    cudaFree(t.d_delta_log_A);
    cudaFree(t.d_delta_log_B);
    cudaFree(t.d_delta_log_C);

    cudaFree(t.d_delta_tail_A);
    cudaFree(t.d_delta_tail_B);
    cudaFree(t.d_delta_tail_C);
}

inline void reset_tiny_task_queues(T_pointers &t) {
    chkerr(cudaMemset(t.d_tiny_tail_A, 0, sizeof(unsigned int)));
    chkerr(cudaMemset(t.d_tiny_tail_B, 0, sizeof(unsigned int)));
    chkerr(cudaMemset(t.d_tiny_tail_C, 0, sizeof(unsigned int)));

    chkerr(cudaMemset(t.d_delta_tail_A, 0, sizeof(unsigned int)));
    chkerr(cudaMemset(t.d_delta_tail_B, 0, sizeof(unsigned int)));
    chkerr(cudaMemset(t.d_delta_tail_C, 0, sizeof(unsigned int)));
}