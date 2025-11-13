void free_graph_gpu_memory(G_pointers &g, D_pointers &d)
{
    chkerr(cudaFree(g.offsets));
    chkerr(cudaFree(g.neighbors));
    chkerr(cudaFree(g.degree));
    chkerr(cudaFree(d.dpos));
    chkerr(cudaFree(d.dseq));
}