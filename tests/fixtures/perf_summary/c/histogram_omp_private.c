#include <stdlib.h>

void histogram(const int* keys, int* hist, int n, int n_bins)
{
    for (int b = 0; b < n_bins; b++)
        hist[b] = 0;
    #pragma omp parallel
    {
        int *local = calloc((size_t)n_bins, sizeof(int));
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < n; i++)
            local[keys[i]]++;
        #pragma omp critical
        for (int b = 0; b < n_bins; b++)
            hist[b] += local[b];
        free(local);
    }
}
