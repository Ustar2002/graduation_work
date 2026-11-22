#include <stdlib.h>

void histogram(const int* data, int* hist, int n, int nbins)
{
    for (int b = 0; b < nbins; b++)
        hist[b] = 0;
    #pragma omp parallel
    {
        int *local = calloc((size_t)nbins, sizeof(int));
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < n; i++)
            local[data[i]]++;
        #pragma omp critical
        for (int b = 0; b < nbins; b++)
            hist[b] += local[b];
        free(local);
    }
}
