#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <unistd.h>

int main(int argc, char** argv)
{
    int runs = 2;
    int threads = 16;
    int elements = 500000;

    double averageCalculationSpeedup = 0;

    for(int i = 0; i < runs; i++)
    {
        char cmd[100];
        sprintf(cmd, "./omp %d %d", threads, elements);
        system(cmd);

        // Read OpenMP results
        FILE* resultFile;
        resultFile = fopen("parallelresult.txt", "r");
        double timingArray[2];

        for(int i = 0; i < 3; i++) 
        {
            fscanf(resultFile, "%lf", &timingArray[i]);
        }
    }

}