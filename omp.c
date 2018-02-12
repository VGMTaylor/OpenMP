
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

// Defaults
#define ARRAY_DEFAULT 5000000
#define THREADS_DEFAULT 8

// Timers
double initializationTime_start, initializationTime_stop, calculationTime_start, calculationTime_stop;
double p_initializationTime_start, p_initializationTime_stop, p_calculationTime_start, p_calculationTime_stop;
double initializationTime, calculationTime, p_initializationTime, p_calculationTime, overallTime, p_overallTime;
double speedupRatio_init, speedupRatio_calc, speedupRatio_overall, mpi_speedupRatio_init, mpi_speedupRatio_calc, mpi_speedupRatio_overall;

double omp_get_wtime(void);

// Input
int ARRAY_SIZE = 0;
int ELEMENTS_PER_PROCESS = 0;
int threads = 0;

// Function declarations
void serialJob();
void parallelJob();
void resetTimers();
void results();
void mpiVersionExecute();
void writeResults();
int convertUserInput(int, char**);
int setNumberOfThreads(int, char**);

int main(int argc, char** argv)
{
    // Input conversion code
    if(argc != 3) {
        printf("Provide an appropriate array size and thread amount to use custom executions - otherwise defaults are set");
        ARRAY_SIZE = ARRAY_DEFAULT;
        threads = THREADS_DEFAULT;
    } else {
        ARRAY_SIZE = convertUserInput(argc, argv);
        threads = setNumberOfThreads(argc, argv);
        printf("%d", argc);
    }
    

    // Get number of elements per process 
    ELEMENTS_PER_PROCESS = ARRAY_SIZE / omp_get_num_threads();
    
    // Compute serial implementation
    serialJob();    

	// Reset wget timers
	resetTimers();
	
    // Compute Parallel Implementation
    parallelJob();

    // Execute MPI version     
    mpiVersionExecute();

    // Print final results & read MPI results file
    results();

    // Write final results
    writeResults();
}

void parallelJob() 
{ 
    // Set num of threads to be equal to the user input
    omp_set_num_threads(threads);

    // Start init timer
    p_initializationTime_start = omp_get_wtime();
        
    // Code to overcome C capabilities of dynamic array allocation
    double* a = malloc(sizeof(double) * ARRAY_SIZE);
    double* b = malloc(sizeof(double) * ARRAY_SIZE);
                            
    // Variables to calculate means and standard deviations
    double totalSumA = 0;
    double totalSumB = 0;        

    // Work out what index the local process should have
    int remainder = ARRAY_SIZE % ELEMENTS_PER_PROCESS;
	int localIteration = omp_get_thread_num() * ELEMENTS_PER_PROCESS;

    // For loop to generate the data arrays
    #pragma omp parallel for reduction(+:totalSumA,totalSumB) schedule(guided)
    for (int i = 0; i < ELEMENTS_PER_PROCESS + remainder; i++) {
        a[i] = sin(localIteration + i);
        totalSumA += a[i];
        b[i] = sin(localIteration + i + 5);
        totalSumB += b[i];
    }

    // Stop init timer
    p_initializationTime_stop = omp_get_wtime();

    // Standard deviation variables
    double squareDistanceA = 0;
    double squareDistanceB = 0;
    double distanceA = 0;
    double distanceB = 0;
    double productOfDifferences = 0;
    double meanA = 0;
    double meanB = 0;
    double standardDevA = 0;
    double standardDevB = 0;

    // Start calc timer
    p_calculationTime_start = omp_get_wtime();

    // Calculate means of the arrays
    #pragma omp parallel sections num_threads(omp_get_num_threads())
    {
        #pragma omp section
            meanA = totalSumA / ARRAY_SIZE;
        #pragma omp section
            meanB = totalSumB / ARRAY_SIZE;
    }
        
    #pragma omp barrier
    
    // Work out square distances from mean
    #pragma omp parallel for reduction(+:distanceA,distanceB,squareDistanceA,squareDistanceB,productOfDifferences) schedule(guided)
    for(int i = 0; i < ARRAY_SIZE; i++) {
        distanceA = a[i] - meanA;
        distanceB = b[i] - meanB;
    
        squareDistanceA += distanceA * distanceA;
        squareDistanceB += distanceB * distanceB;
    
        productOfDifferences += distanceA * distanceB;
    }

    #pragma omp barrier

    // Calculate means of the arrays
    #pragma omp parallel sections num_threads(omp_get_num_threads())
    {
        #pragma omp section
            standardDevA = sqrt(squareDistanceA / ARRAY_SIZE);
        #pragma omp section
            standardDevB = sqrt(squareDistanceB / ARRAY_SIZE);
    }
        
    #pragma omp barrier

    double pearsonCC = (productOfDifferences / ARRAY_SIZE) / (standardDevA * standardDevB);

    #pragma omp barrier
        
    // End calc timer
    p_calculationTime_stop = omp_get_wtime();
        
    // Restore assigned memory
    free(a);
    free(b);
        
    // Variables for timers
    p_initializationTime = p_initializationTime_stop - p_initializationTime_start;
    p_calculationTime = p_calculationTime_stop - p_calculationTime_start;
    p_overallTime = p_initializationTime + p_calculationTime;
    

    // Print assessment text
    printf("\n-------------------------------------------------------------------\n");
    printf("\t\t\tPARALLEL IMPLEMENTATION\n");
    printf("-------------------------------------------------------------------\n");
    printf("Array Length: %d, Pearson Correlation Coefficient: %f\n", ARRAY_SIZE, pearsonCC);
    printf("A: mean = %e, std = %f\n", meanA, standardDevA);
    printf("B: mean = %e, std = %f\n\n", meanB, standardDevB);
    printf("Initialisation completed in %f seconds\n", p_initializationTime);
    printf("Calculation completed in %f seconds\n", p_calculationTime);
    printf("Overall completed in %f seconds", p_overallTime);
    fflush(stdout);
}

void serialJob() 
{    
    // Start init timer
    initializationTime_start = omp_get_wtime();

    // Code to overcome C capabilities of dynamic array allocation
    double* a = malloc(sizeof(double) * ARRAY_SIZE);
    double* b = malloc(sizeof(double) * ARRAY_SIZE);
                    
    // Variables to calculate means and standard deviations
    double totalSumA = 0;
    double totalSumB = 0;

    // For loop to generate the data arrays
    for (int i = 0; i < ARRAY_SIZE; i++) {
        a[i] = sin(i);
        totalSumA += a[i];
        b[i] = sin(i + 5);
        totalSumB += b[i];
    }

    // Start init timer
    initializationTime_stop = omp_get_wtime();

    // Standard deviation variables
    double squareDistanceA = 0;
    double squareDistanceB = 0;
    double distanceA = 0;
    double distanceB = 0;
    double productOfDifferences = 0;

    // Start calc timer
    calculationTime_start = omp_get_wtime();

    // Calculate means of the arrays
    double meanA = totalSumA / ARRAY_SIZE;
    double meanB = totalSumB / ARRAY_SIZE;

    // Work out square distances from mean
    for(int i = 0; i < ARRAY_SIZE; i++) {
        distanceA = a[i] - meanA;
        distanceB = b[i] - meanB;

        squareDistanceA += distanceA * distanceA;
        squareDistanceB += distanceB * distanceB;

        productOfDifferences += distanceA * distanceB;
    }

    // Work out standard deviations
    double standardDevA = sqrt(squareDistanceA / ARRAY_SIZE);
    double standardDevB = sqrt(squareDistanceB / ARRAY_SIZE);

    // Pearson correlation coefficient - covariance / stds
    double pearsonCC = (productOfDifferences / ARRAY_SIZE) / (standardDevA * standardDevB);

    // End calc timer
    calculationTime_stop = omp_get_wtime();

    // Restore assigned memory
    free(a);
    free(b);

    // Variables for timers
    initializationTime = initializationTime_stop - initializationTime_start;
    calculationTime = calculationTime_stop - calculationTime_start;
    overallTime = initializationTime + calculationTime;
    
    // Print assessment text
    printf("\n\n");
    printf("\n==========================================================================================================================================================================================================\n\t\t\t\t\t\t\t\t");
    printf("COMP528-2: Parallel computation of Pearson coefficient using OpenMP\n\n\t\t\t\tStudent Number: 200945941  Full Name: Jack Alan Taylor  Module: COMP528 Multi-Core and Multi-Processor Programming  Lecturer: Alexei Lisitsa");
    printf("\n==========================================================================================================================================================================================================\n\n");

    printf("\n-------------------------------------------------------------------\n");
    printf("\t\t\tSERIAL IMPLEMENTATION\n");
    printf("-------------------------------------------------------------------\n");
    printf("Array Length: %d, Pearson Correlation Coefficient: %f\n", ARRAY_SIZE, pearsonCC);
    printf("A: mean = %e, std = %f\n", meanA, standardDevA);
    printf("B: mean = %e, std = %f\n\n", meanB, standardDevB);
    printf("Initialisation completed in %f seconds\n", initializationTime);
    printf("Calculation completed in %f seconds\n", calculationTime);
    printf("Overall completed in %f seconds\n\n", overallTime);
              
}

// Function that resets all timers for reuse
void resetTimers()
{
	initializationTime_start = 0;
    initializationTime_stop = 0;
    calculationTime_start = 0;
    calculationTime_stop = 0;
}

// Function that prints the final speed up results from the serial version and reads MPI results from a text file
void results()
{
    if(omp_get_thread_num() == 0)
    {
        // Print serial speed up
        speedupRatio_init = initializationTime / p_initializationTime;
        speedupRatio_calc = calculationTime / p_calculationTime;
        speedupRatio_overall = overallTime / p_overallTime;
        printf("\n\n-------------------------------------------------------------------\n");
        printf("\t\t\tSPEEDUP RESULTS");
        printf("\n-------------------------------------------------------------------\n");
        printf("Serial Speedup: (Ratio and Speedup percentage)\n");
        printf("Initialisation Speedup: %f", speedupRatio_init);
        printf("\t\t%f%\n", ((initializationTime - p_initializationTime) / initializationTime) * 100);
        printf("Calculation Speedup: %f", speedupRatio_calc);
        printf("\t\t\t%f%\n", ((calculationTime - p_calculationTime) / calculationTime) * 100);
        printf("Overall Speedup: %f", speedupRatio_overall);
        printf("\t\t\t%f%\n", ((overallTime - p_overallTime) / overallTime) * 100);
        
        // Read in MPI outputs
        FILE* resultFile;
        resultFile = fopen("mpiresults.txt", "r");
        double timingArray[2];
        for(int i = 0; i < 3; i++) 
        {
            fscanf(resultFile, "%lf", &timingArray[i]);
        }

        // Print/calculate mpi speedup
        mpi_speedupRatio_init = timingArray[0] / p_initializationTime;
        mpi_speedupRatio_calc = timingArray[1] / p_calculationTime;
        mpi_speedupRatio_overall = timingArray[2] / p_overallTime;
        printf("\n\nMPI Speedup: (Ratio and Speedup percentage)\n");
        printf("Initialization Speedup: %lf",mpi_speedupRatio_init);
        printf("\t\t%f%\n", ((timingArray[0] - p_initializationTime) / timingArray[0]) * 100);
        printf("Calculation Speedup: %lf", mpi_speedupRatio_calc);
        printf("\t\t\t%f%\n", ((timingArray[1] - p_calculationTime) / timingArray[1]) * 100);
        printf("Overall Speedup: %lf", mpi_speedupRatio_overall);
        printf("\t\t\t%f%\n\n", ((timingArray[2] - p_overallTime) / timingArray[2]) * 100);
    }
}

// Function which executes the attached mpi exec, must recompile MPI program if doesn't work
void mpiVersionExecute()
{
    // Executes the MPI version
    if(omp_get_thread_num() == 0)
    {
        char cmd[100];
        sprintf(cmd, "mpiexec -n %d mpi %d", 8, ARRAY_SIZE);
        system(cmd);
    }
}

// Error checks the supplied array size
int convertUserInput(int argc, char** argv) {
    
    // Variable declaration
    char* c;
    int conversion = strtol(argv[1], &c, 10);

    // Check for non-negative values
    if(conversion > 0) {
        // Check input isn't over 10 numbers long
        int length = log10(fabs(conversion)) + 1;
        if(length < 10)
            return conversion;
    }
        
    // If not valid number return the default array size      
    else {
        return ARRAY_DEFAULT;
    }
}

int setNumberOfThreads(int argc, char** argv) {
    // Variable declaration
    char* c;
    int conversion = strtol(argv[2], &c, 10);

    // Check for non-negative values
    if(conversion > 0) {
        // Check input isn't over 4 numbers long
        int length = log10(fabs(conversion)) + 1;
        if(length < 4)
            return conversion;
    }

    // If not valid number return the default thread amount      
    else {
        return THREADS_DEFAULT;
    }
}

void writeResults() {

    // Write timing file
    FILE* mpiResults;
	mpiResults = fopen("mpispeedupresult.txt", "w+");
	char out[50];
	snprintf(out, 50, "%f", mpi_speedupRatio_init);
	fputs(out, mpiResults);
	fputs("\n", mpiResults);
	snprintf(out, 50, "%f", mpi_speedupRatio_calc);
	fputs(out, mpiResults);
	fputs("\n", mpiResults);
    snprintf(out, 50, "%f", mpi_speedupRatio_overall);
	fputs(out, mpiResults);
    fputs("\n", mpiResults);
    
    // Write timing file
    FILE* parallelResults;
	parallelResults = fopen("parallelresult.txt", "w+");
	char p_out[50];
	snprintf(p_out, 50, "%f", speedupRatio_init);
	fputs(p_out, parallelResults);
	fputs("\n", parallelResults);
	snprintf(p_out, 50, "%f", speedupRatio_calc);
	fputs(p_out, parallelResults);
	fputs("\n", parallelResults);
    snprintf(p_out, 50, "%f", speedupRatio_overall);
	fputs(p_out, parallelResults);
	fputs("\n", parallelResults);
}