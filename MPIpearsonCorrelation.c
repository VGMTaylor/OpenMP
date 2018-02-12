#include <stdio.h>      /* printf */
#include <math.h>       /* sin */
#include <string.h>
#include <mpi.h>
#include <stdlib.h>

// NOTE: Compile with an end arguement dictating the appropriate array size: e.g mpiexec -n 5 pearsonCorrelation 2000000

// Global Variables
int world_size, world_rank;
double initializationTime_start, initializationTime_stop, calculationTime_start, calculationTime_stop; 
double initializationTime, calculationTime, p_initializationTime, p_calculationTime;

typedef enum { false, true } bool;

double p_initializationTime_start, p_initializationTime_stop, p_calculationTime_start, p_calculationTime_stop; 


int main(int argc, char** argv) 
{	

	if(world_rank == 0) 
	{
		printf("WARNING: MAKE SURE YOU APPEND A FINAL COMMAND LINE ARGUMENT FOR THE DATA ARRAY SIZE \n\n\n\n\n");
	}
	
	// Initialisation step		
	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	// Global variables
	char *c;
	long conversion = strtol(argv[1], &c, 10);
	int ARRAY_SIZE = conversion;

	// Initialisation
	int ELEMENTS_PER_PROCESS = ARRAY_SIZE / world_size;
	int remainder = ARRAY_SIZE % ELEMENTS_PER_PROCESS;

	// Serial implementation
	if (world_rank == 0) 
	{
		// Start init timer
		initializationTime_start = MPI_Wtime();

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

		// End init timer
		initializationTime_stop = MPI_Wtime();
		
		// Standard deviation variables
		double squareDistanceA = 0;
		double squareDistanceB = 0;
		double distanceA = 0;
		double distanceB = 0;
		double productOfDifferences = 0;
		
		// Start calc timer
		calculationTime_start = MPI_Wtime();

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
		calculationTime_stop = MPI_Wtime();

		// Restore assigned memory
		free(a);
		free(b);

		// Variables for timers
		initializationTime = initializationTime_stop - initializationTime_start;
		calculationTime = calculationTime_stop - calculationTime_start;

		// Print assessment text
		printf("=============================================================================================================================================\n\t\t\t\tCOMP528-1: Parallel computation of Pearson correlation coefficient using MPI\n=============================================================================================================================================\n");
		printf("Student Number: 200945941  Full Name: Jack Alan Taylor  Module: COMP528 Multi-Core and Multi-Processor Programming  Lecturer: Alexei Lisitsa");
		printf("\n---------------------------------------------------------------------------------------------------------------------------------------------\n\n");
		printf("Processes: %d\tElements per process: %d\t Remainder: %d\n", world_size, ELEMENTS_PER_PROCESS, remainder);
		printf("\n-----------------------SERIAL IMPLEMENTATION-----------------------\n");
		printf("Array Length: %d, Pearson Correlation Coefficient: %f\n", ARRAY_SIZE, pearsonCC);
		printf("A: mean = %e, std = %f\n", meanA, standardDevA);
		printf("B: mean = %e, std = %f\n", meanB, standardDevB);
		printf("Initialisation completed in %f\n", initializationTime);
		printf("Calculation completed in %f\n", calculationTime);
		printf("Overall completed in %f\n", initializationTime + calculationTime);
	}
	
  	// Parallel Implementation - ensure at least one process
  	if (world_size > 0) {

		// Start parallel init timer
		p_initializationTime_start = MPI_Wtime();

		// Code to overcome C capabilities of dynamic array allocation
		double* a = malloc(sizeof(double) * ARRAY_SIZE);
		double* b = malloc(sizeof(double) * ARRAY_SIZE);

		// Variables to calculate local means and standard deviations
		double totalLocalSumA = 0;
		double totalLocalSumB = 0;

		// Work out what index the local process should have
		int localIteration = world_rank * ELEMENTS_PER_PROCESS;

		// Populate arrays - If there is a remainder then ensure last process calculates mean for remainder too
		if (world_rank != world_size - 1) {
			for (int i = 0; i < ELEMENTS_PER_PROCESS; i++) {
				a[i] = sin(localIteration + i);
				b[i] = sin(localIteration + i + 5);
				totalLocalSumA += a[i];
				totalLocalSumB += b[i];
			}	
		}

		// Populate final process with remainder elements
		else {
			for (int i = 0; i < ELEMENTS_PER_PROCESS + remainder; i++) {
				a[i] = sin(localIteration + i);
				b[i] = sin((localIteration + i) + 5);
				totalLocalSumA += a[i];
				totalLocalSumB += b[i];
			}
		}
		
		// Synchronise the proccesses and stop the timer
		MPI_Barrier(MPI_COMM_WORLD);	
		p_initializationTime_stop = MPI_Wtime();
		
		// Start the next calculation timer
		p_calculationTime_start = MPI_Wtime();

		// Compute local mean and declare variables for reduce
		double localMeanA = totalLocalSumA / ARRAY_SIZE;
		double localMeanB = totalLocalSumB / ARRAY_SIZE;
		double globalMeanA = 0;
		double globalMeanB = 0;
		
		// Use Allreduce to broadcast result to all processes
		MPI_Allreduce(&localMeanA, &globalMeanA, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&localMeanB, &globalMeanB, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		// Standard deviation variables
		double squareDistanceA = 0;
		double squareDistanceB = 0;
		double distanceA = 0;
		double distanceB = 0;
		double productOfDifferences = 0;

		// Work out square differences for std and pxy calculations
		if(world_rank != world_size - 1) {

			for (int i = 0; i < ELEMENTS_PER_PROCESS; i++) {
				distanceA = a[i] - globalMeanA;
				distanceB = b[i] - globalMeanB;
						
				squareDistanceA += distanceA * distanceA;
				squareDistanceB += distanceB * distanceB;
						
				productOfDifferences += distanceA * distanceB;		
			}

		}

		// Else statement to deal with remainders
		else {
			for (int i = 0; i < ELEMENTS_PER_PROCESS + remainder; i++) {
				distanceA = a[i] - globalMeanA;
				distanceB = b[i] - globalMeanB;
						
				squareDistanceA += distanceA * distanceA;
				squareDistanceB += distanceB * distanceB;
						
				productOfDifferences += distanceA * distanceB;		
			}
		}

		// Sum the square differences and product of differences to compute pcc and standard dev
		double standardDevA = 0;
		double standardDevB = 0;
		double sumOfDifferences = 0;
		MPI_Reduce(&squareDistanceA, &standardDevA, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&squareDistanceB, &standardDevB, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&productOfDifferences, &sumOfDifferences, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

		if(world_rank == 0) 
		{
			// Compute std and pxy
			standardDevA = sqrt(standardDevA / ARRAY_SIZE);
			standardDevB = sqrt(standardDevB / ARRAY_SIZE);
			double pearsonCC = (sumOfDifferences / ARRAY_SIZE) / (standardDevA * standardDevB);

			// Stop timer
			p_calculationTime_stop = MPI_Wtime();

			// Variables for timers
			p_initializationTime = p_initializationTime_stop - p_initializationTime_start;
			p_calculationTime = p_calculationTime_stop - p_calculationTime_start;
			double overallTime = calculationTime + initializationTime;
			double p_overallTime = p_calculationTime + p_initializationTime;

			// Print out results
			printf("\n-----------------------PARALLEL IMPLEMENTATION-----------------------\n");
			printf("Array Length: %d, Pearson Correlation Coefficient: %f\n", ARRAY_SIZE, pearsonCC);
			printf("A: mean = %e, std = %f\n", globalMeanA, standardDevA);
			printf("B: mean = %e, std = %f\n", globalMeanB, standardDevB);
			printf("Initialisation Time: %f\n", p_initializationTime_stop - p_initializationTime_start);
			printf("Calculation Time: %f\n", p_calculationTime_stop - p_calculationTime_start);		
			printf("Overall Time: %f\n", (p_calculationTime_stop - p_calculationTime_start) + (p_initializationTime_stop - p_initializationTime_start));	
			
			printf("\n-----------------------SPEEDUP RESULTS-----------------------\n");
			printf("Initialisation Speedup: %f", initializationTime - p_initializationTime);
			printf("\t\t%f%\n", ((initializationTime - p_initializationTime) / initializationTime) * 100);
			printf("Calculation Speedup: %f", calculationTime - p_calculationTime);
			printf("\t\t\t%f%\n", ((calculationTime - p_calculationTime) / calculationTime) * 100);
			printf("Overall Speedup: %f", (overallTime - p_overallTime));
			printf("\t\t\t%f%\n", ((overallTime - p_overallTime) / overallTime) * 100);
		}

	}

	MPI_Finalize();
	return 0;
}

