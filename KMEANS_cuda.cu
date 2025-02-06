/*
 * k-Means clustering algorithm - CUDA Parallelized Version
 *
 * Usage:
 *   ./kmeans_cuda [Input Filename] [Number of clusters] [Number of iterations]
 *                 [Percentage of changes] [Centroid precision threshold] [Output Filename]
 *
 * This code is a CUDA-parallelized version of a sequential k-means implementation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include <cuda_runtime.h>

#define MAXLINE 2000
#define MAXCAD 200

// Macros for convenience
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

/*
 * Function: showFileError
 * -------------------------
 * Displays error messages related to file operations.
 */
void showFileError(int error, char* filename)
{
    printf("Error\n");
    switch (error)
    {
        case -1:
            fprintf(stderr,"\tFile %s has too many columns.\n", filename);
            fprintf(stderr,"\tThe maximum number of columns has been exceeded. MAXLINE: %d.\n", MAXLINE);
            break;
        case -2:
            fprintf(stderr,"Error reading file: %s.\n", filename);
            break;
        case -3:
            fprintf(stderr,"Error writing file: %s.\n", filename);
            break;
    }
    fflush(stderr);
}

/*
 * Function: readInput
 * -------------------
 * Reads the input file to determine the number of points (lines) and the number of
 * dimensions per point (columns).
 */
int readInput(char* filename, int *lines, int *samples)
{
    FILE *fp;
    char line[MAXLINE] = "";
    char *ptr;
    const char *delim = "\t";
    int contlines = 0, contsamples = 0;
    
    if ((fp = fopen(filename, "r")) != NULL)
    {
        while (fgets(line, MAXLINE, fp) != NULL)
        {
            if (strchr(line, '\n') == NULL)
            {
                fclose(fp);
                return -1;
            }
            contlines++;
            ptr = strtok(line, delim);
            contsamples = 0;
            while (ptr != NULL)
            {
                contsamples++;
                ptr = strtok(NULL, delim);
            }
        }
        fclose(fp);
        *lines = contlines;
        *samples = contsamples;
        return 0;
    }
    else
    {
        return -2;
    }
}

/*
 * Function: readInput2
 * --------------------
 * Loads the input data from the file into the given data array.
 */
int readInput2(char* filename, float* data)
{
    FILE *fp;
    char line[MAXLINE] = "";
    char *ptr;
    const char *delim = "\t";
    int i = 0;
    
    if ((fp = fopen(filename, "rt")) != NULL)
    {
        while (fgets(line, MAXLINE, fp) != NULL)
        {
            ptr = strtok(line, delim);
            while (ptr != NULL)
            {
                data[i] = atof(ptr);
                i++;
                ptr = strtok(NULL, delim);
            }
        }
        fclose(fp);
        return 0;
    }
    else
    {
        return -2; // No file found
    }
}

/*
 * Function: writeResult
 * ---------------------
 * Writes the final cluster assignment for each point to the output file.
 */
int writeResult(int *classMap, int lines, const char* filename)
{
    FILE *fp;
    
    if ((fp = fopen(filename, "wt")) != NULL)
    {
        for (int i = 0; i < lines; i++)
        {
            fprintf(fp, "%d\n", classMap[i]);
        }
        fclose(fp);
        return 0;
    }
    else
    {
        return -3; // File open error
    }
}

/* 
 * CUDA Kernel: assignClusters
 * ---------------------------
 * For each point (each thread processes one point), compute the squared Euclidean distance
 * to each centroid and assign the point to the nearest centroid.
 */
__global__ void assignClusters(const float *data, const float *centroids, int *cluster_assignments,
                               int n_points, int n_features, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_points)
    {
        float min_dist = FLT_MAX;
        int best_cluster = 0;
        for (int j = 0; j < k; j++)
        {
            float sum = 0.0f;
            for (int d = 0; d < n_features; d++)
            {
                float diff = data[idx * n_features + d] - centroids[j * n_features + d];
                sum += diff * diff;
            }
            // (sqrt is not needed for comparisons)
            if (sum < min_dist)
            {
                min_dist = sum;
                best_cluster = j;
            }
        }
        cluster_assignments[idx] = best_cluster;
    }
}

/* 
 * CUDA Kernel: computeCentroidSums
 * --------------------------------
 * Each thread adds its point’s values to the sum for its assigned centroid. Atomic operations
 * are used to safely update shared sums and counts.
 */
__global__ void computeCentroidSums(const float *data, const int *cluster_assignments,
                                    float *centroid_sums, int *cluster_counts,
                                    int n_points, int n_features)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_points)
    {
        int cluster = cluster_assignments[idx];
        for (int d = 0; d < n_features; d++)
        {
            atomicAdd(&centroid_sums[cluster * n_features + d], data[idx * n_features + d]);
        }
        atomicAdd(&cluster_counts[cluster], 1);
    }
}

int main(int argc, char* argv[])
{
    // Check that the correct number of command-line parameters are provided.
    if (argc != 7)
    {
        fprintf(stderr, "EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
        fprintf(stderr, "Usage: ./kmeans_cuda [Input Filename] [Number of clusters] [Number of iterations] ");
        fprintf(stderr, "[Percentage of changes] [Centroid precision threshold] [Output Filename]\n");
        exit(-1);
    }

    // Read the input data file: number of points (lines) and dimensions (samples)
    int lines = 0, samples = 0;
    int error = readInput(argv[1], &lines, &samples);
    if (error != 0)
    {
        showFileError(error, argv[1]);
        exit(error);
    }
    
    float *h_data = (float*)calloc(lines * samples, sizeof(float));
    if (h_data == NULL)
    {
        fprintf(stderr, "Memory allocation error.\n");
        exit(-4);
    }
    error = readInput2(argv[1], h_data);
    if (error != 0)
    {
        showFileError(error, argv[1]);
        exit(error);
    }
    
    // Program parameters
    int K = atoi(argv[2]);
    int maxIterations = atoi(argv[3]);
    int minChanges = (int)(lines * atof(argv[4]) / 100.0);
    float maxThreshold = atof(argv[5]);
    
    // Allocate and initialize centroids on the host.
    // The initial centroids are selected from random points in the input.
    int *centroidPos = (int*)calloc(K, sizeof(int));
    float *h_centroids = (float*)calloc(K * samples, sizeof(float));
    if (centroidPos == NULL || h_centroids == NULL)
    {
        fprintf(stderr, "Memory allocation error.\n");
        exit(-4);
    }
    srand(0);
    for (int i = 0; i < K; i++)
        centroidPos[i] = rand() % lines;
    
    for (int i = 0; i < K; i++)
    {
        memcpy(&h_centroids[i * samples], &h_data[centroidPos[i] * samples],
               samples * sizeof(float));
    }
    
    // Allocate arrays for cluster assignments on the host.
    int *h_cluster_assignments = (int*)calloc(lines, sizeof(int));
    int *h_old_cluster_assignments = (int*)calloc(lines, sizeof(int));
    if (h_cluster_assignments == NULL || h_old_cluster_assignments == NULL)
    {
        fprintf(stderr, "Memory allocation error.\n");
        exit(-4);
    }
    // Initialize previous assignments to a value (e.g., -1) so that the first iteration always counts changes.
    for (int i = 0; i < lines; i++)
        h_old_cluster_assignments[i] = -1;
    
    // Print the configuration
    printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], lines, samples);
    printf("\tNumber of clusters: %d\n", K);
    printf("\tMaximum number of iterations: %d\n", maxIterations);
    printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges, atof(argv[4]), lines);
    printf("\tMaximum centroid precision: %f\n", maxThreshold);
    
    // Allocate device (GPU) memory.
    float *d_data, *d_centroids, *d_centroid_sums;
    int *d_cluster_assignments, *d_cluster_counts;
    cudaMalloc((void**)&d_data, lines * samples * sizeof(float));
    cudaMalloc((void**)&d_centroids, K * samples * sizeof(float));
    cudaMalloc((void**)&d_cluster_assignments, lines * sizeof(int));
    cudaMalloc((void**)&d_centroid_sums, K * samples * sizeof(float));
    cudaMalloc((void**)&d_cluster_counts, K * sizeof(int));
    
    // Copy input data and initial centroids from host to device.
    cudaMemcpy(d_data, h_data, lines * samples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids, K * samples * sizeof(float), cudaMemcpyHostToDevice);
    
    // Configure the CUDA kernel launch parameters.
    int threadsPerBlock = 256;
    int blocks = (lines + threadsPerBlock - 1) / threadsPerBlock;
    
    int iteration = 0;
    int changes = 0;
    float maxCentroidMovement = FLT_MAX;
    
    clock_t start, end;
    start = clock();
    
    // Main k-means loop.
    do {
        iteration++;
        
        // ----- Step 1: Cluster Assignment -----
        assignClusters<<<blocks, threadsPerBlock>>>(d_data, d_centroids, d_cluster_assignments, lines, samples, K);
        cudaDeviceSynchronize();
        
        // Copy the updated cluster assignments from device to host.
        cudaMemcpy(h_cluster_assignments, d_cluster_assignments, lines * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Count how many points changed their cluster assignment.
        changes = 0;
        for (int i = 0; i < lines; i++) {
            if (h_cluster_assignments[i] != h_old_cluster_assignments[i])
                changes++;
            h_old_cluster_assignments[i] = h_cluster_assignments[i];
        }
        
        // ----- Step 2: Update Centroids -----
        // Zero the centroid sums and counts on the device.
        cudaMemset(d_centroid_sums, 0, K * samples * sizeof(float));
        cudaMemset(d_cluster_counts, 0, K * sizeof(int));
        
        // Launch the kernel to compute the new centroid sums and point counts.
        computeCentroidSums<<<blocks, threadsPerBlock>>>(d_data, d_cluster_assignments, d_centroid_sums, d_cluster_counts, lines, samples);
        cudaDeviceSynchronize();
        
        // Copy the computed sums and counts back to the host.
        float *h_centroid_sums = (float*)malloc(K * samples * sizeof(float));
        int *h_cluster_counts = (int*)malloc(K * sizeof(int));
        cudaMemcpy(h_centroid_sums, d_centroid_sums, K * samples * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_cluster_counts, d_cluster_counts, K * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Update the centroids on the host and compute the maximum centroid movement.
        maxCentroidMovement = 0.0f;
        float *h_old_centroids = (float*)malloc(K * samples * sizeof(float));
        memcpy(h_old_centroids, h_centroids, K * samples * sizeof(float));
        
        for (int j = 0; j < K; j++) {
            if (h_cluster_counts[j] > 0) {
                float movement_sq = 0.0f;
                for (int d = 0; d < samples; d++) {
                    float new_val = h_centroid_sums[j * samples + d] / h_cluster_counts[j];
                    float diff = new_val - h_old_centroids[j * samples + d];
                    movement_sq += diff * diff;
                    h_centroids[j * samples + d] = new_val;
                }
                float movement = sqrtf(movement_sq);
                if (movement > maxCentroidMovement)
                    maxCentroidMovement = movement;
            }
        }
        free(h_old_centroids);
        free(h_centroid_sums);
        free(h_cluster_counts);
        
        // Copy the updated centroids back to the device.
        cudaMemcpy(d_centroids, h_centroids, K * samples * sizeof(float), cudaMemcpyHostToDevice);
        
        // Optionally, print out the iteration statistics.
        printf("[%d] Cluster changes: %d\tMax centroid movement: %f\n", iteration, changes, maxCentroidMovement);
        
    } while ((changes > minChanges) && (iteration < maxIterations) && (maxCentroidMovement > maxThreshold));
    
    end = clock();
    printf("Computation time: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
    
    // Print the termination condition.
    if (changes <= minChanges)
        printf("\nTermination condition: Minimum number of changes reached: %d [%d]\n",
               changes, minChanges);
    else if (iteration >= maxIterations)
        printf("\nTermination condition: Maximum number of iterations reached: %d [%d]\n",
               iteration, maxIterations);
    else
        printf("\nTermination condition: Centroid update precision reached: %f [%f]\n",
               maxCentroidMovement, maxThreshold);
    
    // Write the final cluster assignments to the output file.
    // (Adding 1 to each cluster assignment so that clusters are numbered starting from 1,
    // as in the original code.)
    FILE *fp = fopen(argv[6], "wt");
    if (fp == NULL)
    {
        fprintf(stderr, "Error writing output file: %s\n", argv[6]);
        exit(-3);
    }
    for (int i = 0; i < lines; i++)
    {
        fprintf(fp, "%d\n", h_cluster_assignments[i] + 1);
    }
    fclose(fp);
    
    // Free host memory.
    free(h_data);
    free(h_centroids);
    free(h_cluster_assignments);
    free(h_old_cluster_assignments);
    free(centroidPos);
    
    // Free device memory.
    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_cluster_assignments);
    cudaFree(d_centroid_sums);
    cudaFree(d_cluster_counts);
    
    return 0;
}
