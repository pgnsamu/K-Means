/*
 * k-Means clustering algorithm
 *
 * CUDA version
 *
 * Il codice implementa l'algoritmo k-Means utilizzando CUDA per il parallel computing.
 */


// Inclusione delle librerie standard e CUDA.
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include <cuda.h>

// Definizioni di costanti e macro
#define MAXLINE 2000
#define MAXCAD 200

// Macro per trovare min/max
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

// Macro per controllare eventuali errori nelle chiamate CUDA
#define CHECK_CUDA_CALL( a )	{ \
	cudaError_t ok = a; \
	if ( ok != cudaSuccess ) \
		fprintf(stderr, "-- Error CUDA call in line %d: %s\n", __LINE__, cudaGetErrorString( ok ) ); \
	}
#define CHECK_CUDA_LAST()	{ \
	cudaError_t ok = cudaGetLastError(); \
	if ( ok != cudaSuccess ) \
		fprintf(stderr, "-- Error CUDA last in line %d: %s\n", __LINE__, cudaGetErrorString( ok ) ); \
	}

// Funzione per mostrare gli errori di file I/O
void showFileError(int error, char* filename)
{
	printf("Error\n");
	switch (error)
	{
		case -1:
			fprintf(stderr,"\tFile %s ha troppe colonne.\n", filename);
			fprintf(stderr,"\tNumero massimo di colonne superato. MAXLINE: %d.\n", MAXLINE);
			break;
		case -2:
			fprintf(stderr,"Errore nella lettura del file: %s.\n", filename);
			break;
		case -3:
			fprintf(stderr,"Errore nella scrittura del file: %s.\n", filename);
			break;
	}
	fflush(stderr);	
}

// Funzione readInput:
// Legge il file per determinare il numero di righe (points) e colonne (features)
int readInput(char* filename, int *lines, int *samples)
{
	FILE *fp;
	char line[MAXLINE] = "";
	char *ptr;
	const char *delim = "\t";
	int contlines, contsamples = 0;
	
	contlines = 0;

	if ((fp=fopen(filename,"r"))!=NULL)
	{
		while(fgets(line, MAXLINE, fp)!= NULL) 
		{
			// Se la linea non contiene il carattere di newline, il file potrebbe essere corrotto
			if (strchr(line, '\n') == NULL)
			{
				return -1;
			}
			contlines++;       
			ptr = strtok(line, delim);
			contsamples = 0;
			while(ptr != NULL)
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

// Funzione readInput2:
// Carica i dati da file in un array di float
int readInput2(char* filename, float* data)
{
	FILE *fp;
	char line[MAXLINE] = "";
	char *ptr;
	const char *delim = "\t";
	int i = 0;
	
	if ((fp=fopen(filename,"rt"))!=NULL)
	{
		while(fgets(line, MAXLINE, fp)!= NULL)
		{         
			ptr = strtok(line, delim);
			while(ptr != NULL)
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
		return -2; // File non trovato
	}
}

// Funzione writeResult:
// Scrive su file il cluster assegnato a ciascun punto
int writeResult(int *classMap, int lines, const char* filename)
{	
	FILE *fp;
	
	if ((fp=fopen(filename,"wt"))!=NULL)
	{
		for(int i=0; i<lines; i++)
		{
			fprintf(fp,"%d\n",classMap[i]);
		}
		fclose(fp);  
   
		return 0;
	}
	else
	{
		return -3;
	}
}

// Funzione initCentroids:
// Inizializza i centroidi copiando i dati in posizioni casuali
void initCentroids(const float *data, float* centroids, int* centroidPos, int samples, int K)
{
	int i;
	int idx;
	for(i=0; i<K; i++)
	{
		idx = centroidPos[i];
		memcpy(&centroids[i*samples], &data[idx*samples], (samples*sizeof(float)));
	}
}

// Funzione device per il calcolo della distanza euclidea tra un punto ed un centroide
__device__ float euclideanDistance(float *point, float *center, int samples)
{
	float dist = 0.0;
	for (int i = 0; i < samples; i++) {
		float diff = point[i] - center[i];
		dist += diff * diff;
	}
	return sqrt(dist);
}

// Funzione CPU per il calcolo della distanza euclidea
float euclideanDistance1(float *point, float *center, int samples)
{
	float dist = 0.0;
	for(int i = 0; i < samples; i++) 
	{
		dist += (point[i] - center[i]) * (point[i] - center[i]);
	}
	return sqrt(dist);
}

// Funzione che azzera una matrice float
void zeroFloatMatriz(float *matrix, int rows, int columns)
{
	int i,j;
	for (i=0; i<rows; i++)
		for (j=0; j<columns; j++)
			matrix[i*columns+j] = 0.0;	
}

// Funzione che azzera un array di interi
void zeroIntArray(int *array, int size)
{
	int i;
	for (i=0; i<size; i++)
		array[i] = 0;	
}

// Variabili globali in memoria device per conteggio cambiamenti e massima distanza
__device__ int d_changes;
__device__ float d_maxDist;

/*
 * Kernel CUDA: distanceCalculationForEachPoint
 * - Ogni thread calcola la distanza dal proprio punto a ciascun centroide.
 * - I centroidi sono copiati nella memoria condivisa per velocizzare l'accesso.
 * - Se il cluster assegnato al punto cambia, si aggiorna un contatore globale tramite atomicAdd.
 */
__global__
void distanceCalculationForEachPoint(float *data, float *centroids, int *classMap, int lines, int samples, int K, int iterations) {
	extern __shared__ float s_centroids[];

	// Ogni thread carica una parte dei centroidi nella memoria condivisa.
	int tid = threadIdx.x;
	int totalCentroidsElements = K * samples;
	for (int idx = tid; idx < totalCentroidsElements; idx += blockDim.x) {
		s_centroids[idx] = centroids[idx];
	}
	__syncthreads();

	// Calcola l'indice globale del punto corrente.
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < lines) {
		int old_class = classMap[i];
		float minDist = FLT_MAX;
		int new_class = old_class;
		float *point = &data[i * samples];

		// Valuta la distanza per ciascun centroide
		for (int j = 0; j < K; j++) {
			float dist = 0.0f;
			float *centroid = &s_centroids[j * samples];
			for (int k = 0; k < samples; k++) {
				float diff = point[k] - centroid[k];
				dist += diff * diff;
			}
			dist = sqrtf(dist);
			if (dist < minDist) {
				minDist = dist;
				new_class = j + 1;  // I cluster sono indicizzati da 1
			}
		}
		// Se il cluster cambia, incremento atomico del contatore di cambiamenti
		if (old_class != new_class) {
			atomicAdd(&d_changes, 1);
		}
		// Assegna il nuovo cluster al punto
		classMap[i] = new_class;
	}
}

/*
 * Kernel CUDA: recalculatesCentroids
 * - Per ogni punto, accumula i valori delle feature nel centroide corrispondente.
 * - Viene usata l'operazione atomicAdd per garantire la correttezza in ambiente parallelo.
 */
__global__
void recalculatesCentroids(float *data, int *classMap, int lines, int samples, int K, float* auxCentroids, int* pointsPerClass){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i<lines){
		int classe = classMap[i];
		for(int j=0; j<samples; j++){
			atomicAdd(&auxCentroids[(classe-1)*samples+j], data[i*samples+j]); 
		}
		atomicAdd(&pointsPerClass[classe-1], 1);
	}
}

/*
 * Kernel CUDA: updateCentroids
 * - Calcola la media per ciascun centroide dividendo la somma delle feature per il numero di punti.
 * - Ogni thread lavora su un centroide distinto.
 */
__global__
void updateCentroids(float *auxCentroids, int *pointsPerClass, int samples, int K){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i<K){
		for(int j=0; j<samples; j++){
			auxCentroids[i*samples+j] /= pointsPerClass[i];
		}
	}
}

/*
 * Funzione device atomicMaxFloat:
 * - Aggiorna in maniera atomica il valore massimo (float) in memoria.
 */
__device__ float atomicMaxFloat(float* address, float val) {
	int* address_as_i = (int*)address;
	int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}

/*
 * Kernel CUDA: calculateMaxDist
 * - Calcola la distanza tra i centroidi vecchi ed aggiornati per ogni cluster.
 * - Aggiorna la variabile globale d_maxDist con la massima distanza riscontrata.
 */
__global__
void calculateMaxDist(float *centroids, float *auxCentroids, float *distCentroids, int samples, int K){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i<K){
		distCentroids[i] = euclideanDistance(&centroids[i*samples], &auxCentroids[i*samples], samples);
		if(distCentroids[i] > d_maxDist) {
			atomicMaxFloat(&d_maxDist, distCentroids[i]);
		}
	}
}

/*
 * Funzione main:
 * - Legge e carica i dati.
 * - Inizializza i centroidi.
 * - Esegue iterativamente le seguenti operazioni:
 *    1. Assegna ad ogni punto il centroide più vicino.
 *    2. Ricalcola i centroidi facendo la media dei punti assegnati.
 *    3. Aggiorna i centroidi e valuta la convergenza.
 * - Al termine scrive i risultati su file.
 */
int main(int argc, char* argv[]){

	// Inizializzazione oraria per misurare tempi di esecuzione.
	double start, end;
	start = clock() / (double)CLOCKS_PER_SEC;

	// Controllo dei parametri in ingresso
	if(argc !=  7)
	{
		fprintf(stderr,"EXECUTION ERROR K-MEANS: Parametri non corretti.\n");
		fprintf(stderr,"./KMEANS [Input Filename] [Numero di cluster] [Numero di iterazioni] [Numero di cambiamenti] [Threshold] [Output data file]\n");
		fflush(stderr);
		exit(-1);
	}

	// Lettura dei dati di input: lines = numero di punti, samples = dimensione di ogni punto
	int lines = 0, samples = 0;  
	int error = readInput(argv[1], &lines, &samples);
	if(error != 0)
	{
		showFileError(error,argv[1]);
		exit(error);
	}
	
	// Allocazione e caricamento dei dati in memoria
	float *data = (float*)calloc(lines*samples, sizeof(float));
	if (data == NULL)
	{
		fprintf(stderr,"Memory allocation error.\n");
		exit(-4);
	}
	error = readInput2(argv[1], data);
	if(error != 0)
	{
		showFileError(error, argv[1]);
		exit(error);
	}

	// Parametri dell'algoritmo
	int K = atoi(argv[2]); 
	int maxIterations = atoi(argv[3]);
	int minChanges = (int)(lines * atof(argv[4]) / 100.0);
	float maxThreshold = atof(argv[5]);

	// Allocazione per la posizione iniziale dei centroidi, per i centroidi e per la mappa di classificazione
	int *centroidPos = (int*)calloc(K, sizeof(int));
	float *centroids = (float*)calloc(K * samples, sizeof(float));
	int *classMap = (int*)calloc(lines, sizeof(int));

	if (centroidPos == NULL || centroids == NULL || classMap == NULL)
	{
		fprintf(stderr,"Memory allocation error.\n");
		exit(-4);
	}

	// Inizializzazione casuale dei centroidi
	srand(0);
	for(int i = 0; i < K; i++) 
		centroidPos[i] = rand() % lines;
	
	// Copia dei dati nei centroidi iniziali
	initCentroids(data, centroids, centroidPos, samples, K);

	// Stampa delle informazioni sul dataset ed i parametri
	printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], lines, samples);
	printf("\tNumber of clusters: %d\n", K);
	printf("\tMaximum number of iterations: %d\n", maxIterations);
	printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges, atof(argv[4]), lines);
	printf("\tMaximum centroid precision: %f\n", maxThreshold);
	
	// Misura del tempo per allocazione in memoria
	end = clock() / (double)CLOCKS_PER_SEC;
	printf("\nMemory allocation: %f seconds\n", end - start);
	fflush(stdout);

	CHECK_CUDA_CALL(cudaSetDevice(0));
	CHECK_CUDA_CALL(cudaDeviceSynchronize());

	// Inizio iterazioni dell'algoritmo k-Means
	start = clock() / (double)CLOCKS_PER_SEC;
	char *outputMsg = (char *)calloc(10000, sizeof(char));
	char line[100];
	int it = 0;
	float maxDist;

	// Allocazione per punti per cluster e centroidi ausiliari
	int *pointsPerClass = (int *)malloc(K * sizeof(int));
	float *auxCentroids = (float*)malloc(K * samples * sizeof(float));
	float *distCentroids = (float*)malloc(K * sizeof(float)); 
	if (pointsPerClass == NULL || auxCentroids == NULL || distCentroids == NULL)
	{
		fprintf(stderr,"Memory allocation error.\n");
		exit(-4);
	}

	// Ciclo iterativo: esegue aggiornamento dei cluster finché non si raggiunge la convergenza
	int h_changes;
	do{
		it++;
		h_changes = 0;
		// Reset del contatore dei cambiamenti nella memoria device
		CHECK_CUDA_CALL(cudaMemcpyToSymbol(d_changes, &h_changes, sizeof(int), 0, cudaMemcpyHostToDevice));

		// Definizione del grid e block size per i kernel CUDA
		int blockSize = 128;
		int numBlocks = (lines + blockSize - 1) / blockSize;

		// Allocazione memoria device per dati, centroidi e mappa delle classi
		float *d_data, *d_centroids;
		int *d_classMap;
		CHECK_CUDA_CALL(cudaMalloc((void**)&d_data, lines * samples * sizeof(float)));
		CHECK_CUDA_CALL(cudaMalloc((void**)&d_centroids, K * samples * sizeof(float)));
		CHECK_CUDA_CALL(cudaMalloc((void**)&d_classMap, lines * sizeof(int)));

		// Copia dei dati dalla CPU alla memoria device
		CHECK_CUDA_CALL(cudaMemcpy(d_data, data, lines * samples * sizeof(float), cudaMemcpyHostToDevice));
		CHECK_CUDA_CALL(cudaMemcpy(d_centroids, centroids, K * samples * sizeof(float), cudaMemcpyHostToDevice));
		CHECK_CUDA_CALL(cudaMemcpy(d_classMap, classMap, lines * sizeof(int), cudaMemcpyHostToDevice));

		// Allocazione memoria condivisa per il kernel ed esecuzione kernel per assegnazione dei punti
		size_t sharedMemSize = K * samples * sizeof(float);
		distanceCalculationForEachPoint<<<numBlocks, blockSize, sharedMemSize>>>(d_data, d_centroids, d_classMap, lines, samples, K, it);
		CHECK_CUDA_CALL(cudaDeviceSynchronize());
		
		// Recupero del numero di cambiamenti dall'ambiente device
		CHECK_CUDA_CALL(cudaMemcpyFromSymbol(&h_changes, d_changes, sizeof(int), 0, cudaMemcpyDeviceToHost));
		CHECK_CUDA_CALL(cudaMemcpy(classMap, d_classMap, lines * sizeof(int), cudaMemcpyDeviceToHost));

		// Ricalcolo dei centroidi: inizializza vettori ausiliari
		zeroIntArray(pointsPerClass, K);
		zeroFloatMatriz(auxCentroids, K, samples);
		
		// Allocazione su device per il ricalcolo dei centroidi
		int *d_pointsPerClass;
		float *d_auxCentroids;
		CHECK_CUDA_CALL(cudaMalloc((void**)&d_pointsPerClass, K * sizeof(int)));
		CHECK_CUDA_CALL(cudaMalloc((void**)&d_auxCentroids, K * samples * sizeof(float)));
		CHECK_CUDA_CALL(cudaMemcpy(d_pointsPerClass, pointsPerClass, K * sizeof(int), cudaMemcpyHostToDevice));
		CHECK_CUDA_CALL(cudaMemcpy(d_auxCentroids, auxCentroids, K * samples * sizeof(float), cudaMemcpyHostToDevice));

		// Kernel per accumulare le somme per ogni centroide
		recalculatesCentroids<<<numBlocks, blockSize>>>(d_data, d_classMap, lines, samples, K, d_auxCentroids, d_pointsPerClass);
		CHECK_CUDA_CALL(cudaDeviceSynchronize());
		
		// Aggiornamento del numero di thread per aggiornare i centroidi
		blockSize = 32;
		numBlocks = (K + blockSize - 1) / blockSize;
		updateCentroids<<<numBlocks, blockSize>>>(d_auxCentroids, d_pointsPerClass, samples, K);
		CHECK_CUDA_CALL(cudaDeviceSynchronize());

		// Calcolo della distanza massima tra i centroidi vecchi e quelli aggiornati
		maxDist = FLT_MIN; 
		CHECK_CUDA_CALL(cudaMemcpyToSymbol(d_maxDist, &maxDist, sizeof(float), 0, cudaMemcpyHostToDevice));
		float *d_distCentroids;
		CHECK_CUDA_CALL(cudaMalloc((void**)&d_distCentroids, K * sizeof(float)));
		CHECK_CUDA_CALL(cudaMemcpy(d_distCentroids, distCentroids, K * sizeof(float), cudaMemcpyHostToDevice));

		calculateMaxDist<<<numBlocks, blockSize>>>(d_centroids, d_auxCentroids, d_distCentroids, samples, K);
		CHECK_CUDA_CALL(cudaDeviceSynchronize());
		CHECK_CUDA_CALL(cudaMemcpyFromSymbol(&maxDist, d_maxDist, sizeof(float), 0, cudaMemcpyDeviceToHost)); 

		// Aggiorna i centroidi per il prossimo ciclo iterativo
		CHECK_CUDA_CALL(cudaMemcpy(auxCentroids, d_auxCentroids, K * samples * sizeof(float), cudaMemcpyDeviceToHost));
		memcpy(centroids, auxCentroids, (K * samples * sizeof(float)));
		
		sprintf(line,"\n[%d] Cluster changes: %d\tMax. centroid distance: %f", it, h_changes, maxDist);
		outputMsg = strcat(outputMsg,line);
		
		// Libera la memoria device allocata per questa iterazione
		CHECK_CUDA_CALL(cudaFree(d_data));
		CHECK_CUDA_CALL(cudaFree(d_centroids));
		CHECK_CUDA_CALL(cudaFree(d_classMap));
		CHECK_CUDA_CALL(cudaFree(d_pointsPerClass));
		CHECK_CUDA_CALL(cudaFree(d_auxCentroids));

	} while((h_changes > minChanges) && (it < maxIterations) && (maxDist > maxThreshold));

	// Stampa delle condizioni di terminazione e risultati
	printf("%s", outputMsg);	
	CHECK_CUDA_CALL(cudaDeviceSynchronize());
	end = clock() / (double)CLOCKS_PER_SEC;
	printf("\nComputation: %f seconds", end - start);
	fflush(stdout);

	// Condizioni per la terminazione in base al criterio raggiunto
	if (h_changes <= minChanges) {
		printf("\n\nTermination condition:\nMinimum number of changes reached: %d [%d]", h_changes, minChanges);
	}
	else if (it >= maxIterations) {
		printf("\n\nTermination condition:\nMaximum number of iterations reached: %d [%d]", it, maxIterations);
	}
	else {
		printf("\n\nTermination condition:\nCentroid update precision reached: %g [%g]", maxDist, maxThreshold);
	}	

	// Scrive su file i cluster assegnati a ogni punto
	error = writeResult(classMap, lines, argv[6]);
	if(error != 0)
	{
		showFileError(error, argv[6]);
		exit(error);
	}

	// Libera memoria allocata in host
	free(data);
	free(classMap);
	free(centroidPos);
	free(centroids);
	free(distCentroids);
	free(pointsPerClass);
	free(auxCentroids);

	end = clock() / (double)CLOCKS_PER_SEC;
	printf("\n\nMemory deallocation: %f seconds\n", end - start);
	fflush(stdout);
	return 0;
}
