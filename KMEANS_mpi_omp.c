/*
 * k-Means clustering algorithm
 *
 * MPI version
 *
 * Parallel computing (Degree in Computer Engineering)
 * 2022/2023
 *
 * Version: 1.0
 *
 * (c) 2022 Diego García-Álvarez, Arturo Gonzalez-Escribano
 * Grupo Trasgo, Universidad de Valladolid (Spain)
 *
 * This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.
 * https://creativecommons.org/licenses/by-sa/4.0/
 */
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include <mpi.h>
#include <omp.h>

#define MAXLINE 2000
#define MAXCAD 200
#define NUM_THREADS 4

// Macros per il calcolo di minimo e massimo
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

/* 
Funzione showFileError: visualizza l'errore rilevato durante la lettura o scrittura del file.
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
Funzione readInput: determina il numero di righe (punti) e colonne (dimensioni) dal file.
*/
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
		// Legge il file riga per riga
		while(fgets(line, MAXLINE, fp)!= NULL) 
		{
			// Verifica che la riga sia completa (con carattere newline)
			if (strchr(line, '\n') == NULL)
			{
				return -1;
			}
			contlines++;       
			ptr = strtok(line, delim);
			contsamples = 0;
			// Conta il numero di colonne presenti nella riga
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

/* 
Funzione readInput2: carica i dati numerici dal file in un array di float.
*/
int readInput2(char* filename, float* data)
{
	FILE *fp;
	char line[MAXLINE] = "";
	char *ptr;
	const char *delim = "\t";
	int i = 0;
	
	if ((fp=fopen(filename,"rt"))!=NULL)
	{
		// Per ogni riga del file
		while(fgets(line, MAXLINE, fp)!= NULL)
		{         
			ptr = strtok(line, delim);
			// Converte ogni elemento da string a float
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

/* 
Funzione writeResult: scrive il cluster assegnato a ciascun punto nel file di output.
*/
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
		return -3; // Errore nella scrittura del file
	}
}

/*
Funzione initCentroids: inizializza i centroidi selezionando dei punti casuali dai dati.
*/
void initCentroids(const float *data, float* centroids, int* centroidPos, int samples, int K)
{
	int i;
	int idx;
	#pragma omp parallel for private(i,idx) shared(centroids, data, centroidPos, samples)
	for(i=0; i<K; i++)
	{
		idx = centroidPos[i];
		// Copia il punto selezionato dai dati nell'array dei centroidi
		memcpy(&centroids[i*samples], &data[idx*samples], (samples*sizeof(float)));
	}
}

/*
Funzione euclideanDistance: calcola la distanza euclidea tra un punto ed un centroide.
*/
float euclideanDistance(float *point, float *center, int samples)
{
	float dist=0.0;
	for(int i=0; i<samples; i++) 
	{
		dist += (point[i]-center[i])*(point[i]-center[i]);
	}
	dist = sqrt(dist);
	return(dist);
}

/*
Funzione zeroFloatMatriz: inizializza tutti gli elementi della matrice a zero.
*/
void zeroFloatMatriz(float *matrix, int rows, int columns)
{
	int i,j;
	for (i=0; i<rows; i++)
		for (j=0; j<columns; j++)
			matrix[i*columns+j] = 0.0;	
}

/*
Funzione zeroIntArray: inizializza tutti gli elementi dell'array a zero.
*/
void zeroIntArray(int *array, int size)
{
	int i;
	#pragma omp parallel for private(i) shared(array)
	for (i=0; i<size; i++)
		array[i] = 0;	
}

int main(int argc, char* argv[])
{
	// Inizializzazione di MPI
	MPI_Init( &argc, &argv );
	int rank;
	MPI_Comm_rank( MPI_COMM_WORLD, &rank );
	MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

	// Avvio del timer per la memoria
	double start, end;
	start = MPI_Wtime();

	/*
	* PARAMETRI:
	* argv[1]: File di input
	* argv[2]: Numero di cluster
	* argv[3]: Numero massimo di iterazioni
	* argv[4]: Percentuale minima di cambiamenti (condizione di terminazione)
	* argv[5]: Soglia di precisione per i centroidi
	* argv[6]: File di output con i cluster assegnati
	*/
	if(argc !=  7)
	{
		fprintf(stderr,"EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
		fprintf(stderr,"./KMEANS [Input Filename] [Number of clusters] [Number of iterations] [Number of changes] [Threshold] [Output data file]\n");
		fflush(stderr);
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}

	// Lettura del file di input per determinare il numero di punti e dimensioni
	int lines = 0, samples= 0;  
	int error = readInput(argv[1], &lines, &samples);
	if(error != 0)
	{
		showFileError(error,argv[1]);
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}
	
	// Allocazione dell'array per i dati
	float *data = (float*)calloc(lines*samples,sizeof(float));
	if (data == NULL)
	{
		fprintf(stderr,"Memory allocation error.\n");
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}
	// Caricamento dei dati dal file
	error = readInput2(argv[1], data);
	if(error != 0)
	{
		showFileError(error,argv[1]);
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}

	// Impostazione dei parametri del k-Means
	int K = atoi(argv[2]); 
	int maxIterations = atoi(argv[3]);
	int minChanges = (int)(lines * atof(argv[4]) / 100.0);
	float maxThreshold = atof(argv[5]);

	// Allocazione degli array per i centroidi ed il mapping delle classi
	int *centroidPos = (int*)calloc(K,sizeof(int));
	float *centroids = (float*)calloc(K*samples,sizeof(float));
	int *classMap = (int*)calloc(lines,sizeof(int));

	if (centroidPos == NULL || centroids == NULL || classMap == NULL)
	{
		fprintf(stderr,"Memory allocation error.\n");
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}

	// Inizializzazione dei centroidi con indici random
	srand(0);
	int i;
	for(i=0; i<K; i++) 
		centroidPos[i] = rand() % lines;
	
	// Carica i centroidi iniziali dai dati
	initCentroids(data, centroids, centroidPos, samples, K);

	// Stampa dei parametri di esecuzione (su ogni processo)
	printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], lines, samples);
	printf("\tNumber of clusters: %d\n", K);
	printf("\tMaximum number of iterations: %d\n", maxIterations);
	printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges, atof(argv[4]), lines);
	printf("\tMaximum centroid precision: %f\n", maxThreshold);
	
	// Stampa del tempo di allocazione memoria
	end = MPI_Wtime();
	printf("\nMemory allocation: %f seconds\n", end - start);
	fflush(stdout);

	// Inizio della fase di clustering: si riavvia il timer
	start = MPI_Wtime();

	// Buffer per messaggi di output
	char *outputMsg = (char *)calloc(10000,sizeof(char));
	char line[100];

	int j;
	int class;
	float dist, minDist;
	int it = 0;
	int changes = 0;
	float maxDist;

	// Allocazione delle variabili per la ricalcolazione dei centroidi
	int *pointsPerClass = (int *)malloc(K*sizeof(int));
	float *auxCentroids = (float*)malloc(K*samples*sizeof(float));
	float *distCentroids = (float*)malloc(K*sizeof(float)); 
	if (pointsPerClass == NULL || auxCentroids == NULL || distCentroids == NULL)
	{
		fprintf(stderr,"Memory allocation error.\n");
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}

	// Divisione del lavoro fra processi MPI: i cluster e i punti devono essere distribuibili equamente
	int size; 
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	if(K % size != 0){
		fprintf(stderr,"Number of clusters must be divisible by the number of processes\n");
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}
	if(lines % size != 0){
		fprintf(stderr,"Number of points must be divisible by the number of processes\n");
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}
	int centroidPerProcess = K / size;
	int linesPerProcess = lines / size;
	// Allocazione della memoria per le variabili locali per ogni processo MPI
	int *pointsPerClassLocal = (int*)calloc(K,sizeof(int));
	float *auxCentroidsLocal = (float*)calloc(K*samples,sizeof(float));	
	int *classMaplocal = (int*)calloc(linesPerProcess,sizeof(int));

	// Ciclo iterativo principale del k-Means
	do{
		it++;
		// Reset del contatore dei cambiamenti
		changes = 0;
		// Inizializza le classi locali a zero per ogni punto
		for(int y = 0; y < linesPerProcess; y++){
			classMaplocal[y] = 0;
		}
		int changesLocal = 0;
		// Assegnazione di ogni punto al cluster più vicino, sfruttando OpenMP per il parallelismo
		#pragma omp parallel for private(i, j, class, minDist, dist) reduction(+:changesLocal) num_threads(NUM_THREADS)
		for(i = rank * linesPerProcess; i < (rank+1) * linesPerProcess; i++){
			class = 1;
			minDist = FLT_MAX;
			for(j = 0; j < K; j++){ 
				dist = euclideanDistance(&data[i*samples], &centroids[j*samples], samples);
				if(dist < minDist){
					minDist = dist;
					class = j + 1;
				}
			}
			// Conta il cambiamento rispetto all'iterazione precedente
			if(classMap[i] != class){
				changesLocal++;
			}
			classMaplocal[i - rank * linesPerProcess] = class;
		}

		// Aggrega il numero totale di cambiamenti usando MPI_Allreduce
		MPI_Allreduce(&changesLocal, &changes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

		// Azzeramento degli array per il ricalcolo dei centroidi
		zeroIntArray(pointsPerClass, K);
		zeroFloatMatriz(auxCentroids, K, samples);
		zeroIntArray(pointsPerClassLocal, K);
		zeroFloatMatriz(auxCentroidsLocal, K, samples);

		// Calcolo locale dei nuovi centroidi, sommando i punti appartenenti a ciascun cluster
		#pragma omp parallel for private(i, class, j) shared(pointsPerClass, auxCentroids) reduction(+:pointsPerClassLocal[:K], auxCentroidsLocal[:K*samples]) num_threads(NUM_THREADS)
		for(i = rank * linesPerProcess; i < (rank+1) * linesPerProcess; i++) {
			class = classMaplocal[i - rank * linesPerProcess];
			pointsPerClassLocal[class - 1] += 1;
			for(j = 0; j < samples; j++){
				auxCentroidsLocal[(class - 1) * samples + j] += data[i * samples + j];
			}
		}

		// Somma i contributi locali di ogni processo per ottenere il nuovo centroide globale
		MPI_Allreduce(pointsPerClassLocal, pointsPerClass, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(auxCentroidsLocal, auxCentroids, K * samples, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

		// Dividi la somma per il numero di punti per calcolare la media (nuovo centroide)
		int start = rank * centroidPerProcess;
		int end = (rank + 1) * centroidPerProcess;
		#pragma omp parallel for private(i, j) shared(auxCentroids, pointsPerClass) num_threads(NUM_THREADS)
		for(i = start; i < end; i++) {
			for(j = 0; j < samples; j++){
				auxCentroids[i * samples + j] /= pointsPerClass[i];
			}
		}
		
		// Calcolo della distanza tra i centroidi vecchi ed aggiornati per verificare il criterio di convergenza
		float maxDistLocal = FLT_MIN;
		#pragma omp parallel for private(i) reduction(max:maxDistLocal) num_threads(NUM_THREADS)
		for(i = start; i < end; i++){
			distCentroids[i] = euclideanDistance(&centroids[i * samples], &auxCentroids[i * samples], samples);
			if(distCentroids[i] > maxDistLocal) {
				maxDistLocal = distCentroids[i];
			}
		}
		// Ottiene il massimo valore fra tutti i processi MPI
		MPI_Allreduce(&maxDistLocal, &maxDist, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
		
		// Aggiornamento dei centroidi globali con quelli appena calcolati
		memcpy(centroids, auxCentroids, (K * samples * sizeof(float)));
		sprintf(line, "\n[%d] Cluster changes: %d\tMax. centroid distance: %f", it, changes, maxDist);
		outputMsg = strcat(outputMsg, line);

		// Raccoglie i risultati locali dei cluster assegnati da ciascun processo
		MPI_Allgather(classMaplocal, linesPerProcess, MPI_INT, classMap, linesPerProcess, MPI_INT, MPI_COMM_WORLD);
	} while((changes > minChanges) && (it < maxIterations) && (maxDist > maxThreshold));

	// Liberazione delle variabili locali allocate per ogni processo
	free(pointsPerClassLocal);
	free(auxCentroidsLocal);
	free(classMaplocal);

	// Stampa del messaggio riassuntivo sul criterio di terminazione
	printf("%s", outputMsg);	

	// Calcola e stampa il tempo totale di esecuzione del clustering
	end = MPI_Wtime();
	printf("\nComputation: %f seconds", end - start);
	fflush(stdout);

	// Ulteriore controllo dei criteri di terminazione (minimi cambiamenti, iterazioni max o precisione)
	if (changes <= minChanges) {
		printf("\n\nTermination condition:\nMinimum number of changes reached: %d [%d]", changes, minChanges);
	}
	else if (it >= maxIterations) {
		printf("\n\nTermination condition:\nMaximum number of iterations reached: %d [%d]", it, maxIterations);
	}
	else {
		printf("\n\nTermination condition:\nCentroid update precision reached: %g [%g]", maxDist, maxThreshold);
	}	

	// Scrittura dell'output in file: cluster assegnato a ciascun punto
	error = writeResult(classMap, lines, argv[6]);
	if(error != 0)
	{
		showFileError(error, argv[6]);
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}

	// Liberazione della memoria allocata
	free(data);
	free(classMap);
	free(centroidPos);
	free(centroids);
	free(distCentroids);
	free(pointsPerClass);
	free(auxCentroids);

	// Stampa del tempo impiegato per la deallocazione della memoria
	end = MPI_Wtime();
	printf("\n\nMemory deallocation: %f seconds\n", end - start);
	fflush(stdout);

	MPI_Finalize();
	return 0;
}
