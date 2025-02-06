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
 * Licensed under CC BY-SA 4.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include <mpi.h>

#define MAXLINE 2000
#define MAXCAD 200

// Macros per calcolare minimo e massimo
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

/* 
	Funzione showFileError:
	Visualizza messaggi di errore relativi alla gestione dei file.
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
	Funzione readInput:
	Legge il file per determinare il numero di righe e colonne.
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
		  while(fgets(line, MAXLINE, fp)!= NULL) 
		{
			// Verifica presenza del carattere newline
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

/* 
	Funzione readInput2:
	Carica i dati dal file nell'array data.
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

/* 
	Funzione writeResult:
	Scrive il mapping delle classi (cluster) in un file di output.
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
		return -3; // Errore scrittura file
	}
}

/*
	Funzione initCentroids:
	Inizializza i centroidi copiando i dati dalla posizione specificata.
*/
void initCentroids(const float *data, float* centroids, int* centroidPos, int samples, int K)
{
	int i;
	int idx;
	for(i=0; i<K; i++)
	{
		idx = centroidPos[i];
		// Copia il punto corrispondente in data come centroide
		memcpy(&centroids[i*samples], &data[idx*samples], (samples*sizeof(float)));
	}
}

/*
	Funzione euclideanDistance:
	Calcola la distanza euclidea tra un punto e un centroide.
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
	Funzione zeroFloatMatriz:
	Imposta tutti gli elementi di una matrice float a 0.
*/
void zeroFloatMatriz(float *matrix, int rows, int columns)
{
	int i,j;
	for (i=0; i<rows; i++)
		for (j=0; j<columns; j++)
			matrix[i*columns+j] = 0.0;	
}

/*
	Funzione zeroIntArray:
	Imposta tutti gli elementi di un array intero a 0.
*/
void zeroIntArray(int *array, int size)
{
	int i;
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

	// Avvio timer per la misurazione delle performance (memoria)
	double start, end;
	start = MPI_Wtime();

	/*
	 * PARAMETRI:
	 * argv[1]: File di input
	 * argv[2]: Numero di cluster
	 * argv[3]: Numero massimo di iterazioni
	 * argv[4]: Percentuale minima di cambiamenti di classe (condizione di terminazione)
	 * argv[5]: Soglia di precisione per la distanza tra centroidi (condizione di terminazione)
	 * argv[6]: File di output
	 */
	if(argc !=  7)
	{
		fprintf(stderr,"EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
		fprintf(stderr,"./KMEANS [Input Filename] [Number of clusters] [Number of iterations] [Number of changes] [Threshold] [Output data file]\n");
		fflush(stderr);
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}

	// Lettura del file di input per determinare righe (punti) e colonne (dimensioni)
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
	// Lettura dei dati dal file
	error = readInput2(argv[1], data);
	if(error != 0)
	{
		showFileError(error,argv[1]);
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}

	// Parametri e conversione degli argomenti
	int K = atoi(argv[2]); 
	int maxIterations = atoi(argv[3]);
	int minChanges = (int)(lines * atof(argv[4]) / 100.0);
	float maxThreshold = atof(argv[5]);

	// Allocazione memoria per centroidi, posizioni dei centroidi e mappatura delle classi
	int *centroidPos = (int*)calloc(K,sizeof(int));
	float *centroids = (float*)calloc(K*samples,sizeof(float));
	int *classMap = (int*)calloc(lines,sizeof(int));

	 if (centroidPos == NULL || centroids == NULL || classMap == NULL)
	{
		fprintf(stderr,"Memory allocation error.\n");
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}

	// Inizializzazione casuale dei centroidi
	srand(0);
	for(int i = 0; i < K; i++) 
		centroidPos[i] = rand() % lines;
	
	// Copia dei punti iniziali nei centroidi
	initCentroids(data, centroids, centroidPos, samples, K);

	// Visualizzazione dei parametri di esecuzione
	if(rank == 0){
		printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], lines, samples);
		printf("\tNumber of clusters: %d\n", K);
		printf("\tMaximum number of iterations: %d\n", maxIterations);
		printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges, atof(argv[4]), lines);
		printf("\tMaximum centroid precision: %f\n", maxThreshold);
	}

	// Misurazione tempo allocazione memoria
	end = MPI_Wtime();
	printf("\nMemory allocation: %f seconds\n", end - start);
	fflush(stdout);

	// Inizio del conteggio del tempo di computazione
	start = MPI_Wtime();

	// Allocazione per messaggi di output e gestione della comunicazione MPI
	char *outputMsg = (char *)calloc(10000,sizeof(char));
	char line[100];

	int j, class;
	float dist, minDist;
	int it = 0;
	int changes = 0;
	float maxDist;

	// Allocazione per il calcolo locale dei nuovi centroidi e conteggio punti per cluster
	int *pointsPerClass = (int *)malloc(K*sizeof(int));
	float *auxCentroids = (float*)malloc(K*samples*sizeof(float));
	float *distCentroids = (float*)malloc(K*sizeof(float)); 
	if (pointsPerClass == NULL || auxCentroids == NULL || distCentroids == NULL)
	{
		fprintf(stderr,"Memory allocation error.\n");
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}

	// Verifica della congruenza fra numero di cluster e processi
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
	int centroidPerProcess = K / size;		// Numero di centroidi per processo
	int linesPerProcess = lines / size;		// Numero di righe per processo

	// Allocazioni per dati locali MPI
	int *pointsPerClassLocal = (int*)calloc(K,sizeof(int));
	float *auxCentroidsLocal = (float*)calloc(K*samples,sizeof(float));	
	int *classMaplocal = (int*)calloc(linesPerProcess,sizeof(int));
	int global_continue;

	// Ciclo iterativo del k-Means
	do{
		it++;
		global_continue = 0;

		// 1. CLASSIFICAZIONE: assegna ogni punto al centroide più vicino
		changes = 0;
		// Inizializza la mappa locale delle classi
		for(int y = 0; y < linesPerProcess; y++){
			classMaplocal[y] = 0;
		}
		int changesLocal = 0;
		for(int i = rank * linesPerProcess; i < (rank + 1) * linesPerProcess; i++){
			class = 1;
			minDist = FLT_MAX;
			// Calcola la distanza di ogni punto da ciascun centroide
			for(j = 0; j < K; j++){
				dist = euclideanDistance(&data[i*samples], &centroids[j*samples], samples);
				if(dist < minDist){
					minDist = dist;
					class = j + 1;
				}
			}
			// Conta i cambiamenti rispetto alla precedente assegnazione
			if(classMap[i] != class){
				changesLocal++;
			}
			classMaplocal[i - rank*linesPerProcess] = class;
		}

		// Riduzione degli errori/cambiamenti a livello globale
		MPI_Reduce(&changesLocal, &changes, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		// Raccolta di tutte le assegnazioni nei processi
		MPI_Allgather(classMaplocal, linesPerProcess, MPI_INT, classMap, linesPerProcess, MPI_INT, MPI_COMM_WORLD);

		// Visualizzazione temporanea su file commented out
		/*
		...
		*/

		// Azzeramento dei contatori per la somma dei centroidi
		if(rank == 0){
			zeroIntArray(pointsPerClass,K);
			zeroFloatMatriz(auxCentroids,K,samples);
		}
		zeroIntArray(pointsPerClassLocal,K);
		zeroFloatMatriz(auxCentroidsLocal,K,samples);

		// 2. CALCOLO DEI NUOVI CENTROIDI:
		// Somma i punti per cluster in maniera locale
		for(int i = rank*linesPerProcess; i < (rank+1)*linesPerProcess; i++) {
			class = classMap[i];
			pointsPerClassLocal[class - 1] += 1;
			for(j = 0; j < samples; j++){
				auxCentroidsLocal[(class - 1)*samples + j] += data[i*samples+j];
			}
		}

		// Riduce i totali calcolati dai processi per ottenere la media globale
		MPI_Allreduce(pointsPerClassLocal, pointsPerClass, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(auxCentroidsLocal, auxCentroids, K*samples, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

		// Ogni processo aggiorna i centroidi a lui assegnati
		int start = rank * centroidPerProcess;
		int end = (rank + 1) * centroidPerProcess;
		for(int i = start; i < end; i++) {
			for(j = 0; j < samples; j++){
				auxCentroids[i*samples+j] /= pointsPerClass[i];
			}
		}
		
		// Calcolo della distanza tra vecchi e nuovi centroidi
		maxDist = FLT_MIN;
		float maxDistLocal = FLT_MIN;
		for(int i = rank * centroidPerProcess; i < (rank+1) * centroidPerProcess; i++){
			distCentroids[i] = euclideanDistance(&centroids[i*samples], &auxCentroids[i*samples], samples);
			if(distCentroids[i] > maxDistLocal) {
				maxDistLocal = distCentroids[i];
			}
		}

		// Riduzione per trovare la massima distanza a livello globale
		MPI_Reduce(&maxDistLocal, &maxDist, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
		
		// Raccolta e condivisione dei nuovi centroidi fra i processi
		MPI_Allgather(MPI_IN_PLACE, centroidPerProcess * samples, MPI_FLOAT, auxCentroids, centroidPerProcess * samples, MPI_FLOAT, MPI_COMM_WORLD);
		
		if(rank == 0){
			// Aggiorna globalmente i centroidi
			memcpy(centroids, auxCentroids, (K*samples*sizeof(float)));
			// Aggiorna il messaggio di output con statistiche dell'iterazione
			sprintf(line,"\n[%d] Cluster changes: %d\tMax. centroid distance: %f", it, changes, maxDist);
			outputMsg = strcat(outputMsg,line);
			// Definisce la condizione di continuazione
			global_continue = (changes > minChanges) && (it < maxIterations) && (maxDist > maxThreshold);
		}
		// Broadcast dei nuovi centroidi e della condizione di continuazione agli altri processi
		MPI_Bcast(centroids, K*samples, MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&global_continue, 1, MPI_INT, 0, MPI_COMM_WORLD);
	} while(global_continue);

	// Liberazione della memoria locale per MPI
	free(pointsPerClassLocal);
	free(auxCentroidsLocal);
	free(classMaplocal);

	// Output finale dei messaggi
	printf("%s", outputMsg);	

	// Misurazione del tempo di computazione
	end = MPI_Wtime();
	printf("\nComputation: %f seconds", end - start);
	fflush(stdout);

	// Inizio della deallocazione della memoria
	start = MPI_Wtime();

	// Condizioni di terminazione e messaggi di output
	if (changes <= minChanges) {
		printf("\n\nTermination condition:\nMinimum number of changes reached: %d [%d]", changes, minChanges);
	}
	else if (it >= maxIterations) {
		printf("\n\nTermination condition:\nMaximum number of iterations reached: %d [%d]", it, maxIterations);
	}
	else {
		printf("\n\nTermination condition:\nCentroid update precision reached: %g [%g]", maxDist, maxThreshold);
	}	

	// Scrittura dei risultati nel file di output
	error = writeResult(classMap, lines, argv[6]);
	if(error != 0)
	{
		showFileError(error, argv[6]);
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}

	// Deallocazione della memoria
	free(data);
	free(classMap);
	free(centroidPos);
	free(centroids);
	free(distCentroids);
	free(pointsPerClass);
	free(auxCentroids);

	// Tempo finale per la deallocazione della memoria
	end = MPI_Wtime();
	printf("\n\nMemory deallocation: %f seconds\n", end - start);
	fflush(stdout);

	MPI_Finalize();
	return 0;
}
