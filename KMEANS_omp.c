
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include <omp.h>

#define MAXLINE 2000
#define MAXCAD 200

// Macros
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

/* 
	Funzione showFileError: Visualizza l'errore durante l'apertura/lettura/scrittura del file.
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
	Funzione readInput: Determina il numero di righe e colonne (cioè # di punti e dimensioni).
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
			// Controlla che la riga non sia troppo lunga
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
	Funzione readInput2: Carica i dati dal file in un array float.
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
	Funzione writeResult: Salva l'output (assegnazione dei cluster per ogni punto) su file.
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
	Funzione initCentroids: Inizializza i centroidi copiando i punti corrispondenti in base ad un vettore di posizioni casuali.
*/
void initCentroids(const float *data, float* centroids, int* centroidPos, int samples, int K)
{
	int i;
	int idx;
	#pragma omp parallel for private(i,idx) shared(data,centroids)
	for(i=0; i<K; i++)
	{
		idx = centroidPos[i];
		memcpy(&centroids[i*samples], &data[idx*samples], (samples*sizeof(float)));
	}
}

/*
	Funzione euclideanDistance: Calcola la distanza euclidea tra un punto e un centroide.
*/
float euclideanDistance(float *point, float *center, int samples)
{
	float dist=0.0;
	for(int i=0; i<samples; i++) 
	{
		float diff = point[i] - center[i];
        dist += diff * diff;
	}
	return(dist);
}

/*
	Funzione zeroFloatMatriz: Inizializza a zero una matrice float.
*/
void zeroFloatMatriz(float *matrix, int rows, int columns)
{
	int i,j;
	for (i=0; i<rows; i++)
		for (j=0; j<columns; j++)
			matrix[i*columns+j] = 0.0;	
}

/*
	Funzione zeroIntArray: Inizializza a zero un array int.
*/
void zeroIntArray(int *array, int size)
{
	int i;
	#pragma omp parallel for private(i)
	for (i=0; i<size; i++)
		array[i] = 0;	
}

int main(int argc, char* argv[])
{
	// Avvio del timer per il calcolo della memoria e inizializzazioni
	double start, end;
	start = omp_get_wtime();

	/* 
		PARAMETRI:
		argv[1]: file di input
		argv[2]: numero di cluster (K)
		argv[3]: numero massimo di iterazioni
		argv[4]: soglia percentuale minima di cambiamenti (terminazione)
		argv[5]: soglia di precisione per l'aggiornamento dei centroidi (terminazione)
		argv[6]: file di output (assegnazione cluster per ogni punto)
	*/
	if(argc !=  7)
	{
		fprintf(stderr,"EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
		fprintf(stderr,"./KMEANS [Input Filename] [Number of clusters] [Number of iterations] [Number of changes] [Threshold] [Output data file]\n");
		fflush(stderr);
		exit(-1);
	}

	// Lettura del file di input: determina il numero di punti e dimensioni di ciascun punto.
	int lines = 0, samples= 0;  
	int error = readInput(argv[1], &lines, &samples);
	if(error != 0)
	{
		showFileError(error,argv[1]);
		exit(error);
	}
	
	// Allocazione della memoria per i dati e caricamento dati dal file
	float *data = (float*)calloc(lines*samples,sizeof(float));
	if (data == NULL)
	{
		fprintf(stderr,"Memory allocation error.\n");
		exit(-4);
	}
	error = readInput2(argv[1], data);
	if(error != 0)
	{
		showFileError(error,argv[1]);
		exit(error);
	}

	// Parametri estratti dagli argomenti della riga di comando
	int K=atoi(argv[2]); 
	int maxIterations=atoi(argv[3]);
	int minChanges= (int)(lines*atof(argv[4])/100.0);
	float maxThreshold=atof(argv[5]);

	// Allocazioni per centroidi e mappatura dei cluster
	int *centroidPos = (int*)calloc(K,sizeof(int));
	float *centroids = (float*)calloc(K*samples,sizeof(float));
	int *classMap = (int*)calloc(lines,sizeof(int));

	 if (centroidPos == NULL || centroids == NULL || classMap == NULL)
	{
		fprintf(stderr,"Memory allocation error.\n");
		exit(-4);
	}

	// Inizializzazione casuale dei centroidi (scelti tra i punti)
	srand(0);
	int i;
	for(i=0; i<K; i++) 
		centroidPos[i]=rand()%lines;
	
	// Copia dei punti iniziali per i centroidi
	initCentroids(data, centroids, centroidPos, samples, K);

	// Stampa dei parametri iniziali
	printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], lines, samples);
	printf("\tNumber of clusters: %d\n", K);
	printf("\tMaximum number of iterations: %d\n", maxIterations);
	printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges, atof(argv[4]), lines);
	printf("\tMaximum centroid precision: %f\n", maxThreshold);
	
	end = omp_get_wtime();
	printf("\nMemory allocation: %f seconds\n", end - start);
	fflush(stdout);

	// Inizio del tempo di calcolo per l'algoritmo k-means
	start = omp_get_wtime();

	char *outputMsg = (char *)calloc(10000,sizeof(char));
	char line[100];

	int j;
	int class;
	float dist, minDist;
	int it=0;
	int changes = 0;
	float maxDist;

	// Allocazioni per il calcolo dei nuovi centroidi
	// pointsPerClass: conta il numero di punti per cluster
	// auxCentroids: per calcolare il nuovo centroide come media dei punti assegnati
	// distCentroids: per il calcolo della distanza tra vecchi e nuovi centroidi
	int *pointsPerClass = (int *)malloc(K*sizeof(int));
	float *auxCentroids = (float*)malloc(K*samples*sizeof(float));
	float *distCentroids = (float*)malloc(K*sizeof(float)); 
	if (pointsPerClass == NULL || auxCentroids == NULL || distCentroids == NULL)
	{
		fprintf(stderr,"Memory allocation error.\n");
		exit(-4);
	}

	/*
	 * Ciclo principale dell'algoritmo k-means:
	 * Ad ogni iterazione si assegna ogni punto al cluster più vicino e si aggiornano i centroidi.
	 */
	do {
		it++; // Incrementa il numero di iterazioni

		// 1. Assegnamento: per ogni punto calcola la distanza euclidea da ognuno dei centroidi
		changes = 0; 	
		#pragma omp parallel for private(i,class,minDist) shared(classMap,changes) schedule(dynamic,4)
		for(i=0; i<lines; i++){
			class = 1;          // Valore predefinito per il cluster
			minDist = FLT_MAX;  // Inizializza con il massimo valore possibile
			for(j=0; j<K; j++){
				// Calcola la distanza dal punto i al centroide j
				dist = euclideanDistance(&data[i*samples], &centroids[j*samples], samples);
				if(dist < minDist){
					minDist = dist;
					class = j+1;
				}
			}
			// Se il punto cambia cluster rispetto all'iterazione precedente incrementa il contatore dei cambiamenti
			if(classMap[i] != class){
				#pragma omp atomic
				changes++;
			}
			classMap[i]=class;
		}

		// 2. Calcolo dei nuovi centroidi: media dei punti appartenenti ad ogni cluster
		zeroIntArray(pointsPerClass, K);
		zeroFloatMatriz(auxCentroids, K, samples);
		
		// diviso in due cicli per la riduzione
		#pragma omp parallel for reduction(+:pointsPerClass[:K])
		for (int i = 0; i < lines; i++) {
			int class = classMap[i] - 1;
			pointsPerClass[class]++;
		}

		#pragma omp parallel for collapse(2) reduction(+:auxCentroids[:K*samples])
		for (int i = 0; i < lines; i++) {
			for (int j = 0; j < samples; j++) {
				int class = classMap[i] - 1;
				auxCentroids[class * samples + j] += data[i * samples + j];
			}
		}


		// Media: divide la sommatoria delle coordinate per il numero di punti per ogni cluster
		for(i=0; i<K; i++) {
			for(j=0; j<samples; j++){
				auxCentroids[i*samples+j] /= pointsPerClass[i];
			}
		}
		
		// 3. Valutazione del criterio di uscita:
		// Calcola la massima distanza tra vecchi e nuovi centroidi
		maxDist = FLT_MIN;
		for(i=0; i<K; i++){
			distCentroids[i] = euclideanDistance(&centroids[i*samples], &auxCentroids[i*samples], samples);
			if(distCentroids[i] > maxDist){
				maxDist = distCentroids[i];
			}
		}
		// Aggiorna i centroidi per la prossima iterazione
		memcpy(centroids, auxCentroids, (K*samples*sizeof(float)));

		// Append dei messaggi di output per il log delle iterazioni
		sprintf(line,"\n[%d] Cluster changes: %d\tMax. centroid distance: %f", it, changes, maxDist);
		outputMsg = strcat(outputMsg,line); 

	} while ((changes > minChanges) && (it < maxIterations) && (maxDist > maxThreshold*maxThreshold));
	
	// Stampa sintesi delle condizioni di terminazione
	printf("%s", outputMsg);

	// Report del tempo di calcolo
	end = omp_get_wtime();
	printf("\nComputation: %f seconds", end - start);
	fflush(stdout);

	// Ulteriore report riguardante la condizione di terminazione raggiunta
	if (changes <= minChanges) {
		printf("\n\nTermination condition:\nMinimum number of changes reached: %d [%d]", changes, minChanges);
	}
	else if (it >= maxIterations) {
		printf("\n\nTermination condition:\nMaximum number of iterations reached: %d [%d]", it, maxIterations);
	}
	else {
		printf("\n\nTermination condition:\nCentroid update precision reached: %g [%g]", maxDist, maxThreshold);
	}	

	// Scrive il risultato (assegnazione dei cluster per ogni punto) nel file di output
	error = writeResult(classMap, lines, argv[6]);
	if(error != 0)
	{
		showFileError(error, argv[6]);
		exit(error);
	}

	// Deallocazione della memoria
	free(data);
	free(classMap);
	free(centroidPos);
	free(centroids);
	free(distCentroids);
	free(pointsPerClass);
	free(auxCentroids);

	end = omp_get_wtime();
	printf("\n\nMemory deallocation: %f seconds\n", end - start);
	fflush(stdout);

	return 0;
}
