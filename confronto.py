import subprocess
import time

# Funzione per eseguire il programma MPI e misurare il tempo di esecuzione
def esegui_programma_mpi(num_processi):
    # Costruisci il comando per eseguire il programma MPI
    comando = f"mpirun -np {num_processi} ./KMEANS_mpi test_files/input100D2.inp 1000 3 50 0.01 output2D.inp"
    
    # Inizia il timer
    start_time = time.time()
    
    # Esegui il comando
    subprocess.run(comando, shell=True, check=True)
    
    # Calcola il tempo di esecuzione
    end_time = time.time()
    tempo_esecuzione = end_time - start_time
    return tempo_esecuzione

# Numero di esecuzioni
numero_esecuzioni = 100
tempi_esecuzione = []

# Numero di processi MPI da utilizzare (esempio: 4 processi)
num_processi = 4

# Nome del file dove salvare i tempi
file_tempi = "tempi_esecuzione.txt"

# Apri il file in modalità scrittura (se esiste, viene sovrascritto)
with open(file_tempi, "w") as file:
    # Scrivi l'intestazione nel file
    file.write("Esecuzione,Tempo (secondi)\n")
    
    # Esegui il programma MPI 100 volte e memorizza i tempi
    for i in range(numero_esecuzioni):
        tempo = esegui_programma_mpi(num_processi)
        tempi_esecuzione.append(tempo)
        
        # Scrivi i risultati nel file
        file.write(f"{i+1},{tempo:.4f}\n")
        print(f"Esecuzione {i+1}/{numero_esecuzioni} - Tempo: {tempo:.4f} secondi")

# Analisi dei risultati
media_tempo = sum(tempi_esecuzione) / numero_esecuzioni
min_tempo = min(tempi_esecuzione)
max_tempo = max(tempi_esecuzione)

# Scrivi le statistiche nel file
with open(file_tempi, "a") as file:
    file.write("\nStatistiche:\n")
    file.write(f"Tempo medio: {media_tempo:.4f} secondi\n")
    file.write(f"Tempo minimo: {min_tempo:.4f} secondi\n")
    file.write(f"Tempo massimo: {max_tempo:.4f} secondi\n")

# Stampa le statistiche nella console
print("\nStatistiche dopo 100 esecuzioni:")
print(f"Tempo medio: {media_tempo:.4f} secondi")
print(f"Tempo minimo: {min_tempo:.4f} secondi")
print(f"Tempo massimo: {max_tempo:.4f} secondi")
