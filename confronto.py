import subprocess
import time
import os

def esegui_programma_mpi(comando):
    """Esegue un programma MPI e restituisce True se eseguito con successo."""
    try:
        subprocess.run(comando, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Errore durante l'esecuzione del comando: {comando}")
        print(e)
        return False

def confronta_file(file1, file2):
    """Confronta due file riga per riga e restituisce True se sono uguali."""
    try:
        with open(file1, 'r') as f1, open(file2, 'r') as f2:
            linee1 = f1.readlines()
            linee2 = f2.readlines()

            # Verifica se il numero di righe è diverso
            if len(linee1) != len(linee2):
                print(f"I file hanno un numero di righe diverso: {len(linee1)} vs {len(linee2)}")
                return False

            # Confronta ogni riga
            for i in range(len(linee1)):
                if linee1[i] != linee2[i]:
                    print(f"Riga {i+1} diversa:")
                    print(f"File1: {linee1[i].strip()}")
                    print(f"File2: {linee2[i].strip()}")
                    return False

            return True
    except FileNotFoundError:
        print(f"Uno dei file non esiste: {file1} o {file2}")
        return False
    except Exception as e:
        print(f"Si è verificato un errore durante il confronto dei file: {e}")
        return False

# Configurazioni
num_esecuzioni = 3
file_riferimento = "/home/alex/progetto_embedded/K-Means/output2D.inp"
file_generato = "/home/alex/progetto_embedded/K-Means/output2D2.inp"
comando_mpi = "mpirun -np 4 ./KMEANS_mpi test_files/input100D2.inp 1000 3 45 0.01 output2D.inp"

# File di log per i tempi di esecuzione
file_tempi = "tempi_esecuzione.txt"

# Esegui il programma 100 volte e confronta i file
tempi_esecuzione = []
differenze_trovate = False

with open(file_tempi, "w") as file_log:
    file_log.write("Esecuzione,Tempo (secondi),Confronto\n")
    
    for i in range(num_esecuzioni):
        print(f"\nEsecuzione {i+1}/{num_esecuzioni}...")
        start_time = time.time()
        
        # Esegui il programma MPI
        if not esegui_programma_mpi(comando_mpi):
            print(f"Errore nell'esecuzione del programma MPI alla iterazione {i+1}.")
            break
        
        end_time = time.time()
        tempo_esecuzione = end_time - start_time
        tempi_esecuzione.append(tempo_esecuzione)
        
        # Confronta i file
        if confronta_file(file_riferimento, file_generato):
            confronto = "Uguali"
            print("I file sono uguali.")
        else:
            confronto = "Diversi"
            differenze_trovate = True
            print(f"File diversi alla iterazione {i+1}.")
            # Se desideri fermarti al primo errore, usa: break
        
        # Scrivi i risultati nel log
        file_log.write(f"{i+1},{tempo_esecuzione:.4f},{confronto}\n")
        
    if not differenze_trovate:
        print("\nTutte le esecuzioni completate senza differenze nei file.")
    else:
        print("\nSono state trovate differenze nei file in almeno un'esecuzione.")

# Analisi dei risultati
media_tempo = sum(tempi_esecuzione) / len(tempi_esecuzione) if tempi_esecuzione else 0
print(f"\nStatistiche dei tempi di esecuzione:")
print(f"Tempo medio: {media_tempo:.4f} secondi")

