def confronta_file(file1, file2):
    # Apre entrambi i file in modalità di lettura
    try:
        with open(file1, 'r') as f1, open(file2, 'r') as f2:
            # Legge le righe di entrambi i file
            linee1 = f1.readlines()
            linee2 = f2.readlines()

            # Verifica che i file abbiano lo stesso numero di righe
            if len(linee1) != len(linee2):
                print(f"I file hanno un numero di righe diverso: {len(linee1)} vs {len(linee2)}")
                return

            # Confronta ogni riga dei due file
            for i in range(len(linee1)):
                if linee1[i] == linee2[i]:
                    print(f"Riga {i+1} uguale: {linee1[i].strip()}")
                else:
                    print(f"Riga {i+1} diversa:")
                    print(f"File1: {linee1[i].strip()}")
                    print(f"File2: {linee2[i].strip()}")
    except FileNotFoundError:
        print(f"Uno dei file non esiste. Controlla i percorsi forniti.")
    except Exception as e:
        print(f"Si è verificato un errore: {e}")

# Esegui la funzione con i percorsi dei tuoi file
file1 = "file1.txt"
file2 = "file2.txt"
confronta_file(file1, file2)
