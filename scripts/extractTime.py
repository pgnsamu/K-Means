def estrai_valori_computation(file_input, file_output):
    """
    Legge un file, cerca la stringa 'Computation:', ed estrae i valori successivi,
    scrivendoli su un altro file.
    """
    try:
        with open(file_input, 'r') as fin, open(file_output, 'w') as fout:
            for linea in fin:
                if "Computation:" in linea:
                    # Estrae il valore dopo "Computation:"
                    try:
                        valore = linea.split("Computation:")[1].strip()
                        fout.write(valore + "\n")
                    except IndexError:
                        print(f"Errore nell'estrazione del valore nella riga: {linea.strip()}")
        print(f"Valori estratti e salvati in '{file_output}'.")
    except FileNotFoundError:
        print(f"Il file '{file_input}' non esiste.")
    except Exception as e:
        print(f"Si è verificato un errore: {e}")

# Esempio di utilizzo
file_input = "tempi_esecuzione.txt"  # Sostituisci con il percorso del file sorgente
file_output = "onlyTempi.txt"  # Sostituisci con il percorso del file di destinazione
estrai_valori_computation(file_input, file_output)
