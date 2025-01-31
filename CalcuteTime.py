import os
import re
import csv

def extract_computation_number(file_path):
    """Estrae il numero dopo 'Computation:' da un file .out"""
    with open(file_path, "r") as file:
        for line in file:
            match = re.search(r"Computation:\s*([+-]?\d*\.?\d+)", line)
            if match:
                return float(match.group(1))  # Converte in float
    return None  # Restituisce None se non trova nulla

def process_logs_in_seq(seq_directory, output_csv="medie.csv"):
    """Scansiona seq/, cerca le cartelle logs/ e calcola la media per ogni sottocartella salvandola in un CSV"""
    
    results = []  # Lista per salvare i risultati

    for subdir in os.listdir(seq_directory):  # Scorre le cartelle in seq/
        subdir_path = os.path.join(seq_directory, subdir)
        
        if os.path.isdir(subdir_path):  # Verifica se è una cartella
            logs_path = os.path.join(subdir_path, "logs")  # Percorso della cartella logs
            
            if os.path.isdir(logs_path):  # Controlla se logs/ esiste
                computation_values = []  # Lista per salvare i valori della singola cartella
                
                for file in os.listdir(logs_path):  # Scansiona i file nella cartella logs/
                    if file.endswith(".out"):  # Considera solo i file .out
                        file_path = os.path.join(logs_path, file)
                        value = extract_computation_number(file_path)
                        if value is not None:
                            computation_values.append(value)
                
                # Calcola la media se ci sono valori
                if computation_values:
                    mean_value = sum(computation_values) / len(computation_values)
                    results.append([subdir, mean_value])
                    print(f"Cartella '{subdir}': Media Computation = {mean_value:.2f}")
                else:
                    print(f"Cartella '{subdir}': Nessun valore trovato nei file .out")

    # Salva i risultati in un file CSV
    if results:
        with open(output_csv, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Cartella", "Media Computation"])  # Intestazione
            writer.writerows(results)
        print(f"\nRisultati salvati in '{output_csv}'")

# Esempio di utilizzo
seq_directory = "seq"  # Cambia il percorso se necessario
process_logs_in_seq(seq_directory)
