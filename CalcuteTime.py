import os
import re
import csv

def extract_computation_number(file_path):
    """Estrae il numero dopo 'Computation:' da un file .out"""
    values = []
    with open(file_path, "r") as file:
        for line in file:
            match = re.search(r"Computation:\s*([+-]?\d*\.?\d+)", line)
            if match:
                values.append(float(match.group(1)))  # Converte in float
    return values  # Restituisce una lista di valori trovati

def process_logs_in_seq(seq_directory, output_csv="medie.csv"):
    """Scansiona seq/, cerca le cartelle logs/ e calcola la media per ogni sottocartella, salvando i dati in CSV distinti"""

    results = []  # Lista per salvare le medie generali

    for subdir in os.listdir(seq_directory):  # Scorre le cartelle in seq/
        subdir_path = os.path.join(seq_directory, subdir)

        if os.path.isdir(subdir_path):  # Verifica se è una cartella
            logs_path = os.path.join(subdir_path, "logs")  # Percorso della cartella logs

            if os.path.isdir(logs_path):  # Controlla se logs/ esiste
                computation_values = []  # Lista per salvare i valori della singola cartella
                log_entries = []  # Lista per salvare i dati per il CSV di dettaglio

                for file in os.listdir(logs_path):  # Scansiona i file nella cartella logs/
                    if file.endswith(".out"):  # Considera solo i file .out
                        file_path = os.path.join(logs_path, file)
                        values = extract_computation_number(file_path)
                        if values:
                            computation_values.extend(values)
                            log_entries.extend([[file, v] for v in values])  # Salva file e valore

                # Salva il CSV con i valori estratti per questa cartella logs/
                if log_entries:
                    log_csv_path = os.path.join(seq_directory, f"logs_{subdir}.csv")
                    with open(log_csv_path, mode="w", newline="") as log_csv_file:
                        writer = csv.writer(log_csv_file)
                        writer.writerow(["File", "Computation Value"])  # Intestazione
                        writer.writerows(log_entries)
                    print(f"Salvati dettagli in '{log_csv_path}'")

                # Calcola la media se ci sono valori
                if computation_values:
                    mean_value = sum(computation_values) / len(computation_values)
                    results.append([subdir, mean_value])
                    print(f"Cartella '{subdir}': Media Computation = {mean_value:.2f}")
                else:
                    print(f"Cartella '{subdir}': Nessun valore trovato nei file .out")

    # Salva le medie generali in un file CSV
    if results:
        with open(output_csv, mode="a", newline="") as csv_file:
            writer = csv.writer(csv_file, delimiter=";")
            writer.writerow(["Cartella", "Media Computation"])  # Intestazione
            writer.writerows(results)
        print(f"\nRisultati medi salvati in '{output_csv}'")

# Esempio di utilizzo
seq_directory = "omp"  # Cambia il percorso se necessario
process_logs_in_seq(seq_directory)
