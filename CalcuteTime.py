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

def extract_first_number(text):
    """Estrae il primo numero presente in una stringa per ordinamento"""
    match = re.search(r"\d+", text)
    return int(match.group()) if match else float('inf')

def process_logs_in_seq(seq_directory, output_csv="medie.csv"):
    """Scansiona seq/, cerca le cartelle logs/ e calcola la media per ogni sottocartella, salvando i dati in CSV distinti"""

    results = {}  # Dizionario per salvare le medie generali

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
                            if len(values)>1:
                                values = [max(values)]
                            computation_values.extend(values)
                            log_entries.extend([[file, v] for v in values])  # Salva file e valore

                # Salva il CSV con i valori estratti per questa cartella logs/
                if log_entries:
                    log_csv_path = os.path.join(seq_directory, f"logs_{subdir}.csv")
                    with open(log_csv_path, mode="w", newline="") as log_csv_file:
                        writer = csv.writer(log_csv_file, delimiter=";")
                        writer.writerow(["File", "Computation Value"])  # Intestazione
                        writer.writerows(log_entries)
                    print(f"Salvati dettagli in '{log_csv_path}'")

                # Calcola la media se ci sono valori
                if computation_values:
                    mean_value = sum(computation_values) / len(computation_values)
                    row_key = subdir.split('_')[-1]  # Ultimo elemento per la riga
                    col_key = subdir.split('_')[0]  # Primo elemento per la colonna
                    if row_key not in results:
                        results[row_key] = {}
                    results[row_key][col_key] = mean_value
                    print(f"Cartella '{subdir}': Media Computation = {mean_value:.2f}")
                else:
                    print(f"Cartella '{subdir}': Nessun valore trovato nei file .out")

    # Creazione della tabella delle medie
    if results:
        rows = sorted(results.keys(), key=extract_first_number)
        columns = sorted(set(col for row in results.values() for col in row.keys()), key=extract_first_number)
        
        # Crea una tabella vuota con intestazioni
        table = [[""] + columns]
        for row in rows:
            row_data = [row] + [results[row].get(col, "") for col in columns]
            table.append(row_data)

        # Salva la tabella in un file CSV
        with open(output_csv, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file, delimiter=";")
            writer.writerows(table)
        print(f"\nRisultati medi salvati in '{output_csv}'")

# Esempio di utilizzo
seq_directory = "mpi"  # Cambia il percorso se necessario
#for subdir in os.listdir("."):
    #print(subdir)
process_logs_in_seq(seq_directory)
