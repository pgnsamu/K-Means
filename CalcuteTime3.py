import os
import re
import csv

def extract_execution_time(log_path):
    """
    Legge il file di log e cerca una riga con il tempo di esecuzione.
    Si assume che la riga abbia il formato:
       Execution time: <valore> seconds
    Restituisce il valore (float) se trovato, altrimenti None.
    """
    times = []
    with open(log_path, 'r') as f:
        for line in f:
            match = re.findall(r'^Computation.*?(\d+\.\d+)\s*seconds$', line)
            if match:
                times.extend([float(m) for m in match])
    return times if times else None


def create_csv_table_from_logs(logs_dir="logs", output_csv="results.csv"):
    """
    Scorre la directory dei log (logs_dir) e crea una tabella CSV con:
      - righe: numero di thread MPI (estratti dalla parte '_np<num>' della directory)
      - colonne: nome dell'input (parte iniziale del nome della directory; ad esempio "input100D2")
    In ogni cella viene memorizzato il tempo medio di esecuzione (estratto dai file .log)
    relativi a quella combinazione.
    
    La struttura delle directory attesa è quella creata dallo script Bash:
      logs/
         input100D2_np1/
             run_001.log
             run_002.log
             ...
         input100D2_np2/
             ...
         input100D2_np4/
             ...
         input100D2_np8/
             ...
         input200D2_np1/
             ...
         ...
    """
    # Dizionario per memorizzare i risultati: result[np_threads][input_id] = tempo medio
    result = {}
    # Per tenere traccia dei file di input (senza estensione) trovati
    input_files_set = set()

    # Itera sulle directory in logs_dir
    for d in os.listdir(logs_dir):
        dpath = os.path.join(logs_dir, d)
        if os.path.isdir(dpath):
            # Ci aspettiamo un nome nel formato: <input_id>_np<num>
            m = re.match(r'(.+)_np(\d+)$', d)
            # m = re.match(r'(.+)_np(\d+)_exp(\d+)$', d)
            if not m:
                continue  # salta directory che non rispettano il pattern
            input_id = m.group(1)   # ad esempio "input100D2"
            np_val = int(m.group(2))
            input_files_set.add(input_id)
            if np_val not in result:
                result[np_val] = {}

            # Lista per accumulare i tempi (uno per ogni file di log)
            times = []
            for file in os.listdir(dpath):
                if file.endswith(".log"):
                    log_file = os.path.join(dpath, file)
                    t = extract_execution_time(log_file)
                    if t is not None:
                        times.append(t[0])
                    if len(t) > 1:
                        times.append(max(t))
            # Se sono stati estratti dei tempi, calcola la media
            if times:
                avg_time = sum(times) / len(times)
            else:
                avg_time = None
            result[np_val][input_id] = avg_time

    # Ordina in modo crescente i numeri di processi e i nomi di input
    sorted_np = sorted(result.keys())
    sorted_inputs = ["input2D3", "input4D", "input8D", "input16D", "input32D", "input64D"]
    # input2D3,input4D,input8D,input16D,input32D,input64D

    # Crea e scrive il file CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Scrive la riga di intestazione: la prima colonna sarà "MPI Threads"
        header = ["MPI Threads"] + sorted_inputs
        writer.writerow(header)
        
        for np_val in sorted_np:
            row = [np_val]
            for input_id in sorted_inputs:
                value = result[np_val].get(input_id)
                # Se non è stato trovato alcun valore, lascia la cella vuota
                row.append(f"{value:.3f}" if value is not None else "")
            writer.writerow(row)
    
    print(f"CSV creato in: {output_csv}")

# Esempio di utilizzo:
if __name__ == "__main__":
    # Modifica logs_dir se necessario
    create_csv_table_from_logs(logs_dir="logs", output_csv="results.csv")