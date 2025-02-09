import os
import re
import csv

def extract_max_time(log_path):
    """
    Estrae tutti i tempi di computazione (float) trovati nel file di log e restituisce
    il massimo tra essi. Se non viene trovato nessun tempo, restituisce None.
    
    Si assume che i log contengano righe con il formato:
       Execution time: <valore> seconds
    Modifica la regex se il formato è diverso.
    """
    max_time = None
    # Puoi modificare il pattern se i tuoi log usano un'altra sintassi
    #pattern = re.compile(r'Execution time:\s*([\d\.]+)', re.IGNORECASE)
    pattern = re.compile(r"Computation:\s*([+-]?\d*\.?\d+)", re.IGNORECASE)
    with open(log_path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                try:
                    t = float(m.group(1))
                    if max_time is None or t > max_time:
                        max_time = t
                except ValueError:
                    continue
    return max_time

def create_csv_table_from_logs(logs_dir="logs2", output_csv="results.csv"):
    """
    Scorre la directory logs2, in cui ogni sottocartella deve avere il nome:
       <input_id>_np<num>_exp<num>
    Per ogni directory, raccoglie i file .log, estrae (per ogni file) il tempo massimo 
    di computazione e ne calcola la media.
    
    Viene poi creato un file CSV in cui:
      - Le righe sono identificate dalla coppia (MPI Threads, Experiment)
      - Le colonne sono gli input_id (ossia il nome dell'input senza gli eventuali suffissi)
      - In ogni cella viene riportato il tempo medio (formattato a 3 cifre decimali) per quella combinazione.
    """
    # Dizionario in cui salvare i risultati:
    #   result[(np_val, exp_val)][input_id] = tempo medio (float) oppure None
    result = {}
    # Per raccogliere l'elenco degli input (input_id)
    input_ids_set = set()
    
    # Itera sulle directory presenti in logs_dir
    for d in os.listdir(logs_dir):
        dpath = os.path.join(logs_dir, d)
        if os.path.isdir(dpath):
            # Ci aspettiamo un nome nel formato: <input_id>_np<num>_exp<num>
            m = re.match(r'(.+)_np(\d+)_exp(\d+)$', d)
            if not m:
                print(f"Directory {d} ignorata (non corrisponde al pattern)")
                continue
            input_id = m.group(1)  # es. "input100D2"
            np_val = int(m.group(2))
            exp_val = int(m.group(3))
            input_ids_set.add(input_id)
            key = (np_val, exp_val)
            if key not in result:
                result[key] = {}
            
            # Raccoglie i tempi (massimi per ogni file) per questa directory
            times = []
            for filename in os.listdir(dpath):
                if filename.endswith(".log"):
                    log_file = os.path.join(dpath, filename)
                    t = extract_max_time(log_file)
                    if t is not None:
                        times.append(t)
            if times:
                avg_time = sum(times) / len(times)
            else:
                avg_time = None
            result[key][input_id] = avg_time

    # Ordina le righe per np e exp
    sorted_keys = sorted(result.keys(), key=lambda x: (x[0], x[1]))
    # Ordina gli input in ordine alfabetico
    sorted_inputs = ["input2D3", "input4D", "input8D", "input16D", "input32D", "input64D"]
    
    # Crea il file CSV con intestazione:
    # Le prime colonne saranno "MPI Threads" e "Experiment", poi le colonne per ogni input
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ["MPI Threads", "Experiment"] + sorted_inputs
        writer.writerow(header)
        
        for key in sorted_keys:
            np_val, exp_val = key
            row = [np_val, exp_val]
            for input_id in sorted_inputs:
                val = result[key].get(input_id)
                cell = f"{val:.3f}" if val is not None else ""
                row.append(cell)
            writer.writerow(row)
    
    print(f"CSV creato in: {output_csv}")

# Esempio di utilizzo:
if __name__ == "__main__":
    create_csv_table_from_logs(logs_dir="logs2", output_csv="results2.csv")