#!/bin/bash
#
# Script per eseguire 50 volte il programma MPI per 6 input file diversi
# e per thread (processi MPI) pari a 2^0, 2^1, 2^2, 2^3.
#
# La struttura del comando eseguito è:
# mpirun -np <num_processi> ./KMEANS_mpi test_files/<input_file> 1000 3 50 0.01 <output_file>
#
# L’output (stdout e stderr) di ogni run viene salvato in un file log numerato,
# all’interno di una cartella la cui denominazione dipende dall’input file e dal numero di processi.
#

# Array dei file di input (modifica i nomi secondo le tue necessità)
input_files=("input2D3.inp" "input4D.inp" "input8D.inp" "input16D.inp" "input32D.inp" "input64D.inp")

# Parametri costanti da passare all'eseguibile (modifica se necessario)
param1=128
param2=1000
param3=1
param4=1

# Ciclo sui thread/processi MPI: 2^0, 2^1, 2^2, 2^3
for exp in {0..3}; do
    num_processi=$((2**exp))
    
    # Per ogni input file
    for input in "${input_files[@]}"; do
        # Il file di input è nella cartella test_files/
        input_path="test_files/${input}"
        
        # Genera un nome per l'output sostituendo "input" con "output"
        # (modifica la logica se necessario)
        output=$(echo "$input" | sed 's/input/output/')
        
        # Costruisce la cartella di log in base all’input file e al numero di processi
        # (ad esempio: logs/input100D2_np1)
        log_folder="logs/${input%.*}_np${num_processi}"
        mkdir -p "$log_folder"
        
        echo "------------------------------------------------------------"
        echo "Esecuzione per file: ${input} con np=${num_processi}"
        echo "I log saranno salvati in: ${log_folder}"
        echo "------------------------------------------------------------"
        
        # Esegue 50 volte il comando
        for run in $(seq 1 50); do
            # Nome del file di log (numero formattato con 3 cifre, ad esempio run_001.log)
            logfile="${log_folder}/run_$(printf '%03d' "$run").log"
            
            # Costruisce la stringa del comando
            cmd="mpirun -np ${num_processi} ./KMEANS_mpi ${input_path} ${param1} ${param2} ${param3} ${param4} ${output}"
            
            # Stampa il comando e lo salva nel logfile
            echo "Esecuzione n. ${run}: ${cmd}" | tee "$logfile"
            
            # Esegue il comando, salvando stdout e stderr nel log
            ${cmd} >> "$logfile" 2>&1
            
            echo "----------------------------------------" >> "$logfile"
        done
    done
done