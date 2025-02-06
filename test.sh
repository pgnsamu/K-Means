#!/bin/bash



# Creazione di una lista di stringhe
string_list=("stringa1" "stringa2" "stringa3")
# Accesso al primo elemento della lista
primo_elemento=${string_list[0]}
echo "Il primo elemento della lista è: $primo_elemento"

listaFile=("input2D3.inp" "input4D.inp" "input8D.inp" "input16D.inp")

# Verifica che siano stati passati dei parametri
if [ "$#" -eq 0 ]; then
    echo "Utilizzo: $0 <parametri_eseguibile>"
    exit 1
fi

for j in $(seq 1 4); do

    # Costruisco il nome della cartella a partire dai parametri, sostituendo eventuali spazi con underscore
    FOLDER=$(echo "log/seq/${listaFile[j-1]}_$@" | tr ' ' '_')
    mkdir -p "$FOLDER"

    # Specifica il percorso dell'eseguibile (modifica se necessario)
    EXECUTABLE="./KMEANS_seq"

    # Esecuzione dell'eseguibile 100 volte
    for i in $(seq 1 50); do
        # Costruisco il nome del file di log con numerazione a 3 cifre
        LOGFILE="${FOLDER}/run_$(printf '%03d' "$i").log"
        
        echo "Esecuzione numero $i con parametri: test_files/${listaFile[j-1]} $@" | tee "$LOGFILE"
        
        # Esegue l'eseguibile con gli stessi parametri passati allo script e redirige stdout e stderr
        "$EXECUTABLE" "test_files/${listaFile[j-1]}" "$@" >> "$LOGFILE" 2>&1
        
        echo "------------------------" >> "$LOGFILE"
    done
done

echo "Esecuzioni completate. I file di log sono stati salvati in: $FOLDER"