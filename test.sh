#!/bin/bash

# Imposta il percorso dell'eseguibile
EXECUTABLE="./eseguibile"

# Nome del file di log in cui salvare output ed errori
LOGFILE="esecuzioni.log"

# Se esiste già un file di log, lo cancello per iniziare da zero
if [ -f "$LOGFILE" ]; then
    rm "$LOGFILE"
fi

# Esecuzione dell'eseguibile 100 volte
for i in $(seq 1 100); do
    echo "Esecuzione numero $i:" | tee -a "$LOGFILE"
    # Esegue l'eseguibile e redirige stdout e stderr nel file di log
    "$EXECUTABLE" >> "$LOGFILE" 2>&1
    echo "------------------------" >> "$LOGFILE"
done

echo "Esecuzioni completate. Controlla il file $LOGFILE per i risultati."