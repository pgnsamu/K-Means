import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Funzione per leggere i tempi dal file
def leggi_tempi(file_tempi):
    tempi = []
    with open(file_tempi, "r") as file:
        # Salta l'intestazione
        next(file)
        for line in file:
            # Estrai i tempi (colonna 2)
            _, tempo = line.split(',')
            tempi.append(float(tempo.strip()))
    return tempi

# Nome del file che contiene i tempi di esecuzione
file_tempi = "tempi_esecuzione.txt"

# Leggi i tempi dal file
tempi_esecuzione = leggi_tempi(file_tempi)

# Calcola media e deviazione standard dei tempi
media = np.mean(tempi_esecuzione)
deviazione_standard = np.std(tempi_esecuzione)

# Crea un intervallo per il grafico
x = np.linspace(min(tempi_esecuzione) - 0.01, max(tempi_esecuzione) + 0.01, 1000)

# Crea la distribuzione normale
y = norm.pdf(x, media, deviazione_standard)

# Traccia l'istogramma dei tempi e la distribuzione gaussiana
plt.figure(figsize=(10, 6))
plt.hist(tempi_esecuzione, bins=20, density=True, alpha=0.6, color='g', label='Istogramma dei tempi')
plt.plot(x, y, 'r-', label=f'Gaussiana: Media={media:.4f}, Deviazione standard={deviazione_standard:.4f}')
plt.title('Distribuzione dei Tempi di Esecuzione MPI')
plt.xlabel('Tempo (secondi)')
plt.ylabel('Densità di probabilità')
plt.legend(loc='best')
plt.grid(True)
plt.show()
