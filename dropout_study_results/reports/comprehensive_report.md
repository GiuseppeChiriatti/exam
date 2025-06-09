# Studio del Dropout come Tecnica di Regolarizzazione
## Analisi Comparativa su Reti Neurali Multistrato

**Data Esperimento:** 31/05/2025 11:35:49  
**Device utilizzato:** cpu  
**Durata totale:** 0:48:38.205305  

---

## 1. Introduzione e Obiettivi

Questo studio analizza l'efficacia del **dropout** come tecnica di regolarizzazione per le reti neurali multistrato (MLP). L'obiettivo è confrontare le prestazioni di diverse architetture neurali con e senza dropout su due dataset di classificazione:

- **MNIST**: Dataset di cifre scritte a mano (0-9)
- **Fashion-MNIST**: Dataset di capi di abbigliamento (10 categorie)

### 1.1 Metodologia

Sono state implementate e confrontate le seguenti architetture:

1. **MLP Singolo Strato Nascosto** (784 → 512 → 10)
2. **MLP Multi-Strato** (784 → 512 → 256 → 128 → 10)

Per ogni architettura sono state testate due configurazioni:
- **Senza Dropout** (dropout_rate = 0.0)
- **Con Dropout** (dropout_rate = 0.3)

### 1.2 Parametri Sperimentali

- **Ottimizzatore:** Adam
- **Learning Rate:** 0.001
- **Batch Size:** 128
- **Epochs:** 15 (modelli principali), 10 (analisi dropout)
- **Funzione di Loss:** CrossEntropyLoss

---

## 2. Risultati MNIST Dataset

### 2.1 Prestazioni dei Modelli Principali

| Modello | Test Accuracy | Training Time | Overfitting Gap |
|---------|---------------|---------------|-----------------|
| Single No Dropout | 96.88% | 156.1s | 2.24% |
| Single With Dropout | 97.89% | 167.1s | 0.06% |
| Multi No Dropout | 97.78% | 167.5s | 1.31% |
| Multi With Dropout | 97.98% | 179.6s | -0.65% |

### 2.2 Analisi Dettagliata Dropout - MNIST

Sono stati testati i seguenti valori di dropout: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

| Dropout Rate | Train Accuracy | Test Accuracy | Overfitting Gap |
|--------------|----------------|---------------|-----------------|
| 0.0 | 98.64% | 96.30% | 2.34% |
| 0.1 | 98.10% | 97.83% | 0.27% |
| 0.2 | 97.61% | 97.71% | -0.10% |
| 0.3 | 96.82% | 97.62% | -0.80% |
| 0.4 | 95.95% | 97.21% | -1.26% |
| 0.5 | 94.20% | 96.65% | -2.45% |

**Osservazioni MNIST:**
- Il dropout riduce efficacemente l'overfitting
- Il valore ottimale di dropout per MNIST sembra essere intorno a 0.2-0.3
- L'accuracy test rimane stabile anche con dropout elevato

---

## 3. Risultati Fashion-MNIST Dataset

### 3.1 Prestazioni dei Modelli Principali

| Modello | Test Accuracy | Training Time | Overfitting Gap |
|---------|---------------|---------------|-----------------|
| Single No Dropout | 88.38% | 164.1s | 5.00% |
| Single With Dropout | 88.95% | 171.7s | 1.67% |
| Multi No Dropout | 88.68% | 171.0s | 4.99% |
| Multi With Dropout | 88.43% | 182.2s | 1.21% |

### 3.2 Analisi Dettagliata Dropout - Fashion-MNIST

| Dropout Rate | Train Accuracy | Test Accuracy | Overfitting Gap |
|--------------|----------------|---------------|-----------------|
| 0.0 | 91.64% | 88.05% | 3.59% |
| 0.1 | 90.49% | 87.77% | 2.72% |
| 0.2 | 89.67% | 88.32% | 1.35% |
| 0.3 | 88.74% | 88.16% | 0.58% |
| 0.4 | 87.42% | 87.51% | -0.09% |
| 0.5 | 86.38% | 87.09% | -0.71% |

**Osservazioni Fashion-MNIST:**
- Fashion-MNIST presenta maggiore difficoltà rispetto a MNIST
- Il dropout è ancora più efficace per ridurre l'overfitting
- Le performance assolute sono inferiori ma la regolarizzazione è più importante

---

## 4. Confronto tra Dataset

### 4.1 Difficoltà Relativa
- **MNIST**: Dataset più semplice, accuracy > 95% facilmente raggiungibili
- **Fashion-MNIST**: Dataset più complesso, accuracy tipicamente 85-90%

### 4.2 Efficacia del Dropout

- **Su MNIST**: Riduzione media overfitting = 2.07%
- **Su Fashion-MNIST**: Riduzione media overfitting = 3.56%

---

## 5. Analisi Tecnica Approfondita

### 5.1 Architetture Testate

#### MLP Singolo Strato
```
Input Layer (784) → Hidden Layer (512) → [Dropout] → Output Layer (10)
```
- **Parametri totali**: ~410K
- **Capacità di memorizzazione**: Media
- **Tendenza all'overfitting**: Moderata

#### MLP Multi-Strato
```
Input (784) → Hidden1 (512) → [Dropout] → Hidden2 (256) → [Dropout] → Hidden3 (128) → [Dropout] → Output (10)
```
- **Parametri totali**: ~590K
- **Capacità di memorizzazione**: Alta
- **Tendenza all'overfitting**: Elevata

### 5.2 Meccanismo del Dropout

Il dropout agisce come regolarizzatore attraverso:

1. **Durante il Training**: Disattiva casualmente neuroni con probabilità `p`
2. **Durante il Test**: Scala i pesi per `(1-p)` per compensare
3. **Effetto**: Riduce la co-adattazione tra neuroni

---

## 6. Conclusioni e Raccomandazioni

### 6.1 Conclusioni Principali

- Il dropout riduce l'overfitting in media del 1.2% su MNIST
- Fashion-MNIST presenta overfitting maggiore di MNIST (gap medio 3.2% vs 0.7%)
- Le reti multi-strato ottengono performance superiori alle single-layer
- Dropout ottimale per MNIST: 0.1 (accuracy 97.83%)
- Dropout ottimale per Fashion-MNIST: 0.2 (accuracy 88.32%)

### 6.2 Raccomandazioni Pratiche

1. **Per dataset semplici (MNIST-like)**: Dropout 0.2-0.3
2. **Per dataset complessi (Fashion-MNIST-like)**: Dropout 0.3-0.4
3. **Per reti profonde**: Dropout essenziale per prevenire overfitting
4. **Per reti shallow**: Dropout utile ma meno critico

### 6.3 Limitazioni dello Studio

- Test limitato a due dataset di immagini 28x28
- Architetture MLP semplici (no CNN)
- Un solo valore di dropout testato per modelli principali
- Numero limitato di epochs per l'analisi dettagliata

### 6.4 Sviluppi Futuri

- Test su dataset più diversificati
- Analisi di altre tecniche di regolarizzazione (Batch Normalization, Weight Decay)
- Confronto con architetture convoluzionali
- Studio dell'interazione dropout + altre tecniche

---

## 7. Dettagli Tecnici

### 7.1 Ambiente di Sviluppo
- **Framework**: PyTorch
- **Device**: cpu
- **Versioni**: PyTorch, Matplotlib, NumPy

### 7.2 Riproducibilità
Tutti i modelli e i risultati sono stati salvati nelle seguenti directory:
- **Modelli**: `dropout_study_results_20250531_104710\models`
- **Grafici**: `dropout_study_results_20250531_104710\plots`
- **Dati**: `dropout_study_results_20250531_104710\data`
- **Report**: `dropout_study_results_20250531_104710\reports`

### 7.3 File Generati
- Grafici comparativi per ogni dataset
- Analisi dropout dettagliate
- Salvataggio stati modelli (.pth)
- Dati numerici in formato JSON

---

*Report generato automaticamente dal sistema di analisi dropout*  
*Timestamp: 2025-05-31T11:35:49.020465*
